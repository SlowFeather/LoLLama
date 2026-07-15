from __future__ import annotations

import json
import time
import uuid
from array import array
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from lollama._logging import get_logger
from lollama.config import LayerConfig, MemoryConfig

from .embedding import EmbeddingClient, cosine
from .store import SqliteMemoryStore
from .textutil import char_bigrams, jaccard, normalize

logger = get_logger(__name__)

# 第 1 层工作记忆是对话窗口本身（由调用方维护），这里持久化其余 4 层。
PERSISTED_LAYERS = ("episodic", "semantic", "procedural", "core")

LAYER_LABELS = {
    "episodic": "情景",
    "semantic": "事实",
    "procedural": "偏好",
    "core": "画像",
}

_DUPLICATE_SIMILARITY_MIN_CHARS = 6
_DUPLICATE_SIMILARITY_THRESHOLD = 0.9
_DUPLICATE_CONTAINMENT_MIN_RATIO = 0.72
_NEGATION_MARKERS = frozenset("不没无非未勿别")
# 新记忆与旧记忆高度相似但否定性相反时，旧记忆强度乘以该系数（矛盾削弱）
_CONTRADICTION_DECAY = 0.5


@dataclass(slots=True)
class MemoryItem:
    id: str
    layer: str
    text: str
    importance: float
    strength: float
    created_at: float
    last_accessed: float
    hits: int = 0
    source: str = "conversation"
    meta: dict = field(default_factory=dict)

    def effective_strength(self, *, half_life_hours: float, now: float | None = None) -> float:
        """指数衰减后的当前强度；half_life_hours=0 表示永不衰减。"""
        if half_life_hours <= 0:
            return self.strength
        now = time.time() if now is None else now
        hours = max(0.0, (now - self.last_accessed) / 3600.0)
        return self.strength * (0.5 ** (hours / half_life_hours))


class MemoryManager:
    """MemoryOS 式分层记忆：情景 → 语义/程序性 → 核心画像。

    - 持久化：SQLite 行级增量写入（WAL），另导出 dashboard 只读的 JSON 镜像；
    - 检索：bigram 字面 + FTS5 BM25 + 可选本地 embedding 向量三通道加权融合，
      再叠加重要度与记忆强度；
    - 巩固/晋升/遗忘语义与旧版一致（命中巩固、热度晋升、半衰期衰减、超容量淘汰）。
    """

    def __init__(self, cfg: MemoryConfig, memory_dir: str | Path):
        self.cfg = cfg
        self.dir = Path(memory_dir)
        self.store = SqliteMemoryStore(self.dir / f"{cfg.user_id}.db")
        self._items: dict[str, list[MemoryItem]] = {layer: [] for layer in PERSISTED_LAYERS}
        self._grams: dict[str, frozenset[str]] = {}
        self._vectors: dict[str, array] = {}
        self._embedder: EmbeddingClient | None = None
        self._mirror_written_at = 0.0
        self.load()

    def set_embedder(self, embedder: EmbeddingClient | None) -> None:
        self._embedder = embedder

    def close(self) -> None:
        self.store.close()

    async def aclose(self) -> None:
        self.save()
        if self._embedder is not None:
            await self._embedder.close()
        self.close()

    # ------------------------------------------------------------ persistence

    @property
    def _mirror_file(self) -> Path:
        return self.dir / f"{self.cfg.user_id}.json"

    def load(self) -> None:
        self._items = {layer: [] for layer in PERSISTED_LAYERS}
        self._grams.clear()
        changed = False
        rows = self.store.load()
        if not rows:
            rows = self._load_legacy_json()
            changed = bool(rows)
        for raw in rows:
            layer = raw.get("layer")
            if layer not in PERSISTED_LAYERS:
                logger.warning("skipping memory item with unknown layer: %r", raw)
                changed = True
                continue
            try:
                self._items[layer].append(MemoryItem(**raw))
            except TypeError:
                logger.warning("skipping malformed memory item in layer %s: %r", layer, raw)
                changed = True
        for layer in PERSISTED_LAYERS:
            changed = self._merge_duplicates(layer) or changed
        self._vectors = self.store.load_embeddings()
        known = {item.id for items in self._items.values() for item in items}
        stale = [item_id for item_id in self._vectors if item_id not in known]
        for item_id in stale:
            del self._vectors[item_id]
        if changed:
            self.save()

    def _load_legacy_json(self) -> list[dict]:
        """首次运行时从旧版 JSON 文件迁移；之后 JSON 只作为 dashboard 镜像输出。"""
        try:
            data = json.loads(self._mirror_file.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return []
        except Exception:
            logger.exception("failed to read legacy memory file %s; starting empty", self._mirror_file)
            return []
        rows: list[dict] = []
        for layer in PERSISTED_LAYERS:
            for raw in data.get(layer, []):
                if isinstance(raw, dict):
                    rows.append(raw)
        if rows:
            logger.info("migrating %d legacy JSON memory item(s) into sqlite store", len(rows))
        return rows

    def save(self) -> None:
        """全量落库并强制刷新 JSON 镜像（用于关停、迁移后等场景）。"""
        self.store.upsert_many(item for items in self._items.values() for item in items)
        self._export_mirror(force=True)

    def _persist(self, touched: list[MemoryItem]) -> None:
        if touched:
            self.store.upsert_many(touched)
        self._export_mirror()

    def _export_mirror(self, *, force: bool = False) -> None:
        """导出 dashboard 只读的 JSON 镜像（格式与旧版持久化文件一致），带节流。"""
        now = time.time()
        if not force and now - self._mirror_written_at < self.cfg.snapshot_min_interval_sec:
            return
        self._mirror_written_at = now
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {layer: [asdict(item) for item in items] for layer, items in self._items.items()}
        tmp = self._mirror_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._mirror_file)

    # ------------------------------------------------------------------ write

    def add(
        self,
        layer: str,
        text: str,
        *,
        importance: float = 0.5,
        source: str = "conversation",
        meta: dict | None = None,
        save: bool = True,
    ) -> MemoryItem:
        if layer not in PERSISTED_LAYERS:
            raise ValueError(f"unknown memory layer: {layer}")
        text = text.strip()
        if not text:
            raise ValueError("memory text must not be empty")
        importance = min(1.0, max(0.0, importance))
        now = time.time()
        existing = self._find_duplicate(layer, text)
        if existing is not None:
            # 重复写入视为巩固而不是新增
            existing.importance = max(existing.importance, importance)
            existing.strength = min(1.5, existing.strength + self.cfg.promotion.reinforce_on_recall)
            existing.last_accessed = now
            if save:
                self._persist([existing])
            return existing
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            layer=layer,
            text=text,
            importance=importance,
            strength=1.0,
            created_at=now,
            last_accessed=now,
            source=source,
            meta=meta or {},
        )
        touched = self._weaken_contradictions(layer, item)
        self._items[layer].append(item)
        touched.append(item)
        evicted = self._enforce_capacity(layer, now=now)
        if evicted:
            self._drop_items(evicted)
            evicted_ids = {entry.id for entry in evicted}
            touched = [entry for entry in touched if entry.id not in evicted_ids]
        if save:
            self._persist(touched)
        return item

    def _weaken_contradictions(self, layer: str, item: MemoryItem) -> list[MemoryItem]:
        """新记忆与旧记忆高度相似但否定相反（如“喜欢/不喜欢苹果”）时削弱旧记忆。"""
        new_norm = normalize(item.text)
        touched: list[MemoryItem] = []
        for existing in self._items[layer]:
            old_norm = normalize(existing.text)
            if not _has_negation_mismatch(new_norm, old_norm):
                continue
            if _is_similar_ignoring_negation(new_norm, old_norm):
                existing.strength *= _CONTRADICTION_DECAY
                existing.meta["contradicted_by"] = item.id
                touched.append(existing)
                logger.info("weakening contradicted memory %s: %s", existing.id, existing.text[:60])
        return touched

    def record_turn(self, user_text: str, assistant_text: str) -> MemoryItem | None:
        """把一轮完整对话作为情景记忆存档。"""
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text:
            return None
        summary = f"用户说：{user_text}"
        if assistant_text:
            summary += f"；我回答：{assistant_text}"
        return self.add("episodic", summary[:500], importance=0.3, source="turn")

    # --------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        layers: tuple[str, ...] = PERSISTED_LAYERS,
        query_vector: array | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        query = query.strip()
        if not query:
            return []
        r = self.cfg.retrieval
        top_k = top_k or r.top_k
        now = time.time()
        candidates = [item for layer in layers for item in self._items[layer]]
        if not candidates:
            return []
        similarity = self._fused_similarity(query, candidates, query_vector)
        scored: list[tuple[MemoryItem, float]] = []
        for item in candidates:
            layer_cfg = self._layer_cfg(item.layer)
            strength = item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now)
            sim = similarity.get(item.id, 0.0)
            # 与查询毫无关联的记忆不注入（核心画像除外，画像常与任意话题相关）
            if sim <= 0 and item.layer != "core":
                continue
            score = r.similarity_weight * sim + r.importance_weight * item.importance + r.strength_weight * min(1.0, strength)
            if score >= r.min_score:
                scored.append((item, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = scored[:top_k]
        touched: list[MemoryItem] = []
        for item, _score in result:
            self._reinforce(item)
            touched.append(item)
            self._maybe_promote(item)
        if touched:
            self._persist(touched)
        return result

    async def retrieve_async(
        self,
        query: str,
        *,
        top_k: int | None = None,
        layers: tuple[str, ...] = PERSISTED_LAYERS,
    ) -> list[tuple[MemoryItem, float]]:
        """带向量通道的检索：先取查询向量（失败自动退化为纯字面检索）。"""
        query_vector = None
        if self._embedder is not None and query.strip():
            query_vector = await self._embedder.embed_one(query)
        return self.retrieve(query, top_k=top_k, layers=layers, query_vector=query_vector)

    def _fused_similarity(
        self,
        query: str,
        candidates: list[MemoryItem],
        query_vector: array | None,
    ) -> dict[str, float]:
        """bigram / FTS-BM25 / 向量 三通道按权重加权平均，输出 0~1 的融合相关度。

        只对本次查询实际可用的通道分摊权重：向量通道未启用或 FTS 无结果时，
        融合值退化为剩余通道的口径，保持与 min_score 的标定兼容。
        """
        r = self.cfg.retrieval
        allowed_ids = {item.id for item in candidates}

        channels: list[tuple[float, dict[str, float]]] = []

        query_grams = char_bigrams(query)
        if r.bigram_weight > 0 and query_grams:
            bigram_scores: dict[str, float] = {}
            for item in candidates:
                grams = self._grams.get(item.id)
                if grams is None:
                    grams = char_bigrams(item.text)
                    self._grams[item.id] = grams
                sim = jaccard(query_grams, grams)
                if sim > 0:
                    bigram_scores[item.id] = sim
            channels.append((r.bigram_weight, bigram_scores))

        if r.fts_weight > 0:
            hits = self.store.fts_search(query, limit=r.fts_candidates)
            hits = [(item_id, rank) for item_id, rank in hits if item_id in allowed_ids]
            if hits:
                # bm25 越小越相关（通常为负）；映射到 (0,1]，最优命中为 1
                best = min(rank for _id, rank in hits)
                worst = max(rank for _id, rank in hits)
                span = worst - best
                fts_scores = {
                    item_id: 1.0 if span <= 0 else max(0.05, 1.0 - (rank - best) / span)
                    for item_id, rank in hits
                }
                channels.append((r.fts_weight, fts_scores))

        if r.vector_weight > 0 and query_vector is not None and self._vectors:
            vector_scores: dict[str, float] = {}
            for item in candidates:
                vec = self._vectors.get(item.id)
                if vec is None:
                    continue
                sim = cosine(query_vector, vec)
                if sim > 0:
                    vector_scores[item.id] = min(1.0, sim)
            if vector_scores:
                channels.append((r.vector_weight, vector_scores))

        if not channels:
            return {}
        total_weight = sum(weight for weight, _scores in channels)
        fused: dict[str, float] = {}
        for weight, scores in channels:
            for item_id, sim in scores.items():
                fused[item_id] = fused.get(item_id, 0.0) + weight * sim
        return {item_id: value / total_weight for item_id, value in fused.items()}

    def format_context(self, pairs: list[tuple[MemoryItem, float]]) -> str:
        if not pairs:
            return ""
        lines = [f"- [{LAYER_LABELS.get(item.layer, item.layer)}] {item.text}" for item, _ in pairs]
        return "以下是你的长期记忆中与当前对话相关的内容，可参考但不要机械复述：\n" + "\n".join(lines)

    # ------------------------------------------------------------- embeddings

    async def embed_pending(self, *, limit: int = 32) -> int:
        """给还没有向量的记忆补算 embedding；返回本次补算条数。"""
        if self._embedder is None:
            return 0
        pending = [
            item
            for layer in PERSISTED_LAYERS
            for item in self._items[layer]
            if item.id not in self._vectors
        ][:limit]
        if not pending:
            return 0
        vectors = await self._embedder.embed([item.text for item in pending])
        if not vectors:
            return 0
        for item, vec in zip(pending, vectors):
            self._vectors[item.id] = vec
            self.store.set_embedding(item.id, self._embedder.cfg.model, vec)
        logger.info("embedded %d memory item(s)", len(pending))
        return len(pending)

    # ------------------------------------------------------------- forgetting

    def sweep(self) -> int:
        """遗忘清理：衰减到阈值以下的记忆被移除；超容量的层淘汰最弱项。返回移除数量。"""
        if not self.cfg.forgetting.enabled:
            return 0
        removed: list[MemoryItem] = []
        now = time.time()
        for layer in PERSISTED_LAYERS:
            layer_cfg = self._layer_cfg(layer)
            kept: list[MemoryItem] = []
            for item in self._items[layer]:
                strength = item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now)
                if strength < layer_cfg.min_strength:
                    removed.append(item)
                    logger.debug("forgetting %s memory %s: %s", layer, item.id, item.text[:60])
                else:
                    kept.append(item)
            self._items[layer] = kept
            removed.extend(self._enforce_capacity(layer, now=now))
        if removed:
            self._drop_items(removed)
            self._export_mirror(force=True)
            logger.info("memory sweep removed %d items", len(removed))
        return len(removed)

    def clear(self) -> None:
        self._items = {layer: [] for layer in PERSISTED_LAYERS}
        self._grams.clear()
        self._vectors.clear()
        self.store.clear()
        self._export_mirror(force=True)

    def stats(self) -> dict:
        return {layer: len(items) for layer, items in self._items.items()}

    def items(self, layer: str) -> list[MemoryItem]:
        return list(self._items[layer])

    # ---------------------------------------------------------------- helpers

    def _drop_items(self, dropped: list[MemoryItem]) -> None:
        self.store.delete_many(item.id for item in dropped)
        for item in dropped:
            self._grams.pop(item.id, None)
            self._vectors.pop(item.id, None)

    def _layer_cfg(self, layer: str) -> LayerConfig:
        return getattr(self.cfg.layers, layer)

    def _find_duplicate(self, layer: str, text: str) -> MemoryItem | None:
        return _find_duplicate_in(self._items[layer], text)

    def _merge_duplicates(self, layer: str) -> bool:
        merged: list[MemoryItem] = []
        dropped: list[MemoryItem] = []
        for item in self._items[layer]:
            existing = _find_duplicate_in(merged, item.text)
            if existing is None:
                merged.append(item)
                continue
            _merge_item(existing, item)
            dropped.append(item)
        self._items[layer] = merged
        if dropped:
            self._drop_items(dropped)
        return bool(dropped)

    def _reinforce(self, item: MemoryItem) -> None:
        layer_cfg = self._layer_cfg(item.layer)
        now = time.time()
        current = item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now)
        item.strength = min(1.5, current + self.cfg.promotion.reinforce_on_recall)
        item.last_accessed = now
        item.hits += 1

    def _maybe_promote(self, item: MemoryItem) -> bool:
        """情景记忆热度达标后晋升为语义记忆（MemoryOS 的 heat-based 升层）。"""
        promo = self.cfg.promotion
        if not promo.enabled or item.layer != "episodic":
            return False
        if _is_raw_turn_memory(item):
            # 原始对话流水保留在情景层；语义/偏好/画像由提炼器或显式工具写入，
            # 避免把“用户说：...；我回答：...”整段搬进 semantic。
            return False
        if item.hits < promo.episodic_hits_to_semantic:
            return False
        if item not in self._items["episodic"]:
            return False
        self._items["episodic"].remove(item)
        item.layer = "semantic"
        item.importance = min(1.0, item.importance + 0.2)
        self._items["semantic"].append(item)
        evicted = self._enforce_capacity("semantic", now=time.time())
        if evicted:
            self._drop_items(evicted)
        logger.info("promoted episodic memory to semantic: %s", item.text[:60])
        return True

    def _enforce_capacity(self, layer: str, *, now: float) -> list[MemoryItem]:
        layer_cfg = self._layer_cfg(layer)
        items = self._items[layer]
        overflow = len(items) - layer_cfg.capacity
        if overflow <= 0:
            return []
        items.sort(key=lambda item: item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now))
        evicted = items[:overflow]
        self._items[layer] = items[overflow:]
        for item in evicted:
            logger.debug("evicting %s memory over capacity: %s", layer, item.text[:60])
        return evicted


def _find_duplicate_in(items: list[MemoryItem], text: str) -> MemoryItem | None:
    normalized = normalize(text)
    for item in items:
        if normalize(item.text) == normalized:
            return item
    for item in items:
        if _is_similar_duplicate(normalized, normalize(item.text)):
            return item
    return None


def _is_similar_duplicate(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if _has_negation_mismatch(left, right):
        return False
    return _is_similar_ignoring_negation(left, right)


def _is_similar_ignoring_negation(left: str, right: str) -> bool:
    if not left or not right:
        return False
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) < _DUPLICATE_SIMILARITY_MIN_CHARS:
        return False
    if shorter in longer and len(shorter) / len(longer) >= _DUPLICATE_CONTAINMENT_MIN_RATIO:
        return True
    return SequenceMatcher(None, left, right).ratio() >= _DUPLICATE_SIMILARITY_THRESHOLD


def _has_negation_mismatch(left: str, right: str) -> bool:
    return bool(_NEGATION_MARKERS & set(left)) != bool(_NEGATION_MARKERS & set(right))


def _merge_item(existing: MemoryItem, duplicate: MemoryItem) -> None:
    existing.importance = max(existing.importance, duplicate.importance)
    existing.strength = min(1.5, max(existing.strength, duplicate.strength))
    existing.created_at = min(existing.created_at, duplicate.created_at)
    existing.last_accessed = max(existing.last_accessed, duplicate.last_accessed)
    existing.hits += duplicate.hits


def _is_raw_turn_memory(item: MemoryItem) -> bool:
    return item.source == "turn" or item.text.startswith("用户说：") or "；我回答：" in item.text
