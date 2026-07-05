from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lollama._logging import get_logger
from lollama.config import LayerConfig, MemoryConfig

logger = get_logger(__name__)

# 第 1 层工作记忆是对话窗口本身（由调用方维护），这里持久化其余 4 层。
PERSISTED_LAYERS = ("episodic", "semantic", "procedural", "core")

LAYER_LABELS = {
    "episodic": "情景",
    "semantic": "事实",
    "procedural": "偏好",
    "core": "画像",
}


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

    - 检索命中会巩固强度并累计热度，情景记忆达到热度阈值晋升为语义记忆；
    - 遗忘 = 指数衰减 + 低于 min_strength 清除 + 超容量淘汰最弱项；
    - 以 JSON 文件按 user_id 持久化。
    """

    def __init__(self, cfg: MemoryConfig, memory_dir: str | Path):
        self.cfg = cfg
        self.dir = Path(memory_dir)
        self._items: dict[str, list[MemoryItem]] = {layer: [] for layer in PERSISTED_LAYERS}
        self.load()

    # ------------------------------------------------------------ persistence

    @property
    def _file(self) -> Path:
        return self.dir / f"{self.cfg.user_id}.json"

    def load(self) -> None:
        self._items = {layer: [] for layer in PERSISTED_LAYERS}
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            logger.exception("failed to load memory file %s; starting empty", self._file)
            return
        for layer in PERSISTED_LAYERS:
            for raw in data.get(layer, []):
                try:
                    self._items[layer].append(MemoryItem(**raw))
                except TypeError:
                    logger.warning("skipping malformed memory item in layer %s: %r", layer, raw)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {layer: [asdict(item) for item in items] for layer, items in self._items.items()}
        tmp = self._file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._file)

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
                self.save()
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
        self._items[layer].append(item)
        self._enforce_capacity(layer, now=now)
        if save:
            self.save()
        return item

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

    def retrieve(self, query: str, *, top_k: int | None = None, layers: tuple[str, ...] = PERSISTED_LAYERS) -> list[tuple[MemoryItem, float]]:
        query = query.strip()
        if not query:
            return []
        r = self.cfg.retrieval
        top_k = top_k or r.top_k
        now = time.time()
        query_grams = _char_bigrams(query)
        scored: list[tuple[MemoryItem, float]] = []
        for layer in layers:
            layer_cfg = self._layer_cfg(layer)
            for item in self._items[layer]:
                strength = item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now)
                sim = _jaccard(query_grams, _char_bigrams(item.text))
                score = r.similarity_weight * sim + r.importance_weight * item.importance + r.strength_weight * min(1.0, strength)
                # 与查询毫无字面关联的记忆不注入（核心画像除外，画像常与任意话题相关）
                if sim <= 0 and layer != "core":
                    continue
                if score >= r.min_score:
                    scored.append((item, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = scored[:top_k]
        promoted = False
        for item, _score in result:
            self._reinforce(item)
            promoted = self._maybe_promote(item) or promoted
        if result:
            self.save()
        return result

    def format_context(self, pairs: list[tuple[MemoryItem, float]]) -> str:
        if not pairs:
            return ""
        lines = [f"- [{LAYER_LABELS.get(item.layer, item.layer)}] {item.text}" for item, _ in pairs]
        return "以下是你的长期记忆中与当前对话相关的内容，可参考但不要机械复述：\n" + "\n".join(lines)

    # ------------------------------------------------------------- forgetting

    def sweep(self) -> int:
        """遗忘清理：衰减到阈值以下的记忆被移除；超容量的层淘汰最弱项。返回移除数量。"""
        if not self.cfg.forgetting.enabled:
            return 0
        removed = 0
        now = time.time()
        for layer in PERSISTED_LAYERS:
            layer_cfg = self._layer_cfg(layer)
            kept: list[MemoryItem] = []
            for item in self._items[layer]:
                strength = item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now)
                if strength < layer_cfg.min_strength:
                    removed += 1
                    logger.debug("forgetting %s memory %s: %s", layer, item.id, item.text[:60])
                else:
                    kept.append(item)
            self._items[layer] = kept
            removed += self._enforce_capacity(layer, now=now)
        if removed:
            self.save()
            logger.info("memory sweep removed %d items", removed)
        return removed

    def clear(self) -> None:
        self._items = {layer: [] for layer in PERSISTED_LAYERS}
        self.save()

    def stats(self) -> dict:
        return {layer: len(items) for layer, items in self._items.items()}

    def items(self, layer: str) -> list[MemoryItem]:
        return list(self._items[layer])

    # ---------------------------------------------------------------- helpers

    def _layer_cfg(self, layer: str) -> LayerConfig:
        return getattr(self.cfg.layers, layer)

    def _find_duplicate(self, layer: str, text: str) -> MemoryItem | None:
        normalized = _normalize(text)
        for item in self._items[layer]:
            if _normalize(item.text) == normalized:
                return item
        return None

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
        self._enforce_capacity("semantic", now=time.time())
        logger.info("promoted episodic memory to semantic: %s", item.text[:60])
        return True

    def _enforce_capacity(self, layer: str, *, now: float) -> int:
        layer_cfg = self._layer_cfg(layer)
        items = self._items[layer]
        overflow = len(items) - layer_cfg.capacity
        if overflow <= 0:
            return 0
        items.sort(key=lambda item: item.effective_strength(half_life_hours=layer_cfg.half_life_hours, now=now))
        evicted = items[:overflow]
        self._items[layer] = items[overflow:]
        for item in evicted:
            logger.debug("evicting %s memory over capacity: %s", layer, item.text[:60])
        return overflow


def _normalize(text: str) -> str:
    return "".join(text.split()).lower()


def _is_raw_turn_memory(item: MemoryItem) -> bool:
    return item.source == "turn" or item.text.startswith("用户说：") or "；我回答：" in item.text


def _char_bigrams(text: str) -> set[str]:
    text = _normalize(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)
