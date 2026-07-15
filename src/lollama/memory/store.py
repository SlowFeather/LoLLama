from __future__ import annotations

import json
import sqlite3
from array import array
from collections.abc import Iterable
from pathlib import Path

from lollama._logging import get_logger

from .textutil import normalize

logger = get_logger(__name__)

_MAX_FTS_WINDOWS = 32


class SqliteMemoryStore:
    """记忆的 SQLite 持久化：行级增量写入（WAL）+ FTS5 trigram 全文索引 + 向量表。

    相比整文件 JSON 重写：写入只碰被改动的行，检索可用 BM25 缩小候选集；
    trigram 分词对中文按 3 字滑窗建索引，短于 3 字的查询由上层 bigram 通道兜底。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._fts_ok = self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            logger.exception("failed to close memory store %s", self.path)

    def _init_schema(self) -> bool:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    rowid INTEGER PRIMARY KEY,
                    id TEXT UNIQUE NOT NULL,
                    layer TEXT NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    strength REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    meta TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL
                )
                """
            )
        for tokenizer in ("trigram", "unicode61"):
            try:
                with self._conn:
                    self._conn.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(norm_text, tokenize='{tokenizer}')"
                    )
                return True
            except sqlite3.OperationalError as exc:
                logger.warning("fts5 tokenizer %s unavailable: %s", tokenizer, exc)
        logger.warning("FTS5 unavailable; memory retrieval falls back to bigram channel only")
        return False

    # ------------------------------------------------------------------ items

    def load(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, layer, text, importance, strength, created_at, last_accessed, hits, source, meta FROM memories"
        ).fetchall()
        items: list[dict] = []
        for row in rows:
            try:
                meta = json.loads(row[9]) if row[9] else {}
            except json.JSONDecodeError:
                meta = {}
            items.append(
                {
                    "id": row[0],
                    "layer": row[1],
                    "text": row[2],
                    "importance": row[3],
                    "strength": row[4],
                    "created_at": row[5],
                    "last_accessed": row[6],
                    "hits": row[7],
                    "source": row[8],
                    "meta": meta if isinstance(meta, dict) else {},
                }
            )
        return items

    def upsert_many(self, items: Iterable) -> None:
        """写入或更新记忆行并同步 FTS 索引；items 为 MemoryItem 或同结构对象。"""
        with self._conn:
            for item in items:
                self._conn.execute(
                    """
                    INSERT INTO memories (id, layer, text, importance, strength, created_at, last_accessed, hits, source, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        layer = excluded.layer,
                        text = excluded.text,
                        importance = excluded.importance,
                        strength = excluded.strength,
                        created_at = excluded.created_at,
                        last_accessed = excluded.last_accessed,
                        hits = excluded.hits,
                        source = excluded.source,
                        meta = excluded.meta
                    """,
                    (
                        item.id,
                        item.layer,
                        item.text,
                        item.importance,
                        item.strength,
                        item.created_at,
                        item.last_accessed,
                        item.hits,
                        item.source,
                        json.dumps(item.meta, ensure_ascii=False),
                    ),
                )
                if self._fts_ok:
                    row = self._conn.execute("SELECT rowid FROM memories WHERE id = ?", (item.id,)).fetchone()
                    if row is not None:
                        self._conn.execute("DELETE FROM mem_fts WHERE rowid = ?", (row[0],))
                        self._conn.execute(
                            "INSERT INTO mem_fts (rowid, norm_text) VALUES (?, ?)", (row[0], normalize(item.text))
                        )

    def delete_many(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if not ids:
            return
        with self._conn:
            for item_id in ids:
                if self._fts_ok:
                    row = self._conn.execute("SELECT rowid FROM memories WHERE id = ?", (item_id,)).fetchone()
                    if row is not None:
                        self._conn.execute("DELETE FROM mem_fts WHERE rowid = ?", (row[0],))
                self._conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
                self._conn.execute("DELETE FROM embeddings WHERE id = ?", (item_id,))

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM embeddings")
            if self._fts_ok:
                self._conn.execute("DELETE FROM mem_fts")

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # -------------------------------------------------------------------- fts

    def fts_search(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        """按查询的 3 字滑窗做 OR 匹配，返回 (id, bm25) 升序（bm25 越小越相关）。"""
        if not self._fts_ok:
            return []
        norm = normalize(query)
        if len(norm) < 3:
            return []
        windows: list[str] = []
        seen: set[str] = set()
        for i in range(len(norm) - 2):
            window = norm[i : i + 3].replace('"', "")
            if len(window) == 3 and window not in seen:
                seen.add(window)
                windows.append(window)
            if len(windows) >= _MAX_FTS_WINDOWS:
                break
        if not windows:
            return []
        match = " OR ".join(f'"{window}"' for window in windows)
        try:
            rows = self._conn.execute(
                """
                SELECT m.id, bm25(mem_fts) AS rank
                FROM mem_fts JOIN memories m ON m.rowid = mem_fts.rowid
                WHERE mem_fts MATCH ?
                ORDER BY rank LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("fts search failed for %r: %s", query[:60], exc)
            return []
        return [(str(row[0]), float(row[1])) for row in rows]

    # ------------------------------------------------------------- embeddings

    def load_embeddings(self) -> dict[str, array]:
        vectors: dict[str, array] = {}
        for item_id, blob in self._conn.execute("SELECT id, vec FROM embeddings"):
            vec = array("f")
            vec.frombytes(blob)
            vectors[str(item_id)] = vec
        return vectors

    def set_embedding(self, item_id: str, model: str, vec: array) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (id, model, dim, vec) VALUES (?, ?, ?, ?)",
                (item_id, model, len(vec), vec.tobytes()),
            )
