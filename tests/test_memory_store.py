from __future__ import annotations

import json
from array import array
from pathlib import Path

from lollama.config import MemoryConfig
from lollama.memory import MemoryManager
from lollama.memory.store import SqliteMemoryStore


def make_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(MemoryConfig(), tmp_path)


def test_sqlite_is_source_of_truth_and_mirror_exported(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.add("semantic", "用户养了一只叫团子的猫", importance=0.8)

    assert (tmp_path / "default.db").exists()
    mirror = json.loads((tmp_path / "default.json").read_text(encoding="utf-8"))
    assert len(mirror["semantic"]) == 1
    assert mirror["semantic"][0]["text"] == "用户养了一只叫团子的猫"

    reloaded = MemoryManager(manager.cfg, tmp_path)
    assert reloaded.stats()["semantic"] == 1


def test_legacy_json_migrates_into_sqlite(tmp_path: Path) -> None:
    cfg = MemoryConfig()
    legacy = {
        "episodic": [],
        "semantic": [
            {
                "id": "legacy1",
                "layer": "semantic",
                "text": "用户住在上海",
                "importance": 0.6,
                "strength": 1.0,
                "created_at": 10.0,
                "last_accessed": 20.0,
                "hits": 1,
                "source": "extraction",
                "meta": {},
            }
        ],
        "procedural": [],
        "core": [],
    }
    (tmp_path / f"{cfg.user_id}.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    manager = MemoryManager(cfg, tmp_path)
    assert manager.stats()["semantic"] == 1
    assert manager.store.count() == 1
    # 迁移后 JSON 只是镜像；改动落在 SQLite
    manager.add("core", "用户名字叫小明")
    reloaded = MemoryManager(cfg, tmp_path)
    assert reloaded.stats() == {"episodic": 0, "semantic": 1, "procedural": 0, "core": 1}


def test_fts_search_finds_chinese_substrings(tmp_path: Path) -> None:
    store = SqliteMemoryStore(tmp_path / "t.db")
    manager = make_manager(tmp_path)
    item = manager.add("semantic", "用户养了一只叫团子的猫", importance=0.8)
    hits = manager.store.fts_search("叫团子的猫今天好吗", limit=10)
    assert [hit[0] for hit in hits] == [item.id]
    # 查询不足 3 字时 FTS 不参与（由 bigram 通道兜底）
    assert manager.store.fts_search("团子", limit=10) == []
    store.close()


def test_retrieve_uses_vector_channel_when_query_vector_given(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    cat = manager.add("semantic", "用户养了一只宠物猫", importance=0.5)
    city = manager.add("semantic", "用户住在上海", importance=0.5)
    manager._vectors[cat.id] = array("f", [1.0, 0.0])
    manager._vectors[city.id] = array("f", [0.0, 1.0])

    # 查询与两条记忆都无字面重叠，只有向量通道能区分
    result = manager.retrieve("喵星人", query_vector=array("f", [1.0, 0.0]))
    assert result
    assert result[0][0].id == cat.id
    assert all(item.id != city.id for item, _ in result)


def test_contradiction_weakens_old_memory(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    old = manager.add("semantic", "用户喜欢吃苹果")
    before = old.strength
    new = manager.add("semantic", "用户不喜欢吃苹果")
    assert new is not old
    assert old.strength < before
    assert old.meta.get("contradicted_by") == new.id


def test_deleted_items_are_removed_from_store(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.cfg.layers.episodic.capacity = 1
    manager.add("episodic", "用户说：一号事件")
    manager.add("episodic", "用户说：二号事件")
    assert manager.store.count() == 1

    manager.clear()
    assert manager.store.count() == 0
    mirror = json.loads((tmp_path / "default.json").read_text(encoding="utf-8"))
    assert mirror == {"episodic": [], "semantic": [], "procedural": [], "core": []}
