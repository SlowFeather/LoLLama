from __future__ import annotations

import time
from pathlib import Path

from lollama.config import MemoryConfig
from lollama.memory import MemoryManager
from lollama.memory.extractor import parse_extraction


def make_manager(tmp_path: Path, **overrides) -> MemoryManager:
    cfg = MemoryConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return MemoryManager(cfg, tmp_path)


def test_add_and_retrieve(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.add("semantic", "用户养了一只叫团子的猫", importance=0.8)
    manager.add("semantic", "用户住在上海", importance=0.6)

    result = manager.retrieve("我的猫团子今天怎么样")
    assert result, "expected at least one hit"
    assert result[0][0].text == "用户养了一只叫团子的猫"


def test_duplicate_add_reinforces_instead_of_duplicating(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    first = manager.add("semantic", "用户喜欢喝咖啡")
    second = manager.add("semantic", "用户 喜欢 喝咖啡")
    assert first is second
    assert manager.stats()["semantic"] == 1
    assert second.strength > 1.0


def test_recall_reinforces_and_promotes_episodic(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.cfg.promotion.episodic_hits_to_semantic = 2
    manager.add("episodic", "用户说：今天调试了 ChatCaht 的唤醒功能")

    manager.retrieve("ChatCaht 唤醒")
    assert manager.stats() == {"episodic": 1, "semantic": 0, "procedural": 0, "core": 0}
    manager.retrieve("ChatCaht 唤醒")
    # 第二次命中达到阈值，晋升为语义记忆
    assert manager.stats() == {"episodic": 0, "semantic": 1, "procedural": 0, "core": 0}
    promoted = manager.items("semantic")[0]
    assert promoted.hits == 2


def test_forgetting_removes_decayed_items(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    item = manager.add("episodic", "用户说：昨天天气不错")
    fresh = manager.add("episodic", "用户说：今天在写代码")
    # 手动把一条记忆的最后访问时间拨回 10 个半衰期之前
    half_life = manager.cfg.layers.episodic.half_life_hours
    item.last_accessed = time.time() - 10 * half_life * 3600

    removed = manager.sweep()
    assert removed == 1
    remaining = [entry.text for entry in manager.items("episodic")]
    assert remaining == [fresh.text]


def test_core_layer_never_decays(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    item = manager.add("core", "用户名字叫小明")
    item.last_accessed = time.time() - 365 * 24 * 3600
    assert manager.sweep() == 0
    assert manager.stats()["core"] == 1


def test_capacity_eviction_drops_weakest(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.cfg.layers.episodic.capacity = 2
    old = manager.add("episodic", "用户说：一号事件")
    old.last_accessed = time.time() - 48 * 3600  # 衰减一些
    manager.add("episodic", "用户说：二号事件")
    manager.add("episodic", "用户说：三号事件")
    texts = {entry.text for entry in manager.items("episodic")}
    assert len(texts) == 2
    assert "用户说：一号事件" not in texts


def test_persistence_roundtrip(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.add("core", "用户名字叫小明", importance=1.0)
    manager.add("procedural", "用户喜欢简短的回答", importance=0.9)

    reloaded = MemoryManager(manager.cfg, tmp_path)
    assert reloaded.stats() == {"episodic": 0, "semantic": 0, "procedural": 1, "core": 1}
    assert reloaded.items("core")[0].text == "用户名字叫小明"


def test_clear(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.add("semantic", "用户住在上海")
    manager.clear()
    assert manager.stats() == {"episodic": 0, "semantic": 0, "procedural": 0, "core": 0}
    assert MemoryManager(manager.cfg, tmp_path).stats()["semantic"] == 0


def test_record_turn_stores_episodic(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    item = manager.record_turn("今天写了什么代码", "你在写 LoLLama 的记忆系统")
    assert item is not None
    assert item.layer == "episodic"
    assert "今天写了什么代码" in item.text


def test_parse_extraction_lenient() -> None:
    text = '好的，以下是提炼结果：\n[{"layer":"core","text":"用户叫小明","importance":0.9},' \
        '{"layer":"bogus","text":"无效层"},{"layer":"semantic","text":"用户养猫","importance":"x"}]'
    items = parse_extraction(text, max_items=4)
    assert items == [
        {"layer": "core", "text": "用户叫小明", "importance": 0.9},
        {"layer": "semantic", "text": "用户养猫", "importance": 0.5},
    ]


def test_parse_extraction_garbage_returns_empty() -> None:
    assert parse_extraction("模型输出了一堆废话", max_items=4) == []
    assert parse_extraction("[not json", max_items=4) == []
