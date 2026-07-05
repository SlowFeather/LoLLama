from __future__ import annotations

from pathlib import Path

import pytest

from lollama.config import MemoryConfig, ToolsConfig
from lollama.memory import MemoryManager
from lollama.tools import build_registry


def make_registry(tmp_path: Path, *, with_memory: bool = False, shell: bool = False):
    cfg = ToolsConfig()
    cfg.shell.enabled = shell
    memory = MemoryManager(MemoryConfig(), tmp_path / "memory") if with_memory else None
    return build_registry(cfg, workspace_dir=tmp_path / "workspace", memory=memory), memory


async def test_registry_specs_are_openai_format(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    specs = registry.specs()
    assert specs, "expected builtin tools registered"
    for spec in specs:
        assert spec["type"] == "function"
        assert set(spec["function"]) == {"name", "description", "parameters"}


async def test_calculator(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    assert await registry.call("calculator", '{"expression": "(3+4)*2"}') == "14"
    result = await registry.call("calculator", '{"expression": "__import__(\'os\')"}')
    assert result.startswith("错误")


async def test_file_tools_sandboxed(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    assert "已写入" in await registry.call("write_file", '{"path": "notes/a.txt", "content": "你好"}')
    assert await registry.call("read_file", '{"path": "notes/a.txt"}') == "你好"
    listing = await registry.call("list_dir", "{}")
    assert "notes" in listing

    escaped = await registry.call("read_file", '{"path": "../outside.txt"}')
    assert escaped.startswith("错误")
    escaped = await registry.call("write_file", '{"path": "..\\\\evil.txt", "content": "x"}')
    assert escaped.startswith("错误")


async def test_memory_tools(tmp_path: Path) -> None:
    registry, memory = make_registry(tmp_path, with_memory=True)
    saved = await registry.call("memory_save", '{"text": "用户喜欢简短回答", "layer": "procedural", "importance": 0.9}')
    assert "已记住" in saved
    assert memory.stats()["procedural"] == 1
    found = await registry.call("memory_search", '{"query": "简短回答"}')
    assert "用户喜欢简短回答" in found


async def test_unknown_tool_and_bad_arguments(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    assert (await registry.call("no_such_tool", "{}")).startswith("错误")
    assert (await registry.call("calculator", "not json")).startswith("错误")


async def test_shell_disabled_by_default(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    assert "run_shell" not in registry.names()


async def test_shell_when_enabled(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path, shell=True)
    result = await registry.call("run_shell", '{"command": "echo hello"}')
    assert "exit=0" in result
    assert "hello" in result
