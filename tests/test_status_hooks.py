from __future__ import annotations

import asyncio
from pathlib import Path

from lollama.agent import Agent
from lollama.config import Config
from lollama.memory import MemoryManager
from lollama.tools import Tool, ToolRegistry


class FakeUpstream:
    """假上游：可配置首字延迟；带工具时第一轮请求工具。"""

    def __init__(self, *, reply: str = "你好。", first_token_delay: float = 0.0, tool_name: str | None = None):
        self.reply = reply
        self.first_token_delay = first_token_delay
        self.tool_name = tool_name

    async def stream_chat(self, messages, *, tools=None):
        if _is_extraction(messages):
            yield {"type": "delta", "text": '[{"layer":"semantic","text":"用户在测试状态钩子","importance":0.6}]'}
            yield {"type": "done", "finish_reason": "stop"}
            return
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if self.first_token_delay:
            await asyncio.sleep(self.first_token_delay)
        if self.tool_name and tools and not has_tool_result:
            yield {"type": "tool_calls", "calls": [{"id": "c1", "name": self.tool_name, "arguments": "{}"}]}
            yield {"type": "done", "finish_reason": "tool_calls"}
            return
        for ch in self.reply:
            yield {"type": "delta", "text": ch}
        yield {"type": "done", "finish_reason": "stop"}


def _is_extraction(messages) -> bool:
    return any("记忆整理器" in str(m.get("content")) for m in messages if m.get("role") == "system")


def make_agent(tmp_path: Path, upstream, *, tool: Tool | None = None, extraction: bool = False) -> Agent:
    cfg = Config()
    cfg.paths.memory_dir = str(tmp_path / "memory")
    cfg.paths.workspace_dir = str(tmp_path / "workspace")
    cfg.memory.extraction.enabled = extraction
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    memory = MemoryManager(cfg.memory, cfg.paths.memory_dir)
    return Agent(cfg, upstream=upstream, memory=memory, tools=registry)


async def collect(agent: Agent, text: str, **kwargs) -> list[dict]:
    return [event async for event in agent.respond([{"role": "user", "content": text}], **kwargs)]


def stages(events: list[dict]) -> list[str]:
    return [e["stage"] for e in events if e["type"] == "status"]


async def test_status_events_on_plain_reply(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, FakeUpstream(reply="早上好。"))
    events = await collect(agent, "早")
    assert stages(events) == ["llm_request", "llm_first_token"]
    request = next(e for e in events if e["type"] == "status" and e["stage"] == "llm_request")
    assert request["round"] == 1
    assert request["announce"] == ""  # 默认不播报
    await agent.close()


async def test_memory_recall_status(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, FakeUpstream(reply="记得。"))
    agent.memory.add("semantic", "用户养了一只叫团子的猫", importance=0.9)
    events = await collect(agent, "还记得团子吗")
    recall = next(e for e in events if e["type"] == "status" and e["stage"] == "memory_recall")
    assert recall["count"] >= 1
    await agent.close()


async def test_llm_waiting_heartbeat(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, FakeUpstream(reply="来了。", first_token_delay=0.4))
    agent.cfg.status.llm_waiting_after_sec = 0.1
    agent.cfg.status.llm_waiting_repeat_sec = 0.1
    events = await collect(agent, "你好")
    waiting = [e for e in events if e["type"] == "status" and e["stage"] == "llm_waiting"]
    assert len(waiting) >= 2
    assert waiting[0]["announce"] == "让我想想。"
    # 首字之后不再有等待心跳
    kinds = [(e["type"], e.get("stage")) for e in events]
    assert kinds.index(("status", "llm_first_token")) > kinds.index(("status", "llm_waiting"))
    await agent.close()


async def test_tool_status_with_announce_template(tmp_path: Path) -> None:
    tool = Tool(name="calculator", description="calc", parameters={"type": "object", "properties": {}}, handler=lambda args: "42")
    agent = make_agent(tmp_path, FakeUpstream(reply="是42。", tool_name="calculator"), tool=tool)
    events = await collect(agent, "算一下")
    start = next(e for e in events if e["type"] == "status" and e["stage"] == "tool_start")
    assert start["announce"] == "我算一下。"
    assert start["name"] == "calculator"
    done = next(e for e in events if e["type"] == "status" and e["stage"] == "tool_done")
    assert done["result"] == "42"
    await agent.close()


async def test_tool_error_status(tmp_path: Path) -> None:
    def boom(args: dict) -> str:
        raise RuntimeError("炸了")

    tool = Tool(name="calculator", description="calc", parameters={"type": "object", "properties": {}}, handler=boom)
    agent = make_agent(tmp_path, FakeUpstream(reply="出问题了。", tool_name="calculator"), tool=tool)
    events = await collect(agent, "算一下")
    error = next(e for e in events if e["type"] == "status" and e["stage"] == "tool_error")
    assert error["announce"] == "工具出了点问题，我再想想。"
    await agent.close()


async def test_slow_tool_waiting_heartbeat(tmp_path: Path) -> None:
    async def slow(args: dict) -> str:
        await asyncio.sleep(0.35)
        return "done"

    tool = Tool(name="calculator", description="calc", parameters={"type": "object", "properties": {}}, handler=slow)
    agent = make_agent(tmp_path, FakeUpstream(reply="好了。", tool_name="calculator"), tool=tool)
    agent.cfg.status.tool_waiting_after_sec = 0.1
    agent.cfg.status.tool_waiting_repeat_sec = 0.1
    events = await collect(agent, "跑个慢任务")
    waiting = [e for e in events if e["type"] == "status" and e["stage"] == "tool_waiting"]
    assert len(waiting) >= 2
    assert waiting[0]["announce"] == "还在处理，稍等。"
    await agent.close()


async def test_status_disabled_emits_nothing(tmp_path: Path) -> None:
    tool = Tool(name="calculator", description="calc", parameters={"type": "object", "properties": {}}, handler=lambda args: "42")
    agent = make_agent(tmp_path, FakeUpstream(reply="是42。", tool_name="calculator"), tool=tool)
    agent.cfg.status.enabled = False
    events = await collect(agent, "算一下")
    assert stages(events) == []
    # 正文与工具事件不受影响
    assert any(e["type"] == "tool" for e in events)
    assert events[-1]["type"] == "done"
    await agent.close()


async def test_custom_tool_label_and_announce(tmp_path: Path) -> None:
    tool = Tool(name="calculator", description="calc", parameters={"type": "object", "properties": {}}, handler=lambda args: "42")
    agent = make_agent(tmp_path, FakeUpstream(reply="是42。", tool_name="calculator"), tool=tool)
    agent.cfg.status.tool_labels = {"calculator": "按计算器"}
    agent.cfg.status.announce.tool_start = "稍等，{label}。"
    events = await collect(agent, "算一下")
    start = next(e for e in events if e["type"] == "status" and e["stage"] == "tool_start")
    assert start["announce"] == "稍等，按计算器。"
    await agent.close()


async def test_memory_extracted_background_notify(tmp_path: Path) -> None:
    received: list[dict] = []

    async def notify(event: dict) -> None:
        received.append(event)

    agent = make_agent(tmp_path, FakeUpstream(reply="好的。"), extraction=True)
    await collect(agent, "记住我在测试状态钩子", background_notify=notify)
    await asyncio.gather(*agent._background, return_exceptions=True)
    assert received, "expected memory_extracted notification"
    assert received[0]["stage"] == "memory_extracted"
    assert received[0]["count"] == 1
    await agent.close()
