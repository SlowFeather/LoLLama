from __future__ import annotations

import asyncio
from pathlib import Path

from lollama.agent import Agent
from lollama.config import Config
from lollama.memory import MemoryManager
from lollama.tools import Tool, ToolRegistry


class FakeUpstream:
    """假上游：第一轮请求工具，之后输出正文；记忆提炼请求返回固定 JSON。"""

    def __init__(self, *, tool_name: str | None = None, reply: str = "你好呀。", extraction: str = "[]"):
        self.tool_name = tool_name
        self.reply = reply
        self.extraction = extraction
        self.calls: list[list[dict]] = []

    async def stream_chat(self, messages, *, tools=None):
        self.calls.append(list(messages))
        if _is_extraction(messages):
            yield {"type": "delta", "text": self.extraction}
            yield {"type": "done", "finish_reason": "stop"}
            return
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if self.tool_name and tools and not has_tool_result:
            yield {"type": "delta", "text": "让我看看。"}
            yield {"type": "tool_calls", "calls": [{"id": "call_1", "name": self.tool_name, "arguments": "{}"}]}
            yield {"type": "done", "finish_reason": "tool_calls"}
            return
        for ch in self.reply:
            yield {"type": "delta", "text": ch}
        yield {"type": "done", "finish_reason": "stop"}


def _is_extraction(messages) -> bool:
    return any("记忆整理器" in str(m.get("content")) for m in messages if m.get("role") == "system")


def make_agent(tmp_path: Path, upstream: FakeUpstream, *, tool: Tool | None = None, extraction: bool = False) -> Agent:
    cfg = Config()
    cfg.paths.memory_dir = str(tmp_path / "memory")
    cfg.paths.workspace_dir = str(tmp_path / "workspace")
    cfg.memory.extraction.enabled = extraction
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    memory = MemoryManager(cfg.memory, cfg.paths.memory_dir)
    return Agent(cfg, upstream=upstream, memory=memory, tools=registry)


async def collect(agent: Agent, text: str) -> list[dict]:
    return [event async for event in agent.respond([{"role": "user", "content": text}])]


async def test_plain_reply_streams_and_records_memory(tmp_path: Path) -> None:
    upstream = FakeUpstream(reply="今天天气不错。")
    agent = make_agent(tmp_path, upstream)
    events = await collect(agent, "今天天气怎么样")

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == "今天天气不错。"
    assert events[-1] == {"type": "done", "text": "今天天气不错。"}
    # 每轮对话写入情景记忆
    assert agent.memory.stats()["episodic"] == 1
    await agent.close()


async def test_tool_loop_executes_and_feeds_result_back(tmp_path: Path) -> None:
    calls: list[dict] = []

    def handler(args: dict) -> str:
        calls.append(args)
        return "42"

    tool = Tool(name="answer", description="answer", parameters={"type": "object", "properties": {}}, handler=handler)
    upstream = FakeUpstream(tool_name="answer", reply="答案是42。")
    agent = make_agent(tmp_path, upstream, tool=tool)
    events = await collect(agent, "宇宙的答案是什么")

    kinds = [e["type"] for e in events]
    assert kinds.count("tool") == 2  # start + done
    assert calls == [{}]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"].endswith("答案是42。")
    # 第二轮请求里带上了 tool 结果
    final_round = upstream.calls[-1]
    assert any(m.get("role") == "tool" and m.get("content") == "42" for m in final_round)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in final_round)
    await agent.close()


async def test_memory_injected_into_system_prompt(tmp_path: Path) -> None:
    upstream = FakeUpstream(reply="记得，你的猫叫团子。")
    agent = make_agent(tmp_path, upstream)
    agent.memory.add("semantic", "用户养了一只叫团子的猫", importance=0.9)

    await collect(agent, "还记得我的猫团子吗")
    system_prompt = upstream.calls[0][0]["content"]
    assert "团子" in system_prompt
    assert "长期记忆" in system_prompt
    await agent.close()


async def test_time_query_can_use_time_tool_without_injected_time(tmp_path: Path) -> None:
    upstream = FakeUpstream(tool_name="get_current_time", reply="现在是北京时间21点05分。")
    tool = Tool(
        name="get_current_time",
        description="time",
        parameters={"type": "object", "properties": {}},
        handler=lambda args: "2026-07-05 21:05:00 Sunday",
    )
    agent = make_agent(tmp_path, upstream, tool=tool)
    agent.memory.add("semantic", "用户之前问过现在几点；当时是北京时间08点00分", importance=0.9)
    agent.memory.add("procedural", "用户希望被告知时间时显示为北京时间", importance=0.9)

    events = await collect(agent, "帮我查一下现在是几点")

    assert any(e["type"] == "tool" and e["name"] == "get_current_time" for e in events)
    assert events[-1]["type"] == "done"
    assert "北京时间" in events[-1]["text"]
    system_prompt = upstream.calls[0][0]["content"]
    assert "当前时间：" not in system_prompt
    assert "必须调用 get_current_time 工具" in system_prompt
    assert "工具结果优先于记忆和常识" in system_prompt
    assert "只有用户明确要求写入时才调用 write_file 或 memory_save" in system_prompt
    assert "北京时间08点00分" not in system_prompt
    assert "用户希望被告知时间时显示为北京时间" in system_prompt
    assert agent.memory.stats()["episodic"] == 0
    await agent.close()


async def test_extraction_writes_layered_memories(tmp_path: Path) -> None:
    extraction = '[{"layer":"core","text":"用户叫小明","importance":0.9},{"layer":"procedural","text":"用户喜欢简短回答","importance":0.8}]'
    upstream = FakeUpstream(reply="好的小明。", extraction=extraction)
    agent = make_agent(tmp_path, upstream, extraction=True)

    await collect(agent, "我叫小明，以后回答短一点")
    await asyncio.gather(*agent._background, return_exceptions=True)
    stats = agent.memory.stats()
    assert stats["core"] == 1
    assert stats["procedural"] == 1
    assert agent.memory.items("core")[0].text == "用户叫小明"
    await agent.close()
