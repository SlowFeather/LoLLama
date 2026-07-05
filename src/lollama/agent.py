from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from ._logging import get_logger
from .config import Config
from .memory import MemoryManager
from .memory.extractor import extract_memories
from .tools import ToolRegistry
from .tools.builtin import DEFAULT_TOOL_LABELS
from .upstream import UpstreamClient

logger = get_logger(__name__)

# 后台事件（如记忆提炼完成）的回调：接收一个 status 事件 dict
BackgroundNotify = Callable[[dict], Awaitable[None]]

_PUMP_END = object()


class Agent:
    """回复编排：长期记忆注入 → 上游流式生成（含工具调用循环）→ 记忆回写。

    respond 产出事件：
      {"type":"status","stage",...,"announce"}  — 生命周期状态钩子（见 config status 段）
      {"type":"delta","text"}                    — 正文增量
      {"type":"tool","name","status","detail"}   — 工具调用（与 status 并存，保持兼容）
      {"type":"done","text"}                     — 回复完成

    status.stage 全集：
      memory_recall / llm_request / llm_waiting / llm_first_token /
      tool_start / tool_waiting / tool_done / tool_error / memory_extracted
    （accepted / canceled / error / done 由服务层补充）
    """

    def __init__(
        self,
        cfg: Config,
        *,
        upstream: UpstreamClient,
        memory: MemoryManager | None,
        tools: ToolRegistry,
    ) -> None:
        self.cfg = cfg
        self.upstream = upstream
        self.memory = memory
        self.tools = tools
        self._background: set[asyncio.Task] = set()

    async def close(self) -> None:
        for task in list(self._background):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def respond(
        self,
        messages: list[dict],
        *,
        background_notify: BackgroundNotify | None = None,
    ) -> AsyncIterator[dict]:
        user_text = _last_user_text(messages)
        prompt, recalled = self._build_messages(messages, user_text)
        if recalled:
            event = self.make_status("memory_recall", count=recalled)
            if event is not None:
                yield event
        tool_specs = self.tools.specs() if self.cfg.tools.enabled and len(self.tools) else None
        full_text: list[str] = []

        for round_index in range(self.cfg.tools.max_rounds + 1):
            allow_tools = tool_specs if round_index < self.cfg.tools.max_rounds else None
            event = self.make_status("llm_request", round=round_index + 1)
            if event is not None:
                yield event
            pending_calls: list[dict] = []
            async for event in self._stream_llm_with_waiting(prompt, allow_tools):
                if event["type"] == "delta":
                    full_text.append(event["text"])
                    yield event
                elif event["type"] == "tool_calls":
                    pending_calls = event["calls"]
                else:
                    yield event  # status 事件（llm_waiting / llm_first_token）
            if not pending_calls:
                break
            prompt.append(
                {
                    "role": "assistant",
                    "content": "".join(full_text) or None,
                    "tool_calls": [
                        {
                            "id": call["id"] or f"call_{index}",
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
                        }
                        for index, call in enumerate(pending_calls)
                    ],
                }
            )
            for index, call in enumerate(pending_calls):
                name = call["name"]
                brief = _brief_args(call["arguments"])
                event = self.make_status("tool_start", name=name, label=self.tool_label(name), arguments=brief)
                if event is not None:
                    yield event
                yield {"type": "tool", "name": name, "status": "start", "detail": brief}
                result = ""
                async for event in self._call_tool_with_waiting(name, call["arguments"] or "{}"):
                    if event["type"] == "tool_result":
                        result = event["result"]
                    else:
                        yield event  # tool_waiting 状态
                logger.info("tool %s -> %s", name, result[:120])
                stage = "tool_error" if result.startswith("错误") else "tool_done"
                event = self.make_status(stage, name=name, label=self.tool_label(name), result=result[:200])
                if event is not None:
                    yield event
                yield {"type": "tool", "name": name, "status": "done", "detail": result[:200]}
                prompt.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"] or f"call_{index}",
                        "content": result,
                    }
                )
        else:
            logger.warning("tool loop hit max_rounds=%d without final answer", self.cfg.tools.max_rounds)

        assistant_text = "".join(full_text).strip()
        self._write_back_memory(user_text, assistant_text, background_notify)
        yield {"type": "done", "text": assistant_text}

    # ---------------------------------------------------------------- status

    def make_status(self, stage: str, **detail) -> dict | None:
        """构造一个 status 事件；status.enabled=false 时返回 None。"""
        if not self.cfg.status.enabled:
            return None
        template = getattr(self.cfg.status.announce, stage, "")
        announce = ""
        if template:
            try:
                announce = template.format(**detail)
            except (KeyError, IndexError, ValueError):
                announce = template
        return {"type": "status", "stage": stage, "announce": announce, **detail}

    def tool_label(self, name: str) -> str:
        labels = {**DEFAULT_TOOL_LABELS, **self.cfg.status.tool_labels}
        return labels.get(name, f"用一下 {name} 工具")

    async def _stream_llm_with_waiting(self, prompt: list[dict], tools: list[dict] | None) -> AsyncIterator[dict]:
        """转发上游事件；首字迟迟不来时按配置产出 llm_waiting 心跳和 llm_first_token。"""
        scfg = self.cfg.status
        queue: asyncio.Queue = asyncio.Queue()

        async def pump() -> None:
            try:
                async for event in self.upstream.stream_chat(prompt, tools=tools):
                    await queue.put(event)
                await queue.put(_PUMP_END)
            except BaseException as exc:  # noqa: BLE001 - 转交消费方抛出
                await queue.put(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise

        pump_task = asyncio.create_task(pump())
        got_first_token = False
        started = time.monotonic()
        next_waiting: float | None = None
        if scfg.enabled and scfg.llm_waiting_after_sec > 0:
            next_waiting = started + scfg.llm_waiting_after_sec
        try:
            while True:
                timeout = None
                if not got_first_token and next_waiting is not None:
                    timeout = max(0.05, next_waiting - time.monotonic())
                try:
                    item = (
                        await asyncio.wait_for(queue.get(), timeout=timeout)
                        if timeout is not None
                        else await queue.get()
                    )
                except TimeoutError:
                    event = self.make_status("llm_waiting", waited_sec=round(time.monotonic() - started, 1))
                    if event is not None:
                        yield event
                    if scfg.llm_waiting_repeat_sec > 0:
                        next_waiting = time.monotonic() + scfg.llm_waiting_repeat_sec
                    else:
                        next_waiting = None
                    continue
                if item is _PUMP_END:
                    return
                if isinstance(item, BaseException):
                    raise item
                if item["type"] == "delta" and not got_first_token:
                    got_first_token = True
                    event = self.make_status("llm_first_token", waited_sec=round(time.monotonic() - started, 1))
                    if event is not None:
                        yield event
                if item["type"] == "done":
                    continue
                yield item
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task

    async def _call_tool_with_waiting(self, name: str, arguments: str) -> AsyncIterator[dict]:
        """执行工具；耗时过长时按配置产出 tool_waiting 心跳，最后产出 tool_result。"""
        scfg = self.cfg.status
        task = asyncio.create_task(self.tools.call(name, arguments))
        started = time.monotonic()
        next_waiting: float | None = None
        if scfg.enabled and scfg.tool_waiting_after_sec > 0:
            next_waiting = started + scfg.tool_waiting_after_sec
        try:
            while True:
                timeout = None
                if next_waiting is not None:
                    timeout = max(0.05, next_waiting - time.monotonic())
                try:
                    result = (
                        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                        if timeout is not None
                        else await task
                    )
                except TimeoutError:
                    event = self.make_status(
                        "tool_waiting",
                        name=name,
                        label=self.tool_label(name),
                        waited_sec=round(time.monotonic() - started, 1),
                    )
                    if event is not None:
                        yield event
                    if scfg.tool_waiting_repeat_sec > 0:
                        next_waiting = time.monotonic() + scfg.tool_waiting_repeat_sec
                    else:
                        next_waiting = None
                    continue
                yield {"type": "tool_result", "result": result}
                return
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # ---------------------------------------------------------------- prompt

    def _build_messages(self, messages: list[dict], user_text: str) -> tuple[list[dict], int]:
        system_parts = [self.cfg.agent.system_prompt]
        if self.cfg.agent.inject_time:
            system_parts.append(f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S %A')}")
        recalled = 0
        if self.memory is not None and self.cfg.agent.inject_memory and user_text:
            pairs = self.memory.retrieve(user_text)
            recalled = len(pairs)
            context = self.memory.format_context(pairs)
            if context:
                system_parts.append(context)
        prompt: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        # 调用方自带的 system 提示保留为附加约束，避免覆盖记忆注入
        for message in messages:
            if message.get("role") == "system":
                prompt.append({"role": "system", "content": message.get("content", "")})
            else:
                prompt.append(dict(message))
        return prompt, recalled

    # ---------------------------------------------------------------- memory

    def _write_back_memory(
        self,
        user_text: str,
        assistant_text: str,
        background_notify: BackgroundNotify | None,
    ) -> None:
        if self.memory is None or not user_text:
            return
        try:
            self.memory.record_turn(user_text, assistant_text)
        except Exception:
            logger.exception("failed to record episodic memory")
        extraction = self.cfg.memory.extraction
        if not extraction.enabled or len(user_text) < extraction.min_turn_chars:
            return
        task = asyncio.create_task(self._extract_and_store(user_text, assistant_text, background_notify))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _extract_and_store(
        self,
        user_text: str,
        assistant_text: str,
        background_notify: BackgroundNotify | None,
    ) -> None:
        try:
            items = await extract_memories(self.upstream, user_text, assistant_text, self.cfg.memory.extraction)
            for entry in items:
                self.memory.add(entry["layer"], entry["text"], importance=entry["importance"], source="extraction")
            if items:
                logger.info("extracted %d memory item(s) from turn", len(items))
                if background_notify is not None:
                    event = self.make_status(
                        "memory_extracted",
                        count=len(items),
                        items=[entry["text"] for entry in items],
                    )
                    if event is not None:
                        with contextlib.suppress(Exception):
                            await background_notify(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("memory extraction task failed")


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _brief_args(arguments: str) -> str:
    arguments = (arguments or "").strip()
    if not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
        return json.dumps(parsed, ensure_ascii=False)[:200]
    except json.JSONDecodeError:
        return arguments[:200]
