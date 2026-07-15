from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._logging import get_logger
from .config import UpstreamConfig

logger = get_logger(__name__)


class UpstreamClient:
    """OpenAI 兼容上游（LM Studio）的流式客户端，支持工具调用。

    stream_chat 产出事件：
      {"type": "delta", "text": str}                        — 正文增量
      {"type": "tool_calls", "calls": [{"id","name","arguments"}]} — 模型请求调用工具
      {"type": "done", "finish_reason": str | None}          — 流结束
    """

    def __init__(self, cfg: UpstreamConfig):
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_sec),
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[bool, str]:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            model_ids = [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)]
            if self.cfg.model in model_ids:
                return True, f"model available: {self.cfg.model}"
            if model_ids:
                return False, f"configured model not listed. available={', '.join(model_ids[:5])}"
            return False, "upstream returned no loaded models"
        except Exception as exc:
            return False, str(exc)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)

        started = time.monotonic()
        chars = 0
        finish_reason: str | None = None
        tool_calls: dict[int, dict] = {}
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                data = _loads_json(line)
                if data is None:
                    continue
                for choice in data.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        chars += len(str(content))
                        yield {"type": "delta", "text": str(content)}
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index") or 0)
                        slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = str(tc["id"])
                        fn = tc.get("function") or {}
                        if fn.get("name") and not slot["name"]:
                            slot["name"] = str(fn["name"])
                        if fn.get("arguments"):
                            slot["arguments"] += str(fn["arguments"])
        if tool_calls:
            calls = [tool_calls[index] for index in sorted(tool_calls)]
            logger.info("upstream requested %d tool call(s): %s", len(calls), [c["name"] for c in calls])
            yield {"type": "tool_calls", "calls": calls}
        logger.info(
            "upstream stream done model=%s chars=%d tool_calls=%d finish=%s elapsed=%.2fs",
            self.cfg.model,
            chars,
            len(tool_calls),
            finish_reason,
            time.monotonic() - started,
        )
        yield {"type": "done", "finish_reason": finish_reason}


def _loads_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
