from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from lollama._logging import get_logger

logger = get_logger(__name__)

ToolHandler = Callable[[dict], str | Awaitable[str]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: ToolHandler
    # 状态播报用的动词短语（如“数一下字数”）；None 则由上层回退到默认文案
    label: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def label(self, name: str) -> str | None:
        tool = self._tools.get(name)
        return tool.label if tool is not None else None

    def specs(self) -> list[dict]:
        """OpenAI tools 格式的工具定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def call(self, name: str, arguments_json: str) -> str:
        """执行工具并返回字符串结果；所有异常转为可读错误文本回传给模型。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 {name}"
        try:
            arguments = json.loads(arguments_json) if arguments_json.strip() else {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return f"错误：工具参数不是合法 JSON 对象: {exc}"
        try:
            result = tool.handler(arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return f"错误：工具 {name} 执行失败: {exc}"
