from __future__ import annotations

import asyncio
import json
import re

from lollama._logging import get_logger
from lollama.config import ExtractionConfig

logger = get_logger(__name__)

VALID_LAYERS = {"episodic", "semantic", "procedural", "core"}

EXTRACTION_SYSTEM = (
    "你是一个记忆整理器。阅读一轮用户与助手的对话，提炼值得长期记住的信息。\n"
    "分层规则：\n"
    "- semantic: 关于用户或世界的客观事实（如“用户养了一只猫”）\n"
    "- procedural: 用户的偏好或做事方式（如“用户喜欢简短的回答”）\n"
    "- core: 稳定的用户核心画像（如姓名、职业、长期目标）\n"
    "只输出 JSON 数组，不要其他文字。每项形如 "
    '{"layer": "semantic", "text": "...", "importance": 0.7}，'
    "importance 取 0~1。没有值得记的就输出 []。"
)


async def extract_memories(upstream, user_text: str, assistant_text: str, cfg: ExtractionConfig) -> list[dict]:
    """用上游 LLM 从一轮对话中提炼记忆条目；失败时返回空列表。"""
    turn = f"用户：{user_text}\n助手：{assistant_text}"
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": turn},
    ]
    chunks: list[str] = []
    try:
        async with asyncio.timeout(cfg.timeout_sec):
            async for event in upstream.stream_chat(messages):
                if event["type"] == "delta":
                    chunks.append(event["text"])
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("memory extraction call failed: %s", exc)
        return []
    return parse_extraction("".join(chunks), max_items=cfg.max_items_per_turn)


def parse_extraction(text: str, *, max_items: int) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("memory extraction returned invalid JSON: %s", text[:200])
        return []
    if not isinstance(data, list):
        return []
    items: list[dict] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        layer = str(raw.get("layer") or "").strip()
        content = str(raw.get("text") or "").strip()
        if layer not in VALID_LAYERS or not content:
            continue
        try:
            importance = float(raw.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        items.append({"layer": layer, "text": content[:300], "importance": min(1.0, max(0.0, importance))})
        if len(items) >= max_items:
            break
    return items
