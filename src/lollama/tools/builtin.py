from __future__ import annotations

import ast
import asyncio
import operator
import subprocess
import time
from pathlib import Path

from lollama._logging import get_logger
from lollama.config import ToolsConfig
from lollama.memory import PERSISTED_LAYERS, MemoryManager

from .registry import Tool, ToolRegistry

logger = get_logger(__name__)

_MAX_FILE_CHARS = 20000

# 各工具在状态播报里的动词短语（status.tool_labels 可覆盖）
DEFAULT_TOOL_LABELS = {
    "get_current_time": "查一下时间",
    "calculator": "算一下",
    "read_file": "读一下文件",
    "write_file": "写一下文件",
    "list_dir": "看一下文件目录",
    "memory_search": "翻一下记忆",
    "memory_save": "记一下",
    "run_shell": "执行一下命令",
}


def build_registry(cfg: ToolsConfig, *, workspace_dir: str | Path, memory: MemoryManager | None) -> ToolRegistry:
    registry = ToolRegistry()
    if not cfg.enabled:
        return registry
    workspace = Path(workspace_dir).resolve()
    b = cfg.builtin

    if b.time:
        registry.register(
            Tool(
                name="get_current_time",
                description="获取当前本地日期和时间",
                parameters={"type": "object", "properties": {}},
                handler=lambda args: time.strftime("%Y-%m-%d %H:%M:%S %A"),
            )
        )
    if b.calculator:
        registry.register(
            Tool(
                name="calculator",
                description="计算一个算术表达式，支持 + - * / // % ** 和括号",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "算术表达式，如 (3+4)*2"}},
                    "required": ["expression"],
                },
                handler=lambda args: _calculate(str(args.get("expression", ""))),
            )
        )
    if b.read_file:
        registry.register(
            Tool(
                name="read_file",
                description="读取工作区内一个文本文件的内容",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "相对工作区的文件路径"}},
                    "required": ["path"],
                },
                handler=lambda args: _read_file(workspace, str(args.get("path", ""))),
            )
        )
    if b.write_file:
        registry.register(
            Tool(
                name="write_file",
                description="把文本内容写入工作区内的一个文件（覆盖写）",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作区的文件路径"},
                        "content": {"type": "string", "description": "要写入的文本内容"},
                    },
                    "required": ["path", "content"],
                },
                handler=lambda args: _write_file(workspace, str(args.get("path", "")), str(args.get("content", ""))),
            )
        )
    if b.list_dir:
        registry.register(
            Tool(
                name="list_dir",
                description="列出工作区内一个目录下的文件和子目录",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "相对工作区的目录路径，默认工作区根目录"}},
                },
                handler=lambda args: _list_dir(workspace, str(args.get("path", "") or ".")),
            )
        )
    if memory is not None and b.memory_search:
        registry.register(
            Tool(
                name="memory_search",
                description="在长期记忆（情景/事实/偏好/画像）中检索与查询相关的内容",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
                    "required": ["query"],
                },
                handler=lambda args: _memory_search(memory, str(args.get("query", ""))),
            )
        )
    if memory is not None and b.memory_save:
        registry.register(
            Tool(
                name="memory_save",
                description="把一条重要信息写入长期记忆。layer 取值: episodic(情景) semantic(事实) procedural(偏好) core(核心画像)",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "要记住的内容"},
                        "layer": {"type": "string", "enum": list(PERSISTED_LAYERS), "description": "记忆层"},
                        "importance": {"type": "number", "description": "重要度 0~1，默认 0.7"},
                    },
                    "required": ["text", "layer"],
                },
                handler=lambda args: _memory_save(memory, args),
            )
        )
    if cfg.shell.enabled:
        registry.register(
            Tool(
                name="run_shell",
                description="在本机执行一条 shell 命令并返回输出（谨慎使用）",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "要执行的命令"}},
                    "required": ["command"],
                },
                handler=lambda args: _run_shell(str(args.get("command", "")), timeout=cfg.shell.timeout_sec, cwd=workspace),
            )
        )
    return registry


# ------------------------------------------------------------------ handlers

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate(expression: str) -> str:
    if not expression.strip():
        return "错误：表达式为空"
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式元素: {ast.dump(node)[:60]}")


def _resolve_in_workspace(workspace: Path, relative: str) -> Path:
    if not relative:
        raise ValueError("path 不能为空")
    target = (workspace / relative).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError(f"路径越界：只允许访问工作区 {workspace} 内的文件")
    return target


def _read_file(workspace: Path, relative: str) -> str:
    target = _resolve_in_workspace(workspace, relative)
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_FILE_CHARS:
        return text[:_MAX_FILE_CHARS] + f"\n...（截断，共 {len(text)} 字符）"
    return text or "（空文件）"


def _write_file(workspace: Path, relative: str, content: str) -> str:
    target = _resolve_in_workspace(workspace, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {relative}（{len(content)} 字符）"


def _list_dir(workspace: Path, relative: str) -> str:
    target = _resolve_in_workspace(workspace, relative)
    if not target.exists():
        return "（目录不存在）"
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "（空目录）"
    return "\n".join(f"{'[目录] ' if entry.is_dir() else ''}{entry.name}" for entry in entries[:200])


def _memory_search(memory: MemoryManager, query: str) -> str:
    pairs = memory.retrieve(query)
    if not pairs:
        return "没有找到相关记忆"
    return "\n".join(f"[{item.layer}] {item.text}" for item, _score in pairs)


def _memory_save(memory: MemoryManager, args: dict) -> str:
    text = str(args.get("text", "")).strip()
    layer = str(args.get("layer", "semantic"))
    try:
        importance = float(args.get("importance", 0.7))
    except (TypeError, ValueError):
        importance = 0.7
    item = memory.add(layer, text, importance=importance, source="tool")
    return f"已记住（{item.layer} 层）：{item.text}"


async def _run_shell(command: str, *, timeout: float, cwd: Path) -> str:
    if not command.strip():
        return "错误：命令为空"
    logger.warning("run_shell executing: %s", command)
    cwd.mkdir(parents=True, exist_ok=True)

    def _run() -> str:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        output = output.strip() or "（无输出）"
        if len(output) > _MAX_FILE_CHARS:
            output = output[:_MAX_FILE_CHARS] + "\n...（截断）"
        return f"exit={result.returncode}\n{output}"

    return await asyncio.to_thread(_run)
