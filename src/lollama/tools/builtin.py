from __future__ import annotations

import ast
import asyncio
import fnmatch
import operator
import subprocess
import time
from datetime import datetime
from pathlib import Path

from lollama._logging import get_logger
from lollama.config import ToolsConfig
from lollama.memory import PERSISTED_LAYERS, MemoryManager

from .registry import Tool, ToolRegistry

logger = get_logger(__name__)

_MAX_FILE_CHARS = 20000
_MAX_SEARCH_FILE_BYTES = 1_000_000
_MAX_SEARCH_RESULTS = 200

# 各工具在状态播报里的动词短语（status.tool_labels 可覆盖）
DEFAULT_TOOL_LABELS = {
    "get_current_time": "查一下时间",
    "calculator": "算一下",
    "read_file": "读一下文件",
    "list_dir": "看一下文件目录",
    "file_info": "看一下文件信息",
    "find_files": "找一下文件",
    "search_files": "搜一下文件内容",
    "write_file": "写一下文件",
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
                    "properties": {
                        "path": {"type": "string", "description": "相对工作区的文件路径"},
                        "start_line": {"type": "integer", "description": "可选，起始行号（从 1 开始）"},
                        "max_lines": {"type": "integer", "description": "可选，最多读取多少行"},
                    },
                    "required": ["path"],
                },
                handler=lambda args: _read_file(
                    workspace,
                    str(args.get("path", "")),
                    start_line=args.get("start_line"),
                    max_lines=args.get("max_lines"),
                ),
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
    if b.file_info:
        registry.register(
            Tool(
                name="file_info",
                description="查看工作区内一个文件或目录的元信息（只读）",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "相对工作区的文件或目录路径"}},
                    "required": ["path"],
                },
                handler=lambda args: _file_info(workspace, str(args.get("path", ""))),
            )
        )
    if b.find_files:
        registry.register(
            Tool(
                name="find_files",
                description="在工作区内按文件名关键词或 glob 模式查找文件和目录（只读）",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "文件名关键词或 glob，如 *.py、**/*.md"},
                        "path": {"type": "string", "description": "可选，相对工作区的起始目录，默认根目录"},
                        "max_results": {"type": "integer", "description": "可选，最多返回多少条，默认 50，最多 200"},
                    },
                    "required": ["pattern"],
                },
                handler=lambda args: _find_files(
                    workspace,
                    str(args.get("path", "") or "."),
                    str(args.get("pattern", "")),
                    max_results=args.get("max_results"),
                ),
            )
        )
    if b.search_files:
        registry.register(
            Tool(
                name="search_files",
                description="在工作区内的文本文件中搜索关键词并返回匹配行（只读）",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要搜索的文本"},
                        "path": {"type": "string", "description": "可选，相对工作区的起始目录或文件，默认根目录"},
                        "glob": {"type": "string", "description": "可选，文件 glob 过滤，如 **/*.py，默认所有文件"},
                        "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认 false"},
                        "max_results": {"type": "integer", "description": "可选，最多返回多少条，默认 50，最多 200"},
                    },
                    "required": ["query"],
                },
                handler=lambda args: _search_files(
                    workspace,
                    str(args.get("path", "") or "."),
                    str(args.get("query", "")),
                    glob=str(args.get("glob", "") or "*"),
                    case_sensitive=bool(args.get("case_sensitive", False)),
                    max_results=args.get("max_results"),
                ),
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


def _read_file(workspace: Path, relative: str, *, start_line=None, max_lines=None) -> str:
    target = _resolve_in_workspace(workspace, relative)
    if target.is_dir():
        raise ValueError("path 指向目录，请使用 list_dir")
    text = target.read_text(encoding="utf-8", errors="replace")
    if start_line is not None or max_lines is not None:
        lines = text.splitlines()
        start = _coerce_int(start_line, default=1, minimum=1, maximum=max(len(lines), 1))
        count = _coerce_int(max_lines, default=min(200, max(len(lines) - start + 1, 0)), minimum=1, maximum=500)
        selected = lines[start - 1 : start - 1 + count]
        prefix = f"（第 {start}-{start + len(selected) - 1} 行 / 共 {len(lines)} 行）\n" if selected else f"（第 {start} 行之后无内容 / 共 {len(lines)} 行）"
        text = prefix + "\n".join(selected)
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
    if not target.is_dir():
        return _file_info(workspace, relative)
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "（空目录）"
    return "\n".join(f"{'[目录] ' if entry.is_dir() else ''}{entry.name}" for entry in entries[:200])


def _file_info(workspace: Path, relative: str) -> str:
    target = _resolve_in_workspace(workspace, relative)
    if not target.exists():
        return "（路径不存在）"
    stat = target.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    rel = _display_path(workspace, target)
    if target.is_dir():
        try:
            entry_count = sum(1 for _ in target.iterdir())
        except OSError:
            entry_count = -1
        count_text = "未知" if entry_count < 0 else str(entry_count)
        return "\n".join(
            [
                f"path: {rel}",
                "type: directory",
                f"entries: {count_text}",
                f"modified: {modified}",
            ]
        )
    line_count = ""
    if stat.st_size <= _MAX_SEARCH_FILE_BYTES:
        text = _read_text_file(target)
        if text is not None:
            line_count = f"\nlines: {len(text.splitlines())}"
    return "\n".join(
        [
            f"path: {rel}",
            "type: file",
            f"size_bytes: {stat.st_size}",
            f"modified: {modified}",
        ]
    ) + line_count


def _find_files(workspace: Path, relative: str, pattern: str, *, max_results=None) -> str:
    pattern = pattern.strip()
    if not pattern:
        return "错误：pattern 不能为空"
    root = _resolve_in_workspace(workspace, relative)
    if not root.exists():
        return "（目录不存在）"
    if not root.is_dir():
        root = root.parent
    limit = _coerce_int(max_results, default=50, minimum=1, maximum=_MAX_SEARCH_RESULTS)
    has_glob = any(ch in pattern for ch in "*?[]")
    pattern_lower = pattern.lower()
    matches: list[str] = []
    for entry in root.rglob("*"):
        rel = _display_path(workspace, entry)
        name = entry.name
        if has_glob:
            matched = fnmatch.fnmatch(rel.lower(), pattern_lower) or fnmatch.fnmatch(name.lower(), pattern_lower)
        else:
            matched = pattern_lower in rel.lower() or pattern_lower in name.lower()
        if matched:
            matches.append(f"{'[目录] ' if entry.is_dir() else ''}{rel}")
            if len(matches) >= limit:
                break
    if not matches:
        return "没有找到匹配文件"
    suffix = "\n...（结果已截断）" if len(matches) >= limit else ""
    return "\n".join(matches) + suffix


def _search_files(
    workspace: Path,
    relative: str,
    query: str,
    *,
    glob: str,
    case_sensitive: bool,
    max_results=None,
) -> str:
    query = query.strip()
    if not query:
        return "错误：query 不能为空"
    root = _resolve_in_workspace(workspace, relative)
    if not root.exists():
        return "（路径不存在）"
    limit = _coerce_int(max_results, default=50, minimum=1, maximum=_MAX_SEARCH_RESULTS)
    needle = query if case_sensitive else query.lower()
    matches: list[str] = []
    candidates = [root] if root.is_file() else root.rglob("*")
    glob_lower = (glob or "*").lower()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        rel = _display_path(workspace, candidate)
        if not fnmatch.fnmatch(rel.lower(), glob_lower) and not fnmatch.fnmatch(candidate.name.lower(), glob_lower):
            continue
        try:
            if candidate.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                continue
            text = _read_text_file(candidate)
        except OSError:
            continue
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                matches.append(f"{rel}:{line_number}: {snippet}")
                if len(matches) >= limit:
                    return "\n".join(matches) + "\n...（结果已截断）"
    if not matches:
        return "没有找到匹配内容"
    return "\n".join(matches)


def _display_path(workspace: Path, target: Path) -> str:
    try:
        return target.relative_to(workspace).as_posix() or "."
    except ValueError:
        return target.as_posix()


def _read_text_file(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _coerce_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(maximum, max(minimum, number))


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
