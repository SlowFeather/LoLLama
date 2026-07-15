from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lollama._logging import get_logger
from lollama.config import SkillsConfig
from lollama.tools.registry import Tool, ToolRegistry

logger = get_logger(__name__)

# 技能名：小写字母开头，小写字母/数字/下划线，最长 32
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# 拼进工具 description 的 SKILL.md 正文上限（避免撑爆工具定义）
_MAX_INSTRUCTION_CHARS = 500


@dataclass(slots=True)
class Skill:
    """一个 Agent Skill：目录 + SKILL.md（YAML frontmatter + 说明正文）+ 可选入口脚本。

    参考 Anthropic Agent Skills 的目录规范：frontmatter 声明 name/description，
    正文是给模型看的使用说明；带 entry 的技能注册为工具并在沙盒中执行。
    """

    name: str
    description: str
    dir: Path
    label: str | None = None
    entry: Path | None = None
    instructions: str = ""
    parameters: dict = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    timeout_sec: float | None = None

    @property
    def tool_name(self) -> str:
        return f"skill_{self.name}"


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """扫描技能目录：每个含 SKILL.md 的子目录是一个技能；坏技能跳过不拖累其他。"""
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    seen: set[str] = set()
    for child in sorted(root.iterdir()):
        skill_md = child / "SKILL.md"
        if not child.is_dir() or not skill_md.is_file():
            continue
        try:
            skill = _parse_skill(child, skill_md)
        except Exception as exc:
            logger.warning("skipping invalid skill %s: %s", child.name, exc)
            continue
        if skill.name in seen:
            logger.warning("skipping duplicate skill name %s in %s", skill.name, child)
            continue
        seen.add(skill.name)
        skills.append(skill)
    if skills:
        logger.info("loaded %d skill(s): %s", len(skills), ", ".join(s.name for s in skills))
    return skills


def _parse_skill(skill_dir: Path, skill_md: Path) -> Skill:
    text = skill_md.read_text(encoding="utf-8-sig")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("SKILL.md must start with YAML frontmatter (--- ... ---)")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")

    name = str(meta.get("name") or skill_dir.name).strip()
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid skill name {name!r} (expect ^[a-z][a-z0-9_]{{0,31}}$)")
    description = str(meta.get("description") or "").strip()
    if not description:
        raise ValueError("SKILL.md frontmatter requires a non-empty description")

    entry: Path | None = None
    entry_raw = str(meta.get("entry") or "").strip()
    if entry_raw:
        entry = (skill_dir / entry_raw).resolve()
        skill_root = skill_dir.resolve()
        if skill_root != entry and skill_root not in entry.parents:
            raise ValueError(f"entry escapes skill directory: {entry_raw}")
        if not entry.is_file():
            raise ValueError(f"entry script not found: {entry_raw}")
        if entry.suffix != ".py":
            raise ValueError(f"entry must be a .py script, got: {entry_raw}")

    parameters = meta.get("parameters") or {}
    if not isinstance(parameters, dict) or not all(isinstance(v, dict) for v in parameters.values()):
        raise ValueError("parameters must be a mapping of name -> JSON Schema fragment")
    required = meta.get("required") or []
    if not isinstance(required, list):
        raise ValueError("required must be a list")
    required = [str(key) for key in required]
    unknown = [key for key in required if key not in parameters]
    if unknown:
        raise ValueError(f"required parameters not declared: {unknown}")

    timeout_sec: float | None = None
    if meta.get("timeout_sec") is not None:
        timeout_sec = float(meta["timeout_sec"])
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")

    label = str(meta.get("label") or "").strip() or None
    instructions = text[match.end() :].strip()
    return Skill(
        name=name,
        description=description,
        dir=skill_dir.resolve(),
        label=label,
        entry=entry,
        instructions=instructions,
        parameters=parameters,
        required=required,
        timeout_sec=timeout_sec,
    )


def register_skill_tools(
    registry: ToolRegistry,
    skills: list[Skill],
    *,
    cfg: SkillsConfig,
    runs_dir: str | Path,
) -> list[str]:
    """把带入口脚本的技能注册为工具（skill_<name>）；返回注册的工具名列表。"""
    from .sandbox import run_skill  # 延迟导入，避免环

    runs_dir = Path(runs_dir)
    registered: list[str] = []
    for skill in skills:
        if skill.entry is None:
            logger.info("skill %s has no entry script; not exposed as a tool", skill.name)
            continue
        description = skill.description
        if skill.instructions:
            description += "\n" + skill.instructions[:_MAX_INSTRUCTION_CHARS]

        def make_handler(bound: Skill):
            async def handler(args: dict) -> str:
                return await run_skill(bound, args, cfg=cfg, runs_dir=runs_dir)

            return handler

        registry.register(
            Tool(
                name=skill.tool_name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": skill.parameters,
                    **({"required": skill.required} if skill.required else {}),
                },
                handler=make_handler(skill),
                label=skill.label,
            )
        )
        registered.append(skill.tool_name)
    return registered
