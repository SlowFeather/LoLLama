from __future__ import annotations

import json
from pathlib import Path

from lollama.config import SkillsConfig
from lollama.skills import load_skills, register_skill_tools, run_skill
from lollama.tools import ToolRegistry


def write_skill(root: Path, name: str, *, frontmatter: str, script: str | None = None) -> Path:
    skill_dir = root / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(frontmatter, encoding="utf-8")
    if script is not None:
        (skill_dir / "scripts" / "main.py").write_text(script, encoding="utf-8")
    return skill_dir


ECHO_SCRIPT = """
import json, sys
args = json.loads(sys.stdin.read() or "{}")
print("echo:" + str(args.get("text", "")))
"""

SLEEP_SCRIPT = """
import time
time.sleep(60)
"""

FAIL_SCRIPT = """
import sys
print("坏了", file=sys.stderr)
sys.exit(3)
"""


def echo_frontmatter(name: str = "echo") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: 回显文本\n"
        "label: 回显一下\n"
        "entry: scripts/main.py\n"
        "parameters:\n"
        "  text: {type: string, description: 文本}\n"
        "required: [text]\n"
        "---\n"
        "使用说明正文。\n"
    )


def test_load_skills_parses_frontmatter(tmp_path: Path) -> None:
    write_skill(tmp_path, "echo", frontmatter=echo_frontmatter(), script=ECHO_SCRIPT)
    skills = load_skills(tmp_path)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "echo"
    assert skill.tool_name == "skill_echo"
    assert skill.label == "回显一下"
    assert skill.required == ["text"]
    assert "使用说明正文" in skill.instructions


def test_load_skills_skips_invalid(tmp_path: Path) -> None:
    write_skill(tmp_path, "ok", frontmatter=echo_frontmatter("ok"), script=ECHO_SCRIPT)
    # entry 越界的技能被拒绝
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: bad\ndescription: x\nentry: ../ok/scripts/main.py\n---\n", encoding="utf-8"
    )
    # 缺 description 的技能被拒绝
    nodesc = tmp_path / "nodesc"
    nodesc.mkdir()
    (nodesc / "SKILL.md").write_text("---\nname: nodesc\n---\n", encoding="utf-8")

    skills = load_skills(tmp_path)
    assert [skill.name for skill in skills] == ["ok"]


def test_register_skill_tools(tmp_path: Path) -> None:
    write_skill(tmp_path, "echo", frontmatter=echo_frontmatter(), script=ECHO_SCRIPT)
    registry = ToolRegistry()
    names = register_skill_tools(registry, load_skills(tmp_path), cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
    assert names == ["skill_echo"]
    assert registry.label("skill_echo") == "回显一下"
    spec = registry.specs()[0]["function"]
    assert spec["parameters"]["required"] == ["text"]


async def test_run_skill_roundtrip(tmp_path: Path) -> None:
    write_skill(tmp_path, "echo", frontmatter=echo_frontmatter(), script=ECHO_SCRIPT)
    registry = ToolRegistry()
    register_skill_tools(registry, load_skills(tmp_path), cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
    result = await registry.call("skill_echo", json.dumps({"text": "你好"}))
    assert result == "echo:你好"
    # 运行目录用完即删
    assert not any((tmp_path / "runs").iterdir())


async def test_run_skill_timeout_kills_process(tmp_path: Path) -> None:
    frontmatter = (
        "---\nname: sleepy\ndescription: 睡觉\nentry: scripts/main.py\ntimeout_sec: 2\n---\n"
    )
    write_skill(tmp_path, "sleepy", frontmatter=frontmatter, script=SLEEP_SCRIPT)
    (skill,) = load_skills(tmp_path)
    result = await run_skill(skill, {}, cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
    assert "超时" in result


async def test_run_skill_nonzero_exit_returns_error(tmp_path: Path) -> None:
    write_skill(tmp_path, "fail", frontmatter=(
        "---\nname: fail\ndescription: 出错\nentry: scripts/main.py\n---\n"
    ), script=FAIL_SCRIPT)
    (skill,) = load_skills(tmp_path)
    result = await run_skill(skill, {}, cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
    assert result.startswith("错误")
    assert "坏了" in result


async def test_run_skill_output_truncated(tmp_path: Path) -> None:
    script = "print('a' * 10000)"
    write_skill(tmp_path, "big", frontmatter=(
        "---\nname: big\ndescription: 大输出\nentry: scripts/main.py\n---\n"
    ), script=script)
    (skill,) = load_skills(tmp_path)
    cfg = SkillsConfig(max_output_chars=100)
    result = await run_skill(skill, {}, cfg=cfg, runs_dir=tmp_path / "runs")
    assert "截断" in result
    assert len(result) < 200


async def test_run_skill_env_is_whitelisted(tmp_path: Path) -> None:
    import os

    os.environ["LOLLAMA_TEST_SECRET"] = "s3cret"
    try:
        script = "import os; print(os.environ.get('LOLLAMA_TEST_SECRET', 'MISSING'))"
        write_skill(tmp_path, "env", frontmatter=(
            "---\nname: env\ndescription: 看环境\nentry: scripts/main.py\n---\n"
        ), script=script)
        (skill,) = load_skills(tmp_path)
        result = await run_skill(skill, {}, cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
        assert result == "MISSING"
    finally:
        del os.environ["LOLLAMA_TEST_SECRET"]


def test_instruction_only_skill_not_registered(tmp_path: Path) -> None:
    doc = tmp_path / "doc"
    doc.mkdir()
    (doc / "SKILL.md").write_text("---\nname: doc\ndescription: 只有说明\n---\n正文\n", encoding="utf-8")
    registry = ToolRegistry()
    names = register_skill_tools(registry, load_skills(tmp_path), cfg=SkillsConfig(), runs_dir=tmp_path / "runs")
    assert names == []
    assert len(registry) == 0
