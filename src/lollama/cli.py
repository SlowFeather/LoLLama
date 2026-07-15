from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from ._logging import get_logger, setup_logging
from .agent import Agent
from .config import Config, load_config
from .memory import MemoryManager, build_memory
from .service.server import RealtimeLlmService
from .skills import load_skills, register_skill_tools
from .tools import build_registry
from .upstream import UpstreamClient

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130


def default_config_path() -> str:
    for candidate in ("configs/config.yaml", "configs/config.example.yaml"):
        if Path(candidate).exists():
            return candidate
    return "configs/config.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lollama",
        description="Full-duplex local LLM agent service with layered memory and tool calling",
    )
    parser.add_argument("--config", default=default_config_path(), help="Path to LoLLama YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Run the full-duplex WebSocket agent service")
    sub.add_parser("doctor", help="Check upstream model API and memory storage")

    text = sub.add_parser("text", help="Send one prompt through the agent (memory + tools) and print the reply")
    text.add_argument("prompt", nargs="+")

    memory = sub.add_parser("memory", help="Inspect or manage the layered memory store")
    memory.add_argument("action", choices=["stats", "sweep", "clear", "dump"])

    sub.add_parser("skills", help="List loaded agent skills")

    init = sub.add_parser("init-config", help="Copy the example config to a writable path")
    init.add_argument("--out", default="configs/config.yaml")
    init.add_argument("--force", action="store_true")
    return parser


async def _amain(args: argparse.Namespace) -> int:
    if args.command == "init-config":
        return _init_config(args)

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    setup_logging(cfg.runtime.log_level, str(Path(cfg.paths.logs_dir) / "lollama.log"))
    logger.info("lollama %s starting; config=%s", args.command, args.config)

    if args.command == "serve":
        await RealtimeLlmService(cfg).serve_forever()
        return 0
    if args.command == "doctor":
        return await _doctor(cfg)
    if args.command == "text":
        return await _text(cfg, " ".join(args.prompt))
    if args.command == "memory":
        return _memory(cfg, args.action)
    if args.command == "skills":
        return _skills(cfg)
    raise AssertionError(args.command)


def _init_config(args: argparse.Namespace) -> int:
    src = Path(__file__).resolve().parents[2] / "configs" / "config.example.yaml"
    dst = Path(args.out)
    if dst.exists() and not args.force:
        print(f"配置已存在: {dst}")
        print("需要覆盖时加 --force。")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"已创建配置: {dst}")
    return 0


async def _doctor(cfg: Config) -> int:
    upstream = UpstreamClient(cfg.upstream)
    try:
        ok, detail = await upstream.health()
    finally:
        await upstream.close()
    print(f"[{'OK' if ok else 'FAIL'}] upstream: {detail}")

    memory_ok = True
    if cfg.memory.enabled:
        try:
            manager = MemoryManager(cfg.memory, cfg.paths.memory_dir)
            manager.save()
            print(f"[OK] memory: dir={cfg.paths.memory_dir} stats={manager.stats()}")
        except Exception as exc:
            memory_ok = False
            print(f"[FAIL] memory: {exc}")
    else:
        print("[OK] memory: disabled")

    registry = build_registry(cfg.tools, workspace_dir=cfg.paths.workspace_dir, memory=None)
    if cfg.skills.enabled:
        register_skill_tools(registry, load_skills(cfg.skills.dir), cfg=cfg.skills, runs_dir=cfg.paths.skill_runs_dir)
    print(f"[OK] tools: {', '.join(registry.names()) or 'disabled'}")
    return 0 if ok and memory_ok else 2


async def _text(cfg: Config, prompt: str) -> int:
    upstream = UpstreamClient(cfg.upstream)
    memory = (
        build_memory(cfg.memory, cfg.paths.memory_dir, upstream_base_url=cfg.upstream.base_url)
        if cfg.memory.enabled
        else None
    )
    tools = build_registry(cfg.tools, workspace_dir=cfg.paths.workspace_dir, memory=memory)
    if cfg.skills.enabled:
        register_skill_tools(tools, load_skills(cfg.skills.dir), cfg=cfg.skills, runs_dir=cfg.paths.skill_runs_dir)
    agent = Agent(cfg, upstream=upstream, memory=memory, tools=tools)
    try:
        async for event in agent.respond([{"role": "user", "content": prompt}]):
            if event["type"] == "delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "status":
                announce = f" 「{event['announce']}」" if event.get("announce") else ""
                print(f"\n[status {event['stage']}]{announce}", flush=True)
            elif event["type"] == "tool":
                print(f"\n[tool {event['name']} {event['status']}] {event.get('detail', '')}", flush=True)
        print()
        # 等待后台记忆提炼完成再退出
        await asyncio.gather(*agent._background, return_exceptions=True)
    finally:
        await agent.close()
        await upstream.close()
        if memory is not None:
            await memory.aclose()
    return 0


def _memory(cfg: Config, action: str) -> int:
    if not cfg.memory.enabled:
        print("memory disabled in config")
        return 1
    manager = MemoryManager(cfg.memory, cfg.paths.memory_dir)
    if action == "stats":
        print(manager.stats())
    elif action == "sweep":
        removed = manager.sweep()
        print(f"removed={removed} stats={manager.stats()}")
    elif action == "clear":
        manager.clear()
        print("memory cleared")
    elif action == "dump":
        for layer in ("core", "procedural", "semantic", "episodic"):
            items = manager.items(layer)
            print(f"== {layer} ({len(items)}) ==")
            for item in items:
                print(f"  [{item.importance:.2f}/{item.strength:.2f} hits={item.hits}] {item.text}")
    return 0


def _skills(cfg: Config) -> int:
    if not cfg.skills.enabled:
        print("skills disabled in config")
        return 1
    skills = load_skills(cfg.skills.dir)
    if not skills:
        print(f"（{cfg.skills.dir} 下没有技能；每个技能一个子目录，内含 SKILL.md）")
        return 0
    for skill in skills:
        kind = "工具" if skill.entry is not None else "仅说明"
        params = ", ".join(skill.parameters) or "-"
        print(f"{skill.name} [{kind}] {skill.description}")
        print(f"  参数: {params}  超时: {skill.timeout_sec or cfg.skills.timeout_sec}s  目录: {skill.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
