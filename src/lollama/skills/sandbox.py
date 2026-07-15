from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from lollama._logging import get_logger
from lollama.config import SkillsConfig

from .loader import Skill

logger = get_logger(__name__)

# ---------------------------------------------------------- Windows Job Object

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _create_job(max_memory_mb: int, max_processes: int):
        """创建带内存/进程数上限、关句柄即杀的 Job Object；失败返回 None（尽力而为）。"""
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = max_processes
        info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024
        ok = kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return job

    def _assign_job(job, pid: int) -> bool:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not handle:
            return False
        ok = kernel32.AssignProcessToJobObject(job, handle)
        kernel32.CloseHandle(handle)
        return bool(ok)

    def _terminate_job(job) -> None:
        ctypes.windll.kernel32.TerminateJobObject(job, 1)

    def _close_job(job) -> None:
        ctypes.windll.kernel32.CloseHandle(job)


def _sandbox_env(run_dir: Path) -> dict[str, str]:
    """环境变量白名单：不继承父进程环境，临时目录与 HOME 都指向本次运行目录。"""
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LOLLAMA_SKILL_RUN_DIR": str(run_dir),
        "TEMP": str(run_dir),
        "TMP": str(run_dir),
        "TMPDIR": str(run_dir),
        "HOME": str(run_dir),
        "USERPROFILE": str(run_dir),
    }
    if sys.platform == "win32":
        # Windows 上 socket/ssl 等基础设施依赖这几项
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    else:
        env["PATH"] = "/usr/bin:/bin"
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except OSError:
        pass


async def run_skill(skill: Skill, arguments: dict, *, cfg: SkillsConfig, runs_dir: Path) -> str:
    """在沙盒子进程中执行技能入口脚本。

    协议：参数以单个 JSON 对象写入 stdin，结果从 stdout 读出；退出码非 0 视为失败。
    沙盒措施：python -I 隔离模式、环境变量白名单、独立运行目录（cwd/TEMP/HOME）、
    超时强杀进程树、内存与进程数上限（Windows Job Object / POSIX rlimit）、输出截断。
    """
    if skill.entry is None:
        return f"错误：技能 {skill.name} 没有可执行入口"
    timeout = skill.timeout_sec or cfg.timeout_sec
    python = cfg.python or sys.executable
    run_dir = Path(runs_dir) / f"{skill.name}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(arguments, ensure_ascii=False)
    logger.info("running skill %s timeout=%.0fs run_dir=%s args=%s", skill.name, timeout, run_dir, payload[:200])

    def _run() -> tuple[int | None, str, str, bool]:
        cmd = [python, "-I", "-X", "utf8", str(skill.entry)]
        popen_kwargs: dict = {}
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _posix_preexec(cfg.max_memory_mb)
        process = subprocess.Popen(
            cmd,
            cwd=run_dir,
            env=_sandbox_env(run_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
        job = None
        if sys.platform == "win32":
            job = _create_job(cfg.max_memory_mb, cfg.max_processes)
            if job is not None and not _assign_job(job, process.pid):
                _close_job(job)
                job = None
            if job is None:
                logger.warning("skill %s: job object unavailable; memory/process limits not enforced", skill.name)
        try:
            stdout, stderr = process.communicate(input=payload, timeout=timeout)
            return process.returncode, stdout or "", stderr or "", False
        except subprocess.TimeoutExpired:
            if job is not None:
                _terminate_job(job)
            _kill_tree(process)
            stdout, stderr = process.communicate()
            return None, stdout or "", stderr or "", True
        finally:
            if job is not None:
                _close_job(job)

    try:
        returncode, stdout, stderr, timed_out = await asyncio.to_thread(_run)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    if timed_out:
        logger.warning("skill %s timed out after %.0fs", skill.name, timeout)
        return f"错误：技能 {skill.name} 执行超时（{timeout:.0f} 秒）"
    stdout = stdout.strip()
    stderr = stderr.strip()
    if returncode != 0:
        detail = stderr[-400:] if stderr else stdout[-400:]
        logger.warning("skill %s failed exit=%s: %s", skill.name, returncode, detail[:200])
        return f"错误：技能 {skill.name} 执行失败（退出码 {returncode}）：{detail}"
    output = stdout or "（无输出）"
    if len(output) > cfg.max_output_chars:
        output = output[: cfg.max_output_chars] + f"\n...（截断，共 {len(output)} 字符）"
    logger.info("skill %s done chars=%d", skill.name, len(output))
    return output


def _posix_preexec(max_memory_mb: int):
    def fn() -> None:
        os.setsid()
        try:
            import resource

            limit = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

    return fn
