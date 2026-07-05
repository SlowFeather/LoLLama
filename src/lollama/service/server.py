from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

import websockets
from websockets.exceptions import ConnectionClosed

from lollama._logging import get_logger
from lollama.agent import Agent
from lollama.config import Config
from lollama.memory import MemoryManager
from lollama.tools import build_registry
from lollama.upstream import UpstreamClient

logger = get_logger(__name__)


class _Connection:
    """单个客户端连接的会话状态：服务端工作记忆 + 进行中的回复任务。"""

    __slots__ = ("history", "task", "request_id")

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.request_id: str | None = None


class RealtimeLlmService:
    """全双工 LLM 智能体服务。

    协议（JSON 文本帧）：
      → {"type":"ping"}                       ← {"type":"pong"}
      → {"type":"status"}                     ← {"type":"status", ...}
      → {"type":"chat","request_id"?, "messages":[...] 或 "text":"..."}
          ← {"type":"agent_status","request_id","stage","announce",...} ...  — 状态钩子
          ← {"type":"delta","request_id","text"} ...
          ← {"type":"tool","request_id","name","status","detail"} ...
          ← {"type":"done","request_id","text","canceled":false}
      → {"type":"cancel"}                     随时打断当前回复（barge-in）
      → {"type":"memory","action":"stats|clear|sweep"}
      → {"type":"shutdown"}
    新 chat 到达时自动取消上一条进行中的回复；连接断开也会中止回复。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._started_at = time.monotonic()
        self.upstream = UpstreamClient(cfg.upstream)
        self.memory = MemoryManager(cfg.memory, cfg.paths.memory_dir) if cfg.memory.enabled else None
        self.tools = build_registry(cfg.tools, workspace_dir=cfg.paths.workspace_dir, memory=self.memory)
        self.agent = Agent(cfg, upstream=self.upstream, memory=self.memory, tools=self.tools)
        self._clients: set = set()
        self._shutdown = asyncio.Event()
        self._requests_served = 0

    async def serve_forever(self) -> None:
        self.cfg.ensure_dirs()
        sweep_task = asyncio.create_task(self._sweep_loop())
        try:
            async with websockets.serve(
                self.handler,
                self.cfg.service.host,
                self.cfg.service.port,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ):
                logger.info(
                    "LoLLama WebSocket listening url=ws://%s:%d%s upstream=%s model=%s tools=%s memory=%s",
                    self.cfg.service.host,
                    self.cfg.service.port,
                    self.cfg.service.ws_path,
                    self.cfg.upstream.base_url,
                    self.cfg.upstream.model,
                    ",".join(self.tools.names()) or "disabled",
                    "on" if self.memory is not None else "off",
                )
                await self._shutdown.wait()
        finally:
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task
            await self.agent.close()
            await self.upstream.close()
            if self.memory is not None:
                self.memory.save()

    def shutdown(self) -> None:
        logger.info("shutdown requested")
        self._shutdown.set()

    async def handler(self, websocket) -> None:
        path = _ws_path(websocket)
        client = _client_name(websocket)
        if path != self.cfg.service.ws_path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return
        if len(self._clients) >= self.cfg.service.max_clients:
            await websocket.close(code=1013, reason="too many clients")
            return

        self._clients.add(websocket)
        conn = _Connection()
        logger.info("WebSocket connected client=%s", client)
        try:
            async for raw in websocket:
                if isinstance(raw, bytes):
                    continue
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("message must be a JSON object")
                    await self._handle_json(websocket, conn, payload)
                except ConnectionClosed:
                    raise
                except Exception as exc:
                    logger.warning("invalid WebSocket message from %s: %s", client, exc)
                    await _send_json(websocket, {"type": "error", "message": str(exc)})
        except ConnectionClosed:
            pass
        finally:
            await self._cancel_current(conn)
            self._clients.discard(websocket)
            logger.info("WebSocket disconnected client=%s", client)

    # ---------------------------------------------------------------- routing

    async def _handle_json(self, websocket, conn: _Connection, payload: dict) -> None:
        msg_type = str(payload.get("type") or payload.get("cmd") or "")
        if msg_type == "ping":
            await _send_json(websocket, {"type": "pong"})
        elif msg_type == "status":
            await _send_json(websocket, self.status())
        elif msg_type == "chat":
            await self._start_chat(websocket, conn, payload)
        elif msg_type == "cancel":
            canceled = await self._cancel_current(conn, websocket=websocket)
            await _send_json(websocket, {"type": "ack", "cmd": "cancel", "ok": True, "canceled": canceled})
        elif msg_type == "reset":
            await self._cancel_current(conn)
            conn.history.clear()
            await _send_json(websocket, {"type": "ack", "cmd": "reset", "ok": True})
        elif msg_type == "memory":
            await self._handle_memory(websocket, payload)
        elif msg_type == "shutdown":
            await _send_json(websocket, {"type": "ack", "cmd": "shutdown", "ok": True})
            self.shutdown()
        else:
            raise ValueError(f"unsupported message type: {msg_type}")

    async def _handle_memory(self, websocket, payload: dict) -> None:
        action = str(payload.get("action") or "stats")
        if self.memory is None:
            await _send_json(websocket, {"type": "memory", "action": action, "enabled": False})
            return
        if action == "clear":
            self.memory.clear()
        elif action == "sweep":
            removed = self.memory.sweep()
            await _send_json(websocket, {"type": "memory", "action": action, "removed": removed, "stats": self.memory.stats()})
            return
        elif action != "stats":
            raise ValueError(f"unsupported memory action: {action}")
        await _send_json(websocket, {"type": "memory", "action": action, "enabled": True, "stats": self.memory.stats()})

    def status(self) -> dict:
        return {
            "type": "status",
            "ready": True,
            "model": self.cfg.upstream.model,
            "upstream": self.cfg.upstream.base_url,
            "tools": self.tools.names(),
            "memory": self.memory.stats() if self.memory is not None else None,
            "clients": len(self._clients),
            "requests_served": self._requests_served,
            "uptime_seconds": round(time.monotonic() - self._started_at, 3),
        }

    # ------------------------------------------------------------------- chat

    async def _start_chat(self, websocket, conn: _Connection, payload: dict) -> None:
        messages = payload.get("messages")
        text = str(payload.get("text") or "").strip()
        if messages is not None:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                raise ValueError("chat.messages must be a list of objects")
        elif not text:
            raise ValueError("chat requires 'messages' or non-empty 'text'")
        request_id = str(payload.get("request_id") or uuid.uuid4().hex[:8])

        # 全双工：新请求自动打断进行中的回复
        await self._cancel_current(conn, websocket=websocket)
        conn.request_id = request_id
        conn.task = asyncio.create_task(self._run_chat(websocket, conn, request_id, messages, text))

    async def _run_chat(self, websocket, conn: _Connection, request_id: str, messages: list[dict] | None, text: str) -> None:
        if messages is None:
            conn.history.append({"role": "user", "content": text})
            _trim_history(conn.history, self.cfg.agent.max_history_turns)
            messages = list(conn.history)
        full_text = ""
        canceled = False

        async def notify(event: dict) -> None:
            # 后台事件（记忆提炼完成）：连接可能已断开，尽力而为
            await self._send_status(websocket, request_id, event)

        try:
            await self._send_status(websocket, request_id, self.agent.make_status("accepted"))
            async for event in self.agent.respond(messages, background_notify=notify):
                if event["type"] == "delta":
                    await _send_json(websocket, {"type": "delta", "request_id": request_id, "text": event["text"]})
                elif event["type"] == "status":
                    await self._send_status(websocket, request_id, event)
                elif event["type"] == "tool":
                    await _send_json(
                        websocket,
                        {
                            "type": "tool",
                            "request_id": request_id,
                            "name": event["name"],
                            "status": event["status"],
                            "detail": event.get("detail", ""),
                        },
                    )
                elif event["type"] == "done":
                    full_text = event["text"]
        except asyncio.CancelledError:
            canceled = True
            raise
        except ConnectionClosed:
            canceled = True
        except Exception as exc:
            logger.exception("chat request %s failed", request_id)
            with contextlib.suppress(Exception):
                await self._send_status(websocket, request_id, self.agent.make_status("error", message=str(exc)))
                await _send_json(websocket, {"type": "error", "request_id": request_id, "message": str(exc)})
            return
        finally:
            if conn.request_id == request_id:
                conn.request_id = None
            if not canceled and full_text and messages is not None and text:
                conn.history.append({"role": "assistant", "content": full_text})
                _trim_history(conn.history, self.cfg.agent.max_history_turns)
        self._requests_served += 1
        with contextlib.suppress(Exception):
            await _send_json(websocket, {"type": "done", "request_id": request_id, "text": full_text, "canceled": False})

    async def _cancel_current(self, conn: _Connection, *, websocket=None) -> bool:
        task = conn.task
        conn.task = None
        request_id = conn.request_id
        conn.request_id = None
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        logger.info("chat request %s canceled", request_id)
        if websocket is not None and request_id is not None:
            await self._send_status(websocket, request_id, self.agent.make_status("canceled"))
        return True

    async def _send_status(self, websocket, request_id: str, event: dict | None) -> None:
        if event is None:
            return
        message = {key: value for key, value in event.items() if key != "type"}
        message["type"] = "agent_status"
        message["request_id"] = request_id
        with contextlib.suppress(Exception):
            await _send_json(websocket, message)

    # ----------------------------------------------------------------- sweeps

    async def _sweep_loop(self) -> None:
        if self.memory is None or not self.cfg.memory.forgetting.enabled:
            return
        interval = self.cfg.memory.forgetting.sweep_interval_sec
        while True:
            await asyncio.sleep(interval)
            try:
                self.memory.sweep()
            except Exception:
                logger.exception("memory sweep failed")


def _trim_history(history: list[dict], max_turns: int) -> None:
    keep = max_turns * 2
    if len(history) > keep:
        del history[: len(history) - keep]


async def _send_json(websocket, message: dict) -> None:
    await websocket.send(json.dumps(message, ensure_ascii=False))


def _ws_path(websocket) -> str | None:
    path = getattr(websocket, "path", None)
    if path is not None:
        return path
    request = getattr(websocket, "request", None)
    return getattr(request, "path", None)


def _client_name(websocket) -> str:
    remote = websocket.remote_address
    if isinstance(remote, tuple) and len(remote) >= 2:
        return f"{remote[0]}:{remote[1]}"
    return str(remote)


def run_service(cfg: Config) -> None:
    asyncio.run(RealtimeLlmService(cfg).serve_forever())
