from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import websockets

from lollama.config import Config
from lollama.service.server import RealtimeLlmService


class SlowFakeUpstream:
    def __init__(self, reply: str = "你好，我在。", delay: float = 0.0):
        self.reply = reply
        self.delay = delay
        self.calls: list[list[dict]] = []

    async def stream_chat(self, messages, *, tools=None):
        self.calls.append(list(messages))
        for ch in self.reply:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield {"type": "delta", "text": ch}
        yield {"type": "done", "finish_reason": "stop"}

    async def close(self) -> None:
        return None

    async def health(self) -> tuple[bool, str]:
        return True, "fake model available"


@contextlib.asynccontextmanager
async def running_service(tmp_path: Path, *, delay: float = 0.0, reply: str = "你好，我在。"):
    cfg = Config()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.paths.memory_dir = str(tmp_path / "memory")
    cfg.paths.workspace_dir = str(tmp_path / "workspace")
    cfg.memory.extraction.enabled = False
    service = RealtimeLlmService(cfg)
    await service.upstream.close()
    fake = SlowFakeUpstream(reply=reply, delay=delay)
    service.upstream = fake
    service.agent.upstream = fake

    server = await websockets.serve(service.handler, "127.0.0.1", 0, max_size=None)
    port = server.sockets[0].getsockname()[1]
    try:
        yield service, f"ws://127.0.0.1:{port}{cfg.service.ws_path}"
    finally:
        server.close()
        await server.wait_closed()


async def _recv_json(ws, timeout: float = 5.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def test_ping_status_and_memory_commands(tmp_path: Path) -> None:
    async with running_service(tmp_path) as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            assert (await _recv_json(ws))["type"] == "pong"

            await ws.send(json.dumps({"type": "status"}))
            status = await _recv_json(ws)
            assert status["type"] == "status"
            assert status["ready"] is True
            assert {"ready", "state", "model_loaded", "audio_open", "last_error"} <= status.keys()
            assert "memory_search" in status["tools"]

            await ws.send(json.dumps({"type": "memory", "action": "stats"}))
            memory = await _recv_json(ws)
            assert memory["type"] == "memory"
            assert memory["stats"] == {"episodic": 0, "semantic": 0, "procedural": 0, "core": 0}


async def test_chat_with_messages_streams_deltas_and_done(tmp_path: Path) -> None:
    async with running_service(tmp_path, reply="早上好。") as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "chat",
                        "request_id": "r1",
                        "messages": [{"role": "user", "content": "早上好"}],
                    }
                )
            )
            deltas: list[str] = []
            while True:
                msg = await _recv_json(ws)
                if msg["type"] == "delta":
                    assert msg["request_id"] == "r1"
                    deltas.append(msg["text"])
                elif msg["type"] == "done":
                    assert msg["request_id"] == "r1"
                    assert msg["text"] == "早上好。"
                    assert msg["canceled"] is False
                    break
            assert "".join(deltas) == "早上好。"


async def test_chat_with_bare_text_keeps_server_side_history(tmp_path: Path) -> None:
    async with running_service(tmp_path) as (service, url):
        async with websockets.connect(url) as ws:
            for request_id in ("a", "b"):
                await ws.send(json.dumps({"type": "chat", "request_id": request_id, "text": "你好"}))
                while (await _recv_json(ws))["type"] != "done":
                    pass
        # 服务端工作记忆里累计了两轮 user+assistant
        assert service._requests_served == 2


async def test_chat_with_text_uses_server_prompt_and_working_history(tmp_path: Path) -> None:
    async with running_service(tmp_path, reply="好。") as (service, url):
        async with websockets.connect(url) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "chat",
                        "request_id": "a",
                        "text": "第一句",
                    },
                    ensure_ascii=False,
                )
            )
            while (await _recv_json(ws))["type"] != "done":
                pass

            await ws.send(
                json.dumps(
                    {
                        "type": "chat",
                        "request_id": "b",
                        "text": "第二句",
                    },
                    ensure_ascii=False,
                )
            )
            while (await _recv_json(ws))["type"] != "done":
                pass

    second_prompt = service.upstream.calls[1]
    system_texts = [message["content"] for message in second_prompt if message["role"] == "system"]
    user_texts = [message["content"] for message in second_prompt if message["role"] == "user"]
    assistant_texts = [message["content"] for message in second_prompt if message["role"] == "assistant"]
    assert any("工具纪律" in text for text in system_texts)
    assert all("简短回答" not in text for text in system_texts)
    assert user_texts == ["第一句", "第二句"]
    assert assistant_texts == ["好。"]


async def test_new_chat_cancels_previous_one(tmp_path: Path) -> None:
    async with running_service(tmp_path, delay=0.05, reply="这是一条很长很长的回答。") as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat", "request_id": "old", "text": "第一个问题"}))
            while True:
                first = await _recv_json(ws)
                if first["type"] == "delta":
                    assert first["request_id"] == "old"
                    break

            await ws.send(json.dumps({"type": "chat", "request_id": "new", "text": "换个问题"}))
            saw_new_done = False
            async with asyncio.timeout(10):
                while not saw_new_done:
                    msg = await _recv_json(ws)
                    if msg["type"] == "done" and msg["request_id"] == "new":
                        saw_new_done = True
                    else:
                        # 旧请求被取消后不应再有 old 的 done
                        assert not (msg["type"] == "done" and msg["request_id"] == "old")
            assert saw_new_done


async def test_cancel_command_stops_stream(tmp_path: Path) -> None:
    async with running_service(tmp_path, delay=0.05) as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat", "request_id": "r1", "text": "你好"}))
            while (await _recv_json(ws))["type"] != "delta":
                pass
            await ws.send(json.dumps({"type": "cancel"}))
            async with asyncio.timeout(5):
                while True:
                    msg = await _recv_json(ws)
                    if msg["type"] == "ack" and msg["cmd"] == "cancel":
                        assert msg["canceled"] is True
                        break


async def test_chat_emits_agent_status_events(tmp_path: Path) -> None:
    async with running_service(tmp_path, reply="好的。") as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat", "request_id": "r1", "text": "你好"}))
            seen: list[dict] = []
            while True:
                msg = await _recv_json(ws)
                seen.append(msg)
                if msg["type"] == "done":
                    break
            statuses = [m for m in seen if m["type"] == "agent_status"]
            assert statuses, "expected agent_status events"
            assert statuses[0]["stage"] == "accepted"
            assert all(m["request_id"] == "r1" for m in statuses)
            stages = {m["stage"] for m in statuses}
            assert "llm_request" in stages
            assert "llm_first_token" in stages


async def test_barge_in_sends_canceled_status(tmp_path: Path) -> None:
    async with running_service(tmp_path, delay=0.05, reply="这是一条很长很长的回答。") as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat", "request_id": "old", "text": "第一个问题"}))
            while (await _recv_json(ws))["type"] != "delta":
                pass
            await ws.send(json.dumps({"type": "chat", "request_id": "new", "text": "换个问题"}))
            saw_canceled = False
            async with asyncio.timeout(10):
                while True:
                    msg = await _recv_json(ws)
                    if msg["type"] == "agent_status" and msg["stage"] == "canceled" and msg["request_id"] == "old":
                        saw_canceled = True
                    if msg["type"] == "done" and msg["request_id"] == "new":
                        break
            assert saw_canceled


async def test_invalid_message_returns_error(tmp_path: Path) -> None:
    async with running_service(tmp_path) as (_service, url):
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat"}))
            msg = await _recv_json(ws)
            assert msg["type"] == "error"

            await ws.send("not json")
            msg = await _recv_json(ws)
            assert msg["type"] == "error"
