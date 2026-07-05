# LoLLama

全双工本地 LLM 智能体服务：ChatCaht（编排程序）通过 WebSocket 把用户文本发过来，
LoLLama 检索分层记忆、按需调用本地工具，向 LM Studio（OpenAI 兼容 API）请求生成，
并把回复**流式**发回；生成过程中随时可被打断（barge-in）。

与 GVoice（TTS）、SpText（ASR）、WakeUp（唤醒）同级，是 Chat 全双工实时对话系统的
"大脑"子项目。

## 5 层记忆系统

参考 [Awesome-AI-Memory](https://github.com/IAAR-Shanghai/Awesome-AI-Memory) 中
MemoryOS 的分层 + 热度晋升 + 遗忘架构，做了适合纯本地运行的原生实现（无向量库、无云依赖）：

| 层 | 名称 | 内容 | 默认半衰期 |
|---|------|------|-----------|
| 1 | 工作记忆 working | 当前对话窗口 | 会话内 |
| 2 | 情景记忆 episodic | 每轮对话事件 | 72 小时 |
| 3 | 语义记忆 semantic | 用户与世界的事实 | 30 天 |
| 4 | 程序性记忆 procedural | 偏好与做事方式 | 90 天 |
| 5 | 核心画像 core | 稳定的用户画像 | 永不遗忘 |

- **写入**：每轮对话自动存情景记忆；后台用 LLM 提炼语义/程序性/画像记忆；模型也可通过
  `memory_save` 工具主动记忆。
- **检索**：字面相关度 + 重要度 + 记忆强度加权打分，Top-K 注入 system 提示。
- **巩固与晋升**：被检索命中的记忆强度增加（间隔重复）；情景记忆命中次数达阈值晋升为语义记忆。
- **遗忘**：强度按半衰期指数衰减，低于阈值即清除；层超容量时淘汰最弱记忆。
- 所有容量/半衰期/阈值/权重都在 `configs/config.yaml` 的 `memory:` 段可调。

## 工具调用

`tools:` 段可逐个开关：当前时间、计算器、工作区内文件读/写/列目录、记忆检索/写入、
本地 shell（默认关闭）。单次回复的工具轮数由 `tools.max_rounds` 限制。

## 快速开始

```bash
uv sync --extra dev
uv run lollama init-config          # 生成 configs/config.yaml
uv run lollama doctor               # 检查 LM Studio 与记忆存储
uv run lollama serve                # 启动全双工 WebSocket 服务 (ws://127.0.0.1:8801/v1/llm/ws)
uv run lollama text "你好，记住我喜欢简短的回答"
uv run lollama memory dump          # 查看记忆内容
```

## WebSocket 协议

```jsonc
→ {"type":"chat","request_id":"r1","messages":[{"role":"user","content":"你好"}]}
   // 或 {"type":"chat","text":"你好"}（服务端自己维护工作记忆）
← {"type":"agent_status","request_id":"r1","stage":"accepted","announce":""}      // 状态钩子
← {"type":"agent_status","request_id":"r1","stage":"tool_start","announce":"我算一下。","name":"calculator",...}
← {"type":"delta","request_id":"r1","text":"你"}
← {"type":"tool","request_id":"r1","name":"get_current_time","status":"done","detail":"..."}
← {"type":"done","request_id":"r1","text":"完整回复","canceled":false}
→ {"type":"cancel"}      // 随时打断；新 chat 到达也会自动打断上一条
→ {"type":"ping"} / {"type":"status"} / {"type":"memory","action":"stats"} / {"type":"shutdown"}
```

## 状态钩子

请求生命周期各阶段发送 `agent_status` 事件，`announce` 为可直接朗读的播报文案
（ChatCaht 收到后送 TTS），文案与心跳节奏在 `status:` 段可调、留空即不播报：

| stage | 时机 | 默认播报 |
|-------|------|---------|
| accepted | 请求受理 | 无 |
| memory_recall | 召回了相关记忆（count） | 无 |
| llm_request | 开始请求模型（round） | 无 |
| llm_waiting | 首字超过 `llm_waiting_after_sec` 未到（可重复心跳） | "让我想想。" |
| llm_first_token | 首字产出 | 无 |
| tool_start | 开始调用工具（{label} 播报名） | "我算一下。"等 |
| tool_waiting | 工具执行超过 `tool_waiting_after_sec`（可重复心跳） | "还在处理，稍等。" |
| tool_done / tool_error | 工具完成 / 出错 | 无 / "工具出了点问题，我再想想。" |
| memory_extracted | 后台记忆提炼完成 | 无（可设"这个我记住了。"） |
| canceled / error | 被打断 / 出错 | 无 / "抱歉，出了点问题。" |

ChatCaht 侧把 `llm.provider` 设为 `lollama` 即可切换（见 ChatCaht 的 configs/config.yaml）。
