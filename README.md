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

- **持久化**：SQLite（WAL、行级增量写入），文件 `artifacts/memory/{user_id}.db`；
  同目录的 `{user_id}.json` 是给 ChatCaht dashboard 的只读镜像，旧版 JSON 首次启动自动迁移进库。
- **写入**：每轮对话自动存情景记忆；后台用 LLM 提炼语义/程序性/画像记忆；模型也可通过
  `memory_save` 工具主动记忆。新记忆与旧记忆高度相似但否定相反（“喜欢/不喜欢”）时，
  旧记忆被自动削弱（矛盾衰减）。
- **检索**：三通道混合相关度 —— 字符 bigram 字面匹配（中文短查询兜底）+ SQLite FTS5
  BM25（trigram 分词）+ 可选本地 embedding 向量余弦，按权重融合后再叠加重要度与
  记忆强度，Top-K 注入 system 提示。向量通道在 `memory.retrieval.embedding:` 开启
  （LM Studio 加载一个 embedding 模型即可，向量在后台补算、查询失败自动退化为字面检索）。
- **巩固与晋升**：被检索命中的记忆强度增加（间隔重复）；情景记忆命中次数达阈值晋升为语义记忆。
- **遗忘**：强度按半衰期指数衰减，低于阈值即清除；层超容量时淘汰最弱记忆。
- 所有容量/半衰期/阈值/权重都在 `configs/config.yaml` 的 `memory:` 段可调。

## 工具调用

`tools:` 段可逐个开关，单次回复的工具轮数由 `tools.max_rounds` 限制。当前内置工具：

| 工具 | 作用 | 默认 |
|------|------|------|
| `get_current_time` | 获取当前本地日期和时间 | 开启 |
| `calculator` | 安全计算算术表达式 | 开启 |
| `read_file` | 读取 `workspace_dir` 内文本文件，可指定起始行和行数 | 开启 |
| `list_dir` | 列出 `workspace_dir` 内目录 | 开启 |
| `file_info` | 查看 `workspace_dir` 内文件/目录元信息 | 开启 |
| `find_files` | 在 `workspace_dir` 内按文件名关键词或 glob 查找文件 | 开启 |
| `search_files` | 在 `workspace_dir` 内全文搜索文本文件 | 开启 |
| `write_file` | 写入/覆盖 `workspace_dir` 内文本文件 | 开启 |
| `memory_search` | 模型主动检索长期记忆 | 开启 |
| `memory_save` | 模型主动写入长期记忆 | 开启 |
| `run_shell` | 在 `workspace_dir` 内执行本地 shell 命令 | 关闭 |

文件类工具都被限制在 `paths.workspace_dir` 目录内，路径越界会被拒绝。新补充的
`file_info`、`find_files`、`search_files` 都是只读文件工具；`run_shell` 风险最高，默认关闭。

## Agent Skill（沙盒执行）

参考 Anthropic Agent Skills 的目录规范：`skills/` 下每个子目录一个技能，
`SKILL.md` 的 YAML frontmatter 声明元数据，正文是给模型看的使用说明：

```
skills/
  word_count/
    SKILL.md          # name / description / label / entry / parameters / required / timeout_sec
    scripts/main.py   # 入口脚本：stdin 读 JSON 参数，stdout 输出结果，非 0 退出码视为失败
```

带 `entry` 的技能注册为工具 `skill_<name>`（如 `skill_word_count`），模型可像内置工具一样
调用；`label` 是状态播报用的动词短语（如“数一下字数”）。

**所有技能脚本都在沙盒子进程中执行**，沙盒措施：

| 措施 | 说明 |
|------|------|
| 进程隔离 | `python -I`（isolated 模式，不读用户 site-packages / 环境注入） |
| 环境变量白名单 | 不继承父进程环境，秘钥类变量对脚本不可见 |
| 独立运行目录 | cwd/TEMP/HOME 都指向一次性运行目录，用完即删 |
| 超时强杀 | 超过 `timeout_sec` 杀掉整个进程树 |
| 资源上限 | 内存与进程数上限（Windows Job Object / POSIX rlimit） |
| 输出截断 | stdout 超过 `skills.max_output_chars` 截断 |

注意：这套沙盒防事故不防恶意（不阻断网络与文件读取）。只放可信来源的技能进
`skills/` 目录；需要强隔离时可把 `skills.python` 指向容器/受限环境中的解释器。

## 快速开始

```bash
uv sync --extra dev
uv run lollama init-config          # 生成 configs/config.yaml
uv run lollama doctor               # 检查 LM Studio 与记忆存储
uv run lollama serve                # 启动全双工 WebSocket 服务 (ws://127.0.0.1:8801/v1/llm/ws)
uv run lollama text "你好，记住我喜欢简短的回答"
uv run lollama memory dump          # 查看记忆内容
uv run lollama skills               # 列出已加载的技能
```

## 日志与记忆文件

- 文件日志：`artifacts/logs/lollama.log`（滚动 10MB × 5 份，UTF-8），格式与 ChatCaht 全家统一：
  `2026-07-06 10:09:29,554 INFO lollama.service.server: 消息`
- 记忆持久化：`artifacts/memory/{user_id}.db`（SQLite，含 FTS5 全文索引和 embedding 向量表），
  按 4 层（episodic 情景 / semantic 事实 / procedural 偏好 / core 画像）存储，
  含强度、热度与访问时间；工作记忆只在内存中
- `artifacts/memory/{user_id}.json` 是同格式的只读镜像（带节流导出），
  ChatCaht Dashboard（`ChatCaht/dashboard/`）读它实时展示记忆条目与当前有效强度

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
