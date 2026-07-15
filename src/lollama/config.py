from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    artifacts_dir: str = "artifacts"
    logs_dir: str = "artifacts/logs"
    memory_dir: str = "artifacts/memory"
    workspace_dir: str = "artifacts/workspace"
    # 技能沙盒每次执行的临时运行目录
    skill_runs_dir: str = "artifacts/skill_runs"


@dataclass(slots=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8801
    ws_path: str = "/v1/llm/ws"
    max_clients: int = 8
    health_timeout_sec: float = 5.0


@dataclass(slots=True)
class UpstreamConfig:
    """OpenAI 兼容上游（LM Studio / Ollama）。"""

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen/qwen3.5-9b"
    api_key: str = "lm-studio"
    temperature: float = 0.6
    max_tokens: int = 512
    timeout_sec: float = 120.0
    # 附加请求字段，原样并入 /chat/completions 请求体。
    # 思考型模型（如 qwen3.5）必须 {"reasoning_effort": "none"}，否则 content 为空。
    extra_body: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "none"})


@dataclass(slots=True)
class AgentConfig:
    system_prompt: str = (
        "你是运行在本地电脑上的中文助手，回答自然简洁，通常 1 到 2 句话，适合直接朗读；"
        "不要 Markdown、表情、长列表或客套铺垫，除非用户明确要求展开。"
        "工具纪律：需要当前/实时信息、计算、记忆检索或电脑文件内容时先调用工具，不要猜测或编造；"
        "用户询问当前时间、日期或星期时，必须调用 get_current_time 工具，工具结果优先于记忆和常识；"
        "工具结果足够时直接回答，不重复无关细节；结果不足时简短说明限制。"
        "只有用户明确要求写入时才调用 write_file 或 memory_save；其他文件操作优先使用只读工具。"
    )
    # 客户端只发纯文本时，服务端自己维护的工作记忆轮数
    max_history_turns: int = 8
    inject_memory: bool = True
    inject_time: bool = False


@dataclass(slots=True)
class LayerConfig:
    capacity: int = 300
    # 记忆强度半衰期（小时）；0 表示永不衰减
    half_life_hours: float = 72.0
    # 有效强度低于该值时被遗忘清除
    min_strength: float = 0.15


@dataclass(slots=True)
class LayersConfig:
    # 第 1 层：工作记忆（当前对话窗口，只在内存中）
    working_max_turns: int = 8
    # 第 2 层：情景记忆（对话事件，按时间衰减最快）
    episodic: LayerConfig = field(default_factory=lambda: LayerConfig(capacity=300, half_life_hours=72.0, min_strength=0.15))
    # 第 3 层：语义记忆（用户相关事实）
    semantic: LayerConfig = field(default_factory=lambda: LayerConfig(capacity=500, half_life_hours=720.0, min_strength=0.10))
    # 第 4 层：程序性记忆（偏好、做事方式）
    procedural: LayerConfig = field(default_factory=lambda: LayerConfig(capacity=200, half_life_hours=2160.0, min_strength=0.05))
    # 第 5 层：核心画像（稳定的用户画像，默认永不遗忘）
    core: LayerConfig = field(default_factory=lambda: LayerConfig(capacity=64, half_life_hours=0.0, min_strength=0.0))


@dataclass(slots=True)
class EmbeddingConfig:
    """可选的本地向量检索通道（OpenAI 兼容 /embeddings，如 LM Studio 加载的 embedding 模型）。"""

    enabled: bool = False
    # 留空则复用 upstream.base_url
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_sec: float = 5.0


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 6
    min_score: float = 0.08
    similarity_weight: float = 0.5
    importance_weight: float = 0.25
    strength_weight: float = 0.25
    # 相关度融合的三个通道权重（按本次查询实际可用的通道归一化）：
    # bigram 字符二元组字面匹配 / SQLite FTS5 BM25 / embedding 向量余弦
    bigram_weight: float = 1.0
    fts_weight: float = 0.6
    vector_weight: float = 1.0
    # FTS 通道最多取多少条候选
    fts_candidates: int = 32
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass(slots=True)
class PromotionConfig:
    enabled: bool = True
    # 情景记忆被命中该次数后晋升为语义记忆
    episodic_hits_to_semantic: int = 3
    # 每次被检索命中时的强度增益（类似间隔重复的巩固）
    reinforce_on_recall: float = 0.35


@dataclass(slots=True)
class ForgettingConfig:
    enabled: bool = True
    # 周期性遗忘清理的间隔（秒）
    sweep_interval_sec: float = 600.0


@dataclass(slots=True)
class ExtractionConfig:
    # 会话结束一轮后，用 LLM 从对话中提炼语义/程序性/画像记忆
    enabled: bool = True
    min_turn_chars: int = 6
    max_items_per_turn: int = 4
    timeout_sec: float = 30.0


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    user_id: str = "default"
    # dashboard 只读 JSON 镜像的最小导出间隔（秒）；SQLite 才是持久化事实源
    snapshot_min_interval_sec: float = 5.0
    layers: LayersConfig = field(default_factory=LayersConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)


@dataclass(slots=True)
class StatusAnnounceConfig:
    """各阶段的播报文案；留空表示该阶段不播报。

    tool_start 支持 {label}（工具播报名）和 {name}（工具原名）占位符。
    """

    accepted: str = ""
    memory_recall: str = ""
    llm_request: str = ""
    llm_waiting: str = "让我想想。"
    llm_first_token: str = ""
    tool_start: str = "我{label}。"
    tool_waiting: str = "还在处理，稍等。"
    tool_done: str = ""
    tool_error: str = "工具出了点问题，我再想想。"
    memory_extracted: str = ""
    canceled: str = ""
    error: str = "抱歉，出了点问题。"


@dataclass(slots=True)
class StatusConfig:
    """状态钩子：请求生命周期各阶段向客户端发送 agent_status 事件，供播报。"""

    enabled: bool = True
    # 等待模型首字超过该秒数后播报 llm_waiting；0 表示不播报
    llm_waiting_after_sec: float = 3.0
    # 之后每隔该秒数重复播报；0 表示只播报一次
    llm_waiting_repeat_sec: float = 15.0
    # 单个工具执行超过该秒数后播报 tool_waiting；0 表示不播报
    tool_waiting_after_sec: float = 5.0
    tool_waiting_repeat_sec: float = 15.0
    # 覆盖内置工具的播报名（动词短语，如 calculator: 算一下）
    tool_labels: dict[str, str] = field(default_factory=dict)
    announce: StatusAnnounceConfig = field(default_factory=StatusAnnounceConfig)


@dataclass(slots=True)
class ShellToolConfig:
    # 谨慎开启：允许模型执行本地命令
    enabled: bool = False
    timeout_sec: float = 20.0


@dataclass(slots=True)
class BuiltinToolsConfig:
    time: bool = True
    calculator: bool = True
    read_file: bool = True
    list_dir: bool = True
    file_info: bool = True
    find_files: bool = True
    search_files: bool = True
    memory_search: bool = True
    write_file: bool = True
    memory_save: bool = True


@dataclass(slots=True)
class ToolsConfig:
    enabled: bool = True
    # 单次回复中最多的工具调用轮数
    max_rounds: int = 4
    builtin: BuiltinToolsConfig = field(default_factory=BuiltinToolsConfig)
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)


@dataclass(slots=True)
class SkillsConfig:
    """Agent Skill：skills 目录下每个子目录一个技能（SKILL.md + 脚本），脚本在沙盒子进程中执行。

    沙盒边界 = 独立子进程（python -I）+ 环境变量白名单 + 独立运行目录 + 超时强杀
    + 内存/进程数上限（Windows Job Object / POSIX rlimit）。它防事故不防恶意：
    只加载可信来源的技能。
    """

    enabled: bool = True
    # 技能目录，每个技能一个子目录，内含 SKILL.md
    dir: str = "skills"
    # 单次执行的默认超时（秒），SKILL.md 里可按技能覆盖
    timeout_sec: float = 20.0
    max_output_chars: int = 8000
    # 沙盒进程的内存上限（MB）与最大进程数（含子进程）
    max_memory_mb: int = 256
    max_processes: int = 8
    # 执行脚本用的 Python 解释器；留空用当前解释器
    python: str = ""


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"


@dataclass(slots=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    status: StatusConfig = field(default_factory=StatusConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def ensure_dirs(self) -> None:
        for path in (self.paths.artifacts_dir, self.paths.logs_dir, self.paths.memory_dir, self.paths.workspace_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("config file must contain a mapping")
        _merge(cfg, data)
    if api_key := os.environ.get("LOLLAMA_UPSTREAM_API_KEY"):
        cfg.upstream.api_key = api_key
    if embedding_key := os.environ.get("LOLLAMA_EMBEDDING_API_KEY"):
        cfg.memory.retrieval.embedding.api_key = embedding_key
    validate_config(cfg)
    return cfg


def _merge(target: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)


def validate_config(cfg: Config) -> None:
    if cfg.runtime.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("runtime.log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    if not (0 < cfg.service.port < 65536):
        raise ValueError("service.port must be in 1..65535")
    if not cfg.service.ws_path.startswith("/"):
        raise ValueError("service.ws_path must start with /")
    if cfg.service.max_clients < 1:
        raise ValueError("service.max_clients must be positive")
    if cfg.service.health_timeout_sec <= 0:
        raise ValueError("service.health_timeout_sec must be positive")
    if cfg.upstream.max_tokens < 1:
        raise ValueError("upstream.max_tokens must be positive")
    if cfg.upstream.timeout_sec <= 0:
        raise ValueError("upstream.timeout_sec must be positive")
    if cfg.agent.max_history_turns < 1:
        raise ValueError("agent.max_history_turns must be positive")
    if cfg.memory.layers.working_max_turns < 1:
        raise ValueError("memory.layers.working_max_turns must be positive")
    for name in ("episodic", "semantic", "procedural", "core"):
        layer: LayerConfig = getattr(cfg.memory.layers, name)
        if layer.capacity < 1:
            raise ValueError(f"memory.layers.{name}.capacity must be positive")
        if layer.half_life_hours < 0:
            raise ValueError(f"memory.layers.{name}.half_life_hours must be >= 0 (0 disables decay)")
        if not (0 <= layer.min_strength <= 1):
            raise ValueError(f"memory.layers.{name}.min_strength must be in 0..1")
    if cfg.memory.retrieval.top_k < 1:
        raise ValueError("memory.retrieval.top_k must be positive")
    weights = (
        cfg.memory.retrieval.similarity_weight,
        cfg.memory.retrieval.importance_weight,
        cfg.memory.retrieval.strength_weight,
    )
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("memory.retrieval weights must be >= 0 and not all zero")
    channel_weights = (
        cfg.memory.retrieval.bigram_weight,
        cfg.memory.retrieval.fts_weight,
        cfg.memory.retrieval.vector_weight,
    )
    if any(w < 0 for w in channel_weights) or sum(channel_weights) <= 0:
        raise ValueError("memory.retrieval channel weights must be >= 0 and not all zero")
    if cfg.memory.retrieval.fts_candidates < 1:
        raise ValueError("memory.retrieval.fts_candidates must be positive")
    if cfg.memory.snapshot_min_interval_sec < 0:
        raise ValueError("memory.snapshot_min_interval_sec must be >= 0")
    embedding = cfg.memory.retrieval.embedding
    if embedding.enabled and not embedding.model:
        raise ValueError("memory.retrieval.embedding.model is required when embedding is enabled")
    if embedding.timeout_sec <= 0:
        raise ValueError("memory.retrieval.embedding.timeout_sec must be positive")
    if cfg.memory.promotion.episodic_hits_to_semantic < 1:
        raise ValueError("memory.promotion.episodic_hits_to_semantic must be positive")
    if cfg.memory.forgetting.sweep_interval_sec <= 0:
        raise ValueError("memory.forgetting.sweep_interval_sec must be positive")
    if cfg.memory.extraction.max_items_per_turn < 1:
        raise ValueError("memory.extraction.max_items_per_turn must be positive")
    if cfg.tools.max_rounds < 1:
        raise ValueError("tools.max_rounds must be positive")
    if cfg.tools.shell.timeout_sec <= 0:
        raise ValueError("tools.shell.timeout_sec must be positive")
    if cfg.skills.timeout_sec <= 0:
        raise ValueError("skills.timeout_sec must be positive")
    if cfg.skills.max_output_chars < 1:
        raise ValueError("skills.max_output_chars must be positive")
    if cfg.skills.max_memory_mb < 1:
        raise ValueError("skills.max_memory_mb must be positive")
    if cfg.skills.max_processes < 1:
        raise ValueError("skills.max_processes must be positive")
    for name in ("llm_waiting_after_sec", "llm_waiting_repeat_sec", "tool_waiting_after_sec", "tool_waiting_repeat_sec"):
        if getattr(cfg.status, name) < 0:
            raise ValueError(f"status.{name} must be >= 0 (0 disables)")
