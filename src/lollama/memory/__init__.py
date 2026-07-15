from lollama.config import MemoryConfig

from .embedding import EmbeddingClient
from .manager import LAYER_LABELS, PERSISTED_LAYERS, MemoryItem, MemoryManager

__all__ = [
    "MemoryManager",
    "MemoryItem",
    "PERSISTED_LAYERS",
    "LAYER_LABELS",
    "EmbeddingClient",
    "build_memory",
]


def build_memory(cfg: MemoryConfig, memory_dir, *, upstream_base_url: str = "") -> MemoryManager:
    """按配置构建记忆管理器；开启向量检索时挂上 embedding 客户端。"""
    manager = MemoryManager(cfg, memory_dir)
    if cfg.retrieval.embedding.enabled:
        manager.set_embedder(EmbeddingClient(cfg.retrieval.embedding, fallback_base_url=upstream_base_url))
    return manager
