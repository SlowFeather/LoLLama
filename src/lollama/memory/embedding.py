from __future__ import annotations

import math
from array import array

import httpx

from lollama._logging import get_logger
from lollama.config import EmbeddingConfig

logger = get_logger(__name__)


class EmbeddingClient:
    """OpenAI 兼容 /embeddings 客户端（LM Studio 本地 embedding 模型）。

    所有失败都吞掉并返回 None：向量通道是检索的增强项，不能拖垮主链路。
    """

    def __init__(self, cfg: EmbeddingConfig, *, fallback_base_url: str = ""):
        self.cfg = cfg
        base_url = (cfg.base_url or fallback_base_url).rstrip("/")
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(cfg.timeout_sec), headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[array] | None:
        """批量取单位化向量；任一环节失败返回 None。"""
        if not texts:
            return []
        try:
            response = await self._client.post("/embeddings", json={"model": self.cfg.model, "input": texts})
            response.raise_for_status()
            data = response.json().get("data", [])
            if len(data) != len(texts):
                logger.warning("embedding count mismatch: sent %d got %d", len(texts), len(data))
                return None
            ordered = sorted(data, key=lambda entry: int(entry.get("index", 0)))
            return [_unit(array("f", entry["embedding"])) for entry in ordered]
        except Exception as exc:
            logger.warning("embedding request failed (model=%s): %s", self.cfg.model, exc)
            return None

    async def embed_one(self, text: str) -> array | None:
        vectors = await self.embed([text])
        return vectors[0] if vectors else None


def _unit(vec: array) -> array:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 0:
        return vec
    return array("f", (value / norm for value in vec))


def cosine(a: array, b: array) -> float:
    """两个已单位化向量的余弦相似度（点积），维度不一致返回 0。"""
    if len(a) != len(b) or not len(a):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
