from __future__ import annotations

import unicodedata

_IGNORED_PUNCTUATION = "。．.!！？?，,；;：:、…~～\"'“”‘’()（）[]【】{}《》<>〈〉「」『』"


def normalize(text: str) -> str:
    """NFKC + casefold，去空白与常见中英标点；作为相似度与全文索引的统一口径。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if not ch.isspace() and ch not in _IGNORED_PUNCTUATION)


def char_bigrams(text: str) -> frozenset[str]:
    text = normalize(text)
    if len(text) < 2:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i : i + 2] for i in range(len(text) - 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)
