from __future__ import annotations

from pathlib import Path

import pytest

from lollama.config import Config, load_config, validate_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "config.example.yaml"


def test_default_config_is_valid() -> None:
    validate_config(Config())


def test_load_example_config() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.service.port == 8801
    assert cfg.service.ws_path == "/v1/llm/ws"
    assert cfg.upstream.extra_body == {"reasoning_effort": "none"}
    assert cfg.memory.layers.core.half_life_hours == 0
    assert cfg.memory.layers.episodic.capacity == 300
    assert cfg.tools.shell.enabled is False


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("memory:\n  no_such_key: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)


def test_invalid_values_rejected() -> None:
    cfg = Config()
    cfg.memory.retrieval.top_k = 0
    with pytest.raises(ValueError):
        validate_config(cfg)

    cfg = Config()
    cfg.memory.layers.episodic.half_life_hours = -1
    with pytest.raises(ValueError):
        validate_config(cfg)

    cfg = Config()
    cfg.tools.max_rounds = 0
    with pytest.raises(ValueError):
        validate_config(cfg)
