"""Configuration loading, merging and error reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Config, deep_merge, load_dotenv
from app.errors import ConfigurationError


def test_defaults_are_usable() -> None:
    config = Config.load(use_env=False)
    assert config.llm.provider == "ollama"
    assert config.retrieval.top_k > 0
    assert config.report.language in {"pt", "en"}


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  model: my-model\n  temperature: 0.4\n", encoding="utf-8")
    config = Config.load(path, use_env=False)
    assert config.llm.model == "my-model"
    assert config.llm.temperature == 0.4
    assert config.llm.provider == "ollama"  # untouched keys keep their default


def test_cli_overrides_win_over_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  model: from-file\n", encoding="utf-8")
    config = Config.load(path, overrides={"llm": {"model": "from-cli"}}, use_env=False)
    assert config.llm.model == "from-cli"


def test_none_overrides_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  model: from-file\n", encoding="utf-8")
    config = Config.load(path, overrides={"llm": {"model": None}}, use_env=False)
    assert config.llm.model == "from-file"


def test_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEWER2_LLM__MODEL", "env-model")
    monkeypatch.setenv("REVIEWER2_RETRIEVAL__TOP_K", "9")
    monkeypatch.setenv("REVIEWER2_VERIFICATION__ENABLED", "false")
    config = Config.load(use_env=True)
    assert config.llm.model == "env-model"
    assert config.retrieval.top_k == 9
    assert config.verification.enabled is False


def test_missing_config_file_reports_action() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Config.load(Path("does-not-exist.yaml"), use_env=False)
    assert "recommended action" in excinfo.value.format()


def test_invalid_yaml_reports_cause(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        Config.load(path, use_env=False)


def test_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("REVIEWER2_LLM__MODEL=from-dotenv\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setenv("REVIEWER2_LLM__MODEL", "already-set")
    load_dotenv(env_file)
    import os

    assert os.environ["REVIEWER2_LLM__MODEL"] == "already-set"
    assert os.environ["OTHER"] == "1"


def test_deep_merge_is_recursive_and_pure() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    extra = {"a": {"c": 99}}
    merged = deep_merge(base, extra)
    assert merged == {"a": {"b": 1, "c": 99}, "d": 3}
    assert base["a"]["c"] == 2


def test_paths_ensure_creates_directories(tmp_path: Path) -> None:
    config = Config.load(
        use_env=False, overrides={"paths": {"reports_dir": str(tmp_path / "r"), "data_dir": str(tmp_path / "d")}}
    )
    config.paths.ensure()
    assert (tmp_path / "r").is_dir()


def test_config_yaml_matches_the_schema() -> None:
    """The shipped config.yaml must not drift from the configuration model."""
    import yaml

    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    config = Config.load(Path("config.yaml"), use_env=False)

    for key, value in raw.items():
        assert key in Config.model_fields, f"unknown key in config.yaml: {key}"
        if isinstance(value, dict):
            section = getattr(config, key)
            for sub_key in value:
                assert sub_key in type(section).model_fields, f"unknown key: {key}.{sub_key}"
