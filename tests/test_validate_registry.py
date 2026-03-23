import json
from pathlib import Path

from pipeline.loaders import load_curated_config
from scripts import build_registry as build_registry_module
from scripts import fetch_sources as fetch_sources_module
from scripts.build_registry import build_registry
from scripts.fetch_sources import SOURCE_DESCRIPTORS, snapshot_path_for_run
from scripts.validate import check_registry_quality


def _write_minimal_curated_config(curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "providers.json": {},
        "source_policies.json": {},
        "canonical_aliases.json": {},
        "rejections.json": {},
    }.items():
        (curated_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_path_for_run_nests_under_source_snapshots(tmp_path: Path) -> None:
    assert snapshot_path_for_run(tmp_path, "2026-03-23T06-15-00Z") == tmp_path / "source_snapshots" / "2026-03-23T06-15-00Z"


def test_fetch_sources_declares_expected_aggregator_sources() -> None:
    assert {descriptor.slug for descriptor in SOURCE_DESCRIPTORS} >= {"litellm", "openrouter", "llm_prices"}
    assert {descriptor.filename for descriptor in SOURCE_DESCRIPTORS} >= {
        "litellm_model_prices.json",
        "openrouter_models.json",
        "llm_prices_current.json",
    }


def test_build_registry_returns_top_level_sections(tmp_path: Path) -> None:
    _write_minimal_curated_config(tmp_path)
    result = build_registry(snapshot_dir=tmp_path, curated_dir=tmp_path)
    assert set(result.keys()) == {"version", "updated_at", "providers", "models", "provider_models"}


def test_build_registry_cli_writes_deterministic_outputs(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    monkeypatch.chdir(tmp_path)

    exit_code = build_registry_module.main(
        [
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--report-md",
            str(tmp_path / "tmp" / "build" / "report.md"),
        ]
    )

    assert exit_code == 0

    models_path = tmp_path / "models.json"
    models_min_path = tmp_path / "models.min.json"
    report_json_path = tmp_path / "tmp" / "build" / "report.json"
    report_md_path = tmp_path / "tmp" / "build" / "report.md"

    expected_registry = {
        "version": 1,
        "updated_at": "1970-01-01T00:00:00Z",
        "providers": {},
        "models": {},
        "provider_models": {},
    }
    expected_report = {
        "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
        "duplicate_clusters": [],
        "resolved_duplicates": [],
        "quarantine": [],
        "new_models": [],
    }

    assert json.loads(models_path.read_text(encoding="utf-8")) == expected_registry
    assert models_min_path.read_text(encoding="utf-8").strip() == json.dumps(expected_registry, ensure_ascii=False, separators=(",", ":"))
    assert json.loads(report_json_path.read_text(encoding="utf-8")) == expected_report
    assert report_md_path.read_text(encoding="utf-8").startswith("# Registry Audit Report")


def test_build_registry_cli_fetches_into_exact_snapshot_dir(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    captured: dict[str, Path] = {}

    def fake_fetch_sources(snapshot_dir: Path) -> Path:
        captured["fetched_snapshot_dir"] = snapshot_dir
        return snapshot_dir

    def fake_build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict, dict, list[dict]]:
        captured["build_snapshot_dir"] = snapshot_dir
        return (
            {
                "version": 1,
                "updated_at": "1970-01-01T00:00:00Z",
                "providers": {},
                "models": {},
                "provider_models": {},
            },
            {
                "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
                "duplicate_clusters": [],
                "resolved_duplicates": [],
                "quarantine": [],
                "new_models": [],
            },
            [],
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_registry_module, "fetch_sources_to", fake_fetch_sources)
    monkeypatch.setattr(build_registry_module, "build_registry_artifacts", fake_build_registry_artifacts)

    snapshot_dir = tmp_path / "tmp" / "source_snapshots" / "manual-run"
    exit_code = build_registry_module.main(["--fetch", "--snapshot-dir", str(snapshot_dir)])

    assert exit_code == 0
    assert captured["fetched_snapshot_dir"] == snapshot_dir
    assert captured["build_snapshot_dir"] == snapshot_dir


def test_build_registry_cli_fetch_without_snapshot_dir_uses_latest_snapshot_dir(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    captured: dict[str, Path] = {}

    def fake_fetch_sources(snapshot_dir: Path) -> Path:
        captured["fetched_snapshot_dir"] = snapshot_dir
        return snapshot_dir

    def fake_build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict, dict, list[dict]]:
        captured["build_snapshot_dir"] = snapshot_dir
        return (
            {
                "version": 1,
                "updated_at": "1970-01-01T00:00:00Z",
                "providers": {},
                "models": {},
                "provider_models": {},
            },
            {
                "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
                "duplicate_clusters": [],
                "resolved_duplicates": [],
                "quarantine": [],
                "new_models": [],
            },
            [],
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_registry_module, "fetch_sources_to", fake_fetch_sources)
    monkeypatch.setattr(build_registry_module, "build_registry_artifacts", fake_build_registry_artifacts)

    exit_code = build_registry_module.main(["--fetch"])

    expected_snapshot_dir = tmp_path / "tmp" / "source_snapshots" / "latest"
    assert exit_code == 0
    assert captured["fetched_snapshot_dir"] == expected_snapshot_dir
    assert captured["build_snapshot_dir"] == expected_snapshot_dir


def test_build_registry_cli_defaults_to_latest_snapshot_dir_without_fetch(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_curated_config(tmp_path / "registry" / "curated")

    captured: dict[str, Path] = {}

    def fake_build_registry_artifacts(snapshot_dir: Path, curated_dir: Path) -> tuple[dict, dict, list[dict]]:
        captured["build_snapshot_dir"] = snapshot_dir
        return (
            {
                "version": 1,
                "updated_at": "1970-01-01T00:00:00Z",
                "providers": {},
                "models": {},
                "provider_models": {},
            },
            {
                "summary": {"duplicate_clusters": 0, "quarantine_count": 0},
                "duplicate_clusters": [],
                "resolved_duplicates": [],
                "quarantine": [],
                "new_models": [],
            },
            [],
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_registry_module, "build_registry_artifacts", fake_build_registry_artifacts)

    exit_code = build_registry_module.main([])

    assert exit_code == 0
    assert captured["build_snapshot_dir"] == tmp_path / "tmp" / "source_snapshots" / "latest"


def test_fetch_bytes_retries_with_timeout(monkeypatch) -> None:
    attempts: list[float] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return b"payload"

    def flaky_urlopen(request, timeout: float):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise TimeoutError("slow upstream")
        return FakeResponse()

    monkeypatch.setattr(fetch_sources_module, "urlopen", flaky_urlopen)
    monkeypatch.setattr(fetch_sources_module.time, "sleep", lambda _delay: None)

    payload = fetch_sources_module._fetch_bytes(
        "https://example.com/models.json",
        timeout=12.5,
        retries=3,
        retry_delay=0.01,
    )

    assert payload == b"payload"
    assert attempts == [12.5, 12.5, 12.5]


def test_check_registry_quality_rejects_provider_specific_model_ids() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {"bedrock": {"display_name": "AWS Bedrock"}},
        "models": {
            "anthropic.claude-opus-4-6-v1:0": {
                "display_name": "Bad",
                "modes": ["chat"],
            }
        },
        "provider_models": {},
    }

    errors = check_registry_quality(data)

    assert any("provider-specific model key" in error for error in errors)


def test_check_registry_quality_rejects_unknown_owned_by() -> None:
    data = {
        "version": 1,
        "updated_at": "2026-03-23T00:00:00Z",
        "providers": {"openai": {"display_name": "OpenAI"}},
        "models": {
            "gpt-4o": {
                "display_name": "GPT-4o",
                "modes": ["chat"],
                "owned_by": "unknown",
            }
        },
        "provider_models": {},
    }

    errors = check_registry_quality(data)

    assert any("owned_by" in error for error in errors)


def test_load_curated_config_reads_authority_files(tmp_path: Path) -> None:
    expected = {
        "providers": {"openai": {"display_name": "OpenAI"}},
        "source_policies": {
            "official_sources": ["openai", "anthropic", "gemini", "xai", "mistral", "cohere", "deepseek"],
            "aggregator_sources": ["litellm", "openrouter", "portkey", "llm_prices"],
            "field_authority": {
                "owned_by": ["official"],
                "display_name": ["official"],
                "pricing": ["official", "portkey", "llm_prices", "openrouter", "litellm"],
            },
        },
        "canonical_aliases": {"gpt-4o-mini": "gpt-4o-mini"},
        "rejections": {
            "exact_model_ids": ["sample_spec", "search_api", "model_router"],
            "prefixes": ["1024-x-1024/", "512-x-512/", "256-x-256/"],
        },
    }

    for name, payload in expected.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    config = load_curated_config(tmp_path)
    assert config == expected
