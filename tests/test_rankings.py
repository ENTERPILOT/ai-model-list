from pipeline.loaders import load_snapshot_payloads
from pipeline.rankings import apply_snapshot_rankings, import_arena_rankings


def test_import_arena_rankings_matches_aliases_and_safe_variant_suffixes() -> None:
    models_data = {
        "models": {
            "gpt-5": {
                "display_name": "GPT-5",
                "modes": ["chat"],
            },
            "claude-opus-4-1": {
                "display_name": "Claude Opus 4.1",
                "modes": ["chat"],
                "aliases": ["claude-opus-4-1-20250805"],
            },
            "deepseek-v3.1-terminus": {
                "display_name": "DeepSeek V3.1 Terminus",
                "modes": ["chat"],
            },
            "gemini-3-pro-preview": {
                "display_name": "Gemini 3 Pro Preview",
                "modes": ["chat"],
            },
        }
    }
    leaderboards = {
        "leaderboard-text.json": {
            "full": {
                "gpt-5-high": {"rating": 1510.2},
                "claude-opus-4-1-20250805-thinking-16k": {"rating": 1501.8},
                "deepseek-v3.1-terminus-thinking": {"rating": 1499.6},
                "gemini-3-pro": {"rating": 1498.5},
            }
        }
    }

    updated, skipped, unmatched = import_arena_rankings(
        leaderboards,
        models_data,
        categories={("leaderboard-text.json", "full"): "chatbot_arena"},
        as_of="2026-03-26",
    )

    assert updated == 4
    assert skipped == 0
    assert unmatched == []
    assert models_data["models"]["gpt-5"]["rankings"]["chatbot_arena"] == {
        "elo": 1510,
        "rank": 1,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["claude-opus-4-1"]["rankings"]["chatbot_arena"] == {
        "elo": 1502,
        "rank": 2,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["deepseek-v3.1-terminus"]["rankings"]["chatbot_arena"] == {
        "elo": 1500,
        "rank": 3,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["gemini-3-pro-preview"]["rankings"]["chatbot_arena"] == {
        "elo": 1498,
        "rank": 4,
        "as_of": "2026-03-26",
    }


def test_apply_snapshot_rankings_uses_snapshot_metadata_date() -> None:
    models_data = {
        "models": {
            "gpt-5": {
                "display_name": "GPT-5",
                "modes": ["chat"],
            }
        }
    }
    snapshot_payloads = {
        "arena_catalog": {
            "leaderboard-text.json": {
                "full": {
                    "gpt-5-high": {"rating": 1510.2},
                }
            }
        },
        "arena_catalog_metadata": {
            "fetched_at": "2026-03-26T08:15:30Z",
            "sources": {"leaderboard-text.json": "https://example.com/leaderboard-text.json"},
        },
    }

    summary = apply_snapshot_rankings(models_data, snapshot_payloads)

    assert summary["sources"]["arena_catalog"]["updated_models"] == 1
    assert summary["sources"]["arena_catalog"]["as_of"] == "2026-03-26"
    assert models_data["models"]["gpt-5"]["rankings"]["chatbot_arena"]["as_of"] == "2026-03-26"


def test_load_snapshot_payloads_reads_arena_catalog_files(tmp_path) -> None:
    arena_dir = tmp_path / "arena_catalog"
    arena_dir.mkdir()
    (arena_dir / "leaderboard-text.json").write_text('{"full": {"gpt-5": {"rating": 1500}}}', encoding="utf-8")
    (arena_dir / "leaderboard-vision.json").write_text('{"full": {"gpt-5": {"rating": 1400}}}', encoding="utf-8")
    (arena_dir / "fetch_metadata.json").write_text(
        '{"fetched_at": "2026-03-26T00:00:00Z", "sources": {"leaderboard-text.json": "https://example.com"}}',
        encoding="utf-8",
    )

    payloads = load_snapshot_payloads(tmp_path)

    assert set(payloads["arena_catalog"]) == {"leaderboard-text.json", "leaderboard-vision.json"}
    assert payloads["arena_catalog"]["leaderboard-text.json"]["full"]["gpt-5"]["rating"] == 1500
    assert payloads["arena_catalog_metadata"]["fetched_at"] == "2026-03-26T00:00:00Z"
