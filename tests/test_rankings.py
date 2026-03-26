from pipeline.loaders import load_snapshot_payloads
from pipeline.rankings import (
    apply_snapshot_rankings,
    import_arena_rankings,
    import_artificial_analysis_rankings,
    import_livebench_rankings,
)


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


def test_import_artificial_analysis_rankings_sets_scores_and_ranks() -> None:
    models_data = {
        "models": {
            "gpt-5.2": {
                "display_name": "GPT-5.2",
                "modes": ["chat"],
            },
            "o3-pro": {
                "display_name": "o3-pro",
                "modes": ["chat"],
            },
        }
    }
    payload = {
        "data": [
            {
                "slug": "gpt-5.2",
                "name": "GPT-5.2",
                "model_creator": {"slug": "openai"},
                "evaluations": {
                    "artificial_analysis_intelligence_index": 71.4,
                    "livecodebench": 0.74,
                },
            },
            {
                "slug": "o3-pro",
                "name": "o3-pro",
                "model_creator": {"slug": "openai"},
                "evaluations": {
                    "artificial_analysis_intelligence_index": 78.9,
                    "livecodebench": 0.81,
                },
            },
        ]
    }

    updated, skipped, unmatched = import_artificial_analysis_rankings(
        payload,
        models_data,
        as_of="2026-03-26",
    )

    assert updated == 2
    assert skipped == 0
    assert unmatched == []
    assert models_data["models"]["o3-pro"]["rankings"]["artificial_analysis_intelligence_index"] == {
        "score": 78.9,
        "rank": 1,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["gpt-5.2"]["rankings"]["artificial_analysis_intelligence_index"] == {
        "score": 71.4,
        "rank": 2,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["o3-pro"]["rankings"]["livecodebench"] == {
        "score": 0.81,
        "rank": 1,
        "as_of": "2026-03-26",
    }


def test_apply_snapshot_rankings_imports_artificial_analysis_when_present() -> None:
    models_data = {
        "models": {
            "gpt-5.2": {
                "display_name": "GPT-5.2",
                "modes": ["chat"],
            }
        }
    }
    snapshot_payloads = {
        "artificial_analysis": {
            "data": [
                {
                    "slug": "gpt-5.2",
                    "name": "GPT-5.2",
                    "model_creator": {"slug": "openai"},
                    "evaluations": {
                        "artificial_analysis_intelligence_index": 71.4,
                    },
                }
            ]
        },
        "artificial_analysis_metadata": {
            "fetched_at": "2026-03-26T09:30:00Z",
            "sources": {"llms_models": "https://artificialanalysis.ai/api/v2/data/llms/models"},
        },
    }

    summary = apply_snapshot_rankings(models_data, snapshot_payloads)

    assert summary["sources"]["artificial_analysis"]["updated_models"] == 1
    assert summary["sources"]["artificial_analysis"]["as_of"] == "2026-03-26"
    assert models_data["models"]["gpt-5.2"]["rankings"]["artificial_analysis_intelligence_index"] == {
        "score": 71.4,
        "rank": 1,
        "as_of": "2026-03-26",
    }


def test_import_livebench_rankings_sets_overall_and_category_scores() -> None:
    models_data = {
        "models": {
            "gpt-4.5-preview": {
                "display_name": "GPT-4.5 Preview",
                "modes": ["chat"],
            },
            "o3-pro": {
                "display_name": "o3-pro",
                "modes": ["chat"],
            },
        }
    }
    payload = {
        "categories": {
            "Coding": ["code_generation"],
            "Reasoning": ["reasoning_chains"],
        },
        "table": [
            {
                "model": "gpt-4.5-preview-2025-02-27",
                "code_generation": 72.0,
                "reasoning_chains": 91.0,
            },
            {
                "model": "o3-pro",
                "code_generation": 84.0,
                "reasoning_chains": 97.0,
            },
        ],
    }

    updated, skipped, unmatched = import_livebench_rankings(
        payload,
        models_data,
        as_of="2026-03-26",
        release="2026-01-08",
    )

    assert updated == 2
    assert skipped == 0
    assert unmatched == []
    assert models_data["models"]["o3-pro"]["rankings"]["livebench"] == {
        "score": 90.5,
        "rank": 1,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["gpt-4.5-preview"]["rankings"]["livebench"] == {
        "score": 81.5,
        "rank": 2,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["o3-pro"]["rankings"]["livebench_coding"] == {
        "score": 84.0,
        "rank": 1,
        "as_of": "2026-03-26",
    }
    assert models_data["models"]["o3-pro"]["rankings"]["livebench_reasoning"] == {
        "score": 97.0,
        "rank": 1,
        "as_of": "2026-03-26",
    }


def test_apply_snapshot_rankings_imports_livebench_when_present() -> None:
    models_data = {
        "models": {
            "o3-pro": {
                "display_name": "o3-pro",
                "modes": ["chat"],
            }
        }
    }
    snapshot_payloads = {
        "livebench": {
            "categories": {"Coding": ["code_generation"]},
            "table": [
                {
                    "model": "o3-pro",
                    "code_generation": 84.0,
                }
            ],
        },
        "livebench_metadata": {
            "fetched_at": "2026-03-26T09:45:00Z",
            "release": "2026-01-08",
            "sources": {"table": "https://livebench.ai/table_2026_01_08.csv"},
        },
    }

    summary = apply_snapshot_rankings(models_data, snapshot_payloads)

    assert summary["sources"]["livebench"]["updated_models"] == 1
    assert summary["sources"]["livebench"]["as_of"] == "2026-03-26"
    assert summary["sources"]["livebench"]["release"] == "2026-01-08"
    assert models_data["models"]["o3-pro"]["rankings"]["livebench_coding"] == {
        "score": 84.0,
        "rank": 1,
        "as_of": "2026-03-26",
    }


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


def test_load_snapshot_payloads_reads_artificial_analysis_files(tmp_path) -> None:
    artificial_analysis_dir = tmp_path / "artificial_analysis"
    artificial_analysis_dir.mkdir()
    (artificial_analysis_dir / "llms_models.json").write_text(
        '{"status": 200, "data": [{"slug": "gpt-5.2"}]}',
        encoding="utf-8",
    )
    (artificial_analysis_dir / "fetch_metadata.json").write_text(
        '{"fetched_at": "2026-03-26T00:00:00Z", "sources": {"llms_models": "https://artificialanalysis.ai/api/v2/data/llms/models"}}',
        encoding="utf-8",
    )

    payloads = load_snapshot_payloads(tmp_path)

    assert payloads["artificial_analysis"]["data"][0]["slug"] == "gpt-5.2"
    assert payloads["artificial_analysis_metadata"]["fetched_at"] == "2026-03-26T00:00:00Z"


def test_load_snapshot_payloads_reads_livebench_files(tmp_path) -> None:
    livebench_dir = tmp_path / "livebench"
    livebench_dir.mkdir()
    (livebench_dir / "table.json").write_text(
        '[{"model": "o3-pro", "code_generation": "84.0"}]',
        encoding="utf-8",
    )
    (livebench_dir / "categories.json").write_text(
        '{"Coding": ["code_generation"]}',
        encoding="utf-8",
    )
    (livebench_dir / "fetch_metadata.json").write_text(
        '{"fetched_at": "2026-03-26T00:00:00Z", "release": "2026-01-08", "sources": {"table": "https://livebench.ai/table_2026_01_08.csv"}}',
        encoding="utf-8",
    )

    payloads = load_snapshot_payloads(tmp_path)

    assert payloads["livebench"]["table"][0]["model"] == "o3-pro"
    assert payloads["livebench"]["categories"]["Coding"] == ["code_generation"]
    assert payloads["livebench_metadata"]["fetched_at"] == "2026-03-26T00:00:00Z"
