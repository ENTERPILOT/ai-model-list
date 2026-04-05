from pathlib import Path

from scripts.check_registry_changes import (
    NO_MEANINGFUL_CHANGES_EXIT_CODE,
    has_meaningful_file_changes,
    has_meaningful_json_changes,
    main,
    strip_ignored_date_fields,
)


def test_strip_ignored_date_fields_removes_updated_at_and_as_of_recursively() -> None:
    document = {
        "updated_at": "2026-04-05T18:51:09Z",
        "models": {
            "gpt-5": {
                "display_name": "GPT-5",
                "rankings": {
                    "chatbot_arena": {
                        "elo": 1500,
                        "rank": 1,
                        "as_of": "2026-04-05",
                    }
                },
            }
        },
    }

    assert strip_ignored_date_fields(document) == {
        "models": {
            "gpt-5": {
                "display_name": "GPT-5",
                "rankings": {
                    "chatbot_arena": {
                        "elo": 1500,
                        "rank": 1,
                    }
                },
            }
        }
    }


def test_has_meaningful_json_changes_ignores_updated_at_only_changes() -> None:
    previous = {"version": 1, "updated_at": "2026-04-05T07:12:43Z", "models": {}}
    current = {"version": 1, "updated_at": "2026-04-05T18:51:09Z", "models": {}}

    assert has_meaningful_json_changes(previous, current) is False


def test_has_meaningful_json_changes_ignores_as_of_only_changes() -> None:
    previous = {
        "models": {
            "gpt-5": {
                "rankings": {
                    "chatbot_arena": {"elo": 1500, "rank": 1, "as_of": "2026-04-04"}
                }
            }
        }
    }
    current = {
        "models": {
            "gpt-5": {
                "rankings": {
                    "chatbot_arena": {"elo": 1500, "rank": 1, "as_of": "2026-04-05"}
                }
            }
        }
    }

    assert has_meaningful_json_changes(previous, current) is False


def test_has_meaningful_json_changes_keeps_model_lifecycle_dates_meaningful() -> None:
    previous = {"models": {"gpt-5": {"release_date": "2026-04-01"}}}
    current = {"models": {"gpt-5": {"release_date": "2026-04-02"}}}

    assert has_meaningful_json_changes(previous, current) is True


def test_has_meaningful_file_changes_returns_false_for_ignored_date_only_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text('{"version":1,"updated_at":"2026-04-05T18:51:09Z","models":{}}\n', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.check_registry_changes._load_json_from_git",
        lambda _base_ref, _path: {"version": 1, "updated_at": "2026-04-05T07:12:43Z", "models": {}},
    )

    assert has_meaningful_file_changes("HEAD", [models_path]) is False


def test_main_returns_skip_exit_code_for_ignored_date_only_diff(tmp_path: Path, monkeypatch) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text('{"version":1,"updated_at":"2026-04-05T18:51:09Z","models":{}}\n', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.check_registry_changes.has_meaningful_file_changes",
        lambda _base_ref, _paths: False,
    )

    assert main([str(models_path)]) == NO_MEANINGFUL_CHANGES_EXIT_CODE
