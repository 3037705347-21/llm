import json
import re

import pytest
from click.testing import CliRunner

import llm
from llm.cli import cli
from llm.logs import backfill_resolved_models
from llm.migrations import migrate

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-02-01T00:00:00+00:00"
T2 = "2026-03-01T00:00:00+00:00"


def _add_turn(db, id, model, resolved_model=None, datetime_utc=T0):
    db["turns"].insert(
        {
            "id": id,
            "thread_id": "thread-1",
            "parent_message_hash": "hash-1",
            "tip_message_hash": "hash-1",
            "model": model,
            "resolved_model": resolved_model,
            "datetime_utc": datetime_utc,
        }
    )


def _add_response(db, id, model, resolved_model=None, datetime_utc=T0):
    db["responses"].insert(
        {
            "id": id,
            "model": model,
            "resolved_model": resolved_model,
            "datetime_utc": datetime_utc,
        }
    )


def _resolved(db, table="turns"):
    return {row["id"]: row["resolved_model"] for row in db[table].rows}


@pytest.mark.parametrize("model_id_or_alias", ("gpt-3.5-turbo", "chatgpt"))
def test_set_alias(model_id_or_alias):
    with pytest.raises(llm.UnknownModelError):
        llm.get_model("this-is-a-new-alias")
    llm.set_alias("this-is-a-new-alias", model_id_or_alias)
    assert llm.get_model("this-is-a-new-alias").model_id == "gpt-3.5-turbo"


def test_remove_alias():
    with pytest.raises(KeyError):
        llm.remove_alias("some-other-alias")
    llm.set_alias("some-other-alias", "gpt-3.5-turbo")
    assert llm.get_model("some-other-alias").model_id == "gpt-3.5-turbo"
    llm.remove_alias("some-other-alias")
    with pytest.raises(llm.UnknownModelError):
        llm.get_model("some-other-alias")


@pytest.mark.parametrize("args", (["aliases", "list"], ["aliases"]))
def test_cli_aliases_list(args):
    llm.set_alias("e-demo", "embed-demo")
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    for line in [
        "3.5         : gpt-3.5-turbo",
        "chatgpt     : gpt-3.5-turbo",
        "chatgpt-16k : gpt-3.5-turbo-16k",
        "3.5-16k     : gpt-3.5-turbo-16k",
        "4           : gpt-4",
        "gpt4        : gpt-4",
        "e-demo      : embed-demo (embedding)",
        "ada         : text-embedding-ada-002 (embedding)",
        "",
    ]:
        line = line.strip()
        if not line:
            continue
        # Turn the whitespace into a regex
        regex = r"\s+".join(re.escape(part) for part in line.split())
        assert re.search(regex, result.output)


@pytest.mark.parametrize("args", (["aliases", "list"], ["aliases"]))
def test_cli_aliases_list_empty(monkeypatch, args):
    monkeypatch.setattr("llm.cli.get_model_aliases", dict)
    monkeypatch.setattr("llm.cli.get_embedding_model_aliases", dict)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0
    assert result.output == ""


@pytest.mark.parametrize("args", (["aliases", "list"], ["aliases"]))
def test_cli_aliases_list_json(args):
    llm.set_alias("e-demo", "embed-demo")
    runner = CliRunner()
    result = runner.invoke(cli, args + ["--json"])
    assert result.exit_code == 0
    assert (
        json.loads(result.output).items()
        >= {
            "3.5": "gpt-3.5-turbo",
            "chatgpt": "gpt-3.5-turbo",
            "chatgpt-16k": "gpt-3.5-turbo-16k",
            "3.5-16k": "gpt-3.5-turbo-16k",
            "4": "gpt-4",
            "gpt4": "gpt-4",
            "ada": "text-embedding-ada-002",
            "e-demo": "embed-demo",
        }.items()
    )


@pytest.mark.parametrize(
    "args,expected,expected_error",
    (
        (["foo", "bar"], {"foo": "bar"}, None),
        (["foo", "-q", "mo"], {"foo": "mock"}, None),
        (["foo", "-q", "mog"], None, "No model found matching query: mog"),
    ),
)
def test_cli_aliases_set(user_path, args, expected, expected_error):
    # Should be not aliases.json at start
    assert not (user_path / "aliases.json").exists()
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "set"] + args)
    if not expected_error:
        assert result.exit_code == 0
        assert (user_path / "aliases.json").exists()
        assert json.loads((user_path / "aliases.json").read_text("utf-8")) == expected
    else:
        assert result.exit_code == 1
        assert result.output.strip() == f"Error: {expected_error}"


def test_cli_aliases_path(user_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "path"])
    assert result.exit_code == 0
    assert result.output.strip() == str(user_path / "aliases.json")


def test_cli_aliases_remove(user_path):
    (user_path / "aliases.json").write_text(json.dumps({"foo": "bar"}), "utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "remove", "foo"])
    assert result.exit_code == 0
    assert json.loads((user_path / "aliases.json").read_text("utf-8")) == {}


def test_cli_aliases_remove_invalid(user_path):
    (user_path / "aliases.json").write_text(json.dumps({"foo": "bar"}), "utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "remove", "invalid"])
    assert result.exit_code == 1
    assert result.output == "Error: No such alias: invalid\n"


@pytest.mark.parametrize("args", (["models"], ["models", "list"]))
def test_cli_aliases_are_registered(user_path, args):
    (user_path / "aliases.json").write_text(
        json.dumps({"foo": "bar", "turbo": "gpt-3.5-turbo"}), "utf-8"
    )
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    # Check for model line only, without keys, as --options is not used
    assert "gpt-3.5-turbo (aliases: 3.5, chatgpt, turbo)" in result.output


# -- alias rotation history ---------------------------------------------


def test_set_alias_records_rotation(user_path):
    assert llm.set_alias("luna", "gpt-5") is None
    # Creating an alias records no rotation
    assert not (user_path / "alias_rotations.json").exists()
    # Switching it records old and new
    assert llm.set_alias("luna", "gpt-6") == "gpt-5"
    rotations = json.loads((user_path / "alias_rotations.json").read_text("utf-8"))
    entry = rotations["luna"][0]
    assert entry["old"] == "gpt-5"
    assert entry["new"] == "gpt-6"
    assert entry["datetime"]
    # Setting it to the same model again records nothing
    llm.set_alias("luna", "gpt-6")
    rotations = json.loads((user_path / "alias_rotations.json").read_text("utf-8"))
    assert len(rotations["luna"]) == 1


def test_set_alias_aborts_when_rotation_history_unreadable(user_path):
    (user_path / "aliases.json").write_text(json.dumps({"luna": "gpt-5"}), "utf-8")
    (user_path / "alias_rotations.json").write_text("{broken json", "utf-8")
    with pytest.raises(llm.AliasHistoryError):
        llm.set_alias("luna", "gpt-6")
    # Neither file was modified
    assert json.loads((user_path / "aliases.json").read_text("utf-8")) == {
        "luna": "gpt-5"
    }
    assert (user_path / "alias_rotations.json").read_text("utf-8") == "{broken json"


def test_set_alias_aborts_when_aliases_file_unreadable(user_path):
    (user_path / "aliases.json").write_text("nope", "utf-8")
    with pytest.raises(llm.AliasHistoryError):
        llm.set_alias("luna", "gpt-6")
    assert (user_path / "aliases.json").read_text("utf-8") == "nope"
    assert not (user_path / "alias_rotations.json").exists()


@pytest.mark.parametrize(
    "content",
    (
        "not json",
        json.dumps({"luna": "not a list"}),
        json.dumps({"luna": [{"old": "gpt-5"}]}),
        json.dumps({"luna": [{"old": "gpt-5", "new": 5, "datetime": T1}]}),
    ),
)
def test_load_alias_rotations_validates_shape(user_path, content):
    (user_path / "alias_rotations.json").write_text(content, "utf-8")
    with pytest.raises(llm.AliasHistoryError):
        llm.load_alias_rotations()


# -- resolved-model backfill --------------------------------------------


@pytest.fixture
def migrated_db(logs_db):
    migrate(logs_db)
    return logs_db


def _seed_rotations(user_path, aliases, rotations):
    (user_path / "aliases.json").write_text(json.dumps(aliases), "utf-8")
    (user_path / "alias_rotations.json").write_text(json.dumps(rotations), "utf-8")


def test_backfill_resolves_rotation_windows(user_path, migrated_db):
    _seed_rotations(
        user_path,
        {"luna": "gpt-7"},
        {
            "luna": [
                {"old": "gpt-5", "new": "gpt-6", "datetime": T1},
                {"old": "gpt-6", "new": "gpt-7", "datetime": T2},
            ]
        },
    )
    _add_turn(migrated_db, "before", "luna", datetime_utc=T0)
    _add_turn(migrated_db, "at-first", "luna", datetime_utc=T1)
    _add_turn(migrated_db, "between", "luna", datetime_utc="2026-02-15T00:00:00+00:00")
    _add_turn(migrated_db, "at-second", "luna", datetime_utc=T2)
    _add_response(migrated_db, "legacy-before", "luna", datetime_utc=T0)
    _add_response(migrated_db, "legacy-between", "luna", datetime_utc=T1)

    updated = backfill_resolved_models(migrated_db)

    assert updated == 6
    assert _resolved(migrated_db) == {
        "before": "gpt-5",
        "at-first": "gpt-6",
        "between": "gpt-6",
        "at-second": "gpt-7",
    }
    assert _resolved(migrated_db, "responses") == {
        "legacy-before": "gpt-5",
        "legacy-between": "gpt-6",
    }


def test_backfill_alias_without_rotations_uses_current_model(user_path, migrated_db):
    _seed_rotations(user_path, {"luna": "gpt-5"}, {})
    _add_turn(migrated_db, "one", "luna", datetime_utc=T0)
    _add_turn(migrated_db, "two", "luna")
    # A row without a timestamp is still resolved: no windows to place it in.
    migrated_db["turns"].insert(
        {
            "id": "undated",
            "thread_id": "thread-1",
            "parent_message_hash": "hash-1",
            "tip_message_hash": "hash-1",
            "model": "luna",
            "resolved_model": None,
            "datetime_utc": None,
        }
    )
    assert backfill_resolved_models(migrated_db) == 3
    assert set(_resolved(migrated_db).values()) == {"gpt-5"}


def test_backfill_leaves_known_and_unrelated_rows_alone(user_path, migrated_db):
    _seed_rotations(
        user_path,
        {"luna": "gpt-6"},
        {"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]},
    )
    _add_turn(
        migrated_db,
        "already-known",
        "luna",
        resolved_model="provider-said-gpt-5",
        datetime_utc=T0,
    )
    _add_turn(migrated_db, "concrete-model", "gpt-4", datetime_utc=T0)
    _add_turn(migrated_db, "unknown-alias", "mystery", datetime_utc=T0)
    # Undated rows cannot be placed in a window: leave unresolved.
    migrated_db["turns"].insert(
        {
            "id": "undated",
            "thread_id": "thread-1",
            "parent_message_hash": "hash-1",
            "tip_message_hash": "hash-1",
            "model": "luna",
            "resolved_model": None,
            "datetime_utc": None,
        }
    )

    assert backfill_resolved_models(migrated_db) == 0
    assert _resolved(migrated_db) == {
        "already-known": "provider-said-gpt-5",
        "concrete-model": None,
        "unknown-alias": None,
        "undated": None,
    }


def test_backfill_is_idempotent(user_path, migrated_db):
    _seed_rotations(
        user_path,
        {"luna": "gpt-6"},
        {"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]},
    )
    _add_turn(migrated_db, "before", "luna", datetime_utc=T0)
    _add_turn(migrated_db, "after", "luna", datetime_utc=T2)

    assert backfill_resolved_models(migrated_db) == 2
    first = _resolved(migrated_db)
    assert backfill_resolved_models(migrated_db) == 0
    assert _resolved(migrated_db) == first


def test_backfill_aborts_without_writing_when_inputs_unreadable(user_path, migrated_db):
    _seed_rotations(
        user_path,
        {"luna": "gpt-6"},
        {"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]},
    )
    _add_turn(migrated_db, "before", "luna", datetime_utc=T0)
    _add_response(migrated_db, "legacy", "luna", datetime_utc=T0)

    (user_path / "alias_rotations.json").write_text("{broken", "utf-8")
    with pytest.raises(llm.AliasHistoryError):
        backfill_resolved_models(migrated_db)
    assert all(v is None for v in _resolved(migrated_db).values())
    assert all(v is None for v in _resolved(migrated_db, "responses").values())

    # Same guarantee when aliases.json itself cannot be read
    (user_path / "alias_rotations.json").write_text(
        json.dumps({"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]}),
        "utf-8",
    )
    (user_path / "aliases.json").write_text("{broken", "utf-8")
    with pytest.raises(llm.AliasHistoryError):
        backfill_resolved_models(migrated_db)
    assert all(v is None for v in _resolved(migrated_db).values())
    assert all(v is None for v in _resolved(migrated_db, "responses").values())


def test_backfill_resumes_after_an_interruption(user_path, migrated_db, monkeypatch):
    _seed_rotations(
        user_path,
        {"bbb": "fixed-model"},
        {"aaa": [{"old": "old-a", "new": "new-a", "datetime": T1}]},
    )
    _add_turn(migrated_db, "a-before", "aaa", datetime_utc=T0)
    _add_turn(migrated_db, "b-one", "bbb", datetime_utc=T0)

    import llm.logs

    real_window = llm.logs._backfill_window
    fail_bbb = {"value": True}

    def failing_window(db, table, alias, model_id, start, end):
        if fail_bbb["value"] and alias == "bbb":
            raise RuntimeError("simulated interruption")
        return real_window(db, table, alias, model_id, start, end)

    monkeypatch.setattr(llm.logs, "_backfill_window", failing_window)
    with pytest.raises(RuntimeError):
        backfill_resolved_models(migrated_db)

    # The alias that committed stayed; the interrupted alias rolled back.
    assert _resolved(migrated_db) == {"a-before": "old-a", "b-one": None}

    # Flip the failure off (without monkeypatch.undo(), which would also
    # revert LLM_USER_PATH) and rerun: it finishes the remaining alias.
    fail_bbb["value"] = False
    assert backfill_resolved_models(migrated_db) == 1
    assert _resolved(migrated_db) == {"a-before": "old-a", "b-one": "fixed-model"}


# -- CLI wiring ----------------------------------------------------------


def test_cli_aliases_set_records_rotation_and_backfills(user_path, logs_db):
    migrate(logs_db)
    (user_path / "aliases.json").write_text(json.dumps({"luna": "gpt-5"}), "utf-8")
    _add_turn(logs_db, "before", "luna", datetime_utc=T0)
    _add_response(logs_db, "legacy", "luna", datetime_utc=T0)

    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "set", "luna", "gpt-6"])
    assert result.exit_code == 0
    assert "switched from 'gpt-5' to 'gpt-6'" in result.output
    assert "Resolved 2 logged record(s)" in result.output

    rotations = json.loads((user_path / "alias_rotations.json").read_text("utf-8"))
    assert rotations["luna"][0]["old"] == "gpt-5"
    assert _resolved(logs_db) == {"before": "gpt-5"}
    assert _resolved(logs_db, "responses") == {"legacy": "gpt-5"}


def test_cli_aliases_backfill_command(user_path, logs_db):
    migrate(logs_db)
    _seed_rotations(
        user_path,
        {"luna": "gpt-6"},
        {"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]},
    )
    _add_turn(logs_db, "before", "luna", datetime_utc=T0)

    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "backfill"])
    assert result.exit_code == 0
    assert result.output.strip() == "Resolved 1 logged record(s)"
    assert _resolved(logs_db) == {"before": "gpt-5"}

    # Re-running converges
    result = runner.invoke(cli, ["aliases", "backfill"])
    assert result.exit_code == 0
    assert result.output.strip() == "Resolved 0 logged record(s)"


def test_cli_aliases_backfill_without_database(user_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "backfill"])
    assert result.exit_code == 0
    assert "No log database found" in result.output


def test_cli_aliases_backfill_corrupt_history(user_path, logs_db):
    migrate(logs_db)
    (user_path / "aliases.json").write_text(json.dumps({"luna": "gpt-6"}), "utf-8")
    (user_path / "alias_rotations.json").write_text("{broken", "utf-8")
    _add_turn(logs_db, "before", "luna", datetime_utc=T0)

    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "backfill"])
    assert result.exit_code == 1
    assert "Could not read alias rotation history" in result.output
    assert _resolved(logs_db) == {"before": None}


def test_cli_aliases_history(user_path):
    _seed_rotations(
        user_path,
        {"luna": "gpt-6"},
        {"luna": [{"old": "gpt-5", "new": "gpt-6", "datetime": T1}]},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["aliases", "history", "luna"])
    assert result.exit_code == 0
    assert result.output.strip() == f"{T1}\tluna\tgpt-5\tgpt-6"

    # Unknown alias is an empty listing, not an error
    result = runner.invoke(cli, ["aliases", "history", "nope"])
    assert result.exit_code == 0
    assert result.output == ""
