"""Tests for llm.reconcile - merging equivalent tool instances.

The runtime normalizes constructor arguments as they are logged, so the
duplicates reconciliation deals with come from earlier eras: instance
rows with differently-written but equivalent arguments, rows nothing
references, and legacy tool_results references. Tests build that shape
directly rather than relying on the current writer to reproduce it.
"""

import json

import pytest
import sqlite_utils
from click.testing import CliRunner

import llm
from llm.cli import cli
from llm.logs import LogStore, log_row_extras, merged_log_rows
from llm.reconcile import (
    ReconcileConflictError,
    ToolInstanceConfigError,
    reconcile_tool_instances,
)

OLD_ARGUMENTS = '{"b": 2, "a": 1}'
NEW_ARGUMENTS = '{"a": 1, "b": 2}'


@pytest.fixture
def store(db_factory):
    return LogStore(db_factory(memory=True))


class Box(llm.Toolbox):
    def __init__(self, a: int, b: int):
        pass

    def run(self) -> str:
        "Run the box"
        return "ran"


def _log_box_turn(store, mock_model):
    """Log one turn against Box, as the current writer would."""
    mock_model.enqueue(["ok"])
    response = mock_model.prompt(
        "go",
        tools=[Box(a=1, b=2)],
        tool_results=[
            llm.ToolResult(
                name="Box_run",
                output="x",
                tool_call_id="c1",
                instance=Box(a=1, b=2),
            )
        ],
    )
    response.text()
    response.log_to_db(store.db)
    return next(iter(store.db["turns"].rows))["id"]


def _make_split(store, mock_model):
    """A historical turn that references an old-writing equivalent instance.

    Returns (turn_id, tool_id, new_id, old_id).
    """
    turn_id = _log_box_turn(store, mock_model)
    new_id = next(iter(store.db["tool_instances"].rows))["id"]
    tool_id = next(iter(store.db["turn_tools"].rows))["tool_id"]

    old_id = (
        store.db["tool_instances"]
        .insert({"name": "Box", "plugin": None, "arguments": OLD_ARGUMENTS})
        .last_pk
    )
    store.db["turn_tools"].update((turn_id, tool_id), {"instance_id": old_id})
    store.db["tool_instantiations"].update((turn_id, "c1"), {"instance_id": old_id})
    return turn_id, tool_id, new_id, old_id


# ---- merging -----------------------------------------------------------


def test_equivalent_configs_merge_and_references_rebind(store, mock_model):
    _, _, new_id, old_id = _make_split(store, mock_model)
    assert store.db["tool_instances"].count == 2

    report = reconcile_tool_instances(store.db)

    assert report.merged == {old_id: new_id}
    assert report.deleted == [old_id]
    assert report.repointed == {
        "turn_tools": 1,
        "tool_instantiations": 1,
        "tool_results": 0,
    }
    assert store.db["tool_instances"].count == 1
    turn_tool = next(iter(store.db["turn_tools"].rows))
    assert turn_tool["instance_id"] == new_id
    instantiation = next(iter(store.db["tool_instantiations"].rows))
    assert instantiation["instance_id"] == new_id


def test_distinct_configs_remain_separate(store, mock_model):
    _log_box_turn(store, mock_model)
    # A second turn in another thread with a different config.
    mock_model.enqueue(["ok"])
    response = mock_model.prompt(
        "other",
        tools=[Box(a=3, b=4)],
        tool_results=[
            llm.ToolResult(
                name="Box_run",
                output="y",
                tool_call_id="c2",
                instance=Box(a=3, b=4),
            )
        ],
    )
    response.text()
    response.log_to_db(store.db)

    reconcile_tool_instances(store.db)
    assert store.db["tool_instances"].count == 2


# ---- historical snapshots ----------------------------------------------


def test_rebound_rows_keep_the_arguments_used_then(store, mock_model):
    _, _, new_id, _ = _make_split(store, mock_model)

    reconcile_tool_instances(store.db)

    turn_tool = next(iter(store.db["turn_tools"].rows))
    assert turn_tool["used_arguments"] == OLD_ARGUMENTS
    instantiation = next(iter(store.db["tool_instantiations"].rows))
    assert instantiation["used_arguments"] == OLD_ARGUMENTS
    # The surviving row itself keeps the new writing.
    instance = store.db["tool_instances"].get(new_id)
    assert instance["arguments"] == NEW_ARGUMENTS


def test_history_display_shows_the_then_config(store, mock_model):
    _make_split(store, mock_model)
    reconcile_tool_instances(store.db)

    row = merged_log_rows(store)[0]
    extras = log_row_extras(store, row)
    assert extras["tools"][0]["instance"] == {
        "name": "Box",
        "arguments": OLD_ARGUMENTS,
    }
    assert extras["tool_results"][0]["instance"] == {
        "name": "Box",
        "plugin": None,
        "arguments": OLD_ARGUMENTS,
    }


def test_chained_merges_keep_the_original_snapshot(store, mock_model):
    _, _, y_id, _ = _make_split(store, mock_model)
    # First merge: old writing X -> current writing Y.
    report1 = reconcile_tool_instances(store.db, current=[Box(a=1, b=2)])
    assert y_id not in report1.deleted

    # A newer writing Z: compact text, parse-equivalent, written by a
    # new turn.
    z_arguments = '{"a":1,"b":2}'
    z_id = (
        store.db["tool_instances"]
        .insert({"name": "Box", "plugin": None, "arguments": z_arguments})
        .last_pk
    )

    # Y now merges away into the current row Z (exact-literal match).
    # The X-era reference must keep X's snapshot, not have it
    # overwritten with Y's during the second hop.
    report2 = reconcile_tool_instances(store.db, current=[("Box", None, z_arguments)])
    assert report2.merged == {y_id: z_id}
    instantiation = next(iter(store.db["tool_instantiations"].rows))
    assert instantiation["instance_id"] == z_id
    assert instantiation["used_arguments"] == OLD_ARGUMENTS
    turn_tool = next(iter(store.db["turn_tools"].rows))
    assert turn_tool["used_arguments"] == OLD_ARGUMENTS


# ---- current configuration ---------------------------------------------


def test_current_config_instance_wins(store, mock_model):
    # Old writing has the lower id and the reference; new writing has
    # none. Without a current config the oldest row wins...
    turn_id = _log_box_turn(store, mock_model)
    tool_id = next(iter(store.db["turn_tools"].rows))["tool_id"]
    old_id = next(iter(store.db["tool_instances"].rows))["id"]
    # Retag the existing row as old writing, add an equivalent new row.
    store.db["tool_instances"].update(old_id, {"arguments": OLD_ARGUMENTS})
    new_id = (
        store.db["tool_instances"]
        .insert({"name": "Box", "plugin": None, "arguments": NEW_ARGUMENTS})
        .last_pk
    )
    store.db["turn_tools"].update((turn_id, tool_id), {"instance_id": old_id})
    store.db["tool_instantiations"].update((turn_id, "c1"), {"instance_id": old_id})

    report = reconcile_tool_instances(store.db, current=[("Box", None, NEW_ARGUMENTS)])
    assert report.merged == {old_id: new_id}
    assert next(iter(store.db["tool_instances"].rows))["id"] == new_id


def test_current_config_without_a_row_is_inserted_once(store):
    assert store.db["tool_instances"].count == 0
    report = reconcile_tool_instances(store.db, current=[("Box", None, NEW_ARGUMENTS)])
    (new_id,) = report.inserted
    assert store.db["tool_instances"].get(new_id)["arguments"] == NEW_ARGUMENTS

    # Re-running with the same current configuration inserts nothing.
    report2 = reconcile_tool_instances(store.db, current=[("Box", None, NEW_ARGUMENTS)])
    assert report2.inserted == []
    assert not report2.changed


def test_toolbox_objects_accepted_as_current(store):
    report = reconcile_tool_instances(store.db, current=[Box(a=1, b=2)])
    (new_id,) = report.inserted
    row = store.db["tool_instances"].get(new_id)
    assert row["name"] == "Box"
    assert json.loads(row["arguments"]) == {"a": 1, "b": 2}


# ---- garbage collection ------------------------------------------------


def test_unreferenced_instances_are_garbage_collected(store):
    orphan_a = (
        store.db["tool_instances"]
        .insert({"name": "A", "plugin": None, "arguments": '{"x": 1}'})
        .last_pk
    )
    # Two equivalent orphans: one is merged away, the survivor is then
    # garbage collected as well.
    b1 = (
        store.db["tool_instances"]
        .insert({"name": "B", "plugin": None, "arguments": '{"y": 1}'})
        .last_pk
    )
    b2 = (
        store.db["tool_instances"]
        .insert({"name": "B", "plugin": None, "arguments": '{"y":1}'})
        .last_pk
    )

    report = reconcile_tool_instances(store.db)
    assert set(report.garbage_collected) == {orphan_a, b1}
    assert report.deleted == [b2]
    assert store.db["tool_instances"].count == 0


def test_current_config_protects_against_gc(store):
    store.db["tool_instances"].insert(
        {"name": "A", "plugin": None, "arguments": '{"x": 1}'}
    )
    report = reconcile_tool_instances(store.db, current=[("A", None, '{"x": 1}')])
    assert not report.changed
    assert store.db["tool_instances"].count == 1


# ---- failures: atomic, no half changes ---------------------------------


def _dump(db):
    return {
        name: db.execute(f"select * from {name} order by 1").fetchall()
        for name in ("tool_instances", "turn_tools", "tool_instantiations")
    }


def test_parse_failure_in_stored_row_changes_nothing(store, mock_model):
    _make_split(store, mock_model)
    bad_id = (
        store.db["tool_instances"]
        .insert({"name": "C", "plugin": None, "arguments": "not json"})
        .last_pk
    )
    before = _dump(store.db)

    with pytest.raises(ToolInstanceConfigError) as ex:
        reconcile_tool_instances(store.db)
    assert str(bad_id) in str(ex.value)
    assert _dump(store.db) == before


def test_parse_failure_in_current_config_changes_nothing(store, mock_model):
    _make_split(store, mock_model)
    before = _dump(store.db)
    with pytest.raises(ToolInstanceConfigError):
        reconcile_tool_instances(store.db, current=[("Box", None, "not json")])
    assert _dump(store.db) == before


def test_snapshot_conflict_changes_nothing(store, mock_model):
    turn_id, _, _, _ = _make_split(store, mock_model)
    # A stale snapshot that disagrees with the instance being merged.
    store.db["tool_instantiations"].update(
        (turn_id, "c1"), {"used_arguments": '{"different": 9}'}
    )
    before = _dump(store.db)
    with pytest.raises(ReconcileConflictError):
        reconcile_tool_instances(store.db)
    assert _dump(store.db) == before


def test_dangling_reference_is_a_conflict(store, mock_model):
    _, _, _, old_id = _make_split(store, mock_model)
    # Remove the old row without repointing: a dangling reference.
    store.db.conn.execute("pragma foreign_keys=OFF")
    store.db["tool_instances"].delete_where("id = ?", [old_id])
    with pytest.raises(ReconcileConflictError):
        reconcile_tool_instances(store.db)


def test_failure_during_apply_rolls_back_and_rerun_converges(
    store, mock_model, monkeypatch
):
    _, _, new_id, old_id = _make_split(store, mock_model)
    before = _dump(store.db)

    original_execute = store.db.execute

    def failing_execute(sql, parameters=None, *args, **kwargs):
        if sql.lstrip().startswith("delete"):
            raise RuntimeError("simulated interruption")
        if parameters is None:
            return original_execute(sql, *args, **kwargs)
        return original_execute(sql, parameters, *args, **kwargs)

    monkeypatch.setattr(store.db, "execute", failing_execute)
    with pytest.raises(RuntimeError):
        reconcile_tool_instances(store.db)
    monkeypatch.undo()

    # The transaction rolled back: nothing half changed.
    assert _dump(store.db) == before
    # Re-running converges.
    report = reconcile_tool_instances(store.db)
    assert report.merged == {old_id: new_id}
    assert store.db["tool_instances"].count == 1
    assert not reconcile_tool_instances(store.db).changed


# ---- legacy tool_results -----------------------------------------------


def test_legacy_tool_results_repointed_and_snapshotted(store):
    db = store.db
    # Minimal legacy shape: a response, a tool def, a tool_result.
    db["responses"].insert({"id": "resp1", "model": "mock"})
    tool_id = (
        db["tools"]
        .insert(
            {
                "hash": "h1",
                "name": "Box_run",
                "description": None,
                "input_schema": "{}",
                "plugin": None,
            }
        )
        .last_pk
    )
    new_id = (
        db["tool_instances"]
        .insert({"name": "Box", "plugin": None, "arguments": NEW_ARGUMENTS})
        .last_pk
    )
    old_id = (
        db["tool_instances"]
        .insert({"name": "Box", "plugin": None, "arguments": OLD_ARGUMENTS})
        .last_pk
    )
    result_id = (
        db["tool_results"]
        .insert(
            {
                "response_id": "resp1",
                "tool_id": tool_id,
                "name": "Box_run",
                "output": "x",
                "tool_call_id": "legacy_c",
                "instance_id": old_id,
            }
        )
        .last_pk
    )

    report = reconcile_tool_instances(db)
    assert report.merged == {old_id: new_id}
    assert report.repointed["tool_results"] == 1
    row = db["tool_results"].get(result_id)
    assert row["instance_id"] == new_id
    assert row["used_arguments"] == OLD_ARGUMENTS


# ---- idempotency --------------------------------------------------------


def test_reconcile_is_idempotent(store, mock_model):
    _make_split(store, mock_model)
    first = reconcile_tool_instances(store.db)
    assert first.changed
    state_after_first = _dump(store.db)
    second = reconcile_tool_instances(store.db)
    assert not second.changed
    assert _dump(store.db) == state_after_first


# ---- CLI ----------------------------------------------------------------


def test_tools_reconcile_cli_end_to_end(tmp_path, mock_model, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    db = sqlite_utils.Database(tmp_path / "logs.db")
    store = LogStore(db)
    _make_split(store, mock_model)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["tools", "reconcile", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["merged"]) == 1

    # Second run reports nothing to do.
    result2 = runner.invoke(cli, ["tools", "reconcile"])
    assert result2.exit_code == 0
    assert "already reconciled" in result2.output


def test_tools_reconcile_cli_bad_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli, ["tools", "reconcile", "-T", "NoSuchTool"])
    assert result.exit_code == 1
    assert "NoSuchTool" in result.output
