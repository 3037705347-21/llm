import hashlib
import json
import os
import sqlite3
import textwrap
import types
from importlib.metadata import version
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from llm.cli import cli
from llm.migrations import migrate
from llm.utils import (
    FragmentAliasManifestError,
    ensure_fragment,
    reconcile_fragment_aliases,
)


def _hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_fragments_set_show_remove(user_path, tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    with open("fragment1.txt", "w") as f:
        f.write("Hello fragment 1")

    # llm fragments --aliases should return nothing
    assert runner.invoke(cli, ["fragments", "list", "--aliases"]).output == ""
    assert (
        runner.invoke(cli, ["fragments", "set", "f1", "fragment1.txt"]).exit_code == 0
    )
    result1 = runner.invoke(cli, ["fragments", "show", "f1"])
    assert result1.exit_code == 0
    assert result1.output == "Hello fragment 1\n"

    # Should be in the list now
    def get_list():
        result2 = runner.invoke(cli, ["fragments", "list"])
        assert result2.exit_code == 0
        return yaml.safe_load(result2.output)

    # And in llm fragments --aliases
    assert "f1" in runner.invoke(cli, ["fragments", "list", "--aliases"]).output

    loaded1 = get_list()
    assert set(loaded1[0].keys()) == {
        "aliases",
        "content",
        "datetime_utc",
        "source",
        "hash",
    }
    assert loaded1[0]["content"] == "Hello fragment 1"
    assert loaded1[0]["aliases"] == ["f1"]

    # Show should work against both alias and hash
    for key in ("f1", loaded1[0]["hash"]):
        result3 = runner.invoke(cli, ["fragments", "show", key])
        assert result3.exit_code == 0
        assert result3.output == "Hello fragment 1\n"

    # But not for an invalid alias
    result4 = runner.invoke(cli, ["fragments", "show", "badalias"])
    assert result4.exit_code == 1
    assert "Fragment 'badalias' not found" in result4.output

    # Remove that alias
    result5 = runner.invoke(cli, ["fragments", "remove", "f1"])
    assert result5.exit_code == 0
    # Should still be in list but no alias
    loaded2 = get_list()
    assert loaded2[0]["aliases"] == []
    assert loaded2[0]["content"] == "Hello fragment 1"

    # And --aliases list should be empty
    assert runner.invoke(cli, ["fragments", "list", "--aliases"]).output == ""


def test_fragments_list(db_factory, user_path):
    runner = CliRunner()
    db = db_factory(str(user_path / "logs.db"))
    with db.conn:
        migrate(db)
        db["fragments"].insert_all(
            [
                {
                    "id": 1,
                    "content": "1",
                    "datetime_utc": "2023-10-01T00:00:00Z",
                    "source": "file1.txt",
                    "hash": "hash1",
                },
                {
                    "id": 2,
                    "content": "2",
                    "datetime_utc": "2022-10-01T00:00:00Z",
                    "source": "file2.txt",
                    "hash": "hash2",
                },
                {
                    "id": 3,
                    "content": "3",
                    "datetime_utc": "2024-10-01T00:00:00Z",
                    "source": "file3.txt",
                    "hash": "hash3",
                },
            ]
        )
        db["fragment_aliases"].insert(
            {
                "alias": "f1",
                "fragment_id": 1,
            }
        )
    result = runner.invoke(cli, ["fragments", "list"])
    assert result.exit_code == 0
    assert result.output.strip() == (textwrap.dedent("""
            - hash: hash2
              aliases: []
              datetime_utc: '2022-10-01T00:00:00Z'
              source: file2.txt
              content: '2'
            - hash: hash1
              aliases:
              - f1
              datetime_utc: '2023-10-01T00:00:00Z'
              source: file1.txt
              content: '1'
            - hash: hash3
              aliases: []
              datetime_utc: '2024-10-01T00:00:00Z'
              source: file3.txt
              content: '3'
            """).strip())


def test_fragment_absolute_path(user_path, tmp_path):
    path = tmp_path / "fragment.txt"
    path.write_text("Hello from an absolute path")

    result = CliRunner().invoke(
        cli, ["prompt", "-m", "echo", "-f", str(path)], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "prompt": "Hello from an absolute path",
        "system": "",
        "attachments": [],
        "stream": True,
        "previous": [],
    }


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "X"})
def test_fragment_url_user_agent(mocked_openai_chat, user_path):
    mocked_openai_chat.add_response(
        url="https://example.com/fragment.txt",
        text="Hello from URL",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "prompt",
            "-m",
            "gpt-4o-mini",
            "-f",
            "https://example.com/fragment.txt",
        ],
    )
    assert result.exit_code == 0

    # Verify the User-Agent header was sent for the fragment URL request
    requests = mocked_openai_chat.get_requests()
    fragment_request = next(r for r in requests if "example.com" in str(r.url))
    llm_version = version("llm")
    expected_user_agent = f"llm/{llm_version} (https://llm.datasette.io/)"
    assert fragment_request.headers["User-Agent"] == expected_user_agent


# --- llm fragments sync --------------------------------------------------


@pytest.fixture
def synced_db(db_factory, user_path):
    """A migrated logs.db plus helpers that reopen it per call."""
    path = str(user_path / "logs.db")
    db = db_factory(path)
    with db.conn:
        migrate(db)
    db.close()
    ids = {}

    def add_fragment(content):
        d = db_factory(path)
        with d.conn:
            fragment_id = ensure_fragment(d, content)
        d.close()
        ids[content] = fragment_id
        return fragment_id

    def add_manual_alias(alias, content):
        d = db_factory(path)
        with d.conn:
            d["fragment_aliases"].insert({"alias": alias, "fragment_id": ids[content]})
        d.close()

    return types.SimpleNamespace(
        path=path,
        ids=ids,
        add_fragment=add_fragment,
        add_manual_alias=add_manual_alias,
    )


def _alias_rows(user_path):
    """alias -> (fragment_id, managed_by), read from a fresh connection."""
    conn = sqlite3.connect(str(user_path / "logs.db"))
    try:
        rows = conn.execute(
            "select alias, fragment_id, managed_by from fragment_aliases"
        ).fetchall()
    finally:
        conn.close()
    return {alias: (fragment_id, owner) for alias, fragment_id, owner in rows}


def _fragment_contents(user_path):
    conn = sqlite3.connect(str(user_path / "logs.db"))
    try:
        rows = conn.execute("select hash, content from fragments").fetchall()
    finally:
        conn.close()
    return dict(rows)


def _write_manifest(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_fragments_sync_creates_managed_aliases(user_path, tmp_path, synced_db):
    synced_db.add_fragment("content one")
    synced_db.add_fragment("content two")
    synced_db.add_manual_alias("manual", "content one")
    h1, h2 = _hash("content one"), _hash("content two")
    manifest = _write_manifest(tmp_path, "aliases.json", {"m1": h1, "m2": h2})

    runner = CliRunner()
    result = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 0, result.output
    assert "Added alias: m1" in result.output
    assert "Added alias: m2" in result.output

    owner = str(manifest.resolve())
    rows = _alias_rows(user_path)
    assert rows["m1"] == (synced_db.ids["content one"], owner)
    assert rows["m2"] == (synced_db.ids["content two"], owner)
    # Manual alias is untouched and remains unowned
    assert rows["manual"] == (synced_db.ids["content one"], None)

    # Managed aliases resolve to the right content
    for alias, content in (("m1", "content one"), ("m2", "content two")):
        shown = runner.invoke(cli, ["fragments", "show", alias])
        assert shown.exit_code == 0
        assert shown.output == content + "\n"


def test_fragments_sync_repoints_by_content_hash(user_path, tmp_path, synced_db):
    synced_db.add_fragment("old content")
    synced_db.add_fragment("new content")
    old_hash, new_hash = _hash("old content"), _hash("new content")
    manifest = _write_manifest(tmp_path, "aliases.json", {"docs": old_hash})
    runner = CliRunner()

    assert runner.invoke(cli, ["fragments", "sync", str(manifest)]).exit_code == 0
    assert runner.invoke(cli, ["fragments", "show", "docs"]).output == "old content\n"

    # Manifest now points the same alias at different content
    manifest.write_text(json.dumps({"docs": new_hash}))
    result = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 0
    assert "Updated alias: docs" in result.output
    assert runner.invoke(cli, ["fragments", "show", "docs"]).output == "new content\n"

    # The old fragment content was NOT deleted: still readable by hash
    old = runner.invoke(cli, ["fragments", "show", old_hash])
    assert old.exit_code == 0
    assert old.output == "old content\n"
    contents = _fragment_contents(user_path)
    assert old_hash in contents
    assert new_hash in contents


def test_fragments_sync_removes_deleted_managed_aliases(user_path, tmp_path, synced_db):
    synced_db.add_fragment("one")
    synced_db.add_fragment("two")
    synced_db.add_fragment("three")
    synced_db.add_manual_alias("manual", "three")
    h1, h2 = _hash("one"), _hash("two")
    manifest = _write_manifest(tmp_path, "aliases.json", {"m1": h1, "m2": h2})
    runner = CliRunner()
    assert runner.invoke(cli, ["fragments", "sync", str(manifest)]).exit_code == 0

    # m2 disappears from the manifest; m1 and the manual alias stay
    manifest.write_text(json.dumps({"m1": h1}))
    result = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 0
    assert "Removed alias: m2" in result.output

    rows = _alias_rows(user_path)
    assert set(rows) == {"m1", "manual"}
    assert rows["m1"][1] == str(manifest.resolve())
    assert rows["manual"][1] is None
    # No fragment content was removed
    assert set(_fragment_contents(user_path)) == {h1, h2, _hash("three")}


def test_fragments_sync_is_idempotent(user_path, tmp_path, synced_db):
    synced_db.add_fragment("content one")
    h1 = _hash("content one")
    manifest = _write_manifest(tmp_path, "aliases.json", {"m1": h1})
    runner = CliRunner()

    first = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert first.exit_code == 0
    assert "Added alias: m1" in first.output
    rows_after_first = _alias_rows(user_path)

    second = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert second.exit_code == 0
    assert second.output == "Fragment aliases already in sync\n"
    assert _alias_rows(user_path) == rows_after_first

    # Empty manifest removes managed aliases, and converges when rerun
    manifest.write_text("{}")
    third = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert third.exit_code == 0
    assert "Removed alias: m1" in third.output
    fourth = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert fourth.exit_code == 0
    assert fourth.output == "Fragment aliases already in sync\n"
    assert _alias_rows(user_path) == {}


def test_fragments_sync_missing_hash_is_atomic(user_path, tmp_path, synced_db):
    synced_db.add_fragment("content one")
    synced_db.add_manual_alias("manual", "content one")
    h1 = _hash("content one")
    missing_hash = "a" * 64
    manifest = _write_manifest(
        tmp_path, "aliases.json", {"good": h1, "bad": missing_hash}
    )
    before = _alias_rows(user_path)

    result = CliRunner().invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 1
    assert missing_hash in result.output
    assert "not stored in this database" in result.output
    # Nothing was written: no partial "good" alias, manual alias intact
    assert _alias_rows(user_path) == before
    assert "good" not in _alias_rows(user_path)


@pytest.mark.parametrize(
    "content,fragment",
    (
        ("{not valid json", None),
        ('["aliases", "must", "be", "an", "object"]', None),
        (None, {"bad alias!": "hash"}),
        (None, {"ok": "not-a-hash"}),
        (None, {"ok": 123}),
        ("", None),
    ),
)
def test_fragments_sync_invalid_manifest(
    user_path, tmp_path, synced_db, content, fragment
):
    synced_db.add_fragment("content one")
    h1 = _hash("content one")
    fragment = fragment or {}
    fragment = {k: (h1 if v == "hash" else v) for k, v in fragment.items()}
    if content is None:
        content = json.dumps(fragment)
    manifest = tmp_path / "aliases.json"
    manifest.write_text(content)

    result = CliRunner().invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 1
    assert _alias_rows(user_path) == {}


def test_fragments_sync_multiple_managers_are_independent(
    user_path, tmp_path, synced_db
):
    synced_db.add_fragment("content one")
    synced_db.add_fragment("content two")
    synced_db.add_manual_alias("manual", "content one")
    h1, h2 = _hash("content one"), _hash("content two")
    manifest_a = _write_manifest(tmp_path, "a.json", {"a1": h1, "a2": h1})
    manifest_b = _write_manifest(tmp_path, "b.json", {"b1": h2})
    runner = CliRunner()

    assert runner.invoke(cli, ["fragments", "sync", str(manifest_a)]).exit_code == 0
    assert runner.invoke(cli, ["fragments", "sync", str(manifest_b)]).exit_code == 0

    # Syncing B must not remove aliases owned by A
    rows = _alias_rows(user_path)
    assert set(rows) == {"a1", "a2", "b1", "manual"}

    # Emptying A removes only A's aliases
    manifest_a.write_text("{}")
    result = runner.invoke(cli, ["fragments", "sync", str(manifest_a)])
    assert result.exit_code == 0
    assert set(result.output.split()) >= {"Removed", "alias:", "a1", "a2"}
    rows = _alias_rows(user_path)
    assert set(rows) == {"b1", "manual"}
    assert rows["b1"][1] == str(manifest_b.resolve())
    assert rows["manual"][1] is None


def test_fragments_sync_adopts_manual_alias(user_path, tmp_path, synced_db):
    synced_db.add_fragment("content one")
    synced_db.add_manual_alias("docs", "content one")
    h1 = _hash("content one")
    manifest = _write_manifest(tmp_path, "aliases.json", {"docs": h1})
    runner = CliRunner()

    result = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 0
    assert "Updated alias: docs" in result.output
    rows = _alias_rows(user_path)
    assert rows["docs"][1] == str(manifest.resolve())

    # Now that the manifest owns it, dropping it from the manifest removes it
    manifest.write_text("{}")
    runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert "docs" not in _alias_rows(user_path)


def test_fragments_sync_keeps_history_readable(
    user_path, tmp_path, synced_db, db_factory
):
    fragment_id_old = synced_db.add_fragment("old content")
    synced_db.add_fragment("new content")
    # A historical conversation referenced the old fragment
    db = db_factory(str(user_path / "logs.db"))
    with db.conn:
        db["prompt_fragments"].insert(
            {
                "response_id": "response-from-history",
                "fragment_id": fragment_id_old,
                "order": 0,
            }
        )
    db.close()
    old_hash, new_hash = _hash("old content"), _hash("new content")

    # v1: docs points at the historically referenced content
    manifest = _write_manifest(
        tmp_path, "aliases.json", {"docs": old_hash, "temp": new_hash}
    )
    runner = CliRunner()
    assert runner.invoke(cli, ["fragments", "sync", str(manifest)]).exit_code == 0

    # v2: docs repoints elsewhere and temp alias is deleted entirely
    manifest.write_text(json.dumps({"docs": new_hash}))
    result = runner.invoke(cli, ["fragments", "sync", str(manifest)])
    assert result.exit_code == 0

    # The historical join still resolves the old content
    db = db_factory(str(user_path / "logs.db"))
    row = db.execute(
        """
        select fragments.hash, fragments.content
        from prompt_fragments
        join fragments on fragments.id = prompt_fragments.fragment_id
        where response_id = :response_id
        """,
        {"response_id": "response-from-history"},
    ).fetchone()
    db.close()
    assert row[0] == old_hash
    assert row[1] == "old content"


def test_reconcile_fragment_aliases_summary(db_factory):
    db = db_factory(memory=True)
    migrate(db)
    h1 = _hash("one")
    h2 = _hash("two")
    with db.conn:
        id1 = ensure_fragment(db, "one")
        id2 = ensure_fragment(db, "two")
        # Pre-existing aliases: a manual one, a stale managed one, one
        # that stays put, and one that gets repointed to other content
        db["fragment_aliases"].insert_all(
            [
                {"alias": "manual", "fragment_id": id1, "managed_by": None},
                {"alias": "stale", "fragment_id": id1, "managed_by": "manifest.json"},
                {"alias": "kept", "fragment_id": id1, "managed_by": "manifest.json"},
                {"alias": "moved", "fragment_id": id1, "managed_by": "manifest.json"},
            ]
        )

    summary = reconcile_fragment_aliases(
        db, {"added": h1, "kept": h1, "moved": h2, "manual": h1}, "manifest.json"
    )
    assert summary["added"] == ["added"]
    assert set(summary["updated"]) == {"moved", "manual"}
    assert summary["removed"] == ["stale"]
    assert summary["unchanged"] == ["kept"]

    rows = {
        r["alias"]: (r["fragment_id"], r["managed_by"])
        for r in db["fragment_aliases"].rows
    }
    assert rows["added"] == (id1, "manifest.json")
    assert rows["moved"] == (id2, "manifest.json")
    assert rows["manual"] == (id1, "manifest.json")
    assert rows["kept"] == (id1, "manifest.json")
    assert "stale" not in rows
    # Rerunning converges: nothing changes
    again = reconcile_fragment_aliases(
        db, {"added": h1, "kept": h1, "moved": h2, "manual": h1}, "manifest.json"
    )
    assert again == {
        "added": [],
        "updated": [],
        "removed": [],
        "unchanged": [
            "added",
            "kept",
            "moved",
            "manual",
        ],
    }


def test_reconcile_fragment_aliases_validates(db_factory):
    db = db_factory(memory=True)
    migrate(db)
    with db.conn:
        ensure_fragment(db, "one")
    h1 = _hash("one")
    with pytest.raises(FragmentAliasManifestError):
        reconcile_fragment_aliases(db, {"bad alias!": h1}, "manifest.json")
    with pytest.raises(FragmentAliasManifestError):
        reconcile_fragment_aliases(db, {"ok": "nope"}, "manifest.json")
    with pytest.raises(FragmentAliasManifestError):
        reconcile_fragment_aliases(db, {"ok": "b" * 64}, "manifest.json")
    with pytest.raises(FragmentAliasManifestError):
        reconcile_fragment_aliases(db, [("ok", h1)], "manifest.json")
    with pytest.raises(FragmentAliasManifestError):
        reconcile_fragment_aliases(db, {"ok": h1}, "")
    # Failed validation left no aliases behind
    assert list(db["fragment_aliases"].rows) == []
