"""Reconciliation for the shared ``tool_instances`` table.

A configured toolbox instance is stored once and referenced by id from
historical turns (``turn_tools``), individual tool calls
(``tool_instantiations``) and legacy rows (``tool_results``). The same
configuration can be written more than one way - constructor arguments
supplied in a different order, a JSON object versus key=value form - so
the table can end up with several rows that describe one real instance,
plus rows nothing references any more.

:func:`reconcile_tool_instances` fixes that:

* configurations that parse to the same JSON value are merged into one
  instance, with every reference rebound to the surviving row;
* instances nothing references and the current configuration does not
  use are deleted;
* every rebound reference keeps the arguments text its instance had at
  the time (the ``used_arguments`` snapshot column), so history can
  still show the configuration that was actually in use.

The whole change is planned in memory first - a configuration that
fails to parse, or an integrity conflict, aborts before anything is
written - and applied in one transaction, so the operation is atomic,
idempotent and converges if an interrupted run is retried.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .logs import canonical_json
from .migrations import migrate
from .models import ServerSideTool, Toolbox

__all__ = [
    "ReconcileConflictError",
    "ReconcileError",
    "ReconcileReport",
    "ToolInstanceConfigError",
    "reconcile_tool_instances",
]


class ReconcileError(Exception):
    """Base class for reconciliation failures.

    Raised during planning, before any data is written.
    """


class ToolInstanceConfigError(ReconcileError):
    """A stored or current configuration could not be parsed as JSON."""


class ReconcileConflictError(ReconcileError):
    """References and snapshots disagree in a way that has no safe merge."""


# Reference tables that point at tool_instances, with the columns that
# key a row within each table.
_REF_TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("turn_tools", ("turn_id", "tool_id")),
    ("tool_instantiations", ("turn_id", "tool_call_id")),
    ("tool_results", ("id",)),
]


@dataclass
class ReconcileReport:
    """What a reconciliation changed. Empty when nothing needed doing."""

    # Merged-away instance id -> surviving instance id.
    merged: dict[int, int] = field(default_factory=dict)
    # Instance ids deleted because they were merged into a survivor.
    deleted: list[int] = field(default_factory=list)
    # Instance ids deleted because nothing referenced or configured them.
    garbage_collected: list[int] = field(default_factory=list)
    # Instance ids created for current-config entries that had no row.
    inserted: list[int] = field(default_factory=list)
    # Reference table name -> number of rows rebound.
    repointed: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(
            self.merged
            or self.garbage_collected
            or self.inserted
            or any(self.repointed.values())
        )


def _current_entry(item: Any) -> tuple[str, str | None, str]:
    """Normalize one current-configuration item to (name, plugin, args)."""
    if isinstance(item, (Toolbox, ServerSideTool)):
        return (
            item.__class__.__name__,
            getattr(item, "plugin", None),
            json.dumps(getattr(item, "_config", None) or {}),
        )
    if isinstance(item, dict):
        return item["name"], item.get("plugin"), item["arguments"]
    if isinstance(item, (tuple, list)):
        name, plugin, arguments = item
        return name, plugin, arguments
    raise TypeError(
        "Current configuration items must be Toolbox/ServerSideTool instances "
        "or (name, plugin, arguments) tuples/dicts"
    )


def _canonical(arguments: str | None) -> str:
    """Canonical JSON text for an arguments value.

    Raises ``ToolInstanceConfigError`` when the text is not valid JSON.
    """
    if arguments is None:
        value: Any = None
    else:
        try:
            value = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as ex:
            raise ToolInstanceConfigError(
                f"Configuration is not valid JSON: {arguments!r}"
            ) from ex
    return canonical_json(value)


@dataclass
class _Instance:
    id: int
    plugin: str | None
    name: str
    arguments: str | None
    canonical: str


@dataclass
class _Refpoint:
    key: tuple
    instance_id: int
    used_arguments: str | None


@dataclass
class _Plan:
    survivor_of: dict[int, int]
    deleted: list[int]
    garbage: list[int]
    inserts: list[tuple[str, str | None, str]]
    refpoints: dict[str, list[_Refpoint]]


def reconcile_tool_instances(db, current=None) -> ReconcileReport:
    """Reconcile stored tool instances against each other and current config.

    Parameters
    ----------
    db : sqlite_utils.Database
        Database containing the ``tool_instances`` table. Migrations are
        applied first, so a fresh database works.
    current : optional iterable
        The tool instances the current configuration uses. Each item is
        either a :class:`llm.Toolbox` / :class:`llm.ServerSideTool`
        instance or a ``(name, plugin, arguments)`` tuple/dict where
        ``arguments`` is a JSON string. These rows are protected from
        garbage collection and win when an equivalent group is merged.
        When omitted, only stored rows are merged and unreferenced rows
        removed.

    Returns
    -------
    ReconcileReport
        The merge, deletion, insertion and rebind operations performed.

    Raises
    ------
    ToolInstanceConfigError
        A stored or current configuration cannot be parsed as JSON.
    ReconcileConflictError
        References or snapshots disagree, so no safe merge exists.

    Nothing is written when either error is raised.
    """
    migrate(db)
    current_entries = [_current_entry(item) for item in (current or [])]

    instances: list[_Instance] = []
    for row in db["tool_instances"].rows:
        arguments = row["arguments"]
        try:
            canonical = _canonical(arguments)
        except ToolInstanceConfigError as ex:
            raise ToolInstanceConfigError(
                f"tool_instances id={row['id']} ({row['name']}): {ex}"
            ) from ex
        instances.append(
            _Instance(
                id=row["id"],
                plugin=row["plugin"],
                name=row["name"],
                arguments=arguments,
                canonical=canonical,
            )
        )
    by_id = {inst.id: inst for inst in instances}

    # Group stored instances by equivalent configuration.
    groups: dict[tuple, list[int]] = {}
    for inst in instances:
        groups.setdefault((inst.plugin, inst.name, inst.canonical), []).append(inst.id)

    # Current configuration: validate, map to groups, queue inserts.
    current_keys: set[tuple] = set()
    # The literal arguments text each current configuration uses, per
    # group key, so a row carrying that exact writing wins the group.
    current_literals: dict[tuple, set[str | None]] = {}
    inserts: dict[tuple, tuple[str, str | None, str]] = {}
    for name, plugin, arguments in current_entries:
        try:
            key = (plugin, name, _canonical(arguments))
        except ToolInstanceConfigError as ex:
            raise ToolInstanceConfigError(
                f"current configuration for {name}: {ex}"
            ) from ex
        current_keys.add(key)
        current_literals.setdefault(key, set()).add(arguments)
        if key not in groups:
            inserts.setdefault(key, (name, plugin, arguments))

    # Deterministic survivors. A row carrying the current configuration's
    # exact writing wins (lowest id on ties), then any row in a current
    # group, then the oldest row. Determinism is what makes reruns
    # converge.
    survivor_of: dict[int, int] = {}
    current_ids: set[int] = set()
    for key, ids in groups.items():
        if key in current_keys:
            current_ids.update(ids)
            exact = [i for i in ids if by_id[i].arguments in current_literals[key]]
            survivor = min(exact) if exact else min(ids)
        else:
            survivor = min(ids)
        for i in ids:
            if i != survivor:
                survivor_of[i] = survivor

    # Read every reference and plan rebinds + snapshots.
    referenced: set[int] = set()
    refpoints: dict[str, list[_Refpoint]] = {}
    for table, key_cols in _REF_TABLES:
        points: list[_Refpoint] = []
        for row in db[table].rows:
            iid = row["instance_id"]
            if iid is None:
                continue
            if iid not in by_id:
                raise ReconcileConflictError(
                    f"{table} row {tuple(row[c] for c in key_cols)} references "
                    f"missing tool_instances id={iid}"
                )
            referenced.add(iid)
            new_id = survivor_of.get(iid, iid)
            if new_id == iid:
                continue
            snapshot = row["used_arguments"]
            if snapshot is not None:
                # An earlier rebind left a snapshot: it must describe
                # the same configuration as the group now being merged,
                # else history would silently point at the wrong config.
                if _canonical(snapshot) != by_id[iid].canonical:
                    raise ReconcileConflictError(
                        f"{table} row {tuple(row[c] for c in key_cols)} records "
                        f"arguments {snapshot!r} that do not match instance id={iid}"
                    )
            else:
                snapshot = by_id[iid].arguments
            points.append(
                _Refpoint(
                    key=tuple(row[c] for c in key_cols),
                    instance_id=new_id,
                    used_arguments=snapshot,
                )
            )
        refpoints[table] = points

    # Unreferenced, non-current instances. Losers merge away regardless;
    # a survivor stays when its group is current or gains any reference
    # after rebinding (a reference to one of its losers counts), and is
    # garbage otherwise - covering a whole group nothing references.
    referenced_survivors = {survivor_of.get(i, i) for i in referenced}
    garbage = sorted(
        i
        for i in by_id
        if i not in survivor_of
        and i not in current_ids
        and i not in referenced_survivors
    )

    plan = _Plan(
        survivor_of=survivor_of,
        deleted=sorted(survivor_of),
        garbage=garbage,
        inserts=list(inserts.values()),
        refpoints=refpoints,
    )
    return _apply(db, plan)


def _apply(db, plan: _Plan) -> ReconcileReport:
    """Apply the plan in one transaction or leave everything untouched."""
    repointed: dict[str, int] = {}
    # db.atomic() (not `with db.conn:`): it issues BEGIN itself, so the
    # nested atomic() blocks the table helpers use become savepoints
    # inside this one transaction rather than separate commits.
    with db.atomic():
        inserted_ids: list[int] = []
        for name, plugin, arguments in plan.inserts:
            inserted_ids.append(
                db["tool_instances"]
                .insert({"name": name, "plugin": plugin, "arguments": arguments})
                .last_pk
            )

        for table, key_cols in _REF_TABLES:
            count = 0
            for point in plan.refpoints[table]:
                db[table].update(
                    point.key,
                    {
                        "instance_id": point.instance_id,
                        "used_arguments": point.used_arguments,
                    },
                )
                count += 1
            repointed[table] = count

        all_deleted = sorted(plan.deleted + plan.garbage)
        if all_deleted:
            placeholders = ",".join("?" for _ in all_deleted)
            db.execute(
                f"delete from tool_instances where id in ({placeholders})",
                all_deleted,
            )

    return ReconcileReport(
        merged=dict(plan.survivor_of),
        deleted=plan.deleted,
        garbage_collected=plan.garbage,
        inserted=inserted_ids,
        repointed=repointed,
    )
