"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations
from dotenv import load_dotenv
import argparse
import sys
import json
import datetime
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
load_dotenv()

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

client = MlflowClient()

# shared helpers


def _resolve_version(name, config_id):
    """
    Resolve the registered version whose config_id tag == config_id.
    0 -> error+exit 1; 2+ -> warning, use highest MLflow version. Used by
    BOTH set and rollback"
    """
    mvs = client.search_model_versions(
        f"name='{name}' AND tags.config_id='{config_id}'")
    if not mvs:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(mvs) > 1:
        versions = sorted(int(mv.version) for mv in mvs)
        chosen = max(versions)
        print(f"warning: multiple versions match config_id={config_id} "
              f"(MLflow versions {versions}); using latest ({chosen})")
        return next(mv for mv in mvs if int(mv.version) == chosen)
    return mvs[0]


def _current_config_id(name, alias):
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:               # unset alias is normal, not a crash
        return ""
    return mv.tags.get("config_id", "")


def _append_log(alias, frm, to, op):
    event = {
        "ts": datetime.datetime.now(datetime.timezone.utc)
        .isoformat().replace("+00:00", "Z"),   # spec sample shows 'Z'
        "alias": alias, "from": frm, "to": to, "op": op,  # from/to are config_ids
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _read_log():
    if not LOG_FILE.exists():             # first run: empty log, don't crash
        return []
    with LOG_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

    # subcommands


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    target = _resolve_version(args.name, args.config_id)
    frm = _current_config_id(args.name, args.alias)
    client.set_registered_model_alias(args.name, args.alias, target.version)
    _append_log(args.alias, frm, args.config_id, "set")
    print(f"{args.alias}: {frm if frm else '(unset)'} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except MlflowException:
        print(f"alias {args.alias} is not set")
        sys.exit(1)
    metrics = client.get_run(mv.run_id).data.metrics
    print(f"{args.name} @ {args.alias}")
    print(f" config_id: {mv.tags.get('config_id', '?')}")
    print(f" model: {mv.tags.get('model', '?')}")
    print(
        f" accuracy_overall: {metrics.get('accuracy_overall', float('nan')):.2f}")
    print(
        f"  verdict_rate_leaked: {metrics.get('verdict_rate_leaked', float('nan')):.2f}")
    print(
        f"  total_cost_usd: ${metrics.get('total_cost_usd', float('nan')):.2f}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    aliases = client.get_registered_model(args.name).aliases or {}
    if not aliases:
        print("no aliases set")
        return
    width = max(len(a) for a in aliases)
    for alias, version in sorted(aliases.items()):
        mv = client.get_model_version(args.name, version)
        print(f"{alias.ljust(width)} -> {mv.tags.get('config_id', '?')}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    log = _read_log()
    if not _current_config_id(args.name, args.alias) and \
            not any(e["alias"] == args.alias for e in log):
        print("nothing to roll back")
        return
    last = next((e for e in reversed(log) if e["alias"] == args.alias), None)
    if last is None:
        print(f"no promotion history for alias {args.alias}")
        return
    if last["op"] == "rollback":           # single-step by design
        print(f"{args.alias} was just rolled back; no further history to walk back to")
        return
    if last["op"] == "set" and not last["from"]:
        print(f"{args.alias} has no previous target (first promotion ever)")
        return
    prev = last["from"]
    # same multiplicity rule as set
    target = _resolve_version(args.name, prev)
    cur = _current_config_id(args.name, args.alias)
    client.set_registered_model_alias(args.name, args.alias, target.version)
    _append_log(args.alias, cur, prev, "rollback")
    print(f"{args.alias}: {cur} → {prev} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser(
        "show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser(
        "list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
