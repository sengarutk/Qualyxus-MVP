"""Tell the system that a machine was repaired, or that the mould, material or process changed.

    python -m src.process.baseline reset --machine imm_1 --by asha --reason "new check ring"
    python -m src.process.baseline reset --machine imm_1 --process-change --by asha --reason "mould 12 fitted"

The healthy baseline then starts again from the next cycles. ``--process-change`` also starts
the golden cycle and the risk model again: use it when a good cycle now looks different.
"""
from __future__ import annotations

import argparse

from src.process.store import MAINTENANCE, PROCESS_CHANGE, ProcessStore


def main(argv=None) -> float:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = parser.add_subparsers(dest="cmd", required=True).add_parser("reset")
    p.add_argument("--machine", required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--process-change", action="store_true")
    p.add_argument("--db", default=None, help="process.db (default: process.db_path of the config)")
    args = parser.parse_args(argv)
    db = args.db
    if db is None:
        from src.config.config import Config

        db = Config.load().get("process", "db_path", default="data/events/process.db")
    store = ProcessStore(db)
    try:
        if args.machine not in store.machines():
            parser.error(f"no cycles from machine '{args.machine}' in {db} (known: {store.machines()})")
        ts = store.reset_baseline(args.machine, kind=PROCESS_CHANGE if args.process_change else MAINTENANCE,
                                  reason=args.reason, by=args.by)
    finally:
        store.close()
    print(f"Baseline of {args.machine} starts again now.")
    return ts


if __name__ == "__main__":
    main()
