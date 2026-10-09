#!/usr/bin/env python3
"""Build the supplied runner or a case-specific scientific environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="Case directory names, or 'all'.")
    parser.add_argument("--runner", action="store_true", help="Build the unified runner image.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Docker.")
    parser.add_argument("--network", default="host", help="Docker build network (default: host).")
    parser.add_argument("--build-arg", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    if not args.runner and not args.cases:
        parser.error("Select --runner, one or more case names, or all.")
    available = {p.parent.parent.name: p.parent for p in (ROOT / "case").glob("*/environment/Dockerfile")}
    names = sorted(available) if args.cases == ["all"] else args.cases
    unknown = sorted(set(names) - available.keys())
    if unknown:
        parser.error("Unknown case(s): " + ", ".join(unknown))
    builds = []
    if args.runner:
        builds.append(("fde-bench-runner:v1", ROOT / "env-runner"))
    builds.extend((f"fde-bench:{name}", available[name]) for name in names)
    for tag, context in builds:
        cmd = ["docker", "build", "--network", args.network, "-t", tag]
        for value in args.build_arg:
            cmd += ["--build-arg", value]
        cmd += ["-f", str(context / "Dockerfile"), str(context)]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
