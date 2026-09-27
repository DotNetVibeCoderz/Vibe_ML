"""
Records new public API in PublicAPI.Unshipped.txt (or --shipped: PublicAPI.Shipped.txt) from the RS0016
diagnostics of a build. Run it after intentionally adding public API:

    python tools/update_public_api.py            # whole solution -> Unshipped
    python tools/update_public_api.py --shipped  # during a release

Removed API (RS0017) is reported, never deleted automatically: removing public API is a breaking change.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"^(?P<file>.+?)\(\d+,\d+\): error RS0016: Symbol '(?P<symbol>.+)' is not part of the declared public API")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", action="store_true")
    args = ap.parse_args()
    out = subprocess.run(["dotnet", "build", str(ROOT / "MediaPipeNet.slnx"), "-c", "Release", "--no-incremental"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    added: dict[pathlib.Path, set[str]] = {}
    for line in out.splitlines():
        m = PATTERN.match(line.strip())
        if not m:
            if "RS0017" in line:
                print("removed API:", line.strip()[:200])
            continue
        project = pathlib.Path(m["file"]).parent
        while not any(project.glob("*.csproj")):
            project = project.parent
        added.setdefault(project, set()).add(m["symbol"])
    for project, symbols in added.items():
        target = project / ("PublicAPI.Shipped.txt" if args.shipped else "PublicAPI.Unshipped.txt")
        lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else ["#nullable enable"]
        body = sorted(set(l for l in lines if l and not l.startswith("#")) | symbols)
        target.write_text("\n".join(["#nullable enable", *body]) + "\n", encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: +{len(symbols)}")


if __name__ == "__main__":
    main()
