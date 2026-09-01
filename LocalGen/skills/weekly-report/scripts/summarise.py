#!/usr/bin/env python3
"""Summarise a weekly metrics CSV.

Reads a CSV with the columns: week, commits, prs, issues. Prints totals for the most recent week
alongside the one before it, as JSON, so the model can fill the template without doing arithmetic.

    python summarise.py metrics.csv
"""

import csv
import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: summarise.py <metrics.csv>"}))
        return 1

    path = sys.argv[1]

    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        print(json.dumps({"error": f"could not read {path}: {error}"}))
        return 1

    if not rows:
        print(json.dumps({"error": "the file contains no rows"}))
        return 1

    # Sorted by week so "most recent" holds regardless of the file's own order.
    rows.sort(key=lambda row: int(row.get("week", 0)))

    current = rows[-1]
    previous = rows[-2] if len(rows) > 1 else None

    def number(row, column):
        try:
            return int(row[column])
        except (KeyError, TypeError, ValueError):
            return 0

    result = {"week": number(current, "week")}

    for column in ("commits", "prs", "issues"):
        now = number(current, column)
        before = number(previous, column) if previous else None

        result[column] = now
        result[f"{column}_prev"] = before
        # None rather than 0 when there is no prior week: an absent comparison is not a flat one.
        result[f"{column}_delta"] = None if before is None else now - before

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
