#!/usr/bin/env python3
"""Cross-checks GraviFrame's Arrow IPC support against pyarrow, in both directions.

The C# test suite can only verify that GraviFrame agrees with itself. For an interchange format
that is nearly worthless — the whole point is that *other* implementations can read the file. This
script is the part that actually establishes that, and it is kept out of the test suite because it
needs a Python toolchain the tests do not.

Run it after any change to ArrowIO.cs or FlatBuffers.cs:

    pip install pyarrow pandas
    dotnet run --project tools/verify/ArrowInterop -c Release
    python tools/verify/arrow_interop.py

The `dotnet run` step writes the files this reads and reads the ones this writes; see that project
for the C# half.
"""

import datetime as dt
import os
import subprocess
import sys
import tempfile

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(REPO, "tools", "verify", "ArrowInterop")

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}{('  — ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def run_csharp(*args):
    result = subprocess.run(
        ["dotnet", "run", "--project", HARNESS, "-c", "Release", "--", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"the C# harness failed on: {' '.join(args)}")
    return result.stdout


def main():
    work = tempfile.mkdtemp(prefix="arrow-interop-")

    # ---------------------------------------------------------------- C# -> pyarrow
    print("\n=== 1. GraviFrame writes, pyarrow reads ===")

    written = os.path.join(work, "from_csharp.arrow")
    run_csharp("write", written)

    with pa.memory_map(written, "r") as source:
        table = ipc.open_file(source).read_all()

    check("pyarrow opens the file", table.num_rows == 5, f"got {table.num_rows} rows")
    check("schema types are right",
          [str(f.type) for f in table.schema] == ["double", "string", "bool", "timestamp[us]"],
          str([str(f.type) for f in table.schema]))

    value = table.column("value").to_pylist()
    label = table.column("label").to_pylist()
    flag = table.column("flag").to_pylist()
    when = table.column("when").to_pylist()

    check("doubles survive", value == [1.5, -2.25, None, 1e300, 0.0], str(value))
    check("empty string is not null", label[1] == "" and label[2] is None, str(label[:3]))
    check("multi-byte text survives", "ünïcødé" in (label[3] or ""), str(label[3]))
    check("booleans survive", flag == [True, False, None, True, False], str(flag))
    check("timestamps survive",
          when[0] == dt.datetime(2024, 3, 1, 12, 30, 45) and when[2] is None, str(when[:3]))

    # ---------------------------------------------------------------- pyarrow -> C#
    print("\n=== 2. pyarrow writes, GraviFrame reads ===")

    incoming = os.path.join(work, "from_pyarrow.arrow")
    table = pa.table({
        "value": pa.array([1.5, -2.25, None, 1e300, 0.0], type=pa.float64()),
        "label": pa.array(["alpha", "", None, "ünïcødé 🐱", "last"], type=pa.string()),
        "flag": pa.array([True, False, None, True, False], type=pa.bool_()),
        "when": pa.array([dt.datetime(2024, 3, 1, 12, 30, 45), dt.datetime(1999, 12, 31, 23, 59, 59),
                          None, dt.datetime(1970, 1, 1), dt.datetime(2030, 6, 15)],
                         type=pa.timestamp("us")),
        "count": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
    })
    with pa.OSFile(incoming, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    output = run_csharp("read", incoming)
    check("GraviFrame reads it", "5 rows x 5 columns" in output, output.strip()[:200])
    check("int64 widens correctly", "count=1,2,3,4,5" in output.replace(" ", ""),
          [l for l in output.splitlines() if "count" in l])

    # ---------------------------------------------------------------- integer widths
    print("\n=== 3. Integer widths and signedness ===")

    for name, arrow_type, values in [
        ("int8", pa.int8(), [-128, 0, 127]),
        ("int16", pa.int16(), [-32768, 0, 32767]),
        ("int32", pa.int32(), [-2147483648, 0, 2147483647]),
        ("int64", pa.int64(), [-9007199254740992, 0, 9007199254740992]),
        ("uint8", pa.uint8(), [0, 128, 255]),
        ("uint16", pa.uint16(), [0, 32768, 65535]),
        ("uint32", pa.uint32(), [0, 2147483648, 4294967295]),
    ]:
        path = os.path.join(work, f"{name}.arrow")
        t = pa.table({"n": pa.array(values, type=arrow_type)})
        with pa.OSFile(path, "wb") as sink:
            with ipc.new_file(sink, t.schema) as writer:
                writer.write_table(t)

        out = run_csharp("values", path, "n")
        got = [float(x) for x in out.strip().split(",")]
        check(name, got == [float(v) for v in values], f"expected {values}, got {got}")

    # ---------------------------------------------------------------- timestamp units
    print("\n=== 4. Timestamp units ===")

    moment = dt.datetime(2024, 3, 1, 12, 30, 45)
    for unit in ["s", "ms", "us", "ns"]:
        path = os.path.join(work, f"ts_{unit}.arrow")
        t = pa.table({"when": pa.array([moment], type=pa.timestamp(unit))})
        with pa.OSFile(path, "wb") as sink:
            with ipc.new_file(sink, t.schema) as writer:
                writer.write_table(t)

        out = run_csharp("values", path, "when").strip()
        check(f"timestamp[{unit}]", out.startswith("2024-03-01T12:30:45"), out)

    # ---------------------------------------------------------------- round trip via pandas
    print("\n=== 5. Round trip through pandas ===")

    try:
        import pandas as pd

        path = os.path.join(work, "pandas.arrow")
        frame = pd.DataFrame({
            "ts": pd.to_datetime(["2024-03-01 12:30:45", "1970-01-01 00:00:00", None],
                                 format="ISO8601"),
            "i32": pd.array([100, -200, None], dtype="Int32"),
            "f": [1.25, None, 3.5],
        })
        t = pa.Table.from_pandas(frame, preserve_index=False)
        with pa.OSFile(path, "wb") as sink:
            with ipc.new_file(sink, t.schema) as writer:
                writer.write_table(t)

        out = run_csharp("read", path)
        check("pandas frame reads", "3 rows x 3 columns" in out, out.strip()[:200])
    except ImportError:
        print("  SKIP  pandas not installed")

    # ---------------------------------------------------------------- verdict
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        sys.exit(1)

    print("All Arrow interop checks passed.")


if __name__ == "__main__":
    main()
