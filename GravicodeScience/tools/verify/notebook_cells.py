#!/usr/bin/env python3
"""Compile-checks every code cell of every notebook against the built libraries.

Why this exists: a notebook is JSON, so it validates whether or not the C# inside it compiles.
All six notebooks once called `plot.GetImageHtml(...)`, which ScottPlot 5.1.59 marks
`[Obsolete(error: true)]` — every chart cell would have failed at run time, and nothing in the
build or the test suite could see it.

The check concatenates every code cell of one notebook into a single file and compiles it. That is
*stricter* than the notebook: .NET Interactive lets a variable be redeclared across cells, and this
does not. The stricter reading is deliberate — it catches a name being quietly reused — but it
means a repeated name has to be renamed rather than shadowed.

Usage:
    python tools/verify/notebook_cells.py [notebooks/GraviNum.Notebook.ipynb ...]

With no arguments it checks every notebook in notebooks/. Exits non-zero on the first failure.
"""

import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LIBRARIES = ["GraviNum", "GraviFrame", "GraviLearn", "GraviText", "GraviGraph", "GraviProb"]

# CS0219 unused variable, CS8321 unused local function, CS0168 declared and never used: all
# normal in a notebook, where a cell exists to show a value rather than to consume one.
PROJECT = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>disable</Nullable>
    <ImplicitUsings>true</ImplicitUsings>
    <NoWarn>CS0219;CS8321;CS0168;CS1998</NoWarn>
    <ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="ScottPlot" Version="5.1.59" />
{references}
  </ItemGroup>
</Project>
"""


def flatten(path):
    """Returns the notebook's code cells as one C# file."""
    notebook = json.load(io.open(path, encoding="utf-8"))
    body, usings = [], []

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            text = line.rstrip("\n")
            # `#r` and `#!` are .NET Interactive directives, not C#.
            if text.startswith("#r ") or text.startswith("#!"):
                continue
            # Usings float to the top: a notebook may declare one halfway down.
            if text.startswith("using ") and text.endswith(";") and "=" not in text:
                usings.append(text)
                continue
            body.append(text)
        body.append("")

    # A chart cell ends in a bare `plot.GetPngHtml(...)` with no semicolon. That is the cell's
    # return value, not a C# statement, so it is turned into one.
    lines = []
    for line in body:
        stripped = line.rstrip()
        if "GetPngHtml(" in stripped and stripped.endswith(")"):
            lines.append("_ = " + stripped + ";")
        else:
            lines.append(line)

    return "\n".join(sorted(set(usings))) + "\n\n" + "\n".join(lines) + "\n"


def check(path, workspace):
    source = flatten(path)
    io.open(os.path.join(workspace, "Program.cs"), "w", encoding="utf-8", newline="\n").write(source)

    result = subprocess.run(
        ["dotnet", "build", os.path.join(workspace, "cells.csproj"), "--nologo", "-v", "q"],
        capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  OK      {os.path.basename(path)}  ({source.count(chr(10))} lines)")
        return True

    print(f"  FAILED  {os.path.basename(path)}")
    for line in (result.stdout + result.stderr).splitlines():
        if ": error" in line:
            print("            " + line.strip())
    return False


def main():
    targets = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))
    if not targets:
        print("No notebooks found.")
        return 1

    workspace = tempfile.mkdtemp(prefix="gravi-notebook-")
    try:
        references = "\n".join(
            f'    <ProjectReference Include="{os.path.join(ROOT, "src", name, name + ".csproj")}" />'
            for name in LIBRARIES)
        io.open(os.path.join(workspace, "cells.csproj"), "w", encoding="utf-8", newline="\n").write(
            PROJECT.format(references=references))

        print(f"Compile-checking {len(targets)} notebook(s):")
        failures = [path for path in targets if not check(path, workspace)]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} notebook(s) do not compile.")
        return 1

    print("\nEvery notebook cell compiles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
