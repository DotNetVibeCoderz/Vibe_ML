# Publishing and CI

Six packages ship from `src/`: `Gravicode.Science.GraviNum`, `.GraviFrame`, `.GraviLearn`,
`.GraviText`, `.GraviGraph` and `.GraviProb`. Everything else in the repository — samples,
benchmarks, notebooks, tests, ScienceAppGen — is marked `IsPackable=false` and never leaves.

## Where the metadata lives

All of it is in `Directory.Build.props`, in one property group that applies to any project under
`src/`. Putting it there rather than in six `.csproj` files is what keeps the packages from
drifting apart: there is one version, one licence, one repository URL.

```xml
<PackageProjectUrl>https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience</PackageProjectUrl>
<RepositoryUrl>https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience</RepositoryUrl>
<RepositoryType>git</RepositoryType>
```

Both point at the subdirectory rather than at the repository root, because that is where the
project actually is — GravicodeScience is one project inside the Vibe_ML monorepo, and a reader
who lands on the root has to go looking. The usual advice is to give `RepositoryUrl` a clone URL;
here the deep link is the more useful thing for a person following it from nuget.org.

Note that `PublishRepositoryUrl` is deliberately **not** set. It would let SourceLink overwrite
`RepositoryUrl` with the git origin, which would replace that deep link with the repository root.

Two things ride along with each package:

- **XML documentation.** `GenerateDocumentationFile` is on for packable projects, so the comments
  in this repository reach a consumer through IntelliSense rather than only through the source.
  `CS1591` (missing comment on a public member) is silenced rather than fixed: a gap in the prose
  is a gap in the docs, not a broken build, and treating it as an error would gate packing on
  writing.
- **A `.snupkg` symbol package**, so a debugger can step into library code.

## Cutting a release

The version comes from the tag, so the two cannot disagree:

```bash
git tag gravicode-science-v1.0.0
git push origin gravicode-science-v1.0.0
```

The tag is prefixed because a bare `v1.0.0` would be ambiguous in a monorepo. `release.yml` picks
it up, checks it parses as a semantic version, builds, **runs the whole test suite**, packs at
that version and pushes to nuget.org.

Publishing to nuget.org is irreversible — a version can be unlisted but never replaced — which is
why the pushing workflow is the only one that pushes, why it is tag-triggered rather than
branch-triggered, and why the test suite gates it. The `workflow_dispatch` entry defaults
`dry_run` to true for the same reason.

`GraviNum` is pushed first because every other package depends on it; a dependency that is not yet
indexed leaves the dependents briefly unrestorable. `--skip-duplicate` means a re-run after a
partial failure finishes the rest instead of failing on what already landed.

### One secret

`NUGET_API_KEY`, under **Settings → Secrets and variables → Actions**, from
[nuget.org/account/apikeys](https://www.nuget.org/account/apikeys), scoped to `Gravicode.Science.*`.
Nothing else is needed; the GitHub Release is created with the built-in `GITHUB_TOKEN`.

### Publishing by hand

```bash
dotnet pack Gravicode.Science.sln -c Release -p:Version=1.0.0

dotnet nuget push "artifacts/packages/Gravicode.Science.GraviNum.1.0.0.nupkg" \
  --api-key "$NUGET_API_KEY" --source https://api.nuget.org/v3/index.json

# then the other five, in any order
```

Packages land in `artifacts/packages/`, which is gitignored.

## Continuous integration

`ci.yml` runs on every push to `main` and every pull request that touches `GravicodeScience/**`.

| Job | Runs on | What it does |
|---|---|---|
| `build` | ubuntu-latest, windows-latest | Restore, build Release, run all 1,049 tests, upload the `.trx` |
| `notebooks` | ubuntu-latest | Compiles every code cell of all six notebooks |
| `pack` | ubuntu-latest | Packs the six libraries and uploads them as an artifact |

The matrix is `fail-fast: false` — one platform failing is worth seeing on the other.

**The notebook job earns its place.** A notebook is JSON, so it validates whether or not the C#
inside it compiles. All six notebooks once called `plot.GetImageHtml(...)`, which ScottPlot 5.1.59
marks `[Obsolete(error: true)]`; every chart cell would have failed at run time and neither the
build nor the test suite could see it. `tools/verify/notebook_cells.py` concatenates each
notebook's code cells into one file and compiles it against the libraries — which is *stricter*
than the notebook, since .NET Interactive allows a variable to be redeclared across cells and this
does not.

Warnings are not treated as errors. The six shipping libraries build clean; ScienceAppGen has
three known warnings, and failing on those would gate the packages on the IDE.

### The workflow files live at the repository root

GitHub reads workflows only from `.github/workflows/` at the root of a repository. These files are
kept here beside the project they describe, so after cloning Vibe_ML they need to be in place at
the root:

```bash
mkdir -p .github/workflows
cp GravicodeScience/.github/workflows/*.yml .github/workflows/
```

They are already written for that layout: every step sets `working-directory: GravicodeScience`
and the triggers are filtered on `GravicodeScience/**`, so they ignore changes elsewhere in the
monorepo.

## Before the first publish

- `testkey.txt` holds a live Azure OpenAI key and a Tavily key. It is in `.gitignore` — check that
  it is not in the repository's history before making anything public. ScienceAppGen reads those
  values from `SCIENCEAPPGEN_APIKEY`, `SCIENCEAPPGEN_ENDPOINT` and `TAVILY_API_KEY`, and
  `app.config` ships with both key fields blank, so nothing needs the file to be committed.
- Reserve the `Gravicode.Science.*` prefix on nuget.org if the account owns it, so nobody else can
  publish under it.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
