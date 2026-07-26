# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
dotnet build                                    # whole solution
dotnet run --project src/BlazorML.Web           # run the app (creates + seeds its DB on first start)
dotnet test                                     # everything — 684 tests

dotnet test tests/BlazorML.Tests                # core, ML engine, catalog, serializers, scripts (409)
dotnet test tests/BlazorML.Agents.Tests         # kernel factory, plugins, chat sessions, outputs, dataset preview (89)
dotnet test tests/BlazorML.Web.Tests            # Razor components, scoring API, password reset (123)

dotnet test --filter "FullyQualifiedName~EvaluatorTests"        # one class
dotnet test --filter "FullyQualifiedName~Auc_is_one_half"       # one test
dotnet test --filter "FullyQualifiedName~TrainingIntegration"   # trains real ML.NET models
dotnet test --filter "FullyQualifiedName~TrainerCoverage"       # fits every trainer in the catalog
dotnet test --filter "FullyQualifiedName~ScoringApi"            # boots the real app
```

Seeded sign-in: `admin@gravicode.com` / `StudioML#2026`.

The app writes its SQLite database and its object storage side by side under
`src/BlazorML.Web/App_Data/`. **Delete that folder to force a clean re-seed** — the seeder only
runs when the dataset table is empty.

## Architecture

Five projects plus tests. Dependencies flow one way, left to right:

```
Core ──► Infrastructure ──► ML ──► Agents ──► Web
```

- **`BlazorML.Core`** — entities, `ExperimentGraph`, `TabularData`, `ModuleCatalog`, abstractions,
  options classes. No heavy dependencies, so every layer can use it.
- **`BlazorML.Infrastructure`** — `AppDbContext` (four DB providers), Identity, four storage
  providers behind `IStorageProvider`, `SettingsService`, `DataSeeder`.
- **`BlazorML.ML`** — `MlDataBridge`, `PipelineBuilder`, 8 module executors, `Evaluator`,
  AutoML, `ExperimentRunner`, script runners.
- **`BlazorML.Agents`** — Semantic Kernel factory for 4 providers, chat service, kernel plugins.
- **`BlazorML.Web`** — Blazor Server, design system, D3 canvas, Minimal API scoring.

`docs/arsitektur.md` explains why it is split this way. `PLAN.md` has the roadmap and
`Progress.md` records what was built, what is limited, and every bug found along the way.

### The three things worth understanding before changing anything

**1. `ModuleCatalog` is the single source of truth for modules.** The palette, the canvas
renderer, the parameter inspector, graph validation and the run engine all read from it. Adding
a module is one declaration in `src/BlazorML.Core/Modules/ModuleCatalog.cs` plus handling its id
in the matching executor — **no UI code**. `ModuleCatalogTests` enforces the invariants that make
this safe (unique ids, choices that contain their own default, conditional fields pointing at
real parameters).

**2. `TabularData` is the payload on every `Dataset` edge, not `IDataView`.** Transforms, LLM
modules and user scripts all need to read and write individual cells. Conversion to `IDataView`
happens only in `MlDataBridge`, immediately before a trainer needs one, and goes via a temporary
CSV and a runtime-built `TextLoader` — because the designer never knows the schema at compile time.

**3. Evaluation is computed from the scored table, not from ML.NET's metrics objects.** ML.NET
gives summary numbers but no ROC points and no residual sample, and the charts need both. One
code path in `Evaluator` produces the summary figures *and* the chart shapes, and it also works
for tables scored outside ML.NET — an LLM classification, for instance.

### Traps that have already bitten

These are fixed and covered by tests. Re-introducing any of them will fail the suite, but they
are worth knowing before touching the surrounding code:

- **SQLite cannot `ORDER BY` a `DateTimeOffset`.** `AppDbContext` installs a UTC-ticks value
  converter for SQLite only. Do not remove it; almost every list orders by a timestamp.
- **Only `true`/`false` count as boolean in `InferTypes`.** "ya"/"tidak" and "yes"/"no" read as
  boolean to a human, but ML.NET's `TextLoader` cannot parse them when a column is declared
  Boolean, and it fails the whole load.
- **`PipelineBuilder` normalises the `Features` vector.** Trees do not care; SDCA, the perceptron
  and the gradient trainers underfit badly without it.
- **Stratifying a split by a continuous column is refused.** It used to produce a 399/1 split
  silently and report regression metrics computed from one row.
- **`MlDataBridge.IsReadable` keys on CLR raw type, not on `DataViewType` subclass.** Keying on
  the subclass dropped key-typed columns, which is what K-Means emits as its cluster index.
- **Not every vector column holds `float`.** `BuildGetter` handles `VBuffer<float>` *and*
  `VBuffer<double>`; the spike detector emits `Vector<Double, 3>`, and assuming single precision
  made training succeed and then reading the result fail.
- **PCA anomaly detection clamps its rank to the feature count.** The default rank is 5, so the
  raw ML.NET error fired on any dataset narrower than that — on a default nobody had touched.
  The clamp logs a warning rather than silently adjusting.
- **Naive Bayes binarises its features.** ML.NET's implementation only sees presence or absence,
  never magnitude, so on ordinary numeric columns it lands on chance. Say so in any description;
  do not "fix" it by tuning.
- **JS interop cannot run during prerender.** `Designer.razor` guards on `_canvasReady`.
- **Form controls are styled by what they are not.** `<input @bind="x" />` renders as `<input>`
  with **no type attribute**, so `input[type='text']` — an attribute selector — does not match it.
  The rule in `app.css` excludes checkbox/radio/range/file/buttons and styles everything else;
  never go back to enumerating types. `FormStyleTests` measures computed style in a browser,
  because an unmatched selector produces no error anywhere and identical markup.
- **A row limit must stop the reader, not trim the result.** `ReadJsonAsync` used to parse the
  whole document before looking at a row, so previewing 25 rows of a large JSON upload read all of
  it. Counting returned rows cannot see that — `SerializerTests` measures bytes consumed.
- **A `<textarea>` whose contents change programmatically must bind `value`, not element content.**
  Once the user has typed in one, the browser stops reflecting changes to its child text node, so
  resetting it silently does nothing. The markup updates correctly, so bUnit sees no problem —
  `EndpointConsoleTests` checks the live value through a browser instead.
- **The canvas pan/zoom transform goes on `g.dz-edge-layer`, never on `svg.dz-edges`.** A
  transform on an outermost `<svg>` is not the same operation as one on a group, so the edges
  drifted away from the nodes as soon as the canvas was scaled — over 120 px after one zoom step.
  Nodes are HTML and edges are SVG, so every view change is applied twice in two coordinate
  systems and nothing in the DOM enforces that they agree; `ZoomTests` measures that they do.
- A `ParameterKind.Choice` carries its own options; **use `ParameterKind.DatasetRef`** for the
  dataset picker, whose options come from the workspace at runtime. `ParameterFieldTests` runs
  parametrically over the whole enum, so a new kind with no case fails rather than silently
  rendering the wrong control.
- **Scoring must work on rows with no label column** — that is the entire point of an endpoint.
  `MlDataBridge.EnsureLabelColumn` puts a placeholder back, because the fitted pipeline's schema
  demands the column the caller is asking you to predict.
- **Requiredness is declared on `ParameterSpec.Required`, never inferred.** Guessing from "has no
  default" marked optional fields as blocking and made validation cry wolf.

## Testing the agent layer

Almost none of it needs a provider, and the tests are written that way:

- **Building a kernel is local work.** `KernelFactory` wires up a client without calling anyone,
  so provider selection, the not-configured refusal and plugin attachment all test with
  placeholder keys.
- **The plugins are ordinary code the model happens to invoke.** `DesignerPlugin` and
  `DataPlugin` run against a real temp SQLite workspace — see `WorkspaceFixture`.
- **The LLM modules test through a stub `ILlmActionRunner`.** What is under test is the batching,
  the fenced-JSON handling and the per-row fallback, not the model's answer.

Only `WicakChatService.SendAsync` genuinely needs credentials, and it is the one thing left
uncovered there.

## Conventions

- **UI language is Indonesian**; code, identifiers and comments are English. Routes are
  Indonesian (`/eksperimen`, `/pengaturan`).
- **Design system is hand-written CSS**, no framework and no build step. Theme values are
  declared once as `light-dark(light, dark)` in `wwwroot/app.css`.
- **The six category pens are a validated categorical palette.** Changing one means re-running a
  CVD/contrast validator, not eyeballing it — both themes currently pass.
- **Colour encodes category, never decoration.** A module family's pen appears on its palette
  chip, node bar, ports and edges. Nothing is identified by colour alone.
- Errors say what happened and what to do about it. A module that cannot run explains what is
  missing rather than failing silently or returning empty output.

## Attribution

App and docs credit Gravicode Studios, led by Kang Fadhil.

## Optional build: vision and NLP

Image and text classification are compiled in only with a flag:

```bash
dotnet build -p:EnableVisionNlp=true
```

The two backbones pull roughly **1.2 GB of native binaries** between them — TensorFlow (~864 MB)
for images, libtorch (~319 MB per platform) for text — which is about twenty times the rest of
the solution's restore. That is why it is not the default.

Both modules stay in `ModuleCatalog` in every build, so they are discoverable, documented and
covered by the catalog invariants. Only the training code is behind `#if VISION_NLP`
(`VisionNlpTrainers.cs`); without the flag, running one throws a message naming the flag and the
download size. If you change that file, build it **both ways** — the guarded branch is not
compiled by a normal build, and a mistake in it will not surface until someone enables the flag.

## Canvas tests (browser)

`tests/BlazorML.Canvas.Tests` drives the designer with a real Chromium through Playwright. It is
the only way to verify that surface: D3 lays out SVG and HTML over JS interop on a Blazor
circuit, and none of it exists until a browser runs it.

```bash
# One-off, from the test project's output folder:
.playwright/node/<platform>/node .playwright/package/cli.js install chromium
```

Without the browser every test **skips with the reason** — never passes. The suite starts the
built `BlazorML.Web.dll` as a child process (not `dotnet run`, which reads launchSettings.json
and overrides the port) against a temp database, and blanks every provider credential by
environment variable so results do not depend on whose machine it runs on.

**Layouts do not inherit a page's render mode.** `ThemeToggle`, `UserMenu` and `TopBarSlot` sit in
`MainLayout` and each carry their own `@rendermode InteractiveServer`; without it their click
handlers are never attached and the controls silently do nothing. bUnit cannot catch this — it
renders components interactively by definition — so anything interactive placed in a layout
needs a canvas test.

**`SectionOutlet` needs both the `@using` and the render mode.** Two separate traps that stack:

- `Microsoft.AspNetCore.Components.Sections` must be imported. Without it Razor emits
  `<SectionOutlet>` and `<SectionContent>` as **unknown HTML elements**, with no warning — the
  bar stays empty and every page's title renders in the body instead. It is in `_Imports.razor`.
- The outlet must be in the same renderer as the content. The pages that fill the top bar are
  `InteractiveServer`, so the outlet is wrapped in `TopBarSlot`, which declares the same mode.
  A statically rendered outlet shows the buttons and none of them work.

## Secrets

`appsettings.json` ships with every credential blank, and `ShippedSettingsTests` fails the build
if one reappears there. Put keys in user secrets instead — the project already has a
`UserSecretsId`:

```bash
dotnet user-secrets --project src/BlazorML.Web set "Chat:OpenAI:ApiKey" "sk-..."
```

User secrets are loaded in Development and override `appsettings.json`, so the app behaves
identically without the key ever entering a file that gets committed.
