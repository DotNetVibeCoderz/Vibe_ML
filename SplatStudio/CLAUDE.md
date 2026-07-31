# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
dotnet restore                        # from repo root (SplatStudio.sln)
dotnet build
cd src/SplatStudio.Web && dotnet run   # the only runnable project; http://localhost:5080

dotnet test tests/SplatStudio.Tests
dotnet test tests/SplatStudio.Tests --filter "FullyQualifiedName~Benchmark" \
  --logger "console;verbosity=detailed"     # GPU vs CPU timing table
```

Tests tagged `[GpuFact]` **skip** (not fail) when no CUDA/OpenCL device is present, so the
suite is green on a GPU-less machine. `GpuDiagnosticTests` walks ILGPU initialisation stage by
stage — run it first when the GPU path breaks on a new machine, since a bare "GPU unavailable"
tells you nothing.

There are no EF migrations — the schema is created by `Database.EnsureCreatedAsync()` in
`Program.cs` on first run, so **any entity change requires deleting `App_Data/`** or the new
column silently won't exist.

First run creates `App_Data/splatstudio.db` (SQLite), seeds 6 demo users (all with password
`Demo123!`, documented account `demo@splatstudio.local`), 12 procedurally generated splat
scenes and 6 generated mesh scenes, and converts the splat ones in the background. To
re-seed, delete `App_Data/` — seeding no-ops permanently once any `AspNetUsers` row exists.

## Architecture

.NET 8 Blazor Web App (Interactive Server render mode, global) in four projects:

| Project | Depends on | Holds |
|---|---|---|
| `SplatStudio.Domain` | nothing | `Entities.cs`, `Enums.cs` — POCOs only |
| `SplatStudio.Application` | Domain | `Abstractions.cs` — every port interface, one file |
| `SplatStudio.Infrastructure` | Application, Domain | EF Core, Identity, storage adapters, splat engines, background worker |
| `SplatStudio.Web` | all three | Blazor components, Razor Pages for auth, wwwroot |
| `tests/SplatStudio.Tests` | Infrastructure | Engine + format tests, GPU equivalence, benchmark |

All ports live in the single file `src/SplatStudio.Application/Abstractions.cs`:
`IStorageService`, `IGaussianSplatEngine`, `IConversionQueue`, `ISceneUpdateNotifier`,
`IAppEmailSender`, `IImageProcessingService`. All wiring happens in one method —
`InfrastructureServiceCollectionExtensions.AddSplatInfrastructure`. Add new services there,
not in `Program.cs`.

### The one pipeline that matters

`Upload.razor` → save original + 480px thumbnail via `IStorageService` → insert `ImageAsset` +
`SplatScene` (`Status = Queued`) → `IConversionQueue.QueueSceneConversion(sceneId)` →
`ConversionBackgroundService` dequeues, sets `Processing`, runs `IGaussianSplatEngine`, writes
`splats/{userId}/{sceneId}.splat`, sets `Completed`/`Failed` → `ISceneUpdateNotifier` fires →
open Blazor circuits (`Home.razor`, `MyScenes.razor`, `SceneViewer.razor`) subscribe to
`Notifier.SceneUpdated`, reload, and `StateHasChanged()`. Any page that subscribes **must**
unsubscribe in `DisposeAsync` — all three existing pages do.

Storage keys are always built server-side with the shape `images|thumbnails|splats/{userId}/{id}.ext`.
Binary content never touches the database or the SignalR circuit; the browser fetches it over plain
HTTP from the URL returned by `IStorageService.GetPublicUrlAsync`.

### Conversion modes

The uploader picks one of three `ConversionMode` values, stored on the scene and fixed for
its lifetime. Each is an `IConversionEngine`; `IConversionEngineCatalog` lists them for the
upload picker and resolves one for the worker.

| Mode | Engine | Artifact | Availability |
|---|---|---|---|
| `HeuristicSplat` | `HeuristicConversionEngine` (wraps `IGaussianSplatEngine`) | `.splat` | always |
| `PhotorealSplat` | `HostedPhotorealConversionEngine` | `.splat` | needs `Splatting:Hosted:Photoreal` |
| `Mesh` | `HostedMeshConversionEngine` | `.glb` | needs `Splatting:Hosted:Mesh` |

- **Unavailable modes stay registered.** The picker renders them disabled with
  `UnavailableReason`, which is why the engines are registered unconditionally rather than
  only when configured. Availability is re-checked in the worker, because config can change
  between upload and dequeue.
- **Both hosted engines share `HostedGenerationClient`** — a submit → poll → download loop
  whose vendor-specific naming (paths, multipart field, JSON paths for job id / status /
  result URL) is *configuration*, not code. No file names a specific vendor; point
  `Splatting:Hosted:*` at Rodin, Tripo, Hunyuan or a self-hosted runner.
- **Mesh scenes are a second artifact type**, not a variant of the first: stored under
  `meshes/` as `.glb`, rendered by `wwwroot/js/mesh-viewer.js` (a separate viewer that adds
  lighting and lazy-loads `vendor/GLTFLoader.js`), and offered as a download on the scene
  page. `SplatScene.ArtifactKind` selects the viewer; `OutputStorageKey` hides which of the
  two key columns holds the file. Deleting a scene must clear **both** keys.
  The download is a plain `<a download>` pointing at the storage URL, so it saves directly
  only for same-origin providers (`FileSystem`); remote buckets need `Content-Disposition`.
- Hosted output is validated before storing — splat bytes must be a multiple of 32, meshes
  must start with the `glTF` magic — so a misconfigured provider fails loudly instead of
  producing a scene that renders as nothing.

### Splat engines

The three implementations of `IGaussianSplatEngine` behind `HeuristicSplat`, chosen by
`Splatting:Engine`:

- `LocalHeuristicSplatEngine` — CPU, stateless, clamps its working image to 512×512
  (≈262k points max).
- `GpuSplatEngine` — the *same* heuristic as ILGPU compute kernels (CUDA, OpenCL fallback).
  ~3× faster at 262k points and the only path above that ceiling. Falls back to the CPU
  engine when no device is present. Uses `Splatting:GpuMaxPoints` (250k) rather than
  `MaxPoints`.
- `ExternalApiSplatEngine` — documented skeleton for a real 3DGS backend.

Two constraints that are easy to break:

- **Kernel parameter types must be `public` and top-level.** ILGPU emits
  `ViewImplementation<T>` per buffer element type via Reflection.Emit; an internal or nested
  type fails at kernel-load with `TypeLoadException: Access is denied`. That is why
  `GpuSplatRecord` is its own public file.
- **The GPU engine must never be registered from a scoped factory.** A scoped factory that
  returns a shared instance makes the container dispose it at scope end, which tore down the
  accelerator after the first conversion and silently dropped every later scene to the CPU.
  `LocalHeuristic`/`Gpu` are registered as singletons; only `ExternalApi` is scoped (it holds
  an `HttpClient`).

### Provider switching

Three DB providers (`DatabaseConfiguration.ConfigureProvider`) and four storage providers
(`StorageServiceCollectionExtensions.AddSplatStorage`) are each selected by one config string.
Those two files are the *only* places allowed to branch on vendor — nothing else in the app may
reference a concrete DB or storage SDK. `S3StorageService` backs both `S3` and `MinIO`
(differing only by `ServiceUrl`/`ForcePathStyle`). Config lives in
`src/SplatStudio.Web/appsettings.json` under `Database:`, `Storage:`, `Splatting:`, `Email:`.

### The `.splat` format

32 bytes/point, defined by `SplatFileWriter.cs`, mirrored by `GpuSplatRecord`, and parsed by
`wwwroot/js/splat-viewer.js`: 3×float32 position, 3×float32 scale, 4×uint8 RGBA, 4×uint8
quaternion (byte 0..255 ↔ −1..1). **All three must change together.**

`.splat` and `.glb` are served through the `FileSystem` provider's static-file mapping in
`Program.cs`, which needs an explicit `ContentTypeProvider` entry for each — the middleware
404s unknown extensions, so dropping either mapping makes those scenes fail to load with no
server-side error.

The viewer is hand-written against Three.js r128, **vendored** at
`wwwroot/js/vendor/three.min.js` (it previously used a CDN URL that 404'd, so the viewer never
worked). It skips per-frame depth sorting and ignores the stored rotation quaternion by design.
`gl_PointSize` is derived from viewport height and camera FOV via the `uProjScale` uniform;
a hard-coded constant there under-sizes points below one pixel at high splat counts and the
cloud renders as nothing.

The heuristic — inverse-luminance blended with a centre-weighted radial prior, one splat per
downsampled pixel — is a "2.5D" approximation, *not* real multi-view 3DGS. Don't claim
otherwise in UI copy or docs; `/about` states the limits explicitly and the seeded gallery
deliberately keeps failure cases (`torus`, `dunes`) visible.

## Conventions and constraints

- **Auth is Razor Pages, everything else is Blazor.** Login/Register/Logout/ForgotPassword/
  ResetPassword live in `Pages/Account/*.cshtml` because writing the auth cookie requires a real
  HTTP response, which an established SignalR circuit cannot do. Don't port these to components.
- **Blazor components inject `ApplicationDbContext` directly** (a scoped DbContext per circuit),
  so `SplatStudio.Web` references Infrastructure — a deliberate deviation from strict Clean
  Architecture. Follow the existing pattern rather than introducing a repository layer. Never use
  that DbContext concurrently from two async paths in one component.
- **A page that both mutates and live-reloads must call `Db.ChangeTracker.Clear()` before
  re-querying.** The circuit's DbContext tracks what it loaded, and EF identity resolution
  returns that stale instance instead of the current row — so background-worker progress
  never appears. `SceneViewer` and `MyScenes` both do this; `Home` avoids it with
  `AsNoTracking()` because it never writes.
- **One background worker, not a pool** — conversion is CPU-bound and competes with SignalR
  circuits in the same process. Scale out with more instances. The queue is a bounded channel
  (capacity 256) for backpressure.
- **No MediatR/CQRS**, by design.
- Deleting a scene must delete storage objects *before* removing the rows (see
  `MyScenes.razor` `DeleteAsync`) — EF cascade only cleans the database.
- Types intentionally live in few, large files (`Entities.cs`, `Enums.cs`, `Abstractions.cs`,
  `EmailSenders.cs`). Add to the existing file rather than creating one type per file. The
  exception is `GpuSplatRecord`, which ILGPU forces to be top-level and public.
- Seed assets are generated procedurally, never shipped as binaries. `SampleImageFactory`
  draws the photos (add a `Catalogue` entry plus a `Shade` branch); `SampleMeshFactory` writes
  binary glTF from scratch (add a `Catalogue` entry plus a `Build` branch). Both must stay pure
  functions of their key — `Output_is_deterministic…` asserts it, because a gallery that
  differs per deployment makes screenshots and bug reports useless.
- **Seeded mesh scenes bypass the conversion queue.** Mode 3 has no local engine, so queuing
  them would fail every one on an unconfigured install. They are written straight to storage
  with `Engine = SplatEngineType.SampleData` — never attribute demo geometry to a model that
  did not run.
- `SampleMeshFactory` also rasterises its own thumbnails (small z-buffered triangle filler).
  Mesh cards must show the actual model; reusing an unrelated source photo would misrepresent
  the scene. Curve functions passed to `Sweep` receive their parameter **in radians over
  [0, tau]**, and `Sweep(closed:)` must be false for open curves or a band is drawn from the
  end back to the start.
- XML doc comments on ports and infrastructure classes explain *why* a decision was made; keep
  that style when adding to those layers.

## UI

Single hand-written stylesheet: `wwwroot/css/app.css` — the **"Depth Ramp"** system. No CSS
framework, no build step, no scoped `.razor.css`. Add styles there using the existing tokens.

- The palette is three stops (`--near` amber → `--mid` rose → `--far` indigo) that *encode
  depth*, matching how depth maps are conventionally read. Use them semantically (the hero
  legend, conversion status, the brand mark), not as generic accents.
- Type roles are fixed: Bricolage Grotesque display, Inter body, **IBM Plex Mono for every
  measured value** (splat counts, timings, dates). Numbers should read as instrument output.
- `Components/App.razor` and `Pages/Shared/_Layout.cshtml` both declare the font links and
  chrome — the Razor Pages auth flow uses the second one, so keep them in step.
- Focus rings are scoped to interactive elements, and `[tabindex="-1"]:focus` is explicitly
  cleared: Blazor focuses the page `<h1>` after navigation for screen readers, which otherwise
  draws a ring around every page title.
- **Scope element-type selectors.** A bare `form button[type="submit"]` also matched the
  navbar's sign-out button and knocked it out of line; it is now `.form-panel button[...]`.
  Prefer a class ancestor over a bare tag selector for anything spacing-related.

`.claude/skills/frontend-design/SKILL.md` carries the design brief for new or reshaped UI.

## Docs

`README.md` and `README.id.md` (Indonesian) are kept in sync — update both when changing
behaviour they describe.
