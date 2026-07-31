# SplatStudio

Turn a single uploaded photo into an orbitable 3D Gaussian-splat point cloud, right in the browser — with a gallery, comments, 1-5 star ratings, full account management, and pluggable database and storage backends.

Built with **.NET 8** (Blazor Web App, Interactive Server render mode) using Clean Architecture (Domain / Application / Infrastructure / Web).

> 🇮🇩 Baca dalam Bahasa Indonesia: [README.id.md](README.id.md)

---

## Table of contents

1. [What this actually does (please read this first)](#what-this-actually-does-please-read-this-first)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Getting started](#getting-started)
5. [Configuration](#configuration) — including [Conversion modes](#conversion-modes)
6. [Demo accounts & sample data](#demo-accounts--sample-data)
7. [Plugging in a real 3D Gaussian Splatting backend](#plugging-in-a-real-3d-gaussian-splatting-backend)
8. [Performance notes](#performance-notes)
9. [Known limitations & v1 simplifications](#known-limitations--v1-simplifications)
10. [Deployment notes](#deployment-notes)

---

## What this actually does (please read this first)

**Real 3D Gaussian Splatting** (the technique behind tools like Luma AI or nerfstudio/gsplat) is trained from *many* photos of the same subject taken from different angles. A pipeline like COLMAP first reconstructs camera positions and a rough point cloud via Structure-from-Motion, and then thousands of Gaussians are optimized over many GPU-hours of gradient descent against all those views. A single flat photo simply doesn't contain the multi-view information needed to reconstruct true 3D geometry — no amount of clever code changes that.

What SplatStudio ships instead is a fast, fully offline, deterministic "2.5D" reconstruction, in two interchangeable implementations: `LocalHeuristicSplatEngine` (CPU) and `GpuSplatEngine` (the same maths as ILGPU compute kernels). For each uploaded image it:

1. Downscales the image to a point budget (`Splatting:MaxPoints` / `GpuMaxPoints`).
2. Estimates a per-pixel **pseudo-depth** as a blend of inverse-luminance and a center-weighted radial prior (brighter, more central pixels are pulled toward the camera).
3. Smooths that depth field with a small box blur.
4. Emits one Gaussian splat per pixel, positioned in that depth field, colored from the source pixel.
5. Writes the result as a compact 32-bytes-per-point binary `.splat` file, rendered in-browser with a small hand-written Three.js point-cloud viewer (`wwwroot/js/splat-viewer.js`).

The result is a genuine, freely-orbitable 3D point-cloud world — and it looks surprisingly good for portraits or single objects on a plain background, which is exactly what the bundled sample images demonstrate. It is **not**, however, a substitute for true multi-view reconstruction, and it will not recover real depth/occlusion for complex scenes. Think of it as a stylized "photo to snow-globe" effect rather than photogrammetry.

The GPU engine is faster, not better: it is the identical approximation with the identical limits. Running on a GPU buys throughput, which is what makes a 250,000-point budget practical — it does not buy accuracy. The app states all of this at `/about` too, so the caveat reaches people who never open this file.

If you have access to a real splatting backend, see [Plugging in a real backend](#plugging-in-a-real-3d-gaussian-splatting-backend) — the engine sits behind a clean interface specifically so it can be swapped out.

## Features

- **Upload → convert → view**: upload a JPEG/PNG/WebP, the app queues a background conversion job, and the gallery/viewer update live (via an in-process `ISceneUpdateNotifier`) the moment it finishes — no manual refresh.
- **Three conversion modes**, chosen per upload: the built-in depth heuristic (instant, free, offline), photorealistic 3D Gaussian splatting via a hosted service, or an image-to-3D **mesh** via a model such as TRELLIS, Hunyuan3D or Rodin. See [Conversion modes](#conversion-modes).
- **Gallery**: a public grid of completed scenes ("constellation gallery"), each showing its thumbnail, average rating, view count, and comment count.
- **Two 3D viewers**, both drag-to-orbit and scroll-to-zoom: a custom lightweight point-cloud renderer written for this project's own `.splat` format (no external splat-viewer library), and a small glTF viewer for mesh scenes that adds lighting and loads its parser on demand. Three.js is vendored locally rather than pulled from a CDN, so neither depends on a third party staying up.
- **Comments & ratings**: signed-in users can leave a comment and a 1-5 star rating per scene (one rating per user per scene).
- **Full account system** via ASP.NET Core Identity: register, login/logout, forgot/reset password (via a swappable email sender), profile editing (display name, bio, avatar upload), change password.
- **My Scenes**: manage your own uploads — toggle public/private, delete (cleans up both database rows and stored files).
- **Three database providers**: SQLite (zero-config default), SQL Server, MySQL — switch with one config value.
- **Four storage providers**: local filesystem (zero-config default), Azure Blob Storage, AWS S3, and MinIO/any S3-compatible service — switch with one config value.
- **"Depth Ramp" UI**: a custom design system whose palette is the product's own output — near is amber, mid is rose, far is indigo, the way depth maps are conventionally read. The home page's hero is a real scene rendering live and swaying gently, not a mockup, with the actual splat count and conversion time underneath it.

## Architecture

```
src/
  SplatStudio.Domain          Entities + enums only, zero dependencies
  SplatStudio.Application     Ports (interfaces) the Web/Infrastructure layers implement against
  SplatStudio.Infrastructure  EF Core, Identity, storage providers, the splat engines, background worker
  SplatStudio.Web             Blazor components, Razor Pages for auth, static assets
tests/
  SplatStudio.Tests           Splat format, engines, GPU/CPU equivalence, conversion modes,
                              sample-mesh validity, benchmark
```

A few deliberate choices worth calling out:

- **No MediatR/CQRS.** Given the project's already-large surface area (3 DB providers × 4 storage providers × full auth × real-time gallery), a simplified Clean Architecture without a mediator pipeline keeps the codebase easier to read end-to-end. The port/adapter separation (`Application` interfaces, `Infrastructure` implementations) is kept, so adding CQRS later is a structural addition, not a rewrite.
- **Blazor Web App hosting model + classic Razor Pages for auth.** Login/Register/Logout/ResetPassword need to write the authentication cookie directly on an HTTP response, which only works *before* a Blazor Server circuit's SignalR connection takes over. So those five flows are plain ASP.NET Core Razor Pages (`Pages/Account/*.cshtml`), styled to match the rest of the app, while everything else is an Interactive Server Blazor component. This mirrors the official `dotnet new blazor -au Individual -int Server` template's approach.
- **One background worker, not a pool.** Image→splat conversion is CPU-bound. Running many of these in parallel on a single instance would starve the same process's Blazor Server SignalR circuits of CPU, making the whole app feel laggy for everyone. `ConversionBackgroundService` deliberately processes one scene at a time; scale out by running more instances, not more workers per instance.
- **`EnsureCreatedAsync()` instead of EF migrations.** With three swappable database providers, maintaining three separate migration histories adds real ongoing maintenance cost for comparatively little benefit in a v1. The schema is created directly from the model on first run. See [Known limitations](#known-limitations--v1-simplifications) for the upgrade path.

## Getting started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Nothing else, for the default configuration — SQLite + local filesystem storage need no external services.
- Optional: an NVIDIA/CUDA (or OpenCL) GPU for the accelerated splat mode. Without one the app falls back to the CPU on its own.

### Run it

```bash
dotnet restore
dotnet build
cd src/SplatStudio.Web && dotnet run
```

Then open <http://localhost:5080>. On first run the app will:

- Create the SQLite database at `App_Data/splatstudio.db`.
- Seed six demo accounts, twelve splat scenes and six mesh scenes (see [Demo accounts](#demo-accounts--sample-data)).
- Convert the sample images in the background — the gallery updates itself as each finishes.

To reset to a clean slate, stop the app and delete `App_Data/`. Because the schema is created with `EnsureCreatedAsync()` rather than migrations, you also need to do that after any entity change.

### Tests

```bash
dotnet test tests/SplatStudio.Tests

# GPU vs CPU timing table
dotnet test tests/SplatStudio.Tests --filter "FullyQualifiedName~Benchmark" \
  --logger "console;verbosity=detailed"
```

The suite covers the `.splat` binary layout, the depth heuristic's output bounds, alpha handling, determinism, GPU/CPU equivalence, the conversion-mode availability rules, and the generated sample meshes — including that each `.glb` is a structurally valid container whose accessors fit its binary chunk, and that its thumbnail is not a blank frame. Tests that need a GPU are **skipped**, not failed, on machines without a CUDA/OpenCL device.

## Configuration

All configuration lives in `src/SplatStudio.Web/appsettings.json` (and `appsettings.Development.json` for local overrides). Every provider switch is a single string value — no code changes needed.

### Database

```json
"Database": { "Provider": "Sqlite" },   // Sqlite | SqlServer | MySql
"ConnectionStrings": {
  "Sqlite":    "Data Source=App_Data/splatstudio.db",
  "SqlServer": "Server=localhost;Database=SplatStudio;User Id=sa;Password=YourStrong!Passw0rd;TrustServerCertificate=True;",
  "MySql":     "Server=localhost;Port=3306;Database=splatstudio;User=root;Password=root;"
}
```

Set `Database:Provider` to `SqlServer` or `MySql` and fill in the matching connection string — that's the entire change required.

### Storage

```json
"Storage": {
  "Provider": "FileSystem",   // FileSystem | AzureBlob | S3 | MinIO
  "FileSystem": { "RootPath": "App_Data/storage", "PublicBasePath": "/media" },
  "AzureBlob":  { "ConnectionString": "", "ContainerName": "splatstudio" },
  "S3":         { "ServiceUrl": "", "AccessKey": "", "SecretKey": "", "BucketName": "splatstudio", "Region": "us-east-1", "ForcePathStyle": false, "PublicBaseUrl": "" },
  "MinIO":      { "ServiceUrl": "http://localhost:9000", "AccessKey": "minioadmin", "SecretKey": "minioadmin", "BucketName": "splatstudio", "ForcePathStyle": true, "PublicBaseUrl": "" }
}
```

- **FileSystem** (default): files live under `App_Data/storage` and are served at `/media/...` via static file middleware. Zero external setup.
- **AzureBlob**: set `ConnectionString` (e.g. from the Azure Portal or `AzureWebJobsStorage`) and `ContainerName`. The container is created automatically with public blob access if it doesn't exist.
- **S3**: set `AccessKey`/`SecretKey`/`BucketName`/`Region`. Leave `ServiceUrl` empty to talk to real AWS S3. The bucket is auto-created on startup if missing.
- **MinIO** (or any other S3-compatible service — Cloudflare R2, Wasabi, etc.): same shape as S3, but set `ServiceUrl` to your endpoint and `ForcePathStyle: true` (most self-hosted S3-compatible servers require path-style addressing).

### Conversion modes

Every upload picks one mode. Only the first runs on your own machine; the other two hand the image to a hosted service.

| Mode | Produces | Runs | Needs |
|---|---|---|---|
| **Depth-estimated splat** | `.splat` point cloud | here, in milliseconds | nothing |
| **Photorealistic splat** | `.splat` point cloud | hosted, minutes | `Splatting:Hosted:Photoreal` |
| **3D object (mesh)** | `.glb` textured mesh | hosted, minutes | `Splatting:Hosted:Mesh` |

A mode with no credentials is **shown on the upload page but disabled**, with the exact setting it needs. Nothing is hidden and nothing accepts an upload it cannot fulfil.

The third mode produces geometry rather than a point cloud, so those scenes open in a glTF viewer and carry a **Download .glb** button — the file is ordinary glTF that opens in Blender or any other 3D tool.

#### Pointing the hosted modes at a provider

Rodin/Hyper3D, Tripo, Tencent Hunyuan3D and a self-hosted TRELLIS runner all speak the same protocol — POST the image, get a job id, poll until it finishes, download the asset. Only the naming differs, so that naming is configuration rather than three hand-written vendor clients that would rot on the next API revision:

```json
"Splatting": {
  "Hosted": {
    "Mesh": {
      "BaseUrl": "https://api.your-provider.com",
      "ApiKey": "",                       // use user secrets or the environment
      "SubmitPath": "/v1/generate",
      "ImageFieldName": "image",
      "SubmitFields": { "output_format": "glb" },
      "JobIdPath": "data.job_id",         // dotted path into the JSON response
      "StatusPath": "/v1/generate/{jobId}",
      "StatusFieldPath": "status",
      "SuccessStates": [ "succeeded", "done" ],
      "FailureStates": [ "failed", "cancelled" ],
      "ResultUrlPath": "result.url",      // "results.0.url" also works
      "PollIntervalSeconds": 5,
      "TimeoutSeconds": 900
    }
  }
}
```

The defaults describe that contract but are not any real vendor's API — fill in the paths from your provider's docs. Anything the status endpoint returns that is in neither `SuccessStates` nor `FailureStates` is treated as "still running", so you don't have to enumerate every in-progress word.

Two guards worth knowing about: splat responses must be a whole number of 32-byte records, and mesh responses must begin with the `glTF` magic. A provider configured to emit the wrong format fails the scene with a clear message rather than storing something the viewer renders as nothing.

> **The mesh path was verified end to end against a local stub provider implementing the contract above, not against a commercial API.** Nobody's real endpoint paths are baked in, so plan on filling in the field mappings for whichever vendor you use.

### Splat engine

This configures the built-in **depth-estimated** mode only; the hosted modes are configured above.

```json
"Splatting": {
  "Engine": "Gpu",            // LocalHeuristic | Gpu | ExternalApi
  "MaxPoints": 40000,         // used by LocalHeuristic
  "GpuMaxPoints": 250000,     // used by Gpu
  "ExternalApi": { "Endpoint": "", "ApiKey": "", "TimeoutSeconds": 120 }
}
```

`MaxPoints`/`GpuMaxPoints` control both conversion speed and output file size (32 bytes per point) — lower them for faster, lighter scenes; raise them for denser point clouds at the cost of a larger `.splat` download.

**`Gpu`** runs the identical heuristic as ILGPU compute kernels on a CUDA device (with an OpenCL fallback). It is not more accurate — same approximation, same caveats — it is just faster, which is what makes a much larger point budget practical. If no device is found it logs the reason and quietly falls back to `LocalHeuristic`, so the app still runs on a GPU-less machine.

Measured on an NVIDIA RTX 4060 (`dotnet test --filter Benchmark`, 5 iterations, 1024×1024 source):

| point budget | CPU ms/image | GPU ms/image | speedup |
|---:|---:|---:|---:|
| 10,000 | 21.7 | 19.6 | 1.11× |
| 40,000 | 27.5 | 9.9 | 2.78× |
| 100,000 | 27.1 | 11.9 | 2.28× |
| 262,144 | 59.6 | 19.7 | 3.03× |
| 1,000,000 | — | 37.0 | CPU caps at 262,144 |

Both engines share the same CPU-side JPEG decode and Lanczos resize, which is a fixed floor neither can beat — that is why the advantage only appears once point count dominates.

### Email (password reset)

```json
"Email": {
  "Provider": "File",   // File | Smtp
  "Smtp": { "Host": "", "Port": 587, "EnableSsl": true, "Username": "", "Password": "", "FromAddress": "no-reply@splatstudio.local", "FromName": "SplatStudio" }
}
```

The `File` provider (default) writes each "email" as an `.html` file under `App_Data/emails` and — only in this dev mode — surfaces the reset link directly on the "forgot password" confirmation page, so you can test the full flow with zero SMTP setup. Switch to `Smtp` and fill in real credentials for production.

## Demo accounts & sample data

On first run against an empty database, the app seeds:

- **Six accounts**, all with the password **`Demo123!`** — sign in as **demo@splatstudio.local**, or as `rani@`, `tomas@`, `aiko@`, `marcus@`, `priya@` `splatstudio.local` to see the gallery from someone else's side. Each gets a generated avatar and a bio.
- **Twelve splat scenes** distributed across those accounts (two of them private, so "My Scenes" has something to toggle), with comments from several users and a spread of 3–5 star ratings so averages mean something.
- **Six mesh scenes** — a trefoil knot, a turned vase, a cut gem, a nautilus shell, a ridge field and a Möbius band — so the glTF viewer and the `.glb` download have something to show without a hosted provider configured.

Nothing is shipped as a binary. Photos come from `SampleImageFactory` (drawn gradients and shapes) and meshes from `SampleMeshFactory`, which writes binary glTF from scratch — parametric surfaces, vertex-coloured along the same near/far ramp as the interface. Both are pure functions of their recipe key, so every deployment seeds an identical gallery.

Two honesty notes about the seed:

- The splat scenes go through the **real** upload → storage → queue → background-conversion pipeline, so they double as a smoke test of it on every fresh deployment. The catalogue deliberately includes cases the technique handles badly — a torus whose hole gets filled in, a nearly flat dune field — alongside the ones it flatters.
- The mesh scenes **cannot** go through that pipeline, because mode 3 has no local engine; with no provider configured every one would fail and a fresh install would show only errors. They are written straight to storage and labelled **Sample data** rather than being attributed to a model that never ran.

This only happens once — the moment any user exists in the database (including a real person who signs up first), seeding is skipped permanently.

## Plugging in a real 3D Gaussian Splatting backend

For photorealistic splats and meshes the easiest route is filling in `Splatting:Hosted` (see [Conversion modes](#conversion-modes)) — no code at all.

If your provider does not fit that submit/poll/download shape, the splat engine still sits behind one small interface:

```csharp
public interface IGaussianSplatEngine
{
    SplatEngineType EngineType { get; }
    Task<GaussianSplatGenerationResult> GenerateAsync(Stream imageStream, int maxOutputPoints, CancellationToken ct = default);
}
```

`ExternalApiSplatEngine` (in `SplatStudio.Infrastructure/Splatting/`) is a documented starting point: it POSTs the image to a configured HTTP endpoint and expects raw `.splat` bytes back. Most real-world 3DGS providers (Luma AI, KIRI Engine, a self-hosted nerfstudio/gsplat job runner) are asynchronous/job-based rather than synchronous request-response, so treat this stub as a skeleton to adapt to your specific provider's polling or webhook pattern — not a drop-in production client. Implement `IGaussianSplatEngine` against your provider, register it in `InfrastructureServiceCollectionExtensions.AddSplatInfrastructure`, and set `Splatting:Engine` to `ExternalApi`.

To add a genuinely new conversion mode, implement `IConversionEngine` instead — that is the port the upload page enumerates to build its picker and the background worker resolves to pick an engine.

## Performance notes

- **Response compression** is enabled for both Brotli and Gzip, explicitly including `application/octet-stream` so `.splat` files compress well over the wire.
- **Single background worker** (see [Architecture](#architecture)) avoids CPU contention with Blazor Server's SignalR circuits. This matters less with the GPU engine, where the CPU-side image decode is the bottleneck rather than the splat maths.
- **GPU compute** for the depth field and splat emission when a CUDA/OpenCL device is present — see the benchmark table under [Splat engine](#splat-engine).
- **Bounded channel queue** (`ChannelConversionQueue`, capacity 256) provides natural backpressure — uploads queue rather than spawning unbounded concurrent work.
- **Image downscaling** happens before splat generation (`LocalHeuristicSplatEngine` resizes to a point-budget-derived resolution) and again for gallery thumbnails (480px max dimension), so the browser never has to download a full-resolution original just to render a small grid card.
- **Static file caching**: filesystem-backed media is served with a 30-day immutable `Cache-Control` header.
- **`InvariantGlobalization`** is enabled, trimming ICU data the app doesn't need.
- The Three.js point-cloud viewer renders splats as camera-facing billboards without per-frame back-to-front depth sorting (documented in `splat-viewer.js`) — sorting hundreds of thousands of points every frame would cost more than the visual benefit is worth for these mostly-opaque billboards.
- The mesh viewer fetches its glTF parser (~96 KB) only the first time it is needed, so deployments that never enable mode 3 do not pay for it.

## Known limitations & v1 simplifications

- **Heuristic, not true 3DGS** — see [above](#what-this-actually-does-please-read-this-first).
- **EF Core schema via `EnsureCreatedAsync()`, not migrations.** To move to proper per-provider migrations for production: pick one provider, run `dotnet ef migrations add InitialCreate --context ApplicationDbContext`, repeat per provider with a `--output-dir` per provider, and replace the `EnsureCreatedAsync()` call in `Program.cs` with `Database.MigrateAsync()`.
- **No CQRS/MediatR** — see [Architecture](#architecture). Straightforward to layer in later given the existing port/adapter boundaries.
- **Logout is a real HTTP POST**, not a Blazor event — by design (see Architecture), but means a JS-disabled or aggressively-cached client could theoretically replay it; standard ASP.NET Core antiforgery protection is applied via the built-in `<AntiforgeryToken />` component.
- **Splat viewer has no back-to-front depth sort** — a deliberate performance/complexity trade-off, see [Performance notes](#performance-notes).
- **The GPU engine emits points in a non-deterministic order.** It compacts its output with an atomic counter, so two runs over the same image can order the splats differently. The format carries no ordering semantics and the viewer does not depth-sort, so this is invisible — but it does mean GPU output is not byte-comparable run to run, while CPU output is.
- **No integration or UI tests.** The suite covers the splat format, the engines and the mode-selection layer; the Blazor pages, auth flows and storage adapters are only exercised by hand and by the seeding path on startup.
- **No hosted provider is verified against its real API.** The submit/poll/download contract is verified against a local stub, and the field mappings are configuration — but the first real provider you connect will need its paths filled in, and possibly extra fields in `SubmitFields`.
- **The mesh viewer is deliberately minimal.** Lighting is a fixed three-light rig with no environment map or shadows.
- **`.glb` downloads rely on the `download` attribute**, which browsers only honour same-origin. That covers the default `FileSystem` provider; behind Azure Blob or S3 the link opens the file rather than saving it, unless the bucket sets `Content-Disposition`.

## Deployment notes

- Set `Database:Provider` and `Storage:Provider` for your target environment, and use environment variables or a secrets manager for connection strings/keys rather than committing them to `appsettings.json`. That goes for `Splatting:Hosted:*:ApiKey` too.
- Behind a reverse proxy (nginx, Azure App Service, etc.), make sure WebSockets are enabled and forwarded — Blazor Server's Interactive render mode depends on a persistent SignalR connection.
- If you scale to multiple instances, route by sticky session (or use Azure SignalR Service) — Blazor Server circuits are stateful and tied to one instance.
- `UseHsts`/`UseExceptionHandler` are only applied outside the Development environment; set `ASPNETCORE_ENVIRONMENT=Production` for a real deployment.
- Hosted jobs run for minutes and share the single worker with local conversions. If you expect many mode 2/3 uploads at once, scale out — the queue is bounded (256) and will apply backpressure rather than piling up without limit.
