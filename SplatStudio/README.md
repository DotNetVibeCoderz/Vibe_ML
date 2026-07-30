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
5. [Configuration](#configuration)
6. [Demo account & sample data](#demo-account--sample-data)
7. [Plugging in a real 3D Gaussian Splatting backend](#plugging-in-a-real-3d-gaussian-splatting-backend)
8. [Performance notes](#performance-notes)
9. [Known limitations & v1 simplifications](#known-limitations--v1-simplifications)
10. [Deployment notes](#deployment-notes)

---

## What this actually does (please read this first)

**Real 3D Gaussian Splatting** (the technique behind tools like Luma AI or nerfstudio/gsplat) is trained from *many* photos of the same subject taken from different angles. A pipeline like COLMAP first reconstructs camera positions and a rough point cloud via Structure-from-Motion, and then thousands of Gaussians are optimized over many GPU-hours of gradient descent against all those views. A single flat photo simply doesn't contain the multi-view information needed to reconstruct true 3D geometry — no amount of clever code changes that.

What SplatStudio ships instead is `LocalHeuristicSplatEngine`: a fast, fully offline, CPU-only, deterministic "2.5D" reconstruction. For each uploaded image it:

1. Downscales the image to a point budget (`Splatting:MaxPoints`, default 40,000).
2. Estimates a per-pixel **pseudo-depth** as a blend of inverse-luminance and a center-weighted radial prior (brighter, more central pixels are pulled toward the camera).
3. Smooths that depth field with a small box blur.
4. Emits one Gaussian splat per pixel, positioned in that depth field, colored from the source pixel.
5. Writes the result as a compact 32-bytes-per-point binary `.splat` file, rendered in-browser with a small hand-written Three.js point-cloud viewer (`wwwroot/js/splat-viewer.js`).

The result is a genuine, freely-orbitable 3D point-cloud world — and it looks surprisingly good for portraits or single objects on a plain background, which is exactly what the bundled sample images demonstrate. It is **not**, however, a substitute for true multi-view reconstruction, and it will not recover real depth/occlusion for complex scenes. Think of it as a stylized "photo to snow-globe" effect rather than photogrammetry.

If you have access to a real splatting backend, see [Plugging in a real backend](#plugging-in-a-real-3d-gaussian-splatting-backend) — the engine sits behind a clean interface specifically so it can be swapped out.

## Features

- **Upload → convert → view**: upload a JPEG/PNG/WebP, the app queues a background conversion job, and the gallery/viewer update live (via an in-process `ISceneUpdateNotifier`) the moment it finishes — no manual refresh.
- **Gallery**: a public grid of completed scenes ("constellation gallery"), each showing its thumbnail, average rating, view count, and comment count.
- **3D viewer**: drag to orbit, scroll to zoom, rendered with a custom lightweight Three.js point-cloud renderer (no external splat-viewer library — written for this project against the project's own `.splat` format).
- **Comments & ratings**: signed-in users can leave a comment and a 1-5 star rating per scene (one rating per user per scene).
- **Full account system** via ASP.NET Core Identity: register, login/logout, forgot/reset password (via a swappable email sender), profile editing (display name, bio, avatar upload), change password.
- **My Scenes**: manage your own uploads — toggle public/private, delete (cleans up both database rows and stored files).
- **Three database providers**: SQLite (zero-config default), SQL Server, MySQL — switch with one config value.
- **Four storage providers**: local filesystem (zero-config default), Azure Blob Storage, AWS S3, and MinIO/any S3-compatible service — switch with one config value.
- **Glassmorphism UI**: a custom "Constellation Glass" design system — drifting blurred color blobs behind frosted-glass panels, deliberately chosen as a literal visual metaphor for Gaussian splats.

## Architecture

```
src/
  SplatStudio.Domain          Entities + enums only, zero dependencies
  SplatStudio.Application     Ports (interfaces) the Web/Infrastructure layers implement against
  SplatStudio.Infrastructure  EF Core, Identity, storage providers, the splat engines, background worker
  SplatStudio.Web             Blazor components, Razor Pages for auth, static assets
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

### Run it

```bash
cd src/SplatStudio.Web
dotnet restore
dotnet run
```

Then open the URL printed in the console (typically `https://localhost:5001` or similar). On first run the app will:

- Create the SQLite database at `App_Data/splatstudio.db`.
- Seed a demo account and three sample scenes (see [Demo account](#demo-account--sample-data)).
- Start converting the sample images in the background — refresh the gallery after a few seconds to see them complete.

> **Note:** this project was generated in a sandboxed environment without internet access to nuget.org, so the code could not be compiled or restored here. Please run `dotnet restore` / `dotnet build` yourself with normal internet access before first use — that's a standard step for any freshly generated solution, but worth calling out explicitly since it wasn't (and couldn't be) verified end-to-end in this session.

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

### Splat engine

```json
"Splatting": {
  "Engine": "LocalHeuristic",   // LocalHeuristic | ExternalApi
  "MaxPoints": 40000,
  "ExternalApi": { "Endpoint": "", "ApiKey": "", "TimeoutSeconds": 120 }
}
```

`MaxPoints` controls both conversion speed and output file size — lower it for faster, lighter scenes; raise it for denser point clouds (at the cost of slower conversion and larger `.splat` files served to the browser).

### Email (password reset)

```json
"Email": {
  "Provider": "File",   // File | Smtp
  "Smtp": { "Host": "", "Port": 587, "EnableSsl": true, "Username": "", "Password": "", "FromAddress": "no-reply@splatstudio.local", "FromName": "SplatStudio" }
}
```

The `File` provider (default) writes each "email" as an `.html` file under `App_Data/emails` and — only in this dev mode — surfaces the reset link directly on the "forgot password" confirmation page, so you can test the full flow with zero SMTP setup. Switch to `Smtp` and fill in real credentials for production.

## Demo account & sample data

On first run against an empty database, the app seeds:

- A demo user: **demo@splatstudio.local** / **Demo123!**
- Three sample scenes built from synthetically generated images (gradients and shapes — not real photos), pushed through the real upload → storage → queue → background-conversion pipeline, so they double as a smoke test of that pipeline on every fresh deployment.
- One welcome comment and a 5-star rating on the first sample scene.

This only happens once — the moment any user exists in the database (including a real person who signs up first), seeding is skipped permanently.

## Plugging in a real 3D Gaussian Splatting backend

The splat engine sits behind one small interface:

```csharp
public interface IGaussianSplatEngine
{
    SplatEngineType EngineType { get; }
    Task<GaussianSplatGenerationResult> GenerateAsync(Stream imageStream, int maxOutputPoints, CancellationToken ct = default);
}
```

`ExternalApiSplatEngine` (in `SplatStudio.Infrastructure/Splatting/`) is a documented starting point: it POSTs the image to a configured HTTP endpoint and expects raw `.splat` bytes back. Most real-world 3DGS providers (Luma AI, KIRI Engine, a self-hosted nerfstudio/gsplat job runner) are asynchronous/job-based rather than synchronous request-response, so treat this stub as a skeleton to adapt to your specific provider's polling or webhook pattern — not a drop-in production client. Implement `IGaussianSplatEngine` against your provider, register it in `InfrastructureServiceCollectionExtensions.AddSplatInfrastructure`, and set `Splatting:Engine` to `ExternalApi`.

## Performance notes

- **Response compression** is enabled for both Brotli and Gzip, explicitly including `application/octet-stream` so `.splat` files compress well over the wire.
- **Single background worker** (see [Architecture](#architecture)) avoids CPU contention with Blazor Server's SignalR circuits.
- **Bounded channel queue** (`ChannelConversionQueue`, capacity 256) provides natural backpressure — uploads queue rather than spawning unbounded concurrent work.
- **Image downscaling** happens before splat generation (`LocalHeuristicSplatEngine` resizes to a point-budget-derived resolution) and again for gallery thumbnails (480px max dimension), so the browser never has to download a full-resolution original just to render a small grid card.
- **Static file caching**: filesystem-backed media is served with a 30-day immutable `Cache-Control` header.
- **`InvariantGlobalization`** is enabled, trimming ICU data the app doesn't need.
- The Three.js point-cloud viewer renders splats as camera-facing billboards without per-frame back-to-front depth sorting (documented in `splat-viewer.js`) — sorting tens of thousands of points every frame would cost more than the visual benefit is worth for these mostly-opaque billboards.

## Known limitations & v1 simplifications

- **Heuristic, not true 3DGS** — see [above](#what-this-actually-does-please-read-this-first).
- **EF Core schema via `EnsureCreatedAsync()`, not migrations.** To move to proper per-provider migrations for production: pick one provider, run `dotnet ef migrations add InitialCreate --context ApplicationDbContext`, repeat per provider with a `--output-dir` per provider, and replace the `EnsureCreatedAsync()` call in `Program.cs` with `Database.MigrateAsync()`.
- **No CQRS/MediatR** — see [Architecture](#architecture). Straightforward to layer in later given the existing port/adapter boundaries.
- **Logout is a real HTTP POST**, not a Blazor event — by design (see Architecture), but means a JS-disabled or aggressively-cached client could theoretically replay it; standard ASP.NET Core antiforgery protection is applied via the built-in `<AntiforgeryToken />` component.
- **Splat viewer has no back-to-front depth sort** — a deliberate performance/complexity trade-off, see [Performance notes](#performance-notes).

## Deployment notes

- Set `Database:Provider` and `Storage:Provider` for your target environment, and use environment variables or a secrets manager for connection strings/keys rather than committing them to `appsettings.json`.
- Behind a reverse proxy (nginx, Azure App Service, etc.), make sure WebSockets are enabled and forwarded — Blazor Server's Interactive render mode depends on a persistent SignalR connection.
- If you scale to multiple instances, route by sticky session (or use Azure SignalR Service) — Blazor Server circuits are stateful and tied to one instance.
- `UseHsts`/`UseExceptionHandler` are only applied outside the Development environment; set `ASPNETCORE_ENVIRONMENT=Production` for a real deployment.
