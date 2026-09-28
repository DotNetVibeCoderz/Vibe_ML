# Platforms: desktop, server, mobile and browser

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/platform.md)

## Supported and tested

| Platform | Status | Execution providers |
|---|---|---|
| Windows x64 | ✅ tested in CI (all tests, including golden cross-validation) | CPU, DirectML, CUDA |
| Linux x64 | ✅ tested in CI | CPU, CUDA |
| macOS arm64 | ✅ tested in CI | CPU, CoreML |
| Android (.NET MAUI) | 🟡 sample app builds (`samples/MediaPipeNet.Maui`); not device-tested in CI | CPU, NNAPI |
| iOS (.NET MAUI) | 🟡 same sample, needs a Mac to build | CPU, CoreML |
| Windows ARM64, Linux ARM64, macOS x64 | 🟡 expected to work (ONNX Runtime ships these), not in CI | CPU |

Console apps, ASP.NET Core, worker services, WPF/WinForms/Avalonia desktop apps and Polyglot notebooks all use the
same packages.

## .NET MAUI (Android / iOS)

The libraries target `net10.0` and contain no platform-specific code, so they run inside MAUI apps.
[`samples/MediaPipeNet.Maui`](https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/MediaPipeNet/samples/MediaPipeNet.Maui) is a complete app — pick a photo, run the face
stylizer, the face mesh or object detection, choose the execution provider, see the result and its latency:

```bash
dotnet workload install maui-android              # once
dotnet build samples/MediaPipeNet.Maui -f net10.0-android
dotnet build samples/MediaPipeNet.Maui -t:Run -f net10.0-android   # device or emulator attached
```

What the sample shows, and what your own app needs:

1. **Native runtime** — reference the task packages plus `Microsoft.ML.OnnxRuntime`, which ships the Android `.aar`
   (with NNAPI) and the iOS `.xcframework` (with CoreML). Do not reference `Gravicode.MediaPipeNet`, which pulls the
   desktop natives.
2. **Providers** — `ExecutionProvider.Nnapi` (Android 8.1+, NPU/GPU/DSP, full float32 precision) and
   `ExecutionProvider.CoreML` (iOS). `Auto` tries CUDA → DirectML → CoreML → NNAPI → CPU; operators a mobile provider
   cannot run fall back to the CPU. `ExecutionProviderSelector.GetAvailableProviders()` lists what the device offers.
3. **Models** — model packages copy files to the output folder, which does not exist inside an APK/IPA. Bundle the
   `.onnx` files as `MauiAsset`s, copy them to `FileSystem.AppDataDirectory` on first launch and point
   `BaseOptions.ModelPaths` (catalog id → file) at them — see `ModelInstaller.cs`. The sample bundles the INT8
   variants (a quarter of the size): about 14 MB for five models.
4. **Camera frames** — wrap the platform camera's RGBA buffers with `MPImage.FromPixelData`.

The sample is built on Windows (Android) but has not been run on a device or emulator by the project yet, and mobile
builds are not part of CI (the images lack the MAUI workloads) — treat mobile as a preview in 1.0.

## Blazor WebAssembly — server-side today, in-browser after 1.0

ONNX Runtime's managed API P/Invokes a native library that does not exist in the browser, so the tasks cannot run
inside Blazor WebAssembly as is. The options:

- **Server-side inference** (Blazor Server, or an ASP.NET Core API the WASM client calls) — works today with the
  DI extensions; see [Integration](integration.md).
- **onnxruntime-web** through JS interop, re-using MediaPipe.NET's pre/post-processing (plain C# that compiles to
  WASM). This needs an asynchronous inference backend underneath every task (browser inference is `await`-only, the
  tasks run synchronously today). Designing that abstraction in a hurry would freeze the wrong API into 1.0, so it is
  planned as an additive 1.x feature (a new backend interface next to `OnnxModel`, no breaking change).

## Face stylizer

Supported since 1.0 as `FaceStylizer` — the generator was converted to ONNX after freezing its resource variables and
lowering its training-mode batch norms; see [Tasks](tasks.md#facestylizer) and
[Models](models.md#face-stylizer). No TFLite runtime is needed.
