# Platforms: desktop, server, mobile and browser

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/platform.md)

## Supported and tested

| Platform | Status | Execution providers |
|---|---|---|
| Windows x64 | ✅ tested in CI (all tests, including golden cross-validation) | CPU, DirectML, CUDA |
| Linux x64 | ✅ tested in CI | CPU, CUDA |
| macOS arm64 | ✅ tested in CI | CPU, CoreML |
| Windows ARM64, Linux ARM64, macOS x64 | 🟡 expected to work (ONNX Runtime ships these), not in CI | CPU |

Console apps, ASP.NET Core, worker services, WPF/WinForms/Avalonia desktop apps and Polyglot notebooks all use the
same packages.

## .NET MAUI (Android / iOS) — evaluated, not yet supported

The libraries target `net10.0` and contain no platform-specific code, so they compile into MAUI apps. What is
missing for a supported mobile story:

1. **Native runtime**: reference `Microsoft.ML.OnnxRuntime` (it ships Android `.aar` and iOS `.xcframework` assets)
   instead of `Gravicode.MediaPipeNet`, which pulls the desktop CPU runtime.
2. **Providers**: `ExecutionProvider.CoreML` works on iOS; NNAPI (Android) is not exposed through
   `InferenceOptions` yet — `ExecutionProvider.Cpu` works everywhere.
3. **Models**: model packages copy files to the output folder, which does not exist inside an APK/IPA. Embed the
   `.onnx` files as `MauiAsset`s and serve them with an `EmbeddedResourceModelProvider` or copy them to
   `FileSystem.AppDataDirectory` and point `BaseOptions.ModelDirectory` there.
4. **Camera frames**: wrap the platform camera's RGBA/NV21 buffers with `MPImage.FromPixelData`.
5. **Size**: prefer `ModelPrecision.Int8` / `Float16` variants and only the models you need.

These steps are expected to work but are not covered by CI; a MAUI sample and an NNAPI provider option are on the
roadmap.

## Blazor WebAssembly — evaluated, out of scope

ONNX Runtime's managed API P/Invokes a native library that does not exist in the browser, so the tasks cannot run
inside Blazor WebAssembly as is. The realistic options are:

- **Server-side inference** (Blazor Server, or an ASP.NET Core API the WASM client calls) — works today with the
  DI extensions; see [Integration](integration.md).
- **onnxruntime-web** through JS interop, re-using MediaPipe.NET's pre/post-processing (which is plain C# and does
  compile to WASM) — feasible but requires an `IInferenceBackend` abstraction over `OnnxModel`; tracked for after 1.0.

## Face stylizer

MediaPipe's face stylizer (`face_stylizer_color_sketch.task`) is not ported: its generator graph initializes
resource variables through `CALL_ONCE`/`VAR_HANDLE`/`ASSIGN_VARIABLE`, uses the TensorFlow Flex op
`FusedBatchNormV3` and samples `RANDOM_STANDARD_NORMAL` noise — none of which tf2onnx can convert — and the official
MediaPipe Python package 1.0.1 does not ship the task either, so there is no reference to validate against. It will
be reconsidered when Google publishes a convertible model.
