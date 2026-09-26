# Troubleshooting

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/pemecahan-masalah.md)

**`ModelNotFoundException: Model 'x' could not be resolved`**
The message lists every place that was tried. Add the `Gravicode.MediaPipeNet.Models.*` package for that task, set
`MEDIAPIPENET_MODELS` to a folder with the `.onnx` files, set `BaseOptions.ModelDirectory`, or allow downloads
(`ModelStore.CreateDefault(allowDownload: true)`, the default). Behind a proxy, configure `HttpClient.DefaultProxy`.

**"checksum mismatch"**
The file is not the one the catalog expects (partial download, different conversion). Delete it and let the store
download it again, or point `BaseOptions.ModelPaths[id]` at your file (explicit paths are not verified).

**`Could not create an ONNX Runtime session … Make sure a native ONNX Runtime package … is referenced`**
You referenced only the layer packages (`MediaPipeNet.Tasks.Vision`) without a native runtime. Reference
`Gravicode.MediaPipeNet` (CPU), `Gravicode.MediaPipeNet.DirectML` or `Gravicode.MediaPipeNet.Cuda` — exactly one of them.

**The GPU provider is not used**
Check `OnnxModel.Provider`/the Gallery readout and the logs: `Auto` silently falls back to CPU when CUDA libraries
are missing or DirectML is unavailable. `mediapipenet-cli info` lists the providers compiled into the loaded
runtime. Set `FallbackToCpu = false` to get an exception instead.

**`InvalidOperationException: … was created in Image mode; this method requires Video mode`**
Create the task with the matching `RunningMode`. Video mode needs strictly increasing `timestampMs`.

**Handedness looks swapped**
MediaPipe's handedness assumes a mirrored (selfie) image. For a rear camera or an unmirrored photo, swap it, or
mirror frames (`LiveStreamProcessorOptions.MirrorFrames = true`, `MPImage.FlipHorizontal()`).

**No face on full-body images**
`FaceDetector` uses the short-range model (faces within ~2 m, larger than ~15 % of the frame). Crop with
`ImageProcessingOptions.RegionOfInterest`, or use `HolisticLandmarker`, which derives the face region from the pose.

**Webcam does not open**
Try another index (`WebcamFrameSource.ListCameras()`), close other apps using the camera, and make sure an
OpenCvSharp runtime package for your OS is referenced.

**High CPU when running several tasks**
Each session uses all physical cores by default; set `InferenceOptions.IntraOpThreads` (e.g. 2) per task.

**Linux: `DllNotFoundException: onnxruntime`**
Use a glibc-based distribution (Debian/Ubuntu); Alpine/musl is not supported by ONNX Runtime.

**ImageSharp licensing**
MediaPipe.NET uses SixLabors.ImageSharp 3.1 (Six Labors Split License: Apache-2.0 terms for open source and for
organizations under the revenue threshold). Commercial users above the threshold need a Six Labors license.
