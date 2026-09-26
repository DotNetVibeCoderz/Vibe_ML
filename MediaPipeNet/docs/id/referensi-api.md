# Referensi API (ikhtisar)

> 🇬🇧 [Read in English](../en/api-reference.md)

Setiap tipe publik memiliki dokumentasi XML (IntelliSense). Halaman ini memetakan permukaan API.

## MediaPipeNet (Core)

| Tipe | |
|---|---|
| `Timestamp` | Timestamp packet dalam mikrodetik; `FromMilliseconds`, `Unset`, `PreStream`, `Min`, `Max`, `PostStream`, `Done`. |
| `NormalizedLandmark(X, Y, Z, Visibility?, Presence?)` | Landmark dalam koordinat gambar [0,1]; `ToPixel`. |
| `Landmark(X, Y, Z, …)` | Landmark world dalam meter. |
| `NormalizedKeypoint`, `Category`, `Detection` | Hasil deteksi. |
| `RectF`, `NormalizedRect`, `LetterboxPadding`, `ImageSize` | Geometri; `RectF.IntersectionOverUnion`. |
| `RunningMode` | `Image`, `Video`, `LiveStream`. |
| `Angles` | `NormalizeRadians`, `ComputeRotation`. |
| `MediaPipeException`, `ModelNotFoundException` | Error. |
| `Diagnostics.MediaPipeTelemetry`, `Diagnostics.FrameRateCounter` | Meter/ActivitySource dan penghitung FPS. |
| `Serialization.MediaPipeJson` | Opsi JSON bersama. |

## MediaPipeNet.Imaging

| Tipe | |
|---|---|
| `MPImage` | Gambar RGBA yang di-pool: `Load`/`LoadAsync` (file, stream, byte), `FromImage`, `FromPixelData`, `CopyFrom`, `ToImage`, `CopyTo`, `Clone`, `FlipHorizontal`, `SaveAsPng`/`SaveAsJpeg`. |
| `PixelFormat` | `Rgba32`, `Bgra32`, `Rgb24`, `Bgr24`, `Gray8`. |
| `ImageToTensor`, `ImageToTensorOptions`, `BorderMode` | ROI → tensor float NHWC. |
| `TensorMapping` | `TensorToImage`, `ImageToTensor`, `ScaleZ`. |
| `TensorWarp` | `ProjectToImage`, `Resize`. |
| `IFrameSource`, `VideoFrame`, `ImageFileFrameSource`, `MemoryFrameSource` | Input video. |

## MediaPipeNet.Inference

| Tipe | |
|---|---|
| `OnnxModel` | `Load(path | bytes)`, `Inputs`, `Outputs`, `Provider`, `RentContext()`. |
| `InferenceContext` | `GetInput`, `Run`, `GetOutput`, `Dispose` (kembali ke pool). |
| `InferenceOptions`, `ExecutionProvider` | `Provider`, `DeviceId`, `IntraOpThreads`, `InterOpThreads`, `FallbackToCpu`, `EnableProfiling`, `MaxPooledContexts`. |
| `ExecutionProviderSelector` | `GetAvailableProviders`, `GetRuntimeVersion`, `GetCandidates`, `CreateSessionOptions`. |
| `Models.ModelCatalog`, `Models.ModelDescriptor` | 13 model. |
| `Models.ModelStore` | `Default`, `CreateDefault`, `GetModelPath(Async)`, `FindLocal`, `EnsureModelsAsync`, `ComputeSha256Async`. |
| `Models.IModelProvider` + `DirectoryModelProvider`, `BundledModelProvider`, `EmbeddedResourceModelProvider`, `HttpModelProvider`, `NuGetModelProvider` | Sumber model. |

## MediaPipeNet.Framework

| Tipe | |
|---|---|
| `GraphBuilder`, `NodeBuilder`, `GraphOptions`, `GraphInputOptions`, `QueueOverflowPolicy` | Mendeklarasikan graph. |
| `CalculatorGraph` | `StartAsync`, `AddPacket`, `CloseInputStream(s)`, `ObserveOutputStream`, `CreateOutputStreamPoller`, `WaitUntilIdleAsync`, `WaitUntilDoneAsync`, `CloseAsync`, `Cancel`, `ToMermaid`, `GetEdges`, `State`, `Error`, `DroppedPackets`, `InFlightCount`. |
| `ICalculatorNode`, `CalculatorNode`, `CalculatorContract`, `CalculatorContext`, `InputPolicy`, `PortSpec` | Menulis node. |
| `Packet`, `Packet<T>` | Nilai ber-timestamp. |
| `Nodes.*` | `PassThroughNode`, `LambdaNode`, `AsyncLambdaNode`, `CombineNode`, `SinkNode`, `PacketThinnerNode`, `PacketCounterNode`. |
| `Config.GraphConfig`, `Config.NodeConfig`, `Config.CalculatorRegistry` | Konfigurasi `.pbtxt`. |
| `GraphValidationException` | Error saat build/start. |

## MediaPipeNet.Tasks.Vision

| Tipe | |
|---|---|
| Task | `FaceDetector`, `FaceLandmarker`, `HandLandmarker`, `GestureRecognizer`, `PoseLandmarker`, `HolisticLandmarker`, `ImageSegmenter`, `ObjectDetector`, `ImageClassifier` (masing-masing dengan `Create`, `CreateAsync`, method per mode, `ResetTracking`, `DroppedFrames`, `Dispose`). |
| Opsi | Record `…Options` turunan `VisionTaskOptions<TResult>`; `BaseOptions`; `ImageProcessingOptions`; `PoseModel`. |
| Hasil | `FaceDetectionResult`, `FaceLandmarkResult`/`FaceLandmarks`, `HandLandmarkResult`/`HandLandmarks`, `GestureRecognitionResult`/`RecognizedHand`, `PoseLandmarkResult`/`PoseLandmarks`, `HolisticResult`, `SegmentationResult`/`SegmentationMask`, `ObjectDetectionResult`, `ClassificationResult`. |
| Nama | `HandLandmark`, `PoseLandmark`, `FaceKeypoint`, `Connections` (tangan, pose, kontur wajah). |
| Streaming | `LiveStreamProcessor<T>`, `LiveStreamProcessorOptions`, `LiveFrameResult<T>`, `LiveStreamStats`. |
| `Processing.*` | `SsdAnchors`, `SsdDetector`/`SsdDetectorSpec`, `DetectionDecoder`, `RawDetection`, `NonMaxSuppression`, `RoiCalculator`, `OneEuroFilter`, `LandmarkSmoother`, `Labels`. |
| `Graph.*` | `VisionTaskNode<TTask,TResult>`, `InferenceNode`, `VisionCalculators.AddVisionCalculators`. |
| `ModelLoader` | Me-resolve/memuat model dari `BaseOptions`. |

## Tambahan

- **MediaPipeNet.Visualization** — `LandmarkDrawer`, `BoundingBoxDrawer`, `SegmentationMaskOverlay`, `ResultRenderer`, `DrawingStyle`.
- **MediaPipeNet.Video.OpenCv** — `WebcamFrameSource`, `VideoFileFrameSource`, `OpenCvFrameSource`.
- **MediaPipeNet.Extensions.DI** — `AddMediaPipeNet`, `Add{Task}`, `MediaPipeNetOptions`, `IMediaPipeNetBuilder`.

Untuk menghasilkan referensi HTML lengkap: `dotnet tool install -g docfx` lalu `docfx metadata` atas
`src/*/*.csproj` (file dokumentasi XML dihasilkan setiap build).
