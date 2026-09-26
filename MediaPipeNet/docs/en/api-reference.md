# API reference (overview)

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/referensi-api.md)

Every public type carries XML documentation (IntelliSense). This page maps the surface.

## MediaPipeNet (Core)

| Type | |
|---|---|
| `Timestamp` | Microsecond packet timestamp; `FromMilliseconds`, `Unset`, `PreStream`, `Min`, `Max`, `PostStream`, `Done`. |
| `NormalizedLandmark(X, Y, Z, Visibility?, Presence?)` | Landmark in [0,1] image coordinates; `ToPixel`. |
| `Landmark(X, Y, Z, …)` | World landmark in meters. |
| `NormalizedKeypoint`, `Category`, `Detection` | Detection results. |
| `RectF`, `NormalizedRect`, `LetterboxPadding`, `ImageSize` | Geometry; `RectF.IntersectionOverUnion`. |
| `RunningMode` | `Image`, `Video`, `LiveStream`. |
| `Angles` | `NormalizeRadians`, `ComputeRotation`. |
| `MediaPipeException`, `ModelNotFoundException` | Errors. |
| `Diagnostics.MediaPipeTelemetry`, `Diagnostics.FrameRateCounter` | Meter/ActivitySource and FPS counter. |
| `Serialization.MediaPipeJson` | Shared JSON options. |

## MediaPipeNet.Imaging

| Type | |
|---|---|
| `MPImage` | Pooled RGBA image: `Load`/`LoadAsync` (file, stream, bytes), `FromImage`, `FromPixelData`, `CopyFrom`, `ToImage`, `CopyTo`, `Clone`, `FlipHorizontal`, `SaveAsPng`/`SaveAsJpeg`. |
| `PixelFormat` | `Rgba32`, `Bgra32`, `Rgb24`, `Bgr24`, `Gray8`. |
| `ImageToTensor`, `ImageToTensorOptions`, `BorderMode` | ROI → NHWC float tensor. |
| `TensorMapping` | `TensorToImage`, `ImageToTensor`, `ScaleZ`. |
| `TensorWarp` | `ProjectToImage`, `Resize`. |
| `IFrameSource`, `VideoFrame`, `ImageFileFrameSource`, `MemoryFrameSource` | Video input. |

## MediaPipeNet.Inference

| Type | |
|---|---|
| `OnnxModel` | `Load(path | bytes)`, `Inputs`, `Outputs`, `Provider`, `RentContext()`. |
| `InferenceContext` | `GetInput`, `Run`, `GetOutput`, `Dispose` (returns to pool). |
| `InferenceOptions`, `ExecutionProvider` | `Provider`, `DeviceId`, `IntraOpThreads`, `InterOpThreads`, `FallbackToCpu`, `EnableProfiling`, `MaxPooledContexts`. |
| `ExecutionProviderSelector` | `GetAvailableProviders`, `GetRuntimeVersion`, `GetCandidates`, `CreateSessionOptions`. |
| `Models.ModelCatalog`, `Models.ModelDescriptor` | The 13 models. |
| `Models.ModelStore` | `Default`, `CreateDefault`, `GetModelPath(Async)`, `FindLocal`, `EnsureModelsAsync`, `ComputeSha256Async`. |
| `Models.IModelProvider` + `DirectoryModelProvider`, `BundledModelProvider`, `EmbeddedResourceModelProvider`, `HttpModelProvider`, `NuGetModelProvider` | Model sources. |

## MediaPipeNet.Framework

| Type | |
|---|---|
| `GraphBuilder`, `NodeBuilder`, `GraphOptions`, `GraphInputOptions`, `QueueOverflowPolicy` | Declaring graphs. |
| `CalculatorGraph` | `StartAsync`, `AddPacket`, `CloseInputStream(s)`, `ObserveOutputStream`, `CreateOutputStreamPoller`, `WaitUntilIdleAsync`, `WaitUntilDoneAsync`, `CloseAsync`, `Cancel`, `ToMermaid`, `GetEdges`, `State`, `Error`, `DroppedPackets`, `InFlightCount`. |
| `ICalculatorNode`, `CalculatorNode`, `CalculatorContract`, `CalculatorContext`, `InputPolicy`, `PortSpec` | Writing nodes. |
| `Packet`, `Packet<T>` | Timestamped values. |
| `Nodes.*` | `PassThroughNode`, `LambdaNode`, `AsyncLambdaNode`, `CombineNode`, `SinkNode`, `PacketThinnerNode`, `PacketCounterNode`. |
| `Config.GraphConfig`, `Config.NodeConfig`, `Config.CalculatorRegistry` | `.pbtxt` configs. |
| `GraphValidationException` | Build/start errors. |

## MediaPipeNet.Tasks.Vision

| Type | |
|---|---|
| Tasks | `FaceDetector`, `FaceLandmarker`, `HandLandmarker`, `GestureRecognizer`, `PoseLandmarker`, `HolisticLandmarker`, `ImageSegmenter`, `ObjectDetector`, `ImageClassifier` (each with `Create`, `CreateAsync`, mode-specific methods, `ResetTracking`, `DroppedFrames`, `Dispose`). |
| Options | `…Options` records deriving from `VisionTaskOptions<TResult>`; `BaseOptions`; `ImageProcessingOptions`; `PoseModel`. |
| Results | `FaceDetectionResult`, `FaceLandmarkResult`/`FaceLandmarks`, `HandLandmarkResult`/`HandLandmarks`, `GestureRecognitionResult`/`RecognizedHand`, `PoseLandmarkResult`/`PoseLandmarks`, `HolisticResult`, `SegmentationResult`/`SegmentationMask`, `ObjectDetectionResult`, `ClassificationResult`. |
| Names | `HandLandmark`, `PoseLandmark`, `FaceKeypoint`, `Connections` (Hand, Pose, face contours). |
| Streaming | `LiveStreamProcessor<T>`, `LiveStreamProcessorOptions`, `LiveFrameResult<T>`, `LiveStreamStats`. |
| `Processing.*` | `SsdAnchors`, `SsdDetector`/`SsdDetectorSpec`, `DetectionDecoder`, `RawDetection`, `NonMaxSuppression`, `RoiCalculator`, `OneEuroFilter`, `LandmarkSmoother`, `Labels`. |
| `Graph.*` | `VisionTaskNode<TTask,TResult>`, `InferenceNode`, `VisionCalculators.AddVisionCalculators`. |
| `ModelLoader` | Resolve/load models from `BaseOptions`. |

## Add-ons

- **MediaPipeNet.Visualization** — `LandmarkDrawer`, `BoundingBoxDrawer`, `SegmentationMaskOverlay`, `ResultRenderer`, `DrawingStyle`.
- **MediaPipeNet.Video.OpenCv** — `WebcamFrameSource`, `VideoFileFrameSource`, `OpenCvFrameSource`.
- **MediaPipeNet.Extensions.DI** — `AddMediaPipeNet`, `Add{Task}`, `MediaPipeNetOptions`, `IMediaPipeNetBuilder`.

To generate a full HTML reference: `dotnet tool install -g docfx` then `docfx metadata` over `src/*/*.csproj`
(the XML documentation files are produced by every build).
