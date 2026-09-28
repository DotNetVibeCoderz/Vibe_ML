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
| `TensorWarp` | `ProjectToImage`, `ProjectChannelsToImage` (all channels of an HWC tensor in one pass), `Resize`. |
| `IFrameSource`, `VideoFrame`, `ImageFileFrameSource`, `MemoryFrameSource` | Video input. |

## MediaPipeNet.Inference

| Type | |
|---|---|
| `OnnxModel` | `Load(path | bytes)`, `Inputs`, `Outputs` (`TensorSpec` with `ElementType` float32/int32/int64), `Provider`, `RentContext()`, `RunDynamic(DynamicTensor[])` for variable shapes, `FindInput`/`FindOutput`. |
| `InferenceContext` | `GetInput`, `GetInputInt32`, `GetInputInt64`, `Run`, `GetOutput`, `GetOutputInt32`, `UsesIoBinding`, `Dispose` (returns to pool). |
| `InferenceOptions`, `ExecutionProvider`, `ModelPrecision` | `Provider`, `DeviceId`, `IntraOpThreads`, `InterOpThreads`, `FallbackToCpu`, `EnableProfiling`, `MaxPooledContexts`, `UseIoBinding`, `Precision` (`Float32`, `Float16`, `Int8`). |
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

## 0.2 / 0.3 additions

### MediaPipeNet.Tasks (Tasks.Core) — new assembly

`BaseOptions` and `ModelLoader` moved here from `MediaPipeNet.Tasks.Vision` so that the audio and text packages do
not depend on the imaging stack. **Breaking change:** add `using MediaPipeNet.Tasks;` where you name `BaseOptions`
or `ModelLoader` (property syntax such as `new FaceDetectorOptions { BaseOptions = … }` is unaffected).

| Type | |
|---|---|
| `BaseOptions` | Model store / directory / explicit paths, `InferenceOptions`, logger factory. |
| `ModelLoader` | `LoadAsync`, `ResolvePathAsync` (precision variant → float32 fallback), `SelectVariant`. |
| `ClassifierOptions` | `MaxResults`, `ScoreThreshold`, `CategoryAllowlist`, `CategoryDenylist`, `Select(scores, labels)`. |
| `CustomModels` | `LoadAsync(options, fallback, customPath)`, `ResolveLabels`, `CompanionFile`. |

### MediaPipeNet.Tasks.Vision

| Type | New |
|---|---|
| `FaceDetectorOptions.Model` / `FaceDetectorModel` | `ShortRange`, `FullRange`. |
| `FaceLandmarkerOptions.OutputFacialTransformationMatrixes` | `FaceLandmarks.FacialTransformationMatrix`, `GetTransformMatrix()`. |
| `PoseLandmarkerOptions.SmoothSegmentationMasks` | Heatmap refinement is always on. |
| `ImageSegmenterOptions` | `Model` (`SegmenterModel`), `OutputConfidenceMasks`, `OutputCategoryMask`. |
| `SegmentationResult` | `ConfidenceMasks`, `CategoryMask`, `Labels`, `GetConfidenceMask(label)`; `ConfidenceMask` = foreground. |
| `CategoryMask` | Byte-per-pixel category indices, `Histogram()`, `FromConfidenceMasks`. |
| `InteractiveSegmenter`, `RegionOfInterest` | MagicTouch; `FromKeypoint`, `FromScribble`. |
| `FaceStylizer`, `FaceStylizerOptions`, `FaceStylizerResult` | Color-sketch stylizer; `StylizedImage`, `FaceAlignment`, `FaceRect`, `Composite(image)`, `ComputeFaceRect(landmarks, w, h)`. |
| `FaceLandmarkerOptions.LandmarksModel` | Choose the face mesh (`ModelCatalog.FaceLandmarksDetector` or `FaceLandmarksDetector192`). |
| `ImageEmbedder`, `ImageEmbeddingResult` | MobileNet V3 embeddings; `CosineSimilarity`. |
| `VisionTaskBase.ProcessBatch` / `ProcessBatchAsync` | Parallel batches, results in order. |
| `ImageClassifierOptions`, `ObjectDetectorOptions` | `ModelPath`, `Labels`. `GestureRecognizerOptions`: `ClassifierModelPath`, `Labels`. |
| `Processing.FaceGeometry`, `HeatmapRefinement`, `SegmentationSmoother`, `RoiTracking` | The ported MediaPipe calculators, public for your own pipelines. |
| `Graph.VisionCalculators` | `ImageEmbedderCalculator`; `ImageSegmenterCalculator` option `model`. |

### MediaPipeNet (Core)

`Embedding(Values, QuantizedValues, HeadIndex, HeadName)` with `CosineSimilarity` and `FromTensor`.

### MediaPipeNet.Tasks.Audio / MediaPipeNet.Tasks.Text

See [Audio & text tasks](audio-and-text.md): `AudioData`, `AudioClassifier`, `VoiceActivityDetector`,
`AudioRunningMode`; `TextClassifier`, `TextEmbedder`, `LanguageDetector`, `Tokenizers.BertTokenizer`,
`Tokenizers.RegexTokenizer`, `Tokenizers.NGramHasher`.

### MediaPipeNet.Framework

`NodeBuilder.In(tag, stream, backEdge)`, `NodeBuilder.OnExecutor`, `GraphBuilder.AddExecutor`, `AddSubgraph`,
`GraphOptions.EnableTracing` / `MaxTraceEvents`, `CalculatorGraph.GetTraceEvents` / `WriteChromeTrace` / `IsTracing`,
`GraphTraceEvent`, `PreviousLoopbackNode<T>`, `GraphConfig.ToBuilder`, `Executors`, `EnableTracing`,
`NodeConfig.BackEdgeTags` / `Executor`, `CalculatorRegistry.RegisterSubgraph` / `TryGetSubgraph`.

### API stability

Every shippable library tracks its public surface in `PublicAPI.Shipped.txt` / `PublicAPI.Unshipped.txt`
(Microsoft.CodeAnalysis.PublicApiAnalyzers). An unrecorded public API change fails the build (RS0016/RS0017), so
every change to the surface is deliberate and reviewable; `python tools/update_public_api.py` records intentional
additions.
