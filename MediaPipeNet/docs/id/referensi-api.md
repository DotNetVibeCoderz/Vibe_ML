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
| `TensorWarp` | `ProjectToImage`, `ProjectChannelsToImage` (semua kanal tensor HWC dalam satu pass), `Resize`. |
| `IFrameSource`, `VideoFrame`, `ImageFileFrameSource`, `MemoryFrameSource` | Input video. |

## MediaPipeNet.Inference

| Tipe | |
|---|---|
| `OnnxModel` | `Load(path | bytes)`, `Inputs`, `Outputs` (`TensorSpec` dengan `ElementType` float32/int32/int64), `Provider`, `RentContext()`, `RunDynamic(DynamicTensor[])` untuk shape variabel, `FindInput`/`FindOutput`. |
| `InferenceContext` | `GetInput`, `GetInputInt32`, `GetInputInt64`, `Run`, `GetOutput`, `GetOutputInt32`, `UsesIoBinding`, `Dispose` (kembali ke pool). |
| `InferenceOptions`, `ExecutionProvider`, `ModelPrecision` | `Provider`, `DeviceId`, `IntraOpThreads`, `InterOpThreads`, `FallbackToCpu`, `EnableProfiling`, `MaxPooledContexts`, `UseIoBinding`, `Precision` (`Float32`, `Float16`, `Int8`). |
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

## Tambahan 0.2 / 0.3

### MediaPipeNet.Tasks (Tasks.Core) — assembly baru

`BaseOptions` dan `ModelLoader` dipindahkan ke sini dari `MediaPipeNet.Tasks.Vision` agar paket audio dan teks tidak
bergantung pada stack pengolahan gambar. **Breaking change:** tambahkan `using MediaPipeNet.Tasks;` di tempat Anda
menyebut tipe `BaseOptions` atau `ModelLoader` (sintaks properti seperti `new FaceDetectorOptions { BaseOptions = … }`
tidak terpengaruh).

| Tipe | |
|---|---|
| `BaseOptions` | Model store / direktori / path eksplisit, `InferenceOptions`, logger factory. |
| `ModelLoader` | `LoadAsync`, `ResolvePathAsync` (varian presisi → fallback float32), `SelectVariant`. |
| `ClassifierOptions` | `MaxResults`, `ScoreThreshold`, `CategoryAllowlist`, `CategoryDenylist`, `Select(scores, labels)`. |
| `CustomModels` | `LoadAsync(options, fallback, customPath)`, `ResolveLabels`, `CompanionFile`. |

### MediaPipeNet.Tasks.Vision

| Tipe | Baru |
|---|---|
| `FaceDetectorOptions.Model` / `FaceDetectorModel` | `ShortRange`, `FullRange`. |
| `FaceLandmarkerOptions.OutputFacialTransformationMatrixes` | `FaceLandmarks.FacialTransformationMatrix`, `GetTransformMatrix()`. |
| `PoseLandmarkerOptions.SmoothSegmentationMasks` | Penyempurnaan heatmap selalu aktif. |
| `ImageSegmenterOptions` | `Model` (`SegmenterModel`), `OutputConfidenceMasks`, `OutputCategoryMask`. |
| `SegmentationResult` | `ConfidenceMasks`, `CategoryMask`, `Labels`, `GetConfidenceMask(label)`; `ConfidenceMask` = foreground. |
| `CategoryMask` | Indeks kategori satu byte per piksel, `Histogram()`, `FromConfidenceMasks`. |
| `InteractiveSegmenter`, `RegionOfInterest` | MagicTouch; `FromKeypoint`, `FromScribble`. |
| `ImageEmbedder`, `ImageEmbeddingResult` | Embedding MobileNet V3; `CosineSimilarity`. |
| `VisionTaskBase.ProcessBatch` / `ProcessBatchAsync` | Batch paralel, hasil berurutan. |
| `ImageClassifierOptions`, `ObjectDetectorOptions` | `ModelPath`, `Labels`. `GestureRecognizerOptions`: `ClassifierModelPath`, `Labels`. |
| `Processing.FaceGeometry`, `HeatmapRefinement`, `SegmentationSmoother`, `RoiTracking` | Kalkulator MediaPipe hasil port, publik untuk pipeline Anda. |
| `Graph.VisionCalculators` | `ImageEmbedderCalculator`; opsi `model` pada `ImageSegmenterCalculator`. |

### MediaPipeNet (Core)

`Embedding(Values, QuantizedValues, HeadIndex, HeadName)` dengan `CosineSimilarity` dan `FromTensor`.

### MediaPipeNet.Tasks.Audio / MediaPipeNet.Tasks.Text

Lihat [Task audio & teks](audio-dan-teks.md): `AudioData`, `AudioClassifier`, `VoiceActivityDetector`,
`AudioRunningMode`; `TextClassifier`, `TextEmbedder`, `LanguageDetector`, `Tokenizers.BertTokenizer`,
`Tokenizers.RegexTokenizer`, `Tokenizers.NGramHasher`.

### MediaPipeNet.Framework

`NodeBuilder.In(tag, stream, backEdge)`, `NodeBuilder.OnExecutor`, `GraphBuilder.AddExecutor`, `AddSubgraph`,
`GraphOptions.EnableTracing` / `MaxTraceEvents`, `CalculatorGraph.GetTraceEvents` / `WriteChromeTrace` / `IsTracing`,
`GraphTraceEvent`, `PreviousLoopbackNode<T>`, `GraphConfig.ToBuilder`, `Executors`, `EnableTracing`,
`NodeConfig.BackEdgeTags` / `Executor`, `CalculatorRegistry.RegisterSubgraph` / `TryGetSubgraph`.

### Stabilitas API

Setiap library yang dirilis mencatat permukaan publiknya di `PublicAPI.Shipped.txt` / `PublicAPI.Unshipped.txt`
(Microsoft.CodeAnalysis.PublicApiAnalyzers). Perubahan API publik yang tidak tercatat menggagalkan build
(RS0016/RS0017), sehingga setiap perubahan permukaan disengaja dan dapat direview;
`python tools/update_public_api.py` mencatat penambahan yang disengaja.
