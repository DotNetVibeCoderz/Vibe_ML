# Tasks

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/task.md)

Every task follows the same pattern:

| Mode | Method | Notes |
|---|---|---|
| `RunningMode.Image` (default) | `Detect(image)` / `DetectAsync(image)` | Independent images, thread-safe. |
| `RunningMode.Video` | `DetectForVideo(image, timestampMs)` | Frames in order, increasing timestamps; tracking + smoothing. |
| `RunningMode.LiveStream` | `DetectLiveStream(image, timestampMs)` → `ResultCallback` | Asynchronous; returns `false` when a frame was dropped. |

(`GestureRecognizer` uses `Recognize…`, `ImageSegmenter` `Segment…`, `ImageClassifier` `Classify…`,
`ImageEmbedder` `Embed…`.) All methods accept an optional `ImageProcessingOptions(RegionOfInterest, RotationDegrees)`.

**Batches** — every vision task also has `ProcessBatch(images, options, maxDegreeOfParallelism)` /
`ProcessBatchAsync(…)` for offline workloads: the images run in parallel on the shared model (each worker rents its
own pooled buffers) and the results come back in input order — about 2× the throughput of a sequential loop on a
4-core CPU. Image-mode calls are thread-safe; tracking state is only touched in video / live-stream mode.

Audio and text tasks are described in [Audio & text tasks](audio-and-text.md); your own models in
[Custom models](custom-models.md).

---

## FaceDetector

![Face detection](../images/gallery-faces.png)

BlazeFace: bounding box + 6 keypoints per face (`FaceKeypoint`: right/left eye, nose tip, mouth center,
right/left ear tragion). The short-range model (128×128) suits faces within ~2 m; the **full-range** model
(192×192, one 48×48 anchor grid) finds faces up to ~5 m away — group photos, rooms, surveillance.

| Option | Default | Meaning |
|---|---|---|
| `Model` | `ShortRange` | `FaceDetectorModel.ShortRange` or `FaceDetectorModel.FullRange`. |
| `MinDetectionConfidence` | 0.5 | Minimum face score. |
| `MinSuppressionThreshold` | 0.3 | IoU above which overlapping detections are merged (weighted NMS). |
| `MaxResults` | -1 | Maximum faces (-1 = all). |

Result: `FaceDetectionResult.Detections` → `Detection(BoundingBox px, Categories, Keypoints)`.

## FaceLandmarker (face mesh)

![Face mesh](../images/gallery-face-mesh.png)

478 3-D landmarks (468 mesh + 10 iris) and optionally 52 ARKit-style blendshapes. Pipeline: BlazeFace → rotated
ROI (eyes level, ×1.5) → Face Mesh V2 (256×256) → blendshape model on 146 landmarks. In video mode the next ROI is
derived from the current landmarks, so the detector only runs when a face is lost.

| Option | Default | |
|---|---|---|
| `NumFaces` | 1 | Maximum faces. |
| `MinFaceDetectionConfidence` | 0.5 | Detector threshold. |
| `MinFacePresenceConfidence` | 0.5 | Landmark model presence threshold. |
| `MinTrackingConfidence` | 0.5 | Below this, the face is re-detected (video/live). |
| `OutputFaceBlendshapes` | false | Also compute the 52 blendshapes. |
| `OutputFacialTransformationMatrixes` | false | Also compute each face's 4×4 facial transformation matrix. |
| `SmoothLandmarks` | true | One-Euro smoothing in video/live mode (single face). |

```csharp
var face = landmarker.Detect(image).Faces[0];
float smile = face.GetBlendshape("mouthSmileLeft");
var noseTip = face.Landmarks[1];
Connections.FaceContours   // oval, lips, eyes, eyebrows, irises — for drawing
```

**Facial transformation matrix (head pose for AR).** With `OutputFacialTransformationMatrixes = true`,
`FaceLandmarks.FacialTransformationMatrix` holds the rigid pose and scale that map MediaPipe's canonical 3-D face
model into camera space — a port of MediaPipe's geometry pipeline (perspective camera, 63° vertical field of view,
weighted orthogonal Procrustes on 33 stable landmarks). The 16 values are row-major in MediaPipe's layout
(column-vector convention, translation in centimeters in the last column); `face.GetTransformMatrix()` returns the
same matrix as a `System.Numerics.Matrix4x4` in .NET's row-vector convention, ready for `Vector3.Transform`.
On the portrait fixture it matches MediaPipe within 2° of rotation and 1.5 cm of translation.

```csharp
using var landmarker = FaceLandmarker.Create(new() { OutputFacialTransformationMatrixes = true });
var pose = landmarker.Detect(image).Faces[0].GetTransformMatrix()!.Value;
var noseInCamera = Vector3.Transform(new Vector3(0, -0.5f, 7.5f), pose);   // canonical-model cm → camera cm
```

## HandLandmarker

![Hands](../images/gallery-hands.png)

BlazePalm (192×192) → rotated hand ROI (×2.6, shifted toward the fingers) → hand landmark model (224×224).
Each `HandLandmarks` has `Handedness` ("Left"/"Right", MediaPipe's convention assumes a mirrored selfie image),
21 normalized `Landmarks`, 21 `WorldLandmarks` in meters, `PresenceScore` and the `Roi`.

| Option | Default | |
|---|---|---|
| `NumHands` | 2 | Maximum hands. In video mode the palm detector runs while fewer hands are tracked — use 1 for maximum speed. |
| `MinHandDetectionConfidence` | 0.5 | Palm detector threshold. |
| `MinHandPresenceConfidence` | 0.5 | Landmark presence threshold. |
| `MinTrackingConfidence` | 0.5 | Re-detect below this (video/live). |

```csharp
var tip = hand[HandLandmark.IndexFingerTip];
Connections.Hand   // the 21 bones
```

## GestureRecognizer

![Gestures](../images/gallery-gestures.png)

Hand landmarks → gesture embedder → canned classifier. Gestures: `None`, `Closed_Fist`, `Open_Palm`,
`Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

| Option | Default | |
|---|---|---|
| `Hands` | `new HandLandmarkerOptions()` | Hand tracking options. |
| `MinGestureScore` | 0 | Below this the gesture is reported as `None`. |
| `MaxResults` | 1 | Gestures listed per hand (-1 = all). |
| `CategoryAllowlist` | null | Restrict to these gestures. |

`GestureRecognitionResult.Hands[i].TopGesture`, `.Gestures`, `.Hand`. You can also classify landmarks you already
have: `recognizer.Classify(handLandmarks, width, height)`.

## PoseLandmarker

![Pose](../images/gallery-pose.png)

BlazePose detector (224×224) → alignment-point ROI → BlazePose GHUM landmark model (256×256, Lite or Full). As
in MediaPipe, the regressed landmarks are **refined with the model's 64×64 heatmaps** (sigmoid-weighted centroid of
a 7×7 window around each landmark) before they are projected back to the image — this brings the landmarks 2.7×
closer to MediaPipe's than regression alone.

| Option | Default | |
|---|---|---|
| `Model` | `PoseModel.Lite` | `Lite` (5.5 MB) or `Full` (12.8 MB, more accurate). |
| `NumPoses` | 1 | Maximum people. |
| `MinPoseDetectionConfidence` / `MinPosePresenceConfidence` / `MinTrackingConfidence` | 0.5 | |
| `OutputSegmentationMasks` | false | Person mask for the whole image. |
| `SmoothLandmarks` | true | One-Euro smoothing in video/live mode. |
| `SmoothSegmentationMasks` | true | MediaPipe's segmentation smoothing on the person mask in video/live mode. |

Each landmark carries `Visibility` and `Presence`. `pose[PoseLandmark.LeftWrist]`, `WorldLandmarks` (meters, hips
origin), `SegmentationMask`, `Connections.Pose`.

## HolisticLandmarker

![Holistic](../images/gallery-holistic.png)

Pose, face mesh and both hands of one person, following MediaPipe's holistic graph:

1. **Pose** (BlazePose, Lite or Full) finds the person.
2. **Face** — the ROI comes from the pose's face landmarks (×3), the face detector runs inside it and its
   keypoints give the face mesh ROI. A face too small for the detector (a few dozen pixels) falls back to the
   pose's face landmarks — something MediaPipe does not do — still gated by the mesh's presence score.
3. **Hands** — for each wrist whose visibility is above 0.1, the ROI is built from the pose's wrist, index and
   pinky landmarks (×2.7), refined by the **hand ROI refinement model** (256×256, two points: wrist and middle
   finger), and the hand landmark model runs on it. Hand world landmarks are re-anchored at the pose's world wrist.
4. In video/live mode each part keeps its previous ROI while it agrees with the new one (MediaPipe's
   `RoiTrackingCalculator`: rotation, translation and scale limits, previous landmarks inside the new ROI), which
   keeps crops steady from frame to frame.

The face and both hands run in parallel. `HolisticResult(Pose, Face, LeftHand, RightHand)` — left/right are the
**person's** anatomical sides. On the pose fixture both hands match MediaPipe within 0.01 (normalized).

## ImageSegmenter

![Segmentation](../images/gallery-segment.png)

![Selfie multiclass](../images/segment-multiclass.jpg)

| `Model` | Input | Categories |
|---|---|---|
| `Selfie` (default) | 256×256 | one person mask |
| `SelfieMulticlass` | 256×256 | background, hair, body-skin, face-skin, clothes, others |
| `Hair` | 512×512 (RGBA) | background, hair |
| `DeepLabV3` | 257×257 | the 21 PASCAL VOC classes (person, cat, dog, car, …) |

| Option | Default | |
|---|---|---|
| `OutputConfidenceMasks` | true | One probability mask per category (`SegmentationResult.ConfidenceMasks`). |
| `OutputCategoryMask` | false | The most likely category per pixel (`CategoryMask`, one byte per pixel; 255 = unlabeled). |
| `TemporalSmoothing` | 0.3 | MediaPipe's uncertainty-weighted smoothing across frames (video/live). |

`result.ConfidenceMask` is the foreground: the only mask of the selfie model, otherwise 1 − P(background).
`result.GetConfidenceMask("hair")` picks a category by name, `result.Labels` lists them,
`result.CategoryMask.Histogram()` gives the share of each category. Use
`MediaPipeNet.Visualization.SegmentationMaskOverlay` to blur or replace the background or to colorize categories
(`OverlayCategories`).

```csharp
using var segmenter = ImageSegmenter.Create(new() { Model = SegmenterModel.SelfieMulticlass, OutputCategoryMask = true });
var result = segmenter.Segment(image);
Console.WriteLine($"hair covers {result.GetConfidenceMask("hair")!.Coverage():P1}");
```

## InteractiveSegmenter

![Interactive segmentation](../images/gallery-interactive.png)

MagicTouch (512×512): segments the object under a point or a scribble — "tap to select". Image mode only, like
MediaPipe's. The hint is rendered into the model's fourth input channel exactly as MediaPipe does (a disk whose
thickness scales with the image).

```csharp
using var segmenter = InteractiveSegmenter.Create(new() { OutputCategoryMask = true });
var cutout = segmenter.Segment(image, RegionOfInterest.FromKeypoint(0.62f, 0.5f)).ConfidenceMask;
var scribble = RegionOfInterest.FromScribble([new(0.3f, 0.6f), new(0.35f, 0.62f), new(0.4f, 0.6f)]);
```

## ImageEmbedder

![Image embedding](../images/gallery-embed.png)

MobileNet V3 small (224×224): a 1024-D feature vector per image for visual search, clustering and
de-duplication. Options: `L2Normalize` (false, as in MediaPipe), `Quantize` (int8 values in
`Embedding.QuantizedValues`). Compare with `ImageEmbedder.CosineSimilarity(a, b)`: burger vs. a crop of the same
burger ≈ 0.92, burger vs. cat ≈ 0.05 (MediaPipe: 0.920 and 0.048).

## ObjectDetector

![Objects](../images/gallery-objects.png)

EfficientDet-Lite0 (320×320), 80 COCO classes. Custom EfficientDet-Lite models of any input size can be loaded with
`ModelPath` + `Labels` — see [Custom models](custom-models.md).

| Option | Default | |
|---|---|---|
| `ScoreThreshold` | 0.3 | Minimum score. |
| `MaxResults` | -1 | Maximum detections. |
| `NmsThreshold` | 0.5 | IoU for (class-agnostic) suppression. |
| `CategoryAllowlist` / `CategoryDenylist` | null | Filter labels, e.g. `{ "person", "car" }`. |

## ImageClassifier

![Classification](../images/gallery-classify.png)

EfficientNet-Lite0 (224×224), 1000 ImageNet classes. Options: `MaxResults` (5), `ScoreThreshold`,
`CategoryAllowlist`, `CategoryDenylist`, and `ModelPath` / `Labels` for [your own classifier](custom-models.md).

---

## Regions of interest and rotation

```csharp
// Only look at the right half of the frame, and rotate the content by 90° before the model sees it
// (e.g. a camera mounted sideways). RotationDegrees must be a multiple of 90.
var roi = new ImageProcessingOptions(new NormalizedRect(0.75f, 0.5f, 0.5f, 1f), RotationDegrees: 90);
var result = detector.Detect(image, roi);     // results are still in original full-image coordinates
```

## JSON

Every result has `ToJson(indented)`; `MediaPipeJson.Serialize/Deserialize` use camelCase, skip nulls and encode
segmentation masks compactly as base64 float32.
