# Tasks

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/task.md)

Every task follows the same pattern:

| Mode | Method | Notes |
|---|---|---|
| `RunningMode.Image` (default) | `Detect(image)` / `DetectAsync(image)` | Independent images, thread-safe. |
| `RunningMode.Video` | `DetectForVideo(image, timestampMs)` | Frames in order, increasing timestamps; tracking + smoothing. |
| `RunningMode.LiveStream` | `DetectLiveStream(image, timestampMs)` → `ResultCallback` | Asynchronous; returns `false` when a frame was dropped. |

(`GestureRecognizer` uses `Recognize…`, `ImageSegmenter` `Segment…`, `ImageClassifier` `Classify…`.)
All methods accept an optional `ImageProcessingOptions(RegionOfInterest, RotationDegrees)`.

---

## FaceDetector

![Face detection](../images/gallery-faces.png)

BlazeFace short-range (128×128): bounding box + 6 keypoints per face (`FaceKeypoint`: right/left eye, nose tip,
mouth center, right/left ear tragion). Best for faces within ~2 m.

| Option | Default | Meaning |
|---|---|---|
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
| `SmoothLandmarks` | true | One-Euro smoothing in video/live mode (single face). |

```csharp
var face = landmarker.Detect(image).Faces[0];
float smile = face.GetBlendshape("mouthSmileLeft");
var noseTip = face.Landmarks[1];
Connections.FaceContours   // oval, lips, eyes, eyebrows, irises — for drawing
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

BlazePose detector (224×224) → alignment-point ROI → BlazePose GHUM landmark model (256×256, Lite or Full).

| Option | Default | |
|---|---|---|
| `Model` | `PoseModel.Lite` | `Lite` (5.5 MB) or `Full` (12.8 MB, more accurate). |
| `NumPoses` | 1 | Maximum people. |
| `MinPoseDetectionConfidence` / `MinPosePresenceConfidence` / `MinTrackingConfidence` | 0.5 | |
| `OutputSegmentationMasks` | false | Person mask for the whole image. |
| `SmoothLandmarks` | true | One-Euro smoothing in video/live mode. |

Each landmark carries `Visibility` and `Presence`. `pose[PoseLandmark.LeftWrist]`, `WorldLandmarks` (meters, hips
origin), `SegmentationMask`, `Connections.Pose`.

## HolisticLandmarker

![Holistic](../images/gallery-holistic.png)

Pose, face mesh and both hands of one person. The three pipelines run in parallel; hands are assigned to the
body's left/right wrist; when the face detector misses a small or profile face, the face ROI is derived from the
pose's face landmarks. `HolisticResult(Pose, Face, LeftHand, RightHand)` — left/right are the **person's**
anatomical sides.

## ImageSegmenter (selfie)

![Segmentation](../images/gallery-segment.png)

Per-pixel person probability (`SegmentationMask`, same size as the image). `TemporalSmoothing` (default 0.3) blends
masks across frames in video/live mode. Use `MediaPipeNet.Visualization.SegmentationMaskOverlay` to blur or
replace the background, `mask.Coverage()` for the foreground fraction, `mask.ToBytes(threshold)` for an 8-bit mask.

## ObjectDetector

![Objects](../images/gallery-objects.png)

EfficientDet-Lite0 (320×320), 80 COCO classes.

| Option | Default | |
|---|---|---|
| `ScoreThreshold` | 0.3 | Minimum score. |
| `MaxResults` | -1 | Maximum detections. |
| `NmsThreshold` | 0.5 | IoU for (class-agnostic) suppression. |
| `CategoryAllowlist` / `CategoryDenylist` | null | Filter labels, e.g. `{ "person", "car" }`. |

## ImageClassifier

![Classification](../images/gallery-classify.png)

EfficientNet-Lite0 (224×224), 1000 ImageNet classes. Options: `MaxResults` (5), `ScoreThreshold`,
`CategoryAllowlist`, `CategoryDenylist`.

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
