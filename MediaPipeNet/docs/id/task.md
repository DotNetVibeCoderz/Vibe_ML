# Task

> 🇬🇧 [Read in English](../en/tasks.md)

Semua task mengikuti pola yang sama:

| Mode | Method | Catatan |
|---|---|---|
| `RunningMode.Image` (default) | `Detect(image)` / `DetectAsync(image)` | Gambar independen, thread-safe. |
| `RunningMode.Video` | `DetectForVideo(image, timestampMs)` | Frame berurutan, timestamp naik; tracking + smoothing. |
| `RunningMode.LiveStream` | `DetectLiveStream(image, timestampMs)` → `ResultCallback` | Asinkron; mengembalikan `false` bila frame dibuang. |

(`GestureRecognizer` memakai `Recognize…`, `ImageSegmenter` `Segment…`, `ImageClassifier` `Classify…`.)
Semua method menerima `ImageProcessingOptions(RegionOfInterest, RotationDegrees)` opsional.

---

## FaceDetector

![Deteksi wajah](../images/gallery-faces.png)

BlazeFace short-range (128×128): bounding box + 6 keypoint per wajah (`FaceKeypoint`: mata kanan/kiri, ujung
hidung, tengah mulut, tragus telinga kanan/kiri). Optimal untuk wajah dalam jarak ~2 m.

| Opsi | Default | Arti |
|---|---|---|
| `MinDetectionConfidence` | 0.5 | Skor wajah minimum. |
| `MinSuppressionThreshold` | 0.3 | IoU di atas nilai ini membuat deteksi yang bertumpuk digabung (weighted NMS). |
| `MaxResults` | -1 | Jumlah wajah maksimum (-1 = semua). |

Hasil: `FaceDetectionResult.Detections` → `Detection(BoundingBox px, Categories, Keypoints)`.

## FaceLandmarker (face mesh)

![Face mesh](../images/gallery-face-mesh.png)

478 landmark 3-D (468 mesh + 10 iris) dan opsional 52 blendshape bergaya ARKit. Pipeline: BlazeFace → ROI berotasi
(mata sejajar, ×1,5) → Face Mesh V2 (256×256) → model blendshape dari 146 landmark. Di mode video ROI berikutnya
diturunkan dari landmark saat ini, sehingga detektor hanya berjalan ketika wajah hilang.

| Opsi | Default | |
|---|---|---|
| `NumFaces` | 1 | Jumlah wajah maksimum. |
| `MinFaceDetectionConfidence` | 0.5 | Ambang detektor. |
| `MinFacePresenceConfidence` | 0.5 | Ambang presence model landmark. |
| `MinTrackingConfidence` | 0.5 | Di bawah nilai ini wajah dideteksi ulang (video/live). |
| `OutputFaceBlendshapes` | false | Hitung juga 52 blendshape. |
| `SmoothLandmarks` | true | Smoothing One-Euro di mode video/live (satu wajah). |

```csharp
var face = landmarker.Detect(image).Faces[0];
float senyum = face.GetBlendshape("mouthSmileLeft");
var ujungHidung = face.Landmarks[1];
Connections.FaceContours   // oval, bibir, mata, alis, iris — untuk menggambar
```

## HandLandmarker

![Tangan](../images/gallery-hands.png)

BlazePalm (192×192) → ROI tangan berotasi (×2,6, digeser ke arah jari) → model landmark tangan (224×224). Setiap
`HandLandmarks` memiliki `Handedness` ("Left"/"Right", konvensi MediaPipe mengasumsikan gambar selfie tercermin),
21 `Landmarks` ternormalisasi, 21 `WorldLandmarks` dalam meter, `PresenceScore`, dan `Roi`.

| Opsi | Default | |
|---|---|---|
| `NumHands` | 2 | Jumlah tangan maksimum. Di mode video detektor telapak tetap berjalan selama tangan yang dilacak lebih sedikit — gunakan 1 untuk kecepatan maksimum. |
| `MinHandDetectionConfidence` | 0.5 | Ambang detektor telapak. |
| `MinHandPresenceConfidence` | 0.5 | Ambang presence landmark. |
| `MinTrackingConfidence` | 0.5 | Deteksi ulang di bawah nilai ini (video/live). |

```csharp
var ujung = hand[HandLandmark.IndexFingerTip];
Connections.Hand   // 21 tulang
```

## GestureRecognizer

![Gestur](../images/gallery-gestures.png)

Landmark tangan → gesture embedder → canned classifier. Gestur: `None`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`,
`Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

| Opsi | Default | |
|---|---|---|
| `Hands` | `new HandLandmarkerOptions()` | Opsi tracking tangan. |
| `MinGestureScore` | 0 | Di bawah nilai ini gestur dilaporkan sebagai `None`. |
| `MaxResults` | 1 | Jumlah gestur per tangan (-1 = semua). |
| `CategoryAllowlist` | null | Batasi ke gestur tertentu. |

`GestureRecognitionResult.Hands[i].TopGesture`, `.Gestures`, `.Hand`. Landmark yang sudah ada juga bisa
diklasifikasi: `recognizer.Classify(handLandmarks, lebar, tinggi)`.

## PoseLandmarker

![Pose](../images/gallery-pose.png)

Detektor BlazePose (224×224) → ROI alignment-point → model landmark BlazePose GHUM (256×256, Lite atau Full).

| Opsi | Default | |
|---|---|---|
| `Model` | `PoseModel.Lite` | `Lite` (5,5 MB) atau `Full` (12,8 MB, lebih akurat). |
| `NumPoses` | 1 | Jumlah orang maksimum. |
| `MinPoseDetectionConfidence` / `MinPosePresenceConfidence` / `MinTrackingConfidence` | 0.5 | |
| `OutputSegmentationMasks` | false | Mask orang untuk seluruh gambar. |
| `SmoothLandmarks` | true | Smoothing One-Euro di mode video/live. |

Setiap landmark memiliki `Visibility` dan `Presence`. `pose[PoseLandmark.LeftWrist]`, `WorldLandmarks` (meter, titik
asal di pinggul), `SegmentationMask`, `Connections.Pose`.

## HolisticLandmarker

![Holistic](../images/gallery-holistic.png)

Pose, face mesh, dan kedua tangan satu orang. Ketiga pipeline berjalan paralel; tangan dipasangkan ke pergelangan
kiri/kanan tubuh; bila detektor wajah melewatkan wajah kecil atau menyamping, ROI wajah diturunkan dari landmark
wajah milik pose. `HolisticResult(Pose, Face, LeftHand, RightHand)` — kiri/kanan adalah sisi anatomis **orang
tersebut**.

## ImageSegmenter (selfie)

![Segmentasi](../images/gallery-segment.png)

Probabilitas orang per piksel (`SegmentationMask`, ukuran sama dengan gambar). `TemporalSmoothing` (default 0.3)
memadukan mask antar-frame di mode video/live. Gunakan `MediaPipeNet.Visualization.SegmentationMaskOverlay` untuk
blur atau mengganti latar, `mask.Coverage()` untuk porsi foreground, `mask.ToBytes(ambang)` untuk mask 8-bit.

## ObjectDetector

![Objek](../images/gallery-objects.png)

EfficientDet-Lite0 (320×320), 80 kelas COCO.

| Opsi | Default | |
|---|---|---|
| `ScoreThreshold` | 0.3 | Skor minimum. |
| `MaxResults` | -1 | Jumlah deteksi maksimum. |
| `NmsThreshold` | 0.5 | IoU untuk suppression (lintas kelas). |
| `CategoryAllowlist` / `CategoryDenylist` | null | Filter label, mis. `{ "person", "car" }`. |

## ImageClassifier

![Klasifikasi](../images/gallery-classify.png)

EfficientNet-Lite0 (224×224), 1000 kelas ImageNet. Opsi: `MaxResults` (5), `ScoreThreshold`,
`CategoryAllowlist`, `CategoryDenylist`.

---

## Region of interest dan rotasi

```csharp
// Hanya lihat setengah kanan frame, dan putar konten 90° sebelum dilihat model
// (mis. kamera dipasang miring). RotationDegrees harus kelipatan 90.
var roi = new ImageProcessingOptions(new NormalizedRect(0.75f, 0.5f, 0.5f, 1f), RotationDegrees: 90);
var result = detector.Detect(image, roi);     // hasil tetap dalam koordinat gambar asli
```

## JSON

Setiap hasil punya `ToJson(indented)`; `MediaPipeJson.Serialize/Deserialize` memakai camelCase, melewati nilai null,
dan menyandikan mask segmentasi secara ringkas sebagai float32 base64.
