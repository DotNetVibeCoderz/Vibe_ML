# Task

> 🇬🇧 [Read in English](../en/tasks.md)

Semua task mengikuti pola yang sama:

| Mode | Method | Catatan |
|---|---|---|
| `RunningMode.Image` (default) | `Detect(image)` / `DetectAsync(image)` | Gambar independen, thread-safe. |
| `RunningMode.Video` | `DetectForVideo(image, timestampMs)` | Frame berurutan, timestamp naik; tracking + smoothing. |
| `RunningMode.LiveStream` | `DetectLiveStream(image, timestampMs)` → `ResultCallback` | Asinkron; mengembalikan `false` bila frame dibuang. |

(`GestureRecognizer` memakai `Recognize…`, `ImageSegmenter` `Segment…`, `ImageClassifier` `Classify…`,
`ImageEmbedder` `Embed…`, `FaceStylizer` `Stylize…`.) Semua method menerima `ImageProcessingOptions(RegionOfInterest, RotationDegrees)` opsional.

**Batch** — setiap task vision juga punya `ProcessBatch(images, options, maxDegreeOfParallelism)` /
`ProcessBatchAsync(…)` untuk beban offline: gambar diproses paralel pada model yang sama (tiap worker menyewa buffer
pool sendiri) dan hasilnya kembali sesuai urutan input — sekitar 2× throughput loop berurutan di CPU 4 core.
Pemanggilan mode Image thread-safe; state tracking hanya disentuh di mode video / live-stream.

Task audio dan teks dijelaskan di [Task audio & teks](audio-dan-teks.md); model Anda sendiri di
[Model kustom](model-kustom.md).

---

## FaceDetector

![Deteksi wajah](../images/gallery-faces.png)

BlazeFace: bounding box + 6 keypoint per wajah (`FaceKeypoint`: mata kanan/kiri, ujung hidung, tengah mulut,
tragus telinga kanan/kiri). Model short-range (128×128) cocok untuk wajah dalam jarak ~2 m; model **full-range**
(192×192, satu grid anchor 48×48) menemukan wajah hingga ~5 m — foto grup, ruangan, pemantauan.

| Opsi | Default | Arti |
|---|---|---|
| `Model` | `ShortRange` | `FaceDetectorModel.ShortRange` atau `FaceDetectorModel.FullRange`. |
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
| `OutputFacialTransformationMatrixes` | false | Hitung juga matriks transformasi wajah 4×4 tiap wajah. |
| `SmoothLandmarks` | true | Smoothing One-Euro di mode video/live (satu wajah). |

```csharp
var face = landmarker.Detect(image).Faces[0];
float senyum = face.GetBlendshape("mouthSmileLeft");
var ujungHidung = face.Landmarks[1];
Connections.FaceContours   // oval, bibir, mata, alis, iris — untuk menggambar
```

**Matriks transformasi wajah (pose kepala untuk AR).** Dengan `OutputFacialTransformationMatrixes = true`,
`FaceLandmarks.FacialTransformationMatrix` berisi pose rigid dan skala yang memetakan model wajah 3-D kanonis
MediaPipe ke ruang kamera — port geometry pipeline MediaPipe (kamera perspektif, field of view vertikal 63°, weighted
orthogonal Procrustes pada 33 landmark stabil). Ke-16 nilai tersusun row-major seperti MediaPipe (konvensi vektor
kolom, translasi dalam sentimeter di kolom terakhir); `face.GetTransformMatrix()` mengembalikan matriks yang sama
sebagai `System.Numerics.Matrix4x4` dengan konvensi vektor baris .NET, siap untuk `Vector3.Transform`. Pada fixture
portrait hasilnya cocok dengan MediaPipe dalam selisih rotasi 2° dan translasi 1,5 cm.

```csharp
using var landmarker = FaceLandmarker.Create(new() { OutputFacialTransformationMatrixes = true });
var pose = landmarker.Detect(image).Faces[0].GetTransformMatrix()!.Value;
var hidungDiKamera = Vector3.Transform(new Vector3(0, -0.5f, 7.5f), pose);   // cm model kanonis → cm kamera
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
Seperti MediaPipe, landmark hasil regresi **disempurnakan dengan heatmap 64×64 model** (centroid berbobot sigmoid
pada jendela 7×7 di sekitar tiap landmark) sebelum diproyeksikan ke gambar — membuat landmark 2,7× lebih dekat ke
hasil MediaPipe dibanding regresi saja.

| Opsi | Default | |
|---|---|---|
| `Model` | `PoseModel.Lite` | `Lite` (5,5 MB) atau `Full` (12,8 MB, lebih akurat). |
| `NumPoses` | 1 | Jumlah orang maksimum. |
| `MinPoseDetectionConfidence` / `MinPosePresenceConfidence` / `MinTrackingConfidence` | 0.5 | |
| `OutputSegmentationMasks` | false | Mask orang untuk seluruh gambar. |
| `SmoothLandmarks` | true | Smoothing One-Euro di mode video/live. |
| `SmoothSegmentationMasks` | true | Segmentation smoothing MediaPipe pada mask orang di mode video/live. |

Setiap landmark memiliki `Visibility` dan `Presence`. `pose[PoseLandmark.LeftWrist]`, `WorldLandmarks` (meter, titik
asal di pinggul), `SegmentationMask`, `Connections.Pose`.

## HolisticLandmarker

![Holistic](../images/gallery-holistic.png)

Pose, face mesh, dan kedua tangan satu orang, mengikuti graph holistic MediaPipe:

1. **Pose** (BlazePose, Lite atau Full) menemukan orangnya.
2. **Wajah** — ROI berasal dari landmark wajah milik pose (×3), detektor wajah berjalan di dalamnya dan keypoint-nya
   menjadi ROI face mesh. Wajah yang terlalu kecil untuk detektor (beberapa puluh piksel) jatuh kembali ke landmark
   wajah pose — hal yang tidak dilakukan MediaPipe — tetap disaring oleh skor presence mesh.
3. **Tangan** — untuk tiap pergelangan dengan visibility di atas 0,1, ROI dibangun dari landmark pergelangan,
   telunjuk, dan kelingking pose (×2,7), disempurnakan oleh **model hand ROI refinement** (256×256, dua titik:
   pergelangan dan jari tengah), lalu model landmark tangan dijalankan di sana. World landmark tangan
   dijangkarkan ulang ke pergelangan world milik pose.
4. Di mode video/live tiap bagian mempertahankan ROI sebelumnya selama masih sesuai dengan ROI baru
   (`RoiTrackingCalculator` MediaPipe: batas rotasi, translasi, dan skala, landmark sebelumnya berada di dalam ROI
   baru), sehingga crop stabil dari frame ke frame.

Wajah dan kedua tangan berjalan paralel. `HolisticResult(Pose, Face, LeftHand, RightHand)` — kiri/kanan adalah sisi
anatomis **orang tersebut**. Pada fixture pose, kedua tangan cocok dengan MediaPipe dalam selisih 0,01 (ternormalisasi).

## ImageSegmenter

![Segmentasi](../images/gallery-segment.png)

![Selfie multiclass](../images/segment-multiclass.jpg)

| `Model` | Input | Kategori |
|---|---|---|
| `Selfie` (default) | 256×256 | satu mask orang |
| `SelfieMulticlass` | 256×256 | background, hair, body-skin, face-skin, clothes, others |
| `Hair` | 512×512 (RGBA) | background, hair |
| `DeepLabV3` | 257×257 | 21 kelas PASCAL VOC (person, cat, dog, car, …) |

| Opsi | Default | |
|---|---|---|
| `OutputConfidenceMasks` | true | Satu mask probabilitas per kategori (`SegmentationResult.ConfidenceMasks`). |
| `OutputCategoryMask` | false | Kategori paling mungkin per piksel (`CategoryMask`, satu byte per piksel; 255 = tanpa label). |
| `TemporalSmoothing` | 0.3 | Smoothing berbobot ketidakpastian ala MediaPipe antar-frame (video/live). |

`result.ConfidenceMask` adalah foreground: satu-satunya mask model selfie, atau 1 − P(background) untuk model lain.
`result.GetConfidenceMask("hair")` memilih kategori berdasarkan nama, `result.Labels` mendaftarnya,
`result.CategoryMask.Histogram()` memberi porsi tiap kategori. Gunakan
`MediaPipeNet.Visualization.SegmentationMaskOverlay` untuk blur/mengganti latar atau mewarnai kategori
(`OverlayCategories`).

```csharp
using var segmenter = ImageSegmenter.Create(new() { Model = SegmenterModel.SelfieMulticlass, OutputCategoryMask = true });
var result = segmenter.Segment(image);
Console.WriteLine($"rambut menutupi {result.GetConfidenceMask("hair")!.Coverage():P1}");
```

## InteractiveSegmenter

![Segmentasi interaktif](../images/gallery-interactive.png)

MagicTouch (512×512): mensegmentasi objek di bawah sebuah titik atau goresan — "ketuk untuk memilih". Hanya mode
Image, seperti MediaPipe. Petunjuknya digambar ke kanal input keempat model persis seperti MediaPipe (lingkaran yang
ketebalannya mengikuti ukuran gambar).

```csharp
using var segmenter = InteractiveSegmenter.Create(new() { OutputCategoryMask = true });
var potongan = segmenter.Segment(image, RegionOfInterest.FromKeypoint(0.62f, 0.5f)).ConfidenceMask;
var goresan = RegionOfInterest.FromScribble([new(0.3f, 0.6f), new(0.35f, 0.62f), new(0.4f, 0.6f)]);
```

## FaceStylizer

![Stilisasi wajah](../images/gallery-stylize.png)

Menggambar ulang wajah paling menonjol dengan sebuah gaya — BlazeFaceStylizer "color sketch" (256×256) resmi yang
dikonversi ke ONNX. Pipeline-nya sama dengan MediaPipe: face mesh (mesh 192×192 bawaan stylizer) → mata dan mulut →
persegi berotasi ala `FaceToRectCalculator` (berpusat di antara mata, bergeser 10 % ke arah mulut, sisi
max(3,6 × mata-ke-mulut, 4 × mata-ke-mata)) → generator → gambar 256×256. Kepala yang miring hasilnya tetap tegak.

```csharp
using var stylizer = FaceStylizer.Create(new() { OutputFaceAlignment = true });
using var hasil = stylizer.Stylize(image);          // dispose: gambar memakai buffer pool
hasil.StylizedImage?.SaveAsPng("sketsa.png");       // null bila tidak ada wajah
using var tempel = hasil.Composite(image);          // ditempel kembali ke foto, tepi dihaluskan
Console.WriteLine(hasil.FaceRect);                  // potongan berotasi, ternormalisasi
```

| Opsi | Default | |
|---|---|---|
| `MinFaceDetectionConfidence` / `MinFacePresenceConfidence` | 0,5 | Sama seperti `FaceLandmarker`. |
| `OutputFaceAlignment` | false | Kembalikan juga potongan 256×256 yang dilihat model. |
| `Model` | color sketch | Stylizer lain yang dikonversi dengan `convert_models.py`. |

Generator menyuntikkan noise acak (seperti aslinya), jadi dua kali proses hasilnya sedikit berbeda — di MediaPipe
juga. Terhadap rata-rata 16 kali proses MediaPipe, selisih keluaran 2,4 / 255 per kanal pada 32×32 (sebaran proses
MediaPipe sendiri 0,2 di sekitar rata-rata itu; sisanya pergeseran satu piksel pusat potongan yang dibulatkan).
Varian FP16 (14 MB) dan INT8 (7,6 MB) tetap di bawah 2,8. Cara konversi model: [Model](model.md#face-stylizer).

## ImageEmbedder

![Embedding gambar](../images/gallery-embed.png)

MobileNet V3 small (224×224): vektor fitur 1024-D per gambar untuk pencarian visual, clustering, dan de-duplikasi.
Opsi: `L2Normalize` (false, seperti MediaPipe), `Quantize` (nilai int8 di `Embedding.QuantizedValues`). Bandingkan
dengan `ImageEmbedder.CosineSimilarity(a, b)`: burger vs. potongan burger yang sama ≈ 0,92, burger vs. kucing ≈ 0,05
(MediaPipe: 0,920 dan 0,048).

## ObjectDetector

![Objek](../images/gallery-objects.png)

EfficientDet-Lite0 (320×320), 80 kelas COCO. Model EfficientDet-Lite kustom dengan ukuran input berapa pun dapat
dimuat lewat `ModelPath` + `Labels` — lihat [Model kustom](model-kustom.md).

| Opsi | Default | |
|---|---|---|
| `ScoreThreshold` | 0.3 | Skor minimum. |
| `MaxResults` | -1 | Jumlah deteksi maksimum. |
| `NmsThreshold` | 0.5 | IoU untuk suppression (lintas kelas). |
| `CategoryAllowlist` / `CategoryDenylist` | null | Filter label, mis. `{ "person", "car" }`. |

## ImageClassifier

![Klasifikasi](../images/gallery-classify.png)

EfficientNet-Lite0 (224×224), 1000 kelas ImageNet. Opsi: `MaxResults` (5), `ScoreThreshold`,
`CategoryAllowlist`, `CategoryDenylist`, serta `ModelPath` / `Labels` untuk [classifier Anda sendiri](model-kustom.md).

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
