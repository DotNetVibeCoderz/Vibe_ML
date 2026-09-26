# Aplikasi MediaPipe.Net Gallery

> 🇬🇧 [Read in English](../en/gallery.md)

Aplikasi desktop Avalonia lintas platform (Windows, Linux, macOS) yang memamerkan setiap task: coba pada sampel
bawaan atau gambar dan webcam Anda sendiri, atur opsinya, baca hasilnya, lalu salin kode C# yang menghasilkannya.

```bash
dotnet run --project samples/MediaPipeNet.Gallery -c Release
```

Aplikasi membundel semua model (`Gravicode.MediaPipeNet.Models.All`) sehingga berjalan offline. Di Windows aplikasi memakai
build DirectML dari ONNX Runtime; provider default adalah CPU dan dapat diganti di *Pengaturan*.

## Halaman

| Halaman | Isi |
|---|---|
| Ikhtisar | Hasil holistic langsung di stage dan kartu untuk setiap task. |
| Deteksi wajah · Deteksi objek | Kotak, keypoint, dan label. |
| Face mesh · Landmark tangan · Pengenalan gestur · Landmark pose · Holistic | Overlay landmark beserta ROI berotasi (garis putus-putus) tempat model dijalankan. |
| Segmentasi selfie · Klasifikasi gambar | Efek mask / blur / latar studio; label top-K. |
| Kamera langsung | Task apa pun pada webcam dalam mode video: FPS, latensi, frame dibuang. |
| Graph API | Skema graph kustom (node wajah + tangan paralel, join, dan node pixelation kustom), output-nya, serta konfigurasi `.pbtxt` yang bisa diedit dan dijalankan. |
| Benchmark | Latensi per task di mesin Anda pada 640×480, dengan garis target NFR-1. |
| Model | Status katalog, verifikasi SHA-256, unduh model yang belum ada. |
| Pengaturan | Execution provider, thread CPU, bahasa (English / Bahasa Indonesia), tema (terang / gelap), folder model tambahan, titik landmark. |
| Tentang | Kredit dan lisensi. |

Setiap halaman task punya tiga tab: **Pratinjau** (stage dengan overlay dan readout instrumen: latensi · provider ·
ukuran gambar · ringkasan), **Kode C#** (snippet yang mengikuti nilai opsi saat ini, dengan tombol salin), dan
**JSON** (hasil terserialisasi).

## Screenshot

| | |
|---|---|
| ![Ikhtisar gelap](../images/gallery-home-dark-id.png) | ![Tangan gelap](../images/gallery-hands-dark-id.png) |
| ![Pengaturan gelap](../images/gallery-settings-dark-id.png) | ![Ikhtisar](../images/gallery-home.png) |
| ![Face mesh](../images/gallery-face-mesh.png) | ![Gestur](../images/gallery-gestures.png) |
| ![Pose](../images/gallery-pose.png) | ![Holistic](../images/gallery-holistic.png) |
| ![Segmentasi](../images/gallery-segment.png) | ![Objek](../images/gallery-objects.png) |
| ![Klasifikasi](../images/gallery-classify.png) | ![Deteksi wajah](../images/gallery-faces.png) |
| ![Graph API](../images/gallery-graph.png) | ![Benchmark](../images/gallery-benchmark.png) |
| ![Model](../images/gallery-models.png) | ![Kamera langsung](../images/gallery-live.png) |

## Desain

"Lab optik": chrome terang mengelilingi stage gelap seperti viewfinder, dibingkai tanda registrasi, dengan strip
readout monospace. Cobalt adalah satu-satunya aksen antarmuka; coral dan mint dicadangkan untuk apa yang dilihat
model. Tipografi: *Unbounded* untuk judul halaman, *Inter* untuk teks, *JetBrains Mono* untuk data dan kode (font
berlisensi SIL Open Font License, dibundel di `Assets/Fonts`).

## Membuat ulang screenshot

```bash
dotnet run --project samples/MediaPipeNet.Gallery -- --screenshots docs/images
```

Aplikasi mengunjungi setiap halaman, menunggu hasilnya, me-render jendela ke PNG, mengulang tiga halaman dalam tema
gelap dan Bahasa Indonesia, lalu keluar. `--lang id` dan `--theme Dark` memulai aplikasi interaktif dalam mode
tersebut tanpa mengubah pengaturan tersimpan.

## Peta kode

| Folder | |
|---|---|
| `Services/TaskCatalog.cs` | Definisi sembilan task: opsi, render overlay, baris hasil, snippet kode. |
| `Controls/StageView.cs` | Kontrol viewfinder (fit gambar, overlay vektor, tanda registrasi, readout). |
| `Controls/GraphDiagram.cs` | Skema berlapis untuk `CalculatorGraph` apa pun (memakai `GetEdges()`). |
| `Views/*` | Halaman. `MainWindow.cs` mengatur navigasi dan mode screenshot. |
| `Services/Loc.cs` | String English / Indonesia. |
| `Theme/Tokens.axaml`, `Theme/Styles.axaml` | Token desain (terang/gelap) dan style. |
