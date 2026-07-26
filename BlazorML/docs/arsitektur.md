# Arsitektur

Dokumen ini menjelaskan bagaimana solusi disusun dan kenapa. Untuk roadmap lihat
[PLAN.md](../PLAN.md); untuk status pengerjaan lihat [Progress.md](../Progress.md).

---

## Lima project

```
BlazorML.Core ──────────┬──────────────┬───────────────┐
                        │              │               │
                        ▼              ▼               ▼
                  Infrastructure ──► BlazorML.ML ──► BlazorML.Agents
                        │              │               │
                        └──────────────┴───────────────┴──► BlazorML.Web
```

Dependensi hanya mengalir ke kanan. Tidak ada siklus, dan tidak ada project yang menarik
dependensi berat milik tetangganya.

| Project | Isi | Kenapa terpisah |
|---|---|---|
| `Core` | Entity, enum, `ExperimentGraph`, `TabularData`, `ModuleCatalog`, abstraksi, kelas options | Tanpa dependensi berat, jadi aman dipakai semua layer |
| `Infrastructure` | `AppDbContext`, switch 4 provider database, Identity, 4 storage provider, `SettingsService`, seeder | Semua I/O eksternal terkumpul di satu tempat |
| `ML` | Profiling, 8 executor modul, registry 25 trainer, evaluasi, AutoML, run engine, script runner | Mengisolasi ML.NET beserta binary native-nya dari layer web |
| `Agents` | Kernel factory 4 provider LLM, chat service, 4 kelompok kernel function | LLM opsional; tanpa kredensial aplikasi tetap jalan penuh |
| `Web` | Blazor Server, design system, kanvas D3, Minimal API scoring | Satu-satunya entry point |

---

## Keputusan yang mengikat

### 1. `ExperimentGraph` disimpan sebagai satu dokumen JSON

Graph tidak dinormalisasi ke tabel node dan edge. Seluruhnya disimpan di
`Experiment.GraphJson`.

Konsekuensinya bagus: versioning jadi satu baris `ExperimentVersion` per penyimpanan, dan
tidak ada risiko graph tersimpan setengah. Reproducibility yang diminta spesifikasi jadi
gratis. Kerugiannya, kamu tidak bisa mencari "eksperimen mana yang memakai modul X" lewat SQL —
tapi itu bukan pertanyaan yang muncul di aplikasi ini.

### 2. `ModuleCatalog` adalah satu-satunya sumber kebenaran

Palette, renderer node di kanvas, inspector parameter, validasi graph, dan run engine semuanya
membaca dari `ModuleCatalog`. Menambah modul baru = satu deklarasi, nol kode UI.

```csharp
new()
{
    Id = "tf.contoh",
    Name = "Contoh Transform",
    Category = ModuleCategory.DataTransform,
    Description = "Satu kalimat, ditulis untuk orang yang sedang memutuskan.",
    Inputs = [In("Dataset", PortType.Dataset)],
    Outputs = [Out("Dataset", PortType.Dataset)],
    Parameters = [P("kolom", "Kolom", ParameterKind.ColumnName)]
}
```

Setelah itu tinggal menangani id-nya di executor yang sesuai.

### 3. `TabularData`, bukan `IDataView`, sebagai payload antar-modul

Modul transform, modul LLM, dan script pengguna semuanya perlu membaca dan menulis sel satu
per satu. `IDataView` yang lazy dan berorientasi kolom membuat itu menyakitkan.

Konversi ke `IDataView` dilakukan tepat sebelum trainer membutuhkannya, lewat `MlDataBridge`.
Jalurnya melalui CSV sementara dan `TextLoader` yang skemanya dibangun saat runtime — karena
loader in-memory ML.NET menuntut tipe baris yang diketahui saat kompilasi, sedangkan designer
tidak pernah tahu kolom apa yang akan dibawa pengguna.

### 4. Evaluasi dihitung dari tabel hasil scoring

ML.NET memberi angka ringkasan, tapi tidak memberi titik kurva ROC maupun sampel residual —
dan designer butuh keduanya untuk menggambar apa pun yang berguna.

Menghitung dari baris hasil scoring memberi satu jalur kode yang menghasilkan angka ringkasan
**dan** bentuk yang dibutuhkan chart. Bonusnya: ikut bekerja untuk tabel yang di-score di luar
ML.NET, misalnya hasil klasifikasi oleh LLM.

### 5. Id bertipe `string` di semua entity

GUID format `n`. Skemanya identik di keempat provider database tanpa cabang per-provider, dan
tidak ada masalah identity/sequence yang berbeda antar engine.

### 6. `EnsureCreated`, bukan EF Migrations

Migration multi-provider berarti empat set migration terpisah yang harus dijaga tetap sejalan.
Untuk aplikasi yang di-deploy sendiri oleh penggunanya, itu beban tanpa manfaat.

### 7. Storage selalu lewat `IStorageProvider`

Tidak ada `File.ReadAllText` di luar `FileSystemStorageProvider`. Dataset dan model `.zip`
sama-sama lewat jalur ini, jadi berpindah dari FileSystem ke S3 adalah perubahan konfigurasi.

---

## Cara sebuah run bekerja

1. `ExperimentRunner` memuat eksperimen dan mem-parse `GraphJson`.
2. Graph diurutkan topologis (Kahn). Siklus ditolak dengan menyebut node mana yang terlibat.
3. Untuk tiap node, input diambil dari keluaran node hulu lewat kunci `"nodeId:portIndex"`.
4. Executor yang cocok dipilih lewat `CanExecute(moduleId)`.
5. Hasil, durasi, dan pratinjau dicatat sebagai `NodeRunResult`; log per node ikut tersimpan.
6. Node yang hulunya gagal ditandai **Skipped**, bukan Failed — ia tidak pernah dapat input,
   jadi menyebutnya gagal itu menyesatkan.

Progress dikirim lewat `IProgress<RunProgress>` supaya kanvas bisa menyalakan node selagi run
masih berjalan.

---

## Hal yang sengaja tidak dilakukan

- **Tidak ada state management global** di Blazor. State hidup di komponen atau di database.
- **Tidak ada AutoMapper atau MediatR.** Aplikasinya tidak cukup besar untuk membayar
  indireksinya.
- **Tidak ada repository generik** di atas EF Core. `DbContext` sudah sebuah repository.
- **Tidak ada Bootstrap.** Design system dibangun sendiri, ~1.100 baris CSS, tanpa build step.
