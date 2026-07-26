# PLAN.md — Roadmap Pengembangan Blazor ML Studio

Dokumen ini adalah rencana kerja. Status harian ada di [Progress.md](Progress.md);
spesifikasi produk ada di [requirements.txt](requirements.txt).

---

## 1. Sasaran

Membangun **Blazor ML Studio (BlazorML)** — platform drag-and-drop berbasis web untuk
menyusun, melatih, dan men-deploy model ML tanpa banyak coding, dengan asisten data
scientist "Profesor Wicak" sebagai fitur kelas satu, bukan tempelan.

Dibuat oleh **Gravicode Studios**, di-lead oleh **Kang Fadhil**.

---

## 2. Arsitektur solusi

Lima project, dependensi mengalir satu arah (tidak ada siklus):

```
BlazorML.Core ──────────┬──────────────┬───────────────┐
   domain, graph,       │              │               │
   module catalog,      ▼              ▼               ▼
   abstractions   Infrastructure ──► BlazorML.ML ──► BlazorML.Agents
                  EF Core 4 DB       ML.NET,          Semantic Kernel,
                  4 storage          executor         plugins
                        │              │               │
                        └──────────────┴───────────────┴──► BlazorML.Web
                                                            Blazor Server + Minimal API
```

| Project | Isi | Kenapa dipisah |
|---|---|---|
| `BlazorML.Core` | Entity, enum, `ExperimentGraph`, `ModuleCatalog`, abstraksi, options | Tidak punya dependensi berat, jadi bisa dipakai semua layer |
| `BlazorML.Infrastructure` | `AppDbContext`, provider switch 4 database, Identity, 4 storage provider, repository, seeder | Semua I/O eksternal terkumpul di satu tempat |
| `BlazorML.ML` | Profiling dataset, eksekutor modul, registry trainer, evaluasi, AutoML, scoring, run engine | Isolasi ML.NET supaya Web tidak ikut menarik native binary-nya |
| `BlazorML.Agents` | Kernel factory 4 provider LLM, chat service, kernel functions | LLM opsional; kalau tidak dikonfigurasi, app tetap jalan |
| `BlazorML.Web` | Blazor Server, design system, D3 designer, Minimal API scoring | Satu-satunya entry point |

### Keputusan arsitektur yang mengikat

1. **`ExperimentGraph` sebagai satu dokumen JSON.** Graph disimpan utuh di
   `Experiment.GraphJson`, bukan tabel node/edge ternormalisasi. Versioning jadi
   satu baris `ExperimentVersion` per simpan — murah, dan reproducibility di spec terpenuhi.
2. **`ModuleCatalog` sebagai satu-satunya sumber kebenaran modul.** Palette, renderer node,
   inspector parameter, dan run engine semuanya membaca dari sini. Menambah modul = satu deklarasi.
3. **Id bertipe `string` (GUID "n")** di semua entity, supaya skema identik di SQLite,
   SQL Server, MySQL, dan PostgreSQL tanpa cabang per-provider.
4. **`EnsureCreated` + seeder, bukan EF Migrations.** Migration multi-provider butuh empat set
   migration terpisah. Untuk aplikasi yang di-deploy user sendiri, ini beban tanpa manfaat.
5. **Storage selalu lewat `IStorageProvider`.** Tidak ada `File.ReadAllText` di luar
   `FileSystemStorageProvider`. Dataset dan model `.zip` sama-sama lewat jalur ini.

---

## 3. Dependensi yang sudah dikunci

Semua versi di bawah sudah diverifikasi `dotnet restore` bersih di mesin ini
(.NET SDK 10.0.302), dipusatkan di `Directory.Packages.props`.

| Area | Paket | Versi | Catatan |
|---|---|---|---|
| Data | `Microsoft.EntityFrameworkCore.*` | 10.0.10 | SQLite + SQL Server |
| Data | `Npgsql.EntityFrameworkCore.PostgreSQL` | 10.0.0 | |
| Data | `MySql.EntityFrameworkCore` | 10.0.7 | **Bukan Pomelo** — lihat catatan di bawah |
| ML | `Microsoft.ML` | 5.0.0 | |
| ML | `Microsoft.ML.AutoML` | 0.23.0 | Menuntut `Microsoft.ML` ≥ 5.0.0 |
| ML | `FastTree`, `LightGbm`, `Recommender`, `TimeSeries`, `Mkl.Components` | 5.0.0 / 0.23.0 | |
| LLM | `Microsoft.SemanticKernel` | 1.78.0 | Connector OpenAI ikut di sini |
| LLM | `...Connectors.Google`, `...Connectors.Ollama` | 1.78.0-alpha | Hanya tersedia prerelease |
| LLM | `Anthropic.SDK` | 5.10.0 | Microsoft tidak merilis connector Anthropic |
| Web | `Swashbuckle.AspNetCore` | 9.0.6 | Swagger untuk endpoint scoring |
| Web | `Markdig` | 1.3.2 | Render markdown chat ke HTML |
| Viz | D3.js v7 | vendored | `wwwroot/lib/d3/d3.min.js`, bukan CDN |

### Dua temuan yang mengubah rencana awal

- **Pomelo.EntityFrameworkCore.MySql tidak mendukung EF Core 10.** Versi tertinggi
  (9.0.0) mensyaratkan `EntityFrameworkCore.Relational` ≤ 9.0.999. Memakainya berarti
  menahan seluruh solusi di EF Core 9. Diganti `MySql.EntityFrameworkCore` 10.0.7 dari Oracle.
- **Tidak ada `Microsoft.SemanticKernel.Connectors.Anthropic`.** Paket itu tidak pernah
  ada di nuget.org. Anthropic disambungkan lewat `Anthropic.SDK` yang meng-expose
  `IChatClient`, lalu diadaptasi ke Semantic Kernel dengan `AsChatCompletionService()`.
  Tiga provider lain memakai connector resmi.

---

## 4. Tahapan

Tanda ✅ diperbarui di [Progress.md](Progress.md), bukan di sini.

### Fase 1 — Fondasi
1. Scaffold solusi, central package management, kunci versi
2. Core: entity, enum, `ExperimentGraph` + topological sort, `DatasetProfile`, `ModuleCatalog`
3. Core: abstraksi (`IStorageProvider`, `IDatasetService`, …) dan kelas options `appsettings`

### Fase 2 — Infrastruktur
4. `AppDbContext` + Identity + switch 4 provider database
5. Empat storage provider di balik `IStorageProvider`
6. Repository, seeder, dan layanan `AppSetting` (config bisa diubah dari dalam aplikasi)

### Fase 3 — Mesin ML
7. Loader + profiler dataset (statistik kolom, histogram, sample rows)
8. Eksekutor per kategori modul: transform, algoritma, training, scoring, evaluasi
9. Run engine: topological sort → eksekusi → `NodeRunResult` + log per node
10. AutoML, cross-validation, sweep, permutation feature importance
11. Registry model: simpan `.zip` ML.NET, versioning, schema input untuk API

### Fase 4 — Agent
12. Kernel factory 4 provider + pembacaan setting dari `appsettings`/database
13. Chat service: multi-session, attachment, riwayat, streaming
14. Kernel functions umum: Tavily search, scrape URL, baca file dari URL, tanggal/waktu, math
15. Kernel functions produk: query dataset, **baca dan tulis `ExperimentGraph`**

### Fase 5 — Antarmuka
16. Design system "neo brutalism soft": token, dark/light, tipografi, komponen dasar
17. App shell: navigasi, header, theme switcher, halaman auth
18. Designer D3: canvas, palette, drag-and-drop, routing koneksi, inspector, status run
19. Halaman: dashboard, datasets, experiments, models, endpoints, marketplace, settings, profile
20. Panel chat Profesor Wicak: markdown → HTML, attachment, multi-sesi, show/hide
21. Visualisasi hasil: ROC, confusion matrix, residual, feature importance, leaderboard

### Fase 6 — Deployment dan penutup
22. Minimal API scoring + API key + Swagger
23. Seed: user contoh, dataset contoh, experiment marketplace siap pakai
24. `docs/` lengkap + `README.md` dwibahasa (EN/ID)
25. Build bersih, jalankan, verifikasi satu experiment end-to-end

### Fase 7 — Uji otomatis
26. Logika inti, serializer, evaluator, script runner
27. Layer agent tanpa panggilan penyedia; kanvas D3 lewat Chromium sungguhan
28. Reset sandi, riwayat run, perbandingan hasil, versi eksperimen
29. **Menjalankan setiap trainer dan setiap modul di katalog** — bukan memvalidasi bentuknya,
    tapi benar-benar fit sebuah model lalu men-score dengannya. Digerakkan dari `ModuleCatalog`,
    jadi modul yang ditambahkan nanti otomatis ikut terjaring

### Sisa pekerjaan yang diketahui
30. **Tiga database dan tiga storage provider belum pernah dieksekusi.** Kodenya lengkap dan
    tersambung; yang kurang adalah menjalankan SQL Server, MySQL, PostgreSQL, Azure Blob, S3 dan
    MinIO sungguhan — realistisnya lewat Docker Compose di CI
31. ~~Atribusi hanya tampil di halaman masuk~~ ✅ — komponen `Attribution` membaca
    `WorkspaceOptions` dan tampil di app shell serta halaman masuk

---

## 5. Batasan yang disepakati di muka

Beberapa hal di spec dikerjakan dengan cakupan yang perlu dinyatakan terus terang,
supaya tidak terbaca sebagai selesai penuh:

- **Script R dan Python** dijalankan lewat runtime eksternal (Python.NET / `Rscript`).
  Modulnya ada dan tereksekusi bila runtime terpasang; bila tidak, node gagal dengan
  pesan yang menjelaskan apa yang kurang — bukan diam-diam dilewati.
- **AutoML vision dan NLP** memakai jalur ML.NET yang menarik dependensi TorchSharp
  berukuran besar. Fase 3 mengerjakan classification, regression, dan recommendation
  dulu; vision/NLP menyusul setelah jalur utama terbukti jalan.
- **Kredensial provider LLM tidak di-seed.** Chat aktif setelah user mengisi API key
  di halaman Settings. Tanpa itu panel chat tetap tampil dan menjelaskan apa yang perlu diisi.

---

## 6. Definisi selesai

Sebuah fase dianggap selesai bila:

1. `dotnet build` bersih tanpa error,
2. jalur yang dibangun bisa dijalankan dari UI, bukan hanya lolos compile,
3. status di `Progress.md` diperbarui dengan apa yang benar-benar berjalan dan apa yang belum.
