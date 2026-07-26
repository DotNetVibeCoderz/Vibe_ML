# Progress.md — Status Pengembangan

Pelacakan status. Roadmap dan keputusan arsitektur ada di [PLAN.md](PLAN.md).

**Terakhir diperbarui:** 26 Juli 2026
**Build:** `dotnet build` bersih — 0 error, 0 warning CS
**Tes:** `dotnet test` — **684 lulus, 1 skip, 0 gagal** (409 inti + 89 agent + 123 web + 63 kanvas)
**Runtime:** aplikasi menyala, seluruh halaman merespons 200, 0 galat di log

Keterangan: ✅ selesai & terverifikasi jalan · 🔨 sedang dikerjakan · ⬜ belum · ⚠️ terbatas

---

## Ringkasan

| Fase | Cakupan | Status |
|---|---|---|
| 1 | Fondasi: solusi, Core domain, module catalog | ✅ |
| 2 | Infrastruktur: database, storage, config | ✅ |
| 3 | Mesin ML: profiler, eksekutor, run engine, AutoML | ✅ |
| 4 | Agent: Semantic Kernel, chat, kernel functions | ✅ |
| 5 | Antarmuka: design system, designer D3, halaman, chat panel | ✅ |
| 6 | Deployment API, seed, dokumentasi, verifikasi | ✅ |

---

## Bukti verifikasi

**Build.** `dotnet build` atas solusi penuh: 0 error, 0 warning CS.

**Halaman.** Setelah masuk sebagai `admin@gravicode.com`, seluruh rute merespons `200`:
`/` · `/dataset` · `/eksperimen` · `/eksperimen/{id}` · `/model` · `/endpoint` · `/galeri` ·
`/pengaturan` · `/profil` · `/api-docs`. Log aplikasi bersih tanpa entri `fail:`.

**Halaman designer** merender 63 modul di palette, kanvas, dan inspector.

**Pelatihan end-to-end.** Kelima eksperimen contoh yang tidak butuh kunci LLM dijalankan
lewat `ExperimentRunner` yang sama dengan yang dipakai UI:

| Eksperimen | Hasil | Metrik | Payload chart |
|---|---|---|---|
| Klasifikasi spesies iris | ✅ 0,2 s | MicroAccuracy 0,9667 · MacroAccuracy 0,9667 | confusion 3×3 |
| Prediksi harga rumah | ✅ 0,2 s | MAE 67,19 · RMSE 84,86 | 80 titik residual |
| Prediksi churn pelanggan | ✅ 0,7 s | Accuracy 0,9167 · Precision 0,9160 · Recall 0,9561 | ROC 182 titik · confusion 2×2 |
| AutoML churn | ✅ 60,1 s | AUC 0,9852 · Accuracy 0,9459 · F1 0,9556 | 137 titik residual |
| Rekomendasi film | ✅ | MAE 0,5373 · RMSE 0,7515 | — |

Eksperimen keenam (Analisis sentimen ulasan) tidak diuji karena butuh kunci API penyedia LLM.

Kolom terakhir memastikan bukan hanya angkanya yang benar, tapi payload untuk grafik ROC,
confusion matrix, dan residual benar-benar bertahan sampai tersimpan di `NodeResultsJson`.

---

## Rincian per fase

### Fase 1 — Fondasi ✅

Solusi 5 project net10.0 · central package management dengan versi terverifikasi ·
D3.js v7 di-vendor lokal · 17 enum, 14 entity · `ExperimentGraph` dengan topological sort Kahn ·
`TabularData` + profiling kolom · **`ModuleCatalog` berisi 63 modul** dengan port bertipe dan
spec parameter · 7 section konfigurasi.

### Fase 2 — Infrastruktur ✅

`AppDbContext` dengan skema identik di keempat provider · switch SQLite/SQL Server/MySQL/
PostgreSQL · ASP.NET Identity dengan 3 peran · 4 storage provider di balik `IStorageProvider` ·
`SettingsService` dua lapis (appsettings sebagai dasar, override tersimpan di DB) ·
`ISecretProtector` untuk section berkredensial · serializer CSV/TSV/JSON.

### Fase 3 — Mesin ML ✅

`MlDataBridge` konversi dua arah · featurisasi otomatis dari tipe kolom · **27 trainer** ·
jalur khusus untuk matrix factorization, SSA forecasting, dan spike detection · `Evaluator`
(ROC, AUC rank-based, confusion matrix, residual, clustering) · AutoML binary/multiclass/regresi
dengan leaderboard · cross-validate, sweep, permutation feature importance · **12 executor
transform** · **6 executor modul LLM** · **4 script runner** · `ExperimentRunner` dengan log per
node · `ModelRegistry` dengan versioning.

### Fase 4 — Agent Profesor Wicak ✅

Kernel factory untuk OpenAI, Anthropic, Gemini, Ollama · chat multi-sesi dengan streaming ·
lampiran gambar dan dokumen · **20 kernel function** dalam 4 kelompok: utilitas, web,
dataset, dan designer · `LlmActionRunner` terpisah tanpa akses tools untuk modul data.

### Fase 5 — Antarmuka ✅

Design system "neo brutalism soft" (~1.400 baris CSS, tanpa build step) dengan dark/light lewat
`light-dark()` · palet kategori tervalidasi CVD/kontras di kedua tema · app shell dengan rail
ikon · halaman auth · **kanvas D3** dengan drag-drop, routing ortogonal, pemeriksaan tipe port,
dan animasi jejak saat run · inspector parameter generik dari spec · 9 halaman aplikasi ·
panel chat dengan render markdown penuh.

**Visualisasi evaluasi** ✅ — empat grafik D3, tiap bentuk dipilih sesuai tugas datanya:

| Grafik | Bentuk | Warna |
|---|---|---|
| Kurva ROC | garis + diagonal acuan, crosshair menempel ke titik terdekat | satu warna, seri tunggal |
| Confusion matrix | heatmap, hitungan tercetak di tiap sel | sekuensial satu hue, terang→gelap |
| Prediksi vs sebenarnya | scatter + garis prediksi sempurna | satu warna, cincin surface agar titik bertumpuk tetap terbaca |
| Kepentingan fitur | bar horizontal | divergen — skor negatif itu temuan nyata, bukan derau |

Tiap grafik berpasangan dengan **tampilan tabel** berisi angka yang sama, jadi tidak ada
informasi yang hanya tersampaikan lewat warna.

### Fase 6 — Deployment dan penutup ✅

Minimal API scoring dengan kunci API dan Swagger · seed 5 pengguna, 5 dataset, 6 eksperimen ·
`docs/` berisi 6 dokumen · `README.md` dwibahasa · `CLAUDE.md` diperbarui dengan perintah dan
arsitektur sungguhan.

### Fase 7 — Uji otomatis ✅

**684 tes, semuanya lulus.** Ditulis setelah aplikasinya jalan, dan menemukan **dua puluh satu bug**
yang lolos dari verifikasi manual — itu justifikasi suite ini.

`tests/BlazorML.Tests` (409) — logika inti:

| Berkas | Isi |
|---|---|
| `TabularDataTests` | Inferensi tipe, operasi kolom, escaping CSV, isolasi `Clone`, profil |
| `ExperimentGraphTests` | Urutan topologis, deteksi siklus, edge yatim, round-trip JSON |
| `EvaluatorTests` | AUC dengan jawaban yang diketahui (termasuk kasus seri dan satu kelas), confusion matrix, monotonisitas ROC, metrik regresi |
| `TransformExecutorTests` | Semua transform utama, plus penjaga stratifikasi dan kardinalitas one-hot |
| `ModuleCatalogTests` | Invarian katalog untuk seluruh 63 modul secara parametrik |
| `TrainingIntegrationTests` | Melatih model ML.NET sungguhan atas data bersinyal tertanam, lalu memeriksa modelnya menemukan sinyal itu — termasuk round-trip simpan/muat dan scoring baris tanpa label |
| `LlmExecutorTests` | Batching, parsing balasan berpagar kode, fallback per-baris saat batch tidak sejajar, substitusi `{{kolom}}` — semuanya lewat penyedia tiruan, tanpa kunci API |
| `SerializerTests` | Round-trip CSV/TSV/JSON/Parquet: nilai, gap yang tetap jadi gap, tipe kolom yang bertahan di Parquet, dan stream yang tidak bisa di-seek |
| `ScriptRunnerTests` | C# dan JavaScript selalu jalan; Python dijalankan sungguhan di mesin ini — termasuk memastikan script tak berujung **benar-benar dibunuh saat timeout**. R ter-skip dengan alasan tercatat bila tidak terpasang |
| `VisionNlpTests` | Modul vision/NLP ada di katalog di build mana pun, dan tanpa pack-nya kegagalan menyebutkan cara mengaktifkan beserta ongkosnya |
| `TrainerCoverageTests` | **Setiap trainer di katalog, dilatih sungguhan.** Digerakkan dari `ModuleCatalog` alih-alih daftar tulis-tangan, jadi trainer yang ditambahkan nanti otomatis ikut teruji. Dua sapuan parametrik — semuanya harus berhasil fit lalu dipakai men-score, dan yang punya label harus mengalahkan tebakan atas data bersinyal tertanam — plus jalur khusus matrix factorization, SSA (termasuk penolakan seri terlalu pendek), dan spike detection |
| `ModuleCoverageTests` | Modul non-trainer yang belum pernah dijalankan: Cross Validate, Tune Hyperparameters, Feature Importance, Select Best Features, Featurize Text, Enter Data Manually, Import from SQL (SQLite sementara sungguhan) dan Import from Web (transport tiruan: CSV, JSON, header, 404) |

`tests/BlazorML.Agents.Tests` (89) — layer agent, **tanpa satu pun panggilan ke penyedia**:

| Berkas | Isi |
|---|---|
| `DesignerPluginTests` | Fungsi yang membuat bot bisa menyusun eksperimen: pemeriksaan tipe port, penolakan lingkaran, penggantian edge pada port yang sama, validasi nilai pilihan, dan bahwa **tiap perubahan tercatat sebagai versi baru**. Ditutup dengan membangun alur lengkap dari kanvas kosong lewat fungsi-fungsi itu saja, lalu menjalankannya sampai keluar metrik |
| `DataPluginTests` | Query dataset sungguhan dengan filter, profil kolom, hitung keseimbangan kelas |
| `TimeAndMathPluginTests` | Aritmetika, selisih tanggal, statistik, penanganan input buruk |
| `WebPluginTests` | Tavily dan scraping lewat transport tiruan: sakelar dihormati, pesan kunci hilang, script/style dibuang dari teks |
| `KernelFactoryTests` | **Keempat penyedia benar-benar membangun kernel dengan chat service** — termasuk Anthropic yang lewat adapter — plus penolakan bernama saat kredensial kosong, dan plugin yang sengaja tidak dipasang untuk modul data |
| `WicakChatServiceTests` | Sesi per pemilik, reset, hapus berantai, urutan pesan |
| `OutputModuleTests` | Dua modul yang menulis keluar dari eksperimen, di atas workspace sungguhan: Save Dataset round-trip di ketiga format beserta profil kolomnya, dan Register Model — termasuk **memuat kembali berkas yang ditulis lalu benar-benar men-score dengannya**, karena mencatat baris database sambil menulis model yang tidak bisa dimuat baru akan ketahuan di produksi |
| `DatasetPreviewTests` | `PreviewAsync` di atas workspace sungguhan — sebelumnya ada di interface sejak awal dan **tidak pernah dipanggil sekali pun**: baris teratas berkas, kolom lengkap, permintaan melebihi jumlah baris, gap yang tetap gap, dan dataset yang sudah tidak ada menyebut mana yang dicari |

| `ShippedSettingsTests` | Membaca `appsettings.json` yang benar-benar dikirim: tidak ada kredensial, semua section ada, dan defaultnya yang aman. **Tes inilah yang menemukan kunci API sungguhan di dalam file itu.** |

`tests/BlazorML.Canvas.Tests` (63) — kanvas D3, lewat **browser Chromium sungguhan**:

Satu-satunya cara memverifikasi bagian ini: D3 menata SVG dan HTML lewat JS interop di atas
circuit Blazor, dan semua itu belum ada sampai browser mengeksekusinya. Aplikasinya dijalankan
sebagai proses anak dengan database dan storage di folder sementara, lalu dikemudikan Playwright.

| Cakupan | Isi |
|---|---|
| Render | Node dan edge tergambar, port ada, warna pena kategori sampai ke node, 63 modul di palette |
| Interaksi | Klik memilih node dan membuka inspector · drag memindahkan node **dan perubahannya sampai ke Blazor** · klik palette menambah node · pencarian menyaring · klik garis memutus sambungan · **drag antar-port membuat sambungan** · zoom dan paskan tampilan |
| Tema | Toggle mengubah dokumen dan kanvas ikut, serta pilihannya bertahan setelah reload |
| Panel chat | Buka/tutup, dan tanpa kredensial ia mengatakan apa yang perlu diisi |
| Responsif | Viewport selebar ponsel tidak menghasilkan scroll horizontal |
| Shell & atribusi | Baris kredit muncul di **setiap** halaman shell dan di halaman masuk dengan nama yang sama · judul halaman beserta tombol aksinya benar-benar berada di top bar · tidak ada `sectioncontent`/`sectionoutlet` tersisa di DOM · kanvas designer tidak tumbuh scrollbar setelah bar kredit ditambahkan |
| Zoom & pan | Ujung tiap edge diukur jaraknya ke port terdekat **dalam satuan dunia** — melewati zoom in ×2, zoom out ×4, zoom in ×2, paskan tampilan, dan pan — plus penjaga bahwa kedua lapisan benar-benar menerima skala yang sama, supaya kanvas yang mengabaikan zoom tidak lolos begitu saja |
| Halaman Endpoint | Tab bahasa mengganti snippet · tombol salin mengonfirmasi di dirinya sendiri dan isinya benar-benar sampai ke clipboard · **"Kembalikan contoh" mengubah isi textarea yang sudah diketik** · mengirim permintaan menampilkan status yang dijawab API · snippet curl yang panjang tidak menyeret halaman ke samping di layar ponsel |
| Pratinjau dataset | Baris CSV seeded sungguhan tergambar · pemilih jumlah baris membaca ulang · tab Data/Kolom bertukar · tabel lebar men-scroll di dalam bingkainya sendiri, bukan menyeret halaman · **nama kolom tetap terlihat saat barisnya di-scroll** |
| Gaya form | `<input>` polos mendapat kotak yang sama persis dengan `select` di sebelahnya — radius, tebal border, ukuran font, padding · tipe yang disuntikkan saat itu juga (password, number, search) ikut sama · checkbox, radio, range dan file tetap digambar sendiri · input teks fokus memakai bayangan, checkbox fokus tetap punya outline |

Kredensial penyedia sengaja dikosongkan lewat environment variable saat aplikasi dijalankan, jadi
hasil tes tidak bergantung pada apakah developer punya kunci di user-secrets — dan tidak ada tes
yang bisa membelanjakan uang ke model sungguhan.

`tests/BlazorML.Web.Tests` (123) — layer web:

| Berkas | Isi |
|---|---|
| `ParameterFieldTests` | Tiap `ParameterKind` merender kontrol yang benar; dijalankan parametrik atas seluruh enum sehingga kind baru tanpa penanganan langsung gagal |
| `PasswordResetTests` | Alur lupa-sandi end-to-end: halaman, token asli mengganti sandi dan langsung masuk, token tidak bisa dipakai dua kali, dan **alamat yang terdaftar dijawab persis sama dengan yang tidak** — kalau berbeda, halaman itu jadi cara mengetahui siapa saja yang punya akun |
| `ScoringApiTests` | Boot aplikasi sungguhan lewat `WebApplicationFactory`, latih dan terbitkan model, lalu panggil API seperti integrator luar: jalur sukses, prediksi benar-benar mengikuti sinyal, dan seluruh jalur penolakan |
| `EndpointSnippetTests` | Snippet keempat bahasa: memuat URL, header kunci dan seluruh nama kolom · tidak pernah memuat kunci asli · body curl-nya di-parse ulang dan harus sama persis dengan payload contoh · nama kolom bertanda kutip, spasi, backslash dan apostrof tidak merusak apa pun |
| `EndpointTesterTests` | Setiap badan permintaan yang jelas tidak bisa jalan ditolak **sebelum** ada panggilan sama sekali — factory-nya melempar kalau ada yang mencoba · kunci masuk ke header yang memang dibaca API · 401 dilaporkan sebagai respons, bukan kegagalan · balasan non-JSON ditampilkan apa adanya |
| `GeneratedSnippetIntegrationTests` | Menutup lingkarannya: payload yang ditampilkan halaman untuk model sungguhan yang sudah diterbitkan **benar-benar diskor 200** oleh API yang berjalan — termasuk body yang diambil dari dalam perintah curl yang dihasilkan |
| `EndpointPageTests` | Halaman Endpoint lewat bUnit di atas SQLite sementara: keempat tab, snippet ikut berganti, konsol tertutup sampai diminta, teks yang diketik itu yang dikirim, endpoint berhenti diperingatkan sebelum panggilan, dan tiap endpoint punya tab serta konsolnya sendiri |
| `DatasetPreviewPageTests` | Pratinjau baris di halaman Dataset lewat bUnit dengan service tiruan: gap dirender sebagai em dash bukan sel kosong, kolom angka rata kanan, memilih 500 baris **meminta ulang ke service** alih-alih memotong yang sudah ada, berkas tak terbaca menjelaskan diri, dan menutup dialog melepas barisnya |

---

## ⚠️ Batasan yang perlu dinyatakan

Semua tercantum juga di README supaya tidak ada kejutan:

| Hal | Status |
|---|---|
| **Python & R** | Berjalan sebagai proses anak, bukan Python.NET. Penyimpangan disengaja — alasannya di bawah. Perlu runtime terpasang; kalau tidak ada, node gagal dengan pesan yang jelas. |
| **AutoML vision & NLP** | Ada, tapi **runtime-nya opt-in**. Modulnya selalu tampil di katalog; build dengan `-p:EnableVisionNlp=true` untuk menyertakan backbone-nya. Tanpa flag itu, mencoba menjalankannya memberi pesan yang menyebutkan cara mengaktifkan dan ongkosnya. |
| **Impor Parquet** | ✅ Baca dan tulis, lewat Parquet.Net (14 MB). |
| **Modul LLM** | Perlu kunci API. Tanpa itu node gagal dengan penjelasan. |
| **Connector Gemini & Ollama** | Prerelease dari Microsoft (`1.78.0-alpha`). |
| **Berpindah storage provider** | Tidak memindahkan berkas lama secara otomatis. |
| **Cakupan tes** | 390 tes menutupi logika inti, mesin ML, layer agent, komponen Razor, dan API scoring. Seluruh lapisan kini punya tes otomatis, termasuk kanvas D3 lewat browser sungguhan. |

---

## Catatan teknis dan bug yang ditemukan saat pengerjaan

**Pomelo tidak bisa dipakai untuk MySQL.**
Versi tertinggi 9.0.0 mensyaratkan EF Core ≤ 9.0.999, yang akan menahan seluruh solusi di EF
Core 9. Diganti `MySql.EntityFrameworkCore` 10.0.7 dari Oracle.

**Connector Anthropic untuk Semantic Kernel tidak ada.**
`Microsoft.SemanticKernel.Connectors.Anthropic` tidak pernah dirilis ke nuget.org. Anthropic
disambung lewat `Anthropic.SDK`, yang mengimplementasikan `IChatClient` secara **eksplisit** —
jadi perlu cast dulu sebelum extension `AsBuilder()`/`AsChatCompletionService()` terlihat.

**Python dijalankan sebagai proses anak, bukan Python.NET.**
Penyimpangan dari spec, disengaja. Python.NET meng-host CPython di dalam proses web: satu
interpreter global untuk seluruh server, GIL diperebutkan antar circuit Blazor, timeout tidak
bisa dipaksakan pada script yang macet di native code, dan segfault di kode pengguna
menjatuhkan seluruh aplikasi. Proses anak memberi isolasi per run, timeout yang benar-benar
bekerja, dan kegagalan yang hanya merugikan satu node. Kontrak script-nya identik, jadi
beralih ke in-process nanti tidak mengubah apa pun yang terlihat pengguna. Paket `pythonnet`
sudah dilepas supaya tidak ada dependensi yang menganggur.

**`TabularData` dipilih sebagai payload antar-modul, bukan `IDataView`.**
Transform, modul LLM, dan script pengguna semuanya perlu membaca dan menulis sel satu per satu.
Konversi ke `IDataView` hanya dilakukan tepat sebelum trainer membutuhkannya.

**Evaluasi dihitung dari tabel hasil scoring, bukan objek metrics ML.NET.**
ML.NET tidak memberi titik kurva ROC maupun sampel residual, padahal designer butuh keduanya.
Satu jalur kode menghasilkan angka ringkasan *dan* bentuk yang dibutuhkan chart — dan ikut
bekerja untuk tabel yang di-score di luar ML.NET.

### Bug yang ditemukan dan diperbaiki

| Bug | Perbaikan |
|---|---|
| `TryTopologicalSort` memakai parameter `out` di dalam lambda (CS1628) | List lokal yang di-assign ke `out` di awal method |
| Runner JavaScript mengevaluasi kode pengguna **dua kali** — sekali untuk hasil, sekali untuk `JSON.stringify` — sehingga efek sampingnya berjalan ganda | Dievaluasi sekali ke variabel global, lalu diserialisasi dari sana |
| **SQLite tidak mendukung `ORDER BY` atas `DateTimeOffset`.** Karena SQLite adalah provider bawaan dan hampir semua daftar diurutkan per waktu, ini memutus 5 dari 8 halaman | Value converter ke UTC ticks, dipasang otomatis hanya saat provider-nya SQLite |
| Database SQLite mendarat di `bin/App_Data` sementara storage di `App_Data/storage` folder proyek — dua folder berbeda, dan `dotnet clean` akan menghapus database tapi menyisakan dataset | `DatabaseProviderSetup` menerima content root; keduanya kini satu tempat |
| JS interop dipanggil saat prerender dan saat dispose komponen yang belum pernah interaktif → halaman designer balas 500 | Flag `_canvasReady`, plus dispose yang melewati interop bila kanvas tidak pernah menyala |
| `InferTypes` menandai kolom berisi "ya"/"tidak" sebagai Boolean, tapi `TextLoader` ML.NET tidak bisa mem-parse kata itu sebagai boolean → loader meledak dan **dua eksperimen churn gagal dilatih** | Hanya literal yang benar-benar bisa di-parse `TextLoader` yang dihitung boolean; "ya"/"tidak" tetap kategorikal dan ditangani one-hot encoding |
| Konstanta privat `Close` di `Icons` bentrok dengan ikon publik `Close` | Konstanta privat diganti nama jadi `End` |
| **Split Data menstratifikasi berdasarkan kolom kontinu menghasilkan 399/1, bukan 320/80.** Hampir tiap nilai jadi grupnya sendiri, tiap grup membulat seluruhnya ke keluaran pertama, dan keluaran kedua nyaris kosong. Run tetap "berhasil" dan melaporkan MAE 16,90 — dihitung dari **satu baris** | Dua perbaikan: seeder hanya menstratifikasi untuk task klasifikasi; dan `Split Data` menolak kolom stratifikasi yang rata-rata grupnya di bawah 4 baris, dengan pesan yang menyarankan kolom kategori atau split acak biasa. Metrik jujurnya sekarang MAE 67,19 dari 80 baris uji |
| Token tema dideklarasikan **tiga kali** — light, `[data-theme=dark]`, dan di dalam query `prefers-color-scheme` — tiga peluang untuk melenceng, dan versi media-query masih memakai pena lama yang belum divalidasi | Dikumpulkan jadi satu deklarasi `light-dark(terang, gelap)` per token; `data-theme` hanya menyetel `color-scheme` |
| Enam pena kategori dipilih dengan mata. Validator menemukan **lima dari enam pena dark mode di luar band lightness**, dan amber light mode contrast-nya 2,68:1 | Di-step ulang sampai kedua tema lolos band lightness, chroma floor, pemisahan CVD deutan/tritan, dan kontras 3:1 |
| Teks putih di atas latar pena pada header node dan `.tag--pen` — pena dipilih untuk kontras antar-warna, bukan terhadap putih | Pena jadi bilah/swatch identitas di samping teks; teksnya memakai tinta penuh |

### Empat bug yang ditemukan **oleh test suite**, setelah verifikasi manual menyatakan selesai

| Bug | Perbaikan |
|---|---|
| **Dropdown dataset selalu kosong.** `datasetId` dideklarasikan `ParameterKind.Choice` tanpa opsi statis, sementara cabang dinamis di `ParameterField` dikunci ke `ColumnName` — jadi cabang itu tidak pernah jalan dan `<select>`-nya dirender tanpa satu pun opsi. Modul Import Dataset praktis tidak bisa dikonfigurasi dari inspector | `ParameterKind.DatasetRef` baru yang menyatakan "opsinya datang saat runtime"; `ParameterField` merender daftar dataset dan menautkan ke halaman impor bila masih kosong. Tes katalog kini mewajibkan tiap `Choice` punya opsi dan default yang ada di dalamnya |
| **Pipeline tidak pernah menormalisasi vektor `Features`.** Trainer pohon tidak terpengaruh, tapi SDCA dan perceptron underfit parah: data yang sama memberi ~0,95 lewat pohon dan ~0,75 lewat trainer linear. Terbaca seolah "algoritma itu jelek", padahal inputnya yang tidak diskalakan | `NormalizeMinMax` atas `Features` sebelum trainer. Iris naik 96,67% → 100%; semua trainer linear kini lolos ambang |
| **Hasil clustering tidak pernah sampai ke tabel.** `MlDataBridge.IsReadable` menyaring berdasarkan subclass `DataViewType`, sehingga kolom bertipe key — yang dipakai K-Means sebagai indeks klaster — ikut terbuang diam-diam | Penyaringan berdasarkan tipe CLR-nya, plus getter untuk seluruh tipe integer. Evaluasi clustering akhirnya berjalan |
| Indeks klaster keluar dengan nama `PredictedLabel`, sama seperti label klasifikasi, sehingga tabel hasil tidak bisa dibedakan tugasnya | Diganti nama jadi `PredictedClusterId` saat scoring |

### Dua bug lagi, ditemukan tes layer web

| Bug | Perbaikan |
|---|---|
| **Endpoint terbitan menolak permintaan yang sah dengan 422.** Pipeline menyalin kolom label, sehingga skema inputnya menuntut kolom itu ada — termasuk saat inferensi, ketika label justru yang sedang diprediksi. Artinya pemanggil API harus mengirim jawabannya untuk bisa bertanya. Modul Score Model juga tidak bisa men-score data produksi yang belum berlabel | `MlDataBridge.EnsureLabelColumn` menambahkan kolom label berisi placeholder bila baris datang tanpanya, dipakai baik oleh API maupun modul Score Model. Tes memastikan prediksinya identik dengan dan tanpa label, jadi placeholder-nya benar-benar tidak berpengaruh |
| Tes komponen mengonfirmasi ulang bug dropdown dataset dari sisi render, dan kini `ParameterField` diuji parametrik atas **seluruh** `ParameterKind` | Sebuah kind baru yang tidak ditangani langsung menggagalkan tes, bukan diam-diam merender kontrol yang salah |

### Fase 10 — Reset password dan riwayat ✅

Dua item spec yang sebelumnya terlewat, ditemukan lewat audit ulang terhadap `requirements.txt`.

**Reset password.** Spec menyebut `Auth: … Reset Password`; yang ada baru ganti-sandi dari
halaman profil. Sekarang lengkap: halaman lupa-sandi, tautan bertoken, halaman pilih sandi baru,
lalu langsung masuk.

Dua keputusan yang membentuknya:

- **Jawabannya identik** apakah alamat itu terdaftar atau tidak. Kalau berbeda, halaman ini jadi
  cara mengetahui siapa saja yang punya akun di instalasi tersebut. Berlaku juga di endpoint
  penyetelan: alamat tak dikenal ditolak sama persis seperti token kedaluwarsa.
- **Tanpa server email, alurnya tetap jalan** — tautannya ditulis ke log aplikasi dengan level
  Warning dan kalimat yang menjelaskan kenapa. Aplikasi ini di-deploy sendiri oleh penggunanya,
  sering tanpa server email di tangan, dan reset yang sekadar gagal akan mengunci administrator
  dari instalasinya sendiri. Section Email baru di Pengaturan mengatur SMTP-nya.

**Riwayat run dan versi.** `RunsAsync`, `VersionsAsync`, dan `RestoreVersionAsync` sudah ditulis
sejak Fase 2–3 dan **tersambung ke nol tombol** — datanya direkam, tidak ada yang bisa melihatnya.
Halaman `/eksperimen/{id}/riwayat` sekarang menampilkan keduanya:

| Tab | Isi |
|---|---|
| **Run** | Tabel semua run dengan metriknya, plus grafik perbandingan satu metrik antar-run. Bar chart seri tunggal — run terbaik menonjol lewat bobot warna, bukan hue kedua, jadi tidak ada palet kategorikal yang bisa keliru. Baseline selalu di nol; memotongnya akan melebih-lebihkan selisih kecil. |
| **Versi** | Daftar versi dengan catatan dan jumlah modul, dan tombol kembalikan. Mengembalikan bersifat **append-only** — hasilnya jadi versi baru di puncak, jadi langkah itu sendiri bisa dibatalkan. Tesnya memeriksa properti itu, bukan sekadar bahwa tombolnya bekerja. |

Ini yang membuat dua klaim spec berdiri: *"membandingkan hasil model secara visual"* dan
reproducibility yang sebelumnya benar di level data tapi tidak bisa disentuh pengguna.

### Fase 9 — Kanvas D3 ✅

Tiga bug ditemukan begitu browser sungguhan menjalankan halaman itu:

| Bug | Perbaikan |
|---|---|
| **Theme toggle dan menu pengguna sama sekali tidak berfungsi.** Keduanya ada di `MainLayout`, dan render mode sebuah halaman **tidak mengalir naik ke layout-nya** — jadi keduanya dirender statis dan handler `@onclick`-nya tidak pernah terpasang. Fitur yang dijanjikan spesifikasi (dark/light) mati total, dan tidak satu pun tes unit maupun komponen bisa menemukannya: bUnit merender komponen secara interaktif menurut definisinya | `@rendermode InteractiveServer` pada `ThemeToggle` dan `UserMenu`, menjadikan keduanya pulau interaktif sendiri |
| **Panel chat mengaku tersambung di instalasi baru.** Ollama dianggap terkonfigurasi hanya karena `Endpoint`-nya punya nilai bawaan, sehingga pengguna baru melihat asisten yang tampak siap lalu gagal di pesan pertama | `OllamaOptions.Enabled` yang harus dinyalakan sadar — menjalankan Ollama lokal adalah klaim yang hanya bisa dibuat penggunanya. Ditambah sakelarnya di halaman Pengaturan |
| **Klik dua modul palette menaruh keduanya di koordinat identik**, jadi node kedua tersembunyi persis di bawah yang pertama dan kanvas tampak tidak bereaksi | Posisi node baru kini bertingkat |

### Fase 8 — Parquet, script runtime, vision/NLP ✅

| Item | Hasil |
|---|---|
| **Impor & ekspor Parquet** | Lewat Parquet.Net (14 MB). Kolumnar di disk, baris di memori, jadi kedua arah men-transpose. Berbeda dari CSV, Parquet membawa skemanya sendiri — kolom numerik kembali numerik tanpa perlu ditebak ulang dari nilainya. |
| **Python & R** | Sudah berjalan sebagai proses anak sejak awal; yang ditambahkan di sini **tesnya**. Python 3.12 ada di mesin ini, jadi tesnya menjalankan interpreter sungguhan — termasuk membuktikan `while True: pass` dibunuh saat timeout, yang justru alasan utama memilih proses anak. |
| **AutoML vision & NLP** | Image Classification (TensorFlow) dan Text Classification (TorchSharp), keduanya transfer learning. **Opt-in build** — lihat catatan ongkos di bawah. |

**Kenapa vision/NLP dibuat opt-in.** Diukur, bukan dikira-kira:

| Paket | Ukuran |
|---|---|
| Parquet.Net | 14 MB |
| SciSharp.TensorFlow.Redist | **864 MB** |
| libtorch-cpu (win-x64 saja) | **319 MB** |

~1,2 GB binary native untuk dua modul, sekitar dua puluh kali sisa restore solusi ini — dan itu
baru satu platform. Memaksakannya ke semua orang salah; menghilangkan fiturnya juga salah. Jadi
modulnya **selalu ada di katalog** (bisa ditemukan, terdokumentasi, tervalidasi), sementara
backbone-nya di balik `#if VISION_NLP` yang dinyalakan dengan `-p:EnableVisionNlp=true`. Tanpa
flag itu, mencoba menjalankannya memberi pesan yang menyebutkan cara mengaktifkan **dan** ongkosnya.

Jalur opt-in itu **sudah dibangun dan diverifikasi compile**, bukan sekadar ditulis — dan
verifikasi itu langsung menemukan satu galat: extension `TextClassification` butuh namespace
`Microsoft.ML.TorchSharp` di-scope, yang tidak akan pernah ketahuan kalau saya hanya menulisnya.

### Satu bug lagi, ditemukan tes browser untuk riwayat

| Bug | Perbaikan |
|---|---|
| **Tes kanvas saling merusak.** Semuanya menunjuk satu eksperimen contoh yang sama, dan sebagian besar menyuntingnya — jadi node yang ditambahkan satu tes membuat run tes berikutnya gagal, dan "eksperimen ini belum pernah dijalankan" berhenti benar setelah tes pertama yang menjalankannya. Hasilnya bergantung pada urutan | Tiap tes kini mengambil salinannya sendiri lewat tombol "Pakai contoh ini" di galeri — sekaligus menguji jalur klon itu |

Ditambah satu kesalahan proses yang layak dicatat: saya sempat membaca hasil `dotnet test --no-build`
padahal build-nya gagal, jadi yang dijalankan adalah biner lama dan kegagalannya tampak tidak
berubah meski kodenya sudah diperbaiki. Sejak itu build selalu diperiksa sukses lebih dulu.

### ⚠️ Temuan paling serius: kunci API sungguhan di `appsettings.json`

Ditemukan oleh `ShippedSettingsTests`, yang membaca file yang benar-benar dikirim alih-alih
objek options bawaan.

| | |
|---|---|
| **Apa** | Kunci OpenAI (164 karakter) dan kunci Tavily (41 karakter), keduanya aktif, tertulis di `src/BlazorML.Web/appsettings.json` — file yang README suruh orang `git clone`. |
| **Sejauh mana** | Direktori ini **belum** repositori git, jadi belum ter-commit dan belum ter-push. Kuncinya hanya ada di disk lokal. |
| **Tindakan** | Dipindahkan ke **user-secrets** (`UserSecretsId` sudah dikonfigurasi di `BlazorML.Web.csproj`), lalu dikosongkan dari file. Aplikasi berperilaku persis sama — user-secrets menimpa `appsettings.json` di environment Development — tapi kuncinya tidak lagi berada di file yang akan ikut ter-commit. |
| **Yang perlu Anda lakukan** | Pertimbangkan **memutar (rotate) kedua kunci itu**. Keduanya sempat berada dalam teks biasa di file proyek; memutarnya murah, dan menghilangkan seluruh keraguan. |
| **Penjaga ke depan** | `ShippedSettingsTests` gagal bila kredensial mana pun muncul lagi di file itu. Ditambah `.gitignore` yang sebelumnya tidak ada sama sekali — tanpa itu, database, `bin/`, dan seluruh isi `App_Data/` juga akan ikut ter-commit. |

Ini juga menjelaskan kenapa panel chat mengaku terkonfigurasi selama pengujian: kuncinya memang
ada dan terbaca. Aplikasinya benar; filenya yang tidak seharusnya menyimpannya.

### Satu bug lagi, ditemukan tes layer agent

| Bug | Perbaikan |
|---|---|
| **Validasi menandai parameter opsional sebagai wajib.** `IsRequired` menebak dari "bertipe teks dan tanpa default", sehingga field yang memang opsional — `positiveClass` di Evaluate Model, `newName` di Edit Metadata, `stratifyColumn` di Split Data, `description` di Register Model — dilaporkan sebagai "masih perlu diisi" dan **memblokir run yang sebenarnya sah**. Validasi yang berteriak soal hal yang tidak wajib adalah validasi yang lama-lama diabaikan orang | Requiredness kini **dideklarasikan**, bukan ditebak: `ParameterSpec.Required`, dengan 25 parameter yang benar-benar wajib ditandai eksplisit. Dua invarian katalog baru menjaganya — parameter wajib tidak boleh punya default (kontradiktif), dan daftar field opsional yang dulu salah ditandai kini diuji langsung |

### Menjalankan seluruh trainer: dua bug dan satu deskripsi yang menyesatkan

Sampai putaran ini, **dua belas dari 27 trainer di katalog belum pernah sekali pun dijalankan**.
Mereka ada di palette, tervalidasi oleh invarian katalog, dan kompilasinya bersih — tapi tidak ada
satu pun yang pernah diminta fit sebuah model. Begitu sapuan parametrik menyentuh semuanya, tiga
hal langsung muncul.

| Temuan | Perbaikan |
|---|---|
| **PCA anomaly detection meledak dengan pesan ML.NET mentah.** Default `rank` adalah 5, dan pada dataset dengan 2 kolom fitur ML.NET melempar `"Rank (5) cannot be larger than the original dimension (2)"` — istilah yang tidak muncul di mana pun di UI, pada parameter yang defaultnya sendiri yang salah. Pengguna yang menarik modul ini ke kanvas dan menekan Run tanpa menyentuh apa pun **selalu** kena, kecuali datasetnya kebetulan lebar | `TrainingExecutors.Fit` menurunkan rank ke jumlah fitur yang benar-benar ada dan **mencatatnya sebagai peringatan di log run**. Run tetap jalan, dan pengguna tahu persis apa yang disesuaikan — bukan diam-diam diperbaiki, bukan pula gagal |
| **Spike detection tidak bisa dibaca hasilnya.** Ia mengeluarkan `Vector<Double, 3>`, sementara `MlDataBridge.BuildGetter` mengasumsikan setiap kolom vektor berisi `float`. Hasilnya `"Invalid TValue in GetGetter for column #SpikeVector"` — pelatihan berhasil, lalu pembacaan hasilnya yang gagal, jadi kegagalannya muncul setelah pekerjaan mahal selesai | Getter kini menangani vektor `float` **dan** `double`. Ini varian dari jebakan yang sudah tercatat di CLAUDE.md — `IsReadable` dulu keliru berpatokan pada subclass `DataViewType`; kali ini yang keliru adalah asumsi tipe elemennya |
| **Naive Bayes hanya mencapai 0,333 pada tiga kelas** — persis angka tebak-tebakan. Ini bukan bug: implementasi ML.NET **membinerkan setiap fitur**, jadi ia cuma melihat ada-atau-tidak, tidak pernah seberapa besar. Yang salah adalah deskripsi katalognya, yang berbunyi seperti classifier serba-guna | Deskripsi modul kini menyatakan batasan itu terus terang: untuk fitur ada/tidak-ada seperti n-gram teks, dan ia mengabaikan besar-kecilnya nilai. Tesnya memberi fitur berbentuk presence/absence, dengan alasan tertulis di tempatnya — menilai trainer ini pada kolom kontinu berarti menguji ketidakcocokan yang sudah kita dokumentasikan, bukan menguji wiring-nya |

Satu kesalahan tes milik saya sendiri, dicatat karena sempat terlihat seperti bug produk: saya
menuntut tabel hasil Cross Validate berisi empat baris (satu per fold), padahal isinya **satu baris
per metrik**. Assertion-nya diganti dengan yang benar-benar memeriksa bentuk tabel itu —
kolom `metric`/`mean`/`stdDev`/`min`/`max` — beserta invarian min ≤ max.

### Setiap input teks di aplikasi tidak pernah menerima design system

Dilaporkan pengguna: `<input>` polos tampil tanpa sudut membulat dan lebih besar, sementara
`select`, `input type=number`, `textarea` dan checkbox semuanya benar.

| Bug | Perbaikan |
|---|---|
| **Aturan input mendaftar tipe satu per satu**: `input[type='text'], input[type='email'], …`. Itu **attribute selector** — ia menuntut atributnya benar-benar tertulis. `<input @bind="x" />`, cara biasa menulis field teks di Razor, menghasilkan `<input>` **tanpa atribut `type` sama sekali**: berperilaku sebagai text, tapi tidak cocok dengan satu pun selector itu, lalu jatuh ke kotak bawaan browser. **25 field** kena — seluruh Pengaturan, Profil, form S3, nama endpoint di halaman Model, dan `ParameterField`, yaitu inspector parameter di kanvas designer | Aturannya dibalik: menamai apa yang **bukan** kotak teks (`checkbox`, `radio`, `range`, `file`, `color`, tombol) dan menata sisanya. Sekaligus menutup jebakan yang sama untuk tipe yang belum dipakai siapa pun — `tel`, `time`, `datetime-local` kini ikut tertata, bukan diam-diam terlewat |

Ikut diperbaiki di aturan yang sama: `input:focus { outline: none }` sebelumnya berlaku untuk
**semua** input. Bayangan keras yang menggantikannya terbaca sebagai fokus pada kotak berbingkai,
tapi tidak pada checkbox 17 px — jadi pengguna keyboard kehilangan penanda fokus di sana. Sekarang
aturan fokus memakai daftar pengecualian yang sama, dan checkbox, radio serta slider
mempertahankan cincin `:focus-visible` amber.

Ini kelas bug yang tidak bisa dilihat alat mana pun selain browser: selector yang tidak cocok
**tidak menghasilkan galat di mana pun**. Tidak ada peringatan build, tidak ada exception, dan
markup-nya identik — jadi bUnit melihat semuanya beres. Sebelas tes browser baru mengukur gaya
terkomputasi, dan sudah diverifikasi bisa gagal: dengan selector lama dikembalikan, pesannya
persis keluhan yang dilaporkan — *"A bare &lt;input&gt; has square corners."*

### Pratinjau baris di halaman Dataset

Halaman Dataset dulu hanya bisa menampilkan profil kolom — statistik tentang data, tanpa pernah
memperlihatkan datanya. Sekarang tombol **Lihat data** membuka pratinjau baris, dan dialognya jadi
bertab: **Data** dan **Kolom**. Keduanya tampilan atas hal yang sama, dan membaca barisnya selalu
memunculkan pertanyaan tentang tipe kolomnya — jadi keduanya di balik satu pasang tab, bukan dua
dialog yang tidak bisa dibolak-balik.

Pilihan jumlah baris 25 / 100 / 500, dan mengubahnya **membaca ulang ke service**, bukan memotong
tabel yang sudah ada — kalau memotong, pilihan 500 tidak akan pernah menampilkan baris ke-26.

Detail kecil yang disengaja: sel kosong dirender sebagai em dash yang diredupkan, bukan dibiarkan
kosong. Sel kosong terbaca seperti kesalahan render, padahal gap adalah fakta tentang datanya yang
diperhatikan setiap transform di hilir. Kolom angka rata kanan supaya besarannya bisa dibandingkan
ke bawah, memakai tipe dari profil tersimpan — penilaian yang sama dengan yang dipakai saat melatih.

**`IDatasetService.PreviewAsync` ternyata belum pernah dipanggil dari mana pun.** Ada di interface
dan terimplementasi sejak awal, tidak pernah dieksekusi. Halaman ini pemanggil pertamanya, dan
pratinjau yang diam-diam membaca seluruh unggahan 512 MB ke dalam circuit Blazor demi 25 baris
adalah cara yang buruk untuk mengetahuinya di produksi. Sekarang ada 10 tes untuknya.

### Pembaca JSON membaca seluruh berkas meski dibatasi jumlah barisnya

Ditemukan karena **mengukur**, bukan membaca kode. Tes lama bernama `A_row_limit_stops_reading_early`
sebenarnya hanya menghitung baris yang kembali — memotong tabel yang sudah di-parse penuh akan
lolos dengan sama mulusnya. Tesnya diganti nama menjadi apa yang benar-benar diperiksa, dan tes
baru mengukur berapa byte stream yang benar-benar dikonsumsi.

| Bug | Perbaikan |
|---|---|
| **`ReadJsonAsync` memanggil `JsonDocument.ParseAsync`, yang menarik dan mem-parse seluruh dokumen sebelum baris pertama dilihat.** `break` pada `rowLimit` hanya menghemat penyusunan tabel, bukan pembacaannya. Terukur: membaca 25 dari 20.000 baris mengonsumsi **1.577.781 dari 1.577.781 byte** — seluruhnya. CSV dan TSV memang berhenti lebih awal | Array JSON tingkat atas — bentuk yang hampir selalu dihasilkan sebuah ekspor — kini di-stream lewat `DeserializeAsyncEnumerable` dan berhenti begitu barisnya cukup. Bentuk terbungkus (`{"data":[…]}`) tidak bisa di-stream karena array-nya bisa saja properti terakhir, jadi tetap lewat jalur lama; keduanya dibedakan dengan mengintip satu karakter bermakna pertama lalu memundurkan stream, dan hanya kalau stream-nya bisa di-seek |

Satu tes browser saya sendiri sempat hampa dan itu tercatat: pengujian "nama kolom tetap terlihat
saat di-scroll" **lolos juga ketika pembatas tingginya dihapus** — tanpa cap tidak ada yang meluap,
tidak ada yang ter-scroll, dan "header tidak bergerak" jadi benar secara trivial. Sekarang tesnya
memastikan dulu bahwa `scrollTop` benar-benar bergerak, dan dengan itu ia gagal sebagaimana mestinya.

### Halaman Endpoint: uji coba API dan contoh kode empat bahasa

Halaman ini sebelumnya hanya menampilkan satu perintah `curl` dengan `{"kolom1":123,"kolom2":"nilai"}`
— bentuk yang benar, isi yang tidak berarti apa-apa. Siapa pun yang menyalinnya tetap harus pergi
mencari sendiri nama kolom modelnya sebelum ada yang jalan.

Sekarang: tab **cURL · C# · Python · Node.js**, masing-masing dengan tombol salin, dan panel uji
coba berisi kunci API, badan permintaan yang bisa disunting, serta responsnya lengkap dengan
status dan waktu tempuh.

**Yang membuatnya berguna bukan tab-nya, tapi payload-nya.** `InputSchemaJson` dulu hanya menyimpan
`[{"name":"x"}]`. Sekarang ia menyimpan tipe kolom **dan satu nilai sungguhan dari data latih**,
diambil dari baris pertama yang memang berisi — bukan baris pertama begitu saja, karena baris
pertama dataset nyata sama mungkinnya berisi gap. Hasilnya snippet yang tinggal ditempel dan jalan.
Skema lama tetap terbaca: `ModelInputSchema.Parse` toleran terhadap bentuk yang hanya punya nama.

Beberapa keputusan yang layak dicatat:

| Keputusan | Alasan |
|---|---|
| Uji coba dikirim lewat **HTTP sungguhan** ke rute aplikasi sendiri, bukan memanggil kode scoring langsung | Yang paling mungkin salah justru ada di depan model: kunci, nama header, dan apakah endpoint-nya aktif. Memanggil scoring langsung akan berhasil sambil membiarkan ketiganya rusak |
| Badan permintaan divalidasi di sisi klien sebelum dikirim | Meneruskan JSON yang jelas rusak lalu melaporkan 400 dari server sama dengan menyuruh pengguna mendiagnosis sendiri |
| 401 ditampilkan sebagai **respons**, bukan kegagalan | Itu jawaban yang benar dari API. Menampilkannya sebagai gangguan jaringan mengirim orang mencari masalah yang tidak ada |
| Snippet **tidak** diberi syntax highlighting | Di sistem desain ini warna berarti kategori. Enam warna berebut di dalam snippet enam baris tidak menyampaikan apa pun sambil membantah setiap permukaan lain |
| Kunci tidak pernah masuk ke snippet | Yang tercetak selalu `KUNCI_API_KAMU`. Snippet ini ditempel ke chat dan issue tracker; ada tes yang menggagalkan build kalau `bml_` pernah muncul di sana |
| Membuat kunci baru langsung mengisikannya ke panel uji coba endpoint itu | Itu satu-satunya saat kunci polos ada. Menyuruh menyalinnya lalu menempelkannya dua baris di bawah adalah pekerjaan yang tidak perlu |

### Satu bug lagi, dan hanya browser yang bisa melihatnya

| Bug | Perbaikan |
|---|---|
| **Badan permintaan dirender sebagai isi elemen `<textarea>`.** Begitu sebuah textarea pernah diketik, browser menandainya *dirty* dan berhenti memantulkan perubahan pada child text node-nya. Jadi tombol "Kembalikan contoh" diam-diam tidak melakukan apa-apa — kotaknya tetap berisi apa pun yang tadi diketik. Markup-nya berubah dengan benar, jadi bUnit melihat semuanya beres | Payload dipasang lewat atribut `value`, yang Blazor tetapkan sebagai properti DOM. Diverifikasi dengan mengembalikan bug-nya sebentar: tes browsernya langsung merah, dan `InputValueAsync` memang mengembalikan teks lama |

Satu kesalahan proses ikut tercatat: setelah mengembalikan perbaikan itu dengan `mv`, timestamp
berkasnya ikut mundur, build inkremental melewatinya, dan tes gagal terhadap biner lama — varian
dari pelajaran `--no-build` yang sudah pernah kena sebelumnya. `touch` lalu build ulang.

### Atribusi di app shell — dan bug yang ketahuan karena mengukur tinggi halaman

Kredit pembuat sebelumnya hanya ada di halaman masuk, tertulis keras-keras, padahal
`WorkspaceOptions.BuiltBy` dan `LedBy` sudah lama ada dan **tidak pernah dibaca satu baris kode
pun** — dua sumber kebenaran yang bebas berbeda tanpa ada yang tahu. Sekarang ada satu komponen
`Attribution` yang membaca opsi itu, dipakai di app shell dan di halaman masuk.

Bagian yang tidak diduga: menambahkan bar setinggi 34 px ke shell mengubah tinggi yang harus
dipatuhi setiap halaman setinggi-viewport. Tes yang memeriksa kanvas designer tidak tumbuh
scrollbar langsung gagal — **dengan selisih 60 px, bukan 34**. Selisih 26 px itu yang membuka dua
bug yang sudah lama ada dan tidak ada hubungannya dengan atribusi.

| Bug | Perbaikan |
|---|---|
| **`SectionOutlet` dan `SectionContent` bukan komponen, melainkan elemen HTML tak dikenal.** `_Imports.razor` tidak pernah meng-import `Microsoft.AspNetCore.Components.Sections`, dan Razor **tidak memberi peringatan apa pun** untuk tag PascalCase yang tidak dikenal — ia menuliskannya apa adanya. Akibatnya `<sectionoutlet>` di top bar selalu kosong, dan judul halaman beserta tombol Jalankan/Simpan/asisten dirender di badan halaman. Selama ini terlihat disengaja, karena bloknya mendarat tepat di bawah bar seperti baris kedua | Satu baris `@using` di `_Imports.razor`, plus komentar yang menjelaskan kenapa hilangnya tidak bersuara |
| **Memperbaiki hal di atas mematikan seluruh tombol top bar.** Konten section hanya sampai ke outlet di renderer yang sama. Outlet-nya ada di `MainLayout`, yang **dirender statis** — jebakan yang sudah tercatat: layout tidak mewarisi render mode halaman. Tombolnya tampil, tidak satu pun berfungsi. Enam tes kanvas menangkapnya seketika | Outlet dibungkus `TopBarSlot` yang membawa `@rendermode InteractiveServer` sendiri, sejajar dengan `ThemeToggle` dan `UserMenu`. Semua halaman yang mengisi bar ini memang `InteractiveServer`, jadi keduanya berada di renderer yang sama |

Tinggi bar kredit dideklarasikan sebagai token `--credit`, dan `designer.css` menguranginya dari
`100vh` bersama `--header` — dengan komentar yang menyebutkan kenapa melewatkan salah satunya
memberi scrollbar pada permukaan yang seharusnya di-pan, bukan di-scroll.

### Edge kanvas lepas dari node saat di-zoom — dilaporkan pengguna

| Bug | Perbaikan |
|---|---|
| **Transform pan/zoom dipasang pada elemen `<svg>` terluar.** Node adalah HTML dan edge adalah SVG, jadi tiap perubahan tampilan harus diterapkan dua kali di dua sistem koordinat — dan `transform` pada `<svg>` terluar bukan operasi yang sama dengan `transform` pada sebuah group. Lapisan edge melenceng dari node begitu kanvas diskalakan: diukur, ujung panah berjarak **2–16 px** dari port-nya pada zoom 1, lalu **122–151 px** hanya setelah satu langkah zoom in. Panahnya terlihat putus | Group `<g class="dz-edge-layer">` di dalam svg, dan transform dipasang di situ. Semua path memang sudah digambar di koordinat dunia, jadi tidak ada geometri yang perlu diubah — hanya tempat transform-nya menempel. Setelah perbaikan jaraknya 2–19 px di seluruh rentang zoom |

Tesnya (`ZoomTests`) mengukur jarak tiap ujung edge ke port terdekat **dalam satuan dunia**, bukan
piksel layar: anggaran piksel tetap akan lolos di 0,25× karena alasan yang salah dan gagal di 2,5×
tanpa alasan. Dan sebelum dianggap selesai, bug-nya dikembalikan sebentar untuk memastikan tes ini
**benar-benar bisa gagal** — tiga dari empat langsung merah. Tes regresi yang tidak pernah bisa
gagal tidak menjaga apa pun; suite kanvas ini pernah kena persis masalah itu sekali.

Yang **tidak** dikerjakan, dan alasannya: `BuiltBy`/`LedBy` sengaja tidak diberi field di halaman
Pengaturan. Nama workspace dan tagline memang milik yang memasang aplikasinya, tapi kredit
pembuat adalah fakta tentang siapa yang membangunnya — memberi tombol suntingnya di UI sama saja
menyediakan cara menghapus atribusi yang justru diminta spesifikasi. Nilainya tetap bisa diubah
lewat konfigurasi oleh yang benar-benar perlu.
