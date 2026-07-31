# SplatStudio

Mengubah satu foto yang diunggah menjadi point cloud 3D Gaussian-splat yang bisa diputar-putar (orbit), langsung di browser — lengkap dengan galeri, komentar, rating 1-5 bintang, manajemen akun penuh, serta backend database dan storage yang bisa dipasang-tukar (pluggable).

Dibangun dengan **.NET 8** (Blazor Web App, render mode Interactive Server) menggunakan Clean Architecture (Domain / Application / Infrastructure / Web).

> 🇬🇧 Read in English: [README.md](README.md)

---

## Daftar isi

1. [Apa yang sebenarnya dilakukan aplikasi ini (mohon dibaca dulu)](#apa-yang-sebenarnya-dilakukan-aplikasi-ini-mohon-dibaca-dulu)
2. [Fitur](#fitur)
3. [Arsitektur](#arsitektur)
4. [Memulai](#memulai)
5. [Konfigurasi](#konfigurasi) — termasuk [Mode konversi](#mode-konversi)
6. [Akun demo & data contoh](#akun-demo--data-contoh)
7. [Memasang backend 3D Gaussian Splatting yang sesungguhnya](#memasang-backend-3d-gaussian-splatting-yang-sesungguhnya)
8. [Catatan performa](#catatan-performa)
9. [Keterbatasan & penyederhanaan v1](#keterbatasan--penyederhanaan-v1)
10. [Catatan deployment](#catatan-deployment)

---

## Apa yang sebenarnya dilakukan aplikasi ini (mohon dibaca dulu)

**3D Gaussian Splatting** yang sesungguhnya (teknik di balik Luma AI atau nerfstudio/gsplat) dilatih dari *banyak* foto subjek yang sama dari berbagai sudut. Pipeline seperti COLMAP pertama-tama merekonstruksi posisi kamera dan point cloud kasar lewat Structure-from-Motion, lalu ribuan Gaussian dioptimasi selama berjam-jam di GPU dengan gradient descent terhadap semua sudut pandang tersebut. Satu foto datar saja tidak mengandung informasi multi-sudut yang dibutuhkan untuk merekonstruksi geometri 3D yang sesungguhnya — sepintar apa pun kode yang ditulis, ini tidak bisa diubah.

Yang SplatStudio sediakan sebagai mode bawaannya adalah rekonstruksi "2.5D" yang cepat, sepenuhnya offline, dan deterministik, dalam dua implementasi yang saling menggantikan: `LocalHeuristicSplatEngine` (CPU) dan `GpuSplatEngine` (perhitungan yang sama persis sebagai kernel komputasi ILGPU). Untuk setiap gambar yang diunggah:

1. Mengecilkan ukuran gambar sesuai budget jumlah titik (`Splatting:MaxPoints` / `GpuMaxPoints`).
2. Mengestimasi **pseudo-depth** per piksel sebagai campuran dari invers-luminance dan radial prior yang lebih berat di tengah (piksel yang lebih terang dan lebih ke tengah ditarik lebih dekat ke kamera).
3. Menghaluskan depth field tersebut dengan box blur kecil.
4. Menghasilkan satu Gaussian splat per piksel, diposisikan sesuai depth field tersebut, dengan warna dari piksel sumbernya.
5. Menulis hasilnya sebagai file biner `.splat` 32 byte per titik, dirender di browser dengan viewer point-cloud Three.js kecil yang ditulis khusus untuk proyek ini (`wwwroot/js/splat-viewer.js`).

Hasilnya adalah dunia point-cloud 3D yang sungguh-sungguh bisa diputar bebas — dan cukup meyakinkan untuk potret atau objek tunggal dengan latar polos. Namun ini **bukan** pengganti rekonstruksi multi-sudut yang sesungguhnya, dan tidak akan menghasilkan depth/occlusion yang akurat untuk skena kompleks. Anggap saja ini efek "foto jadi snow-globe" yang bergaya, bukan photogrammetry.

Engine GPU **lebih cepat, bukan lebih akurat**: perkiraan yang sama persis dengan batasan yang sama persis. Menjalankannya di GPU membeli throughput — itulah yang membuat budget 250.000 titik jadi masuk akal — bukan membeli akurasi.

Kalau Anda memang butuh rekonstruksi sungguhan, dua mode lainnya mengirim gambar ke layanan hosted yang menjalankan model generatif betulan — lihat [Mode konversi](#mode-konversi). Aplikasi juga menyatakan semua ini di halaman `/about`, supaya peringatannya sampai ke orang yang tidak pernah membuka berkas ini.

## Fitur

- **Unggah → konversi → lihat**: unggah JPEG/PNG/WebP, aplikasi mengantrekan job konversi di background, dan galeri/viewer ter-update otomatis (lewat `ISceneUpdateNotifier` in-process) begitu selesai — tanpa refresh manual.
- **Tiga mode konversi**, dipilih per unggahan: heuristik kedalaman bawaan (instan, gratis, offline), 3D Gaussian splatting fotorealistik lewat layanan hosted, atau **mesh** 3D lewat model seperti TRELLIS, Hunyuan3D, atau Rodin. Lihat [Mode konversi](#mode-konversi).
- **Galeri**: grid publik dari skena yang sudah selesai, masing-masing menampilkan thumbnail, rating rata-rata, jumlah view, dan jumlah komentar.
- **Dua viewer 3D**, keduanya drag untuk orbit dan scroll untuk zoom: renderer point-cloud ringan yang ditulis khusus untuk format `.splat` proyek ini (tanpa library splat-viewer eksternal), dan viewer glTF kecil untuk skena mesh yang menambahkan pencahayaan serta memuat parser-nya sesuai kebutuhan. Three.js di-vendor secara lokal, bukan diambil dari CDN, sehingga keduanya tidak bergantung pada pihak ketiga.
- **Komentar & rating**: pengguna yang sudah login bisa meninggalkan komentar dan rating 1-5 bintang per skena (satu rating per pengguna per skena).
- **Sistem akun penuh** lewat ASP.NET Core Identity: registrasi, login/logout, lupa/reset password (lewat email sender yang bisa ditukar), edit profil (nama tampilan, bio, unggah avatar), ganti password.
- **My Scenes**: kelola unggahan Anda sendiri — toggle publik/privat, hapus (membersihkan baris database sekaligus file yang tersimpan).
- **Tiga provider database**: SQLite (default tanpa konfigurasi), SQL Server, MySQL — tinggal ganti satu nilai konfigurasi.
- **Empat provider storage**: filesystem lokal (default tanpa konfigurasi), Azure Blob Storage, AWS S3, dan MinIO/layanan kompatibel-S3 apa pun — tinggal ganti satu nilai konfigurasi.
- **UI "Depth Ramp"**: design system custom yang paletnya adalah output produk ini sendiri — dekat itu amber, tengah rose, jauh indigo, sesuai cara depth map lazim dibaca. Hero di halaman depan adalah skena sungguhan yang dirender langsung dan berayun pelan, bukan mockup, lengkap dengan jumlah splat dan waktu konversi yang sebenarnya.

## Arsitektur

```
src/
  SplatStudio.Domain          Entity + enum saja, tanpa dependensi
  SplatStudio.Application     Port (interface) yang diimplementasikan oleh layer Web/Infrastructure
  SplatStudio.Infrastructure  EF Core, Identity, provider storage, splat engine, background worker
  SplatStudio.Web             Komponen Blazor, Razor Pages untuk auth, aset statis
tests/
  SplatStudio.Tests           Format splat, engine, kesetaraan GPU/CPU, mode konversi,
                              validitas mesh contoh, benchmark
```

Beberapa keputusan yang disengaja dan perlu dijelaskan:

- **Tanpa MediatR/CQRS.** Mengingat proyek ini sudah cukup besar (3 provider DB × 4 provider storage × auth lengkap × galeri real-time), Clean Architecture yang disederhanakan tanpa pipeline mediator membuat codebase lebih mudah dibaca dari ujung ke ujung. Pemisahan port/adapter (interface di `Application`, implementasi di `Infrastructure`) tetap dipertahankan, sehingga menambahkan CQRS nanti adalah penambahan struktural, bukan menulis ulang dari awal.
- **Hosting model Blazor Web App + Razor Pages klasik untuk auth.** Login/Register/Logout/ResetPassword perlu menulis cookie autentikasi langsung pada response HTTP, yang hanya berfungsi *sebelum* koneksi SignalR dari circuit Blazor Server mengambil alih. Karena itu kelima flow tersebut adalah Razor Pages ASP.NET Core biasa (`Pages/Account/*.cshtml`), distyling agar serasi dengan tampilan aplikasi lainnya, sementara semua halaman lain adalah komponen Blazor Interactive Server. Ini mengikuti pola template resmi `dotnet new blazor -au Individual -int Server`.
- **Satu background worker, bukan pool.** Konversi gambar→splat adalah CPU-bound. Menjalankan banyak proses ini paralel pada satu instance akan merebut CPU dari circuit SignalR Blazor Server di proses yang sama. `ConversionBackgroundService` sengaja memproses satu skena dalam satu waktu; untuk scale-out, tambah jumlah instance, bukan jumlah worker per instance. Ini kurang berpengaruh pada engine GPU, di mana yang jadi hambatan justru decode gambar di sisi CPU.
- **`EnsureCreatedAsync()` alih-alih migration EF.** Dengan tiga provider database yang bisa ditukar, memelihara tiga riwayat migration terpisah menambah biaya maintenance yang nyata untuk manfaat yang relatif kecil di v1. Skema dibuat langsung dari model saat pertama kali dijalankan. Konsekuensinya: **setiap perubahan entity mengharuskan `App_Data/` dihapus.** Lihat [Keterbatasan](#keterbatasan--penyederhanaan-v1) untuk jalur upgrade-nya.

## Memulai

### Prasyarat

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Tidak ada yang lain, untuk konfigurasi default — SQLite + storage filesystem lokal tidak butuh layanan eksternal apa pun.
- Opsional: GPU NVIDIA/CUDA (atau perangkat OpenCL) untuk mode splat yang dipercepat GPU. Kalau tidak ada, aplikasi otomatis turun ke CPU.

### Menjalankan

```bash
dotnet restore
dotnet build
cd src/SplatStudio.Web && dotnet run
```

Lalu buka <http://localhost:5080>. Saat pertama kali dijalankan, aplikasi akan:

- Membuat database SQLite di `App_Data/splatstudio.db`.
- Mengisi (seed) enam akun demo, dua belas skena splat, dan enam skena mesh (lihat [Akun demo](#akun-demo--data-contoh)).
- Mengonversi gambar contoh di background — galeri memperbarui dirinya sendiri saat tiap skena selesai.

Untuk mengembalikan ke kondisi bersih, hentikan aplikasi lalu hapus `App_Data/`. Karena skema dibuat dengan `EnsureCreatedAsync()` dan bukan migration, langkah ini juga wajib setiap kali ada perubahan entity.

### Pengujian

```bash
dotnet test tests/SplatStudio.Tests

# Tabel perbandingan waktu GPU vs CPU
dotnet test tests/SplatStudio.Tests --filter "FullyQualifiedName~Benchmark" \
  --logger "console;verbosity=detailed"
```

Suite ini mencakup layout biner `.splat`, batas output heuristik kedalaman, penanganan alpha, determinisme, kesetaraan GPU/CPU, aturan ketersediaan mode konversi, serta mesh contoh yang dihasilkan — termasuk bahwa setiap `.glb` adalah container yang valid secara struktur dan thumbnail-nya bukan bingkai kosong. Tes yang membutuhkan GPU akan **di-skip**, bukan gagal, di mesin tanpa perangkat CUDA/OpenCL.

## Konfigurasi

Semua konfigurasi ada di `src/SplatStudio.Web/appsettings.json` (dan `appsettings.Development.json` untuk override lokal). Setiap pergantian provider hanya berupa satu nilai string — tidak perlu mengubah kode.

### Database

```json
"Database": { "Provider": "Sqlite" },   // Sqlite | SqlServer | MySql
"ConnectionStrings": {
  "Sqlite":    "Data Source=App_Data/splatstudio.db",
  "SqlServer": "Server=localhost;Database=SplatStudio;User Id=sa;Password=YourStrong!Passw0rd;TrustServerCertificate=True;",
  "MySql":     "Server=localhost;Port=3306;Database=splatstudio;User=root;Password=root;"
}
```

Set `Database:Provider` ke `SqlServer` atau `MySql` dan isi connection string yang sesuai — itu saja perubahan yang dibutuhkan.

### Storage

```json
"Storage": {
  "Provider": "FileSystem",   // FileSystem | AzureBlob | S3 | MinIO
  "FileSystem": { "RootPath": "App_Data/storage", "PublicBasePath": "/media" },
  "AzureBlob":  { "ConnectionString": "", "ContainerName": "splatstudio" },
  "S3":         { "ServiceUrl": "", "AccessKey": "", "SecretKey": "", "BucketName": "splatstudio", "Region": "us-east-1", "ForcePathStyle": false, "PublicBaseUrl": "" },
  "MinIO":      { "ServiceUrl": "http://localhost:9000", "AccessKey": "minioadmin", "SecretKey": "minioadmin", "BucketName": "splatstudio", "ForcePathStyle": true, "PublicBaseUrl": "" }
}
```

- **FileSystem** (default): file disimpan di `App_Data/storage` dan disajikan di `/media/...` lewat static file middleware. Tanpa setup eksternal.
- **AzureBlob**: isi `ConnectionString` (misalnya dari Azure Portal atau `AzureWebJobsStorage`) dan `ContainerName`. Container otomatis dibuat dengan akses publik jika belum ada.
- **S3**: isi `AccessKey`/`SecretKey`/`BucketName`/`Region`. Biarkan `ServiceUrl` kosong untuk terhubung ke AWS S3 sungguhan. Bucket otomatis dibuat saat startup jika belum ada.
- **MinIO** (atau layanan kompatibel-S3 lain — Cloudflare R2, Wasabi, dll.): bentuknya sama seperti S3, tapi isi `ServiceUrl` dengan endpoint Anda dan `ForcePathStyle: true` (kebanyakan server kompatibel-S3 self-hosted membutuhkan path-style addressing).

### Mode konversi

Setiap unggahan memilih satu mode. Hanya yang pertama berjalan di mesin Anda sendiri; dua lainnya mengirim gambar ke layanan hosted.

| Mode | Menghasilkan | Berjalan di | Membutuhkan |
|---|---|---|---|
| **Splat estimasi kedalaman** | point cloud `.splat` | mesin ini, milidetik | — |
| **Splat fotorealistik** | point cloud `.splat` | hosted, menitan | `Splatting:Hosted:Photoreal` |
| **Objek 3D (mesh)** | mesh bertekstur `.glb` | hosted, menitan | `Splatting:Hosted:Mesh` |

Mode tanpa kredensial **tetap tampil di halaman unggah tetapi dinonaktifkan**, disertai nama setting persis yang dibutuhkannya. Tidak ada yang disembunyikan, dan tidak ada yang menerima unggahan yang tak bisa dipenuhinya.

Mode ketiga menghasilkan geometri, bukan point cloud, jadi skena tersebut dibuka dengan viewer glTF dan punya tombol **Download .glb** — filenya glTF biasa yang bisa dibuka di Blender atau tool 3D lain.

#### Mengarahkan mode hosted ke provider

Rodin/Hyper3D, Tripo, Tencent Hunyuan3D, dan TRELLIS self-hosted semuanya berbicara protokol yang sama — POST gambar, dapat job id, polling sampai selesai, unduh asetnya. Yang berbeda hanya penamaannya, jadi penamaan itu dijadikan konfigurasi alih-alih tiga klien vendor yang ditulis tangan dan akan usang begitu API-nya berubah:

```json
"Splatting": {
  "Hosted": {
    "Mesh": {
      "BaseUrl": "https://api.provider-anda.com",
      "ApiKey": "",                       // pakai user secrets atau environment
      "SubmitPath": "/v1/generate",
      "ImageFieldName": "image",
      "SubmitFields": { "output_format": "glb" },
      "JobIdPath": "data.job_id",         // path bertitik ke dalam response JSON
      "StatusPath": "/v1/generate/{jobId}",
      "StatusFieldPath": "status",
      "SuccessStates": [ "succeeded", "done" ],
      "FailureStates": [ "failed", "cancelled" ],
      "ResultUrlPath": "result.url",      // "results.0.url" juga bisa
      "PollIntervalSeconds": 5,
      "TimeoutSeconds": 900
    }
  }
}
```

Nilai default di atas menggambarkan kontraknya, tapi **bukan** API vendor mana pun — isi path-nya dari dokumentasi provider Anda. Apa pun yang dikembalikan endpoint status yang tidak ada di `SuccessStates` maupun `FailureStates` dianggap "masih berjalan", jadi Anda tidak perlu mendaftar semua kata untuk status "sedang proses".

Dua penjaga yang perlu diketahui: response splat harus kelipatan utuh dari 32 byte per record, dan response mesh harus diawali magic `glTF`. Provider yang salah dikonfigurasi akan menggagalkan skena dengan pesan jelas, bukan menyimpan sesuatu yang dirender viewer sebagai kekosongan.

> **Jalur mesh diverifikasi ujung ke ujung terhadap server tiruan lokal yang mengikuti kontrak di atas, bukan terhadap API komersial.** Tidak ada endpoint vendor sungguhan yang ditanam di kode, jadi siapkan waktu untuk mengisi pemetaan field sesuai vendor yang Anda pakai.

### Splat engine

Bagian ini mengonfigurasi mode **estimasi kedalaman** bawaan saja; mode hosted dikonfigurasi di atas.

```json
"Splatting": {
  "Engine": "Gpu",            // LocalHeuristic | Gpu | ExternalApi
  "MaxPoints": 40000,         // dipakai LocalHeuristic
  "GpuMaxPoints": 250000,     // dipakai Gpu
  "ExternalApi": { "Endpoint": "", "ApiKey": "", "TimeoutSeconds": 120 }
}
```

`MaxPoints`/`GpuMaxPoints` mengatur kecepatan konversi sekaligus ukuran file output (32 byte per titik) — kecilkan untuk skena lebih cepat dan ringan; perbesar untuk point cloud lebih padat dengan konsekuensi unduhan `.splat` yang lebih besar.

**`Gpu`** menjalankan heuristik yang identik sebagai kernel komputasi ILGPU di perangkat CUDA (dengan fallback OpenCL). Kalau tidak ada perangkat yang ditemukan, alasannya dicatat di log dan aplikasi diam-diam kembali ke `LocalHeuristic`, sehingga tetap berjalan di mesin tanpa GPU.

Diukur pada NVIDIA RTX 4060 (`dotnet test --filter Benchmark`, 5 iterasi, sumber 1024×1024):

| budget titik | CPU ms/gambar | GPU ms/gambar | percepatan |
|---:|---:|---:|---:|
| 10.000 | 21,7 | 19,6 | 1,11× |
| 40.000 | 27,5 | 9,9 | 2,78× |
| 100.000 | 27,1 | 11,9 | 2,28× |
| 262.144 | 59,6 | 19,7 | 3,03× |
| 1.000.000 | — | 37,0 | CPU mentok di 262.144 |

Kedua engine sama-sama memakai decode JPEG dan resize Lanczos di sisi CPU, dan itu adalah lantai biaya yang tak bisa dilewati keduanya — karena itulah keunggulan GPU baru terlihat setelah jumlah titik mendominasi.

### Email (reset password)

```json
"Email": {
  "Provider": "File",   // File | Smtp
  "Smtp": { "Host": "", "Port": 587, "EnableSsl": true, "Username": "", "Password": "", "FromAddress": "no-reply@splatstudio.local", "FromName": "SplatStudio" }
}
```

Provider `File` (default) menulis setiap "email" sebagai file `.html` di `App_Data/emails` dan — hanya dalam mode dev ini — menampilkan link reset langsung di halaman konfirmasi "lupa password", sehingga Anda bisa menguji seluruh flow tanpa setup SMTP sama sekali. Ganti ke `Smtp` dan isi kredensial sungguhan untuk produksi.

## Akun demo & data contoh

Saat pertama kali dijalankan terhadap database kosong, aplikasi akan mengisi:

- **Enam akun**, semuanya dengan password **`Demo123!`** — masuk sebagai **demo@splatstudio.local**, atau sebagai `rani@`, `tomas@`, `aiko@`, `marcus@`, `priya@` `splatstudio.local` untuk melihat galeri dari sisi orang lain. Masing-masing punya avatar dan bio yang dibuat otomatis.
- **Dua belas skena splat** yang tersebar di akun-akun tersebut (dua di antaranya privat, supaya "My Scenes" punya sesuatu untuk di-toggle), dengan komentar dari beberapa pengguna dan sebaran rating 3–5 bintang agar rata-ratanya bermakna.
- **Enam skena mesh** — trefoil knot, vas bubut, batu permata terpotong, cangkang nautilus, bentang punggungan, dan pita Möbius — supaya viewer glTF dan tombol unduh `.glb` punya sesuatu untuk ditampilkan tanpa provider hosted terkonfigurasi.

Tidak ada satu pun yang dikirim sebagai file biner. Foto dihasilkan `SampleImageFactory` (gradien dan bentuk yang digambar), mesh oleh `SampleMeshFactory` yang menulis binary glTF dari nol — permukaan parametrik dengan warna per-vertex mengikuti ramp dekat/jauh yang sama dengan antarmukanya. Keduanya fungsi murni dari kunci resepnya, sehingga setiap deployment menghasilkan galeri yang identik.

Dua catatan kejujuran tentang seeding:

- Skena splat melewati pipeline upload → storage → queue → konversi background yang **sesungguhnya**, sehingga sekaligus berfungsi sebagai smoke test pipeline itu di setiap deployment baru. Katalognya sengaja memuat kasus yang ditangani buruk oleh teknik ini — torus yang lubangnya tertutup, bentang dune yang nyaris datar — berdampingan dengan yang hasilnya bagus.
- Skena mesh **tidak bisa** melewati pipeline itu, karena mode 3 tidak punya engine lokal; tanpa provider terkonfigurasi semuanya akan gagal dan instalasi baru hanya akan menampilkan error. Skena-skena itu ditulis langsung ke storage dan dilabeli **Sample data**, bukan diatribusikan ke model yang tidak pernah berjalan.

Ini hanya terjadi sekali — begitu ada satu pun pengguna di database (termasuk orang sungguhan yang mendaftar lebih dulu), proses seeding permanen dilewati.

## Memasang backend 3D Gaussian Splatting yang sesungguhnya

Untuk splat fotorealistik dan mesh, cara termudah adalah mengisi `Splatting:Hosted` (lihat [Mode konversi](#mode-konversi)) — tidak perlu menulis kode sama sekali.

Kalau provider Anda tidak cocok dengan pola submit/poll/download itu, splat engine tetap diletakkan di belakang satu interface kecil:

```csharp
public interface IGaussianSplatEngine
{
    SplatEngineType EngineType { get; }
    Task<GaussianSplatGenerationResult> GenerateAsync(Stream imageStream, int maxOutputPoints, CancellationToken ct = default);
}
```

`ExternalApiSplatEngine` (di `SplatStudio.Infrastructure/Splatting/`) adalah titik awal yang sudah didokumentasikan: ia mengirim (POST) gambar ke endpoint HTTP yang dikonfigurasi dan mengharapkan byte `.splat` mentah sebagai balasan. Implementasikan `IGaussianSplatEngine` sesuai provider Anda, daftarkan di `InfrastructureServiceCollectionExtensions.AddSplatInfrastructure`, dan set `Splatting:Engine` ke `ExternalApi`.

Untuk menambah mode konversi yang benar-benar baru, implementasikan `IConversionEngine` — port itu yang dibaca halaman unggah untuk menyusun pilihannya dan yang dipakai background worker untuk memilih engine.

## Catatan performa

- **Kompresi response** diaktifkan untuk Brotli dan Gzip, secara eksplisit mencakup `application/octet-stream` agar file `.splat` terkompresi dengan baik saat dikirim.
- **Satu background worker** (lihat [Arsitektur](#arsitektur)) menghindari perebutan CPU dengan circuit SignalR Blazor Server.
- **Komputasi GPU** untuk depth field dan emisi splat bila ada perangkat CUDA/OpenCL — lihat tabel benchmark di [Splat engine](#splat-engine).
- **Bounded channel queue** (`ChannelConversionQueue`, kapasitas 256) memberikan backpressure alami — unggahan mengantre, bukan memunculkan kerja paralel tanpa batas.
- **Downscaling gambar** terjadi sebelum pembuatan splat dan sekali lagi untuk thumbnail galeri (dimensi maksimum 480px), sehingga browser tidak perlu mengunduh gambar resolusi penuh hanya untuk merender satu kartu kecil di grid.
- **Caching file statis**: media yang disajikan dari filesystem dikirim dengan header `Cache-Control` immutable 30 hari.
- **`InvariantGlobalization`** diaktifkan, memangkas data ICU yang tidak dibutuhkan aplikasi.
- Viewer point-cloud merender splat sebagai billboard yang selalu menghadap kamera tanpa pengurutan depth back-to-front per frame (didokumentasikan di `splat-viewer.js`) — mengurutkan ratusan ribu titik setiap frame lebih mahal daripada manfaat visualnya untuk billboard yang sebagian besar opaque ini.
- Viewer mesh memuat parser glTF-nya (±96 KB) hanya saat pertama kali dibutuhkan, jadi deployment yang tidak memakai mode 3 tidak ikut membayarnya.

## Keterbatasan & penyederhanaan v1

- **Heuristik, bukan 3DGS sungguhan** — lihat [bagian atas](#apa-yang-sebenarnya-dilakukan-aplikasi-ini-mohon-dibaca-dulu).
- **Skema EF Core lewat `EnsureCreatedAsync()`, bukan migration.** Untuk beralih ke migration per-provider yang proper untuk produksi: pilih satu provider, jalankan `dotnet ef migrations add InitialCreate --context ApplicationDbContext`, ulangi per provider dengan `--output-dir` masing-masing, lalu ganti pemanggilan `EnsureCreatedAsync()` di `Program.cs` dengan `Database.MigrateAsync()`.
- **Tanpa CQRS/MediatR** — lihat [Arsitektur](#arsitektur). Mudah ditambahkan nanti mengingat batas port/adapter yang sudah ada.
- **Logout adalah HTTP POST sungguhan**, bukan event Blazor — ini disengaja (lihat Arsitektur), tapi berarti klien dengan JS dimatikan atau cache agresif secara teoritis bisa me-replay-nya; perlindungan antiforgery standar ASP.NET Core diterapkan lewat komponen built-in `<AntiforgeryToken />`.
- **Splat viewer tidak melakukan depth sort back-to-front** — trade-off performa/kompleksitas yang disengaja, lihat [Catatan performa](#catatan-performa).
- **Engine GPU menghasilkan titik dengan urutan yang tidak deterministik.** Ia memadatkan output-nya dengan atomic counter, jadi dua kali proses atas gambar yang sama bisa mengurutkan splat berbeda. Formatnya tidak punya semantik urutan dan viewer-nya tidak melakukan depth sort, jadi ini tak terlihat — tapi artinya output GPU tidak bisa dibandingkan byte-per-byte antarproses, sementara output CPU bisa.
- **Belum ada tes integrasi atau UI.** Suite-nya mencakup format splat, engine, lapisan pemilihan mode, dan mesh contoh; halaman Blazor, flow auth, dan adapter storage hanya diuji manual dan lewat jalur seeding saat startup.
- **Belum ada provider hosted yang diverifikasi terhadap API aslinya.** Kontrak submit/poll/download diverifikasi terhadap server tiruan lokal, dan pemetaan field-nya berupa konfigurasi — tapi provider sungguhan pertama yang Anda sambungkan tetap perlu path-nya diisi, dan mungkin field tambahan di `SubmitFields`.
- **Viewer mesh sengaja dibuat minimal.** Pencahayaannya rig tiga lampu tetap, tanpa environment map atau bayangan.
- **Unduhan `.glb` mengandalkan atribut `download`**, yang hanya dihormati browser untuk same-origin. Itu mencakup provider `FileSystem` bawaan; di belakang Azure Blob atau S3, link-nya akan membuka file alih-alih menyimpannya, kecuali bucket-nya menyetel `Content-Disposition`.

## Catatan deployment

- Set `Database:Provider` dan `Storage:Provider` sesuai environment target Anda, dan gunakan environment variable atau secrets manager untuk connection string/key, bukan menaruhnya langsung di `appsettings.json`. Ini juga berlaku untuk `Splatting:Hosted:*:ApiKey`.
- Di belakang reverse proxy (nginx, Azure App Service, dll.), pastikan WebSocket diaktifkan dan diteruskan — render mode Interactive Blazor Server bergantung pada koneksi SignalR yang persisten.
- Jika scale ke banyak instance, gunakan routing sticky session (atau Azure SignalR Service) — circuit Blazor Server bersifat stateful dan terikat ke satu instance.
- `UseHsts`/`UseExceptionHandler` hanya diterapkan di luar environment Development; set `ASPNETCORE_ENVIRONMENT=Production` untuk deployment sungguhan.
- Job hosted berjalan menitan dan memakai satu worker yang sama dengan konversi lokal. Kalau Anda mengharapkan banyak unggahan mode 2/3 sekaligus, tambah jumlah instance — antreannya bounded (256) dan akan memberi backpressure, bukan menumpuk tanpa batas.
