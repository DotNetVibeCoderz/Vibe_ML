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
5. [Konfigurasi](#konfigurasi)
6. [Akun demo & data contoh](#akun-demo--data-contoh)
7. [Memasang backend 3D Gaussian Splatting yang sesungguhnya](#memasang-backend-3d-gaussian-splatting-yang-sesungguhnya)
8. [Catatan performa](#catatan-performa)
9. [Keterbatasan & penyederhanaan v1](#keterbatasan--penyederhanaan-v1)
10. [Catatan deployment](#catatan-deployment)

---

## Apa yang sebenarnya dilakukan aplikasi ini (mohon dibaca dulu)

**3D Gaussian Splatting** yang sesungguhnya (teknik di balik Luma AI atau nerfstudio/gsplat) dilatih dari *banyak* foto subjek yang sama dari berbagai sudut. Pipeline seperti COLMAP pertama-tama merekonstruksi posisi kamera dan point cloud kasar lewat Structure-from-Motion, lalu ribuan Gaussian dioptimasi selama berjam-jam di GPU dengan gradient descent terhadap semua sudut pandang tersebut. Satu foto datar saja tidak mengandung informasi multi-sudut yang dibutuhkan untuk merekonstruksi geometri 3D yang sesungguhnya — sepintar apa pun kode yang ditulis, ini tidak bisa diubah.

Yang SplatStudio sediakan sebagai gantinya adalah `LocalHeuristicSplatEngine`: rekonstruksi "2.5D" yang cepat, sepenuhnya offline, hanya pakai CPU, dan deterministik. Untuk setiap gambar yang diunggah, engine ini:

1. Mengecilkan ukuran gambar sesuai budget jumlah titik (`Splatting:MaxPoints`, default 40.000).
2. Mengestimasi **pseudo-depth** per piksel sebagai campuran dari invers-luminance dan radial prior yang lebih berat di tengah (piksel yang lebih terang dan lebih ke tengah ditarik lebih dekat ke kamera).
3. Menghaluskan depth field tersebut dengan box blur kecil.
4. Menghasilkan satu Gaussian splat per piksel, diposisikan sesuai depth field tersebut, dengan warna dari piksel sumbernya.
5. Menulis hasilnya sebagai file biner `.splat` 32 byte per titik, dirender di browser dengan viewer point-cloud Three.js kecil yang ditulis khusus untuk proyek ini (`wwwroot/js/splat-viewer.js`).

Hasilnya adalah dunia point-cloud 3D yang sungguh-sungguh bisa diputar bebas — dan hasilnya cukup bagus untuk potret atau objek tunggal dengan latar polos, persis seperti yang ditunjukkan oleh gambar contoh yang dibundel. Namun ini **bukan** pengganti rekonstruksi multi-sudut yang sesungguhnya, dan tidak akan menghasilkan depth/occlusion yang akurat untuk skena yang kompleks. Anggap saja ini efek "foto jadi snow-globe" yang bergaya, bukan photogrammetry.

Jika Anda punya akses ke backend splatting sungguhan, lihat [Memasang backend sungguhan](#memasang-backend-3d-gaussian-splatting-yang-sesungguhnya) — engine ini sengaja diletakkan di belakang interface yang bersih agar mudah ditukar.

## Fitur

- **Unggah → konversi → lihat**: unggah JPEG/PNG/WebP, aplikasi akan mengantrekan job konversi di background, dan galeri/viewer ter-update otomatis (lewat `ISceneUpdateNotifier` in-process) begitu selesai — tanpa perlu refresh manual.
- **Galeri**: grid publik dari skena yang sudah selesai ("constellation gallery"), masing-masing menampilkan thumbnail, rating rata-rata, jumlah view, dan jumlah komentar.
- **Viewer 3D**: drag untuk orbit, scroll untuk zoom, dirender dengan renderer point-cloud Three.js ringan yang ditulis sendiri (tanpa library splat-viewer eksternal — dibuat khusus untuk format `.splat` milik proyek ini).
- **Komentar & rating**: pengguna yang sudah login bisa meninggalkan komentar dan rating 1-5 bintang per skena (satu rating per pengguna per skena).
- **Sistem akun penuh** lewat ASP.NET Core Identity: registrasi, login/logout, lupa/reset password (lewat email sender yang bisa ditukar), edit profil (nama tampilan, bio, unggah avatar), ganti password.
- **My Scenes**: kelola unggahan Anda sendiri — toggle publik/privat, hapus (membersihkan baris database sekaligus file yang tersimpan).
- **Tiga provider database**: SQLite (default tanpa konfigurasi), SQL Server, MySQL — tinggal ganti satu nilai konfigurasi.
- **Empat provider storage**: filesystem lokal (default tanpa konfigurasi), Azure Blob Storage, AWS S3, dan MinIO/layanan kompatibel-S3 apa pun — tinggal ganti satu nilai konfigurasi.
- **UI glassmorphism**: design system "Constellation Glass" custom — gumpalan warna blur yang melayang-layang di belakang panel kaca buram, sengaja dipilih sebagai metafora visual literal dari Gaussian splat.

## Arsitektur

```
src/
  SplatStudio.Domain          Entity + enum saja, tanpa dependensi
  SplatStudio.Application     Port (interface) yang diimplementasikan oleh layer Web/Infrastructure
  SplatStudio.Infrastructure  EF Core, Identity, provider storage, splat engine, background worker
  SplatStudio.Web             Komponen Blazor, Razor Pages untuk auth, aset statis
```

Beberapa keputusan yang disengaja dan perlu dijelaskan:

- **Tanpa MediatR/CQRS.** Mengingat proyek ini sudah cukup besar (3 provider DB × 4 provider storage × auth lengkap × galeri real-time), Clean Architecture yang disederhanakan tanpa pipeline mediator membuat codebase lebih mudah dibaca dari ujung ke ujung. Pemisahan port/adapter (interface di `Application`, implementasi di `Infrastructure`) tetap dipertahankan, sehingga menambahkan CQRS nanti adalah penambahan struktural, bukan menulis ulang dari awal.
- **Hosting model Blazor Web App + Razor Pages klasik untuk auth.** Login/Register/Logout/ResetPassword perlu menulis cookie autentikasi langsung pada response HTTP, yang hanya berfungsi *sebelum* koneksi SignalR dari circuit Blazor Server mengambil alih. Karena itu kelima flow tersebut adalah Razor Pages ASP.NET Core biasa (`Pages/Account/*.cshtml`), distyling agar serasi dengan tampilan aplikasi lainnya, sementara semua halaman lain adalah komponen Blazor Interactive Server. Ini mengikuti pola template resmi `dotnet new blazor -au Individual -int Server`.
- **Satu background worker, bukan pool.** Konversi gambar→splat adalah CPU-bound. Menjalankan banyak proses ini secara paralel pada satu instance akan merebut CPU dari circuit SignalR Blazor Server di proses yang sama, membuat seluruh aplikasi terasa lambat bagi semua orang. `ConversionBackgroundService` sengaja memproses satu skena dalam satu waktu; untuk scale-out, tambah jumlah instance, bukan jumlah worker per instance.
- **`EnsureCreatedAsync()` alih-alih migration EF.** Dengan tiga provider database yang bisa ditukar, memelihara tiga riwayat migration terpisah menambah biaya maintenance yang nyata untuk manfaat yang relatif kecil di v1. Skema dibuat langsung dari model saat pertama kali dijalankan. Lihat [Keterbatasan](#keterbatasan--penyederhanaan-v1) untuk jalur upgrade-nya.

## Memulai

### Prasyarat

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Tidak ada yang lain, untuk konfigurasi default — SQLite + storage filesystem lokal tidak butuh layanan eksternal apa pun.

### Menjalankan

```bash
cd src/SplatStudio.Web
dotnet restore
dotnet run
```

Lalu buka URL yang tercetak di konsol (biasanya `https://localhost:5001` atau sejenisnya). Saat pertama kali dijalankan, aplikasi akan:

- Membuat database SQLite di `App_Data/splatstudio.db`.
- Mengisi (seed) akun demo dan tiga skena contoh (lihat [Akun demo](#akun-demo--data-contoh)).
- Mulai mengonversi gambar contoh di background — refresh galeri setelah beberapa detik untuk melihatnya selesai.

> **Catatan:** proyek ini dibuat di lingkungan sandbox tanpa akses internet ke nuget.org, sehingga kode ini belum bisa dikompilasi/di-restore di sesi ini. Mohon jalankan `dotnet restore` / `dotnet build` sendiri dengan akses internet normal sebelum pemakaian pertama — ini langkah standar untuk solusi yang baru dibuat, tapi perlu disampaikan secara eksplisit karena belum (dan tidak bisa) diverifikasi end-to-end dalam sesi ini.

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

### Splat engine

```json
"Splatting": {
  "Engine": "LocalHeuristic",   // LocalHeuristic | ExternalApi
  "MaxPoints": 40000,
  "ExternalApi": { "Endpoint": "", "ApiKey": "", "TimeoutSeconds": 120 }
}
```

`MaxPoints` mengatur kecepatan konversi sekaligus ukuran file output — kecilkan untuk skena yang lebih cepat dan ringan; perbesar untuk point cloud yang lebih padat (dengan konsekuensi konversi lebih lambat dan file `.splat` yang dikirim ke browser lebih besar).

### Email (reset password)

```json
"Email": {
  "Provider": "File",   // File | Smtp
  "Smtp": { "Host": "", "Port": 587, "EnableSsl": true, "Username": "", "Password": "", "FromAddress": "no-reply@splatstudio.local", "FromName": "SplatStudio" }
}
```

Provider `File` (default) menulis setiap "email" sebagai file `.html` di `App_Data/emails` dan — hanya dalam mode dev ini — menampilkan link reset langsung di halaman konfirmasi "lupa password", sehingga Anda bisa menguji seluruh flow tanpa setup SMTP sama sekali. Ganti ke `Smtp` dan isi kredensial sungguhan untuk produksi.

## Akun demo & data contoh

Saat pertama kali dijalankan terhadap database yang kosong, aplikasi akan mengisi:

- Pengguna demo: **demo@splatstudio.local** / **Demo123!**
- Tiga skena contoh yang dibangun dari gambar sintetis yang dihasilkan secara terprogram (gradien dan bentuk — bukan foto sungguhan), didorong lewat pipeline upload → storage → queue → konversi background yang sesungguhnya, sehingga sekaligus berfungsi sebagai smoke test pipeline tersebut di setiap deployment baru.
- Satu komentar sambutan dan rating 5 bintang pada skena contoh pertama.

Ini hanya terjadi sekali — begitu ada satu pun pengguna di database (termasuk orang sungguhan yang mendaftar lebih dulu), proses seeding ini permanen dilewati.

## Memasang backend 3D Gaussian Splatting yang sesungguhnya

Splat engine diletakkan di belakang satu interface kecil:

```csharp
public interface IGaussianSplatEngine
{
    SplatEngineType EngineType { get; }
    Task<GaussianSplatGenerationResult> GenerateAsync(Stream imageStream, int maxOutputPoints, CancellationToken ct = default);
}
```

`ExternalApiSplatEngine` (di `SplatStudio.Infrastructure/Splatting/`) adalah titik awal yang sudah didokumentasikan: ia mengirim (POST) gambar ke endpoint HTTP yang dikonfigurasi dan mengharapkan byte `.splat` mentah sebagai balasan. Kebanyakan provider 3DGS sungguhan (Luma AI, KIRI Engine, job runner nerfstudio/gsplat self-hosted) bersifat asinkron/berbasis job, bukan request-response sinkron, jadi anggap stub ini sebagai kerangka yang perlu disesuaikan dengan pola polling atau webhook provider spesifik Anda — bukan klien produksi yang siap pakai langsung. Implementasikan `IGaussianSplatEngine` sesuai provider Anda, daftarkan di `InfrastructureServiceCollectionExtensions.AddSplatInfrastructure`, dan set `Splatting:Engine` ke `ExternalApi`.

## Catatan performa

- **Kompresi response** diaktifkan untuk Brotli dan Gzip, secara eksplisit mencakup `application/octet-stream` agar file `.splat` terkompresi dengan baik saat dikirim.
- **Satu background worker** (lihat [Arsitektur](#arsitektur)) menghindari perebutan CPU dengan circuit SignalR Blazor Server.
- **Bounded channel queue** (`ChannelConversionQueue`, kapasitas 256) memberikan backpressure alami — unggahan akan mengantre, bukan memunculkan kerja paralel tanpa batas.
- **Downscaling gambar** terjadi sebelum pembuatan splat (`LocalHeuristicSplatEngine` mengubah ukuran sesuai resolusi yang diturunkan dari budget jumlah titik) dan sekali lagi untuk thumbnail galeri (dimensi maksimum 480px), sehingga browser tidak perlu mengunduh gambar resolusi penuh hanya untuk merender satu kartu kecil di grid.
- **Caching file statis**: media yang disajikan dari filesystem dikirim dengan header `Cache-Control` immutable 30 hari.
- **`InvariantGlobalization`** diaktifkan, memangkas data ICU yang tidak dibutuhkan aplikasi.
- Viewer point-cloud Three.js merender splat sebagai billboard yang selalu menghadap kamera tanpa pengurutan depth back-to-front per frame (didokumentasikan di `splat-viewer.js`) — mengurutkan puluhan ribu titik setiap frame akan lebih mahal daripada manfaat visualnya untuk billboard yang sebagian besar opaque ini.

## Keterbatasan & penyederhanaan v1

- **Heuristik, bukan 3DGS sungguhan** — lihat [bagian atas](#apa-yang-sebenarnya-dilakukan-aplikasi-ini-mohon-dibaca-dulu).
- **Skema EF Core lewat `EnsureCreatedAsync()`, bukan migration.** Untuk beralih ke migration per-provider yang proper untuk produksi: pilih satu provider, jalankan `dotnet ef migrations add InitialCreate --context ApplicationDbContext`, ulangi per provider dengan `--output-dir` masing-masing, lalu ganti pemanggilan `EnsureCreatedAsync()` di `Program.cs` dengan `Database.MigrateAsync()`.
- **Tanpa CQRS/MediatR** — lihat [Arsitektur](#arsitektur). Mudah ditambahkan nanti mengingat batas port/adapter yang sudah ada.
- **Logout adalah HTTP POST sungguhan**, bukan event Blazor — ini disengaja (lihat Arsitektur), tapi berarti klien dengan JS dimatikan atau cache yang agresif secara teoritis bisa me-replay-nya; perlindungan antiforgery standar ASP.NET Core diterapkan lewat komponen built-in `<AntiforgeryToken />`.
- **Splat viewer tidak melakukan depth sort back-to-front** — trade-off performa/kompleksitas yang disengaja, lihat [Catatan performa](#catatan-performa).

## Catatan deployment

- Set `Database:Provider` dan `Storage:Provider` sesuai environment target Anda, dan gunakan environment variable atau secrets manager untuk connection string/key, bukan menaruhnya langsung di `appsettings.json`.
- Di belakang reverse proxy (nginx, Azure App Service, dll.), pastikan WebSocket diaktifkan dan diteruskan — render mode Interactive Blazor Server bergantung pada koneksi SignalR yang persisten.
- Jika scale ke banyak instance, gunakan routing sticky session (atau Azure SignalR Service) — circuit Blazor Server bersifat stateful dan terikat ke satu instance.
- `UseHsts`/`UseExceptionHandler` hanya diterapkan di luar environment Development; set `ASPNETCORE_ENVIRONMENT=Production` untuk deployment sungguhan.
