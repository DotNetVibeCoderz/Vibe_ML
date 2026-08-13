# ScienceAppGen

*[English](../ScienceAppGen.md)* · Code editor yang membangun aplikasi data science dari sebuah prompt.

![ScienceAppGen](../screenshots/scienceappgen.png)

ScienceAppGen adalah IDE desktop yang dibangun dengan Avalonia. Bagian yang membedakannya adalah
asistennya: **Jack, the Code Bender** tidak mencetak kode ke chat untuk Anda salin — ia membuat
proyeknya, menulis berkasnya, menjalankan build, membaca error kompiler, lalu memperbaikinya.

```bash
dotnet run --project tools/ScienceAppGen
```

---

## Tata letak jendela

| Bagian | Fungsinya |
|---|---|
| **Kiri** | Explorer proyek bergaya VS Code. Klik ganda membuka berkas; `bin`, `obj`, dan `.git` disembunyikan. |
| **Tengah** | Editor dengan nomor baris, pewarnaan sintaks, dan tab. |
| **Bawah** | Panel output: hasil build, panggilan tool, dan error, diwarnai menurut tingkat keparahan. |
| **Kanan** | Chat dengan Jack. Bisa diubah lebarnya, disembunyikan, dengan pemilih model di atas. |
| **Status bar** | Proyek yang terbuka, operasi berjalan, posisi kursor, model aktif. |

Spektrum enam warna di seluruh antarmuka adalah sebuah sistem, bukan hiasan: satu pita per library
Gravicode, dipakai ulang untuk tingkat log dan jenis berkas, sehingga sebuah warna selalu berarti
hal yang sama.

### Warna sintaks

Enam panjang gelombang yang sama juga mewarnai kode, dicerahkan untuk latar gelap dan digelapkan
untuk latar terang. Definisi bawaan AvaloniaEdit mengasumsikan halaman putih — `MethodCall`
berwarna MidnightBlue dan `NumberLiteral` DarkBlue, keduanya praktis tak terlihat di editor gelap —
sehingga `SyntaxTheme` memetakan ulang nama-nama warnanya ke palet di `Themes/Tokens.axaml`,
bukan menggandakan berkas `.xshd`.

| Peran | Panjang gelombang | Gelap | Terang |
|---|---|---|---|
| Keyword | ungu GraviProb | `#9B9BEA` 7,2:1 | `#5A45B8` 7,1:1 |
| Tipe | sian GraviGraph | `#62BEDC` 8,6:1 | `#0E6E8C` 5,8:1 |
| Method | kuning GraviLearn | `#E6D06A` 11,8:1 | `#7E610F` 5,8:1 |
| String | hijau GraviText | `#74C99A` 9,1:1 | `#1E7A45` 5,4:1 |
| Angka | oranye GraviFrame | `#EBA05C` 8,4:1 | `#A6520B` 5,5:1 |
| Preprocessor, exception | merah GraviNum | `#EC7378` 6,3:1 | `#B3272C` 6,5:1 |
| Komentar | abu batu, miring | `#7C8EA1` 5,4:1 | `#5F7183` 5,0:1 |
| Tanda baca | abu batu | `#97A4B4` 7,2:1 | `#47535F` 7,9:1 |

Setiap peran melewati WCAG AA (4,5:1) terhadap latar editornya di kedua tema, dan sebagian besar
melewati AAA. Berkas yang sama dengan tema terang:

![Editor dengan tema terang](../screenshots/scienceappgen-light.png)

### Papan ketik

| | |
|---|---|
| `Ctrl+Shift+N` | Proyek baru |
| `Ctrl+O` | Buka proyek |
| `Ctrl+S` / `Ctrl+Shift+S` | Simpan / simpan semua |
| `Ctrl+G` | Ke nomor baris |
| `Ctrl+K` | Format kode |
| `Ctrl+B` / `Ctrl+L` | Tampil/sembunyikan chat / log |
| `F6` / `F5` | Build / jalankan |
| `Ctrl+,` | Pengaturan |
| `Ctrl+Enter` | Kirim pesan chat |

---

## Konfigurasi

Semuanya berada di `app.config` dan dapat diubah lewat **Tools → Settings**. UI menulis perubahan
kembali ke berkas yang sama, jadi menyunting manual dan lewat UI itu setara.

```xml
<add key="llm.provider" value="AzureOpenAI" />
<add key="llm.model" value="gpt-5-mini" />
<add key="llm.apiKey" value="" />
<add key="llm.endpoint" value="https://resource-anda.openai.azure.com/" />
<add key="llm.temperature" value="0.3" />
<add key="tools.tavilyApiKey" value="" />
```

Kunci API dikosongkan di kontrol versi. Isikan lewat Settings, atau lewat variabel lingkungan yang
selalu menang atas isi berkas:

```
SCIENCEAPPGEN_APIKEY      SCIENCEAPPGEN_ENDPOINT      TAVILY_API_KEY
```

### Provider

| Provider | Butuh | Catatan |
|---|---|---|
| **AzureOpenAI** | kunci + endpoint | `llm.model` adalah nama **deployment** |
| **OpenAI** | kunci | Kosongkan endpoint kecuali memakai gateway yang kompatibel |
| **Anthropic** | kunci | Lihat di bawah |
| **Google** | kunci | Gemini |
| **Ollama** | endpoint | Tanpa kunci. Biasanya `http://localhost:11434`; model harus sudah di-pull |

**Anthropic tidak punya connector Semantic Kernel resmi.** Claude dilayani oleh
`Services/AnthropicChatCompletionService.cs`, ditulis langsung terhadap Messages API. Tiga hal
berbeda dari bentuk OpenAI dan ditangani secara eksplisit: system prompt berupa field `system` di
tingkat atas, bukan sebuah pesan; pesan berperan sama yang berurutan digabung karena API menolaknya;
dan `max_tokens` bersifat wajib, bukan opsional.

**Model reasoning mengubah kontrak permintaan.** Keluarga o1, o3, o4, dan gpt-5 menolak
`max_tokens` dan menuntut `max_completion_tokens`, dan sebagian besar hanya menerima temperature
bawaan. `AssistantService.UsesCompletionTokenLimit` mendeteksinya dan menyesuaikan permintaan. Ini
ditemukan dengan menjalankannya terhadap endpoint sungguhan, bukan dengan membaca dokumentasi.

---

## Template

**File → New Project** menawarkan lima belas template. Semuanya build dan berjalan apa adanya —
klaim itu diperiksa dengan membangkitkan setiap template lalu mem-build-nya, bukan sekadar
dinyatakan.

| Template | Menghasilkan |
|---|---|
| `blank` | Proyek konsol kosong |
| `numerics` | Array, aljabar linear, dekomposisi, statistik |
| `dataframe` | Memuat CSV, group-by, pivot, describe |
| `ml-pipeline` | Split → scale → PCA → random forest, dengan laporan klasifikasi |
| `clustering` | k-means, DBSCAN, dan Gaussian mixture, dinilai dengan silhouette |
| `nlp` | Tokenisasi dwibahasa, stemming, sentimen |
| `graph` | PageRank, komunitas, dan GCN terlatih |
| `bayesian` | Posterior MCMC yang diuji terhadap jawaban konjugat eksak |
| `timeseries` | Rolling window, persentase perubahan, resampling kalender |
| `notebook` | Notebook .NET Interactive dengan grafik |
| `explainability` | Permutation importance, nilai Shapley, dan kalibrasi untuk model terlatih |
| `anomaly` | Deteksi kebaruan one-class SVM dan pengelompokan kerapatan HDBSCAN |
| `tokenizer` | Melatih tokenizer BPE dan bergaya SentencePiece pada korpus Anda sendiri |
| `ner` | Penanda entitas berbasis CRF yang dilatih dari anotasi format CoNLL |
| `forecasting` | Penyaringan Kalman, pemulusan, dan peramalan dengan ketidakpastian yang jujur |

Lima template yang ditambahkan pada v0.4 masuk ke kategori yang sudah ada — *Machine learning*,
*Natural language*, dan *Statistics* — sehingga pemilih mengelompokkannya menurut fungsinya, bukan
menurut kapan template itu ditulis.

Template disimpan di dalam kode, bukan sebagai berkas lepas, sehingga tidak mungkin hilang dari
salinan terpasang dan nama proyek disubstitusi dengan benar, bukan lewat cari-dan-ganti. Berkas
`.csproj` yang dihasilkan memuat path absolut ke sumber Gravicode.Science, sehingga proyek yang
dibuat di luar repositori tetap bisa di-build.

### Satu template, dari awal sampai jalan

Gambar berikut berasal dari satu kali jalan tanpa jeda: `ml-pipeline` dipilih di dialog, lalu
di-build dan dijalankan dari toolbar, tanpa ada yang diketik di antaranya.

Pemilih template memperlihatkan isi tiap template — satu pita per pustaka yang dipakai, dengan
warna pustaka itu, sehingga `ml-pipeline` langsung terbaca sebagai GraviNum + GraviFrame +
GraviLearn.

![Dialog proyek baru dengan template ml-pipeline terpilih](../screenshots/scienceappgen-new-project.png)

![Pemilih template digulir ke template v0.4](../screenshots/scienceappgen-templates.png)

Setelah digulir, tampak lima template yang ditambahkan pada v0.4. Pitanya bekerja dengan cara yang
sama: `ner` dan `tokenizer` terbaca sebagai GraviNum + GraviText, `forecasting` sebagai
GraviNum + GraviProb.

Proyek langsung terbuka dengan `Program.cs` di editor dan log mencatat apa yang dibuat serta di
mana lokasinya.

![Proyek IrisClassifier hasil generate terbuka di editor](../screenshots/scienceappgen-ml-pipeline.png)

**Run** mengompilasi dan menjalankannya di tempat. Panel output menampilkan hasil sungguhan —
classification report dan skor cross-validation lima lipat dari pipeline yang ditulis template.

![Proyek hasil generate berjalan dan mencetak classification report](../screenshots/scienceappgen-run.png)

---

## Tool milik Jack

Asisten ini punya 16 kernel function. Function calling itulah yang membuatnya menjadi pembangun
aplikasi, bukan sekadar jendela obrolan: model memutuskan untuk membuat proyek, menulis berkas, dan
mem-build-nya, lalu Semantic Kernel mengeksekusi panggilan itu dan mengembalikan hasilnya.

### Proyek (`ProjectPlugin`)

| Fungsi | Kegunaan |
|---|---|
| `CreateProject` | Membuat dari template lalu membukanya |
| `ListTemplates` | Apa saja yang tersedia |
| `WriteFile` / `ReadFile` / `DeleteFile` | Akses berkas di dalam proyek |
| `ListFiles` | Orientasi sebelum melakukan perubahan |
| `BuildProject` | Build dan mengembalikan **output kompiler**, sehingga error kembali untuk diperbaiki |
| `RunProject` | Build lalu jalankan, dan kembalikan output programnya |
| `GetProjectInfo` | Keadaan saat ini |

Setiap path diselesaikan terhadap akar proyek yang terbuka lalu diperiksa berada di dalamnya.
Pemeriksaan itu adalah batas keamanan seluruh permukaan tool: model boleh menyebut path apa pun, dan
traversal seperti `../../../etc` ditolak, bukan diikuti.

### Tool umum (`CommonToolsPlugin`)

| Fungsi | Kegunaan |
|---|---|
| `SearchInternet` | Pencarian web Tavily, untuk apa pun setelah batas pengetahuan |
| `ScrapeWebPage` | Mengambil halaman dan membuang markup-nya |
| `MathCalculation` | Aritmetika eksak — `sqrt`, `pow`, `log`, `min`, `max`, dan lainnya |
| `GetCurrentDateTime` | Jam, dengan dukungan zona waktu |
| `CalculateDateDifference` | Selisih dan pergeseran tanggal |

Semua ini ada karena model bahasa justru tidak andal pada hal-hal tersebut: ia tidak bisa tahu
tanggal hari ini, ia kalkulator yang buruk, dan pengetahuannya punya batas waktu.

### Rujukan API (`GravicodeReferencePlugin`)

| Fungsi | Kegunaan |
|---|---|
| `GravicodeReference` | Permukaan API sebenarnya sebuah library, beserta jebakannya |
| `GravicodeExample` | Program lengkap yang berfungsi untuk tugas umum |

Tanpa ini, model menulis pemanggilan yang tampak masuk akal tetapi tidak ada. Rujukan terkurasi
lebih murah daripada siklus build-perbaiki-build-ulang dan lebih andal daripada berharap library-nya
ada dalam data latih.

---

## Terverifikasi ujung ke ujung

`tools/ScienceAppGen` telah dijalankan terhadap endpoint Azure OpenAI sungguhan. Pengujiannya tidak
memeriksa apa yang *dikatakan* asisten, melainkan apa yang benar-benar ada di disk:

```
=== 1. Kernel function lokal ===
  PASS  MathCalculation          sqrt(2) * 100 / (3 + 4) = 20.2030508910442
  PASS  MathCalculation bersarang pow(2, 10) + max(3, 7) = 1031
  PASS  GetCurrentDateTime
  PASS  CalculateDateDifference
  PASS  GravicodeReference
  PASS  GravicodeExample

=== 2. Template ===
  PASS  Jumlah template (15)
  PASS  Template terbentuk
  PASS  Substitusi nama

=== 3. Path traversal ditolak ===
  PASS  Traversal ditolak
  PASS  Path normal diizinkan

=== 4. LLM langsung: pembuatan aplikasi ujung ke ujung ===
  PASS  Kernel terkonfigurasi (16 tool)
  PASS  Direktori proyek dibuat
  PASS  Program.cs ada
  PASS  Proyek hasil generate berhasil build
  PASS  Proyek hasil generate berjalan

----- output program -----
Total revenue by region:
North: 2180.70
South: 1956.00
East: 670.75

Best-selling product overall: Widget (2611.65)

=== 5. Pencarian Tavily ===
  PASS  SearchInternet mengembalikan hasil
```

Dari prompt *"buat proyek konsol bernama SalesReport, definisikan record Sale, cetak pendapatan per
region terurut dari terbesar dan produk terlaris, lalu build"*, Jack membuat proyeknya, menulis
`Program.cs`, dan mem-build-nya — dan harness membangun ulang serta menjalankan hasilnya secara
independen untuk memastikan outputnya benar.

Jalur template diuji terpisah lewat UI-nya sendiri — proses yang tergambar di atas — dan dari situ
bug referensi `.csproj` ditemukan: path-nya relatif terhadap repositori, sehingga setiap proyek
yang dibuat di lokasi lain gagal di-build.

---

## Arsitektur

```
tools/ScienceAppGen/
  Models/AppSettings.cs           app.config bertipe kuat
  Services/
    ConfigurationService.cs       baca/tulis XML yang menjaga komentar
    ProjectService.cs             akses berkas + dotnet CLI, dengan penjaga path
    TemplateService.cs            sepuluh template
    AssistantService.cs           perkabelan Semantic Kernel, streaming, penampilan tool
    AnthropicChatCompletionService.cs   Claude, karena SK tidak punya connector
    LogService.cs                 sink log bersama
  Plugins/                        16 kernel function
  ViewModels/                     shell, explorer, editor, chat
  Views/                          AXAML + code-behind
  Themes/Tokens.axaml             token warna, tipografi, dan spasi
  Themes/Controls.axaml           gaya kontrol
```

Dua catatan implementasi yang perlu diketahui sebelum menyunting:

- **`ConfigurationService` menulis XML secara langsung**, bukan lewat `ConfigurationManager`, yang
  bisa membaca `appSettings` tetapi tidak bisa menulisnya kembali tanpa mengubah urutan dan format
  berkas.
- **Binding tidak boleh memuat cast di dalam path.**
  `{Binding $parent[Window].((vm:ShellViewModel)DataContext).X}` bisa dikompilasi tetapi melempar
  exception saat startup — tipenya tidak dapat diselesaikan pada runtime. Gunakan
  `{Binding $parent[Window].DataContext.X}`.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
