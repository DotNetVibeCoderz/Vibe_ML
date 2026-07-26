# Profesor Wicak

Asisten data scientist yang tinggal di halaman designer. Dibangun di atas Semantic Kernel.

---

## Menyalakannya

**Pengaturan → Profesor Wicak.** Isi kredensial minimal satu penyedia, pilih penyedia aktif,
simpan. Panel chat langsung berfungsi.

| Penyedia | Yang perlu diisi | Catatan |
|---|---|---|
| OpenAI | API key, model | Connector resmi. Endpoint kustom didukung untuk gateway kompatibel. |
| Anthropic | API key, model | Lewat `Anthropic.SDK` → `IChatClient` → Semantic Kernel. Microsoft tidak merilis connector Anthropic. |
| Gemini | API key, model | Connector Microsoft, masih prerelease. |
| Ollama | Endpoint, model | Untuk model lokal. Connector prerelease. |

Kredensial dilindungi dengan ASP.NET Core data protection sebelum ditulis ke database.

---

## Yang bisa diatur

| Pengaturan | Efeknya |
|---|---|
| **Persona / system prompt** | Membentuk cara dia menjawab dan seberapa berani dia bertindak. Ini kendali paling berpengaruh. |
| **Temperature** | Rendah membuat jawaban stabil antar-permintaan. 0.4 adalah bawaan yang sehat untuk pekerjaan data. |
| **Token maksimum** | Batas panjang satu jawaban. |
| **Giliran riwayat** | Berapa banyak giliran lampau yang dikirim ulang. Menaikkannya memperbaiki ingatan tapi memperbesar tiap permintaan. |
| **Izinkan panggil fungsi** | Mematikannya membuat dia hanya bisa bicara, tidak bisa membaca data atau mengubah kanvas. |

---

## Kernel function

Inilah yang memisahkan asisten ini dari chatbot tempelan. Dia tidak menebak isi datamu — dia
memeriksanya.

### Dataset

| Fungsi | Kegunaan |
|---|---|
| `ListDatasets` | Mendaftar dataset beserta ukurannya |
| `DescribeDataset` | Profil tiap kolom: tipe, nilai kosong, unik, rentang, nilai tersering |
| `QueryDataset` | Membaca baris sungguhan, dengan filter opsional |
| `ValueCounts` | Menghitung sebaran nilai sebuah kolom — untuk memeriksa keseimbangan kelas |

### Designer

| Fungsi | Kegunaan |
|---|---|
| `ListModules` | Modul yang tersedia beserta port dan parameternya |
| `GetExperiment` | Alur eksperimen yang sedang terbuka |
| `AddModule` | Menaruh modul di kanvas |
| `ConnectModules` | Menyambungkan dua modul, dengan pemeriksaan tipe port |
| `SetParameter` | Mengatur satu pengaturan modul, dengan validasi pilihan |
| `RemoveModule` | Menghapus modul beserta sambungannya |
| `ValidateExperiment` | Memeriksa masalah tanpa menjalankan |
| `RunExperiment` | Menjalankan dan melaporkan hasilnya |

Setiap perubahan tersimpan sebagai **versi baru** eksperimen, dengan catatan bahwa Wicak yang
melakukannya. Jadi apa pun yang dia kerjakan bisa ditinjau dan dikembalikan seperti perubahan
manusia.

### Umum

| Fungsi | Kegunaan |
|---|---|
| `SearchWeb` | Pencarian web lewat Tavily (perlu kunci di Pengaturan → Tools) |
| `ScrapeUrl` | Membaca teks sebuah halaman web |
| `ReadFileFromUrl` | Mengunduh berkas teks, CSV, atau JSON dari URL |
| `CurrentDateTime` | Tanggal dan waktu, dengan dukungan zona waktu |
| `DaysBetween` | Selisih hari antara dua tanggal |
| `Calculate` | Evaluasi ekspresi aritmetika |
| `Statistics` | Mean, median, min, maks, standar deviasi dari deret angka |

---

## Lampiran

Panel menerima gambar dan dokumen.

**Gambar** dikirim ke model sebagai konten gambar — dia benar-benar melihatnya. Berguna untuk
menanyakan grafik atau tangkapan layar hasil.

**Dokumen** disertakan sebagai tautan di dalam pesan. Kalau isinya relevan, Wicak akan
mengambilnya sendiri lewat `ReadFileFromUrl`.

Berkas diunggah ke storage yang sedang aktif, jadi ikut aturan penyimpanan yang sama dengan
dataset.

---

## Sesi

Tiap sesi punya riwayatnya sendiri. Buat sesi baru untuk topik baru supaya konteksnya tidak
tercampur. **Kosongkan sesi** membuang pesannya tapi mempertahankan sesinya.

Judul sesi diambil otomatis dari pertanyaan pertama.

---

## Modul LLM di kanvas

Berbeda dari panel chat: enam modul LLM (Format, Cleanse, Sentiment, Classify, Extract, Custom
Prompt) memakai penyedia yang sama, tapi **sengaja tanpa akses fungsi**. Sebuah node LLM
Sentiment tugasnya melabeli kolom, bukan memutuskan untuk mengubah eksperimenmu.

Modul-modul ini mengirim baris dalam batch dan mem-parse balasan sebagai array JSON. Kalau satu
batch balasannya tidak sejajar, ia mengulang baris per baris supaya baris yang baik-baik saja
tetap dapat jawabannya — satu baris rewel tidak boleh menggagalkan satu node.
