# Blazor ML Studio (BlazorML)

Platform drag-and-drop berbasis web untuk menyusun, melatih, dan menerbitkan model machine
learning tanpa banyak menulis kode — dengan asisten data scientist **Profesor Wicak** yang bisa
memeriksa data dan menyusun alur eksperimen untuk kamu.

Dibuat oleh **Gravicode Studios**, di-lead oleh **Kang Fadhil**.

🇬🇧 [English version below](#english) · 📋 [Roadmap](PLAN.md) · 📊 [Status pengembangan](Progress.md) · 📚 [Dokumentasi](docs/)

---

## Jalankan dalam satu menit

Prasyarat: **.NET SDK 10**.

```bash
git clone <repo>
cd BlazorML
dotnet run --project src/BlazorML.Web
```

Menjalankan tesnya:

```bash
dotnet test        # 684 tes; 63 di antaranya membuka browser sungguhan
```

Tes kanvas butuh browser Playwright sekali unduh; tanpa itu ke-63 tes tersebut **ter-skip dengan
alasan**, bukan lolos diam-diam.

> 🔑 **Kunci API jangan ditaruh di `appsettings.json`.** File itu ikut ter-commit, dan ada tes
> yang menggagalkan build kalau ada kredensial di sana. Pakai user secrets:
> `dotnet user-secrets --project src/BlazorML.Web set "Chat:OpenAI:ApiKey" "sk-..."`

Buka alamat yang tercetak di konsol. Saat pertama kali menyala, aplikasi membuat skema database
SQLite, lalu mengisi 5 pengguna contoh, 5 dataset, dan 6 eksperimen siap jalan.

**Akun contoh** — semuanya berkata sandi `StudioML#2026`:

| Email | Peran |
|---|---|
| `admin@gravicode.com` | Administrator |
| `wina@gravicode.com` | Data Scientist |
| `bagus@gravicode.com` | Data Scientist |
| `sari@contoh.id` | Data Scientist |
| `tamu@contoh.id` | Viewer |

> ⚠️ Kredensial di atas hanya untuk pengembangan. **Ganti sebelum aplikasi dibuka ke siapa pun.**

Coba langsung: masuk → **Galeri** → *Prediksi churn pelanggan* → **Pakai contoh ini** → **Jalankan**.

---

## Yang bisa dilakukan

**Kanvas visual.** 63 modul terseret-lepas, dikelompokkan per kategori dan diberi warna pena
yang konsisten dari palette sampai port dan garis sambungannya. Menyambungkan port yang tipenya
tidak cocok ditolak dengan penjelasan, bukan didiamkan sampai run gagal.

**Data masuk dari mana saja.** Unggah CSV/TSV/JSON, tarik dari SQL (SQLite, SQL Server, MySQL,
PostgreSQL), ambil dari URL, atau ketik langsung. Statistik kolom dihitung saat impor, dan tiap
dataset bisa dibuka untuk **melihat baris aslinya** — 25, 100, atau 500 baris pertama, dibaca
seperlunya saja, bukan seluruh berkas.

**12 modul transformasi.** Pilih kolom, isi data kosong, buang duplikat, saring baris, bagi
train/test (dengan stratifikasi), join, agregasi, normalisasi, encoding kategori, featurisasi
teks, dan seleksi fitur.

**27 algoritma** di enam keluarga task: klasifikasi biner dan multikelas, regresi, clustering,
deteksi anomali, rekomendasi, dan peramalan.

**AutoML** yang mencari algoritma dan setelan terbaik dalam anggaran waktu, lengkap dengan
leaderboard supaya kamu tahu apa saja yang dicoba — bukan cuma yang menang.

**Evaluasi yang bisa dibaca.** ROC dan AUC, confusion matrix, residual, feature importance
lewat permutasi, dan cross-validation yang melaporkan sebaran antar-fold, bukan satu angka
yang kebetulan bagus.

**6 modul LLM** untuk pekerjaan yang sulit dituliskan aturannya: format, cleansing, sentimen,
klasifikasi zero-shot, ekstraksi field, dan prompt bebas.

**Script kustom** dalam C#, JavaScript, Python, dan R.

**Terbitkan jadi REST API.** Model terlatih menjadi endpoint berkunci API dengan dokumentasi
Swagger dalam satu klik. Halaman Endpoint memberi **contoh kode siap tempel dalam cURL, C#,
Python, dan Node.js** — dengan nama kolom dan nilai contoh dari model itu sendiri, bukan
placeholder — plus panel uji coba untuk mengirim permintaan sungguhan dan melihat responsnya.

**Reproducibility.** Setiap penyimpanan graph menjadi versi baru. Halaman riwayat menampilkan
semua run beserta metriknya, grafik perbandingan antar-run, dan daftar versi yang bisa
dikembalikan — termasuk perubahan yang dilakukan Profesor Wicak.

**Auth lengkap.** Masuk, keluar, daftar, profil, dan reset kata sandi lewat tautan bertoken.
Tanpa server email pun alurnya tetap jalan: tautannya ditulis ke log aplikasi.

### Profesor Wicak

Asisten yang tinggal di halaman designer. Dibangun di atas Semantic Kernel dengan pilihan
penyedia **OpenAI, Anthropic, Gemini, atau Ollama**. Persona, temperature, dan modelnya bisa
diubah dari halaman Pengaturan.

Yang membedakannya dari chatbot tempelan: dia punya fungsi untuk **membaca data yang sebenarnya**
(daftar dataset, profil kolom, query baris, hitung keseimbangan kelas) dan untuk **menyusun serta
mengubah alur eksperimen** di kanvas yang sedang kamu buka. Ditambah pencarian web (Tavily),
pembacaan halaman URL, tanggal/waktu, dan kalkulasi.

Chat mendukung multi-sesi, lampiran gambar dan dokumen, serta render markdown penuh — tabel,
kode, gambar, video, audio.

---

## Arsitektur

Lima project, dependensi mengalir satu arah:

```
BlazorML.Core ──────────┬──────────────┬───────────────┐
  domain, ExperimentGraph,│              │               │
  ModuleCatalog,         ▼              ▼               ▼
  abstraksi        Infrastructure ──► BlazorML.ML ──► BlazorML.Agents
                   EF Core (4 DB)     ML.NET,          Semantic Kernel,
                   4 storage          run engine       kernel functions
                        │              │               │
                        └──────────────┴───────────────┴──► BlazorML.Web
                                                            Blazor Server + Minimal API
```

Alasan pembagiannya, dan keputusan yang mengikat, ada di [PLAN.md](PLAN.md).

**Menambah modul baru cukup satu deklarasi** di `ModuleCatalog`. Palette, renderer node,
inspector parameter, dan validasi graph semuanya membaca dari sana — tidak ada kode UI per-modul.

### Tumpukan teknologi

Blazor Server di .NET 10 · D3.js v7 (di-vendor lokal) · EF Core 10 · ML.NET 5 + AutoML ·
Semantic Kernel 1.78 · Markdig · Roslyn scripting · Jint

---

## Konfigurasi

Semua pengaturan bisa diubah dari dalam aplikasi lewat **Pengaturan**. Nilai di
`appsettings.json` menjadi dasar; perubahan dari UI disimpan di database dan menimpanya, jadi
tidak perlu menyunting berkas di server. Section berisi kredensial dilindungi dengan ASP.NET
Core data protection sebelum ditulis.

Satu pengecualian: **provider database** dibaca dari `appsettings.json` saat aplikasi menyala,
karena pengaturannya jelas tidak bisa disimpan di dalam database yang sedang dikonfigurasi.

| Bagian | Pilihan |
|---|---|
| Database | SQLite (bawaan), SQL Server, MySQL, PostgreSQL |
| Storage | FileSystem (bawaan), Azure Blob, S3, MinIO |
| LLM | OpenAI, Anthropic, Gemini, Ollama |

Ganti database dengan menyunting `Database:Provider` dan connection string yang sesuai, lalu
jalankan ulang aplikasi.

---

## Dokumentasi

| Berkas | Isi |
|---|---|
| [docs/arsitektur.md](docs/arsitektur.md) | Struktur solusi dan keputusan desain |
| [docs/modul.md](docs/modul.md) | Referensi 63 modul beserta port dan parameternya |
| [docs/panduan-pengguna.md](docs/panduan-pengguna.md) | Alur kerja dari impor data sampai endpoint |
| [docs/profesor-wicak.md](docs/profesor-wicak.md) | Konfigurasi asisten dan daftar kernel function |
| [docs/api.md](docs/api.md) | Endpoint scoring, autentikasi, dan contoh panggilan |
| [docs/deployment.md](docs/deployment.md) | Menjalankan di produksi, pilihan database dan storage |
| [PLAN.md](PLAN.md) · [Progress.md](Progress.md) | Roadmap dan status pengerjaan |

---

## Batasan yang perlu diketahui

Dinyatakan terbuka supaya tidak ada kejutan:

- **Python dan R** dijalankan sebagai proses anak, bukan in-process. Spesifikasi menyebut
  Python.NET; alasan penyimpangannya dijelaskan di [Progress.md](Progress.md). Modulnya perlu
  Python 3 / `Rscript` terpasang di mesin, dan akan bilang jelas kalau tidak ada.
- **AutoML vision dan NLP** ada, tapi runtime-nya opt-in: build dengan `-p:EnableVisionNlp=true` karena menarik ~1,2 GB binary native.
- **Impor Parquet** didukung penuh, baca dan tulis.
- **Modul LLM** perlu kunci API. Tanpa itu node-nya gagal dengan penjelasan, bukan diam-diam
  menghasilkan kolom kosong.
- Connector **Gemini dan Ollama** untuk Semantic Kernel masih prerelease dari Microsoft.
- **684 tes** menutupi setiap lapisan — logika inti, mesin ML, layer agent, komponen Razor,
  API scoring, dan kanvas D3 lewat browser sungguhan. **Setiap trainer dan setiap modul di
  katalog benar-benar dijalankan**, bukan sekadar divalidasi bentuknya.
- **Hanya SQLite dan FileSystem yang pernah benar-benar dijalankan.** Tiga database dan tiga
  storage provider lainnya kodenya lengkap dan tersambung, tapi belum satu query atau satu
  upload pun pernah dieksekusi terhadapnya.

---

<a name="english"></a>

# English

Blazor ML Studio is a web-based drag-and-drop platform for building, training and deploying
machine learning models with little or no code, with a data-scientist assistant that can inspect
your data and author experiments for you.

Built by **Gravicode Studios**, led by **Kang Fadhil**.

## Run it

Requires **.NET SDK 10**.

```bash
dotnet run --project src/BlazorML.Web
```

On first start the app creates its SQLite schema and seeds 5 users, 5 datasets and 6 runnable
experiments. Sign in with `admin@gravicode.com` / `StudioML#2026` — and change that before the
app is reachable by anyone.

To see it work end to end: **Galeri** → *Prediksi churn pelanggan* → **Pakai contoh ini** → **Jalankan**.

## What it does

A visual canvas with **63 modules**, colour-coded by category from the palette through to the
ports and the wires between them. **27 algorithms** across binary and multiclass classification,
regression, clustering, anomaly detection, recommendation and forecasting. **AutoML** with a
leaderboard, so you see what was tried rather than only what won. **Twelve transforms** for the
data preparation that actually takes the time. **Six LLM modules** for work that is easier to
describe than to specify. **Custom scripts** in C#, JavaScript, Python and R.

Evaluation gives you ROC and AUC, confusion matrices, residual plots, permutation feature
importance, and cross-validation that reports the spread across folds rather than one lucky
number. Trained models publish as API-key-protected REST endpoints with Swagger docs in one
click, and the Endpoint page hands you ready-to-run cURL, C#, Python and Node.js — built from
that model's own column names and sample values — plus a console for trying a real request.

Every graph save becomes a version you can inspect and roll back — including changes the
assistant made.

### Profesor Wicak

The assistant lives on the designer page, built on Semantic Kernel with a choice of **OpenAI,
Anthropic, Gemini or Ollama**. Persona, temperature and model are editable in the UI.

What makes it more than a bolted-on chatbot: it has functions to read your **actual data**
(list datasets, profile columns, query rows, check class balance) and to **build and edit the
experiment graph** on the canvas you have open. Plus web search, page reading, date arithmetic
and calculation. Multi-session, image and document attachments, full markdown rendering.

## Configuration

Everything is editable from the Settings page. `appsettings.json` provides the baseline; changes
made in the app are stored in the database and overlay it, so nobody edits files on the server.
Credential-bearing sections are protected with ASP.NET Core data protection before storage.

The one exception is the **database provider**, which is read from `appsettings.json` at startup —
it cannot be stored inside the database it configures. Four databases (SQLite, SQL Server, MySQL,
PostgreSQL) and four storage backends (FileSystem, Azure Blob, S3, MinIO) are supported.

## Tests

```bash
dotnet test        # 684 tests; 63 of them drive a real browser
```

`tests/BlazorML.Tests` covers the core logic and trains real ML.NET models on data with a
planted signal, then checks the model found it. Every trainer in the catalogue is fitted and
scored with, driven from `ModuleCatalog` itself so a trainer added later is covered the moment it
is declared. `tests/BlazorML.Agents.Tests` exercises the whole
agent layer — including the functions Profesor Wicak uses to build experiments — with no provider
call at all: building a kernel is local work, and a designer function is ordinary code the model
happens to invoke. `tests/BlazorML.Web.Tests` renders the Razor components and boots the whole
application through `WebApplicationFactory` to call the scoring API as an outside integrator would.

`tests/BlazorML.Canvas.Tests` drives the designer with a real Chromium: nodes and edges are drawn
by D3 over JS interop on a Blazor circuit, so nothing short of a browser can check it.

The suite was written after the app was working, and found twenty-one bugs that manual verification
had missed — including a published endpoint that returned 422 for every legitimate request
because its schema demanded the very column the caller was asking it to predict, a theme toggle
that never worked because a layout does not inherit a page''s render mode, an anomaly detector
that threw a raw ML.NET error on its own default settings, and a live API key committed to
`appsettings.json`.

## Password resets

Sign in, sign out, register, profile and password reset are all present. A reset link is emailed
when SMTP is configured under Settings → Email; when it is not, the link is written to the
application log so a self-hosted administrator can still pass it on. The endpoint answers a known
and an unknown address identically, so it cannot be used to discover who has an account.

## Known limits

Python and R run as child processes rather than in-process — the spec asked for Python.NET, and
the reasoning for the deviation is recorded in [Progress.md](Progress.md). AutoML covers
classification and regression; vision and NLP are there but their runtimes are opt-in (`-p:EnableVisionNlp=true`,
~1.2 GB of native binaries). Parquet import and export are supported. LLM modules
need an API key and say so plainly when one is missing.

Every layer now has automated tests. What is *not* covered is that only SQLite and the file-system
store have ever actually executed: the other three databases and three storage backends compile
and are wired up, but no query or upload has run against them.

## Licence and attribution

Built by Gravicode Studios, led by Kang Fadhil.
