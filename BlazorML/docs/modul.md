# Referensi modul

63 modul, dikelompokkan per kategori. Setiap kategori punya warna pena sendiri yang muncul
konsisten di palette, header node, port, dan garis sambungannya.

Definisi lengkapnya ada di `src/BlazorML.Core/Modules/ModuleCatalog.cs` — satu-satunya tempat
modul dideklarasikan.

**Tipe port:** `Dataset` · `UntrainedModel` · `TrainedModel` · `Metrics`

---

## Data in — pena biru (4)

| Modul | Masuk | Keluar | Catatan |
|---|---|---|---|
| **Import Dataset** | — | Dataset | Membaca dataset yang sudah terdaftar. Batas baris berguna saat masih bereksperimen. |
| **Import from SQL** | — | Dataset | Query ke SQLite, SQL Server, MySQL, atau PostgreSQL. Pakai akun read-only. |
| **Import from Web** | — | Dataset | CSV atau JSON lewat HTTP, dengan header kustom. |
| **Enter Data Manually** | — | Dataset | Tabel kecil yang diketik langsung. Berguna untuk tabel lookup. |

---

## Transform — pena teal (12)

| Modul | Masuk | Keluar | Catatan |
|---|---|---|---|
| **Select Columns** | Dataset | Dataset | Simpan atau buang kolom terpilih. |
| **Edit Metadata** | Dataset | Dataset | Ganti nama kolom atau ubah cara nilainya dibaca. Mengubah tipe ikut mengubah nilainya. |
| **Clean Missing Data** | Dataset | Dataset | Isi dengan mean/median/modus/nilai tetap, atau buang baris/kolomnya. |
| **Remove Duplicate Rows** | Dataset | Dataset | Berdasar kolom kunci, atau seluruh baris. |
| **Filter Rows** | Dataset | Dataset | Perbandingan angka maupun teks. |
| **Split Data** | Dataset | Dataset ×2 | Kiri = latih, kanan = uji. Stratifikasi menjaga keseimbangan kelas di kedua sisi. |
| **Join Datasets** | Dataset ×2 | Dataset | Inner, left, atau full. |
| **Group and Summarise** | Dataset | Dataset | Rata-rata, jumlah, min, maks, atau cacah per grup. |
| **Normalize Data** | Dataset | Dataset | Min-max, z-score, log, atau binning. |
| **Encode Categories** | Dataset | Dataset | One-hot, ordinal, atau hashing. One-hot menolak kolom dengan >200 nilai unik. |
| **Featurize Text** | Dataset | Dataset | Bag-of-n-grams dengan stop word Indonesia dan Inggris. |
| **Select Best Features** | Dataset | Dataset + Metrics | Peringkat lewat korelasi, mutual information, atau varians. |

---

## LLM actions — pena ungu (6)

Semuanya perlu kunci API penyedia model. Baris dikirim dalam batch; batch yang balasannya
tidak sejajar diulang baris per baris.

| Modul | Fungsinya |
|---|---|
| **LLM Format** | Menyeragamkan nilai berantakan ke satu bentuk. |
| **LLM Cleanse** | Memperbaiki typo, spasi liar, dan kapitalisasi tidak konsisten. |
| **LLM Sentiment** | Melabeli sentimen dengan skor keyakinan opsional. |
| **LLM Classify** | Memilah ke kategori yang kamu tentukan, tanpa data latih. |
| **LLM Extract** | Menarik field bernama dari teks bebas menjadi kolom baru. |
| **LLM Custom Prompt** | Prompt bebas per baris. `{{nama_kolom}}` disisipkan nilainya. |

---

## Algorithms — pena kuning (27)

Semuanya mengeluarkan `UntrainedModel`, yang disambungkan ke modul pelatihan.

### Klasifikasi biner (7)

Logistic Regression · SDCA Logistic Regression · Boosted Decision Tree · Decision Forest ·
LightGBM · Averaged Perceptron · Linear SVM

Boosted Decision Tree biasanya yang terkuat untuk data tabular. Logistic Regression memberi
bobot per fitur yang bisa dibaca langsung.

### Klasifikasi multikelas (5)

SDCA Maximum Entropy · L-BFGS Maximum Entropy · One-vs-All Trees · LightGBM Multiclass ·
Naive Bayes

### Regresi (8)

Linear Regression · Ordinary Least Squares · Boosted Decision Tree Regression ·
Decision Forest Regression · LightGBM Regression · Tweedie Regression · Poisson Regression ·
Online Gradient Descent

Tweedie untuk target yang sebagian besar nol dengan ekor panjang (klaim, belanja).
Poisson untuk cacahan.

### Clustering (1) · Anomali (2) · Rekomendasi (1) · Peramalan (1) · Vision (1) · NLP (1)

K-Means · PCA Anomaly Detection · Spike Detection · Matrix Factorization · SSA Forecasting ·
Image Classification · Text Classification

Matrix Factorization, SSA Forecasting, dan Spike Detection memakai jalur pelatihan khusus —
ketiganya tidak memakai vektor `Features` gabungan seperti trainer lain.

---

## Training — pena merah muda (5)

| Modul | Masuk | Keluar |
|---|---|---|
| **Train Model** | UntrainedModel + Dataset | TrainedModel |
| **Train Clustering Model** | UntrainedModel + Dataset | TrainedModel |
| **AutoML** | Dataset | TrainedModel + Metrics (leaderboard) |
| **Tune Hyperparameters** | UntrainedModel + Dataset | TrainedModel + Metrics |
| **Cross Validate** | UntrainedModel + Dataset | Metrics + TrainedModel |

**Train Model** butuh kolom label; **Train Clustering Model** tidak. Menyambungkan algoritma ke
modul yang salah ditolak saat validasi dengan penjelasan.

**Cross Validate** melaporkan rata-rata **dan** simpangan antar-fold. Simpangan yang lebar
berarti masalahnya di data, bukan di pilihan algoritma.

**Tune Hyperparameters** tahan terhadap satu kombinasi yang gagal — kombinasi itu dicatat di
log dan sweep-nya lanjut.

---

## Score & evaluate — pena hijau (3)

| Modul | Masuk | Keluar |
|---|---|---|
| **Score Model** | TrainedModel + Dataset | Dataset (dengan kolom prediksi) |
| **Evaluate Model** | Dataset hasil scoring | Metrics |
| **Feature Importance** | TrainedModel + Dataset | Metrics |

**Evaluate Model** menyesuaikan metriknya dengan tugas yang terdeteksi. Ia menghitung dari
baris hasil scoring, jadi ikut bekerja untuk tabel yang di-score di luar ML.NET — misalnya
hasil klasifikasi oleh modul LLM.

**Feature Importance** mengacak satu kolom pada satu waktu dan mengukur seberapa jauh skornya
jatuh. Kalau satu fitur mendominasi total, curigai kebocoran label.

---

## Script — pena tinta (4)

| Modul | Runtime | Kontrak |
|---|---|---|
| **Execute C#** | Terpasang bersama aplikasi | `rows`, `rows2` sebagai `List<Dictionary<string, object>>`. Kembalikan baris yang diteruskan. |
| **Execute JavaScript** | Terpasang bersama aplikasi | `rows`, `rows2` sebagai array objek. Kembalikan array. |
| **Execute Python** | Perlu Python 3 di mesin | Definisikan `run(dataset1, dataset2)` yang mengembalikan list of dict, atau isi variabel `result`. |
| **Execute R** | Perlu `Rscript` di mesin | `dataset1`, `dataset2` sebagai data.frame. Isi variabel `result`. |

Python dan R berjalan sebagai proses anak dengan timeout yang benar-benar berlaku. Kalau
runtime-nya tidak ada, node gagal dengan pesan yang menyebutkan apa yang perlu dipasang.

---

## Data out — pena biru (2)

| Modul | Masuk | Efeknya |
|---|---|---|
| **Save Dataset** | Dataset | Menulis kembali ke workspace sebagai dataset baru. |
| **Register Model** | TrainedModel | Menyimpan model beserta versinya, siap diterbitkan. |

Mendaftarkan model dengan nama yang sama membuat versi berikutnya, tidak menimpa.

---

## Menambah modul baru

Satu deklarasi di `ModuleCatalog`, lalu tangani id-nya di executor yang sesuai
(`src/BlazorML.ML/Execution/Executors/`). Palette, node di kanvas, inspector parameter, dan
validasi graph akan mengikuti sendiri — tidak ada kode UI yang perlu disentuh.
