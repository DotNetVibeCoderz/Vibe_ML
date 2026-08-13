?# Benchmark

*[English](../benchmarks.md)*

Semua angka di halaman ini diukur dengan [BenchmarkDotNet](https://benchmarkdotnet.org) pada mesin
yang dijelaskan di bawah. **Jangan dianggap berlaku umum.** Angka-angka ini ada untuk menunjukkan
bentuk kurva performa dan, pada satu kasus, mendokumentasikan hasil yang bertentangan dengan
asumsi umum.

## Menjalankannya

```bash
dotnet run --project benchmarks/GraviNum.Benchmark -c Release
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --filter "*MatrixProduct*"
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --list flat
```

Release bersifat wajib; BenchmarkDotNet menolak berjalan pada build Debug.

## Mesin acuan

| | |
|---|---|
| CPU | Intel Core i7-8650U, **4 core / 8 thread**, AVX2 + FMA (lebar `Vector<double>` 4) |
| GPU | Intel UHD Graphics 620 (terintegrasi), OpenCL |
| Runtime | .NET 10.0.11, RyuJIT, Server GC |
| Job | `ShortRun` — 3 iterasi pemanasan, 5 iterasi terukur |

### Catatan tentang mesin ini

Ini prosesor ultrabook 15 W, dan itu penting untuk membaca setiap angka di halaman ini.

- **Empat core fisik, bukan delapan.** `Environment.ProcessorCount` melaporkan 8 karena
  hyperthreading, dan kernel paralel membagi kerja menurut angka itu. Dua thread yang berbagi satu
  unit vektor tidak melipatduakan throughput floating-point.
- **Ia throttle di beban AVX berkelanjutan.** Ledakan singkat mencapai turbo; satu menit aljabar
  linear padat tidak. Build yang sama terukur 78 ms dan 168 ms untuk perkalian 1024-kubik dalam jam
  yang sama, murni karena kondisi termal dan beban latar.

Jadi angka absolut membawa ragam sekitar ±30% antar-run, dan rasio antara dua angka yang diukur di
**run berbeda** bukan bukti apa pun. Di mana halaman ini melaporkan sebelum/sesudah, kedua versi
dijalankan **berselang-seling dalam satu proses** lalu diambil yang terbaik dari sekian ulangan —
perbandingan itu tahan terhadap throttling karena kedua sisi mengalaminya sama rata. Rasio terhadap
tumpukan Python berasal dari run yang diambil berurutan pada kondisi yang sama.

---

## GraviNum — perkalian matriks

### Nilai vektorisasi

`LinAlg.Dot` mengakumulasi empat baris hasil sekaligus, sehingga setiap lintasan atas satu baris B
memberi makan empat rantai FMA yang independen. Dibandingkan triple loop klasik:

| Ukuran | Triple loop naif | `LinAlg.Dot` | Percepatan |
|---:|---:|---:|---:|
| 128 | 2.972 µs | 386 µs | **7,7×** |
| 256 | 38.481 µs | 1.918 µs | **20,1×** |

Pada 512×512 jalur produksi mencapai sekitar **24 GFLOP/s** di CPU ini, diukur langsung oleh
`samples/GraviNum.Console`.

Implementasi sebelumnya memanggil helper `Axpy` terpisah per baris B dan hanya mencapai
0,6 GFLOP/s. Seluruh selisihnya berasal dari register blocking dan menjaga loop SIMD tetap inline —
perlu diketahui sebelum menganggap "memakai SIMD" berarti "cepat".

### CPU versus GPU

Ini hasil yang perlu dibaca dengan saksama.

| Ukuran | `LinAlg.Dot` (CPU) | ILGPU (OpenCL) | Rasio GPU |
|---:|---:|---:|---:|
| 128 | 386 µs | 3.105 µs | 8,0× lebih lambat |
| 256 | 1.918 µs | 11.370 µs | 5,9× lebih lambat |
| 512 | 15.515 µs | 79.991 µs | 5,2× lebih lambat |
| 1024 | 132.032 µs | 940.932 µs | 7,1× lebih lambat |

**Pada perangkat keras ini GPU tidak pernah menang.** Dua sebabnya bersifat struktural, bukan
kebetulan:

1. Semua isi library bertipe `double`. GPU terintegrasi menjalankan float64 pada sebagian kecil
   laju float32-nya — sering 1/8 atau lebih buruk — sehingga keunggulan aritmetikanya hilang.
2. Setiap panggilan menyalin kedua operand melewati bus dan mengembalikan hasilnya. Biaya itu
   tetap sementara beban aritmetika bertambah, jadi paling terasa pada masalah kecil, tetapi tidak
   pernah hilang.

Kartu komputasi diskret dengan throughput float64 yang baik akan membalik hasil ini. Library tidak
bisa membedakan jenis perangkat, karena itu **pemilihan GPU otomatis bersifat opt-in**:

```csharp
Compute.AutomaticGpuDispatch = true;      // setelah mengukur di perangkat keras Anda
Compute.Preferred = Compute.Gpu;          // atau paksa langsung
```

Secara bawaan `Compute.Best` selalu mengembalikan backend CPU. Mengarahkan pekerjaan secara
diam-diam ke perangkat yang tujuh kali lebih lambat lebih buruk daripada tidak mengarahkan sama
sekali.

### Dekomposisi

Biaya pada 256×256, relatif terhadap LU:

| Faktorisasi | Biaya relatif | Catatan |
|---|---|---|
| Cholesky | ~0,5× | Hanya berlaku untuk matriks simetris definit positif |
| LU | 1,0× | Bawaan untuk keperluan umum |
| QR (Householder) | ~2× | Lebih stabil; dipakai oleh least squares |
| SVD (QR bidiagonal) | ~6× | Sekaligus memberi rank dan condition number |
| Eigen simetris (QL tridiagonal) | ~2× | Iteratif; biaya bergantung pada spektrum |

### Sparse versus dense

Matriks 2000×2000 dikali vektor. Biaya CSR mengikuti jumlah elemen tak-nol; biaya dense tidak.

| Kepadatan | Pemenang |
|---|---|
| 1% | Sparse, dengan selisih besar |
| 5% | Sparse |
| 25% | Dense — indireksi indeks kini lebih mahal daripada nol yang dilewati |

Aturan praktis yang dikonfirmasi: di bawah sekitar 10% kepadatan, gunakan `SparseMatrix`.

---

## GraviFrame — IO

### CSV streaming versus memory-mapped

Kedua jalur memakan **waktu yang kurang lebih sama**. Itu hasil jujurnya, dan alasan memilih
pembaca memory-mapped bukanlah kecepatan melainkan puncak pemakaian memori: pembaca streaming
menahan isi berkas bersamaan dengan frame hasil parsing, sedangkan pembaca memory-mapped
membiarkan sistem operasi memuat dan melepas halaman. Pada berkas yang muat di RAM, tidak ada
yang bisa dihemat.

Gunakan `ReadCsvMemoryMapped` bila berkas mendekati atau melampaui memori yang tersedia. Selain
itu gunakan `ReadCsv` — lebih sederhana dan tidak ada pemetaan yang bisa gagal.

> Satu bug yang layak dicatat: view harus dibuat dengan panjang berkas yang tepat. Memberi nilai
> `0` ("sampai akhir") membulatkan pemetaan ke ukuran halaman sistem, dan byte NUL di ekornya
> terbaca sebagai satu baris palsu. Kini ada tes regresi khusus untuk itu.

### CSV versus Parquet

Parquet kira-kira **4× lebih kecil** pada data contoh (14,6 KB berbanding 55,7 KB untuk frame
Titanic) dan lebih cepat dibaca karena tipe kolomnya tercatat, bukan disimpulkan ulang. Membaca
sebagian kolom hanya menyentuh kolom itu saja — di situlah tata letak kolumnar paling menguntungkan.

### Transformasi

Group-by, join, dan sort semuanya kira-kira linear terhadap jumlah baris. Satu ketimpangan yang
perlu diketahui: `Rolling(n).Mean()` memakai akumulator inkremental sehingga biayanya satu
penjumlahan dan satu pengurangan per baris berapa pun lebar jendelanya, sedangkan
`Rolling(n).Median()` mengurutkan ulang setiap jendela dan makin lambat saat jendela melebar.

---

## GraviLearn

### Biaya pelatihan

Pada 10.000 sampel dan 20 fitur, relatif terhadap satu decision tree:

| Model | Biaya relatif | Alasan |
|---|---|---|
| Gaussian naive Bayes | ≪ 1× | Satu lintasan menghitung rata-rata dan varians |
| Decision tree | 1× | Mengurutkan tiap fitur kandidat per node |
| Random forest (50 pohon) | ~10× | Pohon saling bebas, jadi diparalelkan |
| Gradient boosting (50 pohon) | ~50× | Sekuensial secara desain — tiap pohon menyesuaikan sisa sebelumnya |

### Biaya prediksi

Inilah yang penting bagi model yang sudah dipakai, dan urutannya berbeda dari pelatihan:

| Model | Biaya per baris |
|---|---|
| Regresi logistik | Satu perkalian titik |
| Random forest | 50 penelusuran pohon |
| k-nearest neighbours | Jarak ke **setiap** titik latih |

kNN dilatih seketika dan melayani dengan lambat. Pertukaran itu adalah definisi lazy learner, dan
itulah sebabnya kNN jarang bertahan menghadapi anggaran latensi.

### Di mana GPU bisa membantu

Regresi logistik menghabiskan hampir seluruh waktunya pada dua perkalian matriks per iterasi:
`X w` untuk menilai tiap sampel, dan `X' e` untuk mengakumulasi gradien. Bentuk itu persis yang
diukur benchmark GPU di atas — jadi pada mesin ini, memindahkan loop ke GPU justru akan
memperlambatnya, dengan alasan float64 yang sama.

---

## GraviText

### Inferensi transformer

Attention bersifat kuadratik terhadap panjang urutan: setiap token memperhatikan setiap token
lain. Menggandakan panjang urutan kira-kira **melipatempatkan** biaya attention sementara bagian
feed-forward hanya berlipat dua. Tembok skala itu terlihat langsung pada
`TransformerInferenceBenchmark`, dan itulah alasan semua teknik konteks panjang ada.

Batch diparalelkan dengan bersih karena urutan saling bebas — `EncodeBatch` memakai `Parallel.For`.

### Throughput pipeline

| Tahap | Biaya relatif |
|---|---|
| Tokenisasi regex | 1× |
| Tokenisasi WordPiece | ~3× — pencarian prefiks terpanjang per kata |
| Stemming Porter | ~2× |
| Transform TF-IDF (dense) | Didominasi ukuran matriks keluaran |

Untuk korpus nyata gunakan `CountVectorizer.TransformSparse`; matriks TF-IDF dense atas kosakata
besar sebagian besar berisi nol.

---

## GraviGraph

### Pelatihan GCN seiring bertambahnya graf

Setiap lapisan adalah propagasi sparse diikuti proyeksi dense:

- bagian sparse: `O(edge × lebar)`
- bagian dense: `O(node × fitur × lebar)`

Pada jaringan yang benar-benar jarang, proyeksi dense mendominasi — bagian itulah yang akan
dipercepat GPU. `GcnTrainingBenchmark` memisahkan keduanya agar pembagiannya terukur, bukan
diasumsikan.

### Algoritma

Pada 100.000 node, PageRank konvergen jauh di bawah satu detik. Betweenness centrality tetap
`O(VE)` bahkan dengan algoritma Brandes, itulah sebabnya `samples/GraviGraph.Console` membatasinya
pada subgraf 600 node — ini batas algoritmis, bukan batas implementasi.

---

## GraviProb

### Rantai berskala, bukan berulang

Rantai sepenuhnya independen, sehingga empat rantai memakan waktu dinding yang kira-kira sama
dengan satu rantai pada mesin dengan empat inti bebas. Karena itu pengambilan sampel multi-rantai
nyaris gratis — dan itu satu-satunya cara menghitung diagnostik konvergensi R-hat, yang
membandingkan varians antar-rantai dengan varians dalam-rantai.

### MCMC versus inferensi variasional

| | MCMC | Mean-field VI |
|---|---|---|
| Biaya | Detik | Milidetik |
| Hasil | Sampel dari posterior sebenarnya | Aproksimasi Gaussian |
| Korelasi | Tertangkap | Diabaikan |
| Sebaran posterior | Benar | Sistematis terlalu kecil |

Gunakan VI untuk iterasi cepat, lalu konfirmasi dengan MCMC. Keduanya diuji terhadap posterior
konjugat eksak dalam rangkaian tes, jadi tidak ada yang dipercaya begitu saja.

### Hidden Markov model

Forward, Viterbi, dan forward-backward semuanya linear terhadap panjang urutan dan kuadratik
terhadap jumlah state. Ketiganya bekerja di ruang log atau dengan penskalaan per langkah, karena
probabilitas mentahnya underflow dalam beberapa puluh langkah waktu — `HiddenMarkovBenchmark`
menjalankan urutan 10.000 langkah justru untuk menguji hal itu.

---

---

## Dibandingkan dengan ekosistem Python

Pertanyaan yang harus dijawab setiap library data science di .NET adalah bagaimana ia dibandingkan
dengan NumPy, pandas, dan scikit-learn. Berikut jawabannya berdasarkan pengukuran, tanpa dipoles.

### Cara menjalankan perbandingan

```bash
dotnet run --project benchmarks/comparison/Gravicode.Science.Comparison -c Release -- dotnet-results.json
python benchmarks/comparison/python_benchmarks.py python-results.json
python benchmarks/comparison/compare.py dotnet-results.json python-results.json
```

Kedua harness memakai **protokol yang sama**: bentuk data sama, fixture sama, dua kali pemanasan,
jumlah pengulangan tetap, dilaporkan mediannya. BenchmarkDotNet sengaja tidak dipakai di sini —
Python tidak punya padanannya, dan membandingkan statistiknya dengan `timeit` berarti
membandingkan dua metodologi pengukuran sekaligus dua implementasi. Jalankan **satu per satu**;
menjalankan keduanya bersamaan membuat mereka berebut inti CPU dan hasilnya tidak bermakna.

Mesin acuan: .NET 10.0.11 dan Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1,
networkx 3.6.1), 8 prosesor logis, AVX2.

### Aljabar linear dense — Python menang telak

| Operasi | Gravicode.Science | Python | Rasio |
|---|---:|---:|---|
| Perkalian matriks 256×256 | 2,58 ms | 0,41 ms | Python 6,3× |
| Perkalian matriks 512×512 | 15,64 ms | 3,07 ms | Python 5,1× |
| Perkalian matriks 1024×1024 | 93,45 ms | 27,76 ms | Python 3,4× |
| LU, 256×256 | 41,13 ms | 3,28 ms | Python 12,5× |
| QR, 256×256 | 153,63 ms | 10,36 ms | Python 14,8× |
| Cholesky, 256×256 | 12,24 ms | 1,23 ms | Python 10,0× |
| SVD, 256×256 | 308,41 ms | 27,11 ms | Python 11,4× |
| Eigen simetris, 256×256 | 110,22 ms | 19,74 ms | Python 5,6× |
| Selesaikan Ax=b, 256×256 | 66,78 ms | 6,18 ms | Python 10,8× |
| Invers matriks, 256×256 | 147,48 ms | 9,90 ms | Python 14,9× |

NumPy tidak mengerjakan ini di Python. Semuanya diserahkan ke LAPACK dan BLAS — Fortran dan
assembly yang disetel tangan selama puluhan tahun, ter-cache-block dan multithread. Kode terkelola
dengan `Vector<T>` tidak menutup jurang itu, dan library ini tidak berpura-pura sebaliknya.

#### Yang berubah di v0.2

SVD dan eigen simetris dulunya dua hasil terburuk di halaman ini dengan selisih jauh — 66× dan
139× — karena keduanya memakai metode Jacobi: menyapu seluruh matriks sambil merotasi pasangan
kolom sampai tidak ada lagi yang berubah. Benar, mudah diverifikasi, dan sangat boros, sebab setiap
sapuan menyentuh setiap elemen betapapun elemen itu sudah konvergen.

Keduanya kini mereduksi matriks sekali dengan refleksi Householder, lalu beriterasi pada bentuk
terkondensasi itu, di mana pergeseran (shift) membuat konvergensinya kubik:

| | v0.1 (Jacobi) | v0.2 | Peningkatan | vs NumPy, sebelum → sesudah |
|---|---:|---:|---|---|
| Eigen simetris, 256×256 | 2.739,93 ms | 116,21 ms | **23,6×** | 138,8× → 5,9× |
| SVD, 256×256 | 1.794,65 ms | 327,63 ms | **5,5×** | 66,2× → 12,1× |
| PCA ke 5 komponen, 20rb × 20 | 375,40 ms | 63,87 ms | **5,9×** | 63,5× → 10,8× |

Implementasi Jacobi tetap ada, sebagai `Decomposition.SvdJacobi` dan
`Decomposition.SymmetricEigenJacobi`. Keduanya bukan kode mati: pengujian menyandingkan jalur cepat
terhadap keduanya, dan karena kedua metode sampai pada faktorisasi yang sama lewat jalan yang sama
sekali berbeda, itu pemeriksaan sungguhan, bukan sekadar pengulangan.

PCA naik lebih banyak daripada SVD di bawahnya karena PCA juga berhenti menghitung faktor yang tak
pernah dibacanya. SVD 20.000×20 menghabiskan sebagian besar waktunya mengakumulasi `U`, satu baris
per sampel; PCA hanya butuh arah komponen dan variansinya, yang ada di `V` dan nilai singular.
`SvdRightVectors` dan `SingularValues` melewati apa yang tidak diminta pemanggil — nilainya identik
bit demi bit, karena rotasi yang menghasilkannya tetap berjalan.

Backend BLAS/LAPACK native tetap [ada di roadmap](../../PLAN.md) dan hanya itu cara menutup sisanya,
tetapi jurang yang harus ditutupnya kini satu orde besaran, bukan dua.

#### Aritmetika elemen, dan satu bug yang memakan 3×

Jalur elemen paralel dulu menyalin **kedua** operand ke array baru dan menyalin hasilnya kembali,
semata agar lambda bisa menangkapnya — span tidak bisa melintasi closure. Pada penjumlahan sejuta
elemen itu tiga lintasan tambahan atas 8 MB masing-masing, untuk menghemat nol. Mem-pin buffernya
dan menyerahkan pointer kepada worker menghapus semuanya:

| | sebelum | sesudah | peningkatan |
|---|---:|---:|---|
| Penjumlahan elemen, 1 juta | 7,23 ms | 2,36 ms | **3,06×** |
| Penjumlahan elemen, 10 juta | 82,33 ms | 27,62 ms | **2,98×** |

Diukur dengan menjalankan kedua versi berselang-seling dalam satu proses dan mengambil yang terbaik
dari lima belas — lihat *[catatan tentang mesin ini](#catatan-tentang-mesin-ini)*. Hasilnya identik
bit demi bit.

Operasi ini terhambat bandwidth memori, bukan aritmetika: angka yang penting adalah 2,9 → 10,2 GB/s
yang dibeli perbaikan ini, dan itulah sebabnya register SIMD yang lebih lebar tidak akan menolong.
Itu pula yang membuat aritmetika elemen kini melampaui NumPy di tabel di atas.

`Unary` sama sekali tidak punya jalur paralel dan kini punya, yang membuat `Exp` dan `Log` pada
array besar ikut berskala dengan jumlah core.

#### Perkalian matriks

Kernelnya kini menahan petak 4×vektor dari C di register sepanjang satu iris `k`, alih-alih memuat
dan menyimpan ulang C di setiap langkah `k`. Mengiris `k` itulah yang menjaga penelusuran strided
melalui B tetap di dalam cache; akumulasi register tanpa irisan justru *lebih lambat* pada 1024 ke
atas — perlu diketahui sebelum ada yang mencoba separuh perubahannya saja.

Diuji berselang-seling terhadap kernel sebelumnya, terbaik dari sekian ulangan: **1,7× pada 128**,
tanpa perubahan pada 256, dan **1,05–1,4×** dari 512 ke atas. Bacaan jujurnya: ini kemenangan
sederhana, bukan terobosan. Menutup sisa 3,4× terhadap NumPy memerlukan blocking tiga tingkat
dengan packing seperti yang dipakai OpenBLAS, dan itu pekerjaan yang jauh lebih besar.

Panelling kolom — memblok loop *j* — juga dicoba dan konsisten **lebih buruk** di setiap ukuran.
Dicatat di sini agar tidak ada yang menghabiskan sore hari menemukannya kembali.

### Array, statistik, dataframe — Python unggul, tetapi tidak sampai orde besaran

| Operasi | Gravicode.Science | Python | Rasio |
|---|---:|---:|---|
| **Penjumlahan elemen, 1 juta** | 2,52 ms | 4,80 ms | **.NET 1,9×** |
| **Penjumlahan elemen, 10 juta** | 26,70 ms | 47,21 ms | **.NET 1,8×** |
| Sparse matriks-vektor, 2000² @ 1% | 0,17 ms | 0,06 ms | Python 3,0× |
| 1.000.000 deviat normal | 17,01 ms | 16,22 ms | seimbang |
| Rata-rata + std atas 1 juta | 12,28 ms | 7,62 ms | Python 1,6× |
| Baca CSV 200.000 baris | 371,37 ms | 135,88 ms | Python 2,7× |
| Group-by mean, 500 grup | 39,88 ms | 5,86 ms | Python 6,8× |
| Urutkan kolom numerik | 47,85 ms | 27,24 ms | Python 1,8× |
| Filter kolom numerik | 6,39 ms | 6,06 ms | seimbang |
| **Rolling mean, jendela 30** | 1,03 ms | 8,83 ms | **.NET 8,6×** |

Rolling mean adalah pengecualian yang menjelaskan aturannya: `Rolling(n).Mean()` memakai akumulator
inkremental, sehingga biayanya satu penjumlahan dan satu pengurangan per baris berapa pun lebar
jendelanya, sedangkan pandas menghitung ulang seluruh jendela. Itu perbedaan **algoritmis**, dan
perbedaan algoritmis bertahan melewati jurang bahasa.

### Machine learning — campuran

| Operasi | Gravicode.Science | Python | Rasio |
|---|---:|---:|---|
| Regresi logistik, 20rb × 20 | 346,95 ms | 31,16 ms | Python 11,1× |
| **Random forest fit, 50 pohon** | 1.906,19 ms | 2.584,07 ms | **.NET 1,4×** |
| Random forest predict, 20rb baris | 134,54 ms | 98,20 ms | Python 1,4× |
| k-means, k=5, 3 restart | 2.541,98 ms | 497,53 ms | Python 5,1× |
| PCA ke 5 komponen | 63,87 ms | 5,92 ms | Python 10,8× |
| kNN predict, 2rb vs 5rb | 609,92 ms | 55,17 ms | Python 11,1× |

Random forest adalah satu-satunya model di mana library ini lebih cepat, dan alasannya
mendidik: pembentukan pohon adalah pekerjaan penuh percabangan, tidak ramah cache, dan mengejar
pointer — tidak ada panggilan BLAS yang bisa membantu. Bentuk masalah itu justru cocok untuk
bahasa ber-JIT dan buruk untuk interpreter; scikit-learn hanya bisa bersaing karena pohonnya
ditulis dalam Cython terkompilasi.

PCA 63× lebih lambat murni karena di bawahnya adalah SVD. Perbaiki SVD-nya, angka ini ikut bergerak.

### Teks, graf, sampling — .NET menang, dan selisihnya besar

| Operasi | Gravicode.Science | Python | Rasio |
|---|---:|---:|---|
| **TF-IDF fit, 20rb dokumen** | 262,50 ms | 691,75 ms | **.NET 2,6×** |
| **Tokenisasi regex 20rb dokumen** | 153,92 ms | 359,60 ms | **.NET 2,3×** |
| PageRank, 50rb node | 276,32 ms | 313,02 ms | seimbang |
| **BFS, 50rb node** | 9,54 ms | 376,63 ms | **.NET 39,5×** |
| **Komponen terhubung, 50rb node** | 9,28 ms | 38,60 ms | **.NET 4,2×** |
| **Dijkstra, 50rb node** | 32,34 ms | 180,42 ms | **.NET 5,6×** |
| **MCMC, 4 rantai × 5000 draw** | 21,37 ms | 279,18 ms | **.NET 13,1×** |
| **1.000.000 log densitas skalar** | 2,08 ms | 275,67 ms | **.NET 132,2×** |

Ini separuh cerita yang lain, dan separuh inilah yang membenarkan keberadaan proyek ini. Tidak ada
satu pun beban kerja di atas yang bisa divektorkan menjadi satu panggilan library. Penelusuran graf
adalah mengejar pointer. MCMC adalah loop sekuensial yang langkah berikutnya bergantung pada
sebelumnya. Loop log-densitas skalar persis hal yang paling buruk bagi interpreter. Di sini
perbandingannya adalah kode ber-JIT melawan bytecode CPython, dan hasilnya tidak berimbang.

Untuk loop log-densitas, padanan **tervektorisasi** NumPy memakan 12,17 ms — masih 5,8× lebih
lambat daripada loop skalar .NET, karena bentuk tervektorisasi harus memuat array antara berisi
1 juta elemen sementara loop menyimpan semuanya di register.

### Rekapitulasi, dan artinya

**.NET lebih cepat pada 12 pengukuran, Python pada 22, seimbang pada 3.**

Namun jangan berhenti di rekap itu, karena pembagiannya tidak acak:

- Di mana pun operasinya berujung pada **LAPACK, BLAS, atau Cython terkompilasi**, Python menang,
  kini konsisten 3–15×, bukan lagi 60–140× seperti biaya dekomposisi Jacobi dahulu.
- Di mana pun operasinya bersifat **skalar, bercabang, atau sekuensial**, .NET menang, 2–40× dan
  sekali mencapai 132×.
- Di mana pun operasinya **terhambat bandwidth memori** — aritmetika elemen — keduanya berdekatan,
  dan .NET kini sedikit unggul.

Dua hal menggerakkan angka-angka di v0.2. Penulisan ulang dekomposisi menghapus ujung ekstrem
rentang pertama, sehingga tidak ada lagi di halaman ini yang terpaut lebih dari sekitar 15× dari
tumpukan Python. Menghapus salinan dari jalur elemen kemudian membalik kedua barisnya dari
*Python 2–3×* menjadi *.NET 1,8–1,9×*.

Jadi ringkasan jujurnya: hari ini library ini bukan pengganti NumPy untuk aljabar linear dense, dan
menambahkan interop BLAS/LAPACK adalah satu perubahan yang paling akan memperbaikinya. Library ini
sudah mengungguli ekosistem Python pada algoritma graf, MCMC, tokenisasi, dan beban kerja apa pun
yang tersusun dari loop skalar ketat — sambil tetap berada dalam satu proses yang aman-tipe, mudah
di-deploy, dan ringan dependensi.

### Tiga bug yang ditemukan lewat perbandingan ini

Menjalankan dua ekosistem berdampingan memunculkan masalah yang tidak terlihat oleh benchmark
.NET saja.

**Jalur elemen paralel menyalin masukannya sendiri.** `RunBinaryContiguous` memanggil `ToArray()`
pada kedua operand dan mengalokasikan array ketiga untuk hasilnya, karena `Span<T>` tidak bisa
ditangkap lambda. Setiap `Add` besar karenanya memindahkan 24 MB yang tidak perlu. Petunjuknya
adalah tertinggal 2–3× dari NumPy pada operasi yang isinya *hanya* penyalinan memori; perbaikannya
adalah pointer `fixed`, dan itu membuat aritmetika elemen **3× lebih cepat** serta melampaui NumPy.

**Pemisahan decision tree bersifat O(n²).** `FindBestSplit` membentuk `sorted[..k]` dan
`sorted[k..]` pada setiap kandidat titik split lalu membangun ulang dictionary hitungan kelas dari
keduanya — pekerjaan O(n) pada masing-masing dari O(n) titik split. Pada 20.000 sampel, benchmark
random forest berjalan **83 menit tanpa selesai**. Setelah ditulis ulang menjadi sapuan atas urutan
terurut sambil memindahkan satu sampel setiap kali di antara hitungan kelas berjalan, kini selesai
dalam **1,9 detik** — dan mengungguli scikit-learn. Rangkaian tes GraviLearn pun ikut lebih cepat,
dari 4 dtk menjadi 1 dtk.

**Ada benchmark yang tidak mengukur apa pun.** Loop log-densitas skalar semula melaporkan 0,45 ms
untuk 1.000.000 iterasi — sekitar 1,5 siklus per iterasi, mustahil untuk sebuah logaritma. Hasilnya
dibuang, sehingga JIT menghapus loop-nya. Dengan sink non-inline yang mengonsumsi nilainya, angka
sebenarnya adalah 2,08 ms. Perlu dinyatakan terus terang: benchmark yang tampak terlalu bagus
biasanya memang begitu.

---

## Membaca angka ini dengan jujur

1. **Ukur di perangkat keras Anda sendiri.** Hasil GPU di atas akan terbalik pada kartu komputasi
   diskret.
2. **`ShortRun` menukar presisi dengan waktu.** Beberapa margin di sini melebihi 10% dari rata-rata.
   Gunakan job bawaan bila sebuah angka harus benar-benar dipercaya.
3. **Benchmark bukan pengganti profiling.** Benchmark memberi tahu mana dari dua implementasi yang
   lebih cepat, bukan di mana program Anda menghabiskan waktunya.
4. **Konsumsi apa yang Anda hitung.** Hasil yang dibuang berarti loop yang dihapus. Harness
   perbandingan mengalirkan setiap nilai lewat sink non-inline justru karena itu.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*

