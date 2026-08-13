# Benchmark

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
| CPU | x86-64-v3, 8 prosesor logis, AVX2 + FMA (lebar `Vector<double>` 4) |
| GPU | Intel UHD Graphics 620 (terintegrasi), OpenCL |
| Runtime | .NET 10.0.11, RyuJIT, Server GC |
| Job | `ShortRun` — 3 iterasi pemanasan, 5 iterasi terukur |

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
| SVD (Jacobi satu sisi) | ~8× | Sekaligus memberi rank dan condition number |
| Eigen simetris (Jacobi) | ~6× | Iteratif; biaya bergantung pada spektrum |

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

## Membaca angka ini dengan jujur

1. **Ukur di perangkat keras Anda sendiri.** Hasil GPU di atas akan terbalik pada kartu komputasi
   diskret.
2. **`ShortRun` menukar presisi dengan waktu.** Beberapa margin di sini melebihi 10% dari rata-rata.
   Gunakan job bawaan bila sebuah angka harus benar-benar dipercaya.
3. **Benchmark bukan pengganti profiling.** Benchmark memberi tahu mana dari dua implementasi yang
   lebih cepat, bukan di mana program Anda menghabiskan waktunya.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
