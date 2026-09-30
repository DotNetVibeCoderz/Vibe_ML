# Benchmark: HF.Net melawan implementasi referensi Python

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

Tujuan halaman ini bukan untuk menang. Tujuannya adalah mengetahui posisi HF.Net terhadap
implementasi yang dipakai semua orang, supaya dokumentasi bisa memberi tahu alat mana yang tepat —
dan benar.

Kedua sisi mengukur **masukan yang sama di mesin yang sama dalam sesi yang sama**, dan keduanya
melaporkan waktu **terbaik** dari beberapa kali jalan setelah pemanasan. Pada laptop yang mengalami
throttling, rata-rata hanya mengukur suhu ruangan.

| | |
|---|---|
| Mesin | Windows 11 (10.0.26200), 8 core logis |
| Python | 3.12.10 — transformers 5.17.0, tokenizers 0.23.2, torch 2.14.0+cpu (4 thread) |
| .NET | 10.0.12, Release |

Cara mereproduksinya ada di bagian bawah. Keluaran mentahnya ada di
[`benchmarks/comparison/report.md`](../../benchmarks/comparison/report.md).

---

## Tokenisasi — HF.Net lebih cepat

`bert-base-uncased`. Sisi Python adalah crate Rust `tokenizers` di balik binding tipis, yang
merupakan pilihan tercepat di Python; GraviTokenizers adalah C# managed.

| Ukuran | Python (Rust) | HF.Net (C#) | |
|---|---:|---:|---|
| Satu dokumen | 0,03 ms | 0,01 ms | **2,6x lebih cepat** |
| 1.000 dokumen | 15,52 ms | 8,70 ms | **1,78x lebih cepat** |

**64.441 dok/detik** melawan **114.887 dok/detik**.

Hasil ini tidak semengejutkan kelihatannya. Kedua implementasi menjalankan penelusuran
longest-match greedy yang sama; crate Rust membayar penyeberangan batas Python pada setiap panggilan,
sedangkan versi managed menyimpan hasil segmentasi per kata — teks nyata sangat sering mengulang
kata, dan perulangan merge adalah bagian yang mahal.

### Dan id-nya identik

Kecepatan hanya berarti bila keluarannya sama. Dan memang sama:

```
input   Hello, world! Tokenizers are unbelievable.
python  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
hf.net  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
```

## Membaca safetensors — daftar seimbang, pembacaan tidak

`model.safetensors` milik `bert-base-uncased`: 420 MB, 206 tensor.

| Ukuran | Python | HF.Net | |
|---|---:|---:|---|
| Membuka dan mendaftar semua tensor | 0,49 ms | 0,79 ms | 1,6x lebih lambat |
| Membaca satu tensor 30.522 × 768 | 0,49 ms | 149 ms | 306x lebih lambat |

Pendaftaran seimbang, karena keduanya hanya membaca header.

**Membaca tensor tidak seimbang, dan alasannya struktural.** PyTorch mengembalikan view tanpa
salinan atas byte F32 sebagaimana tersimpan di disk. HF.Net melebarkan setiap nilai ke `double`,
karena itulah yang ditampung `NdArray` — 23,4 juta nilai, 187 MB dimaterialisasi. Biaya ini dibayar
sekali per tensor saat memuat, bukan pada setiap forward pass.

> Tabel ini juga tempat ditemukannya **bug kinerja 600x**. Pembaca dulu mengalokasikan dan menyalin
> prefiks tetap 256 MB untuk mengurai header sekitar dua puluh kilobyte, sehingga mendaftar tensor
> memakan 429 ms. Membaca panjang header yang dideklarasikan terlebih dulu, lalu tepat sebanyak itu,
> membawanya ke bawah satu milidetik. Hanya perbandingan dengan implementasi referensi yang bisa
> membuatnya terlihat — sendirian, ia tampak cukup cepat.

## Inferensi encoder

Satu forward pass atas 12 token.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 dokumen | 37,6 ms | 48,1 ms | 1,28x lebih lambat |
| bert-base-uncased, 8 dokumen | 175 ms | 386 ms | 2,2x lebih lambat |
| bert-tiny, 1 dokumen | 1,15 ms | 0,83 ms | **1,39x lebih cepat** |

Versi-versi halaman ini sebelumnya mencatat **300–600 ms** untuk satu dokumen bert-base, lalu 111 ms.
Apa yang berubah dijelaskan di bawah; yang tidak berubah adalah presisinya. Encoder managed
sepakat dengan torch **dalam float64 hingga sekitar 1e-13** pada hidden state bert-base — angka yang
menyatakan bahwa kecepatan ini tidak dibeli dengan mengorbankan ketepatan.

### Ke mana waktunya habis, dan apa yang dilakukan

Diukur per blok encoder sebelum apa pun diubah:

| | 12 token | 197 posisi (ViT) | 577 posisi (ViT 384 px) |
|---|---:|---:|---:|
| Attention milik fondasi, perulangan inti | ~2 ms | **~298 ms** | **~2.760 ms** |
| Empat proyeksi Q/K/V/O | 4,2 ms | 38 ms | 76 ms |
| Pasangan feed-forward | 9,6–15 ms | 91–165 ms | 178–618 ms |

Empat perubahan, masing-masing diukur sebelum dipertahankan:

1. **Attention.** `MultiHeadAttention` milik fondasi adalah perulangan indexer yang berurutan. Milik
   HF.Net menyalin key dan value tiap head ke blok yang bersebelahan, menjadikan setiap skor satu dot
   product tervektorisasi, dan membagi kerja per head dan per blok query. Perulangan intinya **sekitar
   20x lebih cepat**, dengan hasil yang sama hingga 1e-15.
2. **Satu kernel linear untuk semuanya: GEMM dengan blok register.** Bobot disimpan dalam panel
   berisi dua belas output, dan kernel intinya memegang blok hasil 4 baris kali 12 output di dua belas
   register AVX: satu input yang di-broadcast, tiga vektor bobot, dua belas fused multiply-add per
   langkah. Kernel dot product yang digantikannya butuh satu load untuk setiap multiply-add. Bobot
   float32 dilebarkan ke buffer per thread 256 input sekaligus dan dipakai ulang oleh setiap kelompok
   empat baris. Melebarkan di dalam perulangan inti 40% lebih lambat, karena konversinya berebut port
   yang sama dengan multiply-add. Dengan satu thread ia berjalan 17,6 GMAC/s dibanding 5,6 milik
   kernel lama; di semua core, 35-54 GMAC/s pada 80 baris dan sekitar 40 pada 577, dibanding sekitar
   12 sebelumnya.
3. **Bobot dalam float32, aktivasi dalam double.** Setiap checkpoint menyimpan F32 atau lebih sempit,
   jadi float32 memuat bobot secara eksak sambil mengurangi separuh byte yang dialirkan per forward
   pass. Aktivasi dan setiap penjumlahan tetap `double`, itulah sebabnya kesepakatan 1e-13 bertahan.
4. **JIT.** Metode yang penuh perulangan dan hanya dipanggil beberapa kali berjalan sebagai kode tier-0
   yang belum dioptimalkan — dan satu forward pass memang hanya "beberapa kali": kernel linear terukur
   **12x lebih lambat** sebelum tiering menyusul. Metode yang panas ditandai `AggressiveOptimization`.

Dua perbaikan kecil lain muncul dari profiling. GELU eksak menghabiskan 40% waktu satu lapisan
feed-forward di dalam deret erf 50 suku milik fondasi; erf berbasis tabel ditambah Taylor 3x lebih
cepat dan lebih dekat ke erf yang dibulatkan dengan benar. Embedding patch ViT membaca piksel lewat
indexer yang mengalokasikan memori pada setiap panggilan: 74 ms, kini 8.

### Yang tersisa

GEMM ini membawa satu dokumen bert-base dari 111 ms ke 48 ms dan delapan dokumen dari 1.104 ms ke
386 ms, dengan hasil yang sama: fill-mask masih sepakat dengan torch hingga desimal keempat sebuah
persentase, dan lima teratas ViT hingga 6e-16. Sisa selisihnya adalah soal presisi. torch berjalan
dalam float32 dan mengerjakan dua kali lebih banyak multiply-add per instruksi; HF.Net menyimpan
setiap aktivasi dan penjumlahan dalam `double`, dan di situlah kesepakatan 1e-13 bertumpu. Mode
float32 sebagai opsi ada di [PLAN.md](../../PLAN.md). Pada ViT, GELU eksak di feed-forward dan atensi
atas 197 posisi kini mengambil porsi waktu yang lebih besar.

### Vision

`google/vit-base-patch16-224` pada satu gambar 224 × 224. Gambarnya adalah rumus yang dihitung kedua
sisi, bukan foto, jadi tidak ada resampler di antara keduanya.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 gambar | 214 ms | 948 ms | 4,4x lebih lambat |

Versi-versi halaman ini sebelumnya mencatat **12 detik**, lalu 1.964 ms. Pada 384 piksel, yang
memerlukan interpolasi position embedding, waktunya 6,8 detik dengan kernel linear sebelumnya,
dibanding 45,8 detik sebelum itu.

### Jalur produksi — lebih cepat daripada torch

Checkpoint `bert-base-uncased` yang sama, diekspor ke ONNX oleh sisi Python dan dijalankan dari .NET
melalui GraviOptimum pada provider CPU milik ONNX Runtime:

| Jalur | Satu dokumen, 12 token | terhadap torch |
|---|---:|---|
| torch (Python) | 37,6 ms | — |
| HF.Net managed | 48,1 ms | 1,28x lebih lambat |
| **HF.Net melalui ONNX Runtime** | **31,5 ms** | **1,19x lebih cepat** |

Hidden state ONNX berbeda dari yang managed paling banyak **4,7e-6**, sebesar aritmetika float32.
Versi halaman ini sebelumnya mengukur jalur ini pada model uji yang kecil karena ekspor model aslinya
belum tersedia; kini benchmark mengekspornya sendiri.

**Bila Anda butuh throughput, inilah jawabannya**, dan ini bukan kompromi: ia lebih cepat daripada
implementasi referensi, dari .NET. Lihat [GraviOptimum](GraviOptimum.md).

## Apakah keduanya sepakat? — hingga digit terakhir yang bermakna

Kecepatan adalah separuh yang mudah. Inilah separuh yang menentukan apakah semuanya bisa dipakai.

**`The capital of France is [MASK].`**

| Peringkat | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41,6790% | paris | 41,6788% |
| 2 | lille | 7,1416% | lille | 7,1416% |
| 3 | lyon | 6,3393% | lyon | 6,3392% |
| 4 | marseille | 4,4448% | marseille | 4,4447% |
| 5 | tours | 3,0297% | tours | 3,0297% |

**`He was a [MASK] player in the national team.`**

| Peringkat | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54,8921% | regular | 54,8917% |
| 2 | key | 19,4748% | key | 19,4747% |
| 3 | former | 5,8151% | former | 5,8151% |
| 4 | capped | 2,1575% | capped | 2,1575% |
| 5 | prominent | 1,3643% | prominent | 1,3643% |

Kolom Python adalah `pipeline`, yang menjalankan torch dalam float32; selisih di desimal keempat
sebuah persentase adalah milik float32. Terhadap torch dalam **float64**, probabilitas fill-mask
HF.Net sepakat hingga sepuluh angka desimal dan hidden state bert-base hingga sekitar 1e-13.

**`google/vit-base-patch16-224`**, torch dalam float64 melawan HF.Net pada piksel yang sama:

| Peringkat | torch | | HF.Net | |
|---|---|---:|---|---:|
| 1 | binder, ring-binder | 0,1186940263 | binder, ring-binder | 0,1186940263 |
| 2 | coil, spiral, volute, whorl, helix | 0,0446382711 | coil, spiral, volute, whorl, helix | 0,0446382711 |
| 3 | screen, CRT screen | 0,0433549419 | screen, CRT screen | 0,0433549419 |
| 4 | television, television system | 0,0315982656 | television, television system | 0,0315982656 |
| 5 | rubber eraser, rubber, pencil eraser | 0,0241216848 | rubber eraser, rubber, pencil eraser | 0,0241216848 |

Selisih terbesar: **1,3e-15**.

> **Apa yang ditemukan pemeriksaan presisi.** Versi halaman ini sebelumnya menyatakan encoder managed
> sepakat dengan torch "hingga sekitar sepersepuluh poin persentase" dan menyebut sisanya sebagai
> `double` melawan `float`. Itu keliru. Kedua encoder menjalankan **aproksimasi tanh** dari GELU,
> padahal setiap checkpoint ini meminta yang eksak, berbasis erf. Akibatnya hidden state bert-base
> meleset **2,8e-2** dari torch. Ini ditemukan dengan memberi kedua sisi masukan yang identik dalam
> presisi yang sama, yang menyingkirkan setiap penjelasan lain.

## Dua hal yang tidak bisa dibuka Python

Keduanya muncul saat menulis benchmark ini, dan keduanya ditangani HF.Net tanpa diminta:

- **`prajjwal1/bert-tiny` tidak punya `tokenizer.json`.** `transformers` 5.x menolak membangun
  tokenizer cepat dari `vocab.txt` tanpa `sentencepiece` terpasang. HF.Net kembali ke `vocab.txt`
  dengan sendirinya.
- **`config.json` milik `prajjwal1/bert-tiny` tidak punya kunci `model_type`.** `AutoModel`
  langsung menolaknya; arsitekturnya harus disebut eksplisit. HF.Net menganggap `model_type` yang
  hilang sebagai `bert` dan memuatnya.

Tak satu pun adalah hasil kinerja. Keduanya adalah jenis hal yang menentukan apakah sebuah pustaka
bisa membuka model yang benar-benar Anda miliki.

## Kesimpulan praktis

| Bila Anda sedang | Pakai |
|---|---|
| Men-tokenisasi teks dalam volume besar | **HF.Net** — lebih cepat, keluaran identik |
| Memuat dan memeriksa checkpoint | **HF.Net** — seimbang pada header, dan membaca format yang ditolak jalur cepat Python |
| Menjalankan encoder di produksi | **ONNX melalui GraviOptimum** — lebih cepat daripada torch, dari .NET |
| Menjalankan encoder untuk memahaminya, atau memeriksa sebuah ekspor | **HF.Net managed** — 3x torch untuk satu kalimat, dan eksak hingga 1e-13 dalam float64 |
| Melatih LoRA pada ratusan contoh | **HF.Net** — `PeftModel.Train`, eksak, dan adapternya bisa dimuat di PEFT Python; lihat [GraviPEFT](GraviPEFT.md) |
| Pelatihan skala besar | **Python.** HF.Net melatih di CPU, beberapa detik per epoch untuk beberapa lusin kalimat; latih di sana lalu layani adapternya di sini |

## Mereproduksi ini

```bash
cd benchmarks/comparison
pip install tokenizers transformers onnx
pip install torch --index-url https://download.pytorch.org/whl/cpu

python python/bench.py --out python/python.json     # juga mengekspor onnx/bert-base-uncased.onnx
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Jalankan kedua sisi dalam sesi yang sama. Jangan membandingkan angka yang diambil pada hari berbeda —
pada laptop yang mengalami throttling, selisih itu melampaui sebagian besar yang sedang diukur.
