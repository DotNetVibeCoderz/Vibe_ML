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
| Satu dokumen | 0,04 ms | 0,01 ms | **3,1x lebih cepat** |
| 1.000 dokumen | 20,60 ms | 9,60 ms | **2,15x lebih cepat** |

**48.553 dok/detik** melawan **104.154 dok/detik**.

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
| Membuka dan mendaftar semua tensor | 0,70 ms | 0,71 ms | seimbang |
| Membaca satu tensor 30.522 × 768 | 0,74 ms | 185 ms | 251x lebih lambat |

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
| bert-base-uncased, 1 dokumen | 50,1 ms | 59,5 ms | 1,19x lebih lambat |
| bert-base-uncased, 8 dokumen | 263 ms | 428 ms | 1,63x lebih lambat |
| bert-tiny, 1 dokumen | 1,75 ms | 1,05 ms | **1,66x lebih cepat** |

Versi-versi halaman ini sebelumnya mencatat **300–600 ms** untuk satu dokumen bert-base, lalu 111 ms.
Apa yang berubah dijelaskan di bawah; yang tidak berubah adalah presisinya. Encoder managed
sepakat dengan torch **dalam float64 hingga sekitar 1e-13** pada hidden state bert-base — angka yang
menyatakan bahwa kecepatan ini tidak dibeli dengan mengorbankan ketepatan.


Dengan `ComputeOptions.LinearLayers = Precision.Single` - lapisan linear dalam float32, seperti torch
menjalankannya - encoder managed menyalip torch untuk satu kalimat:

| Model | Python (torch) | HF.Net, linear float32 | |
|---|---:|---:|---|
| bert-base-uncased, 1 dokumen | 50,1 ms | 45,0 ms | **1,11x lebih cepat** |
| bert-base-uncased, 8 dokumen | 263 ms | 287 ms | 1,09x lebih lambat |
| bert-tiny, 1 dokumen | 1,75 ms | 1,37 ms | **1,28x lebih cepat** |

Hidden state-nya lalu bergeser paling jauh 9,9e-6 dari yang berpresisi ganda.

Setiap waktu torch di halaman ini lebih lambat daripada di versi sebelumnya (di sini 50 ms, dulu 37,6):
laptopnya berjalan lebih panas. Kedua sisi dijalankan berurutan dalam sesi yang sama, jadi rasionya
bisa dibandingkan; angka absolut dari hari yang berbeda tidak.

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
float32 opsional, di atas, adalah jawabannya. Pada ViT, GELU eksak di feed-forward dan atensi
atas 197 posisi kini mengambil porsi waktu yang lebih besar.

### Vision

`google/vit-base-patch16-224` pada satu gambar 224 × 224. Gambarnya adalah rumus yang dihitung kedua
sisi, bukan foto, jadi tidak ada resampler di antara keduanya.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 gambar | 345 ms | 913 ms | 2,65x lebih lambat |
| vit-base-patch16-224, linear float32 | 345 ms | 682 ms | 1,98x lebih lambat |

Versi-versi halaman ini sebelumnya mencatat **12 detik**, lalu 1.964 ms. Pada 384 piksel, yang
memerlukan interpolasi position embedding, waktunya 6,8 detik dengan kernel linear sebelumnya,
dibanding 45,8 detik sebelum itu.

### Jalur produksi — lebih cepat daripada torch

Checkpoint `bert-base-uncased` yang sama, diekspor ke ONNX oleh sisi Python dan dijalankan dari .NET
melalui GraviOptimum pada provider CPU milik ONNX Runtime:

| Jalur | Satu dokumen, 12 token | terhadap torch |
|---|---:|---|
| torch (Python) | 50,1 ms | — |
| HF.Net managed | 59,5 ms | 1,19x lebih lambat |
| **HF.Net melalui ONNX Runtime** | **28,9 ms** | **1,73x lebih cepat** |

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

Selisih terbesar: **5,7e-16**.

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

## Rangkaian per pustaka

`benchmarks/HFNet.Benchmarks` adalah satu proyek BenchmarkDotNet yang mencakup setiap pustaka,
dijalankan di laptop yang sama: Intel Core i7-8650U, 4 core dan 8 thread, .NET 10.0.12. Angka di bawah
adalah rerata delapan iterasi setelah tiga pemanasan; jalankan dengan
`dotnet run -c Release --project benchmarks/HFNet.Benchmarks` atau satu rangkaian dengan
`-- --filter *Tokenizer*`.

**GraviTransformers.** Satu perkalian `[128, 768] x [768, 3072]` - lapisan feed-forward pertama
bert-base pada 128 token - lalu model utuh, dalam kedua presisi:

| | double | linear float32 |
|---|---:|---:|
| GEMM HF.Net, perkalian itu | 9,2 ms | 6,5 ms |
| `MatMul` fondasi (CPU), perkalian itu | 68,7 ms | |
| ILGPU fondasi, perkalian itu | 155 ms | |
| Forward pass bert-base, 8 token | 44 ms | 33 ms |
| bert-base, 32 token | 108 ms | 79 ms |
| bert-base, 128 token | 434 ms | 326 ms |
| bert-base, 512 token | 2.271 ms | 1.891 ms |
| vit-base-patch16-224, satu gambar | 743 ms | 551 ms |
| clip-vit-base-patch32, satu gambar melawan lima label | 231 ms | 159 ms |
| gpt2, 32 token baru dengan KV cache | 1.480 ms | 1.258 ms |

Baris GPU adalah jalur presisi ganda milik fondasi pada GPU terintegrasi, yang menjalankan double
dengan sebagian kecil laju float32-nya - lebih lambat daripada CPU, sebagaimana ditemukan fondasi.
Pada 512 token float32 hanya menambah 17%, karena attention, yang tumbuh dengan kuadrat panjangnya
dan tetap double, adalah sebagian besar biayanya di sana.

**GraviTokenizers.** 1.000 dokumen dari korpus multibahasa:

| Tokenizer | satu thread | `EncodeBatch` |
|---|---:|---:|
| bert-base-multilingual-cased | 5,1 ms | 3,4 ms |
| gpt2 | 9,4 ms | 6,6 ms |
| xlm-roberta-base | 29,3 ms | 11,2 ms |

**GraviDatasets.** 200.000 baris buatan: pemuatan CSV penuh 356 ms, berkas yang sama dipetakan ke
memori 349 ms, Parquet 56 ms. Pemetaan memori menghemat memori, bukan waktu, pada berkas yang muat di
RAM; Parquet enam kali lebih cepat dibaca.

**GraviPEFT.** Satu epoch LoRA untuk 32 kalimat dalam batch berisi 8: 63 ms pada bert-tiny, 3,3 detik
pada bert-base.

**GraviAccelerate.** Gradien data-paralel atas 400.000 baris: 45 ms dengan satu worker, 26 dengan dua,
24 dengan empat, 16 dengan delapan - 2,8x, pada empat core fisik.

**GraviOptimum.** bert-base untuk satu kalimat: 63 ms managed, 37 ms melalui ONNX Runtime.

**GraviDiffusers.** Pipeline uji kecil, 10 langkah pada 64x64 dengan guidance: 455 ms. 25 langkah DDIM
pada laten Stable Diffusion 64x64, scheduler saja: 16 ms; 25 langkah Euler: 27 ms.

**GraviHub.** Membuka `model.safetensors` milik bert-base dan mendaftar 206 tensornya: 0,45 ms.
Mengunduh pickle bert-tiny 17,7 MB ke cache yang masih kosong: 2,7 detik.

## Kesimpulan praktis

| Bila Anda sedang | Pakai |
|---|---|
| Men-tokenisasi teks dalam volume besar | **HF.Net** — lebih cepat, keluaran identik |
| Memuat dan memeriksa checkpoint | **HF.Net** — seimbang pada header, dan membaca format yang ditolak jalur cepat Python |
| Menjalankan encoder di produksi | **ONNX melalui GraviOptimum** — lebih cepat daripada torch, dari .NET |
| Menjalankan encoder untuk memahaminya, atau memeriksa sebuah ekspor | **HF.Net managed** — dalam 1,2x torch untuk satu kalimat dan eksak hingga 1e-13 dalam float64; lebih cepat daripada torch dengan lapisan linear float32 |
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
