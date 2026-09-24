# Benchmark: HF.Net versus implementasi rujukan Python

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

Tujuan halaman ini bukan untuk menang, melainkan untuk tahu posisi HF.Net terhadap implementasi yang
dipakai semua orang — supaya dokumentasi bisa memberi tahu Anda alat mana yang harus dipakai, dan
tidak keliru.

Kedua sisi mengukur **masukan yang sama, di mesin yang sama, dalam sesi yang sama**, dan keduanya
melaporkan waktu **terbaik** dari beberapa kali jalan setelah pemanasan. Pada laptop yang mengalami
throttling, rata-rata justru mengukur suhu ruangan.

| | |
|---|---|
| Mesin | Windows 11 (10.0.26200), 8 core logis |
| Python | 3.12.10 — transformers 5.17.0, tokenizers 0.23.2, torch 2.14.0+cpu (4 thread) |
| .NET | 10.0.12, Release |

Keluaran mentahnya ada di [`benchmarks/comparison/report.md`](../../benchmarks/comparison/report.md).

---

## Tokenisasi — HF.Net lebih cepat

`bert-base-uncased`. Sisi Python memakai crate Rust `tokenizers` di balik binding tipis, yang
merupakan hal tercepat yang dimiliki Python; GraviTokenizers adalah C# terkelola.

| Ukuran | Python (Rust) | HF.Net (C#) | |
|---|---:|---:|---|
| Satu dokumen | 0,03 ms | 0,01 ms | **2,8x lebih cepat** |
| 1.000 dokumen | 18,13 ms | 10,94 ms | **1,66x lebih cepat** |

**55.144 dok/dtk** melawan **91.423 dok/dtk**.

Hasil ini tidak sekejutan kelihatannya. Keduanya melakukan penelusuran pencocokan terpanjang serakah
yang sama; crate Rust membayar satu penyeberangan batas Python per panggilan, sedangkan versi
terkelola meng-cache segmentasi per kata — teks nyata banyak mengulang kata, dan loop merge itulah
bagian yang mahal.

### Dan id-nya identik

Kecepatan hanya berarti bila keluarannya sama. Dan memang sama:

```
input   Hello, world! Tokenizers are unbelievable.
python  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
hf.net  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
```

## Membaca safetensors — seimbang

`model.safetensors` untuk `bert-base-uncased`: 420 MB, 206 tensor.

| Ukuran | Python | HF.Net | |
|---|---:|---:|---|
| Membuka dan mendaftar semua tensor | 0,65 ms | 0,71 ms | 1,09x lebih lambat |
| Membaca satu tensor 30.522 × 768 | 1,14 ms | 196,96 ms | 173x lebih lambat |

Pendaftaran seimbang, karena keduanya hanya membaca header.

**Membaca tensor tidak seimbang, dan alasannya struktural, bukan sesuatu yang bisa ditambal.**
PyTorch mengembalikan view tanpa-salin atas byte F32 sebagaimana tersimpan di disk. HF.Net
melebarkan setiap nilai ke `double`, karena itulah yang disimpan `NdArray` — 23,4 juta nilai, 187 MB
termaterialisasi. Biayanya nyata, dan itu harga keseragaman dengan seluruh tumpukan Gravicode.

> Tabel ini juga tempat ditemukannya **bug kinerja 600x**. Pembacanya mengalokasikan dan menyalin
> prefix tetap 256 MB hanya untuk mengurai header sekitar dua puluh kilobyte, sehingga mendaftar
> tensor memakan 429 ms. Membaca panjang header yang dideklarasikan lebih dulu lalu mengambil persis
> sebanyak itu menurunkannya ke 0,71 ms. Tidak ada yang bisa memunculkannya selain perbandingan
> dengan implementasi rujukan — sendirian, angkanya tampak cukup cepat.

## Inferensi encoder — Python menang, jauh

Satu forward pass atas 12 token.

| Model | Python (torch) | HF.Net (terkelola) | |
|---|---:|---:|---|
| bert-base-uncased, 1 dokumen | 50,8 ms | 300–600 ms | **6–12x lebih lambat** |
| bert-base-uncased, 8 dokumen | 261,6 ms | 1.736 ms | 6,6x lebih lambat |
| bert-tiny, 1 dokumen | 1,5 ms | 2,1 ms | 1,4x lebih lambat |

Angka satu-dokumen bert-base bergerak antara 304 ms dan 598 ms pada dua kali jalan berselang satu jam
dengan build yang sama — throttling termal, sekaligus contoh bagus mengapa satu angka yang dikutip
tanpa kondisinya nyaris tak berarti. Karena itu yang diberikan rentang, bukan presisi palsu.

**Selisih ini wajar dan bukan cacat.** Encoder terkelola bertipe `double` sepenuhnya dan ditulis
untuk dibaca; torch menyalurkan kerja ke kernel presisi tunggal yang disetel tangan dengan attention
terfusi. GraviTransformers ada supaya sebuah model bisa **dimuat, diperiksa dan dipahami** dalam .NET
murni.

Perhatikan bahwa selisihnya menyempit tajam pada model kecil — 1,4x pada bert-tiny melawan 6–12x pada
bert-base. Sebagian besar kemenangan torch ada di perkalian matriks besar, bukan di framework-nya.

### Vision

Satu gambar pada 224x224, yang berarti 197 posisi, bukan 12.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 gambar | 441 ms | 12,0 d | 26x lebih lambat |

Dua pertiga angka managed itu adalah attention, yang merupakan milik fondasi. Pasangan feed-forward
dan proyeksi patch adalah milik HF.Net, dan keduanya dipercepat dengan mengubah tata letak memorinya,
bukan aritmetikanya: bobotnya tetap dalam urutan `(outputs, inputs)` milik checkpoint sehingga tiap
dot product menyusuri memori yang bersebelahan dan bisa divektorkan.

Arah itu cukup berlawanan dengan dugaan sehingga layak dinyatakan terang-terangan. Menransposnya ke
urutan `(inputs, outputs)` yang "diinginkan" perulangan biasa, lalu memparalelkannya per baris,
terukur **lima kali lebih lambat daripada versi berurutan yang hendak digantikannya** — pada 3072
kolom setiap langkah perulangan dalam adalah satu cache line baru, dan operand berlangkah tidak bisa
dimuat ke register vektor sama sekali.

### Jalur produksi

Pekerjaan yang sama lewat ONNX Runtime, via GraviOptimum, pada sebuah model uji kecil:

**0,67 ms** terbaik, 0,85 ms median.

Tidak sebanding dengan tabel di atas — modelnya berbeda — tetapi ini menunjukkan bentuk jawabannya:
kelas kernel presisi tunggal yang sama dengan yang dipakai torch, dijangkau dari .NET. **Ketika Anda
butuh throughput, ekspor ke ONNX.** Lihat [GraviOptimum](GraviOptimum.md).

## Apakah keduanya sepakat? — ya

Kecepatan itu bagian yang mudah. Bagian inilah yang menentukan apakah semuanya berguna.

**`The capital of France is [MASK].`**

| Peringkat | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41,68% | paris | 41,53% |
| 2 | lille | 7,14% | lille | 7,16% |
| 3 | lyon | 6,34% | lyon | 6,31% |
| 4 | marseille | 4,44% | marseille | 4,46% |
| 5 | tours | 3,03% | tours | 3,02% |

**`He was a [MASK] player in the national team.`**

| Peringkat | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54,89% | regular | 55,00% |
| 2 | key | 19,47% | key | 19,39% |
| 3 | former | 5,82% | former | 5,83% |
| 4 | capped | 2,16% | capped | 2,14% |
| 5 | prominent | 1,36% | prominent | 1,36% |

**`google/vit-base-patch16-224`**, pada foto contoh milik Hub sendiri dan pada gambar dua kucing yang
kanonik. Sisa selisih kecilnya berasal dari resampler — bilinear milik PIL melawan milik ImageSharp —
bukan dari modelnya.

| Gambar | Python | | HF.Net | |
|---|---|---:|---|---:|
| bee.jpg | bee | 94,38% | bee | 94,46% |
| | pot, flowerpot | 1,36% | pot, flowerpot | 1,32% |
| cats.jpg | Egyptian cat | 93,74% | Egyptian cat | 93,81% |
| | tabby, tabby cat | 3,84% | tabby, tabby cat | 3,80% |

Urutan sama, lima kandidat sama, probabilitas sepakat sampai sekitar sepersepuluh poin persen.
Selisih sisanya adalah aritmetika `double` melawan `float` — yaitu HF.Net yang *lebih* presisi, bukan
kurang.

## Dua hal yang tidak bisa dibuka Python

Keduanya muncul saat menulis benchmark ini, dan keduanya ditangani HF.Net tanpa diminta:

- **`prajjwal1/bert-tiny` tidak punya `tokenizer.json`.** `transformers` 5.x menolak membangun
  tokenizer cepat dari `vocab.txt` tanpa `sentencepiece` terpasang. HF.Net jatuh ke `vocab.txt`
  dengan sendirinya.
- **`config.json` milik `prajjwal1/bert-tiny` tidak punya kunci `model_type`.** `AutoModel`
  menolaknya mentah-mentah; arsitekturnya harus disebut eksplisit. HF.Net memberi nilai bawaan
  `bert` dan memuatnya.

Keduanya bukan hasil kinerja. Keduanya adalah hal yang menentukan apakah sebuah pustaka bisa membuka
model yang benar-benar Anda punya.

## Kesimpulan praktis

| Kalau Anda | Pakai |
|---|---|
| Menokenisasi teks dalam volume besar | **HF.Net** — lebih cepat, keluaran identik |
| Memuat dan memeriksa checkpoint | **HF.Net** — seimbang pada header, dan membaca format yang ditolak jalur cepat Python |
| Menjalankan encoder di produksi | **ONNX lewat GraviOptimum**, bukan jalur terkelola |
| Menjalankan encoder untuk memahaminya | **HF.Net terkelola** — lebih lambat, dan bisa dibaca |
| Melatih | **Python.** HF.Net belum melakukan backpropagation ke encoder terlatih; lihat [PLAN.md](../../PLAN.md) |

## Mereproduksi ini

```bash
cd benchmarks/comparison
pip install tokenizers transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu

python python/bench.py --out python/python.json
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Jalankan kedua sisi dalam satu sesi. Jangan bandingkan angka dari hari yang berbeda — pada laptop
yang throttling, selisih itu melampaui hampir semua yang sedang diukur.
