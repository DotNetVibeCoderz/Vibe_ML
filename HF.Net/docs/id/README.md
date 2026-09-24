# Dokumentasi HF.Net

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

**English:** [docs/](../)

## Mulai dari sini

- [Memulai](memulai.md) — pasang, model pertama, prediksi pertama
- [HF Gallery](hf-gallery.md) — sembilan use case berjalan terhadap model sungguhan, dalam satu jendela
- [HFAppGen](HFAppGen.md) — IDE yang menuliskan aplikasi HF.Net untuk Anda
- [Benchmark](benchmarks.md) — HF.Net diukur terhadap rujukan Python
- Notebook: [01 memulai](../../notebooks/01-hugging-face-from-dotnet.ipynb) · [02 kinerja](../../notebooks/02-where-the-time-goes.ipynb)

## Pustaka

| Halaman | Pustaka | Padanan Python |
|---|---|---|
| [GraviHub](GraviHub.md) | Klien Hub, safetensors, checkpoint PyTorch | `huggingface_hub` |
| [GraviTokenizers](GraviTokenizers.md) | WordPiece, BPE, Unigram, `tokenizer.json` | `tokenizers` |
| [GraviDatasets](GraviDatasets.md) | Berkas, dataset Hub, split, streaming | `datasets` |
| [GraviTransformers](GraviTransformers.md) | Encoder terlatih dan task head | `transformers` |
| [GraviPEFT](GraviPEFT.md) | Adapter LoRA | `peft` |
| [GraviAccelerate](GraviAccelerate.md) | Perangkat, sharding, pengukuran | `accelerate` |
| [GraviOptimum](GraviOptimum.md) | ONNX Runtime, kuantisasi | `optimum` |
| [GraviDiffusers](GraviDiffusers.md) | Scheduler, Stable Diffusion | `diffusers` |

## Konvensi yang berlaku di seluruh pustaka

**Semuanya bertipe `double`.** `NdArray` menyimpan `double`, jadi checkpoint yang tersimpan sebagai
F16 atau F32 dilebarkan saat dibaca. Ini memakan memori tetapi menjaga keseragaman dengan seluruh
tumpukan Gravicode; ketika throughput yang dibutuhkan, jawabannya adalah
[GraviOptimum](GraviOptimum.md), bukan tipe array yang berbeda.

**Penolakan dinyatakan terang-terangan.** Ketika HF.Net tidak bisa melakukan sesuatu, ia mengatakannya
dan menyebutkan alasannya — arsitektur yang tidak didukung, format berkas yang tidak dikenal, operasi
yang memerlukan gradien yang tidak bisa dihitung. Ia tidak memuat model setengah jalan lalu membiarkan
Anda menemukan masalahnya di keluaran.

**Offset itu nyata.** Offset tokenizer menunjuk ke string asli yang tidak diubah, sehingga sebuah span
selalu bisa dikembalikan sebagai substring, bukan sebagai indeks token.
