# GraviDiffusers

**Difusi denoising: scheduler, dan Stable Diffusion lewat ONNX — teks ke gambar, gambar ke gambar, inpainting, LoRA.**

Padanan `diffusers`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviDiffusers;
```

## Jadwal derau

Setiap sampler adalah cara berbeda menyusuri proses derau maju secara mundur, jadi jadwalnya harus
cocok dengan yang dipakai saat bobot dilatih — kalau tidak, sisanya tidak ada artinya.

```csharp
NoiseSchedule.StableDiffusion;   // 1000 langkah, 0.00085 -> 0.012, ScaledLinear
NoiseSchedule.Ddpm;              // 1000 langkah, 1e-4 -> 0.02, Linear

new NoiseSchedule(trainTimesteps: 1000, betaStart: 0.00085, betaEnd: 0.012,
                  BetaSchedule.ScaledLinear);
```

| Jadwal | Bentuk |
|---|---|
| `Linear` | Beta berjarak linear |
| `ScaledLinear` | Kuadrat dari tanjakan linear pada `sqrt(beta)` — inilah yang dipakai Stable Diffusion |
| `SquaredCosine` | Kosinus, menambahkan derau lebih lembut di kedua ujung |

**Memakai beta linear biasa dengan titik ujung Stable Diffusion** menghasilkan gambar yang bentuknya
masih terbaca tetapi konsisten pucat — mudah disalahartikan sebagai prompt yang buruk.

```csharp
schedule.Betas; schedule.Alphas; schedule.AlphasCumulative;
schedule.AlphaBar(t);    // 1.0 untuk t < 0, yang mendaratkan langkah terakhir pada sampel bersih
schedule.Sigma(t);
```

## Sampler

```csharp
// Dibangun dari scheduler_config.json milik repositori sendiri - cara yang lazim.
var config = SchedulerConfig.Load("scheduler/scheduler_config.json");
IScheduler scheduler = new DdimScheduler(config, eta: 0);   // deterministik
                     = new DdpmScheduler(config);          // rujukan
                     = new EulerScheduler(config);         // bagus di 20-30 langkah
```

`SchedulerConfig` memuat semua yang dibaca diffusers dari berkas itu: beta, `timestep_spacing`
(`leading`, `linspace`, `trailing`), `steps_offset`, `set_alpha_to_one`, `prediction_type`
(`epsilon`, `v_prediction`, `sample`) dan `clip_sample`. Masing-masing mengubah timestep mana yang
dikunjungi atau cara sebuah langkah diambil, jadi sampler yang dibangun tanpanya menyusuri jalur yang
berbeda dari yang dipakai saat model diterbitkan. Konstruktor lama `(NoiseSchedule)` tetap berfungsi,
untuk eksperimen.

**DDIM** merekonstruksi taksiran sampel bersih di setiap langkah lalu memberinya derau kembali ke
tingkat berikutnya. Itulah yang memungkinkannya melompati timestep: ia tetap konsisten saat
mengunjungi 25 dari 1000, sementara pembaruan DDPM mengandaikan langkah bersebelahan dan memburuk
tajam. Pada `eta = 0` ia sepenuhnya deterministik.

**DDPM** setia dan lambat, satu langkah per timestep pelatihan. Ia ada sebagai rujukan untuk
memeriksa sampler yang lebih cepat.

**Euler** memandang proses mundur sebagai ODE dalam tingkat derau dan mengambil langkah Euler biasa di
sepanjangnya, itulah sebabnya dua puluh atau tiga puluh langkah sudah cukup. Timestep-nya bisa jatuh di
antara langkah pelatihan; sigma diinterpolasi di sana, seperti yang dilakukan diffusers.

### Bagaimana koefisiennya diverifikasi

Dua cara. Bangun `x_t` dari `x₀` dan derau yang diketahui, lalu biarkan model mengembalikan derau itu
persis di setiap langkah: lintasan DDIM harus mendarat kembali di `x₀`, dan memang sampai **1e-9**. Lalu
timestep, sigma, dan keluaran setiap langkah dari tiap sampler dipatok ke nilai yang dicetak oleh
diffusers 0.40 sendiri, untuk setiap spacing dan jenis prediksi.

## Teks ke gambar

```csharp
using var pipeline = DiffusionPipeline.FromPretrained("nmkd/stable-diffusion-1.5-onnx-fp16");

var options = new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Width: 512, Height: 512, Seed: 42,
                                    NegativePrompt: "blurry");

using var lighthouse = pipeline.Generate(
    "a red lighthouse on a cliff at dawn, oil painting", options,
    onStep: (step, total) => Console.Write($"\r  {step}/{total}"));

lighthouse.Save("lighthouse.png");
```

![Mercusuar merah, dibuat oleh HF.Net](../screenshots/diffusion-lighthouse.png)

Gambar itu, 25 langkah DDIM di CPU laptop, sekitar enam menit. `OnnxStableDiffusionPipeline` milik
diffusers sendiri dengan seed yang sama memberi gambar yang sama, sampai PSNR 54,7 dB.

### Ini pipeline diffusers, langkah demi langkah

Pipeline ini mengikuti pipeline ONNX diffusers baris demi baris, jadi setelan yang sama memberi gambar
yang sama:

- **Derau awal** adalah `np.random.RandomState(seed).randn(...)`: `NumpyRandom` mereproduksi Mersenne
  Twister NumPy dan cara penarikan Gaussian-nya bit demi bit.
- **Prompt** di-tokenisasi seperti tokenizer CLIP melakukannya - huruf kecil, dipecah dengan regex,
  BPE tingkat byte dengan `</w>` - dan diberi padding dengan token pad milik model sendiri
  (`<|endoftext|>` pada SD 1.x, `!` pada 2.x).
- **Guidance** menjalankan UNet sekali pada batch `[tanpa syarat, bersyarat]`.
- **Geometrinya** diambil dari repositori: faktor skala VAE dari `block_out_channels`, penskalaan laten
  dari `scaling_factor`, kanal masukan UNet dari konfigurasinya. Model dengan VAE berbeda menghasilkan
  gambar seukuran yang diminta.
- **Piksel** dibulatkan, bukan dipotong - pemotongan menggelapkan setiap piksel setengah tingkat.

Pada ekspor float32 hasilnya identik dengan diffusers sampai piksel terakhir; ini diuji pada model
ekspor kecil, untuk DDIM, Euler, gambar ke gambar, dan inpainting. Pada ekspor float16 diffusers
menyimpan laten dalam float16 di antara langkah dan HF.Net dalam double, sehingga keduanya sepakat
sampai sekitar 55 dB.

### Repositori yang berfungsi

| Repositori | Ukuran | Catatan |
|---|---|---|
| `nmkd/stable-diffusion-1.5-onnx-fp16` | 2,1 GB | SD 1.5, float16. Dicocokkan dengan diffusers di atas. |
| `onnx-community/stable-diffusion-v1-5-ONNX` | 4,3 GB | SD 1.5, float32. |
| `amd/stable-diffusion-1.5_io16_amdgpu` | 2,1 GB | SD 1.5, float16. |
| `onnxruntime/sd-turbo` | 2,6 GB | **Hanya CUDA.** Dibangun dari kernel GPU gabungan ONNX Runtime (`NhwcConv`, `GroupNorm`); di CPU ditolak dengan menyebut namanya. |
| `optimum-internal-testing/tiny-stable-diffusion-onnx` | 9 MB | Bobot acak, untuk pengujian. |

Repositori membutuhkan `text_encoder/`, `unet/`, dan `vae_decoder/` masing-masing dengan `model.onnx`,
serta `vae_encoder/` untuk gambar ke gambar dan inpainting. Repositori yang hanya memuat bobot PyTorch
ditolak saat dimuat; konversikan dengan `optimum-cli export onnx --model <id> <out>`.
`DiffusionPipeline.FromDirectory` membuka ekspor yang ada di disk.

## Gambar ke gambar

```csharp
using var night = pipeline.ImageToImage("the same lighthouse at night, stars", lighthouse,
                                        strength: 0.6, options);
```

Gambar dienkode, diberi derau sampai sejauh `strength` di sepanjang jadwal, lalu dibersihkan dari
sana: 0 mengembalikannya apa adanya, 1 mengabaikannya.

**Encoder VAE sebagaimana diekspor tidak deterministik.** Ia mengambil sampel latennya di dalam graf
dengan `RandomNormalLike` tanpa seed, jadi setiap panggilan mengembalikan laten berbeda - di Python
juga. HF.Net menulis salinannya sekali di sebelahnya, `vae_encoder/model.mean.onnx`, dengan skala node
itu diatur ke nol, yang memberi rerata distribusinya. Seed kemudian mereproduksi hasil.

Pada SD 1.5 dengan strength 0,6 hasilnya sepakat dengan `OnnxStableDiffusionImg2ImgPipeline` milik
diffusers, yang dijalankan dengan encoder rerata dan seed yang sama, sampai PSNR 57,8 dB.

## Inpainting

```csharp
using var repainted = pipeline.Inpaint("a full moon", lighthouse, mask, options);   // putih = lukis ulang
```

Dengan **UNet inpainting** (sembilan kanal masukan) ini adalah pipeline inpainting diffusers: UNet
melihat laten, masker pada resolusi laten, dan laten gambar dengan area bermasker dihitamkan. Dengan
**UNet biasa** ia memadukan: setelah setiap langkah, area tanpa masker diganti gambar asli yang diberi
derau setingkat langkah itu - berfungsi di checkpoint mana pun dan lebih lembut di tepinya.

![Mercusuar dengan bulan dilukis ke langit bermasker](../screenshots/diffusion-inpaint.png)

Mercusuar di atas, dengan pojok kanan atas langit dimasker dan `"a full moon in the sky"`, pada UNet
SD 1.5 biasa.

## Penekanan

```csharp
pipeline.PromptWeighting = true;
pipeline.Generate("a (red:1.4) lighthouse at [dawn], ((oil painting))", options);
```

Sintaks dan aritmetika AUTOMATIC1111: `(kata)` mengalikan 1,1, `[kata]` membaginya, `(kata:1.4)`
mengatur bobot, kurung bisa bersarang dan `\(` meloloskan. Keluaran text encoder untuk setiap token
diskalakan dengan bobotnya, lalu keseluruhannya diskalakan kembali ke rerata aslinya. Fitur ini mati
secara bawaan, jadi prompt biasa selalu dienkode persis seperti diffusers melakukannya.
`PromptWeights.Parse` menunjukkan potongan-potongannya.

## LoRA

```csharp
var report = pipeline.LoadLora("some-user/some-sd15-lora", scale: 0.8);   // berkas, folder, atau repo
Console.WriteLine(report);      // LoRA: 128 UNet and 72 text-encoder layers
pipeline.UnloadLoras();
```

Tata letak kohya (`lora_unet_..._to_q.lora_down.weight`) maupun diffusers atau PEFT
(`unet....to_q.lora_A.weight`) sama-sama dibaca. Bobot ekspor ONNX tidak bernama
(`onnx::MatMul_2567`), tetapi nama node-nya menyimpan jalur modul PyTorch, jadi setiap adapter
dicocokkan lewat node yang memakai bobot itu. Pembaruannya ditambahkan ke bobot tersimpan dan diberikan
ke ONNX Runtime sebagai initializer pengganti; berkas model tidak pernah ditulis ulang. LoRA yang
modulnya tidak cocok dengan apa pun ditolak - ia dilatih untuk model lain.

Diuji terhadap LoRA yang digabungkan ke UNet dan text encoder di PyTorch lalu diekspor: keduanya
sepakat kecuali tiga dari 12.288 nilai yang berbeda satu tingkat.

## Hal yang akan menjebak Anda

**Guidance scale di atas 1 menjalankan UNet pada batch berisi dua**, bersyarat dan tidak. Nilai tepat
1 bukan sekadar setelan lemah: ia memangkas separuh kerja.

**Lebar dan tinggi harus kelipatan faktor skala VAE** - 8 untuk Stable Diffusion. Pipeline ONNX
diffusers menganggap 8 apa pun kata VAE-nya; pada model yang VAE-nya memperkecil lebih sedikit, ia
mengembalikan gambar yang lebih kecil daripada yang diminta.

**Siapkan sekitar 8 GB memori untuk SD 1.5 di CPU**, bahkan dari ekspor float16 2 GB: ONNX Runtime hanya
punya sedikit kernel float16 di CPU, jadi ia menyisipkan konversi dan menyimpan salinan float32 dari bobot
yang dibutuhkannya.

**Ekspor float16 memakai float16 sampai ke dasar**, termasuk timestep. HF.Net mengonversi di
perbatasan; tidak ada yang perlu diatur.

## Lihat juga

[GraviOptimum](GraviOptimum.md) · [GraviTokenizers](GraviTokenizers.md) ·
[notebook 05](../../notebooks/05-stable-diffusion.ipynb)
