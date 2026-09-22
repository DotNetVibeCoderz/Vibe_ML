# GraviDiffusers

**Difusi denoising: scheduler, dan teks-ke-gambar lewat ONNX.**

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
IScheduler scheduler = new DdimScheduler(eta: 0);   // deterministik
                     = new DdpmScheduler();          // rujukan
                     = new EulerScheduler();         // bagus di 20-30 langkah
```

**DDIM** merekonstruksi taksiran sampel bersih di setiap langkah lalu memberinya derau kembali ke
tingkat berikutnya. Itulah yang memungkinkannya melompati timestep: ia tetap konsisten saat
mengunjungi 25 dari 1000, sementara pembaruan DDPM mengandaikan langkah bersebelahan dan memburuk
tajam. Pada `eta = 0` ia sepenuhnya deterministik, sehingga seed dan prompt mereproduksi gambar yang
persis sama.

**DDPM** setia dan lambat, satu langkah per timestep pelatihan. Ia ada sebagai rujukan untuk
memeriksa sampler yang lebih cepat.

**Euler** memandang proses mundur sebagai ODE dalam tingkat derau dan mengambil langkah Euler biasa di
sepanjangnya. Cara pandang itulah yang membuatnya menghasilkan gambar layak dalam dua puluh atau tiga
puluh langkah — jumlah langkah menjadi pilihan ketelitian integrasi, bukan sifat jadwal yang dilatih.

### Bagaimana koefisiennya diverifikasi

Bangun `x_t` dari `x₀` yang diketahui dan derau yang diketahui, lalu biarkan "model" mengembalikan
derau itu persis di setiap langkah. Rekonstruksi DDIM jadi tepat di tiap langkah, sehingga lintasannya
harus mendarat kembali di `x₀` — dan itu terjadi sampai **1e-9** hanya jika `sqrt(abar)`,
`sqrt(1 - abar)` dan suku arahnya semuanya berada di tempat yang benar. Uji itu ada di
`tests/GraviDiffusers.Tests`.

Perlu dicatat bahwa prediksi derau *konstan* bukanlah kontraksi: DDIM justru memperbesar sebesar
`sqrt(abar₀ / abar_T)` di sepanjang jalan, kira-kira lima belas kali lipat pada jadwal Stable
Diffusion. Itu perilaku yang benar, bukan bug — model sungguhan memprediksi derau yang sebanding
dengan isi sampelnya.

## Teks ke gambar

```csharp
using var pipeline = DiffusionPipeline.FromPretrained(
    "some-user/stable-diffusion-onnx",
    scheduler: new EulerScheduler());

using var image = pipeline.Generate(
    "a watercolour of a mountain village at dawn",
    new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Width: 512, Height: 512, Seed: 42),
    onStep: (step, total) => Console.Write($"\r  {step}/{total}"));

image.Save("output.png");
```

### Yang dibutuhkannya

Repositori dengan tata letak sebagaimana ekspor ONNX:

```
text_encoder/model.onnx
unet/model.onnx
vae_decoder/model.onnx
```

Repositori yang hanya memuat bobot PyTorch **tidak akan bekerja** dan mengatakannya saat dimuat, bukan
gagal belakangan. Konversikan dengan `optimum-cli export onnx --model <id> <out>`.

Ketiga jaringan berjalan lewat ONNX Runtime karena satu kali difusi mengevaluasi UNet puluhan kali,
dan jalur `double` terkelola adalah alat yang keliru untuk itu, meleset dua orde besaran.

### Hal yang akan menjebak Anda

**Lebar dan tinggi harus kelipatan 8.** Kisi laten delapan kali lebih kecil daripada gambar di tiap
dimensi — dari sanalah semua aritmetika `/ 8` berasal.

**Guidance scale di atas 1 menjalankan UNet dua kali per langkah**, sekali terkondisi dan sekali
tidak. Nilai tepat 1 bukan sekadar setelan lemah: ia memangkas separuh kerja.

**Faktor penskalaan laten adalah 0,18215.** Ia diterapkan di jalan masuk dan dibagi kembali sebelum
VAE mendekode; lupa salah satu arah menghasilkan gambar pucat atau terlalu jenuh.

## Reproduksibilitas

```csharp
new GenerationOptions(Seed: 42)
```

Laten awal diambil dari `GraviRandom` ber-seed, dan DDIM pada `eta = 0` maupun Euler sama-sama
deterministik. Prompt dan seed yang sama menghasilkan gambar yang sama setiap kali — dan itulah dasar
membagikan sebuah *generasi*, bukan sekadar gambarnya.

## Lihat juga

[GraviOptimum](GraviOptimum.md) · [GraviTokenizers](GraviTokenizers.md)
