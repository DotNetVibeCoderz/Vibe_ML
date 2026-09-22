# GraviAccelerate

**Pemilihan perangkat, sharding, dan pengukuran yang bisa dipercaya.**

Padanan `accelerate`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviAccelerate;
```

## Perangkat keras apa ini?

```csharp
foreach (var device in Accelerator.Devices()) Console.WriteLine(device);
// Cpu: 8 logical cores, SIMD width 4
// Gpu: <nama perangkat> / unavailable (DllNotFoundException)

Console.WriteLine(Accelerator.Describe());
```

Menjajaki GPU akan menginisialisasi ILGPU, yang bisa melempar exception pada mesin dengan driver
rusak. Akselerator yang tidak ada adalah kondisi normal, bukan kesalahan yang layak diteruskan, jadi
penjajakannya tidak pernah melempar exception.

## GPU bukan pilihan bawaan

```csharp
var accelerator = new Accelerator(DeviceKind.Auto);      // Cpu | Gpu | Auto
accelerator.Backend(elementCount);
accelerator.Dot(a, b);
```

Ini keputusan **hasil pengukuran**, bukan kehati-hatian. Seluruh tumpukan bertipe `double`, dan GPU
konsumen maupun terintegrasi menjalankan aritmetika presisi ganda pada sebagian kecil dari laju
presisi tunggalnya — fondasi mengukur jalur ILGPU-nya **5 sampai 8 kali lebih lambat** daripada CPU
pada perangkat terintegrasi.

Karena itu `DeviceKind.Auto` hanya meraih GPU di atas `Accelerator.GpuThreshold` (2²⁰ elemen), di mana
transfer dan aritmetikanya sama-sama terbayar. Di bawah itu, peluncuran kernel dan perjalanan
bolak-balik melalui PCIe mendominasi, sehingga GPU kalah bahkan ketika ia benar-benar lebih cepat per
elemen.

Kalau yang Anda inginkan throughput presisi tunggal, itu tugas [GraviOptimum](GraviOptimum.md).

## Sharding

```csharp
var shards = accelerator.Partition(itemCount);

var averaged = accelerator.ParallelGradient(itemCount, shard =>
{
    var total = 0.0;
    foreach (var index in shard.Indices) total += Loss(index);
    return new NdArray([total / shard.Count], 1);
});
```

**Perataannya berbobot menurut ukuran shard.** Rata-rata polos dari rata-rata tiap worker sama dengan
rata-rata global hanya bila ukuran shard-nya sama, dan `Partition` menghasilkan shard tak-rata setiap
kali jumlah worker tidak membagi habis jumlah item. Tanpa pembobotan, model berlatih menuju sesuatu
yang sedikit salah dan tidak akan tertangkap oleh pemeriksaan bentuk maupun konvergensi.

Shard dikumpulkan dalam **urutan worker, bukan urutan selesai**. Penjumlahan floating-point tidak
asosiatif, jadi urutan kedatangan akan membuat hasilnya bergantung pada penjadwalan thread — dan
pelatihan yang tidak bisa direproduksi tidak bisa di-debug.

## Pelatihan

```csharp
var report = accelerator.Train(estimator, dataset, labelColumn: "survived", epochs: 5);

Console.WriteLine(report);
// 5 epochs, 4,450 examples in 0.82s (5,427/s) on Auto across 8 worker(s), score 0.8112
```

Ini menyiapkan matriks, menjalankan `Fit` milik estimator sekali per epoch, dan melaporkan biaya
jalannya. Ia sengaja **tidak** mem-shard proses fit: sharding hanya tersusun benar untuk estimator
yang parameternya bisa dirata-ratakan, dan merata-ratakan koefisien dua pohon keputusan menghasilkan
sesuatu yang bukan pohon. Ketika perataan memang sah, susun loop-nya dari `Partition` dan
`ParallelGradient`.

## Pengukuran

```csharp
var elapsed = Accelerator.Measure(() => model.Embed(text), iterations: 20, warmup: 40);
```

**Jumlah pemanasan bukan pengganjal.** Tiered JIT mengompilasi ulang metode panas setelah kira-kira
tiga puluh panggilan, jadi pengukuran dengan dua iterasi pemanasan justru mengukur keluaran
interpreter. Fondasi pernah mencatat kubus 192 elemen terukur delapan kali lebih lambat daripada kubus
224 elemen karena hal ini.

Waktu **terbaik** yang dikembalikan, bukan rata-rata, dengan alasan yang sama seperti harness
benchmark: pada laptop yang throttling, rata-rata mengukur suhu ruangan. Jangan pernah membandingkan
waktu dari dua jalan terpisah — jalankan variannya bergantian dalam satu proses.

## Melepas GPU

```csharp
Accelerator.ReleaseGpu();
```

## Lihat juga

[GraviOptimum](GraviOptimum.md) · [GraviDatasets](GraviDatasets.md)
