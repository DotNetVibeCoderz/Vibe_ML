# GraviPEFT

**Adapter LoRA, dalam format Hugging Face PEFT.**

Padanan `peft`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviPEFT;
```

## Apa yang dilakukannya

Ia **melatih** adapter LoRA bersama sebuah classification head, dengan backpropagation melewati
encoder terlatih yang bobotnya sendiri tetap dibekukan. Ia **memasang** adapter, termasuk yang
dilatih dengan PEFT di Python, dan **melipatnya** ke dalam bobot secara persis. Ia **menyimpan dan
memuat** tata letak Hugging Face PEFT, dan adapter yang dilatih di sini bisa dimuat di Python. Ia juga
bisa melatih head saja di atas encoder yang dibekukan, baseline murah yang layak dicoba lebih dulu.

```csharp
PeftModel.SupportsAdapterTraining   // true, sejak 0.3
```

## Memasang adapter

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var (adapter, encoder, fraction) = peft.ParameterEfficiency();
Console.WriteLine($"{adapter:N0} dari {encoder:N0} = {fraction:P3}");
```

Model yang dikembalikan berperilaku **identik** dengan masukannya, karena matriks `B` setiap adapter
bernilai nol. Itulah sifat penentu LoRA, bukan detail implementasi: pelatihan bermula dari perilaku
terlatih, bukan dari gangguan atasnya. Menginisialisasi kedua matriks secara acak tetap berlatih,
tetap konvergen, dan berakhir di tempat yang terukur lebih buruk.

`A` diinisialisasi Kaiming-uniform dan bukan nol, kalau tidak gradien tidak akan pernah mengalir.

## Konfigurasi

```csharp
new LoraConfig(Rank: 8, Alpha: 16, TargetModules: ["query", "value"], Dropout: 0);
```

`Scaling` adalah `Alpha / Rank`. Normalisasi itulah yang menjadikan rank sebagai tombol **kapasitas**,
bukan tombol laju belajar: tanpanya, menggandakan rank menggandakan besar pembaruan saat inisialisasi.

Hanya mengadaptasi proyeksi query dan value adalah rekomendasi makalah aslinya dan menjadi bawaan di
sini. Menambahkan key dan output kira-kira menggandakan parameter yang dilatih demi perubahan yang
biasanya masih di dalam derau.

## Memuat adapter terbitan

```csharp
var peft = PEFT.LoadAdapter(model, "some-user/some-lora", merge: true);
```

Membaca `adapter_model.safetensors` (atau `.bin`) berdampingan dengan `adapter_config.json`, persis
dalam tata letak yang ditulis PEFT:

```
base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight
base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight
```

Adapter yang hanya punya separuh pasangan **ditolak**: menerapkannya akan menambahkan pembaruan
berperingkat nol yang tampak seperti adapter yang bekerja padahal tidak melakukan apa-apa.

## Melipat (merge)

```csharp
peft.Merge();
```

Melipat setiap adapter ke dalam bobot dasar, di tempat dan secara persis. Setelah dilipat, inferensi
memakan biaya persis sama dengan model dasar — tidak ada adapter tersisa untuk dievaluasi.

Ini tepat dilakukan sebelum melayani dan keliru dilakukan sebelum menukar adapter, karena lipatan itu
**tidak bisa dibatalkan** hanya dari bobot hasilnya.

Path adapter dicocokkan berdasarkan indeks lapisan ditambah akhiran, bukan path penuh, karena adapter
yang sama diterbitkan dengan beberapa awalan. Bila tidak ada yang cocok, `Merge` melempar exception
alih-alih diam-diam tidak menerapkan apa pun.

## Melatih head

```csharp
peft.FitHead(texts, labels);

peft.Predict("this is excellent");      // -> Prediction[]
peft.Score(heldOutTexts, heldOutLabels);
peft.HeadLabels;
```

Melatih regresi logistik pada embedding beku dari encoder yang sudah diadaptasi. Ini murah karena
encoder dievaluasi **sekali per contoh** lalu dipakai ulang: forward pass transformer mendominasi
biaya, dan melatih head kemudian hanyalah regresi atas beberapa ratus dimensi.

Inilah baseline yang layak dijalankan sebelum `Train`. Bila fitur beku sudah memisahkan kelas-kelasnya,
hanya ini yang Anda perlukan. Bila belum, seperti pada negasi, di mana "good" dan "not good" berbagi
hampir semua token, adapter punya sesuatu untuk ditambahkan.

## Melatih adapter

```csharp
using var model = TransformerModel.Load("bert-base-uncased");
var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var report = peft.Train(texts, labels, new TrainingOptions
{
    Epochs = 8,
    BatchSize = 8,
    LearningRate = 1e-3,
    Progress = new Progress<TrainingProgress>(p => Console.WriteLine($"{p.Step}/{p.TotalSteps} {p.Loss:F4}")),
});

Console.WriteLine(report);                 // 32 steps in about a minute, loss 0.70 -> ... -> 0.003
peft.Predict("The staff were not friendly at all.");
```

Setiap langkah menjalankan tiap contoh melewati encoder dengan adapter di dalam jalurnya, memasang
head linear di atas hidden state terakhir yang di-mean-pool, lalu melakukan backpropagation atas
loss cross-entropy ke `A` dan `B` setiap adapter serta ke head. Setelah itu ia mengambil satu langkah
AdamW. Bobot dasar tidak pernah berubah. Adapter diperbarui di tempat, sehingga `SaveAdapter` dan
`Merge` sesudahnya melihat nilai hasil pelatihan.

| Opsi | Bawaan | |
|---|---|---|
| `Epochs` | 3 | |
| `BatchSize` | 8 | contoh yang rata-rata loss-nya membentuk satu micro-batch |
| `GradientAccumulation` | 1 | micro-batch per langkah optimizer |
| `LearningRate` | 5e-4 | puncak; LoRA butuh kira-kira sepuluh kali laju fine-tuning penuh |
| `WeightDecay` | 0 | terpisah, gaya AdamW; bias dikecualikan |
| `WarmupFraction` | 0 | warm-up linear, lalu turun linear sampai nol |
| `MaxGradientNorm` | 1.0 | clipping norma global; 0 mematikannya |
| `MaxLength` | 128 | pemotongan, dalam token |
| `Seed` | 42 | pengacakan, inisialisasi head, dropout adapter |

Optimizer, jadwal dan clipping-nya adalah milik `Trainer` Hugging Face, ditulis ulang dari
`torch.optim.AdamW`, `get_linear_schedule_with_warmup` dan `clip_grad_norm_`, dan pengujiannya
mengunci ketiganya pada rumus-rumus itu. Satu akibatnya kerap mengejutkan: bila ada warm-up
sedikit pun, langkah pertama diambil dengan laju belajar tepat nol, sama seperti di Python.
`LoraConfig.Dropout` hanya berlaku saat pelatihan.

### Mengapa ia punya backward pass sendiri

Fondasi sudah memiliki encoder autodiff, dan encoder itu tidak dipakai karena satu alasan spesifik:
ia tidak punya bias pada proyeksi query, key dan value, padahal setiap BERT terlatih memilikinya.
Gradien yang diambil melewatinya adalah gradien dari *model yang sedikit berbeda*. Ia akan berlatih,
konvergen, dan menghasilkan adapter yang diam-diam salah untuk model tempat adapter itu dipasang.

Backward pass di sini ditulis tangan di atas kernel yang sama dengan yang dipakai inferensi. Karena
bobot dasar dibekukan, satu-satunya gradien yang dibutuhkan adalah milik adapter. Selebihnya
dipropagasikan lalu dibuang, dan tidak ada yang dipropagasikan di bawah lapisan teradaptasi yang
paling rendah.

### Bagaimana ia diperiksa

- **Gradien.** Setiap entri setiap adapter, pada keenam proyeksi sebuah model dua lapis yang setiap
  bias dan norm-nya digeser dari nilai awalnya, dicocokkan dengan beda-hingga pusat. Selisihnya
  di bawah 1e-6 untuk GELU eksak maupun GELU tanh, dan tetap begitu saat dropout aktif. Backward
  pass yang sengaja dibuat salah (skala atensinya dihilangkan) gagal dengan selisih dua kali lipat.
- **Forward.** Tanpa adapter, forward pass pelatihan sama dengan encoder inferensi hingga 1e-12. Pada
  `bert-base-uncased` sesudah pelatihan, prediksi dari adapter di dalam jalur dan dari bobot yang
  sudah dilipat sama hingga 6e-12.
- **Pulang-pergi.** Adapter yang dilatih dan disimpan di sini, lalu dimuat ke PEFT 0.21 di Python,
  memberi hidden state yang sama dengan model HF.Net yang sudah dilipat hingga 7e-7. Itulah
  pembulatan float32 dari bobot yang dilipat. Adapternya sendiri menggeser hidden state itu sampai 4.

### Biayanya, dan batasnya

- **CPU, satu sekuens sekali jalan, tanpa padding.** Batch adalah satuan perataan, bukan
  vektorisasi. Satu langkah memakan sekitar tiga forward pass per contoh. Di sini, bert-base atas
  32 kalimat pendek selama 8 epoch memakan 58 sampai 79 detik antar-run.
- **Memori.** Pelatihan menyimpan salinan float32 dari bobot encoder di samping model rujukan,
  sekitar 340 MB untuk bert-base.
- **Hanya klasifikasi sekuens**, lewat head yang di-mean-pool. Head klasifikasi token dan tanya-jawab
  tidak dilatih di sini.
- **Head tidak disimpan.** `SaveAdapter` menulis adapternya dalam tata letak PEFT. Head yang dilatih
  `Train` tetap tinggal di memori.
- **`Train` sesudah `Merge` ditolak.** Pembaruannya akan terhitung dua kali. Untuk melanjutkan
  pelatihan adapter yang sudah dilipat, muat ulang model dasar lalu pasang adapter tanpa dilipat
  dengan `PeftModel.WithAdapters(model, adapters, merge: false)`.

Untuk lebih dari beberapa ribu contoh, latih dengan PEFT di Python lalu muat hasilnya di sini. Jalur
itu juga persis.

## Menyimpan

```csharp
peft.SaveAdapter("./my-adapter");
```

Menulis `adapter_model.safetensors` dan `adapter_config.json` dalam tata letak PEFT, sehingga hasilnya
bisa dimuat di Python.

Adapter buatan `ApplyLoRA` diberi kunci berupa **path modul checkpoint itu sendiri**, misalnya
`bert.encoder.layer.0.attention.self.query`, karena PEFT mencocokkan tensor ke modul berdasarkan
nama dan melewatkan, tanpa galat, tensor yang tidak bisa ditempatkannya. Muat adapter ke kelas model
yang memakai nama yang sama. Untuk `bert-base-uncased`, yang nama-nama checkpoint-nya diawali
`bert.`, artinya `AutoModelForSequenceClassification` atau `BertForMaskedLM`, bukan `BertModel`
polos. Sebelum 0.3 kuncinya `layer.0.query`, yang oleh PEFT di Python dimuat sebagai tidak ada adapter
sama sekali.

## Mengapa jumlah parameter itu penting

```
bobot 768 x 768        589.824 nilai
adapter rank 8          12.288 nilai     ~2 %
```

Adapter rank-8 atas satu proyeksi kira-kira dua persen dari bobot yang diadaptasinya. Itulah seluruh
argumennya: adapter khusus-tugas berukuran beberapa megabyte, dikirim bersama satu model dasar
bersama, dan bisa ditukar tanpa memuat ulang model itu.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviHub](GraviHub.md)
