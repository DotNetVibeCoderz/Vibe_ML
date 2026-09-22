# GraviPEFT

**Adapter LoRA, dalam format Hugging Face PEFT.**

Padanan `peft`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviPEFT;
```

## Apa yang bisa dan tidak bisa dilakukannya

**Bisa:** memasang adapter, melipatnya ke dalam bobot secara persis, menyimpan dan memuat tata letak
Hugging Face PEFT, serta melatih task head di atas encoder yang dibekukan.

**Tidak bisa:** melakukan backpropagation ke matriks adapter itu sendiri.

```csharp
PeftModel.SupportsAdapterTraining   // false
```

Alasannya spesifik. Encoder autodiff yang tersedia di fondasi menghilangkan bias pada proyeksi Q/K/V,
sedangkan setiap BERT terlatih memilikinya — jadi gradien yang diambil melewatinya adalah gradien dari
*model yang sedikit berbeda*. Ia akan berlatih, konvergen, dan menghasilkan bobot yang diam-diam
salah. Hal itu dinyatakan di sini alih-alih dihampiri; [PLAN.md](../../PLAN.md) memuat jalan
perbaikannya.

**Jalur jujur hari ini:** latih adapter dengan PEFT di Python, layani di sini.

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

Untuk beberapa ribu contoh berlabel, inilah pilihan yang efektif sekaligus jujur.

## Menyimpan

```csharp
peft.SaveAdapter("./my-adapter");
```

Menulis `adapter_model.safetensors` dan `adapter_config.json` dalam tata letak PEFT, sehingga hasilnya
bisa dimuat di Python.

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
