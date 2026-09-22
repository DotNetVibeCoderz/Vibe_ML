# GraviDatasets

**Memuat dan membentuk dataset, di atas GraviFrame.**

Padanan `datasets`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviDatasets;
```

## Memuat

```csharp
Dataset.Load("titanic");                    // bawaan, jalan tanpa jaringan
Dataset.Load("stanfordnlp/imdb", "train");  // apa pun yang mengandung garis miring adalah id Hub
Dataset.LoadAll("stanfordnlp/imdb");        // semua split

Dataset.FromFile("data.csv");               // .csv .tsv .parquet .json .jsonl
Dataset.FromCsvMemoryMapped("huge.csv");
Dataset.FromFrame(dataFrame, "name");
```

Nama polos dicari di registri bawaan **lebih dulu**, baru kemudian di Hub. Urutan itulah yang menjaga
`Dataset.Load("titanic")` tetap bekerja tanpa jaringan dan tanpa kredensial.

Bawaan: `titanic` `iris` `imdb` `finance` `sms_spam`.

## Membentuk

```csharp
data.Head(5);
data.Slice(start, count);
data.Select(indices);
data.SelectColumns("text", "label");
data.RemoveColumns("id");
data.RenameColumn("review", "text");
data.Shuffle(seed: 42);
data.Filter(row => row.Number("age") > 18);
data.Map("length", i => data.TextColumn("text")[i].Length);
```

**Setiap operasi mengembalikan dataset baru.** Dataset dioper antar-tahap prapemrosesan, dan `Shuffle`
yang mengubah di tempat akan diam-diam mengubah apa yang sudah diambil split sebelumnya — jenis
kebocoran yang menggelembungkan skor validasi tanpa memunculkan kesalahan yang terlihat.

Ketika sebuah operasi hanya memilih baris, frame di bawahnya dibagi pakai alih-alih disalin, sehingga
mengambil seratus split dari dataset besar tidak melipatgandakan memorinya.

## Memecah

```csharp
var split = data.TrainTestSplit(testSize: 0.2, seed: 42);

split.Train; split.Test; split.Validation;
split["train"]; split.Splits; split.Contains("validation"); split.TotalRows;
```

**Pengacakan aktif secara bawaan dan itu lebih penting daripada kelihatannya.** Banyak CSV yang
diterbitkan tersusun menurut label, dan split tanpa acak atas salah satunya menaruh seluruh contoh
positif di satu sisi.

Meminta split yang tidak ada melempar exception dengan menyebutkan nama-nama yang tersedia — dataset
yang diterbitkan dengan `train` dan `validation` tetapi tanpa `test` cukup umum sehingga pesannya
harus menyebut split mana yang ada, bukan hanya yang diminta.

## Membaca baris

```csharp
foreach (var row in data.Rows())                 // streaming, satu baris pada satu waktu
    Console.WriteLine(row.Number("age"));

foreach (var batch in data.Batches(32, dropLast: true))
    Train(batch);

data.TextColumn("label");     // bekerja pada kolom teks MAUPUN numerik
data.NumericColumn("age");
data.ToMatrix("age", "fare"); // NdArray [baris, kolom]
data.Describe();
```

`TextColumn` yang menerima kedua tipe itu disengaja: dataset yang diterbitkan tidak sepakat apakah
kolom label bertipe teks atau numerik, dan pemuat yang melempar exception pada tipe yang tidak
diharapkannya adalah pemuat yang hanya bekerja untuk separuh Hub.

## Dataset Hub

```csharp
HubDatasets.Load("stanfordnlp/imdb", "train");
HubDatasets.LoadAll("stanfordnlp/imdb");
```

Repositori dataset tidak punya satu tata letak baku. Kebanyakan kini menerbitkan Parquet di bawah
direktori per-konfigurasi, yang lebih lama menerbitkan CSV atau JSON Lines di akar, dan sebagian hanya
menerbitkan **skrip pemuat**, yang berbahasa Python dan tidak bisa dijalankan di sini.

Karena itu strateginya:

1. Tanyakan ke dataset server milik Hub berkas Parquet mana yang menopang tiap split. Hub mengonversi
   sebagian besar dataset publik secara otomatis, dan memakai indeks itulah yang membuat dataset
   berbasis skrip bisa dimuat sama sekali.
2. Jatuh ke pencocokan pola atas daftar berkas repositori itu sendiri, menebak split dari path-nya.

Dataset yang hanya berupa skrip dan tanpa konversi Parquet **ditolak dengan pesan yang menjelaskannya**,
bukan ditebak-tebak.

## JSON dan JSON Lines

```csharp
JsonLines.Read("data.jsonl");
```

Menangani array JSON maupun satu objek per baris, diputuskan dengan **memeriksa karakter bukan-spasi
pertama**, bukan dari ekstensinya: banyak repositori menerbitkan array JSON dalam berkas bernama
`.jsonl`, dan sebaliknya.

Urutan kolom mengikuti kemunculan pertama, sehingga pulang-pergi mempertahankan bentuk yang diharapkan
pembaca alih-alih mengurutkan menurut abjad. Field yang absen di sebagian baris menjadi nilai kosong,
bukan kesalahan.

## Menulis

```csharp
data.WriteCsv("out.csv");
data.WriteParquet("out.parquet");
```

## Pembacaan memory-mapped

```csharp
Dataset.FromCsvMemoryMapped("huge.csv");
```

Hanya sepadan untuk berkas yang besar relatif terhadap RAM. Di bawah itu, page fault dari pemetaan
lebih mahal daripada pembacaan langsung — dan itulah yang diukur
`benchmarks/GraviDatasets.Benchmark`, bukan diasumsikan.

## Lihat juga

[GraviHub](GraviHub.md) · [GraviTransformers](GraviTransformers.md) · [GraviAccelerate](GraviAccelerate.md)
