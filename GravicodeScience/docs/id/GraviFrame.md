# GraviFrame

*[English](../GraviFrame.md)* · pandas untuk .NET — kolom bertipe, group-by, join, dan deret waktu.

## Model

Sebuah `DataFrame` adalah daftar terurut berisi `Series`, masing-masing didukung satu array
bertipe. Tata letak kolumnar itulah yang membuat statistik dan filter kolom cepat: masing-masing
menyentuh buffer kontigu alih-alih melangkahi objek baris. Operasi per baris (filter, sort, join)
membangun vektor indeks lalu meminta setiap kolom melakukan `Take` — satu lintasan per kolom, bukan
per sel.

Frame bersifat **immutable**. Setiap transformasi mengembalikan frame baru; array kolom di
bawahnya dibagi pakai bila memungkinkan.

## Tipe kolom

| Tipe | Penyimpanan | Nilai kosong |
|---|---|---|
| `NumericSeries` | `double[]` | `double.NaN` |
| `TextSeries` | `string?[]` | `null` |
| `BooleanSeries` | `bool?[]` | `null` |
| `DateTimeSeries` | `DateTime?[]` | `null` |

Memakai NaN sebagai penanda kosong untuk kolom numerik itulah yang memungkinkan aritmetika kolom
langsung memakai kernel SIMD GraviNum — tidak ada mask null paralel yang perlu diperiksa di dalam
loop, dan aturan IEEE merambatkan nilai kosong secara gratis.

## Membaca data

```csharp
var df = DataFrame.ReadCsv("data.csv");
var df2 = DataFrame.ParseCsv(csvString);
var big = DataFrame.ReadCsvMemoryMapped("huge.csv");   // saat berkas mungkin tak muat di RAM

ParquetIO.Write(df, "data.parquet");
var back = ParquetIO.Read("data.parquet");
var subset = ParquetIO.Read("data.parquet", ["value", "date"]);   // kolom tertentu saja
```

Tipe disimpulkan per kolom dari 1.000 baris pertama secara bawaan. Timpa bila perlu:

```csharp
var df = DataFrame.ReadCsv("data.csv", new CsvOptions
{
    Delimiter = ";",
    HasHeader = true,
    MissingTokens = { "N/A", "-", "?" },
    ColumnTypes = { ["zip"] = DataType.Text },   // cegah ID berbentuk angka jadi numerik
    MaxRows = 100_000,
});
```

Field berkutip, pemisah di dalam kutip, dan kutip ganda sebagai escape sudah ditangani.

## Memeriksa

```csharp
df.Shape;              // (baris, kolom)
df.ColumnNames;
Console.WriteLine(df);            // tabel lebar tetap, 10 baris pertama
Console.WriteLine(df.ToString(50));
Console.WriteLine(df.Info());     // tipe dan jumlah nilai kosong per kolom
Console.WriteLine(df.Describe()); // count, mean, std, kuartil per kolom numerik
```

## Memilih dan memfilter

```csharp
df["age"];                        // Series
df.Numeric("age");                // NumericSeries — melempar pesan jelas bila tipenya salah
df.Text("name");  df.DateTimes("date");  df.Booleans("active");

df.Head(10);  df.Tail(10);  df.Rows(100, 50);  df.Sample(20, seed: 42);
df.SelectColumns("age", "fare");
df.Drop("deck", "embarked");
df.Rename("fare", "ticket_price");

df.FilterBy("age", v => v > 30);
df.Filter(row => row.String("sex") == "female" && row.Number("age") > 30);
df.Filter(df.Numeric("fare").Where(v => v > 100));   // mask boolean

df.SortBy("fare", ascending: false);
df.SortBy([("pclass", true), ("fare", false)]);      // leksikografis
```

## Menambah kolom

```csharp
df.WithColumn(new NumericSeries("ratio", values));
df.WithColumn("price_per_person", i => df.Numeric("fare")[i] / df.Numeric("size")[i]);
df.WithColumn(df.Numeric("age").Standardize().Rename("age_z"));
```

## Nilai kosong

```csharp
df.MissingCounts();                   // per kolom
df.DropMissing();                     // ada nilai kosong di baris tersebut
df.DropMissing("all");                // hanya baris yang seluruhnya kosong
df.DropMissing(subset: ["age"]);
df.FillMissing(0.0);
df.FillMissingWithMean();

var age = df.Numeric("age");
age.FillMissingWithMedian();
age.ForwardFill();  age.BackwardFill();
age.Interpolate();                    // linear di antara nilai terisi
```

## GroupBy

Pengelompokan dilakukan sekali di awal, sehingga `Mean`, `Sum`, dan `Count` pada `GroupBy` yang
sama masing-masing hanya satu lintasan atas nilai, bukan mengelompokkan ulang frame. Urutan grup
mengikuti kemunculan pertama.

```csharp
df.GroupBy("category").Mean("sales");
df.GroupBy("category", "region").Sum("sales");    // kunci gabungan
df.GroupBy("category").Count("n");

df.GroupBy("category").Aggregate("median", "sales");
df.GroupBy("category").Aggregate("sales", v => v.Max() - v.Min(), "range");
df.GroupBy("category").AggregateMany([("sales", "sum"), ("sales", "mean"), ("units", "max")]);
```

Agregat bernama: `mean`, `sum`, `min`, `max`, `median`, `std`, `var`, `count`, `first`, `last`.

## Perubahan bentuk

```csharp
df.Pivot("date", "category", "sales");                  // panjang -> lebar
df.Pivot("date", "category", "sales", aggregate: "sum");
df.Melt(["date"], ["a", "b"], "variable", "value");     // lebar -> panjang

Reshaping.PivotTable(df, "date", "category", ["sales", "units"]);
Reshaping.OneHot(df, "category");                       // satu kolom indikator per kategori
```

Kombinasi yang tidak ada pada pivot menghasilkan NaN, bukan error — itulah yang membuatnya bisa
dipakai pada data nyata yang tidak rapi.

## Join

Sisi kanan di-hash sekali lalu sisi kiri dipindai, sehingga biayanya linear, bukan kuadratik.
Kunci ganda di sisi kanan menghasilkan beberapa baris keluaran, sesuai SQL dan pandas.

```csharp
left.Join(right, "id");                              // inner
left.Join(right, "id", JoinKind.Left);               // Left, Right, Outer
left.Merge(right, leftOn: "user_id", rightOn: "id");
Joins.MergeOn(left, right, ["date", "region"]);      // beberapa kunci

DataFrame.Concat([a, b, c]);                         // tumpuk baris
```

Nama kolom yang bertabrakan diberi akhiran `_right`.

## Deret waktu

```csharp
var close = df.Numeric("close");

close.Shift(2);                    close.Diff();
close.PercentChange();             close.CumulativeSum();

close.Rolling(window: 7).Mean();   // inkremental — O(1) per baris berapa pun lebarnya
close.Rolling(7).Sum();  .Std();  .Min();  .Max();
close.Rolling(7).Median();         // mengurutkan ulang tiap jendela; lambat pada jendela lebar
close.Rolling(7, minPeriods: 1).Mean();
close.Rolling(20).Apply(v => v.Max() - v.Min(), "range");

close.Expanding().Mean();          // kumulatif
close.ExponentialMovingAverageBySpan(20);
```

### Resampling

Ember berasal dari kalender, bukan hitungan tick tetap, sehingga resample bulanan jatuh tepat pada
batas bulan berapa pun panjang bulannya dan tanpa terganggu tahun kabisat.

```csharp
Resampling.Resample(df, "date", ResampleFrequency.Monthly, "mean");
Resampling.Resample(df, "date", ResampleFrequency.Weekly, "sum", ["sales"]);
```

Frekuensi: `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, `Yearly`.

## Analitik dan interoperabilitas

```csharp
df.Describe();
df.CorrelationMatrix();

df.ToNdArray();                              // semua kolom numerik
df.ToNdArray("age", "income", "score");      // kolom tertentu, sesuai urutan
NumericSeries.FromNdArray("x", array);
DataFrame.FromMatrix(matrix, ["a", "b", "c"]);

var (codes, categories) = df.Text("category").Factorize();   // teks -> kode integer
```

`ToNdArray` adalah jembatan ke GraviLearn:

```csharp
var features = df.ToNdArray("age", "income");
var labels = df.Numeric("churn").ToNdArray();
model.Fit(features, labels);
```

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| `InvalidOperationException` pada `Numeric(...)` | Kolomnya disimpulkan sebagai teks — periksa `Info()` |
| Kolom ID menjadi angka | Paksa dengan `ColumnTypes = { ["id"] = DataType.Text }` |
| Transformasi tampak tidak berpengaruh | Frame immutable; gunakan nilai kembaliannya |
| Rolling median lambat | Ia mengurutkan ulang tiap jendela; gunakan `Mean` bila cukup |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
