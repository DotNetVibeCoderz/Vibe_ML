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

## Fungsi window

Fungsi window dievaluasi di dalam grup, dan itulah yang membedakannya dari agregat biasa.

```csharp
Windowing.Rank(frame, ["customer"], "amount", descending: true);
Windowing.CumulativeSum(frame, ["customer"], "amount");
Windowing.RollingMean(frame, ["customer"], "amount", window: 7);
Windowing.Lag(frame, ["customer"], "amount");
Windowing.Lead(frame, ["customer"], "amount");
```

Daftar partisi adalah intinya. Rata-rata bergulir yang dihitung atas frame berisi beberapa pelanggan
mencampur riwayat satu pelanggan ke pelanggan lain, dan — lebih buruk lagi — `Lag` tanpa partisi
membuat baris pertama setiap grup menjangkau baris terakhir grup sebelumnya. Itu jenis kebocoran
yang diam-diam menggelembungkan skor model dan tidak terlihat pada keluaran.

Dua pilihan yang disengaja:

- **`RollingMean` membiarkan jendela yang belum penuh tetap kosong** alih-alih merata-ratakan apa
  yang ada. Mengisinya adalah pilihan yang lebih umum dan lebih menyesatkan: beberapa nilai pertama
  lalu membawa varians jauh lebih besar daripada sisanya, tanpa ada yang menyatakannya di keluaran.
- **Nilai seri pada `Rank` mengikuti flag `dense`.** `false` menyisakan celah sesudahnya
  (1, 2, 2, 4) dan `true` tidak (1, 2, 2, 3).

Daftar partisi kosong memperlakukan seluruh frame sebagai satu grup, sebagaimana yang dilakukan SQL.

## As-of join

Join biasa pada timestamp hanya cocok bila dua sistem mencatat momen yang persis sama, dan jam
sungguhan tidak pernah begitu. `AsOfJoin` mencocokkan setiap baris kiri dengan baris kanan terkini
pada atau sebelum kuncinya.

```csharp
Windowing.AsOfJoin(trades, quotes, on: "time");
Windowing.AsOfJoin(trades, quotes, on: "time", tolerance: 30);
```

**Arahnya sengaja hanya ke belakang.** Mencocokkan baris *terdekat* ke arah mana pun mudah ditulis
dan merupakan look-ahead: ia membiarkan nilai yang tercatat setelah peristiwa menginformasikan baris
yang menggambarkan peristiwa itu, dan begitulah sebuah backtest berakhir dengan memprediksi masa
lalu. Baris kiri yang mendahului semua baris kanan dikembalikan sebagai kosong, bukan dicocokkan
dengan baris masa depan pertama.

`tolerance` adalah batas seberapa jauh ke belakang kecocokan boleh diambil. Kuotasi yang basi
sembilan puluh detik sering lebih buruk daripada tidak ada kuotasi, dan inilah cara menyatakannya.
Join ini mengurutkan frame kanan sendiri alih-alih menuntut masukan terurut, karena menuntutnya
adalah cara mudah mendapat jawaban salah secara diam-diam.

## Kolom kategorikal

`CategoricalSeries` menyimpan teks sebagai kode integer ke dalam kamus bersama.

```csharp
var sizes = CategoricalSeries.FromValues(
    "size", values, categories: ["low", "medium", "high"], ordered: true);

sizes.ArgSort();                    // menurut peringkat yang dideklarasikan, bukan abjad
sizes.CategoryCounts();             // termasuk kategori yang tidak punya baris
sizes.OneHot(dropFirst: true);
sizes.ToCodes();                    // nilai hilang menjadi NaN, bukan -1
sizes.ReorderCategories(["low", "medium", "high"]);
sizes.RemoveUnusedCategories();
```

Ada dua alasan memakainya dan keduanya saling bebas. Yang pertama adalah ukuran: kolom berisi nama
negara per baris menyimpan beberapa lusin string yang sama ratusan ribu kali, dan pengkodean
mengubahnya menjadi `int[]` sehingga pengelompokan dan penggabungan menjadi perbandingan integer.

Yang kedua adalah **pengurutan**. `TextSeries` hanya bisa mengurut secara abjad, yang menempatkan
"high" sebelum "low" sebelum "medium" — jawaban salah yang nyata dan mudah terlewat untuk data
ordinal. Kategorikal terurut mengurut menurut urutan kategori yang dideklarasikan pemanggil.

**Kategori adalah bagian dari tipe kolom, bukan ringkasan isinya.** Kategori tanpa baris tetap ada,
dan itulah yang membuat group-by menghasilkan grup kosong alih-alih menghilangkannya diam-diam.
Karena itu `Take` mempertahankan setiap kategori — menyaring baris tidak boleh mengubah tipe kolom —
dan `RemoveUnusedCategories` adalah cara eksplisit untuk membuangnya. Dengan alasan sama, menetapkan
nilai di luar himpunan kategori akan melempar kesalahan alih-alih melebarkan kamus: itu akan membuat
tipe kolom bergantung pada urutan penulisan yang kebetulan terjadi.

`ToCodes` memetakan nilai hilang ke `NaN`, bukan -1, karena -1 akan terbaca sebagai kategori dengan
peringkat di bawah semua kategori lain, dan itu justru hal yang salah untuk diberikan kepada model.

## SQL

`SqlReader` dan `SqlWriter` ditulis terhadap `System.Data.Common`, sehingga bekerja dengan SQL
Server, PostgreSQL, SQLite, MySQL, atau apa pun yang menyediakan `DbConnection` — dan tidak
memerlukan dependensi paket untuk itu. Pemanggil menyediakan koneksinya, sehingga string koneksi,
pooling, dan kredensial tetap berada di tempatnya.

```csharp
using var connection = new SqliteConnection("Data Source=data.db");

var frame = SqlReader.Read(connection,
    "SELECT id, name, score FROM people WHERE score > $floor",
    new Dictionary<string, object?> { ["$floor"] = 85.0 });

SqlWriter.Write(frame, connection, "metrics");
```

**Sampaikan nilai lewat `parameters`, jangan pernah dengan merangkai string SQL.** Menyisipkan
masukan pengguna ke dalam teks kueri adalah asal mula SQL injection, dan kenyataan bahwa ia berhasil
saat pengujian justru itulah yang membuatnya berbahaya.

Tipe kolom berasal dari penyedia, bukan disimpulkan, dan itulah alasan utama memilih ini daripada
mengekspor ke CSV lalu membacanya kembali: basis data sudah tahu bahwa sebuah kolom adalah tanggal,
bukan string yang tampak seperti tanggal. Nama kolom ganda — sah di SQL, tidak sah di frame — diberi
akhiran alih-alih saling menimpa diam-diam, dan koneksi dikembalikan ke keadaan semula.

Nama tabel tidak bisa dijadikan parameter, sehingga `SqlWriter` menyisipkannya dan menolak apa pun
yang bukan pengenal polos. Penulisan dikelompokkan ke dalam transaksi dan digulung balik saat gagal:
data yang tertulis separuh lebih buruk daripada tidak ada, karena tidak ada apa pun di tabel yang
menyatakan separuh yang mana.

## Excel

`ExcelReader` dan `ExcelWriter` menangani `.xlsx` tanpa pustaka spreadsheet. Formatnya adalah arsip
zip berisi bagian-bagian XML, dan semuanya sudah bisa dibuka oleh BCL.

```csharp
ExcelReader.SheetNames("report.xlsx");
var frame = ExcelReader.Read("report.xlsx", new ExcelOptions { SheetName = "Results" });
ExcelWriter.Write(frame, "output.xlsx", sheetName: "Cities");
```

Tiga hal tentang format ini menjebak siapa pun yang menulis pembaca untuk pertama kalinya, dan
semuanya ditangani:

- **Sel kosong itu tidak ada, bukan berisi kosong.** Sebuah baris hanya mencatat sel yang berisi
  sesuatu, sehingga elemen `<c>` ketiga pada suatu baris belum tentu kolom C. Referensi `r` milik sel
  itulah yang menyatakan letaknya, dan membaca berdasarkan posisi akan menggeser diam-diam setiap
  nilai setelah celah.
- **Tanggal adalah angka.** Excel menyimpannya sebagai hari sejak 1899-12-30 dan menandainya hanya
  lewat format angka, sehingga kolom tanggal tiba tampak seperti bilangan bulat lima digit.
- **Bug tahun kabisat 1900.** Excel meyakini 1900 adalah tahun kabisat, demi kompatibilitas dengan
  Lotus 1-2-3. Epoknya adalah 1899-12-30, bukan 1899-12-31, dan itulah yang membuat setiap tanggal
  sejak 1900-03-01 keluar dengan benar.

Lembar kerja dibaca secara mengalir dengan `XmlReader`, bukan dimuat sebagai dokumen, karena itulah
satu-satunya bagian workbook yang bisa benar-benar besar. Rumus tidak dievaluasi; sel rumus
menghasilkan nilai tersimpannya, yang memang itulah yang dicatat berkas dan hampir selalu yang
diinginkan pemanggil.

---

## Visualisasi

Dihasilkan oleh `samples/GraviFrame.Console`; `notebooks/GraviFrame.Notebook.ipynb` menambahkan
grafik volatilitas bergulir yang dibangun dengan fungsi window v0.4.

![Deret harga dengan rata-rata bergulir di atasnya](../screenshots/graviframe_trend.png)

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
