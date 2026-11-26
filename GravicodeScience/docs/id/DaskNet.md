# Dask.Net

*[English](../DaskNet.md)* · Komputasi paralel dan terdistribusi untuk .NET: task graph yang
lazy, array ber-chunk, DataFrame berpartisi dengan optimiser query, SQL, serta cluster lokal
maupun multi-mesin yang pulih ketika kehilangan worker.

```bash
dotnet add package Gravicode.Science.DaskNet
```

Paketnya `Gravicode.Science.DaskNet`, namespace-nya `Gravicode.Science.DaskNet`, dan dibangun di
atas GraviNum (chunk array) dan GraviFrame (partisi DataFrame).

## Modelnya

Semuanya adalah **task graph**. Sebuah koleksi — `Delayed<T>`, `Bag<T>`, `DaskArray`,
`DaskFrame` — adalah resep untuk sebuah graph: membuatnya tidak mengerjakan apa pun. `Compute`
menyerahkan graph ke **scheduler**, yang menjalankan task sesuai urutan dependensi dan
mengembalikan hasilnya.

Sebelum dijadwalkan, graph di-**cull** (task yang tidak dibutuhkan dibuang) dan di-**fuse**:
rantai task yang masing-masing hanya memberi makan satu task lain digabung menjadi satu task,
sehingga satu putaran penjadwalan dan satu hasil antara per mata rantai hilang.

| Scheduler | Menjalankan task di | Dipakai untuk |
|---|---|---|
| `SynchronousScheduler` | thread pemanggil, urut prioritas | debugging, profiling, stack trace yang deterministik |
| `ThreadScheduler` (default) | thread pool | apa pun yang muat di satu mesin |
| `Client` (dari `LocalCluster` atau `Client.ConnectAsync`) | proses worker lewat TCP | isolasi, lebih dari satu mesin |

```csharp
using Gravicode.Science.DaskNet;

var x = Dask.Value(3.0);
var y = Dask.Delayed(Square, x);          // belum ada yang berjalan
y.Compute();                              // scheduler default
y.Compute(new SynchronousScheduler());    // eksplisit

Dask.DefaultScheduler = new ThreadScheduler(maxWorkers: 4);
using (Dask.UseScheduler(new SynchronousScheduler())) { y.Compute(); }   // ber-scope, ikut alur async
```

Scheduler lokal melepas sebuah hasil begitu pembaca terakhirnya selesai, sehingga graph dengan
ribuan partisi tidak menyimpan semuanya di memori. `RunStatistics.PeakResultsHeld` melaporkan
puncaknya. Task diprioritaskan depth-first, seperti Dask, sehingga reduksi mengonsumsi partisi
segera setelah dihasilkan alih-alih mematerialisasi semua daun lebih dulu.

**Retry** diatur per scheduler: `new ThreadScheduler(retry: new RetryPolicy(MaxRetries: 3,
Delay: TimeSpan.FromMilliseconds(100)))`. Task yang tetap gagal memunculkan `DaskTaskException`
yang menyebut key dan label task-nya, dengan exception aslinya di dalam.

### Memvisualisasikan graph

```csharp
var graph = Dask.Graph(total);
graph.ToMermaid();               // tempel ke renderer Markdown mana pun
graph.ToDot();                   // Graphviz
graph.ToMermaid(highLevel: true) // satu node per layer (map, reduce, shuffle...) bukan per task
```

## Delayed (`dask.delayed`)

```csharp
var parts = files.Select(f => Dask.Delayed(Load, Dask.Value(f))).ToList();
var cleaned = parts.Select(p => p.Then(Clean));
var total = Dask.All(cleaned).Then(xs => xs.Sum());
var (sum, count) = Dask.Compute(total, Dask.All(parts).Then(p => p.Length));   // satu graph, pekerjaan bersama hanya jalan sekali
```

**Lambda** hanya berjalan di scheduler lokal. Agar bisa berjalan di cluster, berikan **method
group** dari method public static yang ditandai `[RemoteTask]` — yang dikirim lewat jaringan
adalah nama method-nya, tidak pernah kodenya:

```csharp
public static class Tasks
{
    [RemoteTask] public static double Square(double x) => x * x;
}
var y = Dask.Delayed(Tasks.Square, x);   // bisa dijalankan remote
```

## Bag (`dask.bag`)

Barisan berpartisi berisi `T` apa pun, untuk data semi-terstruktur — baris log, record JSON, teks.

```csharp
var words = Bag.ReadText("logs/*.txt", blockSize: 64 << 20)
    .FlatMap(line => line.Split(' '))
    .Filter(w => w.Length > 3);

words.Count().Compute();
words.Frequencies().Compute();              // dihitung per partisi, lalu digabung
words.TopK(10, w => w.Length).Compute();
words.FoldBy(w => w[0], () => 0, (n, _) => n + 1, (a, b) => a + b).Compute();
words.GroupBy(w => w[0]).Compute();         // shuffle penuh - FoldBy lebih murah bila cukup
words.Distinct().Repartition(4);

// Sintaks query LINQ juga bisa:
var q = from line in Bag.ReadText("data.txt") where line.Length > 0 select line.Trim();
```

`ReadText` memecah berkas menjadi rentang byte; sebuah baris milik blok tempat byte pertamanya
jatuh, sehingga blok bisa dibaca terpisah tanpa ada baris yang terhitung dua kali atau hilang.
Reduksi numerik (`Mean`, `Var`, `Std`) menggabungkan momen Welford dengan rumus Chan, yang tetap
akurat ketika jumlah kuadrat naif mengalami pembatalan (nilai sekitar 1e9 dengan sebaran kecil).

## Array (`dask.array`)

Kisi (grid) chunk `NdArray` GraviNum.

```csharp
var x = DaskArray.RandomNormal([20_000, 5_000], chunks: [2_000, 1_000], seed: 7);
var gram = x.T.Dot(x) / x.Shape[0];
NdArray result = gram.Compute();

x.Std().Compute();                 // reduksi penuh
x.Sum(axis: 1).Compute();          // reduksi per sumbu, digabung bertingkat
x.Rechunk([5_000, 5_000]);         // ubah ukuran chunk
x.MapBlocks(b => b * 2);           // fungsi NdArray apa pun, per blok (hanya scheduler lokal)
```

Operasi element-wise membutuhkan bentuk yang sama (belum ada broadcasting); array dengan chunking
berbeda di-rechunk ke kisi bersama lebih dulu. `Random` memberi seed tiap blok dari `(seed,
indeks blok)`, sehingga hasilnya tidak bergantung pada berapa thread yang menjalankannya. Setiap
kernel array adalah `[RemoteTask]`, jadi graph array langsung bisa berjalan di cluster.

## DataFrame (`dask.dataframe`)

`DaskFrame` adalah **rencana logis** atas partisi-partisi, masing-masing sebuah `DataFrame`
GraviFrame. Operasi per baris menerima **ekspresi**, bukan lambda:

```csharp
using static Gravicode.Science.DaskNet.Expr;

var sales = DaskFrame.ReadCsv("sales/*.csv", blockSize: 64 << 20);
var report = sales
    .Filter(Col("region").In("north", "east") & (Col("price") > 20))
    .WithColumn("revenue", Col("price") * Col("qty"))
    .Join(DaskFrame.ReadCsv("products.csv"), "product")
    .GroupBy("region", "category").Agg(("revenue", "sum"), ("price", "mean"))
    .SortBy("revenue_sum", ascending: false);

Console.WriteLine(report.Explain());     // rencana yang sudah dioptimasi
DataFrame result = report.Compute();
```

Ekspresi adalah alasan optimiser ada: predikat yang hanya menyebut `region` bisa dipindah ke
bawah join dan masuk ke scan CSV; proyeksi memberi tahu scan kolom mana yang perlu di-parse.
`Func<RowView, bool>` tidak terbaca oleh keduanya. Ekspresi juga bisa diserialisasi, dan itulah
yang membuat sebuah rencana bisa berjalan di cluster. Nilai hilang mengikuti logika tiga nilai
SQL: `NULL > 5` bernilai unknown, dan filter hanya menyimpan baris yang predikatnya true.

**Yang dikerjakan optimiser**

- **Predicate pushdown** melewati proyeksi, rename, kolom baru (bila predikat tidak membacanya),
  sort, repartition, concat, agregasi (predikat pada key grup) dan join. Pada left join, hanya
  predikat sisi kiri yang turun; menurunkan predikat sisi kanan ke bawah left join akan mengubah
  pasangan yang tidak cocok menjadi baris yang terbuang.
- **Projection pushdown**: scan hanya mem-parse kolom yang dibaca oleh langkah berikutnya.
- **Pipeline fusion**: semua langkah per baris di antara dua shuffle berjalan dalam satu task
  per partisi.

Bandingkan `Explain(optimized: false)` dengan `Explain()` untuk melihat apa yang dipindahkan.

**Strategi eksekusi**

| Operasi | Strategi |
|---|---|
| `GroupBy().Agg` | agregat parsial per partisi, digabung bertingkat (`DefaultSplitEvery`, default 8), lalu diselesaikan sekali. Mean/var/std membawa state (count, mean, M2). |
| `Join` / `Merge` | **broadcast** bila sisi kanan satu partisi (inner dan left join); selain itu **hash shuffle** agar key yang cocok bertemu. Hash-nya FNV-1a atas key bertipe, karena `string.GetHashCode` berbeda antar proses. |
| `SortBy` | range partitioning dengan batas yang diambil dari sampel data, sehingga menggabungkan partisi secara berurutan menghasilkan frame terurut. |
| `Repartition`, `Limit`, `Concat` | menggabung/memecah partisi bertetangga; tanpa shuffle. |

Agregasi: `sum`, `mean`, `count`, `size`, `min`, `max`, `var`, `std` (sampel, ddof = 1),
`var_pop`, `std_pop`.

**Berkas.** `ReadCsv` memecah berkas besar menjadi blok byte dan mengunci tipe kolom dari sebuah
sampel, sehingga setiap blok mem-parse kolom dengan cara yang sama. `ReadArrow` membaca berkas
Arrow IPC, satu partisi per berkas. `ReadFiles(pattern, reader)` menerima reader apa pun —
`ReadFiles("*.parquet", ParquetIO.Read)` dengan paket GraviFrame.Parquet. `ToCsv(dir)` dan
`ToArrow(dir)` menulis satu berkas per partisi, ke `.partial` lalu di-rename, sehingga run yang
crash tidak meninggalkan bagian yang setengah tertulis.

**Checkpoint.** `frame.Checkpoint("ckpt/")` menulis setiap partisi ke Arrow dan mengembalikan
frame yang membaca dari sana; partisi yang sudah ada tidak dihitung ulang, sehingga pipeline
panjang yang dijalankan ulang setelah crash melanjutkan dari titik berhentinya. `Persist()`
melakukan hal yang sama di memori.

## SQL (`dask-sql`)

```csharp
using Gravicode.Science.DaskNet.Sql;

var sql = new SqlContext()
    .RegisterCsv("titanic", "datasets/titanic.csv")
    .Register("ports", portsFrame);

DataFrame r = sql.Execute("""
    SELECT t.class, p.country, COUNT(*) AS n, ROUND(AVG(t.survived) * 100, 1) AS pct
    FROM titanic t LEFT JOIN ports p ON t.embark_town = p.town
    WHERE t.age IS NOT NULL
    GROUP BY t.class, p.country
    HAVING COUNT(*) > 10
    ORDER BY pct DESC
    LIMIT 10
    """);

DaskFrame lazy = sql.Sql("SELECT ...");    // susun lebih lanjut, atau Compute di cluster
sql.Explain("SELECT ...");                 // rencana yang sudah dioptimasi
```

Query dikompilasi menjadi rencana `DaskFrame` biasa, sehingga mendapat optimiser yang sama dan
berjalan di scheduler mana pun.

Didukung: `SELECT [DISTINCT]`, `*` dan `t.*`, alias, tabel dan subquery di `FROM`,
`[INNER|LEFT|RIGHT|FULL] JOIN ... ON` (kesamaan antara kedua sisi, plus kondisi tambahan pada
inner join), `WHERE`, `GROUP BY` (ekspresi, alias output, dan posisi), `HAVING`, `ORDER BY`
(output, posisi, atau ekspresi apa pun), `LIMIT`. Ekspresi: aritmetika, `||`, perbandingan,
`AND/OR/NOT`, `IS [NOT] NULL`, `[NOT] IN (...)`, `[NOT] BETWEEN`, `[NOT] LIKE`, `CASE` (kedua
bentuk), `CAST`, `DATE '...'`. Agregat: `COUNT(*)`, `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`,
`STDDEV[_SAMP|_POP]`, `VARIANCE`/`VAR_SAMP`/`VAR_POP`. Fungsi: `ABS`, `SQRT`, `LN`/`LOG`, `EXP`,
`POWER`, `ROUND`, `FLOOR`, `CEIL`, `LOWER`, `UPPER`, `TRIM`, `LENGTH`, `COALESCE`, `IFNULL`,
`NULLIF`, `CONCAT`, `YEAR`, `MONTH`, `DAY`.

Setelah join, setiap kolom bisa dirujuk sebagai `alias.kolom`; nama tanpa kualifikasi berlaku
bila hanya satu tabel yang memilikinya. Dengan `SELECT *`, nama yang bentrok keluar dengan
kualifikasi (`s.product`, `p.product`); baris left join yang tidak cocok memiliki key sisi kanan
yang **hilang**, sesuai SQL, bukan disalin dari sisi kiri.

Ditolak dengan `SqlException` yang menyebut posisinya, alih-alih diaproksimasi: `OFFSET`,
`UNION`, `CROSS JOIN`, `COUNT(DISTINCT ...)`, subquery di luar `FROM`, window function, CTE, dan
join tanpa kesamaan antar sisi.

## Eksekusi terdistribusi (`dask.distributed`)

```csharp
using Gravicode.Science.DaskNet.Distributed;

// Worker di proses ini - tetap lewat TCP dan diserialisasi, jadi yang jalan di sini jalan juga di cluster sungguhan.
await using var cluster = await LocalCluster.StartAsync(workers: 4, threadsPerWorker: 2);
var result = report.Compute(cluster.Client);

// Worker sebagai proses anak (isolasi; crash di sebuah task tidak menjatuhkan client).
await using var procs = await LocalCluster.StartProcessesAsync("DaskNet.Worker.dll", workers: 4,
    preload: ["MyTasks.dll"]);
```

Di mesin lain, jalankan host worker:

```bash
export DASKNET_TOKEN=<minimal 16 karakter, dipakai bersama oleh cluster>
dotnet DaskNet.Worker.dll --host 0.0.0.0 --port 8786 --threads 8 --name node-1 --preload MyTasks.dll
```

lalu sambungkan: `await Client.ConnectAsync(["node-1:8786", "node-2:8786"], token)`.

**Apa yang dikirim lewat jaringan.** Task adalah referensi method plus argumen. Hanya method
`public static` bertanda `[RemoteTask]` yang bisa disebut — worker menolak yang lain, sehingga
client tidak bisa menyuruhnya menjalankan `File.Delete`. Nilai memakai codec biner tertutup
(angka, string, array, `NdArray`, `DataFrame` dengan nilai hilang dan kategorikal, referensi
task); `WireCodec.Register` menambah tipe. Graph yang berisi lambda ditolak dengan
`NotSupportedException` sebelum ada yang dikirim.

**Penjadwalan.** Client menempatkan setiap task pada worker yang sudah memegang input-nya paling
banyak, dengan pemecah seri berupa jumlah task berjalan paling sedikit. Worker mengambil input
yang kurang langsung dari sesama worker, bukan lewat client, dan client melepas hasil begitu
tidak ada lagi yang membutuhkannya.

**Toleransi kegagalan.** Ketika sebuah worker hilang, client menandai setiap hasil yang
dipegangnya sebagai hilang dan menyusun ulang rencana: task yang input-nya hilang dihitung ulang
dari lineage graph, lalu run berlanjut di worker yang tersisa. Retry menangani task yang
melempar exception. Keduanya diuji dengan membunuh worker (in-process maupun sebagai proses
sungguhan) di tengah run.

**Keamanan.** Koneksi diawali tantangan HMAC-SHA256 atas token bersama, sehingga hanya pemegang
token yang bisa mengirim task. Lalu lintasnya **tidak dienkripsi** — tempatkan cluster di
jaringan tepercaya atau lewat tunnel; TLS ada di roadmap.

## Observabilitas

Setiap task adalah sebuah `Activity` pada `ActivitySource` bernama `Gravicode.Science.DaskNet`,
dan `Meter` dengan nama yang sama menerbitkan `dask.tasks.completed`, `dask.tasks.failed`,
`dask.tasks.retried` serta histogram `dask.task.duration`. Sambungkan ke OpenTelemetry seperti
biasa (`AddSource(DaskTelemetry.Name)`, `AddMeter(DaskTelemetry.Name)`). Untuk melihat cepat,
scheduler dan `Client` menerima callback `OnEvent` (`Started`, `Finished`, `Retrying`, `Failed`,
`Recomputing`), dan `LastRun` berisi `RunStatistics` (task per worker, retry, puncak hasil yang
disimpan).

## Performa

BenchmarkDotNet, `benchmarks/DaskNet.Benchmark`, Release, laptop 4-core 15 W yang dijelaskan di
[benchmarks.md](benchmarks.md), jalur managed (tanpa BLAS native). Rasio diambil dalam satu run;
mesin ini mengalami throttling, sehingga waktu absolut berubah antar run.

| Benchmark | Hasil |
|---|---|
| Overhead scheduler | ~5 µs per task sinkron, ~6–9 µs di thread pool (100–1000 task sepele). Usahakan task ≥ 1 ms, sama seperti Dask. |
| `Std` dari 2000² / 4000² dalam 16 chunk | thread scheduler **0,61× / 0,50×** waktu GraviNum pada array utuh; sinkron 1,39× / 1,20× |
| Filter + group-by, 200 rb / 1 jt baris | GraviFrame pada satu frame lebih cepat di ukuran ini (Dask.Net 3,2× / 1,55× waktunya, dengan varians tinggi). Partisi menguntungkan ketika data tidak muat, atau dengan cluster. |
| Reduksi array, cluster in-process vs thread | 1,12× — batas bawah biaya TCP loopback dan serialisasi |

Ringkasan jujurnya: ketika data muat di memori dan satu panggilan library sudah paralel, pakai
Dask.Net untuk data yang tidak muat, bukan demi kecepatan.

## Cakupan terhadap daftar fitur

`Dask.Net-Features.md` menggambarkan produk yang jauh lebih besar. Inilah yang sudah ada hari ini.

| Area | Status |
|---|---|
| DataFrame berpartisi dan lazy; optimasi query; predicate dan projection pushdown | **Selesai** |
| CSV (dipecah per blok), Arrow IPC, berkas apa pun via `ReadFiles` (Parquet lewat GraviFrame.Parquet) | **Selesai** |
| JSON, Excel, Delta Lake, Iceberg, Hudi | Roadmap (`ReadFiles` dengan reader sudah bisa hari ini) |
| Mesin lazy: eksekusi tertunda, planner, pipeline fusion | **Selesai**; optimasi berbasis biaya masih roadmap (baru berbasis aturan) |
| DAG scheduler: visualisasi (DOT/Mermaid), cull/fuse, retry policy, pelacakan dependensi | **Selesai** |
| Cluster lokal: multi-core, isolasi proses | **Selesai**; batas memori per worker masih roadmap |
| Cluster multi-node lewat TCP | **Selesai**; transport gRPC/QUIC/HTTP/3 masih roadmap |
| Penemuan worker (Consul, Kubernetes, Service Fabric, Aspire) | Roadmap — alamat diberikan secara eksplisit |
| Parallel LINQ: map, reduce, fold, aggregate, shuffle, broadcast | **Selesai** (`Bag`) |
| Koleksi terdistribusi (`DistributedList` dsb.) | Roadmap; `Bag`, `DaskArray` dan `Persist` menangani kasus yang dominan baca |
| Arrow: zero-copy, lintas bahasa | Berkas Arrow IPC **selesai**; shared memory antar proses masih roadmap |
| SQL: join, agregasi | **Selesai**; window function, CTE dan ANSI penuh masih roadmap |
| GIS, ML, ETL/streaming, AI, basis data vektor | Roadmap (modul terpisah) |
| Observabilitas: OpenTelemetry | **Selesai** (ActivitySource + Meter); Prometheus/Grafana/Jaeger lewat exporter OTel |
| Operator Kubernetes, dashboard Aspire | Roadmap |
| Toleransi kegagalan: retry task, checkpoint, pemulihan worker | **Selesai**; eksekusi spekulatif masih roadmap |
| Konektor penyimpanan (basis data, data lake, warehouse) | Roadmap |
| Scheduler: thread, cluster | **Selesai**; pool proses adalah `LocalCluster.StartProcessesAsync`; Kubernetes masih roadmap |
| Keamanan: worker dengan autentikasi token | **Selesai**; TLS/mTLS/OAuth2/Entra ID, multi-tenancy dan kuota masih roadmap |

## Batasan yang diketahui

- Penyusunan graph untuk `SortBy` langsung menjalankan pengambilan sampel, yang membaca input
  sekali lagi.
- `MapPartitions`, `Bag.Map` dengan lambda dan `DaskArray.MapBlocks` hanya untuk scheduler
  lokal; tulis method `[RemoteTask]` untuk menjalankan kode kustom di cluster.
- Join dengan beberapa key membutuhkan nama key yang sama di kedua sisi (SQL menangani nama yang
  berbeda).
- Client adalah satu-satunya scheduler sebuah run; bila client mati, run-nya hilang (worker tetap
  hidup dan menerima client berikutnya).

---

Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil
