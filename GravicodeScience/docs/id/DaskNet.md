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
var x = DaskArray.RandomNormal([20_000, 5_000], chunkSize: [2_000, 1_000], seed: 7);
var gram = x.T.Dot(x) / x.Shape[0];
NdArray result = gram.Compute();

x.Std().Compute();                 // reduksi penuh
x.Sum(axis: 1).Compute();          // reduksi per sumbu, digabung bertingkat
x.Rechunk([5_000, 5_000]);         // ubah ukuran chunk
x.MapBlocks(b => b * 2);           // fungsi NdArray apa pun, per blok (hanya scheduler lokal)
```

Operasi element-wise melakukan **broadcasting** dengan aturan NumPy, sehingga `x - x.Mean(axis: 0)`
memusatkan kolom dan `x / x.Sum(axis: 1).ExpandDims(1)` menormalkan baris. Operand yang
direntangkan memakai satu chunk sepanjang sumbu itu dan setiap blok keluaran membacanya; selain
itu array dengan chunking berbeda di-rechunk ke kisi bersama lebih dulu. `Random` memberi seed tiap blok dari `(seed,
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
| `Join` / `Merge` | **broadcast** (inner dan left join) bila sisi kanan satu partisi atau diperkirakan paling besar `Dask.BroadcastJoinBytes` (16 MB); inner join yang sisi *kiri*-nya kecil dibalik. Selain itu **hash shuffle** agar key yang cocok bertemu. `broadcast: true/false` menimpa pilihan itu, seperti `broadcast=` di Dask. Hash-nya FNV-1a atas key bertipe, karena `string.GetHashCode` berbeda antar proses. |
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

**Estimasi ukuran.** Strategi join diambil dari estimasi ukuran, yang ditampilkan `Explain()` di
setiap join (`[est. left 1.2 MB, right 4 KB]`). Partisi di memori diukur, berkas dihitung dari
panjangnya di disk, filter dianggap menyisakan separuh input dan proyeksi sebanding jumlah
kolomnya. Bila keluaran tidak bisa dibatasi - agregat ber-group, join - estimasinya sengaja
kebesaran, karena estimasi yang kekecilan adalah kesalahan yang mahal: data besar akan di-broadcast
ke setiap partisi. Partisi dari `FromDelayed` atau `MapPartitions` tidak punya estimasi dan tidak
pernah di-broadcast berdasarkan ukuran. Membalik join hanya mengubah urutan baris, yang memang
tidak pernah dijanjikan join yang di-shuffle.

### Konektor: JSON dan basis data

```csharp
// JSON Lines, dipecah menjadi blok byte seperti CSV; atau lines: false untuk berkas berisi satu array.
var events = DaskFrame.ReadJson("logs/*.jsonl", blockSize: 32L << 20);
events.ToJson("out/");                                    // satu .jsonl per partisi

// Provider ADO.NET apa pun (SQL Server, PostgreSQL, SQLite, MySQL ...), dipecah menurut kolom numerik.
var orders = DaskFrame.ReadSql(NpgsqlFactory.Instance, connectionString, "sales.orders", index: "id", partitions: 16);
// ... atau satu partisi per query, untuk pemecahan yang tidak bisa dinyatakan dengan rentang.
var byMonth = DaskFrame.ReadSql(SqliteFactory.Instance, cs, months.Select(m =>
    new SqlPartition("SELECT * FROM orders WHERE month = @m", new Dictionary<string, object?> { ["m"] = m })).ToList());
```

**JSON.** Kolom dan tipenya ditetapkan dari sampel berkas pertama (`inferenceRows`), lalu setiap
blok dibaca dengan skema itu, sehingga posisi awal blok tidak bisa mengubah tipe kolom. String
angka di kolom numerik di-parse, nilai yang tidak cocok menjadi hilang, objek dan array bersarang
disimpan sebagai teks JSON-nya, dan string menjadi tanggal hanya bila setiap sampelnya ISO-8601.
Properti yang baru muncul setelah sampel diabaikan.

**Basis data.** Dask.Net tidak bergantung pada driver apa pun: ia terikat ke `System.Data.Common`
lewat `SqlReader` milik GraviFrame, dan worker membuat koneksinya sendiri dari `DbProviderFactory`
provider (dikirim lewat wire sebagai nama tipe, jadi worker cluster harus sudah memuat provider-nya
- `--preload`). Pembacaan tabel membagi `[MIN, MAX]` kolom indeks secara merata; rentang pertama
terbuka ke bawah dan yang terakhir terbuka ke atas serta mengambil baris berindeks NULL, sehingga
setiap baris terbaca tepat sekali walau batas yang diberikan terlalu sempit. Hanya kolom yang
dibutuhkan query yang di-SELECT - projection pushdown sampai ke SQL. Tipe di-pin dari probe skema
sehingga rentang yang kosong atau seluruhnya NULL tidak kembali dengan tipe berbeda (sebuah tes
mereproduksi persis hal itu dengan kolom SQLite tanpa tipe). Nama tabel dan kolom ditulis ke dalam
SQL dan harus berupa identifier biasa; nilai selalu lewat parameter. Connection string ikut
terkirim bersama setiap task - gunakan TLS di cluster.

**Filter juga sampai ke basis data.** Filter pada pembacaan tabel (`orders.Filter(Col("qty") > 10)`,
atau `WHERE` SQL di atas tabel `ReadSql` yang terdaftar) diterjemahkan ke klausa `WHERE` partisi
sejauh mungkin, sehingga basis data melewati baris alih-alih mengirimkannya. Aturannya *aman
sebagai superset*: klausa yang didorong boleh menyisakan lebih banyak baris daripada filter,
tidak pernah lebih sedikit, dan filternya sendiri tetap dijalankan pada hasil yang kembali, jadi
jawabannya tidak bisa berubah. Perbandingan numerik, `IS [NOT] NULL` dan `BETWEEN` diterjemahkan
persis; `=`/`IN` pada teks didorong tetapi diperiksa ulang, karena collation basis data bisa
membandingkan tanpa memedulikan huruf besar atau spasi (sebuah test memakai `NOCASE` SQLite untuk
membuktikan pemeriksaan ulang itu perlu); `NOT` hanya didorong di atas term yang persis; `AND`
mempertahankan sisi mana pun yang bisa diterjemahkan, `OR` hanya bila kedua sisinya bisa. Berikan
`pushFilters: false` untuk tabel yang tipe kolom terdeklarasinya tidak menggambarkan nilainya
(SQLite membolehkan kolom `REAL` berisi teks).

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

**Urutan join.** Rangkaian tiga tabel atau lebih yang digabung dengan inner equi-join diurutkan
ulang menurut estimasi ukuran (greedy operator ordering: berulang kali menggabungkan pasangan
terhubung yang estimasi gabungannya paling kecil, input yang lebih besar di kiri), dan setiap
konjungsi `WHERE` pada satu tabel membagi dua estimasi tabel itu. Kolom output kembali dalam
urutan yang ditulis. Query dibiarkan seperti ditulis bila ada join yang outer, bila ada estimasi
yang tidak tersedia, bila tabel-tabelnya tidak terhubung oleh kesamaan, atau bila urutan tertulis
sudah merupakan urutan terpilih; `SqlContext.ReorderJoins = false` mematikannya. Pengurutan ulang
mengubah rencana, tidak pernah barisnya - test membandingkan setiap query yang diurutkan ulang
dengan urutan tertulis.

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

`UNION [ALL]` didukung: cabang dicocokkan menurut posisi dan digabung dari kiri ke kanan;
`ORDER BY`/`LIMIT` setelah cabang terakhir berlaku untuk seluruh union. Setiap agregat juga bisa
ditulis `AGG(DISTINCT x)`; `COUNT(DISTINCT x)` dihitung sebagai pasangan (key, x) yang unik,
diagregasi per key lalu di-join kembali ke agregat lain, jadi biayanya satu agregasi dan satu
join tambahan.

**Window function**: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`/`LEAD` (dengan offset) dan
`COUNT`/`SUM`/`AVG`/`MIN`/`MAX` `OVER ([PARTITION BY ...] [ORDER BY ...])`, juga di atas query
yang di-group (`RANK() OVER (ORDER BY SUM(qty) DESC)`, `SUM(SUM(qty)) OVER ()`). Baris di-shuffle
dengan hash menurut key partisi sehingga setiap partisi window dihitung utuh dalam satu task (tanpa
`PARTITION BY` semuanya dikumpulkan ke satu task). Agregat dengan `ORDER BY` adalah nilai berjalan
yang menyertakan baris yang seri dengan baris saat ini - frame bawaan SQL, satu-satunya yang
didukung. Operasi yang sama tersedia sebagai `DaskFrame.Window(partitionBy, orderBy, new
WindowFunction("rn", "row_number"))`.

Ditolak dengan `SqlException` yang menyebut posisinya, alih-alih diaproksimasi: `OFFSET`,
`CROSS JOIN`, subquery di luar `FROM`, frame window eksplisit (`ROWS`/`RANGE BETWEEN`), nilai
bawaan `LAG`/`LEAD`, CTE, dan join tanpa kesamaan antar sisi.

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

**Penemuan worker.** Alih-alih alamat, berikan client sebuah `IWorkerDiscovery`. Client
tersambung ke setiap worker yang menjawab, melewati yang tidak menjawab (pod yang masih
menyala), dan dengan `refreshEvery` terus mencari: worker yang muncul belakangan ikut bergabung,
di antara run maupun di tengah run, dan dipakai oleh keputusan penempatan berikutnya.
`RefreshAsync()` menjalankan satu putaran secara manual.

```csharp
var discovery = WorkerDiscovery.Dns("dask-workers.analytics.svc.cluster.local", 8786);  // Kubernetes
// WorkerDiscovery.Aspire("dask-worker-*", "tcp")        service discovery .NET Aspire
// WorkerDiscovery.EnvironmentVariable("DASKNET_WORKERS") "host:port,host:port"
// WorkerDiscovery.Static(...), .From(lookup), .Combine(a, b)
await using var client = await Client.ConnectAsync(discovery, token, tls, refreshEvery: TimeSpan.FromSeconds(30));
```

Keduanya tidak memerlukan paket. Di Kubernetes, Service headless di-resolve menjadi satu record
per pod yang siap, sehingga menambah replika Deployment itulah yang memperbesar cluster:

```yaml
apiVersion: v1
kind: Service
metadata: { name: dask-workers }
spec:
  clusterIP: None              # headless: the name resolves to every ready pod
  selector: { app: dask-worker }
  ports: [{ port: 8786 }]
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: dask-worker }
spec:
  replicas: 4
  selector: { matchLabels: { app: dask-worker } }
  template:
    metadata: { labels: { app: dask-worker } }
    spec:
      containers:
        - name: worker
          image: registry.example/dasknet-worker:2.2.0   # dotnet publish tools/DaskNet.Worker
          args: ["--threads", "4", "--memory-limit", "6GB", "--preload", "/app/MyTasks.dll"]
          env:
            - { name: DASKNET_HOST, value: "0.0.0.0" }
            - { name: DASKNET_PORT, value: "8786" }
            - { name: DASKNET_TOKEN, valueFrom: { secretKeyRef: { name: dask, key: token } } }
          resources: { limits: { memory: 8Gi } }
```

Dengan .NET Aspire, AppHost menyuntikkan variabel `services__{nama}__{endpoint}__{i}` ke setiap
resource yang mereferensikannya; `WorkerDiscovery.Aspire` membacanya. Host worker mengambil port
dari `DASKNET_PORT` dan alamat bind dari `DASKNET_HOST` bila flag-nya tidak diberikan.
`WithReplicas` akan menaruh replika di belakang satu alamat proxy yang terlihat seperti satu
worker, jadi deklarasikan satu resource per worker dan cocokkan dengan wildcard:

```csharp
// AppHost: satu resource per worker, sehingga masing-masing punya alamat sendiri.
var token = builder.AddParameter("dask-token", secret: true);
var app = builder.AddProject<Projects.MyApp>("app").WithEnvironment("DASKNET_TOKEN", token);
for (var i = 0; i < 3; i++)
{
    var worker = builder.AddProject<Projects.DaskNet_Worker>($"dask-worker-{i}")
        .WithEndpoint(name: "tcp", scheme: "tcp", env: "DASKNET_PORT")
        .WithEnvironment("DASKNET_TOKEN", token);
    app.WithReference(worker.GetEndpoint("tcp"));
}

// MyApp
var client = await Client.ConnectAsync(WorkerDiscovery.Aspire("dask-worker-*", "tcp"), token);
```

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

**Batas memori dan spill.** `WorkerOptions.MemoryLimit` (CLI `--memory-limit 4GB`,
`LocalCluster.StartAsync(memoryLimit: ...)`) membatasi hasil yang disimpan worker di memori.
Melewati `MemoryTarget` dari batas itu (0,6, seperti Dask), hasil yang paling lama tidak dipakai
ditulis ke `SpillDirectory` dalam encoding wire-nya dan dibaca kembali saat sebuah task
membutuhkannya; peer yang meminta hasil yang sudah di-spill dikirimi byte berkasnya tanpa
di-decode. Hasil yang sendirian lebih besar dari target dibaca langsung dan tetap di disk, alih-alih
mengusir semua yang lain. Ukurannya adalah estimasi (8 byte per angka, UTF-16 untuk teks;
`SizeOf.Estimate`), bukan working set proses, dan `GetWorkerInfoAsync` melaporkan memori, jumlah
key dan byte yang di-spill per worker. Sebuah tes memusatkan array 8 MB di worker yang dibatasi
1 MB dan memeriksa hasilnya identik bit demi bit.

**Toleransi kegagalan.** Ketika sebuah worker hilang, client menandai setiap hasil yang
dipegangnya sebagai hilang dan menyusun ulang rencana: task yang input-nya hilang dihitung ulang
dari lineage graph, lalu run berlanjut di worker yang tersisa. Retry menangani task yang
melempar exception. Keduanya diuji dengan membunuh worker (in-process maupun sebagai proses
sungguhan) di tengah run.

**Keamanan.** Koneksi diawali tantangan HMAC-SHA256 atas token bersama, sehingga hanya pemegang
token yang bisa mengirim task. Tanpa TLS lalu lintasnya berupa teks biasa: siapa pun di jalur
jaringan bisa membaca argumen dan hasil task.

**TLS.** Berikan `TlsOptions` yang sama ke setiap worker dan ke client; worker juga memakainya
untuk transfer antar-worker. TLS dinegosiasikan lebih dulu dan handshake token berjalan di
dalamnya.

```csharp
// Satu sertifikat self-signed untuk seluruh cluster, di-pin menurut hash, sertifikat client wajib.
var tls = TlsOptions.SelfSigned();
await using var worker = await Worker.StartAsync(new WorkerOptions { Token = token, Tls = tls });
await using var client = await Client.ConnectAsync([worker.Address], token, tls);

// Atau biarkan LocalCluster yang mengatur (proses anak menerima kunci lewat .pfx sementara,
// yang dihapus saat dispose).
await using var cluster = await LocalCluster.StartAsync(workers: 4, tls: true);
```

```bash
export DASKNET_TLS_PASSWORD=...
dotnet DaskNet.Worker.dll --port 8786 --tls-pfx cluster.pfx [--tls-trust client.cer]
```

Peer diterima bila hash SHA-256 sertifikatnya ada di `TrustedCertificates` (pinning), atau, bila
tidak ada yang di-pin, bila pemeriksaan rantai dan nama host sistem operasi lolos.
`FromCertificate`/`FromPfx`/`SelfSigned` mem-pin satu sertifikat itu dan mewajibkannya dari client
(mutual TLS). Pengaturan yang tidak cocok langsung gagal dengan alasan: client biasa ke worker TLS
mendapat `TimeoutException` yang menyebut TLS, client TLS ke worker biasa atau sertifikat yang tidak
dipercaya mendapat `AuthenticationException`, dan token yang salah tetap
`UnauthorizedAccessException`. Sebuah tes meneruskan lalu lintas lewat proxy perekam dan memeriksa
bahwa penanda yang dikirim sebagai argumen task terbaca tanpa TLS dan tidak ada dengan TLS.

### Run terlepas: cluster tetap bekerja ketika client pergi

`Compute` menjadikan client scheduler run-nya, sehingga client yang mati kehilangan run-nya.
`SubmitAsync` menyerahkan graph ke sebuah worker, yang menjadwalkannya ke seluruh cluster
(termasuk dirinya) dan menyimpan hasilnya sampai dilupakan:

```csharp
RunHandle run = await client.SubmitAsync(report);          // Delayed, DaskArray, DaskFrame, atau graph
File.WriteAllText("run.json", run.ToString());             // handle-nya JSON biasa

// ... nanti, dari proses mana pun yang memegang token:
await using var other = await Client.ConnectAsync(addresses, token);
var handle = RunHandle.Parse(File.ReadAllText("run.json"));
RunStatus status = await other.StatusAsync(handle);        // Running / Finished / Failed / Cancelled, dengan progres
DataFrame result = await other.FrameResultAsync(handle);   // menunggu sampai selesai
await other.ForgetAsync(handle);                           // melepas hasilnya di worker
```

Hanya graph `[RemoteTask]` yang bisa di-submit - lambda tidak bisa meninggalkan proses, dan
`SubmitAsync` menolaknya dengan `NotSupportedException`.

**Scheduler punya cadangan (standby).** `SubmitAsync` juga mengirim graph ke worker kedua
(`RunHandle.Standby`), yang menanyai worker penjadwal setiap 200 ms. Setelah tiga kali gagal -
atau jawaban bahwa scheduler tidak mengenal run itu, yang berarti ia di-restart - standby
menjalankan graph itu sendiri di semua worker lain dan menyimpan hasilnya. Itu mencakup scheduler
yang mati di tengah run maupun yang mati setelah selesai (hasilnya ikut hilang, jadi dihitung
ulang). `StatusAsync`, `ResultsAsync`, `CancelAsync` dan `ForgetAsync` otomatis beralih ke
standby; cancel dan forget dikirim ke standby *lebih dulu*, karena kalau tidak, standby akan
melihat scheduler melupakan run itu lalu mengambil alih. Kehilangan kedua worker berarti run
hilang, dan cluster satu worker tidak punya standby. Test membunuh scheduler berdasarkan event
(setelah run mengalami kemajuan) dan setelah selesai, lalu memeriksa hasil dan `Worker.Takeovers`.

### Shared memory antar worker di satu mesin

Hasil berukuran minimal `WorkerOptions.SharedMemoryThreshold` (1 MiB secara bawaan) yang
dibutuhkan pembaca di mesin yang sama - peer yang mengambil input task, atau client yang
mengumpulkan hasil - diserahkan lewat shared memory, bukan socket: pemegang meng-encode-nya
langsung ke segmen yang di-map (named mapping di Windows, file di `/dev/shm` di tempat lain) dan
pembaca men-decode langsung dari mapping-nya sendiri. Lokasi yang sama diklaim lewat host id dan
dibuktikan dengan membuka segmen; pembaca yang gagal membukanya kembali ke socket dan berhenti
meminta ke pemegang itu. `Client.SharedMemory = false` menonaktifkannya untuk client,
`--shared-memory-threshold off` untuk worker; `WorkerInfo` melaporkan `SharedMemoryBytesSent` dan
`SocketBytesSent`.

Segmen tetap di-map, dan justru itulah kuncinya. Versi pertama memetakan view baru untuk setiap
transfer dan terukur *lebih lambat* daripada TCP loopback (0,87x pada 64 MiB): setiap transfer
terkena page fault di seluruh segmen pada kedua sisi. Sekarang pemegang menyimpan pool segmen yang
sudah di-map (hingga 256 MiB) dan memakainya ulang setelah pembaca menyatakan selesai, dan pembaca
menyimpan empat segmen terakhirnya tetap di-map. Mengambil satu `NdArray` dari worker, kedua jalur
bergantian dalam satu proses, terbaik dari 32:

| Ukuran | Socket | Shared memory |
|---|---|---|
| 8 MiB | 19,8 ms | 3,7 ms (5,3x) |
| 64 MiB | 124,9 ms | 25,2 ms (5,0x) |

Yang menyeberang adalah encoding wire milik Dask.Net sendiri, bukan Arrow IPC: pembaca tetap
menyalin data ke array baru, jadi penghematannya adalah salinan milik socket, bukan view
zero-copy. Sebuah `Compute` utuh mendapat keuntungan lebih kecil daripada yang tersirat di tabel
ini bila task-nya sendiri yang dominan.

### Eksekusi spekulatif: task yang tertinggal mendapat percobaan kedua

```csharp
client.Speculation = new SpeculationPolicy(Multiplier: 3, Minimum: TimeSpan.FromSeconds(1), MinimumSamples: 3);
```

Bila sebuah task sudah berjalan `Multiplier` kali lebih lama daripada median task selesai dengan
label yang sama (dan setidaknya `Minimum`), dan tidak ada yang menunggu dijalankan, task itu
dijalankan lagi di worker yang menganggur. Percobaan mana pun yang selesai lebih dulu menang;
hasil yang lain dilepas ketika tiba. Ini bentuk work stealing di cluster, untuk mesin yang lambat
atau tetangga yang berisik. Mati secara default, dan hanya untuk task tanpa efek samping, karena
kedua percobaan bisa berjalan sampai akhir. `RunStatistics.Speculated` dan `SpeculationWins`
menghitungnya, dan event `Speculating` dimunculkan. Sebuah test membuat satu task butuh 6 detik
pada percobaan pertama dan 60 ms pada percobaan kedua, dan run-nya selesai di bawah 4 detik.

## Paket tambahan

Empat paket memperluas Dask.Net tanpa menambah dependensi padanya. Masing-masing paket NuGet
terpisah yang kernel-nya method `[RemoteTask]`, sehingga berjalan di cluster begitu assembly-nya
dimuat di worker (`--preload` untuk worker berbentuk proses).

### Lakehouse: Delta Lake dan Apache Iceberg (`Gravicode.Science.DaskNet.Lakehouse`)

```csharp
using Gravicode.Science.DaskNet.Lakehouse;

DaskFrame sales = DeltaLake.Read("lake/sales");                         // versi terbaru
DaskFrame v3 = DeltaLake.Read("lake/sales", version: 3);                // time travel
DaskFrame then = DeltaLake.Read("lake/sales", DateTimeOffset.Parse("2026-01-31T00:00Z"));
IReadOnlyList<DeltaCommit> log = DeltaLake.History("lake/sales");

DaskFrame events = Iceberg.Read("warehouse/db/events");                 // snapshot saat ini
DaskFrame old = Iceberg.Read("warehouse/db/events", snapshotId: 4871234123L);
```

Satu partisi per file data. Nilai partisi diambil dari metadata tabel, sehingga filter pada kolom
partisi memangkas seluruh file sebelum apa pun dibaca, dan hanya kolom yang diperlukan yang dibaca
dari file yang tersisa. **Delta**: log diputar ulang dari checkpoint terbaru (termasuk checkpoint
multi-bagian) ditambah commit JSON sesudahnya; evolusi skema dengan kolom tambahan berfungsi.
**Deletion vector** dibaca (penyimpanan inline, relatif dan absolut; format RoaringBitmapArray
dengan container array, bitmap dan run; checksum diperiksa) dan baris yang dihapus dibuang
*sebelum* filter apa pun yang didorong ke bawah, karena posisinya adalah posisi di dalam file.
Identitas sebuah file adalah path-nya *dan* deletion vector-nya, karena Spark menulis
`add(P, newDV)` dan `remove(P, oldDV)` dalam satu commit. **Column mapping** mode `name` dan `id`:
file dibaca menurut nama fisik atau field id Parquet dan kolom kembali dengan nama logisnya,
termasuk yang diganti nama dan dihapus. Checkpoint v2 (bernama UUID) ditolak dengan menyebut
fiturnya. **Iceberg** (format v1 dan v2): metadata JSON, lalu manifest list dan
manifest Avro; setiap snapshot dibaca dengan skemanya sendiri, kolom yang diganti nama dilacak
lewat field id, dan tabel boleh dipindahkan (path dipasang ulang terhadap lokasi yang tercatat).
File delete ditolak, bukan diabaikan.

```csharp
DaskFrame trips = Hudi.Read("lake/trips");                              // instant terakhir yang selesai
DaskFrame then = Hudi.Read("lake/trips", instant: "20261101093000000");  // time travel
DaskFrame fast = Hudi.Read("lake/events_mor", readOptimized: true);      // file dasar tabel MOR
IReadOnlyList<HudiInstant> timeline = Hudi.Instants("lake/trips");
```

**Hudi**: kedua tata letak timeline (0.x `.hoodie/<instant>.<action>` dengan metadata JSON, 1.x
`timeline/` dengan Avro), hanya instant yang selesai, file dasar terbaru dari setiap file group,
file group yang digantikan `replacecommit` (insert-overwrite, clustering) dibuang, nilai partisi
dari path bergaya hive maupun biasa dengan pemangkasan. Kolom `_hoodie_*` dibuang kecuali
diminta. Pembacaan snapshot atas tabel merge-on-read yang punya log file hidup ditolak -
penggabungan blok log Avro belum diimplementasikan - dan pesannya menyarankan
`readOptimized: true`.

Setiap versi fixture dibandingkan baris demi baris dengan yang dibaca kembali oleh penulis
rujukan: delta-rs dan pyiceberg (`tools/verify/make_lake_fixtures.py`), serta Spark 3.5 dengan
delta-spark 3.3 dan Hudi 1.0 untuk deletion vector, column mapping dan Hudi
(`tools/verify/make_lake_fixtures_spark.py`).

### Geo: spatial join (`Gravicode.Science.DaskNet.Geo`)

```csharp
using Gravicode.Science.DaskNet.Geo;

var nearby = customers.DistanceJoin(stores, radiusKm: 5, ("lat", "lon"), ("lat", "lon"));  // menambah distance_km
var tagged = trips.PointInPolygonJoin(districts, ("lat", "lon"), wktColumn: "geometry", keepUnmatched: true);
var inside = trips.WithinPolygon("lat", "lon", Polygon.Parse("POLYGON ((106.7 -6.4, 107.0 -6.4, 107.0 -6.1, 106.7 -6.1, 106.7 -6.4))"));
var local = trips.SpatialPartition("lat", "lon", partitions: 16);       // baris yang berdekatan berbagi partisi
string cell = Geohash.Encode(-6.2, 106.8, precision: 7);
```

Titik adalah kolom lintang/bujur dalam derajat; poligon berupa WKT (`POLYGON`, `MULTIPOLYGON`,
termasuk lubang; x = bujur). Kedua join mem-broadcast frame sebelah kanan, jadi letakkan sisi
yang lebih kecil di sana. Di dalam task, sisi kanan diindeks - grid atas bounding box poligon,
atau grid 3-D titik pada bola untuk distance join, sehingga antimeridian dan kutub tidak perlu
kasus khusus - dan setiap kandidat diperiksa persis (haversine; titik di batas dihitung di dalam).
`SpatialPartition` mempartisi per rentang geohash, sebuah kurva Z-order. Kedua join dibandingkan
dengan loop ganda brute force, secara lokal dan di cluster.

Operasi geometri pada kolom WKT berjalan di atas NetTopologySuite:

```csharp
var zones = parcels.Buffer("geometry", 0.001)                 // dalam satuan koordinatnya
    .Simplify("geometry", 0.0001)                             // mempertahankan topologi
    .Dissolve("geometry", by: ["district"])                   // union per distrik, dengan hitungan
    .WithMeasures("geometry");                                // kolom luas dan panjang
var shared = parcels.Overlay(floodZones, "geometry", "geometry");                          // irisan
var dry = parcels.Overlay(floodZones, "geometry", "geometry", OverlayKind.Difference);      // sisanya
var centres = parcels.Centroid("geometry", output: "centre");
```

`Dissolve` meng-union grup di setiap partisi dulu lalu menggabungkan union parsialnya berpasangan
dalam sebuah tree, sehingga tidak ada task yang memegang semua geometri. `Overlay` mem-broadcast
layer kanan dan mengindeksnya dengan STR-tree; pasangan yang hanya bersentuhan di sepanjang tepi
dibuang. Aritmetikanya planar: untuk bujur/lintang, buffer 0,001 berarti 0,001 derajat dan luas
dalam derajat persegi, jadi proyeksikan dulu bila jarak harus berarti meter. Input yang tidak
valid diperbaiki dengan `GeometryFixer` (bow-tie menjadi dua segitiganya), bukan ditolak. Semuanya
diperiksa terhadap shapely/GEOS pada 60 poligon berbentuk bintang dengan lubang
(`tools/verify/make_geometry_fixtures.py`): dissolve, overlay, simplify dan centroid cocok hingga
1e-9 relatif, dengan 1, 5 dan 16 partisi dan di cluster. Buffer hanya cocok hingga galat
aproksimasi busur (terburuk 1,8e-4 relatif) - GEOS dan NTS mengaproksimasi sambungan bulat secara
berbeda - dan test membatasi selisihnya dengan luas yang hilang dari lingkaran pada segi-32.

### ML: pencarian hyperparameter dan prediksi batch (`Gravicode.Science.DaskNet.ML`)

```csharp
using Gravicode.Science.DaskNet.ML;

var result = HyperparameterSearch.Grid<KNearestNeighborsClassifier>(
    new Dictionary<string, object?[]> { ["k"] = [1, 5, 15], ["metric"] = [DistanceMetric.Euclidean, DistanceMetric.Manhattan] },
    x, y, folds: 5, scheduler: client);
Console.WriteLine(result.Best.Spec);              // KNearestNeighborsClassifier(k=15, metric=Manhattan)
DataFrame table = result.ToFrame();               // seperti cv_results_ scikit-learn
IEstimator model = result.BestEstimator!;         // di-fit ulang pada seluruh data

var scored = bigFrame.Predict(EstimatorSpec.Of<RandomForestClassifier>(("nTrees", 200), ("seed", 1)),
    trainX, trainY, features: ["x1", "x2"], probabilities: true);   // berjalan di cluster
```

Satu task per (kandidat, fold). Data latih dikirim sekali dan diambil sekali per worker; estimator
dikirim sebagai `EstimatorSpec` - nama tipe dan argumen konstruktornya, dicocokkan menurut nama,
enum menurut namanya - sehingga estimator GraviLearn mana pun bisa dipakai tanpa kode serialisasi.
Fold ditetapkan sebelum apa pun berjalan, sehingga seed yang sama memberi skor yang sama di
scheduler mana pun, dan test memeriksa setiap skor terhadap loop biasa atas fold yang sama.
Scoring: `accuracy`, `f1_macro`, `f1_weighted`, `r2`, `neg_mean_squared_error`,
`neg_root_mean_squared_error`. `Predict` dengan spec mem-fit model sekali per proses worker dan
menilai setiap partisi yang diterimanya (`ParallelPostFit` milik dask-ml), sehingga estimatornya
harus deterministik untuk seed-nya; `Predict` dengan model yang sudah di-fit juga bisa, tetapi
hanya secara lokal.

**Pelatihan inkremental** (`Incremental` milik dask-ml) untuk data yang tidak muat di memori:

```csharp
var model = Incremental.Fit(new SgdClassifier(classes: [0, 1, 2], learningRate: 0.05),
    bigFrame, features: ["x1", "x2", "x3"], target: "label", epochs: 3, seed: 4, scheduler: client);
var scored = bigFrame.Predict(model, ["x1", "x2", "x3"]);            // state-nya ikut dikirim, juga di cluster
```

`SgdRegressor`, `SgdClassifier` (logistik multinomial) dan `IncrementalGaussianNaiveBayes`
mengimplementasikan `IIncrementalModel` (`PartialFit`, ditambah `State()` berisi nilai biasa).
Pelatihan adalah rangkaian task, satu per partisi per epoch, masing-masing mempelajari partisinya
di tempat partisi itu berada lalu meneruskan state-nya; setiap epoch mengunjungi partisi dalam
urutan ber-seed. SGD pada dasarnya berurutan, jadi rangkaiannya tidak paralel: keuntungannya
adalah memori dan lokalitas, bukan kecepatan. Test memeriksa koefisien hasil pelatihan bit demi
bit terhadap loop `PartialFit` biasa atas partisi yang sama dalam urutan yang sama, secara lokal
dan di cluster; naive Bayes menggabungkan momen per kelas dengan rumus Chan dan sama dengan
`GaussianNaiveBayes` batch untuk 1, 4 dan 9 partisi.

**Scoring ONNX** menjalankan model ONNX apa pun atas sebuah frame, satu sesi ONNX Runtime per
proses worker:

```csharp
byte[] onnx = OnnxExport.Build(pipeline, features: 4);     // atau File.ReadAllBytes("model.onnx") dari skl2onnx, PyTorch...
var scored = bigFrame.ScoreOnnx(onnx, ["x0", "x1", "x2", "x3"], columns: ["score"]);
```

Input pertama model menerima kolom fitur (sebagai float atau double, sesuai deklarasinya); output
yang dipilih menjadi satu kolom, atau kolom `{name}_{i}` bila lebih lebar. Byte model adalah satu
nilai di graph, diambil sekali per worker, dan sesi di-cache menurut SHA-256. Diperiksa terhadap
pipeline GraviLearn asal ekspornya (hingga presisi float32), secara lokal dan di cluster, dan
terhadap model yang ditulis paket `onnx` Python yang outputnya diketahui dalam bentuk tertutup.

### Streaming: micro-batch, window, ETL yang bisa dilanjutkan (`Gravicode.Science.DaskNet.Streaming`)

```csharp
using Gravicode.Science.DaskNet.Streaming;

var query = DaskStream.FromDirectory("inbox", watch: true)            // setiap file CSV baru, sekali
    .Filter(Col("value") >= 0)
    .WithScheduler(client).Partitions(4)                               // setiap batch di cluster
    .Window("time", TimeSpan.FromMinutes(10), keys: ["sensor"],
            [(null, "count"), ("value", "mean"), ("value", "max")],
            allowedLateness: TimeSpan.FromMinutes(2))
    .ToCsvDirectory("out");
await query.RunAsync(checkpointDirectory: "ckpt", cancellationToken);
```

Sumber: `IAsyncEnumerable<DataFrame>` apa pun, sebuah `Channel`, atau direktori file inkremental.
Langkah tanpa state (`Filter`, `WithColumns`, `Select`, `Sql` per batch) berjalan di scheduler
query; `Map` berjalan di dalam proses. `Window` mengagregasi menurut waktu kejadian (tumbling,
atau sliding dengan `slide:`), diselaraskan ke epoch Unix. Sebuah window dikeluarkan ketika
watermark - waktu kejadian terbaru dikurangi keterlambatan yang diizinkan - melewati akhirnya;
baris yang tiba untuk window yang sudah dikeluarkan dibuang dan dihitung (`LateRows`). Dengan
direktori checkpoint, file yang sudah dibaca, jumlah batch dan window yang masih terbuka disimpan
setelah setiap batch, sehingga query yang dijalankan ulang hanya membaca file baru dan membawa
window-nya; sebuah test menghentikan query setelah empat file, menambah dua lagi, menjalankannya
ulang, dan mendapat window yang sama dengan satu run tanpa jeda. Pengiriman at-least-once; sink
CSV menamai file menurut id batch, sehingga batch yang diputar ulang menimpa file-nya sendiri.

**Message broker.** Pesan MQTT dan Kafka menjadi baris lewat `MessageFormat` - objek JSON dengan
skema bertipe, atau baris CSV:

```csharp
var format = MessageFormat.Json(("sensor", ColumnType.Text), ("time", ColumnType.DateTime), ("value", ColumnType.Number));
var mqtt = DaskStream.FromMqtt("broker.local", ["plant/+/temperature"], format,
    new MqttOptions { MaxBatchRows = 500, MaxBatchDelay = TimeSpan.FromSeconds(1) });
var kafka = DaskStream.FromKafka("kafka:9092", ["orders"], groupId: "etl", format,
    new KafkaOptions { KeyColumn = "key", OffsetColumn = "offset" });
```

Sebuah batch ditutup pada `MaxBatchRows` atau setelah `MaxBatchDelay`; `IdleTimeout` mengakhiri
stream yang sudah sepi, sehingga window-nya dikeluarkan. Tipe setiap kolom dipatok oleh skema
(atau oleh nilai pertamanya), sehingga skema batch tidak pernah berubah; nilai yang tidak cocok
menjadi kosong. **Kafka** menyimpan offset per partisi di checkpoint query dan meng-commit-nya ke
group hanya setelah sink selesai: at-least-once, dan restart melanjutkan dari akhir batch terakhir
yang selesai. Diuji terhadap model broker di memori (sink yang gagal di tengah jalan memutar ulang
tepat batch yang gagal) dan terhadap Apache Kafka 4.3 mode KRaft (test yang berjalan bila
`DASKNET_KAFKA` di-set). **MQTT** tidak punya replay: pesan di-acknowledge saat diterima, dan hanya
client id tetap dengan `CleanSession = false` yang membuat broker mengantrekan pesan selama query
mati. Diuji ujung ke ujung terhadap broker in-process milik MQTTnet.

**Join stream-stream** mencocokkan baris dua stream menurut key dan waktu kejadian:

```csharp
var attributed = clicks.IntervalJoin(impressions, on: ["ad"], leftTime: "clicked", rightTime: "shown",
        lower: TimeSpan.FromHours(-1), upper: TimeSpan.Zero, how: StreamJoinType.Left)
    .Window("clicked", TimeSpan.FromMinutes(10), keys: ["ad"], [(null, "count")])
    .ToCsvDirectory("out");
```

`Join(..., within:)` adalah bentuk simetrisnya; `how` berupa inner, left, right atau outer. Kedua
sisi disimpan sebagai state, dan sebuah baris dikeluarkan dari state begitu watermark sisi *lain*
melewati waktu terakhir ia masih bisa cocok - itulah yang membatasi memori, dan sebabnya hasil
pencocokan tidak bergantung pada bagaimana batch kedua stream berselang-seling. Sisi outer
mengeluarkan baris yang tak cocok saat dikeluarkan dari state. State disimpan di checkpoint
seperti window. Inner join diperiksa terhadap join brute force atas semua baris dengan kedatangan
kiri-dulu, kanan-dulu, bergantian dan acak serta beberapa ukuran batch; state tetap di bawah 400
baris dengan 8.000 baris mengalir.

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
| JSON, Excel, Delta Lake, Iceberg, Hudi | JSON (Lines dan array, baca dan tulis) **selesai**; Delta Lake (deletion vector, column mapping), Iceberg dan Hudi (copy-on-write, merge-on-read read-optimized) **selesai** di paket Lakehouse, dengan time travel dan pemangkasan partisi; Excel lewat `ReadFiles` dengan `ExcelReader` milik GraviFrame |
| Mesin lazy: eksekusi tertunda, planner, pipeline fusion | **Selesai**; strategi join berbasis biaya (estimasi ukuran) dan rangkaian join SQL diurutkan ulang menurut estimasi ukuran; rewrite lain berbasis aturan |
| DAG scheduler: visualisasi (DOT/Mermaid), cull/fuse, retry policy, pelacakan dependensi | **Selesai** |
| Cluster lokal: multi-core, isolasi proses, batas memori per worker dengan spill ke disk | **Selesai** |
| Cluster multi-node lewat TCP | **Selesai**; transport gRPC/QUIC/HTTP/3 masih roadmap |
| Penemuan worker (Consul, Kubernetes, Service Fabric, Aspire) | **Selesai** untuk Kubernetes (DNS Service headless) dan Aspire, dengan refresh di latar belakang sehingga worker bisa bergabung ke client yang sedang berjalan; Consul/Service Fabric lewat `WorkerDiscovery.From` |
| Parallel LINQ: map, reduce, fold, aggregate, shuffle, broadcast | **Selesai** (`Bag`) |
| Koleksi terdistribusi (`DistributedList` dsb.) | Roadmap; `Bag`, `DaskArray` dan `Persist` menangani kasus yang dominan baca |
| Arrow: zero-copy, lintas bahasa | Berkas Arrow IPC **selesai**; hasil berpindah antar proses di satu mesin lewat shared memory (encoding wire Dask.Net, bukan Arrow; 5x lebih cepat daripada TCP loopback) |
| SQL: join, agregasi, UNION, window function | **Selesai**; frame window eksplisit, CTE dan ANSI penuh masih roadmap |
| GIS, ML, ETL/streaming, AI, basis data vektor | GIS (Geo: geohash, distance join dan point-in-polygon join, buffer/simplify/dissolve/overlay), ML (pencarian hyperparameter, prediksi batch, pelatihan inkremental, scoring ONNX) dan streaming (file, channel, MQTT dan Kafka; window waktu kejadian; join stream-stream; checkpoint) **selesai** sebagai paket terpisah; AI dan basis data vektor masih roadmap |
| Observabilitas: OpenTelemetry | **Selesai** (ActivitySource + Meter); Prometheus/Grafana/Jaeger lewat exporter OTel |
| Operator Kubernetes, dashboard Aspire | Roadmap |
| Toleransi kegagalan: retry task, checkpoint, pemulihan worker, eksekusi spekulatif | **Selesai**; run bisa dilepas dari client (`SubmitAsync`), dan worker standby mengambil alih run bila worker penjadwalnya mati |
| Konektor penyimpanan (basis data, data lake, warehouse) | Basis data **selesai** lewat provider ADO.NET apa pun (dipartisi per rentang atau per query, filter didorong ke SQL, bisa jalan di cluster); Delta Lake dan Iceberg **selesai**; warehouse masih roadmap |
| Scheduler: thread, cluster | **Selesai**; pool proses adalah `LocalCluster.StartProcessesAsync`; Kubernetes masih roadmap |
| Keamanan: worker dengan autentikasi token, TLS dan mutual TLS | **Selesai**; OAuth2/Entra ID, multi-tenancy dan kuota masih roadmap |

## Batasan yang diketahui

- Penyusunan graph untuk `SortBy` langsung menjalankan pengambilan sampel, yang membaca input
  sekali lagi.
- `MapPartitions`, `Bag.Map` dengan lambda dan `DaskArray.MapBlocks` hanya untuk scheduler
  lokal; tulis method `[RemoteTask]` untuk menjalankan kode kustom di cluster.
- Join dengan beberapa key membutuhkan nama key yang sama di kedua sisi (SQL menangani nama yang
  berbeda).
- Dengan `Compute`, client adalah scheduler run-nya, dan client yang mati kehilangan run itu;
  `SubmitAsync` memindahkan peran itu ke sebuah worker dengan standby, sehingga run bertahan bila
  salah satunya mati, tetapi tidak bila keduanya mati.
- Tabel Hudi merge-on-read dengan log file hidup hanya dibaca sebagai `readOptimized`; checkpoint
  Delta v2 dan tipe bersarang ditolak.
- Aritmetika geometri bersifat planar; proyeksikan bujur/lintang sebelum mem-buffer dengan jarak.
- Pengiriman Kafka dan MQTT bersifat at-least-once; MQTT tidak punya replay. Join stream hanya
  mencocokkan key yang sama.
- Pengurutan ulang join hanya berlaku untuk rangkaian inner join SQL, bukan untuk pemanggilan
  `DaskFrame.Join`, yang berjalan sesuai urutan tertulis.

---

Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil
