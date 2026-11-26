# Dask.Net

*[Bahasa Indonesia](id/DaskNet.md)* · Parallel and distributed computing for .NET: lazy task
graphs, chunked arrays, partitioned DataFrames with a query optimiser, SQL, and a local or
multi-machine cluster that recovers from losing a worker.

```bash
dotnet add package Gravicode.Science.DaskNet
```

The package is `Gravicode.Science.DaskNet`, the namespace `Gravicode.Science.DaskNet`, and it
builds on GraviNum (array chunks) and GraviFrame (DataFrame partitions).

## The model

Everything is a **task graph**. A collection — `Delayed<T>`, `Bag<T>`, `DaskArray`,
`DaskFrame` — is a recipe for a graph: building one does no work. `Compute` hands the graph to a
**scheduler**, which runs the tasks in dependency order and returns the result.

Before scheduling, the graph is **culled** (tasks nobody needs are dropped) and **fused**: a
chain of tasks where each feeds exactly one other becomes one task, which removes a scheduling
round-trip and an intermediate result per link.

| Scheduler | Runs tasks on | Use for |
|---|---|---|
| `SynchronousScheduler` | the calling thread, in priority order | debugging, profiling, deterministic stack traces |
| `ThreadScheduler` (default) | the thread pool | anything that fits on one machine |
| `Client` (from `LocalCluster` or `Client.ConnectAsync`) | worker processes over TCP | isolation, more than one machine |

```csharp
using Gravicode.Science.DaskNet;

var x = Dask.Value(3.0);
var y = Dask.Delayed(Square, x);          // nothing runs yet
y.Compute();                              // default scheduler
y.Compute(new SynchronousScheduler());    // explicit

Dask.DefaultScheduler = new ThreadScheduler(maxWorkers: 4);
using (Dask.UseScheduler(new SynchronousScheduler())) { y.Compute(); }   // scoped, async-flowing
```

The local schedulers release a result as soon as its last reader has finished, so a graph with
thousands of partitions does not keep them all in memory. `RunStatistics.PeakResultsHeld`
reports the high-water mark. Tasks are prioritised depth-first, as Dask does, so a reduction
consumes partitions as they are produced instead of materialising every leaf first.

**Retries** are per scheduler: `new ThreadScheduler(retry: new RetryPolicy(MaxRetries: 3,
Delay: TimeSpan.FromMilliseconds(100)))`. A task that still fails raises `DaskTaskException`
naming the key and label of the task, with the original exception inside.

### Visualising a graph

```csharp
var graph = Dask.Graph(total);
graph.ToMermaid();               // paste into any Markdown renderer
graph.ToDot();                   // Graphviz
graph.ToMermaid(highLevel: true) // one node per layer (map, reduce, shuffle...) instead of per task
```

## Delayed (`dask.delayed`)

```csharp
var parts = files.Select(f => Dask.Delayed(Load, Dask.Value(f))).ToList();
var cleaned = parts.Select(p => p.Then(Clean));
var total = Dask.All(cleaned).Then(xs => xs.Sum());
var (sum, count) = Dask.Compute(total, Dask.All(parts).Then(p => p.Length));   // one graph, shared work runs once
```

A **lambda** runs only on local schedulers. To run on a cluster, pass a **method group** of a
public static method marked `[RemoteTask]` — the method's name is what crosses the wire, never
code:

```csharp
public static class Tasks
{
    [RemoteTask] public static double Square(double x) => x * x;
}
var y = Dask.Delayed(Tasks.Square, x);   // remotable
```

## Bag (`dask.bag`)

A partitioned sequence of any `T`, for semi-structured data — log lines, JSON records, text.

```csharp
var words = Bag.ReadText("logs/*.txt", blockSize: 64 << 20)
    .FlatMap(line => line.Split(' '))
    .Filter(w => w.Length > 3);

words.Count().Compute();
words.Frequencies().Compute();              // counted per partition, then merged
words.TopK(10, w => w.Length).Compute();
words.FoldBy(w => w[0], () => 0, (n, _) => n + 1, (a, b) => a + b).Compute();
words.GroupBy(w => w[0]).Compute();         // full shuffle - FoldBy is cheaper when it fits
words.Distinct().Repartition(4);

// LINQ query syntax works:
var q = from line in Bag.ReadText("data.txt") where line.Length > 0 select line.Trim();
```

`ReadText` splits files into byte ranges; a line belongs to the block its first byte falls in,
so blocks can be read independently without a line being counted twice or lost. Numeric
reductions (`Mean`, `Var`, `Std`) merge Welford moments with Chan's formula, which stays accurate
where a naive sum of squares cancels (values around 1e9 with small spread).

## Array (`dask.array`)

A grid of GraviNum `NdArray` chunks.

```csharp
var x = DaskArray.RandomNormal([20_000, 5_000], chunks: [2_000, 1_000], seed: 7);
var gram = x.T.Dot(x) / x.Shape[0];
NdArray result = gram.Compute();

x.Std().Compute();                 // full reduction
x.Sum(axis: 1).Compute();          // axis reduction, tree-combined
x.Rechunk([5_000, 5_000]);         // re-tile
x.MapBlocks(b => b * 2);           // any NdArray function, per block (local schedulers only)
```

Element-wise operations need equal shapes (there is no broadcasting yet); arrays with different
chunkings are rechunked to a common grid first.
`Random` seeds each block from `(seed, block index)`, so a result does not depend on how many
threads ran it. Every array kernel is a `[RemoteTask]`, so array graphs run on a cluster as is.

## DataFrame (`dask.dataframe`)

A `DaskFrame` is a **logical plan** over partitions, each a GraviFrame `DataFrame`. Row-wise
operations take **expressions** rather than lambdas:

```csharp
using static Gravicode.Science.DaskNet.Expr;

var sales = DaskFrame.ReadCsv("sales/*.csv", blockSize: 64 << 20);
var report = sales
    .Filter(Col("region").In("north", "east") & (Col("price") > 20))
    .WithColumn("revenue", Col("price") * Col("qty"))
    .Join(DaskFrame.ReadCsv("products.csv"), "product")
    .GroupBy("region", "category").Agg(("revenue", "sum"), ("price", "mean"))
    .SortBy("revenue_sum", ascending: false);

Console.WriteLine(report.Explain());     // the optimised plan
DataFrame result = report.Compute();
```

Expressions are the reason the optimiser exists: a predicate that names only `region` can move
below a join and into the CSV scan; a projection tells the scan which columns to parse. A
`Func<RowView, bool>` is opaque to both. Expressions also serialise, which is what lets a plan run
on a cluster. Missing values follow SQL three-valued logic: `NULL > 5` is unknown, and a filter
keeps only rows where the predicate is true.

**What the optimiser does**

- **Predicate pushdown** through projections, renames, new columns (when the predicate does not
  read them), sorts, repartitions, concatenations, aggregations (predicates on group keys) and
  joins. On a left join, only predicates on the left side move down; pushing a right-side
  predicate below a left join would turn missing matches into dropped rows.
- **Projection pushdown**: the scan parses only the columns some later step reads.
- **Pipeline fusion**: all row-wise steps between two shuffles run in one task per partition.

Compare `Explain(optimized: false)` with `Explain()` to see what moved.

**Execution strategies**

| Operation | Strategy |
|---|---|
| `GroupBy().Agg` | partial aggregate per partition, tree-combined (`DefaultSplitEvery`, default 8), finished once. Mean/var/std carry (count, mean, M2) states. |
| `Join` / `Merge` | **broadcast** when the right side is one partition (inner and left joins); otherwise a **hash shuffle** so matching keys meet. Hashing is FNV-1a over a type-tagged key, because `string.GetHashCode` differs between processes. |
| `SortBy` | range partitioning on boundaries sampled from the data, so concatenating partitions in order gives a sorted frame. |
| `Repartition`, `Limit`, `Concat` | merge/split neighbouring partitions; no shuffle. |

Aggregations: `sum`, `mean`, `count`, `size`, `min`, `max`, `var`, `std` (sample, ddof = 1),
`var_pop`, `std_pop`.

**Files.** `ReadCsv` splits large files into byte blocks and pins column types from a sample,
so every block parses a column the same way. `ReadArrow` reads Arrow IPC files, one partition
per file. `ReadFiles(pattern, reader)` takes any reader — `ReadFiles("*.parquet", ParquetIO.Read)`
with the GraviFrame.Parquet package. `ToCsv(dir)` and `ToArrow(dir)` write one file per
partition, writing to `.partial` and renaming so a crashed run leaves no half-written part.

**Checkpointing.** `frame.Checkpoint("ckpt/")` writes every partition to Arrow and returns a
frame that reads from there; partitions already present are not recomputed, so a long pipeline
that is re-run after a crash resumes where it stopped. `Persist()` does the same in memory.

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

DaskFrame lazy = sql.Sql("SELECT ...");    // compose further, or Compute on a cluster
sql.Explain("SELECT ...");                 // the optimised plan
```

A query compiles to an ordinary `DaskFrame` plan, so it gets the same optimiser and runs on any
scheduler.

Supported: `SELECT [DISTINCT]`, `*` and `t.*`, aliases, `FROM` tables and subqueries,
`[INNER|LEFT|RIGHT|FULL] JOIN ... ON` (equalities between the two sides, plus any extra
conditions on inner joins), `WHERE`, `GROUP BY` (expressions, output aliases and positions),
`HAVING`, `ORDER BY` (outputs, positions or any expression), `LIMIT`. Expressions: arithmetic,
`||`, comparisons, `AND/OR/NOT`, `IS [NOT] NULL`, `[NOT] IN (...)`, `[NOT] BETWEEN`,
`[NOT] LIKE`, `CASE` (both forms), `CAST`, `DATE '...'`. Aggregates: `COUNT(*)`, `COUNT`, `SUM`,
`AVG`, `MIN`, `MAX`, `STDDEV[_SAMP|_POP]`, `VARIANCE`/`VAR_SAMP`/`VAR_POP`. Functions: `ABS`,
`SQRT`, `LN`/`LOG`, `EXP`, `POWER`, `ROUND`, `FLOOR`, `CEIL`, `LOWER`, `UPPER`, `TRIM`, `LENGTH`,
`COALESCE`, `IFNULL`, `NULLIF`, `CONCAT`, `YEAR`, `MONTH`, `DAY`.

After a join, every column is addressable as `alias.column`; an unqualified name works when only
one table has it. With `SELECT *`, clashing names come out qualified (`s.product`,
`p.product`); a left join's unmatched rows have the right side's key **missing**, as SQL says,
not copied from the left.

Refused with a `SqlException` that names the position, rather than approximated: `OFFSET`,
`UNION`, `CROSS JOIN`, `COUNT(DISTINCT ...)`, subqueries outside `FROM`, window functions, CTEs,
and joins with no equality between the sides.

## Distributed execution (`dask.distributed`)

```csharp
using Gravicode.Science.DaskNet.Distributed;

// Workers in this process - still TCP and serialised, so what runs here runs on a real cluster.
await using var cluster = await LocalCluster.StartAsync(workers: 4, threadsPerWorker: 2);
var result = report.Compute(cluster.Client);

// Workers as child processes (isolation; a crash in a task does not take the client down).
await using var procs = await LocalCluster.StartProcessesAsync("DaskNet.Worker.dll", workers: 4,
    preload: ["MyTasks.dll"]);
```

On other machines, run the worker host:

```bash
export DASKNET_TOKEN=<at least 16 characters, shared by the cluster>
dotnet DaskNet.Worker.dll --host 0.0.0.0 --port 8786 --threads 8 --name node-1 --preload MyTasks.dll
```

and connect: `await Client.ConnectAsync(["node-1:8786", "node-2:8786"], token)`.

**What crosses the wire.** Tasks are a method reference plus arguments. Only `public static`
methods marked `[RemoteTask]` can be named — a worker refuses anything else, so a client can't
make it run `File.Delete`. Values use a closed binary codec (numbers, strings, arrays, `NdArray`,
`DataFrame` with missing values and categoricals, task references); `WireCodec.Register` adds a
type. Graphs that contain lambdas are rejected with `NotSupportedException` before anything is
sent.

**Scheduling.** The client places each task on the worker that already holds the most of its
inputs, breaking ties by the fewest tasks in flight. Workers fetch missing inputs from each other directly,
not through the client, and the client releases results once nothing needs them.

**Fault tolerance.** When a worker disappears, the client marks every result it held as lost and
re-plans: tasks whose inputs are gone are recomputed from the graph's lineage, then the run
continues on the remaining workers. Retries cover tasks that throw. Both are tested by killing a
worker (in-process and as a real process) halfway through a run.

**Security.** Connections start with an HMAC-SHA256 challenge on the shared token, so only
holders of the token can submit tasks. Traffic is **not encrypted** — keep the cluster on a
trusted network or tunnel it; TLS is on the roadmap.

## Observability

Every task is an `Activity` on the `ActivitySource` named `Gravicode.Science.DaskNet`, and the
`Meter` of the same name publishes `dask.tasks.completed`, `dask.tasks.failed`,
`dask.tasks.retried` and the `dask.task.duration` histogram. Wire them to OpenTelemetry the
usual way (`AddSource(DaskTelemetry.Name)`, `AddMeter(DaskTelemetry.Name)`). For a quick look,
schedulers and `Client` accept an `OnEvent` callback (`Started`, `Finished`, `Retrying`,
`Failed`, `Recomputing`), and `LastRun` holds `RunStatistics` (tasks per worker, retries, peak
results held).

## Performance

BenchmarkDotNet, `benchmarks/DaskNet.Benchmark`, Release, the 4-core 15 W laptop described in
[benchmarks.md](benchmarks.md), managed path (no native BLAS). Ratios within one run; the machine
throttles, so absolute times move between runs.

| Benchmark | Result |
|---|---|
| Scheduler overhead | ~5 µs per task synchronous, ~6–9 µs on the thread pool (100–1000 trivial tasks). Aim for tasks of ≥ 1 ms, as with Dask. |
| `Std` of 2000² / 4000² in 16 chunks | thread scheduler **0.61× / 0.50×** the time of GraviNum on the whole array; synchronous 1.39× / 1.20× |
| Filter + group-by, 200k / 1M rows | GraviFrame on one frame is faster at this size (Dask.Net 3.2× / 1.55× the time, with high variance). Partitioning pays off when the data does not fit, or with a cluster. |
| Array reduction, in-process cluster vs threads | 1.12× — the floor of what loopback TCP and serialisation cost |

The honest summary: when the data fits in memory and one library call is already parallel, add
Dask.Net for the data that doesn't, not for speed.

## Coverage of the feature list

`Dask.Net-Features.md` describes a much larger product. This is what exists today.

| Area | Status |
|---|---|
| Partitioned, lazy DataFrame; query optimisation; predicate and projection pushdown | **Done** |
| CSV (block-split), Arrow IPC, any file via `ReadFiles` (Parquet through GraviFrame.Parquet) | **Done** |
| JSON, Excel, Delta Lake, Iceberg, Hudi | Roadmap (`ReadFiles` with a reader works today) |
| Lazy engine: deferred execution, planner, pipeline fusion | **Done**; cost-based optimisation is roadmap (rules only) |
| DAG scheduler: visualisation (DOT/Mermaid), cull/fuse, retry policy, dependency tracking | **Done** |
| Local cluster: multi-core, process isolation | **Done**; per-worker memory limits are roadmap |
| Multi-node cluster over TCP | **Done**; gRPC/QUIC/HTTP/3 transports are roadmap |
| Worker discovery (Consul, Kubernetes, Service Fabric, Aspire) | Roadmap — addresses are passed explicitly |
| Parallel LINQ: map, reduce, fold, aggregate, shuffle, broadcast | **Done** (`Bag`) |
| Distributed collections (`DistributedList` etc.) | Roadmap; `Bag`, `DaskArray` and `Persist` cover the read-mostly cases |
| Arrow: zero-copy, cross-language | Arrow IPC files **done**; shared memory between processes is roadmap |
| SQL: joins, aggregations | **Done**; window functions, CTE and full ANSI are roadmap |
| GIS, ML, ETL/streaming, AI, vector databases | Roadmap (separate modules) |
| Observability: OpenTelemetry | **Done** (ActivitySource + Meter); Prometheus/Grafana/Jaeger via the OTel exporters |
| Kubernetes operator, Aspire dashboard | Roadmap |
| Fault tolerance: task retry, checkpointing, worker recovery | **Done**; speculative execution is roadmap |
| Storage connectors (databases, data lakes, warehouses) | Roadmap |
| Schedulers: thread, cluster | **Done**; a process pool is `LocalCluster.StartProcessesAsync`; Kubernetes is roadmap |
| Security: token-authenticated workers | **Done**; TLS/mTLS/OAuth2/Entra ID, multi-tenancy and quotas are roadmap |

## Known limits

- Graph construction for `SortBy` runs a sampling pass immediately, which reads the input once.
- `MapPartitions`, `Bag.Map` with lambdas and `DaskArray.MapBlocks` are local-only; write a
  `[RemoteTask]` method to run custom code on a cluster.
- Joins on several keys need the same key names on both sides (SQL handles different names).
- The client is the single scheduler of a run; if the client dies, the run is lost (workers
  survive and accept the next client).

---

Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil
