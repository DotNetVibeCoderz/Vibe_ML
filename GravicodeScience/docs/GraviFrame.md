# GraviFrame

*[Bahasa Indonesia](id/GraviFrame.md)* · pandas for .NET — typed columns, group-by, joins and time series.

## The model

A `DataFrame` is an ordered list of `Series`, each backed by a single typed array. That columnar
layout is why column statistics and filters are fast: each one touches a contiguous buffer instead
of striding across row objects. Row-wise operations (filter, sort, join) build an index vector and
then ask every column to `Take` it — one pass per column, not per cell.

Frames are **immutable**. Every transformation returns a new frame; the underlying column arrays
are shared where possible.

## Column types

| Type | Backing store | Missing value |
|---|---|---|
| `NumericSeries` | `double[]` | `double.NaN` |
| `TextSeries` | `string?[]` | `null` |
| `BooleanSeries` | `bool?[]` | `null` |
| `DateTimeSeries` | `DateTime?[]` | `null` |

Using NaN as the numeric missing marker is what lets column arithmetic go straight to GraviNum's
SIMD kernels — no parallel null mask to check inside the hot loop, and IEEE rules propagate
missing-ness for free.

## Reading data

```csharp
var df = DataFrame.ReadCsv("data.csv");
var df2 = DataFrame.ParseCsv(csvString);
var big = DataFrame.ReadCsvMemoryMapped("huge.csv");   // when the file may not fit in RAM

ParquetIO.Write(df, "data.parquet");
var back = ParquetIO.Read("data.parquet");
var subset = ParquetIO.Read("data.parquet", ["value", "date"]);   // columns only
```

Types are inferred per column from the first 1,000 rows by default. Override when needed:

```csharp
var df = DataFrame.ReadCsv("data.csv", new CsvOptions
{
    Delimiter = ";",
    HasHeader = true,
    MissingTokens = { "N/A", "-", "?" },
    ColumnTypes = { ["zip"] = DataType.Text },   // stop a numeric-looking ID becoming a number
    MaxRows = 100_000,
});
```

Quoted fields, embedded delimiters and doubled escape quotes are handled.

## Inspecting

```csharp
df.Shape;              // (rows, columns)
df.ColumnNames;
Console.WriteLine(df);            // fixed-width table, first 10 rows
Console.WriteLine(df.ToString(50));
Console.WriteLine(df.Info());     // types and missing counts per column
Console.WriteLine(df.Describe()); // count, mean, std, quartiles per numeric column
```

## Selecting and filtering

```csharp
df["age"];                        // Series
df.Numeric("age");                // NumericSeries — throws with a clear message on a type mismatch
df.Text("name");  df.DateTimes("date");  df.Booleans("active");

df.Head(10);  df.Tail(10);  df.Rows(100, 50);  df.Sample(20, seed: 42);
df.SelectColumns("age", "fare");
df.Drop("deck", "embarked");
df.Rename("fare", "ticket_price");

df.FilterBy("age", v => v > 30);
df.Filter(row => row.String("sex") == "female" && row.Number("age") > 30);
df.Filter(df.Numeric("fare").Where(v => v > 100));   // boolean mask

df.SortBy("fare", ascending: false);
df.SortBy([("pclass", true), ("fare", false)]);      // lexicographic
```

## Adding columns

```csharp
df.WithColumn(new NumericSeries("ratio", values));
df.WithColumn("price_per_person", i => df.Numeric("fare")[i] / df.Numeric("size")[i]);
df.WithColumn(df.Numeric("age").Standardize().Rename("age_z"));
```

## Missing values

```csharp
df.MissingCounts();                   // per column
df.DropMissing();                     // any missing value in the row
df.DropMissing("all");                // only fully missing rows
df.DropMissing(subset: ["age"]);
df.FillMissing(0.0);
df.FillMissingWithMean();

var age = df.Numeric("age");
age.FillMissingWithMedian();
age.ForwardFill();  age.BackwardFill();
age.Interpolate();                    // linear between surrounding present values
```

## GroupBy

Grouping happens once, eagerly, so `Mean`, `Sum` and `Count` on the same `GroupBy` each cost one
pass over the values rather than re-bucketing the frame. Groups keep first-seen order.

```csharp
df.GroupBy("category").Mean("sales");
df.GroupBy("category", "region").Sum("sales");    // composite key
df.GroupBy("category").Count("n");

df.GroupBy("category").Aggregate("median", "sales");
df.GroupBy("category").Aggregate("sales", v => v.Max() - v.Min(), "range");
df.GroupBy("category").AggregateMany([("sales", "sum"), ("sales", "mean"), ("units", "max")]);
```

Named aggregates: `mean`, `sum`, `min`, `max`, `median`, `std`, `var`, `count`, `first`, `last`.

## Reshaping

```csharp
df.Pivot("date", "category", "sales");                  // long -> wide
df.Pivot("date", "category", "sales", aggregate: "sum");
df.Melt(["date"], ["a", "b"], "variable", "value");     // wide -> long

Reshaping.PivotTable(df, "date", "category", ["sales", "units"]);
Reshaping.OneHot(df, "category");                       // one indicator column per category
```

Missing combinations in a pivot come back as NaN rather than raising, which is what makes it
usable on ragged real data.

## Joining

The right side is hashed once and the left side scanned, so cost is linear rather than quadratic.
Duplicate keys on the right fan out into multiple rows, matching SQL and pandas.

```csharp
left.Join(right, "id");                              // inner
left.Join(right, "id", JoinKind.Left);               // Left, Right, Outer
left.Merge(right, leftOn: "user_id", rightOn: "id");
Joins.MergeOn(left, right, ["date", "region"]);      // several keys

DataFrame.Concat([a, b, c]);                         // stack rows
```

Colliding column names take a `_right` suffix.

## Time series

```csharp
var close = df.Numeric("close");

close.Shift(2);                    close.Diff();
close.PercentChange();             close.CumulativeSum();

close.Rolling(window: 7).Mean();   // incremental — O(1) per row regardless of width
close.Rolling(7).Sum();  .Std();  .Min();  .Max();
close.Rolling(7).Median();         // re-sorts each window; slower on wide windows
close.Rolling(7, minPeriods: 1).Mean();
close.Rolling(20).Apply(v => v.Max() - v.Min(), "range");

close.Expanding().Mean();          // cumulative
close.ExponentialMovingAverageBySpan(20);
```

### Resampling

Buckets come from the calendar, not from fixed tick counts, so a monthly resample lands on real
month boundaries regardless of month length or leap years.

```csharp
Resampling.Resample(df, "date", ResampleFrequency.Monthly, "mean");
Resampling.Resample(df, "date", ResampleFrequency.Weekly, "sum", ["sales"]);
```

Frequencies: `Hourly`, `Daily`, `Weekly`, `Monthly`, `Quarterly`, `Yearly`.

## Analytics and interop

```csharp
df.Describe();
df.CorrelationMatrix();

df.ToNdArray();                              // every numeric column
df.ToNdArray("age", "income", "score");      // named columns, in order
NumericSeries.FromNdArray("x", array);
DataFrame.FromMatrix(matrix, ["a", "b", "c"]);

var (codes, categories) = df.Text("category").Factorize();   // text -> integer codes
```

`ToNdArray` is the bridge into GraviLearn:

```csharp
var features = df.ToNdArray("age", "income");
var labels = df.Numeric("churn").ToNdArray();
model.Fit(features, labels);
```

## Common mistakes

| Symptom | Cause |
|---|---|
| `InvalidOperationException` on `Numeric(...)` | The column was inferred as text — check `Info()` |
| An ID column became a number | Force it with `ColumnTypes = { ["id"] = DataType.Text }` |
| A transformation seems to do nothing | Frames are immutable; use the returned value |
| Rolling median is slow | It re-sorts each window; use `Mean` where it will do |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
