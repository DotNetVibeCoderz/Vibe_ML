# GraviDatasets

**Dataset loading and shaping, over GraviFrame.**

Mirrors `datasets`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviDatasets;
```

## Loading

```csharp
Dataset.Load("titanic");                    // built in, works offline
Dataset.Load("stanfordnlp/imdb", "train");  // anything with a slash is a Hub id
Dataset.LoadAll("stanfordnlp/imdb");        // every split

Dataset.FromFile("data.csv");               // .csv .tsv .parquet .json .jsonl
Dataset.FromCsvMemoryMapped("huge.csv");
Dataset.FromFrame(dataFrame, "name");
```

A bare name is looked up in the built-in registry **first**, and only then on the Hub. That ordering
is what keeps `Dataset.Load("titanic")` working with no network and no credentials.

Built in: `titanic` `iris` `imdb` `finance` `sms_spam`.

## Shaping

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

**Every operation returns a new dataset.** Datasets are passed between preprocessing steps, and a
mutating `Shuffle` would silently change what an earlier split had already taken - the kind of leak
that inflates a validation score without producing any visible error.

Where an operation only selects rows, the underlying frame is shared rather than copied, so taking a
hundred splits of a large dataset does not multiply its memory.

## Splitting

```csharp
var split = data.TrainTestSplit(testSize: 0.2, seed: 42);

split.Train; split.Test; split.Validation;
split["train"]; split.Splits; split.Contains("validation"); split.TotalRows;
```

**Shuffling is on by default and matters more than it looks.** Many published CSVs are sorted by
label, and an unshuffled split of one of those puts every positive example on one side.

Asking for a split that does not exist throws with the available names listed - a dataset published
with `train` and `validation` but no `test` is common enough that the error has to say which splits
exist, not only which one was asked for.

## Reading rows

```csharp
foreach (var row in data.Rows())                 // streaming, one row at a time
    Console.WriteLine(row.Number("age"));

foreach (var batch in data.Batches(32, dropLast: true))
    Train(batch);

data.TextColumn("label");     // works on a text OR a numeric column
data.NumericColumn("age");
data.ToMatrix("age", "fare"); // [rows, columns] NdArray
data.Describe();
```

`TextColumn` coping with either type is deliberate: published datasets disagree about whether a
label column is text or numeric, and a loader that throws on the one it did not expect is a loader
that works on half the Hub.

## Hub datasets

```csharp
HubDatasets.Load("stanfordnlp/imdb", "train");
HubDatasets.LoadAll("stanfordnlp/imdb");
```

A dataset repository has no single layout. Most now publish Parquet under a per-config directory,
older ones publish CSV or JSON Lines at the root, and some publish only a **loading script**, which
is Python and cannot run here.

The strategy is therefore:

1. Ask the Hub's dataset server which Parquet files back each split. The Hub converts most public
   datasets automatically, and using that index is what makes a script-based dataset loadable at
   all.
2. Fall back to pattern-matching the repository's own file listing, guessing the split from the
   path.

A script-only dataset with no Parquet conversion is **refused with a message saying so**, rather
than guessed at.

## JSON and JSON Lines

```csharp
JsonLines.Read("data.jsonl");
```

Handles a JSON array or one object per line, decided by **inspecting the first non-whitespace
character** rather than by the extension: plenty of repositories publish a JSON array in a file
named `.jsonl` and the reverse.

Column order follows first appearance, so a round trip preserves the shape a reader expects rather
than sorting alphabetically. A field missing from some rows becomes a missing value rather than an
error.

## Writing

```csharp
data.WriteCsv("out.csv");
data.WriteParquet("out.parquet");
```

## Memory-mapped reads

```csharp
Dataset.FromCsvMemoryMapped("huge.csv");
```

Worth it only for a file large relative to RAM. Below that the mapping's page faults cost more than
a straight read - which is what `benchmarks/GraviDatasets.Benchmark` exists to measure rather than
assume.

## See also

[GraviHub](GraviHub.md) · [GraviTransformers](GraviTransformers.md) · [GraviAccelerate](GraviAccelerate.md)
