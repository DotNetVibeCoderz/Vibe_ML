using System.Diagnostics;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Gravicode.Science.GraviNum;

Console.WriteLine(GraviInfo.Banner("GraviFrame"));

var datasets = SampleSupport.ResolveDatasetDirectory();
var screenshots = SampleSupport.ResolveScreenshotDirectory();

// ---------------------------------------------------------------- load
Section("1. CSV to DataFrame with type inference");

var titanic = DataFrame.ReadCsv(Path.Combine(datasets, "titanic.csv"));
Console.WriteLine($"  Loaded titanic.csv -> {titanic.RowCount} rows x {titanic.ColumnCount} columns");
Console.WriteLine();
Console.WriteLine(titanic.SelectColumns("survived", "pclass", "sex", "age", "fare", "embark_town").ToString(8));
Console.WriteLine();
Console.WriteLine(titanic.SelectColumns("survived", "pclass", "sex", "age", "fare").Info());

// ---------------------------------------------------------------- missing data
Section("2. Missing values");

foreach (var (column, missing) in titanic.MissingCounts().Where(kv => kv.Value > 0).OrderByDescending(kv => kv.Value))
    Console.WriteLine($"  {column,-16}{missing,6} missing ({(double)missing / titanic.RowCount:P1})");

var ages = titanic.Numeric("age");
Console.WriteLine($"  age  : {ages.Count} present, median {ages.Median():F1}");
Console.WriteLine($"  after FillMissingWithMedian: {ages.FillMissingWithMedian().Count} present");
Console.WriteLine();

// ---------------------------------------------------------------- group by
Section("3. GroupBy");

var clean = titanic.WithColumn(ages.FillMissingWithMedian().Rename("age_filled"));

Console.WriteLine("  Survival rate by passenger class:");
Console.WriteLine(clean.GroupBy("pclass").Mean("survived").SortBy("pclass").ToString());
Console.WriteLine();

Console.WriteLine("  Survival rate and mean fare by class and sex:");
var bySexAndClass = clean.GroupBy("pclass", "sex")
    .AggregateMany([("survived", "mean"), ("fare", "mean"), ("age_filled", "mean")]);
Console.WriteLine(bySexAndClass.SortBy([("pclass", true), ("sex", true)]).ToString(12));
Console.WriteLine();

Console.WriteLine("  Passenger counts by embarkation town:");
Console.WriteLine(clean.GroupBy("embark_town").Count("passengers").ToString());
Console.WriteLine();

// ---------------------------------------------------------------- pivot and join
Section("4. Pivot and join");

var pivoted = clean.Pivot("pclass", "sex", "survived");
Console.WriteLine("  Survival rate, class down the side and sex across the top:");
Console.WriteLine(pivoted.ToString());
Console.WriteLine();

var classNames = DataFrame.ParseCsv("""
    pclass,class_label
    1,First
    2,Second
    3,Third
    """);
var labelled = clean.GroupBy("pclass").Mean("survived").Join(classNames, "pclass");
Console.WriteLine("  Joined with a lookup table:");
Console.WriteLine(labelled.SortBy("pclass").ToString());
Console.WriteLine();

// ---------------------------------------------------------------- time series
Section("5. Time series");

var prices = DataFrame.ReadCsv(Path.Combine(datasets, "finance_timeseries.csv"));
var grvc = prices.Filter(row => row.String("ticker") == "GRVC").SortBy("date");
var tradingDays = grvc.DateTimes("date");
Console.WriteLine($"  GRVC: {grvc.RowCount} trading days from {tradingDays[0]:yyyy-MM-dd} to {tradingDays[grvc.RowCount - 1]:yyyy-MM-dd}");

var close = grvc.Numeric("close");
var enriched = grvc
    .WithColumn(close.Rolling(window: 7).Mean().Rename("ma7"))
    .WithColumn(close.Rolling(window: 30).Mean().Rename("ma30"))
    .WithColumn(close.PercentChange().Rename("daily_return"))
    .WithColumn(close.Rolling(window: 30).Std().Rename("volatility30"));

Console.WriteLine();
Console.WriteLine(enriched.SelectColumns("date", "close", "ma7", "ma30", "daily_return").Tail(8).ToString());
Console.WriteLine();

var returns = enriched.Numeric("daily_return");
Console.WriteLine($"  mean daily return : {returns.Mean():P4}");
Console.WriteLine($"  daily volatility  : {returns.Std():P4}");
Console.WriteLine($"  annualised vol    : {returns.Std() * Math.Sqrt(252):P2}");
Console.WriteLine();

Console.WriteLine("  Monthly resample (mean close):");
var monthly = Resampling.Resample(grvc, "date", ResampleFrequency.Monthly, "mean", ["close"]);
Console.WriteLine(monthly.Head(6).ToString());
Console.WriteLine();

// ---------------------------------------------------------------- analytics
Section("6. Descriptive statistics and correlation");

Console.WriteLine(clean.SelectColumns("survived", "pclass", "age_filled", "fare").Describe().ToString());
Console.WriteLine();
Console.WriteLine("  Correlation matrix:");
Console.WriteLine(clean.SelectColumns("survived", "pclass", "age_filled", "fare", "sibsp", "parch").CorrelationMatrix().ToString());
Console.WriteLine();

// ---------------------------------------------------------------- IO performance
Section("7. Streaming versus memory-mapped CSV");

var financePath = Path.Combine(datasets, "finance_timeseries.csv");
var sizeKb = new FileInfo(financePath).Length / 1024.0;

var watch = Stopwatch.StartNew();
var streamed = DataFrame.ReadCsv(financePath);
watch.Stop();
var streamedMs = watch.Elapsed.TotalMilliseconds;

watch.Restart();
var mapped = DataFrame.ReadCsvMemoryMapped(financePath);
watch.Stop();

Console.WriteLine($"  file            : {sizeKb:F1} KB, {streamed.RowCount} rows");
Console.WriteLine($"  StreamReader    : {streamedMs,7:F1} ms");
Console.WriteLine($"  memory mapped   : {watch.Elapsed.TotalMilliseconds,7:F1} ms");
Console.WriteLine($"  identical result: {streamed.RowCount == mapped.RowCount}");
Console.WriteLine("  (the mapped path wins on files large enough that the managed string would not fit)");
Console.WriteLine();

Section("8. Parquet round trip");

var temporary = Directory.CreateTempSubdirectory("graviframe-sample");
try
{
    var parquetPath = Path.Combine(temporary.FullName, "titanic.parquet");
    ParquetIO.Write(clean, parquetPath);
    var reloaded = ParquetIO.Read(parquetPath);

    Console.WriteLine($"  csv     : {new FileInfo(Path.Combine(datasets, "titanic.csv")).Length / 1024.0:F1} KB");
    Console.WriteLine($"  parquet : {new FileInfo(parquetPath).Length / 1024.0:F1} KB");
    Console.WriteLine($"  round trip preserved {reloaded.RowCount} rows and {reloaded.ColumnCount} columns");
    Console.WriteLine($"  column types survive without re-inference: age_filled is {reloaded["age_filled"].DataType}");

    var subset = ParquetIO.Read(parquetPath, ["survived", "fare"]);
    Console.WriteLine($"  reading two columns only: {subset.ColumnCount} columns loaded");
}
finally { temporary.Delete(recursive: true); }
Console.WriteLine();

// ---------------------------------------------------------------- chart
Section("9. Trend chart");

var chartPath = Path.Combine(screenshots, "graviframe_trend.png");
var days = Enumerable.Range(0, grvc.RowCount).Select(i => (double)i).ToArray();

var plot = new ScottPlot.Plot();
var closeLine = plot.Add.Scatter(days, close.Values);
closeLine.LegendText = "close";
closeLine.MarkerSize = 0;

var ma7 = plot.Add.Scatter(days, enriched.Numeric("ma7").Values);
ma7.LegendText = "7-day average";
ma7.MarkerSize = 0;

var ma30 = plot.Add.Scatter(days, enriched.Numeric("ma30").Values);
ma30.LegendText = "30-day average";
ma30.MarkerSize = 0;

plot.Title("GraviFrame - GRVC close with rolling averages");
plot.XLabel("trading day");
plot.YLabel("close");
plot.ShowLegend();
plot.SavePng(chartPath, 1000, 600);
Console.WriteLine($"  saved {chartPath}");

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

/// <summary>Locates the repository directories the samples read from and write to.</summary>
internal static class SampleSupport
{
    public static string ResolveDatasetDirectory() => Resolve("datasets")
        ?? throw new DirectoryNotFoundException("Run this sample from inside the repository so 'datasets' can be found.");

    public static string ResolveScreenshotDirectory()
    {
        var found = Resolve(Path.Combine("docs", "screenshots"));
        if (found is not null) return found;

        var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
        Directory.CreateDirectory(fallback);
        return fallback;
    }

    private static string? Resolve(string relative)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, relative);
            if (Directory.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
        return null;
    }
}
