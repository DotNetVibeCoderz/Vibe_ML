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

// ---------------------------------------------------------------- v0.4: windows
Section("10. Window functions");

var sales = new DataFrame(
[
    new TextSeries("customer", ["a", "b", "a", "b", "a", "b"]),
    new NumericSeries("day", [1, 1, 2, 2, 3, 3]),
    new NumericSeries("amount", [10, 100, 20, 200, 30, 300]),
]);

Console.WriteLine("  Two interleaved customers, so a partition bug cannot hide:");
Console.WriteLine(sales.ToString());

var running = Windowing.CumulativeSum(sales, ["customer"], "amount");
Console.WriteLine($"  CumulativeSum per customer = [{string.Join(", ", running.Values.ToArray())}]");

var previous = Windowing.Lag(sales, ["customer"], "amount");
Console.WriteLine($"  Lag per customer           = [{string.Join(", ", previous.Values.ToArray().Select(Missing))}]");
Console.WriteLine("    the two NaNs are each group's first row - without partitioning, customer b's");
Console.WriteLine("    first row would reach back into customer a's last, which is a real leak");

var rolling = Windowing.RollingMean(sales, ["customer"], "amount", window: 2);
Console.WriteLine($"  RollingMean(window: 2)     = [{string.Join(", ", rolling.Values.ToArray().Select(Missing))}]");
Console.WriteLine("    incomplete windows stay missing rather than averaging over what is there");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: as-of join
Section("11. As-of join");

var trades = new DataFrame(
[
    new NumericSeries("time", [10, 25, 40]),
    new NumericSeries("size", [1, 2, 3]),
]);

var quotes = new DataFrame(
[
    new NumericSeries("time", [30, 5, 20, 50]),      // deliberately unordered
    new NumericSeries("price", [300, 100, 200, 400]),
]);

Console.WriteLine("  An equality join on a timestamp matches almost nothing - real clocks differ.");
Console.WriteLine("  As-of takes the most recent quote at or before each trade:");
Console.WriteLine(Windowing.AsOfJoin(trades, quotes, "time").ToString());
Console.WriteLine("    t=40 sees the quote from t=30, never the one from t=50. Matching the *nearest*");
Console.WriteLine("    row in either direction is look-ahead, and is how a backtest predicts the past.");

var stale = Windowing.AsOfJoin(trades, quotes, "time", tolerance: 8);
Console.WriteLine($"  with tolerance 8, quotes older than that are dropped: " +
                  $"[{string.Join(", ", stale.Numeric("price").Values.ToArray().Select(Missing))}]");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: categorical
Section("12. Categorical columns");

var sizes = CategoricalSeries.FromValues(
    "size", ["medium", "low", "high", "low", null, "medium"],
    categories: ["low", "medium", "high"], ordered: true);

Console.WriteLine($"  categories = [{string.Join(", ", sizes.Categories)}], ordered = {sizes.IsOrdered}");
Console.WriteLine($"  codes      = [{string.Join(", ", sizes.Codes.ToArray())}]   (-1 is missing)");
Console.WriteLine("  A TextSeries can only sort alphabetically, which puts high < low < medium.");
Console.WriteLine($"  Ordered sort = [{string.Join(", ", sizes.ArgSort().Select(i => sizes[i] ?? "<missing>"))}]");

foreach (var (category, count) in sizes.CategoryCounts())
    Console.WriteLine($"    {category,-8} {count}");
Console.WriteLine("  Categories with no rows are still reported - the set is part of the column's type.");

var indicators = sizes.OneHot(dropFirst: true);
Console.WriteLine($"  OneHot(dropFirst: true) -> {string.Join(", ", indicators.Select(c => c.Name))}");
Console.WriteLine("    the dropped category is the baseline; keeping all of them alongside an");
Console.WriteLine("    intercept makes the design matrix rank-deficient");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: SQL and Excel
Section("13. SQL and Excel round trip");

var workbook = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.xlsx");
try
{
    var report = new DataFrame(
    [
        new TextSeries("city", ["Bandung", "Jakarta", "Surabaya"]),
        new NumericSeries("population", [2.5e6, 10.6e6, 2.9e6]),
        new DateTimeSeries("surveyed", [new DateTime(2024, 3, 1), new DateTime(2024, 6, 15), null]),
    ]);

    ExcelWriter.Write(report, workbook, sheetName: "Cities");
    Console.WriteLine($"  wrote an .xlsx with no spreadsheet library - the format is a zip of XML");
    Console.WriteLine($"  sheets = [{string.Join(", ", ExcelReader.SheetNames(workbook))}]");

    var back = ExcelReader.Read(workbook);
    Console.WriteLine("  read back with types intact:");
    Console.WriteLine(back.ToString());
    Console.WriteLine($"    the missing date came back missing: {back["surveyed"].IsMissing(2)}");
    Console.WriteLine("    Excel stores dates as day counts marked only by a number format, so a");
    Console.WriteLine("    reader that ignores styles gets five-digit integers instead");
}
finally
{
    if (File.Exists(workbook)) File.Delete(workbook);
}

Console.WriteLine();
Console.WriteLine("  SqlReader/SqlWriter work the same way over any ADO.NET provider:");
Console.WriteLine("    var frame = SqlReader.Read(connection, \"SELECT * FROM t WHERE x > $lo\",");
Console.WriteLine("        new Dictionary<string, object?> { [\"$lo\"] = 85.0 });");
Console.WriteLine("  Values go through parameters, never string concatenation - that is what makes");
Console.WriteLine("  injection impossible, and column types come from the provider rather than inference.");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static string Missing(double value) => double.IsNaN(value) ? "NaN" : value.ToString("G6");

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
