using System.Text.Json;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Training;
using Microsoft.Data.Sqlite;

namespace BlazorML.Tests;

/// <summary>
/// The modules that had never executed. Each of these compiled and appeared in the palette, and
/// none of them had ever been asked to do its job.
/// </summary>
public class ModuleCoverageTests
{
    private readonly AlgorithmExecutor _algorithms = new();
    private readonly TrainingExecutor _training = new();
    private readonly ScoringExecutor _scoring = new();
    private readonly TransformExecutor _transforms = new();
    private readonly DataInputExecutor _inputs = new();

    private static TabularData Labelled(int rows = 200)
    {
        var random = new Random(59);
        var table = TabularData.WithColumns("sinyal", "derau", "kelas");

        for (var i = 0; i < rows; i++)
        {
            var signal = random.Next(0, 100);

            // "derau" is pure noise; "sinyal" alone decides the label.
            table.AddRow(signal, random.Next(0, 100), signal > 50 ? "1" : "0");
        }

        return table;
    }

    // ------------------------------------------------------------- training

    [Fact]
    public void Cross_validate_reports_a_mean_and_a_spread_across_folds()
    {
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));

        var context = TestContext.For("train.crossValidate",
            [("labelColumn", "kelas"), ("folds", "4")], spec, Labelled());

        var results = _training.ExecuteAsync(context).GetAwaiter().GetResult();
        var metrics = (MetricsBundle)results[0]!;

        Assert.Contains("4-fold", metrics.Title);
        Assert.True(metrics.Values.ContainsKey("Accuracy"));

        // The spread is the point of cross-validation — a single fold's score hides how stable
        // the model is, and reporting only a mean would throw that away.
        Assert.True(metrics.Values.ContainsKey("Accuracy_stdDev"));
        Assert.True(metrics.Values["Accuracy"] > 0.8);

        // It also hands back the best fold's model, so the flow can carry on.
        Assert.NotNull(results[1] as TrainedModelHandle);

        // One row per metric — mean, spread and range — not one row per fold.
        Assert.Equal(["metric", "mean", "stdDev", "min", "max"],
            metrics.Table!.Columns.Select(c => c.Name));

        Assert.Contains(metrics.Table.Rows, r => Convert.ToString(r[0]) == "Accuracy");
        Assert.All(metrics.Table.Rows, r =>
            Assert.True(TabularData.ToNumber(r[3]) <= TabularData.ToNumber(r[4]),
                "A metric's minimum came out above its maximum."));
    }

    [Fact]
    public void Tune_hyperparameters_tries_the_grid_and_keeps_the_best()
    {
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));

        var context = TestContext.For("train.sweep",
            [("labelColumn", "kelas"), ("mode", "grid"),
             ("sweep", "numberOfLeaves: 4, 12\nnumberOfTrees: 10, 40")], spec, Labelled());

        var results = _training.ExecuteAsync(context).GetAwaiter().GetResult();

        var best = results[0] as TrainedModelHandle;
        var sweep = (MetricsBundle)results[1]!;

        Assert.NotNull(best);

        // Two values for each of two parameters: four combinations, all of them run.
        Assert.Equal(4, sweep.Table!.RowCount);
        Assert.True(sweep.Values["BestScore"] > 0.5);
    }

    [Fact]
    public void A_sweep_survives_one_bad_combination()
    {
        // One combination is nonsense; the sweep must record it and carry on rather than
        // throwing away the runs that worked.
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));

        var context = TestContext.For("train.sweep",
            [("labelColumn", "kelas"), ("mode", "grid"),
             ("sweep", "numberOfLeaves: 4, -1")], spec, Labelled());

        var results = _training.ExecuteAsync(context).GetAwaiter().GetResult();
        var sweep = (MetricsBundle)results[1]!;

        Assert.NotNull(results[0] as TrainedModelHandle);
        Assert.Equal(2, sweep.Table!.RowCount);
    }

    [Fact]
    public void An_empty_grid_is_refused_with_an_example()
    {
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));

        var context = TestContext.For("train.sweep",
            [("labelColumn", "kelas"), ("sweep", "bukan sebuah grid")], spec, Labelled());

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("numberOfLeaves", error.Message);
    }

    // ----------------------------------------------------------- evaluation

    /// <summary>
    /// The planted signal decides the label outright and the other column is noise, so a working
    /// permutation test has to rank them in that order. Anything else means it is measuring the
    /// wrong thing.
    /// </summary>
    [Fact]
    public void Feature_importance_ranks_the_signal_above_the_noise()
    {
        var data = Labelled();
        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.bin.fastTree"));

        var handle = _training.Output<TrainedModelHandle>(
            TestContext.For("train.model", [("labelColumn", "kelas")], spec, data));

        var context = TestContext.For("score.featureImportance",
            [("labelColumn", "kelas"), ("permutations", "3")], handle, data);

        var metrics = _scoring.Output<MetricsBundle>(context);
        var ranked = metrics.Evaluation!.FeatureImportance!;

        Assert.Equal("sinyal", ranked[0].Feature);
        Assert.True(ranked[0].Weight > ranked[^1].Weight);
    }

    // ------------------------------------------------------------ transforms

    [Fact]
    public void Featurize_text_builds_a_vocabulary_and_a_vector_per_row()
    {
        var input = TabularData.WithColumns("ulasan");
        foreach (var text in (string[])
                 ["barang bagus sekali pengiriman cepat", "barang bagus dan cepat sampai",
                  "sangat mengecewakan barang rusak", "mengecewakan sekali barang rusak parah",
                  "pengiriman cepat barang bagus", "barang rusak dan mengecewakan"])
        {
            input.AddRow(text);
        }

        var output = _transforms.Output<TabularData>(TestContext.For("tf.featurizeText",
            [("column", "ulasan"), ("outputColumn", "fitur"), ("wordGramLength", "2")], input));

        Assert.True(output.Has("fitur"));

        var first = output.Text(0, "fitur")!;
        var counts = first.Split(' ');

        Assert.True(counts.Length > 3, "The vocabulary came out too small to be useful.");
        Assert.All(counts, c => Assert.True(int.TryParse(c, out _)));

        // Every row gets a vector of the same width, or nothing downstream can consume it.
        Assert.All(output.Rows, r =>
            Assert.Equal(counts.Length, Convert.ToString(r[output.IndexOf("fitur")])!.Split(' ').Length));
    }

    [Fact]
    public void Select_best_features_keeps_the_signal_and_drops_the_noise()
    {
        var input = Labelled();
        input.AddColumn("derau2", ColumnDataType.Numeric);

        var random = new Random(61);
        foreach (var row in input.Rows)
        {
            row[^1] = random.Next(0, 100);
        }

        var results = _transforms.ExecuteAsync(TestContext.For("tf.featureSelection",
            [("labelColumn", "kelas"), ("topN", "1"), ("method", "correlation")], input))
            .GetAwaiter().GetResult();

        var kept = (TabularData)results[0]!;
        var scores = (MetricsBundle)results[1]!;

        Assert.True(kept.Has("sinyal"), "The column that decides the label was dropped.");
        Assert.True(kept.Has("kelas"), "The label itself must never be dropped.");
        Assert.Equal("sinyal", scores.Table!.Text(0, "feature"));
    }

    [Fact]
    public void Mutual_information_also_ranks_the_signal_first()
    {
        // Correlation and mutual information take different routes to the same answer here; the
        // second exists for relationships that are not linear.
        var results = _transforms.ExecuteAsync(TestContext.For("tf.featureSelection",
            [("labelColumn", "kelas"), ("topN", "1"), ("method", "mutualInformation")], Labelled()))
            .GetAwaiter().GetResult();

        Assert.Equal("sinyal", ((MetricsBundle)results[1]!).Table!.Text(0, "feature"));
    }

    // ---------------------------------------------------------------- inputs

    [Fact]
    public void Import_from_sql_reads_a_real_query()
    {
        var path = Path.Combine(Path.GetTempPath(), $"blazorml-sql-{Guid.NewGuid():n}.db");

        try
        {
            using (var connection = new SqliteConnection($"Data Source={path}"))
            {
                connection.Open();

                using var create = connection.CreateCommand();
                create.CommandText =
                    """
                    CREATE TABLE pelanggan (id INTEGER, nama TEXT, umur INTEGER);
                    INSERT INTO pelanggan VALUES (1, 'Ani', 34), (2, 'Budi', 28), (3, 'Citra', NULL);
                    """;
                create.ExecuteNonQuery();
            }

            var context = TestContext.For("data.sql",
                [("provider", "Sqlite"), ("connectionString", $"Data Source={path}"),
                 ("query", "SELECT id, nama, umur FROM pelanggan ORDER BY id")]);

            var output = _inputs.Output<TabularData>(context);

            Assert.Equal(3, output.RowCount);
            Assert.Equal(["id", "nama", "umur"], output.Columns.Select(c => c.Name));
            Assert.Equal("Ani", output.Text(0, "nama"));

            // A NULL in the database has to arrive as a gap, not as the string "NULL".
            Assert.True(TabularData.IsBlank(output.Value(2, "umur")));
        }
        finally
        {
            try
            {
                SqliteConnection.ClearAllPools();
                File.Delete(path);
            }
            catch (IOException)
            {
                // A leftover temp file is not worth failing a run over.
            }
        }
    }

    [Fact]
    public void Import_from_web_reads_csv_over_http()
    {
        const string csv = "nama,umur\nAni,34\nBudi,28\n";
        var context = TestContext.With("data.web", StubHttp.Serving(csv),
            [("url", "https://contoh.id/data.csv"), ("format", "Csv")]);

        var output = _inputs.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("Budi", output.Text(1, "nama"));
    }

    [Fact]
    public void Import_from_web_reads_json_and_sends_the_headers_it_was_given()
    {
        var stub = StubHttp.Serving("""[{"a":1,"b":"x"},{"a":2,"b":"y"}]""");

        var context = TestContext.With("data.web", stub,
            [("url", "https://contoh.id/data.json"), ("format", "Json"),
             ("headers", "Authorization: Bearer abc\nX-Uji: ya")]);

        var output = _inputs.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("x", output.Text(0, "b"));
        Assert.Contains("Authorization", stub.LastHeaders);
        Assert.Contains("X-Uji", stub.LastHeaders);
    }

    [Fact]
    public void Import_from_web_surfaces_a_failed_request()
    {
        var context = TestContext.With("data.web",
            StubHttp.Serving("nope", System.Net.HttpStatusCode.NotFound),
            [("url", "https://contoh.id/hilang.csv"), ("format", "Csv")]);

        Assert.ThrowsAny<HttpRequestException>(() =>
            _inputs.ExecuteAsync(context).GetAwaiter().GetResult());
    }

    [Fact]
    public void Enter_data_manually_parses_the_typed_table()
    {
        var context = TestContext.For("data.manual", [("csv", "kota,jumlah\nBandung,10\nJakarta,20")]);
        var output = _inputs.Output<TabularData>(context);

        Assert.Equal(2, output.RowCount);
        Assert.Equal(20d, TabularData.ToNumber(output.Value(1, "jumlah")));
    }

    // ----------------------------------------------------------------- stub

    private sealed class StubHttp(string body, System.Net.HttpStatusCode status)
        : IServiceProvider, IHttpClientFactory
    {
        public List<string> LastHeaders { get; } = [];

        public static StubHttp Serving(string body,
            System.Net.HttpStatusCode status = System.Net.HttpStatusCode.OK) => new(body, status);

        public object? GetService(Type serviceType) =>
            serviceType == typeof(IHttpClientFactory) ? this : null;

        public HttpClient CreateClient(string name) => new(new Handler(this, body, status));

        private sealed class Handler(StubHttp owner, string body, System.Net.HttpStatusCode status)
            : HttpMessageHandler
        {
            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request, CancellationToken cancellationToken)
            {
                owner.LastHeaders.Clear();
                owner.LastHeaders.AddRange(request.Headers.Select(h => h.Key));

                return Task.FromResult(new HttpResponseMessage(status)
                {
                    Content = new StringContent(body)
                });
            }
        }
    }
}
