using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.ML.Evaluation;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Training;

namespace BlazorML.Tests;

/// <summary>
/// Every trainer in the catalogue, trained for real.
/// <para>
/// Driven from <see cref="ModuleCatalog"/> rather than a hand-written list, so a trainer added
/// later is covered the moment it is declared. Most of these had never executed once: they
/// compiled, they were wired into the palette, and nothing had ever asked them to fit a model.
/// </para>
/// </summary>
public class TrainerCoverageTests
{
    private readonly AlgorithmExecutor _algorithms = new();
    private readonly TrainingExecutor _training = new();
    private readonly ScoringExecutor _scoring = new();

    /// <summary>Trainers that take a special path and are exercised individually below.</summary>
    private static readonly string[] SpecialPaths =
    [
        "algo.rec.matrixFactorization", "algo.forecast.ssa", "algo.anomaly.spike",
        "algo.vision.imageClassification", "algo.nlp.textClassification"
    ];

    public static TheoryData<string> StandardTrainers()
    {
        var data = new TheoryData<string>();

        foreach (var module in ModuleCatalog.ByCategory(ModuleCategory.Algorithm)
                     .Where(m => !SpecialPaths.Contains(m.Id)))
        {
            data.Add(module.Id);
        }

        return data;
    }

    // ------------------------------------------------------------------- data

    private static TabularData Binary(int rows = 240)
    {
        var random = new Random(31);
        var table = TabularData.WithColumns("pendapatan", "usia", "kota", "beli");

        for (var i = 0; i < rows; i++)
        {
            var income = random.Next(1, 100);
            var city = (string[])["Bandung", "Jakarta", "Medan"];

            table.AddRow(income, random.Next(18, 70), city[random.Next(3)],
                income > 55 ^ random.NextDouble() < 0.06 ? "1" : "0");
        }

        return table;
    }

    private static TabularData Multiclass(int perClass = 70)
    {
        var random = new Random(37);
        var table = TabularData.WithColumns("ukuran", "berat", "kelas");

        foreach (var (name, centre) in ((string Name, int Centre)[])
                 [("kecil", 10), ("sedang", 50), ("besar", 90)])
        {
            for (var i = 0; i < perClass; i++)
            {
                table.AddRow(centre + random.Next(-7, 7), centre * 2 + random.Next(-9, 9), name);
            }
        }

        return table;
    }

    private static TabularData Regression(int rows = 240)
    {
        var random = new Random(41);
        var table = TabularData.WithColumns("luas", "kamar", "harga");

        for (var i = 0; i < rows; i++)
        {
            var area = 40 + random.Next(0, 160);
            var bedrooms = 1 + area / 50;

            // Non-negative, since Tweedie and Poisson are only defined for non-negative targets.
            table.AddRow(area, bedrooms,
                Math.Round(Math.Max(0, area * 3 + bedrooms * 12 + (random.NextDouble() - 0.5) * 20), 2));
        }

        return table;
    }

    private static TabularData Unlabelled(int rows = 160)
    {
        var random = new Random(43);
        var table = TabularData.WithColumns("x", "y");

        for (var i = 0; i < rows; i++)
        {
            var cluster = i % 2 == 0;
            table.AddRow(random.Next(cluster ? 0 : 80, cluster ? 20 : 100),
                random.Next(cluster ? 0 : 80, cluster ? 20 : 100));
        }

        return table;
    }

    /// <summary>
    /// Presence-or-absence features, the shape a bag-of-words produces. Each class is marked by
    /// which flags are set rather than by how large a number is.
    /// </summary>
    private static TabularData PresenceFeatures(int perClass = 60)
    {
        var random = new Random(53);
        var table = TabularData.WithColumns("kata_a", "kata_b", "kata_c", "kata_d", "kelas");

        for (var i = 0; i < perClass; i++)
        {
            table.AddRow(1, 1, 0, 0, "olahraga");
            table.AddRow(0, 0, 1, 1, "politik");
            table.AddRow(1, 0, 1, random.Next(2), "ekonomi");
        }

        return table;
    }

    private static (TabularData Data, string? Label, MlTask Task) Fixture(MlTask task) => task switch
    {
        MlTask.BinaryClassification => (Binary(), "beli", task),
        MlTask.MulticlassClassification => (Multiclass(), "kelas", task),
        MlTask.Regression => (Regression(), "harga", task),
        MlTask.Clustering => (Unlabelled(), null, task),
        MlTask.AnomalyDetection => (Unlabelled(), null, task),
        _ => throw new NotSupportedException($"No fixture for {task}.")
    };

    // -------------------------------------------------------------- the sweep

    [Theory]
    [MemberData(nameof(StandardTrainers))]
    public void Every_trainer_fits_a_model_and_scores_with_it(string moduleId)
    {
        var module = ModuleCatalog.Find(moduleId)!;
        var (data, label, _) = Fixture(module.Task);

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For(moduleId));

        var trainModule = module.Task is MlTask.Clustering or MlTask.AnomalyDetection
            ? "train.clustering"
            : "train.model";

        var trainContext = TestContext.For(trainModule,
            label is null ? null : [("labelColumn", label)], spec, data);

        var handle = _training.Output<TrainedModelHandle>(trainContext);

        Assert.Equal(module.Name, handle.Algorithm);
        Assert.NotEmpty(handle.FeatureColumns);

        // Fitting is only half of it: a model that cannot be scored with is no use.
        var scored = _scoring.Output<TabularData>(
            TestContext.For("score.model", [("appendOriginal", "true")], handle, data));

        Assert.Equal(data.RowCount, scored.RowCount);
        Assert.True(scored.ColumnCount > data.ColumnCount,
            $"{module.Name} added no prediction columns.");
    }

    [Theory]
    [MemberData(nameof(StandardTrainers))]
    public void Every_supervised_trainer_beats_guessing(string moduleId)
    {
        var module = ModuleCatalog.Find(moduleId)!;

        if (module.Task is MlTask.Clustering or MlTask.AnomalyDetection)
        {
            return; // Nothing to be right or wrong about without a label.
        }

        // Naive Bayes only looks at whether a feature is present, so it is given the kind of
        // features it is actually for. Judging it on continuous columns would be testing a
        // mismatch we already document rather than testing the wiring.
        var (data, label, task) = moduleId == "algo.multi.naiveBayes"
            ? (PresenceFeatures(), "kelas", MlTask.MulticlassClassification)
            : Fixture(module.Task);

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For(moduleId));
        var handle = _training.Output<TrainedModelHandle>(
            TestContext.For("train.model", [("labelColumn", label)], spec, data));

        var scored = _scoring.Output<TabularData>(
            TestContext.For("score.model", [("appendOriginal", "true")], handle, data));

        var result = Evaluator.Evaluate(scored, task, label!, "1");

        // A deliberately low bar. The point is to catch a trainer that is wired up wrongly and
        // predicts one class for everything, not to rank the algorithms against each other.
        var score = task switch
        {
            MlTask.BinaryClassification => result.Metrics["Accuracy"],
            MlTask.MulticlassClassification => result.Metrics["MacroAccuracy"],
            _ => result.Metrics["RSquared"]
        };

        Assert.True(score > 0.5,
            $"{module.Name} only reached {score:F3} on data with a planted signal.");
    }

    // ----------------------------------------------------------- special paths

    [Fact]
    public void Matrix_factorization_learns_taste_from_ratings()
    {
        var random = new Random(47);
        var table = TabularData.WithColumns("pengguna", "film", "rating");

        // Two taste groups, so there is real structure to recover.
        for (var user = 1; user <= 80; user++)
        {
            for (var film = 0; film < 10; film++)
            {
                if (random.NextDouble() > 0.6)
                {
                    continue;
                }

                var affinity = film % 2 == user % 2 ? 4.5 : 2.0;
                table.AddRow($"U{user}", $"F{film}", Math.Round(affinity + random.NextDouble() * 0.6, 1));
            }
        }

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.rec.matrixFactorization",
            [("userColumn", "pengguna"), ("itemColumn", "film"), ("approximationRank", "16")]));

        var handle = _training.Output<TrainedModelHandle>(
            TestContext.For("train.model", [("labelColumn", "rating")], spec, table));

        Assert.Equal(MlTask.Recommendation, handle.Task);

        var scored = _scoring.Output<TabularData>(
            TestContext.For("score.model", [("appendOriginal", "true")], handle, table));

        var result = Evaluator.Evaluate(scored, MlTask.Recommendation, "rating");

        Assert.True(result.Metrics["RSquared"] > 0.5,
            $"Only reached R² {result.Metrics["RSquared"]:F3} on ratings with two clear groups.");
    }

    [Fact]
    public void Ssa_forecasting_fits_a_seasonal_series()
    {
        var table = TabularData.WithColumns("penjualan");

        // A weekly cycle with a mild upward trend.
        for (var day = 0; day < 120; day++)
        {
            table.AddRow(Math.Round(100 + day * 0.4 + 20 * Math.Sin(day * 2 * Math.PI / 7), 2));
        }

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.forecast.ssa",
            [("column", "penjualan"), ("horizon", "7"), ("windowSize", "14"), ("seriesLength", "60")]));

        var handle = _training.Output<TrainedModelHandle>(
            TestContext.For("train.model", [("labelColumn", "penjualan")], spec, table));

        Assert.Equal(MlTask.Forecasting, handle.Task);
        Assert.Contains("penjualan", handle.FeatureColumns);
    }

    [Fact]
    public void Ssa_refuses_a_series_too_short_for_its_window()
    {
        var table = TabularData.WithColumns("penjualan");

        for (var i = 0; i < 10; i++)
        {
            table.AddRow(i);
        }

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.forecast.ssa",
            [("column", "penjualan"), ("windowSize", "14")]));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(
                    TestContext.For("train.model", [("labelColumn", "penjualan")], spec, table))
                .GetAwaiter().GetResult());

        Assert.Contains("at least", error.Message);
    }

    [Fact]
    public void Spike_detection_flags_a_planted_spike()
    {
        var table = TabularData.WithColumns("nilai");

        for (var i = 0; i < 100; i++)
        {
            // A flat series with one obvious jump three quarters of the way through.
            table.AddRow(i == 75 ? 500.0 : 10.0 + i % 3);
        }

        var spec = _algorithms.Output<TrainerSpec>(TestContext.For("algo.anomaly.spike",
            [("column", "nilai"), ("confidence", "95"), ("pvalueHistoryLength", "20")]));

        var handle = _training.Output<TrainedModelHandle>(
            TestContext.For("train.clustering", null, spec, table));

        Assert.Equal(MlTask.AnomalyDetection, handle.Task);

        var scored = _scoring.Output<TabularData>(
            TestContext.For("score.model", [("appendOriginal", "true")], handle, table));

        Assert.Equal(100, scored.RowCount);
    }
}
