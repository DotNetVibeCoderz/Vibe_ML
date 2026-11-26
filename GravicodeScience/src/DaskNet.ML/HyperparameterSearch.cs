using System.Collections.Concurrent;
using System.Diagnostics;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.ML;

/// <summary>One candidate's cross-validated scores.</summary>
/// <param name="Spec">The estimator and its parameters.</param>
/// <param name="FoldScores">The score on each held-out fold, in fold order.</param>
/// <param name="FitMilliseconds">Fit plus score time per fold, as measured on the worker.</param>
public sealed record SearchCandidate(EstimatorSpec Spec, double[] FoldScores, double[] FitMilliseconds)
{
    /// <summary>The mean fold score.</summary>
    public double Mean => FoldScores.Average();

    /// <summary>The population standard deviation of the fold scores (scikit-learn's <c>std_test_score</c>).</summary>
    public double StandardDeviation => Math.Sqrt(FoldScores.Average(s => (s - Mean) * (s - Mean)));
}

/// <summary>The outcome of a search: every candidate, best first.</summary>
public sealed class SearchResult
{
    internal SearchResult(IReadOnlyList<SearchCandidate> candidates, string scoring, IEstimator? best)
    {
        Candidates = candidates;
        Scoring = scoring;
        BestEstimator = best;
    }

    /// <summary>Every candidate, ordered by mean score, best first (ties keep search order).</summary>
    public IReadOnlyList<SearchCandidate> Candidates { get; }

    /// <summary>The best candidate.</summary>
    public SearchCandidate Best => Candidates[0];

    /// <summary>The metric; higher is always better (errors are negated, as in scikit-learn).</summary>
    public string Scoring { get; }

    /// <summary>The best candidate refitted on all the data, or null when <c>refit: false</c>.</summary>
    public IEstimator? BestEstimator { get; }

    /// <summary>The results as a frame: one row per candidate, a column per parameter, then
    /// mean, std and rank (scikit-learn's <c>cv_results_</c>).</summary>
    public DataFrame ToFrame()
    {
        var names = Candidates.SelectMany(c => c.Spec.Parameters.Keys).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        var columns = new List<Series>();
        foreach (var n in names)
        {
            var values = Candidates.Select(c => c.Spec.Parameters.TryGetValue(n, out var v) ? v : null).ToArray();
            columns.Add(values.All(v => v is null or int or long or double or float)
                ? new NumericSeries("param_" + n, values.Select(v => v is null ? double.NaN : Convert.ToDouble(v)).ToArray())
                : new TextSeries("param_" + n, values.Select(v => v is null ? null : EstimatorSpec.Format(v)).ToArray()));
        }
        columns.Add(new NumericSeries("mean_score", Candidates.Select(c => c.Mean).ToArray()));
        columns.Add(new NumericSeries("std_score", Candidates.Select(c => c.StandardDeviation).ToArray()));
        columns.Add(new NumericSeries("rank", Enumerable.Range(1, Candidates.Count).Select(r => (double)r).ToArray()));
        return new DataFrame(columns);
    }
}

/// <summary>
/// Hyperparameter search with cross-validation, one task per (candidate, fold) - dask-ml's
/// <c>GridSearchCV</c>/<c>RandomizedSearchCV</c>. On a <see cref="Distributed.Client"/> the
/// training data is sent once and fetched by each worker once; the estimator travels as
/// <see cref="EstimatorSpec"/> (type name + parameter values), so its assembly must be loaded on
/// the workers.
/// </summary>
/// <remarks>
/// Folds are fixed before anything runs (<see cref="Folds"/>), so every candidate sees the same
/// splits and the search is reproducible: the same seed gives the same scores on any scheduler.
/// </remarks>
public static class HyperparameterSearch
{
    /// <summary>Scoring names: <c>accuracy</c>, <c>f1_macro</c>, <c>f1_weighted</c>, <c>r2</c>,
    /// <c>neg_mean_squared_error</c>, <c>neg_root_mean_squared_error</c>; <c>auto</c> is accuracy
    /// for classifiers and r2 otherwise.</summary>
    public static IReadOnlyList<string> ScoringNames { get; } =
        ["accuracy", "f1_macro", "f1_weighted", "r2", "neg_mean_squared_error", "neg_root_mean_squared_error"];

    /// <summary>
    /// Every combination of <paramref name="grid"/>'s values (in order: the last parameter varies
    /// fastest), each cross-validated.
    /// </summary>
    public static SearchResult Grid(Type estimator, IReadOnlyDictionary<string, object?[]> grid, NdArray x, NdArray y,
        int folds = 5, string scoring = "auto", bool shuffle = true, int seed = 42, bool refit = true, IScheduler? scheduler = null)
    {
        ArgumentNullException.ThrowIfNull(grid);
        var names = grid.Keys.ToArray();
        IEnumerable<object?[]> combos = [[]];
        foreach (var n in names)
        {
            var values = grid[n];
            if (values.Length == 0) throw new ArgumentException($"'{n}' has no values to try.", nameof(grid));
            combos = combos.SelectMany(c => values.Select(v => (object?[])[.. c, v]));
        }
        var specs = combos.Select(c => new EstimatorSpec(estimator, names.Select((n, i) => (n, c[i])).ToDictionary(p => p.n, p => p.Item2))).ToArray();
        return Run(specs, x, y, folds, scoring, shuffle, seed, refit, scheduler);
    }

    /// <summary><see cref="Grid(Type, IReadOnlyDictionary{string, object?[]}, NdArray, NdArray, int, string, bool, int, bool, IScheduler?)"/> for <typeparamref name="T"/>.</summary>
    public static SearchResult Grid<T>(IReadOnlyDictionary<string, object?[]> grid, NdArray x, NdArray y,
        int folds = 5, string scoring = "auto", bool shuffle = true, int seed = 42, bool refit = true, IScheduler? scheduler = null) where T : IEstimator
        => Grid(typeof(T), grid, x, y, folds, scoring, shuffle, seed, refit, scheduler);

    /// <summary>
    /// <paramref name="iterations"/> distinct candidates, each parameter drawn uniformly from its
    /// list (seeded). Asking for more than the grid holds tries the whole grid.
    /// </summary>
    public static SearchResult Random<T>(IReadOnlyDictionary<string, object?[]> space, int iterations, NdArray x, NdArray y,
        int folds = 5, string scoring = "auto", bool shuffle = true, int seed = 42, bool refit = true, IScheduler? scheduler = null) where T : IEstimator
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentOutOfRangeException.ThrowIfLessThan(iterations, 1);
        var names = space.Keys.ToArray();
        var total = names.Aggregate(1L, (n, k) => n * Math.Max(1, space[k].Length));
        var random = new Random(seed);
        var seen = new HashSet<string>();
        var specs = new List<EstimatorSpec>();
        while (specs.Count < Math.Min(iterations, total))
        {
            var pick = names.ToDictionary(n => n, n => space[n][random.Next(space[n].Length)]);
            var spec = new EstimatorSpec(typeof(T), pick);
            if (seen.Add(spec.ToString())) specs.Add(spec);
        }
        return Run([.. specs], x, y, folds, scoring, shuffle, seed, refit, scheduler);
    }

    /// <summary>Cross-validates given candidates.</summary>
    public static SearchResult Evaluate(IReadOnlyList<EstimatorSpec> candidates, NdArray x, NdArray y,
        int folds = 5, string scoring = "auto", bool shuffle = true, int seed = 42, bool refit = true, IScheduler? scheduler = null)
        => Run([.. candidates], x, y, folds, scoring, shuffle, seed, refit, scheduler);

    /// <summary>
    /// The fold each sample is held out in: samples are dealt round-robin (after a seeded shuffle)
    /// so fold sizes differ by at most one.
    /// </summary>
    public static int[] Folds(int samples, int folds, bool shuffle = true, int seed = 42)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(folds, 2);
        if (samples < folds) throw new ArgumentException($"{samples} samples cannot make {folds} folds.");
        var order = Enumerable.Range(0, samples).ToArray();
        if (shuffle) new Random(seed).Shuffle(order);
        var foldOf = new int[samples];
        for (var i = 0; i < samples; i++) foldOf[order[i]] = i % folds;
        return foldOf;
    }

    private static SearchResult Run(EstimatorSpec[] specs, NdArray x, NdArray y, int folds, string scoring,
        bool shuffle, int seed, bool refit, IScheduler? scheduler)
    {
        if (specs.Length == 0) throw new ArgumentException("There is nothing to search.");
        if (x.Rank != 2) throw new ArgumentException("x must be a matrix (samples x features).", nameof(x));
        if (y.Shape[0] != x.Shape[0]) throw new ArgumentException("x and y must have the same number of samples.");
        var metric = scoring == "auto" ? (specs[0].IsClassifier ? "accuracy" : "r2") : scoring;
        if (!ScoringNames.Contains(metric)) throw new ArgumentException($"Unknown scoring '{scoring}'. Use one of: auto, {string.Join(", ", ScoringNames)}.", nameof(scoring));

        var foldOf = Folds(x.Shape[0], folds, shuffle, seed);
        var dx = Dask.Value(x.AsContiguous(), "search-x");
        var dy = Dask.Value(y.AsContiguous(), "search-y");
        var df = Dask.Value(foldOf, "search-folds");
        var tasks = new List<Delayed<double[]>>();
        foreach (var spec in specs)
            for (var f = 0; f < folds; f++)
                tasks.Add(Dask.Delayed(MLKernels.FitAndScore, dx, dy, df,
                    Dask.Value<object?[]>([spec.TypeId, spec.Parameters.Keys.ToArray(), spec.WireValues, metric, f]), "fit-score"));
        var results = Dask.Compute(tasks, scheduler);

        var candidates = specs.Select((s, i) => new SearchCandidate(s,
            [.. Enumerable.Range(0, folds).Select(f => results[i * folds + f][0])],
            [.. Enumerable.Range(0, folds).Select(f => results[i * folds + f][1])])).ToList();
        var ranked = candidates.Select((c, i) => (c, i)).OrderByDescending(p => p.c.Mean).ThenBy(p => p.i).Select(p => p.c).ToArray();
        IEstimator? best = null;
        if (refit)
        {
            best = ranked[0].Spec.Create();
            best.Fit(x, y);
        }
        return new SearchResult(ranked, metric, best);
    }

    internal static double Score(string metric, NdArray truth, NdArray predicted) => metric switch
    {
        "accuracy" => Metrics.Accuracy(truth, predicted),
        "f1_macro" => Metrics.F1Average(truth, predicted, "macro"),
        "f1_weighted" => Metrics.F1Average(truth, predicted, "weighted"),
        "r2" => Metrics.R2Score(truth, predicted),
        "neg_mean_squared_error" => -Metrics.MeanSquaredError(truth, predicted),
        "neg_root_mean_squared_error" => -Metrics.RootMeanSquaredError(truth, predicted),
        _ => throw new ArgumentException($"Unknown scoring '{metric}'."),
    };
}

/// <summary>The per-task kernels behind <see cref="HyperparameterSearch"/> and <see cref="MLFrame"/>.</summary>
public static class MLKernels
{
    /// <summary>
    /// Fits one candidate on every fold but one and scores it on that one. <paramref name="spec"/>
    /// is (type id, parameter names, parameter values, metric, held-out fold). Returns
    /// [score, milliseconds].
    /// </summary>
    [RemoteTask]
    public static double[] FitAndScore(NdArray x, NdArray y, int[] foldOf, object?[] spec)
    {
        var clock = Stopwatch.StartNew();
        var fold = Convert.ToInt32(spec[4]);
        var train = new List<int>();
        var test = new List<int>();
        for (var i = 0; i < foldOf.Length; i++) (foldOf[i] == fold ? test : train).Add(i);
        var model = EstimatorSpec.Build(EstimatorSpec.Resolve((string)spec[0]!), (string[])spec[1]!, (object?[])spec[2]!);
        model.Fit(x.Take(train), y.Take(train));
        var score = HyperparameterSearch.Score((string)spec[3]!, y.Take(test), model.Predict(x.Take(test)));
        return [score, clock.Elapsed.TotalMilliseconds];
    }

    // A fitted model per (training data, spec) in each process, so a worker fits once and then
    // scores every partition it is given. Keyed by a token the client draws per call.
    private static readonly ConcurrentDictionary<string, Lazy<IEstimator>> Fitted = new();
    private static readonly Queue<string> Order = new();
    private const int MaxCachedModels = 8;

    /// <summary>
    /// Predicts one partition. <paramref name="spec"/> is (token, type id, names, values,
    /// feature columns, output column, probabilities?). The model is fitted on
    /// <paramref name="x"/>/<paramref name="y"/> the first time a process sees the token.
    /// </summary>
    [RemoteTask]
    public static DataFrame PredictPartition(DataFrame partition, NdArray x, NdArray y, object?[] spec)
    {
        var token = (string)spec[0]!;
        if (!Fitted.ContainsKey(token))
        {
            // Bounded: a long-lived worker serving many Predict calls keeps only the recent models.
            lock (Order)
            {
                Order.Enqueue(token);
                while (Order.Count > MaxCachedModels && Order.TryDequeue(out var old)) Fitted.TryRemove(old, out _);
            }
        }
        var model = Fitted.GetOrAdd(token, _ => new Lazy<IEstimator>(() =>
        {
            var m = EstimatorSpec.Build(EstimatorSpec.Resolve((string)spec[1]!), (string[])spec[2]!, (object?[])spec[3]!);
            m.Fit(x, y);
            return m;
        })).Value;
        return MLFrame.Predict(partition, model, (string[])spec[4]!, (string)spec[5]!, (bool)spec[6]!);
    }

    /// <summary>Forgets the models fitted for <paramref name="token"/> in this process.</summary>
    internal static void Forget(string token) => Fitted.TryRemove(token, out _);

    /// <summary>Number of fitted models cached in this process (for tests).</summary>
    internal static int CachedModels => Fitted.Count;
}
