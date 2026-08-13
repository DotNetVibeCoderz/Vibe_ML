using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.ModelSelection;

/// <summary>A train/test partition of a dataset.</summary>
/// <param name="TrainX">Training features.</param>
/// <param name="TestX">Test features.</param>
/// <param name="TrainY">Training targets.</param>
/// <param name="TestY">Test targets.</param>
public readonly record struct TrainTestSplit(NdArray TrainX, NdArray TestX, NdArray TrainY, NdArray TestY);

/// <summary>Splitting, cross-validation and hyper-parameter search.</summary>
public static class Selection
{
    /// <summary>
    /// Splits features and targets into a training and a test set.
    /// </summary>
    /// <remarks>
    /// With <paramref name="stratify"/> the class proportions are preserved in both halves. That
    /// matters whenever a class is rare: an unstratified split can leave a class out of training
    /// entirely, which makes the resulting score meaningless rather than merely noisy.
    /// </remarks>
    public static TrainTestSplit Split(NdArray x, NdArray y, double testSize = 0.25,
        int seed = 42, bool shuffle = true, bool stratify = false)
    {
        var samples = x.Shape[0];
        if (y.Size != samples) throw new ArgumentException("Features and targets must have the same length.");
        if (testSize is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(testSize));

        var rng = new GraviRandom(seed);
        List<int> trainIndices = [];
        List<int> testIndices = [];

        if (stratify)
        {
            foreach (var group in Enumerable.Range(0, samples).GroupBy(i => y.At(i)))
            {
                var members = group.ToList();
                if (shuffle) rng.Shuffle(members);
                var testCount = (int)Math.Round(members.Count * testSize);
                testIndices.AddRange(members.Take(testCount));
                trainIndices.AddRange(members.Skip(testCount));
            }
        }
        else
        {
            var order = shuffle ? rng.Permutation(samples) : Enumerable.Range(0, samples).ToArray();
            var testCount = (int)Math.Round(samples * testSize);
            testIndices.AddRange(order.Take(testCount));
            trainIndices.AddRange(order.Skip(testCount));
        }

        return new TrainTestSplit(
            x.Take(trainIndices), x.Take(testIndices),
            TakeTargets(y, trainIndices), TakeTargets(y, testIndices));
    }

    private static NdArray TakeTargets(NdArray y, IReadOnlyList<int> indices)
    {
        var result = NdArray.Zeros(indices.Count);
        for (var i = 0; i < indices.Count; i++) result.SetAt(i, y.At(indices[i]));
        return result;
    }

    /// <summary>Index folds for k-fold cross-validation.</summary>
    public static IReadOnlyList<(int[] Train, int[] Test)> KFold(int samples, int folds = 5,
        bool shuffle = true, int seed = 42)
    {
        if (folds < 2) throw new ArgumentOutOfRangeException(nameof(folds), "Need at least two folds.");
        if (folds > samples) throw new ArgumentOutOfRangeException(nameof(folds), "More folds than samples.");

        var rng = new GraviRandom(seed);
        var order = shuffle ? rng.Permutation(samples) : Enumerable.Range(0, samples).ToArray();

        var result = new List<(int[], int[])>();
        // Distribute the remainder across the first folds so sizes differ by at most one.
        var baseSize = samples / folds;
        var remainder = samples % folds;

        var position = 0;
        for (var f = 0; f < folds; f++)
        {
            var size = baseSize + (f < remainder ? 1 : 0);
            var test = order.Skip(position).Take(size).ToArray();
            var train = order.Take(position).Concat(order.Skip(position + size)).ToArray();
            result.Add((train, test));
            position += size;
        }
        return result;
    }

    /// <summary>Folds that preserve each class's proportion.</summary>
    public static IReadOnlyList<(int[] Train, int[] Test)> StratifiedKFold(NdArray y, int folds = 5,
        bool shuffle = true, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var buckets = new List<int>[folds];
        for (var f = 0; f < folds; f++) buckets[f] = [];

        foreach (var group in Enumerable.Range(0, y.Size).GroupBy(i => y.At(i)))
        {
            var members = group.ToList();
            if (shuffle) rng.Shuffle(members);
            // Deal each class round-robin so every fold gets a proportional share.
            for (var i = 0; i < members.Count; i++) buckets[i % folds].Add(members[i]);
        }

        var result = new List<(int[], int[])>();
        for (var f = 0; f < folds; f++)
        {
            var test = buckets[f].ToArray();
            var train = Enumerable.Range(0, folds).Where(g => g != f).SelectMany(g => buckets[g]).ToArray();
            result.Add((train, test));
        }
        return result;
    }

    /// <summary>The result of a cross-validation run.</summary>
    /// <param name="Scores">One score per fold.</param>
    /// <param name="Mean">Mean score.</param>
    /// <param name="StandardDeviation">Spread across folds, which indicates stability.</param>
    public readonly record struct CrossValidationResult(double[] Scores, double Mean, double StandardDeviation)
    {
        /// <inheritdoc />
        public override string ToString() => $"{Mean:F4} +/- {StandardDeviation:F4} over {Scores.Length} folds";
    }

    /// <summary>
    /// Cross-validates a model. <paramref name="modelFactory"/> must return a fresh, unfitted
    /// model on each call, because reusing one instance would carry the previous fold's fit.
    /// </summary>
    public static CrossValidationResult CrossValidate(
        Func<IEstimator> modelFactory, NdArray x, NdArray y,
        int folds = 5, bool stratified = false, int seed = 42,
        Func<NdArray, NdArray, double>? scorer = null)
    {
        var splits = stratified
            ? StratifiedKFold(y, folds, seed: seed)
            : KFold(x.Shape[0], folds, seed: seed);

        var score = scorer ?? Metrics.Accuracy;
        var scores = new double[splits.Count];

        for (var f = 0; f < splits.Count; f++)
        {
            var (trainIndices, testIndices) = splits[f];
            var model = modelFactory();
            model.Fit(x.Take(trainIndices), TakeTargets(y, trainIndices));
            scores[f] = score(TakeTargets(y, testIndices), model.Predict(x.Take(testIndices)));
        }

        var array = NdArray.FromValues(scores);
        return new CrossValidationResult(scores, Statistics.Mean(array), Statistics.Std(array));
    }

    /// <summary>Cross-validates a whole pipeline, refitting every step on each training fold.</summary>
    public static CrossValidationResult CrossValidatePipeline(
        Func<Pipeline> pipelineFactory, NdArray x, NdArray y,
        int folds = 5, bool stratified = false, int seed = 42,
        Func<NdArray, NdArray, double>? scorer = null)
    {
        var splits = stratified
            ? StratifiedKFold(y, folds, seed: seed)
            : KFold(x.Shape[0], folds, seed: seed);

        var score = scorer ?? Metrics.Accuracy;
        var scores = new double[splits.Count];

        for (var f = 0; f < splits.Count; f++)
        {
            var (trainIndices, testIndices) = splits[f];
            var pipeline = pipelineFactory();
            pipeline.Fit(x.Take(trainIndices), TakeTargets(y, trainIndices));
            scores[f] = score(TakeTargets(y, testIndices), pipeline.Predict(x.Take(testIndices)));
        }

        var array = NdArray.FromValues(scores);
        return new CrossValidationResult(scores, Statistics.Mean(array), Statistics.Std(array));
    }
}

/// <summary>One point in a hyper-parameter search: the settings and the score they achieved.</summary>
/// <param name="Parameters">The parameter values tried.</param>
/// <param name="MeanScore">Mean cross-validated score.</param>
/// <param name="StandardDeviation">Spread across folds.</param>
public sealed record SearchResult(
    IReadOnlyDictionary<string, object> Parameters,
    double MeanScore,
    double StandardDeviation)
{
    /// <inheritdoc />
    public override string ToString()
        => $"{MeanScore:F4} +/- {StandardDeviation:F4}  {{{string.Join(", ", Parameters.Select(kv => $"{kv.Key}={kv.Value}"))}}}";
}

/// <summary>
/// Exhaustive hyper-parameter search with cross-validation.
/// </summary>
/// <remarks>
/// Every combination in the grid is scored by k-fold cross-validation rather than on a single
/// split, because a single split makes the search itself a source of overfitting: with enough
/// combinations, one will look good on any particular held-out set by luck alone.
/// </remarks>
public sealed class GridSearch(
    Func<IReadOnlyDictionary<string, object>, IEstimator> modelFactory,
    int folds = 5,
    bool stratified = true,
    int seed = 42)
{
    private readonly Dictionary<string, object[]> _grid = new(StringComparer.Ordinal);

    /// <summary>Scores for every combination tried, best first.</summary>
    public IReadOnlyList<SearchResult> Results { get; private set; } = [];

    /// <summary>The best combination found.</summary>
    public SearchResult? Best => Results.Count > 0 ? Results[0] : null;

    /// <summary>The model refitted on all the data with the best parameters.</summary>
    public IEstimator? BestModel { get; private set; }

    /// <summary>Adds a parameter and the values to try for it.</summary>
    public GridSearch AddParameter(string name, params object[] values)
    {
        _grid[name] = values;
        return this;
    }

    /// <summary>Runs the search and refits the winner on the full dataset.</summary>
    public GridSearch Fit(NdArray x, NdArray y, Func<NdArray, NdArray, double>? scorer = null)
    {
        if (_grid.Count == 0) throw new InvalidOperationException("No parameters were added to the grid.");

        var combinations = Expand(_grid.Keys.ToArray(), 0, new Dictionary<string, object>()).ToList();
        var results = new List<SearchResult>();

        foreach (var combination in combinations)
        {
            var evaluation = Selection.CrossValidate(
                () => modelFactory(combination), x, y, folds, stratified, seed, scorer);
            results.Add(new SearchResult(combination, evaluation.Mean, evaluation.StandardDeviation));
        }

        Results = results.OrderByDescending(r => r.MeanScore).ToList();

        if (Best is not null)
        {
            BestModel = modelFactory(Best.Parameters);
            BestModel.Fit(x, y);
        }
        return this;
    }

    private IEnumerable<Dictionary<string, object>> Expand(string[] keys, int index, Dictionary<string, object> current)
    {
        if (index == keys.Length)
        {
            yield return new Dictionary<string, object>(current, StringComparer.Ordinal);
            yield break;
        }

        foreach (var value in _grid[keys[index]])
        {
            current[keys[index]] = value;
            foreach (var expanded in Expand(keys, index + 1, current)) yield return expanded;
        }
        current.Remove(keys[index]);
    }

    /// <summary>Renders the full search table, best first.</summary>
    public string Report(int top = 10)
        => string.Join(Environment.NewLine, Results.Take(top).Select((r, i) => $"  {i + 1,2}. {r}"));
}
