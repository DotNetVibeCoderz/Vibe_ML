using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Explain;

/// <summary>How much a model's score drops when one feature is scrambled.</summary>
/// <param name="Feature">Index of the feature.</param>
/// <param name="Mean">Average drop in score across repeats.</param>
/// <param name="StandardDeviation">Spread across repeats — small importances are mostly noise.</param>
public readonly record struct FeatureImportance(int Feature, double Mean, double StandardDeviation)
{
    /// <inheritdoc />
    public override string ToString() => $"feature {Feature}: {Mean:F4} ± {StandardDeviation:F4}";
}

/// <summary>
/// Model-agnostic explanations: which features a fitted model actually relies on.
/// </summary>
/// <remarks>
/// These treat the model as a black box and ask what happens to its score when the data changes.
/// That makes them work for any estimator, and it makes them measure something different from a
/// tree's built-in importances — those describe how the tree was <em>built</em>, and are known to
/// favour high-cardinality features regardless of whether they predict anything.
/// </remarks>
public static class PermutationImportance
{
    /// <summary>
    /// Scores each feature by how much accuracy is lost when its column is shuffled.
    /// </summary>
    /// <param name="model">A fitted estimator.</param>
    /// <param name="x">Features to evaluate on — ideally held-out data, not the training set.</param>
    /// <param name="y">True labels.</param>
    /// <param name="repeats">Shuffles per feature; more repeats, less noise.</param>
    /// <param name="seed">Seed for the shuffling.</param>
    /// <remarks>
    /// <para>
    /// Shuffling a column keeps its distribution and destroys its relationship with the target, so
    /// the drop in score is what that relationship was worth. A feature the model ignores scores
    /// about zero; one it depends on scores high.
    /// </para>
    /// <para>
    /// <b>Run this on held-out data.</b> On the training set it measures what the model memorised
    /// rather than what generalises, and an overfitted model will report every feature as vital.
    /// </para>
    /// <para>
    /// Correlated features share the blame and each looks unimportant: shuffling one leaves the
    /// other carrying the same information, so neither drop is large. That is a real limitation of
    /// the method, not a bug — the honest reading is "these two together matter", and the standard
    /// deviation is what warns you the estimate is unstable.
    /// </para>
    /// </remarks>
    public static IReadOnlyList<FeatureImportance> Compute(IEstimator model, NdArray x, NdArray y,
        int repeats = 10, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(x);

        var rows = x.Shape[0];
        var features = x.Shape[1];
        var baseline = Accuracy(model, x, y);

        var rng = new GraviRandom(seed);
        var results = new List<FeatureImportance>(features);

        for (var feature = 0; feature < features; feature++)
        {
            var drops = new double[repeats];

            for (var repeat = 0; repeat < repeats; repeat++)
            {
                var permuted = x.Copy();
                var order = Enumerable.Range(0, rows).ToArray();

                for (var i = rows - 1; i > 0; i--)
                {
                    var j = rng.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }

                for (var i = 0; i < rows; i++) permuted[i, feature] = x[order[i], feature];
                drops[repeat] = baseline - Accuracy(model, permuted, y);
            }

            var mean = drops.Average();
            var variance = drops.Sum(d => (d - mean) * (d - mean)) / Math.Max(1, repeats - 1);
            results.Add(new FeatureImportance(feature, mean, Math.Sqrt(variance)));
        }

        return results;
    }

    /// <summary>The same importances, largest first.</summary>
    public static IReadOnlyList<FeatureImportance> Ranked(IEstimator model, NdArray x, NdArray y,
        int repeats = 10, int seed = 42)
        => [.. Compute(model, x, y, repeats, seed).OrderByDescending(i => i.Mean)];

    private static double Accuracy(IEstimator model, NdArray x, NdArray y)
    {
        var predictions = model.Predict(x);
        var correct = 0;
        for (var i = 0; i < y.Size; i++)
            if (Math.Abs(predictions.At(i) - y.At(i)) < 1e-9) correct++;
        return (double)correct / y.Size;
    }
}

/// <summary>One point of a reliability diagram.</summary>
/// <param name="MeanPredicted">Average predicted probability in this bin.</param>
/// <param name="ObservedFraction">Fraction that actually turned out positive.</param>
/// <param name="Count">How many samples fell in the bin.</param>
public readonly record struct CalibrationPoint(double MeanPredicted, double ObservedFraction, int Count);

/// <summary>
/// Whether a model's probabilities mean what they say.
/// </summary>
/// <remarks>
/// A model can rank perfectly and still be badly calibrated. If everything it calls "90% likely"
/// happens 60% of the time, its ordering is fine and its numbers are not — and any decision made
/// on a threshold, an expected value or a cost trade-off is then wrong. Accuracy and AUC cannot
/// see this; a reliability curve is what shows it.
/// </remarks>
public static class Calibration
{
    /// <summary>
    /// Bins predicted probabilities and reports the observed frequency in each.
    /// </summary>
    /// <param name="probabilities">Predicted probability of the positive class.</param>
    /// <param name="labels">Actual outcomes, 0 or 1.</param>
    /// <param name="bins">Number of equal-width bins across [0, 1].</param>
    /// <remarks>
    /// A perfectly calibrated model puts every point on the diagonal. Empty bins are dropped
    /// rather than reported as zero, which would draw a curve through a region where there is no
    /// evidence at all.
    /// </remarks>
    public static IReadOnlyList<CalibrationPoint> Curve(NdArray probabilities, NdArray labels, int bins = 10)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        ArgumentNullException.ThrowIfNull(labels);
        if (bins <= 0) throw new ArgumentOutOfRangeException(nameof(bins));

        var sums = new double[bins];
        var positives = new double[bins];
        var counts = new int[bins];

        for (var i = 0; i < probabilities.Size; i++)
        {
            var p = Math.Clamp(probabilities.At(i), 0.0, 1.0);
            var bin = Math.Min(bins - 1, (int)(p * bins));

            sums[bin] += p;
            positives[bin] += labels.At(i) > 0.5 ? 1 : 0;
            counts[bin]++;
        }

        var result = new List<CalibrationPoint>(bins);
        for (var b = 0; b < bins; b++)
            if (counts[b] > 0)
                result.Add(new CalibrationPoint(sums[b] / counts[b], positives[b] / counts[b], counts[b]));

        return result;
    }

    /// <summary>
    /// Expected calibration error: the average gap between claimed and observed probability.
    /// </summary>
    /// <remarks>
    /// Weighted by bin population, so a wild miss in a bin holding three samples does not outweigh
    /// a small one in a bin holding a thousand. Zero means perfectly calibrated.
    /// </remarks>
    public static double ExpectedError(NdArray probabilities, NdArray labels, int bins = 10)
    {
        var curve = Curve(probabilities, labels, bins);
        var total = curve.Sum(p => p.Count);
        if (total == 0) return 0.0;

        return curve.Sum(p => p.Count * Math.Abs(p.MeanPredicted - p.ObservedFraction)) / total;
    }

    /// <summary>
    /// The Brier score: mean squared error of the probabilities.
    /// </summary>
    /// <remarks>
    /// Unlike accuracy, this rewards being right <em>and</em> being appropriately confident. A
    /// model that says 0.99 and is wrong is punished far more than one that said 0.6.
    /// </remarks>
    public static double BrierScore(NdArray probabilities, NdArray labels)
    {
        var total = 0.0;
        for (var i = 0; i < probabilities.Size; i++)
        {
            var error = probabilities.At(i) - labels.At(i);
            total += error * error;
        }
        return total / probabilities.Size;
    }
}

/// <summary>
/// Isotonic regression: the best fit that never decreases.
/// </summary>
/// <remarks>
/// <para>
/// Fits a step function that is monotonically non-decreasing and minimises squared error — no
/// other assumption about shape. That makes it the standard way to recalibrate a classifier: the
/// model's ranking is usually good, so the fix is a monotone remapping of its scores onto honest
/// probabilities, which is exactly what this finds.
/// </para>
/// <para>
/// The algorithm is pool-adjacent-violators. Walk left to right; whenever a block's mean falls
/// below its predecessor's, merge the two and re-average. Merging can create a new violation with
/// the block before, so the merge repeats backwards — that inner loop is the whole algorithm, and
/// leaving it out gives something that looks right on smooth data and fails on the ragged data
/// this is for.
/// </para>
/// </remarks>
public sealed class IsotonicRegression
{
    private double[] _thresholds = [];
    private double[] _values = [];

    /// <summary>True once fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Fits a non-decreasing step function of <paramref name="x"/> onto <paramref name="y"/>.</summary>
    public IsotonicRegression Fit(NdArray x, NdArray y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Size != y.Size) throw new ArgumentException("x and y must have the same length.");
        if (x.Size == 0) throw new ArgumentException("There is nothing to fit.");

        var order = Enumerable.Range(0, x.Size).OrderBy(i => x.At(i)).ToArray();

        // Each block records its running total and weight, so merging is one addition.
        var blockValue = new double[order.Length];
        var blockWeight = new double[order.Length];
        var blockEnd = new double[order.Length];
        var blocks = 0;

        foreach (var index in order)
        {
            blockValue[blocks] = y.At(index);
            blockWeight[blocks] = 1;
            blockEnd[blocks] = x.At(index);
            blocks++;

            // Pool backwards while the last block violates monotonicity with the one before it.
            while (blocks > 1 && blockValue[blocks - 2] > blockValue[blocks - 1])
            {
                var weight = blockWeight[blocks - 2] + blockWeight[blocks - 1];
                var value = (blockValue[blocks - 2] * blockWeight[blocks - 2]
                             + blockValue[blocks - 1] * blockWeight[blocks - 1]) / weight;

                blockValue[blocks - 2] = value;
                blockWeight[blocks - 2] = weight;
                blockEnd[blocks - 2] = blockEnd[blocks - 1];
                blocks--;
            }
        }

        _thresholds = blockEnd[..blocks];
        _values = blockValue[..blocks];
        IsFitted = true;
        return this;
    }

    /// <summary>Maps new values through the fitted step function.</summary>
    /// <remarks>
    /// Linear interpolation between block boundaries rather than a hard step, which keeps the
    /// output continuous; below the first block and above the last it holds flat, because there is
    /// no evidence out there to extrapolate from.
    /// </remarks>
    public NdArray Predict(NdArray x)
    {
        if (!IsFitted) throw new InvalidOperationException("The model must be fitted before use.");

        var result = NdArray.Zeros(x.Size);
        for (var i = 0; i < x.Size; i++) result.SetAt(i, PredictOne(x.At(i)));
        return result;
    }

    /// <summary>Maps a single value.</summary>
    public double PredictOne(double value)
    {
        if (!IsFitted) throw new InvalidOperationException("The model must be fitted before use.");

        if (value <= _thresholds[0]) return _values[0];
        if (value >= _thresholds[^1]) return _values[^1];

        for (var i = 1; i < _thresholds.Length; i++)
        {
            if (value > _thresholds[i]) continue;

            var span = _thresholds[i] - _thresholds[i - 1];
            if (span <= 0) return _values[i];

            var t = (value - _thresholds[i - 1]) / span;
            return _values[i - 1] + t * (_values[i] - _values[i - 1]);
        }

        return _values[^1];
    }

    /// <summary>The fitted step boundaries and their values, for inspection or plotting.</summary>
    public IReadOnlyList<(double Threshold, double Value)> Steps
        => [.. _thresholds.Zip(_values)];
}
