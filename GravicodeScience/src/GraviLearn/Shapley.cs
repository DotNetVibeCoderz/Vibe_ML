using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Explain;

/// <summary>
/// An additive explanation of a single prediction.
/// </summary>
/// <param name="baseValue">
/// What the model predicts on average over the background data — where the explanation starts.
/// </param>
/// <param name="contributions">One value per feature; they sum to the prediction minus the base.</param>
public sealed class Attribution(double baseValue, double[] contributions)
{
    /// <summary>The average prediction over the background data.</summary>
    public double BaseValue { get; } = baseValue;

    /// <summary>Each feature's share of the gap between the base value and this prediction.</summary>
    public IReadOnlyList<double> Contributions { get; } = contributions;

    /// <summary>The prediction the explanation reconstructs: base value plus every contribution.</summary>
    public double Prediction => BaseValue + Contributions.Sum();

    /// <summary>Features ordered by how much they moved this prediction, in either direction.</summary>
    public IReadOnlyList<(int Feature, double Contribution)> Ranked
        => [.. Contributions
            .Select((value, index) => (Feature: index, Contribution: value))
            .OrderByDescending(p => Math.Abs(p.Contribution))];
}

/// <summary>
/// Shapley values: how a model split the credit for one prediction among its features.
/// </summary>
/// <remarks>
/// <para>
/// Permutation importance answers "which features does this model rely on overall". This answers a
/// different question — "why did it say <em>that</em>, for <em>this</em> row" — and the two
/// routinely disagree. A feature can be globally unimportant and decisive for one applicant.
/// </para>
/// <para>
/// The Shapley value is borrowed from cooperative game theory: add features to the model one at a
/// time in a random order and record how much each one moves the prediction when it joins; average
/// over every possible order. It is the unique attribution satisfying efficiency (the parts sum to
/// the whole), symmetry (features that always contribute equally get equal credit), and the dummy
/// property (a feature that never changes anything gets zero). No cheaper heuristic has all three.
/// </para>
/// <para>
/// "Absent" means replaced by a value drawn from the background dataset, so the background is part
/// of the explanation: contributions are always relative to it. Explaining a loan refusal against a
/// background of approved applicants answers a different question from explaining it against all
/// applicants, and the honest reading of any attribution requires knowing which was used.
/// </para>
/// <para>
/// Cost is why there are two methods here. <see cref="Exact"/> enumerates all 2ⁿ subsets and is
/// only usable below about fifteen features; <see cref="Sample"/> is Monte Carlo over permutations
/// and is what to use above that.
/// </para>
/// </remarks>
public static class ShapleyValues
{
    /// <summary>Above this many features, exact enumeration is refused rather than left to run.</summary>
    public const int ExactFeatureLimit = 20;

    /// <summary>
    /// Exact Shapley values by enumerating every subset of features.
    /// </summary>
    /// <param name="predict">The model's output for a batch of rows.</param>
    /// <param name="instance">The single row being explained.</param>
    /// <param name="background">Rows that stand in for "feature absent". A sample of training data.</param>
    /// <remarks>
    /// Cost is 2ⁿ subsets × the background size, so this is for small feature counts. It is worth
    /// having because it is exact: <see cref="Sample"/> is checked against it.
    /// </remarks>
    public static Attribution Exact(Func<NdArray, NdArray> predict, NdArray instance, NdArray background)
    {
        ArgumentNullException.ThrowIfNull(predict);
        ArgumentNullException.ThrowIfNull(instance);
        ArgumentNullException.ThrowIfNull(background);

        var features = background.Shape[1];
        if (features > ExactFeatureLimit)
            throw new ArgumentException(
                $"Exact Shapley values enumerate 2^n subsets; {features} features is too many. Use Sample instead.");

        var row = RowOf(instance, features);

        // v(S) for every subset S, indexed by bitmask.
        var values = new double[1 << features];
        for (var mask = 0; mask < values.Length; mask++) values[mask] = Value(predict, row, background, mask, features);

        var contributions = new double[features];
        var factorials = Factorials(features);

        for (var feature = 0; feature < features; feature++)
        {
            var bit = 1 << feature;
            var total = 0.0;

            foreach (var mask in Enumerable.Range(0, values.Length).Where(m => (m & bit) == 0))
            {
                // The classic weight: |S|! (n - |S| - 1)! / n! is the fraction of orderings in
                // which exactly this coalition precedes the feature.
                var size = System.Numerics.BitOperations.PopCount((uint)mask);
                var weight = factorials[size] * factorials[features - size - 1] / factorials[features];
                total += weight * (values[mask | bit] - values[mask]);
            }

            contributions[feature] = total;
        }

        return new Attribution(values[0], contributions);
    }

    /// <summary>
    /// Shapley values estimated by sampling random feature orderings.
    /// </summary>
    /// <param name="predict">The model's output for a batch of rows.</param>
    /// <param name="instance">The row being explained.</param>
    /// <param name="background">Rows that stand in for "feature absent".</param>
    /// <param name="samples">Permutations drawn. Error falls as 1/√samples.</param>
    /// <param name="seed">Seed for the permutations and background draws.</param>
    /// <remarks>
    /// <para>
    /// Walk a random permutation, revealing one feature at a time, and credit each feature with the
    /// change it caused when revealed. Averaged over enough permutations this converges on the
    /// exact value, and cost is linear in the feature count rather than exponential.
    /// </para>
    /// <para>
    /// The telescoping is what keeps efficiency exact rather than approximate: along any single
    /// permutation the increments sum to <c>f(instance) − f(background row)</c> by construction, so
    /// the estimate satisfies "contributions sum to the prediction" at every sample size. Only the
    /// split between features is approximate.
    /// </para>
    /// </remarks>
    public static Attribution Sample(Func<NdArray, NdArray> predict, NdArray instance, NdArray background,
        int samples = 200, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(predict);
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples));

        var features = background.Shape[1];
        var row = RowOf(instance, features);
        var rng = new GraviRandom(seed);

        var totals = new double[features];
        var order = Enumerable.Range(0, features).ToArray();

        // Each sample needs the prediction after every reveal: n + 1 rows, evaluated in one batch
        // so the model sees a matrix rather than n + 1 separate calls.
        var batch = NdArray.Zeros(features + 1, features);
        var baseTotal = 0.0;

        for (var sample = 0; sample < samples; sample++)
        {
            for (var i = features - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            var reference = rng.Next(background.Shape[0]);
            var current = new double[features];
            for (var f = 0; f < features; f++) current[f] = background[reference, f];

            for (var f = 0; f < features; f++) batch[0, f] = current[f];

            for (var step = 0; step < features; step++)
            {
                current[order[step]] = row[order[step]];
                for (var f = 0; f < features; f++) batch[step + 1, f] = current[f];
            }

            var predictions = predict(batch);
            baseTotal += predictions.At(0);

            for (var step = 0; step < features; step++)
                totals[order[step]] += predictions.At(step + 1) - predictions.At(step);
        }

        for (var f = 0; f < features; f++) totals[f] /= samples;
        return new Attribution(baseTotal / samples, totals);
    }

    /// <summary>
    /// Explains every row of <paramref name="instances"/>, giving a per-feature importance too.
    /// </summary>
    /// <remarks>
    /// The mean absolute contribution across rows is the usual global summary — and unlike a tree's
    /// built-in importances it is in the units of the prediction, so it can be read as "this feature
    /// moves the output by about this much".
    /// </remarks>
    public static (IReadOnlyList<Attribution> PerRow, IReadOnlyList<double> MeanAbsolute) Explain(
        Func<NdArray, NdArray> predict, NdArray instances, NdArray background,
        int samples = 100, int seed = 42)
    {
        var features = background.Shape[1];
        var rows = new List<Attribution>(instances.Shape[0]);
        var totals = new double[features];

        for (var i = 0; i < instances.Shape[0]; i++)
        {
            var instance = NdArray.Zeros(1, features);
            for (var f = 0; f < features; f++) instance[0, f] = instances[i, f];

            var attribution = Sample(predict, instance, background, samples, seed + i);
            rows.Add(attribution);

            for (var f = 0; f < features; f++) totals[f] += Math.Abs(attribution.Contributions[f]);
        }

        for (var f = 0; f < features; f++) totals[f] /= instances.Shape[0];
        return (rows, totals);
    }

    // ------------------------------------------------------------------- helpers

    /// <summary>
    /// The model's expected output when the features in <paramref name="mask"/> come from the
    /// instance and the rest come from the background.
    /// </summary>
    private static double Value(Func<NdArray, NdArray> predict, double[] row, NdArray background,
        int mask, int features)
    {
        var rows = background.Shape[0];
        var batch = NdArray.Zeros(rows, features);

        for (var i = 0; i < rows; i++)
            for (var f = 0; f < features; f++)
                batch[i, f] = (mask & (1 << f)) != 0 ? row[f] : background[i, f];

        var predictions = predict(batch);
        var total = 0.0;
        for (var i = 0; i < rows; i++) total += predictions.At(i);
        return total / rows;
    }

    private static double[] RowOf(NdArray instance, int features)
    {
        if (instance.Size != features)
            throw new ArgumentException(
                $"The instance has {instance.Size} values but the background has {features} features.");

        var row = new double[features];
        for (var f = 0; f < features; f++) row[f] = instance.At(f);
        return row;
    }

    private static double[] Factorials(int n)
    {
        var values = new double[n + 1];
        values[0] = 1;
        for (var i = 1; i <= n; i++) values[i] = values[i - 1] * i;
        return values;
    }
}
