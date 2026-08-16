using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Resampling;

/// <summary>A resampled dataset: features and targets rebalanced together.</summary>
/// <param name="X">The resampled feature matrix.</param>
/// <param name="Y">The matching targets.</param>
public readonly record struct ResampledData(NdArray X, NdArray Y);

/// <summary>
/// Rebalancing strategies for datasets where one class vastly outnumbers another.
/// </summary>
/// <remarks>
/// <para>
/// The problem these address is that most learners minimise total error, and on a dataset that is
/// 99% negative the cheapest way to do that is to predict "negative" for everything. That model
/// scores 99% accuracy and is worthless, which is why accuracy is the wrong metric here and why
/// the training distribution is worth changing.
/// </para>
/// <para>
/// <b>Resample the training split only.</b> Rebalancing before splitting puts synthetic points —
/// or duplicates of real ones — on both sides of the split, so the test set contains rows derived
/// from training rows and the score comes back optimistic. This is the single most common way to
/// get an unreproducible result out of an imbalanced problem.
/// </para>
/// <para>
/// None of this is free. Oversampling makes the minority class look denser than it is and can
/// encourage overfitting to a handful of real examples; undersampling throws away real data. Class
/// weights, where the learner supports them, often work as well and discard nothing — see
/// <see cref="ClassWeights"/>.
/// </para>
/// </remarks>
public static class Resampler
{
    /// <summary>
    /// Duplicates minority-class rows at random until every class matches the largest.
    /// </summary>
    /// <remarks>
    /// The simplest thing that works, and the right first attempt: it invents nothing, so any
    /// improvement is attributable to the rebalancing rather than to synthetic data. What it cannot
    /// do is add information — the duplicates sit exactly on top of the originals, so a flexible
    /// model can memorise a few minority points and report a perfect training score.
    /// </remarks>
    public static ResampledData OverSample(NdArray x, NdArray y, int seed = 42)
    {
        var groups = GroupByClass(x, y);
        var target = groups.Values.Max(g => g.Count);
        var rng = new GraviRandom(seed);
        var rows = new List<int>();

        foreach (var group in groups.Values)
        {
            rows.AddRange(group);
            for (var i = group.Count; i < target; i++) rows.Add(group[rng.Next(group.Count)]);
        }

        return Gather(x, y, rows, rng);
    }

    /// <summary>
    /// Drops majority-class rows at random until every class matches the smallest.
    /// </summary>
    /// <remarks>
    /// Cheap to train on and honest about density, but it discards real observations — on a dataset
    /// that is 1000:1 it keeps a thousandth of the majority class, and whatever structure lived in
    /// the rest is simply gone. Sensible when the majority class is genuinely redundant and the
    /// dataset is large; wasteful otherwise.
    /// </remarks>
    public static ResampledData UnderSample(NdArray x, NdArray y, int seed = 42)
    {
        var groups = GroupByClass(x, y);
        var target = groups.Values.Min(g => g.Count);
        var rng = new GraviRandom(seed);
        var rows = new List<int>();

        foreach (var group in groups.Values)
        {
            var pool = group.ToArray();
            Shuffle(pool, rng);
            rows.AddRange(pool.Take(target));
        }

        return Gather(x, y, rows, rng);
    }

    /// <summary>
    /// SMOTE: synthesises new minority rows by interpolating between near neighbours.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Targets.</param>
    /// <param name="neighbours">How many nearest neighbours a synthetic point may be drawn towards.</param>
    /// <param name="seed">Seed for neighbour and interpolation choices.</param>
    /// <remarks>
    /// <para>
    /// For each minority row, pick one of its <paramref name="neighbours"/> nearest same-class
    /// neighbours and place a new point at a random position on the segment between them. Because
    /// the new points land between real ones rather than on top of them, the classifier sees a
    /// region rather than a set of dots, and the decision boundary it draws is correspondingly
    /// broader.
    /// </para>
    /// <para>
    /// The assumption is that the segment between two same-class neighbours is also that class.
    /// Where the minority class is not convex — two separate clusters with majority points between
    /// them — that is false, and SMOTE will manufacture minority points in the middle of majority
    /// territory. Scale features first: neighbours are found by Euclidean distance, so an
    /// unscaled column in the thousands decides every neighbourhood on its own.
    /// </para>
    /// </remarks>
    public static ResampledData Smote(NdArray x, NdArray y, int neighbours = 5, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (neighbours <= 0) throw new ArgumentOutOfRangeException(nameof(neighbours));

        var features = x.Shape[1];
        var groups = GroupByClass(x, y);
        var target = groups.Values.Max(g => g.Count);
        var rng = new GraviRandom(seed);

        var xRows = new List<double[]>();
        var yValues = new List<double>();

        foreach (var (label, group) in groups)
        {
            foreach (var row in group) { xRows.Add(Row(x, row, features)); yValues.Add(label); }

            var needed = target - group.Count;
            if (needed <= 0) continue;

            if (group.Count == 1)
            {
                // Nothing to interpolate towards. Duplicating is the honest fallback; inventing a
                // neighbour would be making the spread up.
                for (var i = 0; i < needed; i++) { xRows.Add(Row(x, group[0], features)); yValues.Add(label); }
                continue;
            }

            var neighbourhood = NearestNeighbours(x, group, Math.Min(neighbours, group.Count - 1), features);

            for (var i = 0; i < needed; i++)
            {
                var pick = i % group.Count;
                var from = Row(x, group[pick], features);
                var candidates = neighbourhood[pick];
                var to = Row(x, candidates[rng.Next(candidates.Length)], features);

                var t = rng.NextDouble();
                var synthetic = new double[features];
                for (var f = 0; f < features; f++) synthetic[f] = from[f] + t * (to[f] - from[f]);

                xRows.Add(synthetic);
                yValues.Add(label);
            }
        }

        return Materialise(xRows, yValues, features, rng);
    }

    /// <summary>
    /// Weights that make every class contribute equally to the loss, without touching the data.
    /// </summary>
    /// <remarks>
    /// The same effect as oversampling, expressed as a per-class multiplier: <c>n / (k · nᶜ)</c>,
    /// so a class holding a tenth of the rows gets ten times the weight. Where a learner accepts
    /// sample weights this is the better tool — it discards nothing, invents nothing, and adds no
    /// rows to train on.
    /// </remarks>
    public static IReadOnlyDictionary<double, double> ClassWeights(NdArray y)
    {
        ArgumentNullException.ThrowIfNull(y);

        var counts = new Dictionary<double, int>();
        for (var i = 0; i < y.Size; i++) counts[y.At(i)] = counts.GetValueOrDefault(y.At(i)) + 1;

        return counts.ToDictionary(
            kv => kv.Key,
            kv => (double)y.Size / (counts.Count * kv.Value));
    }

    /// <summary>The count of each class label, largest first — how to see the imbalance.</summary>
    public static IReadOnlyList<(double Label, int Count)> ClassBalance(NdArray y)
    {
        var counts = new Dictionary<double, int>();
        for (var i = 0; i < y.Size; i++) counts[y.At(i)] = counts.GetValueOrDefault(y.At(i)) + 1;

        return [.. counts.OrderByDescending(kv => kv.Value).Select(kv => (kv.Key, kv.Value))];
    }

    // ------------------------------------------------------------------- helpers

    private static SortedDictionary<double, List<int>> GroupByClass(NdArray x, NdArray y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Rank != 2) throw new ArgumentException("x must be a rank 2 array of shape (samples, features).");
        if (y.Size != x.Shape[0]) throw new ArgumentException("x and y must describe the same rows.");

        var groups = new SortedDictionary<double, List<int>>();
        for (var i = 0; i < y.Size; i++)
        {
            if (!groups.TryGetValue(y.At(i), out var list)) groups[y.At(i)] = list = [];
            list.Add(i);
        }

        if (groups.Count == 0) throw new ArgumentException("There is nothing to resample.");
        return groups;
    }

    /// <summary>
    /// For each member of a group, the indices of its <paramref name="k"/> nearest same-class rows.
    /// </summary>
    /// <remarks>
    /// Brute force, which is what the minority class can afford: it is small by definition, and the
    /// cost is quadratic in <em>its</em> size rather than the dataset's.
    /// </remarks>
    private static int[][] NearestNeighbours(NdArray x, List<int> group, int k, int features)
    {
        var result = new int[group.Count][];

        for (var i = 0; i < group.Count; i++)
        {
            var distances = new (double Distance, int Index)[group.Count - 1];
            var next = 0;

            for (var j = 0; j < group.Count; j++)
            {
                if (i == j) continue;       // a point is its own nearest neighbour, and useless here

                var sum = 0.0;
                for (var f = 0; f < features; f++)
                {
                    var d = x[group[i], f] - x[group[j], f];
                    sum += d * d;
                }

                distances[next++] = (sum, group[j]);
            }

            Array.Sort(distances, (a, b) => a.Distance.CompareTo(b.Distance));
            result[i] = [.. distances.Take(k).Select(d => d.Index)];
        }

        return result;
    }

    private static double[] Row(NdArray x, int row, int features)
    {
        var values = new double[features];
        for (var f = 0; f < features; f++) values[f] = x[row, f];
        return values;
    }

    private static ResampledData Gather(NdArray x, NdArray y, List<int> rows, GraviRandom rng)
    {
        var features = x.Shape[1];
        var order = rows.ToArray();

        // Shuffled, because the rows come out grouped by class and a learner that reads them in
        // order — anything doing mini-batches — would see entire batches of a single label.
        Shuffle(order, rng);

        var resampledX = NdArray.Zeros(order.Length, features);
        var resampledY = NdArray.Zeros(order.Length);

        for (var i = 0; i < order.Length; i++)
        {
            for (var f = 0; f < features; f++) resampledX[i, f] = x[order[i], f];
            resampledY.SetAt(i, y.At(order[i]));
        }

        return new ResampledData(resampledX, resampledY);
    }

    private static ResampledData Materialise(List<double[]> xRows, List<double> yValues, int features,
        GraviRandom rng)
    {
        var order = Enumerable.Range(0, xRows.Count).ToArray();
        Shuffle(order, rng);

        var x = NdArray.Zeros(order.Length, features);
        var y = NdArray.Zeros(order.Length);

        for (var i = 0; i < order.Length; i++)
        {
            for (var f = 0; f < features; f++) x[i, f] = xRows[order[i]][f];
            y.SetAt(i, yValues[order[i]]);
        }

        return new ResampledData(x, y);
    }

    private static void Shuffle(int[] items, GraviRandom rng)
    {
        for (var i = items.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}
