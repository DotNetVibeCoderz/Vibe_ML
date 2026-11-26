using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.ML;

/// <summary>
/// Batch prediction over partitioned frames (dask-ml's <c>ParallelPostFit</c>): fit on data that
/// fits in memory, score a frame that does not, one task per partition.
/// </summary>
public static class MLFrame
{
    /// <summary>
    /// Adds a prediction column using an already-fitted <paramref name="model"/>. The model is
    /// captured in-process, so this runs on local schedulers only; use the
    /// <see cref="EstimatorSpec"/> overload on a cluster.
    /// </summary>
    /// <remarks>With <c>probabilities</c>, a classifier also adds <c>{output}_p{class}</c> columns.</remarks>
    public static DaskFrame Predict(this DaskFrame frame, IEstimator model, string[] features, string output = "prediction", bool probabilities = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!model.IsFitted) throw new ArgumentException("Fit the model first.", nameof(model));
        CheckFeatures(frame, features);
        return frame.MapPartitions(p => Predict(p, model, features, output, probabilities),
            [.. frame.Columns.Where(c => c != output), output, .. ProbabilityColumns(model, output, probabilities)], "predict");
    }

    /// <summary>
    /// Adds a prediction column on any scheduler, a cluster included: each worker fits
    /// <paramref name="spec"/> on <paramref name="x"/>/<paramref name="y"/> once (the training data
    /// is sent once and fetched once per worker) and then scores every partition it receives.
    /// The estimator must be deterministic for its seed, or workers would disagree.
    /// </summary>
    public static DaskFrame Predict(this DaskFrame frame, EstimatorSpec spec, NdArray x, NdArray y, string[] features,
        string output = "prediction", bool probabilities = false)
    {
        ArgumentNullException.ThrowIfNull(spec);
        CheckFeatures(frame, features);
        if (probabilities && !spec.IsClassifier) throw new ArgumentException($"{spec.Type.Name} is not a classifier.", nameof(probabilities));
        var probe = probabilities ? spec.Create() : null;
        probe?.Fit(x, y);                                  // the class list names the columns
        var token = Guid.NewGuid().ToString("N");
        var dx = Dask.Value(x.AsContiguous(), "predict-x");
        var dy = Dask.Value(y.AsContiguous(), "predict-y");
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(MLKernels.PredictPartition, p, dx, dy,
            Dask.Value<object?[]>([token, spec.TypeId, spec.Parameters.Keys.ToArray(), spec.WireValues, features, output, probabilities]),
            "predict")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. frame.Columns.Where(c => c != output), output, .. ProbabilityColumns(probe, output, probabilities)]);
    }

    /// <summary>The feature columns of a (computed) frame as a samples x features matrix, rows with
    /// a missing feature included as NaN.</summary>
    public static NdArray Features(DataFrame frame, string[] features)
    {
        var n = frame.RowCount;
        var data = new double[n * features.Length];
        for (var j = 0; j < features.Length; j++)
        {
            var column = frame.Numeric(features[j]).Values;
            for (var i = 0; i < n; i++) data[i * features.Length + j] = column[i];
        }
        return new NdArray(data, n, features.Length);
    }

    internal static DataFrame Predict(DataFrame partition, IEstimator model, string[] features, string output, bool probabilities)
    {
        var kept = partition.ColumnNames.Where(c => c != output).Select(c => partition[c]).ToList();
        if (partition.RowCount == 0)
        {
            kept.Add(new NumericSeries(output, []));
            foreach (var c in ProbabilityColumns(model, output, probabilities)) kept.Add(new NumericSeries(c, []));
            return new DataFrame(kept);
        }
        var x = Features(partition, features);
        kept.Add(new NumericSeries(output, model.Predict(x).ToArray()));
        if (probabilities)
        {
            var classifier = (IClassifier)model;
            var p = classifier.PredictProbabilities(x);
            var names = ProbabilityColumns(model, output, true);
            for (var k = 0; k < names.Length; k++)
                kept.Add(new NumericSeries(names[k], Enumerable.Range(0, partition.RowCount).Select(i => p[i, k]).ToArray()));
        }
        return new DataFrame(kept);
    }

    private static string[] ProbabilityColumns(IEstimator? model, string output, bool probabilities)
    {
        if (!probabilities) return [];
        if (model is not IClassifier c) throw new ArgumentException("Probabilities need a classifier.");
        return [.. c.Classes.Select(k => $"{output}_p{EstimatorSpec.Format(k)}")];
    }

    private static void CheckFeatures(DaskFrame frame, string[] features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Length == 0) throw new ArgumentException("Name at least one feature column.", nameof(features));
        foreach (var f in features)
            if (!frame.Columns.Contains(f)) throw new ArgumentException($"No column '{f}'. Columns: {string.Join(", ", frame.Columns)}.", nameof(features));
    }
}
