using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.ML;

/// <summary>
/// A model that learns one batch at a time (scikit-learn's <c>partial_fit</c>), so it can be
/// trained over a frame larger than memory one partition at a time. Its whole state is a list of
/// plain values (<see cref="State"/>), which is what lets the training chain run on a cluster:
/// every step receives the state, learns one partition, and passes the new state on.
/// </summary>
public interface IIncrementalModel : IEstimator
{
    /// <summary>Learns from one more batch. Labels outside the classes given up front are refused.</summary>
    void PartialFit(NdArray x, NdArray y);

    /// <summary>The model's state as wire-encodable values.</summary>
    object?[] State();

    /// <summary>Restores <see cref="State"/> (of the same model type and settings).</summary>
    void Load(object?[] state);
}

/// <summary>Linear regression by stochastic gradient descent on the squared loss with an L2
/// penalty, one sample at a time in row order (scikit-learn's <c>SGDRegressor</c>).</summary>
/// <param name="learningRate">The initial step, eta0.</param>
/// <param name="alpha">The L2 penalty.</param>
/// <param name="power">The step decays as eta0 / t^power (scikit-learn's "invscaling", 0.25).</param>
public sealed class SgdRegressor(double learningRate = 0.01, double alpha = 1e-4, double power = 0.25) : IIncrementalModel
{
    /// <summary>The initial step.</summary>
    public double LearningRate { get; } = learningRate;
    /// <summary>The L2 penalty.</summary>
    public double Alpha { get; } = alpha;
    /// <summary>The step's decay exponent.</summary>
    public double Power { get; } = power;

    private double[] _w = [];
    private double _b;
    private long _t;

    /// <summary>The coefficients.</summary>
    public IReadOnlyList<double> Coefficients => _w;

    /// <summary>The intercept.</summary>
    public double Intercept => _b;

    /// <summary>Samples seen.</summary>
    public long Steps => _t;

    /// <inheritdoc />
    public bool IsFitted => _t > 0;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        _w = [];
        _b = 0;
        _t = 0;
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public void PartialFit(NdArray x, NdArray y)
    {
        var (n, d) = Check(x, y, _w.Length);
        if (_w.Length == 0) _w = new double[d];
        for (var i = 0; i < n; i++)
        {
            _t++;
            var eta = LearningRate / Math.Pow(_t, Power);
            var p = _b;
            for (var j = 0; j < d; j++) p += _w[j] * x[i, j];
            var g = p - y.At(i);
            for (var j = 0; j < d; j++) _w[j] -= eta * (g * x[i, j] + Alpha * _w[j]);
            _b -= eta * g;
        }
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        if (!IsFitted) throw new InvalidOperationException("Fit the model first.");
        var result = new double[x.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var p = _b;
            for (var j = 0; j < _w.Length; j++) p += _w[j] * x[i, j];
            result[i] = p;
        }
        return new NdArray(result, result.Length);
    }

    /// <inheritdoc />
    public object?[] State() => [_w.ToArray(), _b, _t];

    /// <inheritdoc />
    public void Load(object?[] state)
    {
        _w = ((double[])state[0]!).ToArray();
        _b = (double)state[1]!;
        _t = (long)state[2]!;
    }

    internal static (int N, int D) Check(NdArray x, NdArray y, int width)
    {
        if (x.Rank != 2) throw new ArgumentException("x must be samples x features.", nameof(x));
        if (y.Size != x.Shape[0]) throw new ArgumentException($"y has {y.Size} values for {x.Shape[0]} rows.", nameof(y));
        if (width != 0 && x.Shape[1] != width) throw new ArgumentException($"The model has {width} features; this batch has {x.Shape[1]}.", nameof(x));
        return (x.Shape[0], x.Shape[1]);
    }
}

/// <summary>
/// Multinomial logistic regression by stochastic gradient descent, one sample at a time (the
/// incremental counterpart of <c>SGDClassifier(loss="log_loss")</c>). The classes must be given
/// up front, as <c>partial_fit</c>'s <c>classes</c> argument is: the first batch need not contain
/// them all.
/// </summary>
/// <param name="classes">Every label the model will see.</param>
/// <param name="learningRate">The initial step, eta0.</param>
/// <param name="alpha">The L2 penalty.</param>
/// <param name="power">The step decays as eta0 / t^power.</param>
public sealed class SgdClassifier(double[] classes, double learningRate = 0.1, double alpha = 1e-4, double power = 0.25)
    : IIncrementalModel, IClassifier
{
    private readonly double[] _classes = [.. (classes ?? throw new ArgumentNullException(nameof(classes))).Distinct().Order()];
    private double[] _w = [];      // classes x features, row-major

    /// <summary>The initial step.</summary>
    public double LearningRate { get; } = learningRate;
    /// <summary>The L2 penalty.</summary>
    public double Alpha { get; } = alpha;
    /// <summary>The step's decay exponent.</summary>
    public double Power { get; } = power;
    private double[] _b = [];
    private long _t;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <inheritdoc />
    public bool IsFitted => _t > 0;

    /// <summary>Samples seen.</summary>
    public long Steps => _t;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        _w = [];
        _t = 0;
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public void PartialFit(NdArray x, NdArray y)
    {
        if (_classes.Length < 2) throw new InvalidOperationException("A classifier needs at least two classes.");
        var k = _classes.Length;
        var (n, d) = SgdRegressor.Check(x, y, _w.Length == 0 ? 0 : _w.Length / k);
        if (_w.Length == 0)
        {
            _w = new double[k * d];
            _b = new double[k];
        }
        var p = new double[k];
        for (var i = 0; i < n; i++)
        {
            var label = Array.BinarySearch(_classes, y.At(i));
            if (label < 0) throw new ArgumentException($"Label {y.At(i)} is not one of the classes given: {string.Join(", ", _classes)}.", nameof(y));
            _t++;
            var eta = LearningRate / Math.Pow(_t, Power);
            Scores(x, i, d, p);
            Softmax(p);
            for (var c = 0; c < k; c++)
            {
                var g = p[c] - (c == label ? 1 : 0);
                for (var j = 0; j < d; j++) _w[c * d + j] -= eta * (g * x[i, j] + Alpha * _w[c * d + j]);
                _b[c] -= eta * g;
            }
        }
    }

    private void Scores(NdArray x, int row, int d, double[] p)
    {
        for (var c = 0; c < _classes.Length; c++)
        {
            var s = _b[c];
            for (var j = 0; j < d; j++) s += _w[c * d + j] * x[row, j];
            p[c] = s;
        }
    }

    private static void Softmax(double[] p)
    {
        var max = p.Max();
        var total = 0.0;
        for (var c = 0; c < p.Length; c++) total += p[c] = Math.Exp(p[c] - max);
        for (var c = 0; c < p.Length; c++) p[c] /= total;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        if (!IsFitted) throw new InvalidOperationException("Fit the model first.");
        var k = _classes.Length;
        var d = _w.Length / k;
        var result = NdArray.Zeros(x.Shape[0], k);
        var p = new double[k];
        for (var i = 0; i < x.Shape[0]; i++)
        {
            Scores(x, i, d, p);
            Softmax(p);
            for (var c = 0; c < k; c++) result[i, c] = p[c];
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var p = PredictProbabilities(x);
        var result = new double[x.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++) if (p[i, c] > p[i, best]) best = c;
            result[i] = _classes[best];
        }
        return new NdArray(result, result.Length);
    }

    /// <inheritdoc />
    public object?[] State() => [_w.ToArray(), _b.ToArray(), _t];

    /// <inheritdoc />
    public void Load(object?[] state)
    {
        _w = ((double[])state[0]!).ToArray();
        _b = ((double[])state[1]!).ToArray();
        _t = (long)state[2]!;
    }
}

/// <summary>
/// Gaussian naive Bayes learned batch by batch. Per-class counts, means and squared deviations
/// are merged with Chan's formula, so the fitted model equals a <see cref="GraviLearn.Neighbors.GaussianNaiveBayes"/>
/// fitted on all the rows at once (to rounding), whatever the batches and their order.
/// </summary>
/// <param name="classes">Every label the model will see.</param>
/// <param name="varianceSmoothing">As <see cref="GraviLearn.Neighbors.GaussianNaiveBayes"/>'s: this fraction of the
/// largest feature variance is added to every variance.</param>
public sealed class IncrementalGaussianNaiveBayes(double[] classes, double varianceSmoothing = 1e-9) : IIncrementalModel, IClassifier
{
    /// <summary>The fraction of the largest feature variance added to every variance.</summary>
    public double VarianceSmoothing { get; } = varianceSmoothing;

    private readonly double[] _classes = [.. (classes ?? throw new ArgumentNullException(nameof(classes))).Distinct().Order()];
    private double[] _count = [];  // per class
    private double[] _mean = [];   // classes x features
    private double[] _m2 = [];
    private double _all;           // every row, for the smoothing term
    private double[] _allMean = [];
    private double[] _allM2 = [];
    private int _d;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <inheritdoc />
    public bool IsFitted => _all > 0;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        _d = 0;
        _all = 0;
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public void PartialFit(NdArray x, NdArray y)
    {
        var k = _classes.Length;
        var (n, d) = SgdRegressor.Check(x, y, _d);
        if (_d == 0 || _all == 0)
        {
            _d = d;
            _count = new double[k];
            _mean = new double[k * d];
            _m2 = new double[k * d];
            _allMean = new double[d];
            _allM2 = new double[d];
            _all = 0;
        }
        // This batch's moments, then one Chan merge per class: merging row by row (Welford) gives
        // the same value but rounds differently batch to batch.
        var count = new double[k];
        var mean = new double[k * d];
        var m2 = new double[k * d];
        var labels = new int[n];
        for (var i = 0; i < n; i++)
        {
            var c = Array.BinarySearch(_classes, y.At(i));
            if (c < 0) throw new ArgumentException($"Label {y.At(i)} is not one of the classes given: {string.Join(", ", _classes)}.", nameof(y));
            labels[i] = c;
            count[c]++;
            for (var j = 0; j < d; j++) mean[c * d + j] += x[i, j];
        }
        for (var c = 0; c < k; c++)
            if (count[c] > 0) for (var j = 0; j < d; j++) mean[c * d + j] /= count[c];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < d; j++)
            {
                var delta = x[i, j] - mean[labels[i] * d + j];
                m2[labels[i] * d + j] += delta * delta;
            }
        for (var c = 0; c < k; c++)
            for (var j = 0; j < d; j++)
                Merge(ref _count[c], ref _mean[c * d + j], ref _m2[c * d + j], count[c], mean[c * d + j], m2[c * d + j], j == d - 1);

        var batchMean = new double[d];
        var batchM2 = new double[d];
        for (var j = 0; j < d; j++)
        {
            for (var i = 0; i < n; i++) batchMean[j] += x[i, j];
            batchMean[j] /= n;
            for (var i = 0; i < n; i++) batchM2[j] += (x[i, j] - batchMean[j]) * (x[i, j] - batchMean[j]);
        }
        var all = _all;
        for (var j = 0; j < d; j++)
        {
            var a = all;
            Merge(ref a, ref _allMean[j], ref _allM2[j], n, batchMean[j], batchM2[j], true);
        }
        _all = all + n;
    }

    /// <summary>Chan et al.'s pairwise merge of (count, mean, M2); the count is updated only on
    /// the last feature so every feature merges against the same prior count.</summary>
    private static void Merge(ref double na, ref double meanA, ref double m2A, double nb, double meanB, double m2B, bool updateCount)
    {
        if (nb == 0) return;
        var n = na + nb;
        var delta = meanB - meanA;
        meanA += delta * nb / n;
        m2A += m2B + delta * delta * na * nb / n;
        if (updateCount) na = n;
    }

    /// <summary>Unnormalised log posterior per class.</summary>
    public NdArray JointLogLikelihood(NdArray x)
    {
        if (!IsFitted) throw new InvalidOperationException("Fit the model first.");
        var k = _classes.Length;
        var global = 0.0;
        for (var j = 0; j < _d; j++) global = Math.Max(global, _allM2[j] / _all);
        var smoothing = VarianceSmoothing * Math.Max(global, 1e-12);
        var result = NdArray.Zeros(x.Shape[0], k);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var c = 0; c < k; c++)
            {
                if (_count[c] == 0)
                {
                    result[i, c] = double.NegativeInfinity;     // a class never seen has prior 0
                    continue;
                }
                var lp = Math.Log(_count[c] / _all);
                for (var j = 0; j < _d; j++)
                {
                    var variance = (_count[c] > 1 ? _m2[c * _d + j] / _count[c] : 0.0) + smoothing;
                    var delta = x[i, j] - _mean[c * _d + j];
                    lp += -0.5 * Math.Log(2 * Math.PI * variance) - delta * delta / (2 * variance);
                }
                result[i, c] = lp;
            }
        return result;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var k = _classes.Length;
        var result = NdArray.Zeros(x.Shape[0], k);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var max = double.NegativeInfinity;
            for (var c = 0; c < k; c++) max = Math.Max(max, joint[i, c]);
            var total = 0.0;
            for (var c = 0; c < k; c++) total += Math.Exp(joint[i, c] - max);
            for (var c = 0; c < k; c++) result[i, c] = Math.Exp(joint[i, c] - max) / total;
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var result = new double[x.Shape[0]];
        for (var i = 0; i < result.Length; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++) if (joint[i, c] > joint[i, best]) best = c;
            result[i] = _classes[best];
        }
        return new NdArray(result, result.Length);
    }

    /// <inheritdoc />
    public object?[] State() => [_count.ToArray(), _mean.ToArray(), _m2.ToArray(), _all, _allMean.ToArray(), _allM2.ToArray(), _d];

    /// <inheritdoc />
    public void Load(object?[] state)
    {
        _count = ((double[])state[0]!).ToArray();
        _mean = ((double[])state[1]!).ToArray();
        _m2 = ((double[])state[2]!).ToArray();
        _all = (double)state[3]!;
        _allMean = ((double[])state[4]!).ToArray();
        _allM2 = ((double[])state[5]!).ToArray();
        _d = (int)state[6]!;
    }
}

/// <summary>
/// Trains an <see cref="IIncrementalModel"/> over a partitioned frame (dask-ml's
/// <c>Incremental</c>): a chain of tasks, one per partition per epoch, each learning its partition
/// and handing the model state to the next. Only the state moves - partitions are read where they
/// are - so on a cluster each step runs on the worker that holds its partition.
/// </summary>
/// <remarks>
/// Training is sequential by nature: SGD's next step depends on the last, so the chain has no
/// parallelism, and the win is memory and locality, not speed. Each epoch visits the partitions
/// in an order shuffled by <c>seed</c>; the same seed gives the same model on any scheduler.
/// </remarks>
public static class Incremental
{
    /// <summary>
    /// The delayed training chain: computing it gives the fitted state, which
    /// <see cref="IIncrementalModel.Load"/> restores.
    /// </summary>
    public static Delayed<object?[]> FitDelayed(IIncrementalModel model, DaskFrame frame, string[] features, string target,
        int epochs = 1, int? seed = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentOutOfRangeException.ThrowIfLessThan(epochs, 1);
        foreach (var c in features.Append(target))
            if (!frame.Columns.Contains(c)) throw new ArgumentException($"No column '{c}'. Columns: {string.Join(", ", frame.Columns)}.");
        var spec = ModelSpec(model);
        var parts = frame.ToDelayed();
        var state = Dask.Value<object?[]?>(model.IsFitted ? model.State() : null, "incremental-start");
        var meta = Dask.Value<object?[]>([.. spec, features, target], "incremental-spec");
        var random = seed is { } s ? new Random(s) : null;
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var order = Enumerable.Range(0, parts.Length).ToArray();
            random?.Shuffle(order);
            foreach (var p in order)
                state = Dask.Delayed(IncrementalKernels.PartialFit, state, parts[p], meta, $"partial-fit-{epoch}");
        }
        return Dask.Delayed(IncrementalKernels.Finish, state, "incremental-fitted");
    }

    /// <summary>Trains <paramref name="model"/> in place over <paramref name="frame"/> and returns it.</summary>
    public static TModel Fit<TModel>(TModel model, DaskFrame frame, string[] features, string target, int epochs = 1,
        int? seed = 0, IScheduler? scheduler = null) where TModel : IIncrementalModel
    {
        var state = FitDelayed(model, frame, features, target, epochs, seed).Compute(scheduler);
        model.Load(state);
        return model;
    }

    /// <summary>
    /// Adds a prediction column on any scheduler: the model's state travels with each partition's
    /// task, so a fitted incremental model scores on a cluster without refitting.
    /// </summary>
    public static DaskFrame Predict(this DaskFrame frame, IIncrementalModel model, string[] features, string output = "prediction")
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!model.IsFitted) throw new ArgumentException("Fit the model first.", nameof(model));
        foreach (var f in features)
            if (!frame.Columns.Contains(f)) throw new ArgumentException($"No column '{f}'.", nameof(features));
        var meta = Dask.Value<object?[]>([.. ModelSpec(model), model.State(), features, output], "incremental-model");
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(IncrementalKernels.Predict, p, meta, "incremental-predict")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. frame.Columns.Where(c => c != output), output]);
    }

    /// <summary>The model's type and constructor settings, enough for a worker to make an empty one.</summary>
    internal static object?[] ModelSpec(IIncrementalModel model) => model switch
    {
        SgdRegressor r => ["sgd-regressor", new object?[] { r.LearningRate, r.Alpha, r.Power }],
        SgdClassifier c => ["sgd-classifier", new object?[] { c.Classes.ToArray(), c.LearningRate, c.Alpha, c.Power }],
        IncrementalGaussianNaiveBayes g => ["gaussian-nb", new object?[] { g.Classes.ToArray(), g.VarianceSmoothing }],
        _ => throw new NotSupportedException(
            $"{model.GetType().Name} has no wire spec; distributed training supports SgdRegressor, SgdClassifier and IncrementalGaussianNaiveBayes."),
    };

    internal static IIncrementalModel Make(string kind, object?[] s) => kind switch
    {
        "sgd-regressor" => new SgdRegressor((double)s[0]!, (double)s[1]!, (double)s[2]!),
        "sgd-classifier" => new SgdClassifier((double[])s[0]!, (double)s[1]!, (double)s[2]!, (double)s[3]!),
        "gaussian-nb" => new IncrementalGaussianNaiveBayes((double[])s[0]!, (double)s[1]!),
        _ => throw new NotSupportedException($"Unknown incremental model '{kind}'."),
    };
}

/// <summary>The remotable steps of incremental training and scoring.</summary>
public static class IncrementalKernels
{
    /// <summary>Learns one partition: state in, state out.</summary>
    [RemoteTask]
    public static object?[]? PartialFit(object?[]? state, DataFrame partition, object?[] meta)
    {
        var model = Incremental.Make((string)meta[0]!, (object?[])meta[1]!);
        if (state is not null) model.Load(state);
        if (partition.RowCount == 0) return state;
        var features = (string[])meta[2]!;
        var target = (string)meta[3]!;
        model.PartialFit(MLFrame.Features(partition, features), new NdArray(partition.Numeric(target).Values.ToArray(), partition.RowCount));
        return model.State();
    }

    /// <summary>The fitted state, refusing a chain that never saw a row.</summary>
    [RemoteTask]
    public static object?[] Finish(object?[]? state)
        => state ?? throw new InvalidOperationException("Incremental training saw no rows.");

    /// <summary>Scores one partition with a model rebuilt from its state.</summary>
    [RemoteTask]
    public static DataFrame Predict(DataFrame partition, object?[] meta)
    {
        var model = Incremental.Make((string)meta[0]!, (object?[])meta[1]!);
        model.Load((object?[])meta[2]!);
        return MLFrame.Predict(partition, model, (string[])meta[3]!, (string)meta[4]!, probabilities: false);
    }
}
