using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Neighbors;

/// <summary>How distance between two samples is measured.</summary>
public enum DistanceMetric
{
    /// <summary>Straight-line distance.</summary>
    Euclidean,

    /// <summary>Sum of absolute differences; less sensitive to a single large discrepancy.</summary>
    Manhattan,

    /// <summary>Largest single-axis difference.</summary>
    Chebyshev,

    /// <summary>One minus cosine similarity; ignores magnitude, which suits text vectors.</summary>
    Cosine,
}

/// <summary>Distance functions shared by the neighbour and clustering models.</summary>
public static class Distances
{
    /// <summary>Distance between row <paramref name="i"/> of <paramref name="a"/> and row <paramref name="j"/> of <paramref name="b"/>.</summary>
    public static double Between(NdArray a, int i, NdArray b, int j, DistanceMetric metric)
    {
        var features = a.Shape[1];
        switch (metric)
        {
            case DistanceMetric.Manhattan:
            {
                var acc = 0.0;
                for (var d = 0; d < features; d++) acc += Math.Abs(a[i, d] - b[j, d]);
                return acc;
            }
            case DistanceMetric.Chebyshev:
            {
                var best = 0.0;
                for (var d = 0; d < features; d++) best = Math.Max(best, Math.Abs(a[i, d] - b[j, d]));
                return best;
            }
            case DistanceMetric.Cosine:
            {
                double dot = 0, na = 0, nb = 0;
                for (var d = 0; d < features; d++)
                {
                    dot += a[i, d] * b[j, d];
                    na += a[i, d] * a[i, d];
                    nb += b[j, d] * b[j, d];
                }
                var denominator = Math.Sqrt(na) * Math.Sqrt(nb);
                return denominator < 1e-12 ? 1.0 : 1.0 - dot / denominator;
            }
            default:
            {
                var acc = 0.0;
                for (var d = 0; d < features; d++)
                {
                    var delta = a[i, d] - b[j, d];
                    acc += delta * delta;
                }
                return Math.Sqrt(acc);
            }
        }
    }

    /// <summary>Euclidean distance between two vectors.</summary>
    public static double Euclidean(NdArray a, NdArray b)
    {
        var acc = 0.0;
        for (var i = 0; i < a.Size; i++)
        {
            var delta = a.At(i) - b.At(i);
            acc += delta * delta;
        }
        return Math.Sqrt(acc);
    }
}

/// <summary>
/// k-nearest neighbours classification.
/// </summary>
/// <remarks>
/// There is no training beyond storing the data - all the work happens at prediction time, which
/// is why kNN is the canonical "lazy" learner. Two consequences matter in practice: features must
/// be scaled (the distance is dominated by whichever column has the largest units), and prediction
/// cost grows with the training set, so queries are parallelised across rows.
/// </remarks>
public sealed class KNearestNeighborsClassifier(
    int k = 5,
    DistanceMetric metric = DistanceMetric.Euclidean,
    bool distanceWeighted = false) : ModelBase, IClassifier
{
    private NdArray _trainX = NdArray.Zeros(0, 0);
    private NdArray _trainY = NdArray.Zeros(0);
    private double[] _classes = [];

    /// <summary>Number of neighbours consulted.</summary>
    public int K { get; } = k;

    /// <summary>Distance function.</summary>
    public DistanceMetric Metric { get; } = metric;

    /// <summary>Whether nearer neighbours get a larger vote.</summary>
    public bool DistanceWeighted { get; } = distanceWeighted;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        if (K > samples) throw new ArgumentException($"k = {K} exceeds the {samples} training samples.");

        FeatureCount = x.Shape[1];
        _trainX = x.Copy();
        _trainY = y.Copy();
        _classes = DistinctClasses(y);
        IsFitted = true;
    }

    /// <summary>The indices and distances of the k nearest training samples for each query row.</summary>
    public (int[][] Indices, double[][] Distances) KNeighbors(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var indices = new int[x.Shape[0]][];
        var distances = new double[x.Shape[0]][];

        Parallel.For(0, x.Shape[0], i =>
        {
            var all = new (int Index, double Distance)[_trainX.Shape[0]];
            for (var j = 0; j < _trainX.Shape[0]; j++)
                all[j] = (j, Distances.Between(x, i, _trainX, j, Metric));

            Array.Sort(all, (a, b) => a.Distance.CompareTo(b.Distance));
            indices[i] = all.Take(K).Select(t => t.Index).ToArray();
            distances[i] = all.Take(K).Select(t => t.Distance).ToArray();
        });

        return (indices, distances);
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var (indices, distances) = KNeighbors(x);
        var result = NdArray.Zeros(x.Shape[0], _classes.Length);

        for (var i = 0; i < x.Shape[0]; i++)
        {
            var weights = new double[_classes.Length];
            var total = 0.0;
            for (var n = 0; n < indices[i].Length; n++)
            {
                // Inverse-distance weighting; the epsilon keeps an exact match finite.
                var weight = DistanceWeighted ? 1.0 / (distances[i][n] + 1e-9) : 1.0;
                var c = Array.IndexOf(_classes, _trainY.At(indices[i][n]));
                if (c < 0) continue;
                weights[c] += weight;
                total += weight;
            }
            for (var c = 0; c < _classes.Length; c++)
                result[i, c] = total > 0 ? weights[c] / total : 0.0;
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var probabilities = PredictProbabilities(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++)
                if (probabilities[i, c] > probabilities[i, best]) best = c;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}

/// <summary>k-nearest neighbours regression: the target is the (optionally weighted) neighbour average.</summary>
public sealed class KNearestNeighborsRegressor(
    int k = 5, DistanceMetric metric = DistanceMetric.Euclidean, bool distanceWeighted = false)
    : ModelBase, IEstimator
{
    private NdArray _trainX = NdArray.Zeros(0, 0);
    private NdArray _trainY = NdArray.Zeros(0);

    /// <summary>Number of neighbours consulted.</summary>
    public int K { get; } = k;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        if (K > samples) throw new ArgumentException($"k = {K} exceeds the {samples} training samples.");

        FeatureCount = x.Shape[1];
        _trainX = x.Copy();
        _trainY = y.Copy();
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0]);
        Parallel.For(0, x.Shape[0], i =>
        {
            var all = new (int Index, double Distance)[_trainX.Shape[0]];
            for (var j = 0; j < _trainX.Shape[0]; j++)
                all[j] = (j, Distances.Between(x, i, _trainX, j, metric));
            Array.Sort(all, (a, b) => a.Distance.CompareTo(b.Distance));

            double weighted = 0, total = 0;
            for (var n = 0; n < K; n++)
            {
                var weight = distanceWeighted ? 1.0 / (all[n].Distance + 1e-9) : 1.0;
                weighted += weight * _trainY.At(all[n].Index);
                total += weight;
            }
            result.SetAt(i, total > 0 ? weighted / total : 0.0);
        });
        return result;
    }

    /// <summary>R-squared on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.R2Score(y, Predict(x));
}

/// <summary>
/// Gaussian naive Bayes: models each feature as normally distributed within each class and
/// assumes the features are conditionally independent.
/// </summary>
/// <remarks>
/// The independence assumption is almost always wrong, yet the classifier is often competitive:
/// getting the argmax right does not require the probabilities themselves to be right. Training
/// is a single pass computing per-class means and variances, which makes it the fastest baseline
/// worth running.
/// </remarks>
public sealed class GaussianNaiveBayes(double variancesmoothing = 1e-9) : ModelBase, IClassifier
{
    private double[] _classes = [];
    private double[] _priors = [];
    private NdArray _means = NdArray.Zeros(0, 0);
    private NdArray _variances = NdArray.Zeros(0, 0);

    /// <summary>Constant added to every variance so a constant feature cannot divide by zero.</summary>
    public double VarianceSmoothing { get; } = variancesmoothing;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>Class prior probabilities, aligned with <see cref="Classes"/>.</summary>
    public IReadOnlyList<double> Priors => _priors;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _classes = DistinctClasses(y);

        _priors = new double[_classes.Length];
        _means = NdArray.Zeros(_classes.Length, FeatureCount);
        _variances = NdArray.Zeros(_classes.Length, FeatureCount);

        var globalVariance = 0.0;
        for (var j = 0; j < FeatureCount; j++) globalVariance = Math.Max(globalVariance, Statistics.Var(x.Column(j).Copy()));
        var smoothing = VarianceSmoothing * Math.Max(globalVariance, 1e-12);

        for (var c = 0; c < _classes.Length; c++)
        {
            var rows = Enumerable.Range(0, samples).Where(i => y.At(i) == _classes[c]).ToArray();
            _priors[c] = (double)rows.Length / samples;

            for (var j = 0; j < FeatureCount; j++)
            {
                var values = rows.Select(i => x[i, j]).ToArray();
                var mean = values.Average();
                var variance = values.Length > 1
                    ? values.Sum(v => (v - mean) * (v - mean)) / values.Length
                    : 0.0;
                _means[c, j] = mean;
                _variances[c, j] = variance + smoothing;
            }
        }
        IsFitted = true;
    }

    /// <summary>Unnormalised log posterior for each class.</summary>
    public NdArray JointLogLikelihood(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var c = 0; c < _classes.Length; c++)
            {
                // Work in log space: multiplying many small densities underflows otherwise.
                var logProbability = Math.Log(_priors[c]);
                for (var j = 0; j < FeatureCount; j++)
                {
                    var variance = _variances[c, j];
                    var delta = x[i, j] - _means[c, j];
                    logProbability += -0.5 * Math.Log(2 * Math.PI * variance) - delta * delta / (2 * variance);
                }
                result[i, c] = logProbability;
            }
        return result;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var row = new double[_classes.Length];
            for (var c = 0; c < _classes.Length; c++) row[c] = joint[i, c];
            var probabilities = MathUtil.Softmax(row);
            for (var c = 0; c < _classes.Length; c++) result[i, c] = probabilities[c];
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++) if (joint[i, c] > joint[i, best]) best = c;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}

/// <summary>
/// Multinomial naive Bayes for count features, the standard baseline for text classification.
/// </summary>
public sealed class MultinomialNaiveBayes(double alpha = 1.0) : ModelBase, IClassifier
{
    private double[] _classes = [];
    private double[] _logPriors = [];
    private NdArray _logLikelihood = NdArray.Zeros(0, 0);

    /// <summary>Additive (Laplace) smoothing, which stops an unseen token forcing a zero probability.</summary>
    public double Alpha { get; } = alpha;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _classes = DistinctClasses(y);

        _logPriors = new double[_classes.Length];
        _logLikelihood = NdArray.Zeros(_classes.Length, FeatureCount);

        for (var c = 0; c < _classes.Length; c++)
        {
            var rows = Enumerable.Range(0, samples).Where(i => y.At(i) == _classes[c]).ToArray();
            _logPriors[c] = Math.Log((double)rows.Length / samples);

            var counts = new double[FeatureCount];
            var total = 0.0;
            foreach (var i in rows)
                for (var j = 0; j < FeatureCount; j++)
                {
                    counts[j] += x[i, j];
                    total += x[i, j];
                }

            var denominator = total + Alpha * FeatureCount;
            for (var j = 0; j < FeatureCount; j++)
                _logLikelihood[c, j] = Math.Log((counts[j] + Alpha) / denominator);
        }
        IsFitted = true;
    }

    /// <summary>Unnormalised log posterior for each class.</summary>
    public NdArray JointLogLikelihood(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var c = 0; c < _classes.Length; c++)
            {
                var acc = _logPriors[c];
                for (var j = 0; j < FeatureCount; j++) acc += x[i, j] * _logLikelihood[c, j];
                result[i, c] = acc;
            }
        return result;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var row = new double[_classes.Length];
            for (var c = 0; c < _classes.Length; c++) row[c] = joint[i, c];
            var probabilities = MathUtil.Softmax(row);
            for (var c = 0; c < _classes.Length; c++) result[i, c] = probabilities[c];
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var joint = JointLogLikelihood(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            for (var c = 1; c < _classes.Length; c++) if (joint[i, c] > joint[i, best]) best = c;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}
