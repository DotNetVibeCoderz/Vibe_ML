using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Preprocessing;

/// <summary>
/// Centres each feature on zero and scales it to unit variance.
/// </summary>
/// <remarks>
/// This is the default first step of almost every pipeline: distance-based models (kNN, k-means,
/// SVM) and gradient-descent models are both dominated by whichever feature happens to have the
/// largest units unless the columns are put on a common scale.
/// </remarks>
public sealed class StandardScaler : ModelBase, ITransformer
{
    /// <summary>Per-feature mean learned during <see cref="Fit"/>.</summary>
    public NdArray Mean { get; private set; } = NdArray.Zeros(0);

    /// <summary>Per-feature standard deviation learned during <see cref="Fit"/>.</summary>
    public NdArray Scale { get; private set; } = NdArray.Zeros(0);

    /// <summary>Whether to subtract the mean.</summary>
    public bool WithMean { get; init; } = true;

    /// <summary>Whether to divide by the standard deviation.</summary>
    public bool WithStd { get; init; } = true;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        Mean = NdArray.Zeros(FeatureCount);
        Scale = NdArray.Ones(FeatureCount);

        for (var j = 0; j < FeatureCount; j++)
        {
            var column = x.Column(j).Copy();
            Mean.SetAt(j, WithMean ? Statistics.Mean(column) : 0.0);
            if (!WithStd) continue;

            var sd = Statistics.Std(column);
            // A constant column has no scale; leaving it at 1 keeps it at zero after centring
            // instead of producing NaN or infinity.
            Scale.SetAt(j, sd > 1e-12 ? sd : 1.0);
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var j = 0; j < FeatureCount; j++)
        {
            var mean = Mean.At(j);
            var scale = Scale.At(j);
            for (var i = 0; i < x.Shape[0]; i++) result[i, j] = (x[i, j] - mean) / scale;
        }
        return result;
    }

    /// <summary>Reverses the scaling, mapping standardised values back to the original units.</summary>
    public NdArray InverseTransform(NdArray x)
    {
        RequireFitted();
        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var j = 0; j < FeatureCount; j++)
            for (var i = 0; i < x.Shape[0]; i++)
                result[i, j] = x[i, j] * Scale.At(j) + Mean.At(j);
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x)
    {
        Fit(x);
        return Transform(x);
    }
}

/// <summary>Rescales each feature into a fixed range, by default <c>[0, 1]</c>.</summary>
public sealed class MinMaxScaler(double minimum = 0.0, double maximum = 1.0) : ModelBase, ITransformer
{
    private double[] _min = [];
    private double[] _range = [];

    /// <summary>Lower bound of the output range.</summary>
    public double Minimum { get; } = minimum;

    /// <summary>Upper bound of the output range.</summary>
    public double Maximum { get; } = maximum;

    /// <summary>Per-feature minimum seen during <see cref="Fit"/>.</summary>
    public NdArray DataMinimum => NdArray.FromValues(_min);

    /// <summary>Per-feature range seen during <see cref="Fit"/>, with zero ranges left as one.</summary>
    /// <remarks>
    /// Exposed for the same reason <see cref="StandardScaler.Mean"/> is: the fitted state is what
    /// an exporter or a diagnostic needs, and hiding it forces callers to re-derive it.
    /// </remarks>
    public NdArray DataRange => NdArray.FromValues(_range);

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _min = new double[FeatureCount];
        _range = new double[FeatureCount];

        for (var j = 0; j < FeatureCount; j++)
        {
            var column = x.Column(j).Copy();
            _min[j] = Statistics.Min(column);
            var range = Statistics.Max(column) - _min[j];
            _range[j] = range > 1e-12 ? range : 1.0;
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var span = Maximum - Minimum;
        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var j = 0; j < FeatureCount; j++)
            for (var i = 0; i < x.Shape[0]; i++)
                result[i, j] = (x[i, j] - _min[j]) / _range[j] * span + Minimum;
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>
/// Scales features using the median and interquartile range, which outliers barely move.
/// </summary>
public sealed class RobustScaler : ModelBase, ITransformer
{
    private double[] _center = [];
    private double[] _scale = [];

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _center = new double[FeatureCount];
        _scale = new double[FeatureCount];

        for (var j = 0; j < FeatureCount; j++)
        {
            var column = x.Column(j).Copy();
            _center[j] = Statistics.Median(column);
            var iqr = Statistics.InterQuartileRange(column);
            _scale[j] = iqr > 1e-12 ? iqr : 1.0;
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var j = 0; j < FeatureCount; j++)
            for (var i = 0; i < x.Shape[0]; i++)
                result[i, j] = (x[i, j] - _center[j]) / _scale[j];
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>Scales each sample (row) to unit norm, which is what cosine-similarity models expect.</summary>
public sealed class Normalizer(double p = 2.0) : ModelBase, ITransformer
{
    /// <summary>The norm to divide by; 1 for Manhattan, 2 for Euclidean.</summary>
    public double P { get; } = p;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var norm = 0.0;
            for (var j = 0; j < FeatureCount; j++) norm += Math.Pow(Math.Abs(x[i, j]), P);
            norm = Math.Pow(norm, 1.0 / P);
            if (norm < 1e-12) norm = 1.0;
            for (var j = 0; j < FeatureCount; j++) result[i, j] = x[i, j] / norm;
        }
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>How <see cref="SimpleImputer"/> fills a gap.</summary>
public enum ImputationStrategy
{
    /// <summary>Column mean.</summary>
    Mean,

    /// <summary>Column median, which resists outliers.</summary>
    Median,

    /// <summary>Most frequent value, appropriate for encoded categories.</summary>
    MostFrequent,

    /// <summary>A fixed constant.</summary>
    Constant,
}

/// <summary>Replaces missing values (NaN) with a statistic learned per column.</summary>
public sealed class SimpleImputer(ImputationStrategy strategy = ImputationStrategy.Mean, double fillValue = 0.0)
    : ModelBase, ITransformer
{
    private double[] _fill = [];

    /// <summary>The strategy used to compute each column's replacement.</summary>
    public ImputationStrategy Strategy { get; } = strategy;

    /// <summary>The constant used when <see cref="Strategy"/> is <see cref="ImputationStrategy.Constant"/>.</summary>
    public double FillValue { get; } = fillValue;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _fill = new double[FeatureCount];

        for (var j = 0; j < FeatureCount; j++)
        {
            var present = new List<double>();
            for (var i = 0; i < x.Shape[0]; i++)
            {
                var v = x[i, j];
                if (!double.IsNaN(v)) present.Add(v);
            }

            _fill[j] = Strategy switch
            {
                ImputationStrategy.Constant => FillValue,
                _ when present.Count == 0 => FillValue,
                ImputationStrategy.Mean => present.Average(),
                ImputationStrategy.Median => Statistics.Median(NdArray.FromValues(present)),
                ImputationStrategy.MostFrequent => Statistics.Mode(NdArray.FromValues(present)),
                _ => FillValue,
            };
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < FeatureCount; j++)
            {
                var v = x[i, j];
                result[i, j] = double.IsNaN(v) ? _fill[j] : v;
            }
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>Maps arbitrary labels onto the contiguous integers <c>0 .. k-1</c>.</summary>
public sealed class LabelEncoder
{
    private readonly Dictionary<double, int> _forward = [];
    private double[] _inverse = [];

    /// <summary>The distinct labels seen during fitting, in ascending order.</summary>
    public IReadOnlyList<double> Classes => _inverse;

    /// <summary>Learns the label set.</summary>
    public LabelEncoder Fit(NdArray y)
    {
        _inverse = y.ToArray().Distinct().OrderBy(v => v).ToArray();
        _forward.Clear();
        for (var i = 0; i < _inverse.Length; i++) _forward[_inverse[i]] = i;
        return this;
    }

    /// <summary>Encodes labels as integer codes.</summary>
    public NdArray Transform(NdArray y)
    {
        var result = NdArray.Zeros(y.Size);
        for (var i = 0; i < y.Size; i++)
        {
            if (!_forward.TryGetValue(y.At(i), out var code))
                throw new ArgumentException($"Label {y.At(i)} was not seen during fitting.");
            result.SetAt(i, code);
        }
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray y) { Fit(y); return Transform(y); }

    /// <summary>Maps integer codes back to the original labels.</summary>
    public NdArray InverseTransform(NdArray codes)
    {
        var result = NdArray.Zeros(codes.Size);
        for (var i = 0; i < codes.Size; i++) result.SetAt(i, _inverse[(int)codes.At(i)]);
        return result;
    }
}

/// <summary>Expands an integer-coded categorical column into one indicator column per category.</summary>
public sealed class OneHotEncoder(bool dropFirst = false) : ModelBase, ITransformer
{
    private double[][] _categories = [];

    /// <summary>Whether the first category of each feature is dropped to avoid collinearity.</summary>
    public bool DropFirst { get; } = dropFirst;

    /// <summary>Total number of output columns.</summary>
    public int OutputWidth => _categories.Sum(c => Math.Max(0, c.Length - (DropFirst ? 1 : 0)));

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _categories = new double[FeatureCount][];
        for (var j = 0; j < FeatureCount; j++)
            _categories[j] = x.Column(j).ToArray().Distinct().OrderBy(v => v).ToArray();
        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], OutputWidth);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var offset = 0;
            for (var j = 0; j < FeatureCount; j++)
            {
                var categories = _categories[j];
                var start = DropFirst ? 1 : 0;
                var index = Array.IndexOf(categories, x[i, j]);
                if (index >= start) result[i, offset + index - start] = 1.0;
                offset += categories.Length - start;
            }
        }
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>
/// Adds polynomial and interaction terms, which lets a linear model fit curved relationships.
/// </summary>
public sealed class PolynomialFeatures(int degree = 2, bool includeBias = false) : ModelBase, ITransformer
{
    private List<int[]> _combinations = [];

    /// <summary>Highest polynomial degree generated.</summary>
    public int Degree { get; } = degree;

    /// <summary>Whether a constant 1 column is included.</summary>
    public bool IncludeBias { get; } = includeBias;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _combinations = [];

        if (IncludeBias) _combinations.Add([]);
        for (var d = 1; d <= Degree; d++) Enumerate([], 0, d);
        IsFitted = true;
    }

    private void Enumerate(List<int> current, int start, int remaining)
    {
        if (remaining == 0) { _combinations.Add(current.ToArray()); return; }
        for (var j = start; j < FeatureCount; j++)
        {
            current.Add(j);
            Enumerate(current, j, remaining - 1);
            current.RemoveAt(current.Count - 1);
        }
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _combinations.Count);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var c = 0; c < _combinations.Count; c++)
            {
                var product = 1.0;
                foreach (var j in _combinations[c]) product *= x[i, j];
                result[i, c] = product;
            }
        return result;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}
