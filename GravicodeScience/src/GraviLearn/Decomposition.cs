using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Decomposition;

/// <summary>
/// Principal component analysis: rotates the data onto the orthogonal directions of greatest variance.
/// </summary>
/// <remarks>
/// The components come from the SVD of the centred data matrix rather than from an eigen
/// decomposition of the covariance matrix. Both give the same answer in exact arithmetic, but
/// forming <c>X'X</c> squares the condition number, so the SVD route keeps small components
/// accurate on ill-conditioned data.
/// </remarks>
public class PrincipalComponentAnalysis(int components) : ModelBase, ITransformer
{
    /// <summary>Number of components kept.</summary>
    public int Components { get; } = components;

    /// <summary>Per-feature mean removed before projection.</summary>
    public NdArray Mean { get; private set; } = NdArray.Zeros(0);

    /// <summary>The principal axes, one component per row.</summary>
    public NdArray ComponentVectors { get; private set; } = NdArray.Zeros(0, 0);

    /// <summary>Variance captured by each component.</summary>
    public NdArray ExplainedVariance { get; private set; } = NdArray.Zeros(0);

    /// <summary>Fraction of total variance captured by each component.</summary>
    public NdArray ExplainedVarianceRatio { get; private set; } = NdArray.Zeros(0);

    /// <summary>Running total of <see cref="ExplainedVarianceRatio"/>.</summary>
    public NdArray CumulativeExplainedVariance
    {
        get
        {
            var cumulative = NdArray.Zeros(ExplainedVarianceRatio.Size);
            var acc = 0.0;
            for (var i = 0; i < ExplainedVarianceRatio.Size; i++)
            {
                acc += ExplainedVarianceRatio.At(i);
                cumulative.SetAt(i, acc);
            }
            return cumulative;
        }
    }

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        if (Components > Math.Min(samples, FeatureCount))
            throw new ArgumentException(
                $"Cannot keep {Components} components from data with {samples} samples and {FeatureCount} features.");

        Mean = Statistics.Mean(x, axis: 0);
        var centred = NdArray.Zeros(samples, FeatureCount);
        for (var i = 0; i < samples; i++)
            for (var j = 0; j < FeatureCount; j++)
                centred[i, j] = x[i, j] - Mean.At(j);

        var svd = GraviNum.Decomposition.Svd(centred);

        ComponentVectors = NdArray.Zeros(Components, FeatureCount);
        ExplainedVariance = NdArray.Zeros(Components);
        for (var c = 0; c < Components; c++)
        {
            for (var j = 0; j < FeatureCount; j++) ComponentVectors[c, j] = svd.V[j, c];
            var singular = svd.SingularValues.At(c);
            ExplainedVariance.SetAt(c, singular * singular / Math.Max(1, samples - 1));
        }

        var totalVariance = 0.0;
        for (var j = 0; j < FeatureCount; j++) totalVariance += Statistics.Var(centred.Column(j).Copy(), ddof: 1);

        ExplainedVarianceRatio = NdArray.Zeros(Components);
        for (var c = 0; c < Components; c++)
            ExplainedVarianceRatio.SetAt(c, totalVariance > 0 ? ExplainedVariance.At(c) / totalVariance : 0.0);

        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var centred = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < FeatureCount; j++)
                centred[i, j] = x[i, j] - Mean.At(j);

        return LinAlg.Dot(centred, ComponentVectors.T);
    }

    /// <summary>Projects back into the original feature space, losing the discarded components.</summary>
    public NdArray InverseTransform(NdArray projected)
    {
        RequireFitted();
        var reconstructed = LinAlg.Dot(projected, ComponentVectors);
        for (var i = 0; i < reconstructed.Shape[0]; i++)
            for (var j = 0; j < FeatureCount; j++)
                reconstructed[i, j] += Mean.At(j);
        return reconstructed;
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x) { Fit(x); return Transform(x); }
}

/// <summary>
/// Short alias for <see cref="PrincipalComponentAnalysis"/>, so pipelines read the way they do
/// in scikit-learn: <c>new PCA(components: 10)</c>.
/// </summary>
public sealed class PCA(int components) : PrincipalComponentAnalysis(components);

/// <summary>
/// Linear discriminant analysis: finds the projection that best separates known classes.
/// </summary>
/// <remarks>
/// Where PCA maximises total variance without looking at labels, LDA maximises the ratio of
/// between-class to within-class scatter, so it is the better dimensionality reduction when the
/// goal is classification. At most <c>classes - 1</c> useful directions exist.
/// </remarks>
public sealed class LinearDiscriminantAnalysis(int components = 0) : ModelBase, ITransformer
{
    private int _components = components;

    /// <summary>Number of discriminant directions kept.</summary>
    public int Components => _components;

    /// <summary>The discriminant directions, one per row.</summary>
    public NdArray Directions { get; private set; } = NdArray.Zeros(0, 0);

    /// <summary>Per-class feature means.</summary>
    public NdArray ClassMeans { get; private set; } = NdArray.Zeros(0, 0);

    /// <summary>The class labels, ascending.</summary>
    public IReadOnlyList<double> Classes { get; private set; } = [];

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray? y = null)
    {
        ArgumentNullException.ThrowIfNull(y, nameof(y));
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        var classes = DistinctClasses(y);
        Classes = classes;
        if (_components <= 0) _components = Math.Min(classes.Length - 1, FeatureCount);
        if (_components < 1) throw new ArgumentException("LDA needs at least two classes.");

        var overallMean = Statistics.Mean(x, axis: 0);
        ClassMeans = NdArray.Zeros(classes.Length, FeatureCount);

        var within = NdArray.Zeros(FeatureCount, FeatureCount);
        var between = NdArray.Zeros(FeatureCount, FeatureCount);

        for (var c = 0; c < classes.Length; c++)
        {
            var rows = new List<int>();
            for (var i = 0; i < samples; i++) if (y.At(i) == classes[c]) rows.Add(i);

            var subset = x.Take(rows);
            var mean = Statistics.Mean(subset, axis: 0);
            for (var j = 0; j < FeatureCount; j++) ClassMeans[c, j] = mean.At(j);

            // Within-class scatter: how spread the class is around its own mean.
            foreach (var row in rows)
                for (var a = 0; a < FeatureCount; a++)
                {
                    var da = x[row, a] - mean.At(a);
                    for (var b = 0; b < FeatureCount; b++)
                        within[a, b] += da * (x[row, b] - mean.At(b));
                }

            // Between-class scatter: how far the class mean sits from the overall mean.
            for (var a = 0; a < FeatureCount; a++)
            {
                var da = mean.At(a) - overallMean.At(a);
                for (var b = 0; b < FeatureCount; b++)
                    between[a, b] += rows.Count * da * (mean.At(b) - overallMean.At(b));
            }
        }

        // Regularise the within-class scatter so the inverse exists even for collinear features.
        for (var j = 0; j < FeatureCount; j++) within[j, j] += 1e-6;

        var target = LinAlg.Dot(LinAlg.Inverse(within), between);
        // The product is not symmetric in general, so symmetrise before the eigen solve; the
        // discriminant directions are unchanged and the solver stays stable.
        var symmetric = (target + target.T) * 0.5;
        var eigen = GraviNum.Decomposition.SymmetricEigen(symmetric);

        Directions = NdArray.Zeros(_components, FeatureCount);
        for (var c = 0; c < _components; c++)
            for (var j = 0; j < FeatureCount; j++)
                Directions[c, j] = eigen.Vectors[j, c];

        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        return LinAlg.Dot(x, Directions.T);
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(NdArray x, NdArray y) { Fit(x, y); return Transform(x); }
}

/// <summary>
/// t-distributed stochastic neighbour embedding: a non-linear projection to two or three
/// dimensions that preserves local neighbourhoods.
/// </summary>
/// <remarks>
/// t-SNE is a visualisation tool, not a general transformer: it has no out-of-sample
/// <c>Transform</c>, distances in the output are not meaningful, and cluster sizes carry no
/// information. What it does preserve is which points are near which, which is exactly what makes
/// it the standard way to look at word or graph embeddings.
/// </remarks>
public sealed class TStochasticNeighborEmbedding(
    int components = 2,
    double perplexity = 30.0,
    int iterations = 500,
    double learningRate = 200.0,
    int seed = 42)
{
    /// <summary>Output dimensionality.</summary>
    public int Components { get; } = components;

    /// <summary>Effective number of neighbours each point is attracted to.</summary>
    public double Perplexity { get; } = perplexity;

    /// <summary>Gradient descent iterations.</summary>
    public int Iterations { get; } = iterations;

    /// <summary>Gradient descent step size.</summary>
    public double LearningRate { get; } = learningRate;

    /// <summary>Seed for the random initialisation.</summary>
    public int Seed { get; } = seed;

    /// <summary>Final value of the KL divergence being minimised.</summary>
    public double FinalDivergence { get; private set; }

    /// <summary>Computes the embedding for <paramref name="x"/>.</summary>
    public NdArray FitTransform(NdArray x)
    {
        var n = x.Shape[0];
        if (n < 3) throw new ArgumentException("t-SNE needs at least three samples.");

        var p = ComputeAffinities(x, n);

        var rng = new GraviRandom(Seed);
        var embedding = rng.Normal(0.0, 1e-4, n, Components);
        var velocity = NdArray.Zeros(n, Components);
        var gains = NdArray.Ones(n, Components);

        for (var iteration = 0; iteration < Iterations; iteration++)
        {
            // Early exaggeration pulls clusters apart before the layout settles.
            var exaggeration = iteration < 100 ? 4.0 : 1.0;
            var momentum = iteration < 250 ? 0.5 : 0.8;

            var (gradient, divergence) = Gradient(p, embedding, n, exaggeration);
            FinalDivergence = divergence;

            for (var i = 0; i < n; i++)
                for (var d = 0; d < Components; d++)
                {
                    // Jacobs' adaptive gains: speed up axes moving consistently, damp oscillation.
                    var sameDirection = Math.Sign(gradient[i, d]) == Math.Sign(velocity[i, d]);
                    gains[i, d] = sameDirection ? gains[i, d] * 0.8 : gains[i, d] + 0.2;
                    if (gains[i, d] < 0.01) gains[i, d] = 0.01;

                    velocity[i, d] = momentum * velocity[i, d] - LearningRate * gains[i, d] * gradient[i, d];
                    embedding[i, d] += velocity[i, d];
                }

            // Re-centre so the embedding does not drift.
            var mean = Statistics.Mean(embedding, axis: 0);
            for (var i = 0; i < n; i++)
                for (var d = 0; d < Components; d++)
                    embedding[i, d] -= mean.At(d);
        }

        return embedding;
    }

    private double[,] ComputeAffinities(NdArray x, int n)
    {
        var distances = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var acc = 0.0;
                for (var d = 0; d < x.Shape[1]; d++)
                {
                    var delta = x[i, d] - x[j, d];
                    acc += delta * delta;
                }
                distances[i, j] = distances[j, i] = acc;
            }

        var p = new double[n, n];
        var targetEntropy = Math.Log(Perplexity);

        for (var i = 0; i < n; i++)
        {
            // Binary search for the bandwidth that gives this point the requested perplexity.
            var low = 1e-20;
            var high = 1e20;
            var beta = 1.0;
            var row = new double[n];

            for (var attempt = 0; attempt < 50; attempt++)
            {
                var sum = 0.0;
                for (var j = 0; j < n; j++)
                {
                    if (i == j) { row[j] = 0.0; continue; }
                    row[j] = Math.Exp(-distances[i, j] * beta);
                    sum += row[j];
                }
                if (sum < 1e-300) sum = 1e-300;

                var entropy = 0.0;
                for (var j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    var value = row[j] / sum;
                    if (value > 1e-12) entropy -= value * Math.Log(value);
                }

                if (Math.Abs(entropy - targetEntropy) < 1e-5) break;
                if (entropy > targetEntropy) { low = beta; beta = high > 1e19 ? beta * 2 : (beta + high) / 2; }
                else { high = beta; beta = (beta + low) / 2; }
            }

            var total = 0.0;
            for (var j = 0; j < n; j++) total += row[j];
            if (total < 1e-300) total = 1e-300;
            for (var j = 0; j < n; j++) p[i, j] = row[j] / total;
        }

        // Symmetrise and normalise so the affinities form a joint distribution.
        var grand = 0.0;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var value = (p[i, j] + p[j, i]) / (2.0 * n);
                p[i, j] = value;
                grand += value;
            }
        if (grand > 0)
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                    p[i, j] = Math.Max(p[i, j] / grand, 1e-12);

        return p;
    }

    private (NdArray Gradient, double Divergence) Gradient(double[,] p, NdArray y, int n, double exaggeration)
    {
        var num = new double[n, n];
        var total = 0.0;

        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var acc = 0.0;
                for (var d = 0; d < Components; d++)
                {
                    var delta = y[i, d] - y[j, d];
                    acc += delta * delta;
                }
                // Student-t kernel with one degree of freedom: heavy tails prevent crowding.
                var value = 1.0 / (1.0 + acc);
                num[i, j] = num[j, i] = value;
                total += 2 * value;
            }
        if (total < 1e-300) total = 1e-300;

        var gradient = NdArray.Zeros(n, Components);
        var divergence = 0.0;

        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                if (i == j) continue;
                var q = Math.Max(num[i, j] / total, 1e-12);
                var pij = p[i, j] * exaggeration;
                divergence += pij * Math.Log(pij / q);

                var factor = (pij - q) * num[i, j];
                for (var d = 0; d < Components; d++)
                    gradient[i, d] += 4.0 * factor * (y[i, d] - y[j, d]);
            }

        return (gradient, divergence);
    }
}
