using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Linear;

/// <summary>
/// Ordinary least squares regression.
/// </summary>
/// <remarks>
/// Solved through the pseudo-inverse (SVD) rather than by inverting <c>X'X</c>. That costs more
/// arithmetic but stays well behaved when features are collinear, which is the normal case in
/// real data and the case where the normal equations silently return nonsense.
/// </remarks>
public class LinearRegression(bool fitIntercept = true) : ModelBase, IEstimator
{
    /// <summary>Whether a constant term is fitted.</summary>
    public bool FitIntercept { get; } = fitIntercept;

    /// <summary>One coefficient per feature.</summary>
    public NdArray Coefficients { get; protected set; } = NdArray.Zeros(0);

    /// <summary>The constant term, or zero when <see cref="FitIntercept"/> is false.</summary>
    public double Intercept { get; protected set; }

    /// <inheritdoc />
    public virtual void Fit(NdArray x, NdArray y)
    {
        ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        var design = FitIntercept ? WithInterceptColumn(x) : x;
        var solution = LinAlg.LeastSquares(design, y.Reshape(y.Size));

        if (FitIntercept)
        {
            Intercept = solution.At(FeatureCount);
            Coefficients = solution.Slice(Slice.Range(0, FeatureCount)).Copy();
        }
        else
        {
            Intercept = 0.0;
            Coefficients = solution.Copy();
        }
        IsFitted = true;
    }

    /// <inheritdoc />
    public virtual NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var predictions = LinAlg.Dot(x, Coefficients);
        return Intercept == 0.0 ? predictions : predictions + Intercept;
    }

    /// <summary>Coefficient of determination on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.R2Score(y, Predict(x));

    /// <summary>Appends a column of ones so the intercept can be fitted as another coefficient.</summary>
    protected static NdArray WithInterceptColumn(NdArray x)
    {
        var design = NdArray.Zeros(x.Shape[0], x.Shape[1] + 1);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            for (var j = 0; j < x.Shape[1]; j++) design[i, j] = x[i, j];
            design[i, x.Shape[1]] = 1.0;
        }
        return design;
    }
}

/// <summary>
/// Least squares with an L2 penalty, which shrinks coefficients and stabilises collinear features.
/// </summary>
public sealed class RidgeRegression(double alpha = 1.0, bool fitIntercept = true)
    : LinearRegression(fitIntercept)
{
    /// <summary>Strength of the L2 penalty.</summary>
    public double Alpha { get; } = alpha;

    /// <inheritdoc />
    public override void Fit(NdArray x, NdArray y)
    {
        ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        // Centre the data so the intercept is not penalised along with the slopes.
        var featureMeans = Statistics.Mean(x, axis: 0);
        var targetMean = FitIntercept ? Statistics.Mean(y) : 0.0;

        var centred = NdArray.Zeros(x.Shape[0], FeatureCount);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var j = 0; j < FeatureCount; j++)
                centred[i, j] = x[i, j] - (FitIntercept ? featureMeans.At(j) : 0.0);

        var centredTarget = NdArray.Zeros(y.Size);
        for (var i = 0; i < y.Size; i++) centredTarget.SetAt(i, y.At(i) - targetMean);

        // (X'X + alpha I) w = X'y
        var gram = LinAlg.Dot(centred.T, centred);
        for (var j = 0; j < FeatureCount; j++) gram[j, j] += Alpha;

        Coefficients = LinAlg.Solve(gram, LinAlg.Dot(centred.T, centredTarget));

        if (FitIntercept)
        {
            var offset = 0.0;
            for (var j = 0; j < FeatureCount; j++) offset += Coefficients.At(j) * featureMeans.At(j);
            Intercept = targetMean - offset;
        }
        else Intercept = 0.0;

        IsFitted = true;
    }
}

/// <summary>
/// Least squares with an L1 penalty, fitted by coordinate descent.
/// </summary>
/// <remarks>
/// The L1 penalty drives coefficients exactly to zero, so Lasso doubles as feature selection.
/// Coordinate descent is used because the penalty is not differentiable at zero, which rules out
/// plain gradient descent; each coordinate has a closed-form soft-threshold update.
/// </remarks>
public sealed class LassoRegression(double alpha = 1.0, int maxIterations = 1000, double tolerance = 1e-6)
    : LinearRegression(fitIntercept: true)
{
    /// <summary>Strength of the L1 penalty.</summary>
    public double Alpha { get; } = alpha;

    /// <summary>Maximum coordinate descent sweeps.</summary>
    public int MaxIterations { get; } = maxIterations;

    /// <summary>Convergence threshold on the largest coefficient change.</summary>
    public double Tolerance { get; } = tolerance;

    /// <summary>Number of coefficients driven exactly to zero.</summary>
    public int ZeroCoefficients
    {
        get
        {
            var count = 0;
            for (var j = 0; j < Coefficients.Size; j++) if (Coefficients.At(j) == 0.0) count++;
            return count;
        }
    }

    /// <inheritdoc />
    public override void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        var featureMeans = Statistics.Mean(x, axis: 0);
        var targetMean = Statistics.Mean(y);

        var centred = NdArray.Zeros(samples, FeatureCount);
        for (var i = 0; i < samples; i++)
            for (var j = 0; j < FeatureCount; j++)
                centred[i, j] = x[i, j] - featureMeans.At(j);

        var residual = new double[samples];
        for (var i = 0; i < samples; i++) residual[i] = y.At(i) - targetMean;

        var weights = new double[FeatureCount];
        var columnNorms = new double[FeatureCount];
        for (var j = 0; j < FeatureCount; j++)
        {
            var acc = 0.0;
            for (var i = 0; i < samples; i++) acc += centred[i, j] * centred[i, j];
            columnNorms[j] = acc;
        }

        for (var iteration = 0; iteration < MaxIterations; iteration++)
        {
            var largestChange = 0.0;
            for (var j = 0; j < FeatureCount; j++)
            {
                if (columnNorms[j] < 1e-12) continue;

                // Temporarily remove this coordinate's contribution from the residual.
                var rho = 0.0;
                for (var i = 0; i < samples; i++)
                    rho += centred[i, j] * (residual[i] + centred[i, j] * weights[j]);

                var updated = SoftThreshold(rho, Alpha * samples) / columnNorms[j];
                var delta = updated - weights[j];
                if (delta != 0.0)
                {
                    for (var i = 0; i < samples; i++) residual[i] -= centred[i, j] * delta;
                    weights[j] = updated;
                    largestChange = Math.Max(largestChange, Math.Abs(delta));
                }
            }
            if (largestChange < Tolerance) break;
        }

        Coefficients = new NdArray(weights, FeatureCount);
        var offset = 0.0;
        for (var j = 0; j < FeatureCount; j++) offset += weights[j] * featureMeans.At(j);
        Intercept = targetMean - offset;
        IsFitted = true;
    }

    private static double SoftThreshold(double value, double threshold)
    {
        if (value > threshold) return value - threshold;
        if (value < -threshold) return value + threshold;
        return 0.0;
    }
}

/// <summary>
/// Logistic regression for binary and multi-class problems.
/// </summary>
/// <remarks>
/// Binary problems are fitted directly on the log-odds; multi-class problems use one-vs-rest, so
/// a k-class model is k independent binary fits whose scores are softmaxed at prediction time.
/// Optimisation is batch gradient descent with an L2 penalty - simple, deterministic, and easy to
/// hand to a GPU backend, which is what the CPU-vs-GPU benchmark exercises.
/// </remarks>
public sealed class LogisticRegression(
    double learningRate = 0.1,
    int maxIterations = 1000,
    double l2Penalty = 0.0,
    double tolerance = 1e-7) : ModelBase, IClassifier
{
    private double[][] _weights = [];
    private double[] _intercepts = [];
    private double[] _classes = [];

    /// <summary>Gradient descent step size.</summary>
    public double LearningRate { get; } = learningRate;

    /// <summary>Maximum gradient descent iterations.</summary>
    public int MaxIterations { get; } = maxIterations;

    /// <summary>Strength of the L2 penalty.</summary>
    public double L2Penalty { get; } = l2Penalty;

    /// <summary>Convergence threshold on the change in loss.</summary>
    public double Tolerance { get; } = tolerance;

    /// <summary>Iterations actually run before convergence.</summary>
    public int IterationsRun { get; private set; }

    /// <summary>Final training loss.</summary>
    public double FinalLoss { get; private set; }

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>Coefficients of a binary model.</summary>
    public NdArray Coefficients => _weights.Length == 1
        ? new NdArray((double[])_weights[0].Clone(), FeatureCount)
        : throw new InvalidOperationException("Use CoefficientMatrix for multi-class models.");

    /// <summary>Coefficients of every one-vs-rest sub-model, one per row.</summary>
    public NdArray CoefficientMatrix
    {
        get
        {
            var result = NdArray.Zeros(_weights.Length, FeatureCount);
            for (var k = 0; k < _weights.Length; k++)
                for (var j = 0; j < FeatureCount; j++)
                    result[k, j] = _weights[k][j];
            return result;
        }
    }

    /// <summary>The intercept of a binary model.</summary>
    public double Intercept => _intercepts.Length > 0 ? _intercepts[0] : 0.0;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        _classes = DistinctClasses(y);
        if (_classes.Length < 2) throw new ArgumentException("Logistic regression needs at least two classes.");

        var binaryProblems = _classes.Length == 2 ? 1 : _classes.Length;
        _weights = new double[binaryProblems][];
        _intercepts = new double[binaryProblems];

        var features = x.AsContiguous();

        for (var k = 0; k < binaryProblems; k++)
        {
            // For two classes, the positive label is the second one; otherwise it is class k.
            var positive = binaryProblems == 1 ? _classes[1] : _classes[k];
            var targets = new double[samples];
            for (var i = 0; i < samples; i++) targets[i] = y.At(i) == positive ? 1.0 : 0.0;

            var (weights, intercept, loss, iterations) = FitBinary(features, targets, samples);
            _weights[k] = weights;
            _intercepts[k] = intercept;
            FinalLoss = loss;
            IterationsRun = Math.Max(IterationsRun, iterations);
        }

        IsFitted = true;
    }

    private (double[] Weights, double Intercept, double Loss, int Iterations) FitBinary(
        NdArray x, double[] y, int samples)
    {
        var weights = new double[FeatureCount];
        var intercept = 0.0;
        var previousLoss = double.MaxValue;
        var iteration = 0;
        var loss = 0.0;

        var gradient = new double[FeatureCount];
        var predictions = new double[samples];

        for (; iteration < MaxIterations; iteration++)
        {
            loss = 0.0;
            Array.Clear(gradient);
            var interceptGradient = 0.0;

            for (var i = 0; i < samples; i++)
            {
                var z = intercept;
                for (var j = 0; j < FeatureCount; j++) z += weights[j] * x[i, j];
                var p = MathUtil.Sigmoid(z);
                predictions[i] = p;

                // Cross-entropy, clamped so a saturated prediction cannot produce infinity.
                var clamped = Math.Clamp(p, 1e-15, 1 - 1e-15);
                loss -= y[i] * Math.Log(clamped) + (1 - y[i]) * Math.Log(1 - clamped);

                var error = p - y[i];
                for (var j = 0; j < FeatureCount; j++) gradient[j] += error * x[i, j];
                interceptGradient += error;
            }

            loss /= samples;
            for (var j = 0; j < FeatureCount; j++)
            {
                gradient[j] = gradient[j] / samples + L2Penalty * weights[j];
                weights[j] -= LearningRate * gradient[j];
            }
            intercept -= LearningRate * interceptGradient / samples;

            if (L2Penalty > 0)
            {
                var penalty = 0.0;
                for (var j = 0; j < FeatureCount; j++) penalty += weights[j] * weights[j];
                loss += 0.5 * L2Penalty * penalty;
            }

            if (Math.Abs(previousLoss - loss) < Tolerance) { iteration++; break; }
            previousLoss = loss;
        }

        return (weights, intercept, loss, iteration);
    }

    /// <summary>Raw decision scores, before any probability transform.</summary>
    public NdArray DecisionFunction(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _weights.Length);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var k = 0; k < _weights.Length; k++)
            {
                var z = _intercepts[k];
                for (var j = 0; j < FeatureCount; j++) z += _weights[k][j] * x[i, j];
                result[i, k] = z;
            }
        return result;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var scores = DecisionFunction(x);
        var result = NdArray.Zeros(x.Shape[0], _classes.Length);

        for (var i = 0; i < x.Shape[0]; i++)
        {
            if (_weights.Length == 1)
            {
                var p = MathUtil.Sigmoid(scores[i, 0]);
                result[i, 0] = 1 - p;
                result[i, 1] = p;
                continue;
            }

            // One-vs-rest scores are turned into a distribution with a softmax.
            var row = new double[_classes.Length];
            for (var k = 0; k < _classes.Length; k++) row[k] = scores[i, k];
            var probabilities = MathUtil.Softmax(row);
            for (var k = 0; k < _classes.Length; k++) result[i, k] = probabilities[k];
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
            for (var k = 1; k < _classes.Length; k++)
                if (probabilities[i, k] > probabilities[i, best]) best = k;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}

/// <summary>
/// A linear support vector classifier trained by sub-gradient descent on the hinge loss.
/// </summary>
/// <remarks>
/// The hinge loss only penalises points inside the margin, so the fitted boundary depends on the
/// support vectors rather than on every sample - which is what makes it robust to points that sit
/// far on the correct side. Multi-class problems use one-vs-rest.
/// </remarks>
public sealed class LinearSupportVectorClassifier(
    double c = 1.0,
    int maxIterations = 1000,
    double learningRate = 0.01) : ModelBase, IEstimator
{
    private double[][] _weights = [];
    private double[] _intercepts = [];
    private double[] _classes = [];

    /// <summary>Regularisation trade-off; larger values fit the training data harder.</summary>
    public double C { get; } = c;

    /// <summary>Maximum epochs.</summary>
    public int MaxIterations { get; } = maxIterations;

    /// <summary>Step size.</summary>
    public double LearningRate { get; } = learningRate;

    /// <summary>The class labels, ascending.</summary>
    public IReadOnlyList<double> Classes => _classes;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        _classes = DistinctClasses(y);
        var problems = _classes.Length == 2 ? 1 : _classes.Length;
        _weights = new double[problems][];
        _intercepts = new double[problems];

        for (var k = 0; k < problems; k++)
        {
            var positive = problems == 1 ? _classes[1] : _classes[k];
            var weights = new double[FeatureCount];
            var intercept = 0.0;

            for (var epoch = 0; epoch < MaxIterations; epoch++)
            {
                // Decaying step size keeps the sub-gradient iteration convergent.
                var step = LearningRate / (1.0 + epoch * 0.01);

                for (var i = 0; i < samples; i++)
                {
                    var label = y.At(i) == positive ? 1.0 : -1.0;
                    var margin = intercept;
                    for (var j = 0; j < FeatureCount; j++) margin += weights[j] * x[i, j];

                    if (label * margin < 1.0)
                    {
                        for (var j = 0; j < FeatureCount; j++)
                            weights[j] += step * (C * label * x[i, j] - weights[j] / samples);
                        intercept += step * C * label;
                    }
                    else
                    {
                        for (var j = 0; j < FeatureCount; j++) weights[j] -= step * weights[j] / samples;
                    }
                }
            }

            _weights[k] = weights;
            _intercepts[k] = intercept;
        }
        IsFitted = true;
    }

    /// <summary>Signed distance from the decision boundary.</summary>
    public NdArray DecisionFunction(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0], _weights.Length);
        for (var i = 0; i < x.Shape[0]; i++)
            for (var k = 0; k < _weights.Length; k++)
            {
                var margin = _intercepts[k];
                for (var j = 0; j < FeatureCount; j++) margin += _weights[k][j] * x[i, j];
                result[i, k] = margin;
            }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var scores = DecisionFunction(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            if (_weights.Length == 1)
            {
                result.SetAt(i, scores[i, 0] >= 0 ? _classes[1] : _classes[0]);
                continue;
            }
            var best = 0;
            for (var k = 1; k < _weights.Length; k++) if (scores[i, k] > scores[i, best]) best = k;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}
