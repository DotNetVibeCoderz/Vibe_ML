using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Linear;

/// <summary>
/// Logistic regression trained directly on a sparse matrix, without densifying it.
/// </summary>
/// <remarks>
/// <para>
/// The reason this exists is a memory wall, not a speed one. A bag-of-words matrix over a 30,000
/// word vocabulary is around 0.1% non-zero; densified, 50,000 documents is 12&#160;GB of doubles,
/// almost all of them zeros that cost the same to store and to multiply as anything else. In CSR
/// the same matrix is a few tens of megabytes.
/// </para>
/// <para>
/// The algorithm is the same gradient descent <see cref="LogisticRegression"/> runs; what changes
/// is that every loop walks the non-zeros of a row rather than its full width. That makes an epoch
/// cost <c>O(nnz)</c> instead of <c>O(rows × features)</c> — on a 0.1% dense matrix, a thousandfold
/// less arithmetic for exactly the same answer.
/// </para>
/// <para>
/// <b>The one thing that is not sparse is the weight vector.</b> It has one entry per feature
/// whether or not that feature ever occurs, so the vocabulary still has to fit in memory. That is
/// almost never the binding constraint — 30,000 doubles is nothing — but it is worth knowing that
/// this bounds the feature count, not the document count.
/// </para>
/// <para>
/// An L2 penalty is applied to the accumulated gradient rather than by decaying every weight each
/// step. Decaying is what a dense implementation does and it is <c>O(features)</c> per sample,
/// which would put the dense cost straight back in and make the whole exercise pointless.
/// </para>
/// </remarks>
public sealed class SparseLogisticRegression(
    double learningRate = 0.1,
    int maxIterations = 200,
    double l2Penalty = 0.0,
    double tolerance = 1e-7)
{
    private double[][] _weights = [];
    private double[] _intercepts = [];
    private double[] _classes = [];

    /// <summary>Gradient descent step size.</summary>
    public double LearningRate { get; } = learningRate;

    /// <summary>Maximum iterations.</summary>
    public int MaxIterations { get; } = maxIterations;

    /// <summary>Strength of the L2 penalty.</summary>
    public double L2Penalty { get; } = l2Penalty;

    /// <summary>Convergence threshold on the change in loss.</summary>
    public double Tolerance { get; } = tolerance;

    /// <summary>Number of features seen during training.</summary>
    public int FeatureCount { get; private set; }

    /// <summary>Iterations actually run.</summary>
    public int IterationsRun { get; private set; }

    /// <summary>Final training loss.</summary>
    public double FinalLoss { get; private set; }

    /// <summary>True once fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>The distinct class labels, ascending.</summary>
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>
    /// Trains on a sparse feature matrix.
    /// </summary>
    /// <param name="x">Features in CSR, one row per sample.</param>
    /// <param name="y">Class labels, one per row.</param>
    /// <remarks>
    /// Multi-class is one-versus-rest, matching the dense implementation: <c>k</c> binary problems,
    /// each trained independently, and the highest score wins at prediction time. That is not the
    /// same model as a softmax over all classes and can disagree with it on the boundaries, which
    /// is the accepted trade for being able to train each problem separately.
    /// </remarks>
    public SparseLogisticRegression Fit(SparseMatrix x, NdArray y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        if (x.Rows == 0) throw new ArgumentException("There is nothing to train on.", nameof(x));
        if (y.Size != x.Rows)
            throw new ArgumentException($"Got {y.Size} labels for {x.Rows} rows.", nameof(y));

        FeatureCount = x.Columns;
        _classes = [.. y.ToArray().Distinct().Order()];

        var problems = _classes.Length <= 2 ? 1 : _classes.Length;
        _weights = new double[problems][];
        _intercepts = new double[problems];

        var totalLoss = 0.0;

        for (var k = 0; k < problems; k++)
        {
            var positive = _classes.Length <= 2 ? _classes[^1] : _classes[k];

            var targets = new double[x.Rows];
            for (var i = 0; i < x.Rows; i++) targets[i] = y.At(i) == positive ? 1.0 : 0.0;

            var (weights, intercept, loss, iterations) = FitBinary(x, targets);

            _weights[k] = weights;
            _intercepts[k] = intercept;
            totalLoss += loss;
            IterationsRun = Math.Max(IterationsRun, iterations);
        }

        FinalLoss = totalLoss / problems;
        IsFitted = true;
        return this;
    }

    /// <summary>
    /// One binary problem, by full-batch gradient descent over the non-zeros.
    /// </summary>
    /// <remarks>
    /// Two passes per iteration and both are <c>O(nnz)</c>: one to score every row, one to
    /// accumulate the gradient. Scoring cannot be folded into the gradient pass because the
    /// gradient needs the residual, which needs the score.
    /// </remarks>
    private (double[] Weights, double Intercept, double Loss, int Iterations) FitBinary(
        SparseMatrix x, double[] targets)
    {
        var weights = new double[x.Columns];
        var intercept = 0.0;

        var rowPointers = x.RowPointers;
        var columns = x.ColumnIndices;
        var values = x.Values;

        var residual = new double[x.Rows];
        var gradient = new double[x.Columns];

        var previousLoss = double.MaxValue;
        var iteration = 0;
        var loss = 0.0;

        for (; iteration < MaxIterations; iteration++)
        {
            loss = 0.0;

            // Pass one: score each row over its non-zeros only.
            for (var i = 0; i < x.Rows; i++)
            {
                var z = intercept;
                for (var k = rowPointers[i]; k < rowPointers[i + 1]; k++) z += weights[columns[k]] * values[k];

                var p = Sigmoid(z);
                residual[i] = p - targets[i];

                // Clamped so a saturated probability cannot produce a negative infinity here.
                loss -= targets[i] * Math.Log(Math.Max(p, 1e-15))
                        + (1 - targets[i]) * Math.Log(Math.Max(1 - p, 1e-15));
            }

            loss /= x.Rows;

            // Pass two: accumulate the gradient, again over non-zeros only. Only the features that
            // actually occur are touched, which is the whole saving.
            Array.Clear(gradient);
            var interceptGradient = 0.0;

            for (var i = 0; i < x.Rows; i++)
            {
                interceptGradient += residual[i];
                for (var k = rowPointers[i]; k < rowPointers[i + 1]; k++)
                    gradient[columns[k]] += residual[i] * values[k];
            }

            var scale = 1.0 / x.Rows;

            // The penalty goes into the gradient rather than decaying every weight each step:
            // decaying is O(features) per update and would reintroduce the dense cost.
            for (var j = 0; j < x.Columns; j++)
                weights[j] -= LearningRate * (gradient[j] * scale + L2Penalty * weights[j]);

            intercept -= LearningRate * interceptGradient * scale;

            if (Math.Abs(previousLoss - loss) < Tolerance) { iteration++; break; }
            previousLoss = loss;
        }

        return (weights, intercept, loss, iteration);
    }

    /// <summary>Class probabilities, one row per sample and one column per class.</summary>
    public NdArray PredictProbabilities(SparseMatrix x)
    {
        RequireFitted(x);

        var rowPointers = x.RowPointers;
        var columns = x.ColumnIndices;
        var values = x.Values;

        var scores = NdArray.Zeros(x.Rows, _weights.Length);

        for (var i = 0; i < x.Rows; i++)
            for (var k = 0; k < _weights.Length; k++)
            {
                var z = _intercepts[k];
                for (var e = rowPointers[i]; e < rowPointers[i + 1]; e++)
                    z += _weights[k][columns[e]] * values[e];

                scores[i, k] = Sigmoid(z);
            }

        if (_weights.Length > 1) NormaliseRows(scores);
        return scores;
    }

    /// <summary>Predicted class per row.</summary>
    public NdArray Predict(SparseMatrix x)
    {
        var probabilities = PredictProbabilities(x);
        var result = NdArray.Zeros(x.Rows);

        for (var i = 0; i < x.Rows; i++)
        {
            if (_weights.Length == 1)
            {
                result.SetAt(i, probabilities[i, 0] >= 0.5 ? _classes[^1] : _classes[0]);
                continue;
            }

            var best = 0;
            for (var k = 1; k < _weights.Length; k++)
                if (probabilities[i, k] > probabilities[i, best]) best = k;

            result.SetAt(i, _classes[best]);
        }

        return result;
    }

    /// <summary>Fraction of rows predicted correctly.</summary>
    public double Score(SparseMatrix x, NdArray y)
    {
        var predictions = Predict(x);
        var correct = 0;

        for (var i = 0; i < y.Size; i++)
            if (Math.Abs(predictions.At(i) - y.At(i)) < 1e-9) correct++;

        return (double)correct / y.Size;
    }

    /// <summary>
    /// The learned coefficients, densified.
    /// </summary>
    /// <remarks>
    /// The weight vector is dense by construction, so this is not a densification of the training
    /// data — it is one row per binary problem and one column per feature, which is exactly what
    /// was stored.
    /// </remarks>
    public NdArray Coefficients
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

    /// <summary>The intercepts, one per binary problem.</summary>
    public NdArray Intercepts => NdArray.FromValues(_intercepts);

    /// <summary>
    /// The features that push hardest towards a class, largest weight first.
    /// </summary>
    /// <remarks>
    /// The practical reason to keep a linear model on text: a coefficient is directly readable as
    /// "this word moves the decision this far", which no tree ensemble or transformer offers.
    /// </remarks>
    public IReadOnlyList<(int Feature, double Weight)> TopFeatures(int count = 20, int problem = 0)
    {
        RequireFittedOnly();

        return [.. _weights[problem]
            .Select((weight, feature) => (Feature: feature, Weight: weight))
            .OrderByDescending(p => p.Weight)
            .Take(count)];
    }

    private static double Sigmoid(double z)
        // Split on the sign so the exponent is always negative: exp of a large positive number
        // overflows, and the algebraically identical other branch does not.
        => z >= 0 ? 1.0 / (1.0 + Math.Exp(-z)) : Math.Exp(z) / (1.0 + Math.Exp(z));

    private static void NormaliseRows(NdArray scores)
    {
        for (var i = 0; i < scores.Shape[0]; i++)
        {
            var total = 0.0;
            for (var k = 0; k < scores.Shape[1]; k++) total += scores[i, k];
            if (total <= 0) continue;

            for (var k = 0; k < scores.Shape[1]; k++) scores[i, k] /= total;
        }
    }

    private void RequireFitted(SparseMatrix x)
    {
        RequireFittedOnly();

        if (x.Columns != FeatureCount)
            throw new ArgumentException(
                $"Model was fitted on {FeatureCount} features but received {x.Columns}.", nameof(x));
    }

    private void RequireFittedOnly()
    {
        if (!IsFitted) throw new InvalidOperationException("The model must be fitted before use.");
    }
}
