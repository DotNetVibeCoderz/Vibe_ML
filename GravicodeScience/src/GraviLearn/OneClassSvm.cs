using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Anomaly;

/// <summary>The kernel a one-class SVM measures similarity with.</summary>
public enum SvmKernel
{
    /// <summary>Plain inner product. The boundary is a hyperplane, which rarely suits novelty detection.</summary>
    Linear,

    /// <summary>Gaussian radial basis function. The boundary wraps the data; the usual choice.</summary>
    Rbf,
}

/// <summary>
/// One-class SVM: learns the shape of "normal" from unlabelled data, then flags what falls outside.
/// </summary>
/// <remarks>
/// <para>
/// Novelty detection is not classification with one class missing. There are no negative examples
/// to learn a boundary <em>between</em>; the task is to find a region containing most of the
/// training data and as little else as possible. Schölkopf's formulation does that by separating
/// the data from the origin in feature space with maximum margin, which — with an RBF kernel, where
/// every point has unit norm and non-negative similarity — becomes a boundary that wraps the data.
/// </para>
/// <para>
/// <b><see cref="Nu"/> is the dial that matters.</b> It is simultaneously an upper bound on the
/// fraction of training points that end up outside the boundary and a lower bound on the fraction
/// that become support vectors. So <c>nu: 0.05</c> is a statement that around 5% of the training
/// data is contamination worth excluding — not a tolerance to be tuned until the answer looks
/// right. If the training data is genuinely clean, a small nu is still needed: nu of 0 has no
/// solution.
/// </para>
/// <para>
/// Scale the features first. An RBF kernel is a function of Euclidean distance, so a column
/// measured in thousands determines every similarity on its own and <see cref="Gamma"/> becomes
/// meaningless.
/// </para>
/// </remarks>
public sealed class OneClassSvm : ModelBase
{
    private double[,] _supportVectors = new double[0, 0];
    private double[] _coefficients = [];
    private double _rho;

    /// <summary>Creates a detector.</summary>
    /// <param name="nu">
    /// Roughly the fraction of training data allowed to fall outside the boundary. Must be in
    /// (0, 1].
    /// </param>
    /// <param name="kernel">Similarity measure.</param>
    /// <param name="gamma">
    /// RBF width. <c>null</c> uses the "scale" heuristic — <c>1 / (features · variance)</c> — which
    /// adapts to the data rather than assuming it was standardised.
    /// </param>
    /// <param name="tolerance">
    /// Convergence tolerance on the KKT conditions. Loosening this past about 1e-5 is visible in
    /// the results, not just the runtime: at 1e-3 the nu-property stops holding, because the
    /// multipliers are still far enough from optimal that points which should sit at the box bound
    /// have not got there.
    /// </param>
    /// <param name="maxIterations">Cap on optimiser passes.</param>
    public OneClassSvm(double nu = 0.5, SvmKernel kernel = SvmKernel.Rbf, double? gamma = null,
        double tolerance = 1e-6, int maxIterations = 10000)
    {
        if (nu <= 0 || nu > 1) throw new ArgumentOutOfRangeException(nameof(nu), "nu must be in (0, 1].");

        Nu = nu;
        Kernel = kernel;
        Gamma = gamma ?? 0;
        _explicitGamma = gamma is not null;
        Tolerance = tolerance;
        MaxIterations = maxIterations;
    }

    private readonly bool _explicitGamma;

    /// <summary>The fraction of training data allowed outside the boundary.</summary>
    public double Nu { get; }

    /// <summary>The kernel in use.</summary>
    public SvmKernel Kernel { get; }

    /// <summary>The RBF width actually used, resolved from the data when it was not given.</summary>
    public double Gamma { get; private set; }

    /// <summary>KKT convergence tolerance.</summary>
    public double Tolerance { get; }

    /// <summary>Cap on optimiser passes.</summary>
    public int MaxIterations { get; }

    /// <summary>How many training points ended up on or outside the boundary.</summary>
    public int SupportVectorCount => _coefficients.Length;

    /// <summary>The learned offset. The boundary is where the decision function crosses zero.</summary>
    public double Offset => _rho;

    /// <summary>Learns the region of normality from unlabelled data.</summary>
    public OneClassSvm Fit(NdArray x)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];

        if (!_explicitGamma) Gamma = ScaleHeuristic(x);

        var data = ToRows(x, samples, FeatureCount);
        var kernel = KernelMatrix(data, samples);
        var alpha = Solve(kernel, samples);

        // Only points with a non-zero multiplier appear in the decision function; for a well-chosen
        // nu that is a small fraction of the data, and the rest can be forgotten.
        var support = Enumerable.Range(0, samples).Where(i => alpha[i] > 1e-8).ToArray();

        _supportVectors = new double[support.Length, FeatureCount];
        _coefficients = new double[support.Length];

        for (var i = 0; i < support.Length; i++)
        {
            _coefficients[i] = alpha[support[i]];
            for (var f = 0; f < FeatureCount; f++) _supportVectors[i, f] = data[support[i]][f];
        }

        _rho = ResolveRho(kernel, alpha, samples);
        IsFitted = true;
        return this;
    }

    /// <summary>
    /// The signed distance from the boundary: positive inside the learned region, negative outside.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Prefer this to <see cref="Predict"/> when ranking. The sign answers "is this an anomaly";
    /// the magnitude answers "how confident is that", which is what an alert queue needs to sort on.
    /// </para>
    /// <para>
    /// <b>Do not judge the model by scoring its own training data.</b> A support vector appears in
    /// its own decision function — the <c>k(x, x) = 1</c> term — so an isolated training point gets
    /// credit for being near itself and always scores higher than an identical point held out.
    /// For a point whose kernel to everything else is zero the KKT conditions give <c>g = α</c>,
    /// so when <c>ρ &lt; 1/(νn)</c> it settles at <c>α = ρ</c> and scores exactly zero: it lands
    /// <em>on</em> the boundary rather than outside it, and which side of zero it reports is then
    /// decided by floating-point noise. Held out it scores <c>−ρ</c> and is flagged cleanly.
    /// Evaluate on data the model has not seen.
    /// </para>
    /// </remarks>
    public NdArray DecisionFunction(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var sum = 0.0;
            for (var s = 0; s < _coefficients.Length; s++)
                sum += _coefficients[s] * KernelValue(x, i, s);

            result.SetAt(i, sum - _rho);
        }

        return result;
    }

    /// <summary>Labels each row: <c>1</c> for normal, <c>-1</c> for an anomaly.</summary>
    public NdArray Predict(NdArray x)
    {
        var scores = DecisionFunction(x);
        var result = NdArray.Zeros(scores.Size);
        for (var i = 0; i < scores.Size; i++) result.SetAt(i, scores.At(i) >= 0 ? 1 : -1);
        return result;
    }

    // ------------------------------------------------------------------- optimiser

    /// <summary>
    /// Solves the dual by sequential minimal optimisation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The problem is <c>min ½ αᵀKα</c> subject to <c>0 ≤ αᵢ ≤ 1/(νn)</c> and <c>Σα = 1</c>. The
    /// equality constraint is why the updates come in pairs: moving one multiplier alone would
    /// break the sum, so two move in opposite directions and the sum is preserved exactly at every
    /// step rather than restored afterwards.
    /// </para>
    /// <para>
    /// The initialisation matters more than it looks. Spreading the mass over the first
    /// <c>⌈νn⌉</c> points gives a feasible starting point — every multiplier within its box and the
    /// sum already 1 — so the optimiser never has to repair feasibility.
    /// </para>
    /// </remarks>
    private double[] Solve(double[,] kernel, int samples)
    {
        var upper = 1.0 / (Nu * samples);
        var alpha = new double[samples];

        var active = (int)Math.Ceiling(Nu * samples);
        for (var i = 0; i < active; i++) alpha[i] = 1.0 / active;

        // Gradient of ½αᵀKα is Kα, maintained incrementally: a pair update touches two columns.
        var gradient = new double[samples];
        for (var i = 0; i < samples; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < samples; j++) sum += kernel[i, j] * alpha[j];
            gradient[i] = sum;
        }

        for (var iteration = 0; iteration < MaxIterations; iteration++)
        {
            // The most violating pair: the point that would most like to gain mass, and the one
            // that would most like to shed it.
            var low = -1;
            var high = -1;
            var lowGradient = double.PositiveInfinity;
            var highGradient = double.NegativeInfinity;

            for (var i = 0; i < samples; i++)
            {
                if (alpha[i] < upper - 1e-12 && gradient[i] < lowGradient) { lowGradient = gradient[i]; low = i; }
                if (alpha[i] > 1e-12 && gradient[i] > highGradient) { highGradient = gradient[i]; high = i; }
            }

            if (low < 0 || high < 0 || highGradient - lowGradient < Tolerance) break;

            // Move mass from high to low. The unconstrained optimum along this direction, then
            // clipped to keep both multipliers in their boxes.
            var curvature = kernel[low, low] + kernel[high, high] - 2 * kernel[low, high];
            if (curvature <= 1e-12) curvature = 1e-12;

            var step = (highGradient - lowGradient) / curvature;
            step = Math.Min(step, Math.Min(upper - alpha[low], alpha[high]));
            if (step <= 1e-14) break;

            alpha[low] += step;
            alpha[high] -= step;

            for (var i = 0; i < samples; i++)
                gradient[i] += step * (kernel[i, low] - kernel[i, high]);
        }

        return alpha;
    }

    /// <summary>
    /// Recovers the offset from the points sitting strictly inside their box.
    /// </summary>
    /// <remarks>
    /// Those are the ones exactly on the boundary, where the decision function is zero by the KKT
    /// conditions, so rho is their decision value. Averaging over all of them is steadier than
    /// taking any single one. When none is strictly interior — which happens at large nu — the
    /// fallback is the smallest value among the support vectors, which keeps the boundary
    /// enclosing them.
    /// </remarks>
    private double ResolveRho(double[,] kernel, double[] alpha, int samples)
    {
        var upper = 1.0 / (Nu * samples);
        var total = 0.0;
        var count = 0;
        var smallest = double.PositiveInfinity;

        for (var i = 0; i < samples; i++)
        {
            if (alpha[i] <= 1e-8) continue;

            var value = 0.0;
            for (var j = 0; j < samples; j++) value += alpha[j] * kernel[i, j];

            smallest = Math.Min(smallest, value);

            if (alpha[i] < upper - 1e-8) { total += value; count++; }
        }

        if (count > 0) return total / count;
        return double.IsPositiveInfinity(smallest) ? 0.0 : smallest;
    }

    // ------------------------------------------------------------------- kernels

    private double[,] KernelMatrix(double[][] data, int samples)
    {
        var kernel = new double[samples, samples];

        for (var i = 0; i < samples; i++)
            for (var j = i; j < samples; j++)
            {
                var value = Evaluate(data[i], data[j]);
                kernel[i, j] = value;
                kernel[j, i] = value;
            }

        return kernel;
    }

    private double Evaluate(double[] a, double[] b)
    {
        if (Kernel == SvmKernel.Linear)
        {
            var dot = 0.0;
            for (var f = 0; f < a.Length; f++) dot += a[f] * b[f];
            return dot;
        }

        var squared = 0.0;
        for (var f = 0; f < a.Length; f++)
        {
            var d = a[f] - b[f];
            squared += d * d;
        }
        return Math.Exp(-Gamma * squared);
    }

    private double KernelValue(NdArray x, int row, int support)
    {
        if (Kernel == SvmKernel.Linear)
        {
            var dot = 0.0;
            for (var f = 0; f < FeatureCount; f++) dot += x[row, f] * _supportVectors[support, f];
            return dot;
        }

        var squared = 0.0;
        for (var f = 0; f < FeatureCount; f++)
        {
            var d = x[row, f] - _supportVectors[support, f];
            squared += d * d;
        }
        return Math.Exp(-Gamma * squared);
    }

    /// <summary>The "scale" heuristic: 1 / (features × variance of the whole matrix).</summary>
    private static double ScaleHeuristic(NdArray x)
    {
        var mean = 0.0;
        for (var i = 0; i < x.Size; i++) mean += x.At(i);
        mean /= x.Size;

        var variance = 0.0;
        for (var i = 0; i < x.Size; i++)
        {
            var d = x.At(i) - mean;
            variance += d * d;
        }
        variance /= x.Size;

        return variance > 1e-12 ? 1.0 / (x.Shape[1] * variance) : 1.0;
    }

    private static double[][] ToRows(NdArray x, int samples, int features)
    {
        var rows = new double[samples][];
        for (var i = 0; i < samples; i++)
        {
            rows[i] = new double[features];
            for (var f = 0; f < features; f++) rows[i][f] = x[i, f];
        }
        return rows;
    }
}
