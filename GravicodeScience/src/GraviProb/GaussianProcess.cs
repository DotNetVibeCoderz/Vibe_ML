using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>
/// A covariance function: how similar the process believes two inputs' outputs are.
/// </summary>
/// <remarks>
/// The kernel <em>is</em> the model. It encodes every assumption a Gaussian process makes — how
/// smooth the function is, what length scale it varies on, whether it repeats — and choosing it is
/// the modelling decision, not a hyperparameter detail.
/// </remarks>
public abstract class Kernel
{
    /// <summary>Name used in reports.</summary>
    public abstract string Name { get; }

    /// <summary>Covariance between the outputs at two inputs.</summary>
    public abstract double Evaluate(NdArray a, NdArray b);

    /// <summary>The covariance matrix between two sets of inputs, one per row.</summary>
    public NdArray Matrix(NdArray left, NdArray right)
    {
        var result = NdArray.Zeros(left.Shape[0], right.Shape[0]);

        for (var i = 0; i < left.Shape[0]; i++)
            for (var j = 0; j < right.Shape[0]; j++)
                result[i, j] = Evaluate(left.Row(i), right.Row(j));

        return result;
    }

    /// <summary>Squared Euclidean distance between two input vectors.</summary>
    protected static double SquaredDistance(NdArray a, NdArray b)
    {
        var total = 0.0;
        for (var i = 0; i < a.Size; i++)
        {
            var d = a.At(i) - b.At(i);
            total += d * d;
        }
        return total;
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// The squared-exponential (RBF) kernel — the default, and the one that assumes a very smooth function.
/// </summary>
/// <remarks>
/// <para>
/// Samples from a GP with this kernel are infinitely differentiable. That is a strong assumption and
/// often too strong: real processes with kinks or abrupt changes are better served by a Matérn,
/// whose roughness is tunable. The smoothness is why an RBF GP extrapolates so confidently and so
/// badly beyond the data.
/// </para>
/// <para>
/// The length scale sets how far apart two inputs must be before their outputs stop being
/// correlated. Too short and the posterior reverts to the prior mean between every pair of
/// observations; too long and it cannot follow the data.
/// </para>
/// </remarks>
public sealed class RbfKernel(double lengthScale = 1.0, double variance = 1.0) : Kernel
{
    /// <summary>Distance at which correlation falls away.</summary>
    public double LengthScale { get; } = lengthScale > 0
        ? lengthScale
        : throw new ArgumentOutOfRangeException(nameof(lengthScale), "The length scale must be positive.");

    /// <summary>Overall output scale.</summary>
    public double Variance { get; } = variance > 0
        ? variance
        : throw new ArgumentOutOfRangeException(nameof(variance), "The variance must be positive.");

    /// <inheritdoc />
    public override string Name => $"RBF(length={LengthScale:G4}, variance={Variance:G4})";

    /// <inheritdoc />
    public override double Evaluate(NdArray a, NdArray b)
        => Variance * Math.Exp(-0.5 * SquaredDistance(a, b) / (LengthScale * LengthScale));
}

/// <summary>
/// The Matérn kernel, whose smoothness is a parameter rather than an assumption.
/// </summary>
/// <remarks>
/// Only the half-integer cases have a closed form, and those are the ones worth having: ν = 1/2 is
/// the exponential kernel and gives continuous but nowhere-differentiable paths, ν = 3/2 gives once-
/// differentiable ones, and ν = 5/2 twice. Machine-learning practice mostly uses 5/2 as a
/// less-credulous stand-in for the RBF, which is the ν → ∞ limit.
/// </remarks>
public sealed class MaternKernel(double nu = 2.5, double lengthScale = 1.0, double variance = 1.0) : Kernel
{
    /// <summary>Smoothness: 0.5, 1.5 or 2.5.</summary>
    public double Nu { get; } = nu is 0.5 or 1.5 or 2.5
        ? nu
        : throw new ArgumentException("Only nu of 0.5, 1.5 or 2.5 has a closed form.", nameof(nu));

    /// <summary>Distance at which correlation falls away.</summary>
    public double LengthScale { get; } = lengthScale > 0
        ? lengthScale
        : throw new ArgumentOutOfRangeException(nameof(lengthScale));

    /// <summary>Overall output scale.</summary>
    public double Variance { get; } = variance > 0
        ? variance
        : throw new ArgumentOutOfRangeException(nameof(variance));

    /// <inheritdoc />
    public override string Name => $"Matern(nu={Nu:G2}, length={LengthScale:G4})";

    /// <inheritdoc />
    public override double Evaluate(NdArray a, NdArray b)
    {
        var distance = Math.Sqrt(SquaredDistance(a, b)) / LengthScale;

        return Nu switch
        {
            0.5 => Variance * Math.Exp(-distance),
            1.5 => Variance * (1 + Math.Sqrt(3) * distance) * Math.Exp(-Math.Sqrt(3) * distance),
            _ => Variance * (1 + Math.Sqrt(5) * distance + 5 * distance * distance / 3)
                 * Math.Exp(-Math.Sqrt(5) * distance),
        };
    }
}

/// <summary>A kernel for functions that repeat.</summary>
/// <remarks>
/// Distance enters through a sine, so inputs one period apart are perfectly correlated. Worth using
/// only when the periodicity is a genuine belief about the process — it forces the posterior to
/// repeat forever, which is rarely true beyond the data.
/// </remarks>
public sealed class PeriodicKernel(double period = 1.0, double lengthScale = 1.0, double variance = 1.0) : Kernel
{
    /// <summary>The repeat interval.</summary>
    public double Period { get; } = period > 0 ? period : throw new ArgumentOutOfRangeException(nameof(period));

    /// <summary>Smoothness within one period.</summary>
    public double LengthScale { get; } = lengthScale > 0
        ? lengthScale
        : throw new ArgumentOutOfRangeException(nameof(lengthScale));

    /// <summary>Overall output scale.</summary>
    public double Variance { get; } = variance > 0
        ? variance
        : throw new ArgumentOutOfRangeException(nameof(variance));

    /// <inheritdoc />
    public override string Name => $"Periodic(period={Period:G4}, length={LengthScale:G4})";

    /// <inheritdoc />
    public override double Evaluate(NdArray a, NdArray b)
    {
        var distance = Math.Sqrt(SquaredDistance(a, b));
        var sine = Math.Sin(Math.PI * distance / Period);
        return Variance * Math.Exp(-2 * sine * sine / (LengthScale * LengthScale));
    }
}

/// <summary>The sum of two kernels, which models a function that is the sum of two behaviours.</summary>
/// <remarks>
/// Adding kernels is how a trend plus a seasonal cycle is expressed. The sum of two valid covariance
/// functions is a valid covariance function, so no check is needed — unlike most ways of combining
/// them.
/// </remarks>
public sealed class SumKernel(Kernel left, Kernel right) : Kernel
{
    /// <inheritdoc />
    public override string Name => $"({left.Name} + {right.Name})";

    /// <inheritdoc />
    public override double Evaluate(NdArray a, NdArray b) => left.Evaluate(a, b) + right.Evaluate(a, b);
}

/// <summary>A prediction: the posterior mean and its uncertainty.</summary>
/// <param name="Mean">Predicted values, one per test input.</param>
/// <param name="Variance">Predictive variance at each test input.</param>
public readonly record struct GpPrediction(NdArray Mean, NdArray Variance)
{
    /// <summary>Predictive standard deviation at each test input.</summary>
    public NdArray StandardDeviation
    {
        get
        {
            var result = NdArray.Zeros(Variance.Size);
            for (var i = 0; i < Variance.Size; i++) result.SetAt(i, Math.Sqrt(Math.Max(0, Variance.At(i))));
            return result;
        }
    }

    /// <summary>A central credible interval at the given level.</summary>
    public (NdArray Lower, NdArray Upper) Interval(double level = 0.95)
    {
        if (level <= 0 || level >= 1) throw new ArgumentOutOfRangeException(nameof(level));

        var z = MathUtil.NormalQuantile(0.5 + level / 2);
        var deviation = StandardDeviation;

        var lower = NdArray.Zeros(Mean.Size);
        var upper = NdArray.Zeros(Mean.Size);

        for (var i = 0; i < Mean.Size; i++)
        {
            lower.SetAt(i, Mean.At(i) - z * deviation.At(i));
            upper.SetAt(i, Mean.At(i) + z * deviation.At(i));
        }

        return (lower, upper);
    }
}

/// <summary>
/// Gaussian process regression: a distribution over functions, conditioned on data.
/// </summary>
/// <remarks>
/// <para>
/// The idea is to put a prior directly on the function rather than on the parameters of one. Any
/// finite set of inputs has a jointly normal set of outputs, with covariance given by the
/// <see cref="Kernel"/>; conditioning that normal on the observed outputs — the standard Gaussian
/// conditioning formula — gives another normal, and that is the posterior. No optimisation is
/// involved, and the predictive uncertainty comes out with the prediction rather than needing to be
/// estimated separately.
/// </para>
/// <para>
/// <b>The cost is cubic in the number of observations</b>, because the training covariance must be
/// factorised. A few thousand points is the practical ceiling for the exact method implemented here;
/// beyond that, sparse or inducing-point approximations exist and are a different algorithm rather
/// than a tuning of this one.
/// </para>
/// <para>
/// <b>Noise is not optional.</b> The <c>noise</c> term added to the diagonal is both the
/// observation-error model and what keeps the covariance invertible — with duplicate or nearly
/// duplicate inputs it is singular without it, and the factorisation fails. A zero-noise GP that
/// works is one that happened to have well-separated inputs.
/// </para>
/// </remarks>
public sealed class GaussianProcess
{
    private readonly Kernel _kernel;
    private readonly double _noise;
    private NdArray _inputs = NdArray.Zeros(0, 0);
    private NdArray _weights = NdArray.Zeros(0, 0);
    private NdArray _cholesky = NdArray.Zeros(0, 0);
    private double[] _centredTargets = [];
    private double _targetMean;

    /// <summary>Creates a Gaussian process regressor.</summary>
    /// <param name="kernel">The covariance function. Defaults to an RBF with unit length scale.</param>
    /// <param name="noise">
    /// Observation noise variance, added to the diagonal. Also the jitter that keeps the
    /// factorisation stable, so it should stay strictly positive.
    /// </param>
    public GaussianProcess(Kernel? kernel = null, double noise = 1e-6)
    {
        if (noise < 0) throw new ArgumentOutOfRangeException(nameof(noise), "Noise variance cannot be negative.");

        _kernel = kernel ?? new RbfKernel();
        _noise = noise;
    }

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>The kernel in use.</summary>
    public Kernel Kernel => _kernel;

    /// <summary>Number of training points.</summary>
    public int ObservationCount => _inputs.Rank == 2 ? _inputs.Shape[0] : 0;

    /// <summary>
    /// Conditions the process on observed data.
    /// </summary>
    /// <param name="x">Inputs, one per row.</param>
    /// <param name="y">Observed outputs.</param>
    /// <remarks>
    /// The single Cholesky factorisation here is what every later prediction reuses. The targets are
    /// centred first: a GP has a zero prior mean, so without centring it pulls predictions towards
    /// zero rather than towards the data's own level, which looks like a mysterious bias on any
    /// series that does not happen to straddle the origin.
    /// </remarks>
    public GaussianProcess Fit(NdArray x, NdArray y)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);

        if (x.Rank != 2) throw new ArgumentException("Inputs must be a rank 2 array of shape (samples, features).");
        if (x.Shape[0] != y.Size) throw new ArgumentException("There must be one target per input row.");
        if (x.Shape[0] == 0) throw new ArgumentException("There is nothing to fit.");

        _inputs = x.Copy();

        _targetMean = 0.0;
        for (var i = 0; i < y.Size; i++) _targetMean += y.At(i);
        _targetMean /= y.Size;

        var n = x.Shape[0];
        var covariance = _kernel.Matrix(_inputs, _inputs);
        for (var i = 0; i < n; i++) covariance[i, i] += _noise;

        try
        {
            _cholesky = Decomposition.Cholesky(covariance);
        }
        catch (Exception error)
        {
            throw new InvalidOperationException(
                "The training covariance is not positive definite. This usually means the noise term " +
                "is too small for the data — duplicate or near-duplicate inputs make it singular.", error);
        }

        var centred = NdArray.Zeros(n, 1);
        _centredTargets = new double[n];
        for (var i = 0; i < n; i++)
        {
            _centredTargets[i] = y.At(i) - _targetMean;
            centred[i, 0] = _centredTargets[i];
        }

        // α = K⁻¹(y − ȳ), by two triangular solves rather than an inverse.
        _weights = SolveCholesky(centred);

        IsFitted = true;
        return this;
    }

    /// <summary>Posterior mean and variance at new inputs.</summary>
    /// <remarks>
    /// The variance is the prior variance minus what the data explained, which is why it grows back
    /// towards the prior far from any observation. That behaviour — honest uncertainty away from the
    /// data — is the main reason to use a GP rather than a point-estimate regressor.
    /// </remarks>
    public GpPrediction Predict(NdArray x)
    {
        RequireFitted();
        if (x.Rank != 2) throw new ArgumentException("Inputs must be a rank 2 array of shape (samples, features).");

        var cross = _kernel.Matrix(_inputs, x);          // K(train, test)
        var mean = NdArray.Zeros(x.Shape[0]);
        var variance = NdArray.Zeros(x.Shape[0]);

        var product = LinAlg.Dot(Transpose(cross), _weights);
        for (var i = 0; i < x.Shape[0]; i++) mean.SetAt(i, product[i, 0] + _targetMean);

        // v = L⁻¹ K(train, test); the explained variance is ‖v‖² per test point.
        var v = ForwardSubstitute(cross);

        for (var j = 0; j < x.Shape[0]; j++)
        {
            var explained = 0.0;
            for (var i = 0; i < ObservationCount; i++) explained += v[i, j] * v[i, j];

            var prior = _kernel.Evaluate(x.Row(j), x.Row(j));
            variance.SetAt(j, Math.Max(0.0, prior - explained));
        }

        return new GpPrediction(mean, variance);
    }

    /// <summary>Posterior mean only.</summary>
    public NdArray PredictMean(NdArray x) => Predict(x).Mean;

    /// <summary>
    /// The log marginal likelihood of the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The quantity to maximise when choosing kernel hyperparameters. Unlike a training-set
    /// likelihood it does not simply improve as the model gets more flexible: the log-determinant
    /// term is a complexity penalty that grows as the kernel lets the function wiggle, so the
    /// maximum sits at a genuine trade-off rather than at the most flexible model available.
    /// </para>
    /// <para>
    /// This is what <see cref="Optimise"/> searches, and it is worth reporting on its own — a model
    /// whose marginal likelihood is far below another's is not competitive whatever its fit looks
    /// like.
    /// </para>
    /// </remarks>
    public double LogMarginalLikelihood()
    {
        RequireFitted();

        var n = ObservationCount;

        // (y − ȳ)ᵀ K⁻¹ (y − ȳ), which the stored weights already are the right-hand half of.
        var fit = 0.0;
        for (var i = 0; i < n; i++) fit += _weights[i, 0] * _centredTargets[i];

        var logDeterminant = 0.0;
        for (var i = 0; i < n; i++) logDeterminant += Math.Log(_cholesky[i, i]);
        logDeterminant *= 2;

        return -0.5 * fit - 0.5 * logDeterminant - 0.5 * n * Math.Log(2 * Math.PI);
    }

    /// <summary>Draws whole functions from the posterior, evaluated at the given inputs.</summary>
    /// <remarks>
    /// Distinct from the marginal intervals <see cref="GpPrediction.Interval"/> reports: those give
    /// a band each point independently stays within, while these are coherent functions. A band
    /// cannot tell you whether the function wiggles inside it or stays flat, and for anything that
    /// depends on the shape rather than the level, the samples are what to look at.
    /// </remarks>
    public NdArray SamplePosterior(NdArray x, int count, GraviRandom rng)
    {
        RequireFitted();
        ArgumentNullException.ThrowIfNull(rng);

        var prediction = Predict(x);
        var m = x.Shape[0];

        var cross = _kernel.Matrix(_inputs, x);
        var v = ForwardSubstitute(cross);

        var covariance = NdArray.Zeros(m, m);
        for (var i = 0; i < m; i++)
            for (var j = 0; j < m; j++)
            {
                var explained = 0.0;
                for (var k = 0; k < ObservationCount; k++) explained += v[k, i] * v[k, j];

                covariance[i, j] = _kernel.Evaluate(x.Row(i), x.Row(j)) - explained;
            }

        // The posterior covariance is positive semi-definite in exact arithmetic and routinely a
        // hair short of it in floating point, so the factorisation needs a jitter to survive.
        for (var i = 0; i < m; i++) covariance[i, i] += 1e-9;

        var posterior = new MultivariateNormal(prediction.Mean, covariance);
        return posterior.Sample(rng, count);
    }

    /// <summary>
    /// Picks an RBF length scale and noise level by grid search on the log marginal likelihood.
    /// </summary>
    /// <remarks>
    /// A grid rather than a gradient method, deliberately. The marginal likelihood is not concave in
    /// the hyperparameters and has genuine local optima with different interpretations — one
    /// explaining the data as signal, another as noise — so a local optimiser lands wherever it
    /// started. A coarse grid at least sees both.
    /// </remarks>
    public static GaussianProcess Optimise(NdArray x, NdArray y,
        IReadOnlyList<double>? lengthScales = null, IReadOnlyList<double>? noises = null)
    {
        lengthScales ??= [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
        noises ??= [1e-6, 1e-4, 1e-2, 0.1, 0.5];

        GaussianProcess? best = null;
        var bestScore = double.NegativeInfinity;

        foreach (var lengthScale in lengthScales)
            foreach (var noise in noises)
            {
                try
                {
                    var candidate = new GaussianProcess(new RbfKernel(lengthScale), noise).Fit(x, y);
                    var score = candidate.LogMarginalLikelihood();

                    if (double.IsNaN(score) || score <= bestScore) continue;

                    bestScore = score;
                    best = candidate;
                }
                catch (InvalidOperationException)
                {
                    // A covariance that will not factorise at this noise level: skip the point
                    // rather than abandoning the search.
                }
            }

        return best ?? throw new InvalidOperationException(
            "No hyperparameter combination produced a usable model. Try a larger noise term.");
    }

    // ------------------------------------------------------------------ helpers

    private void RequireFitted()
    {
        if (!IsFitted) throw new InvalidOperationException("The process must be fitted before use.");
    }

    /// <summary>Solves <c>K z = b</c> using the stored Cholesky factor.</summary>
    private NdArray SolveCholesky(NdArray b)
    {
        var y = ForwardSubstitute(b);
        return BackSubstitute(y);
    }

    /// <summary>Solves <c>L y = b</c> for each column of <paramref name="b"/>.</summary>
    private NdArray ForwardSubstitute(NdArray b)
    {
        var n = ObservationCount;
        var columns = b.Shape[1];
        var y = NdArray.Zeros(n, columns);

        for (var c = 0; c < columns; c++)
            for (var i = 0; i < n; i++)
            {
                var sum = b[i, c];
                for (var j = 0; j < i; j++) sum -= _cholesky[i, j] * y[j, c];
                y[i, c] = sum / _cholesky[i, i];
            }

        return y;
    }

    /// <summary>Solves <c>Lᵀ z = y</c> for each column.</summary>
    private NdArray BackSubstitute(NdArray y)
    {
        var n = ObservationCount;
        var columns = y.Shape[1];
        var z = NdArray.Zeros(n, columns);

        for (var c = 0; c < columns; c++)
            for (var i = n - 1; i >= 0; i--)
            {
                var sum = y[i, c];
                for (var j = i + 1; j < n; j++) sum -= _cholesky[j, i] * z[j, c];
                z[i, c] = sum / _cholesky[i, i];
            }

        return z;
    }

    private static NdArray Transpose(NdArray a)
    {
        var result = NdArray.Zeros(a.Shape[1], a.Shape[0]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[j, i] = a[i, j];
        return result;
    }
}
