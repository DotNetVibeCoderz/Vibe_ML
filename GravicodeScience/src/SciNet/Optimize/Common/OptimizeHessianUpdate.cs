using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Which matrix a <see cref="HessianUpdateStrategy"/> approximates.</summary>
public enum HessianApproxType
{
    /// <summary>The Hessian B itself (what <c>trust-constr</c> uses).</summary>
    Hess,
    /// <summary>The inverse Hessian H = B⁻¹ (what line-search quasi-Newton methods use).</summary>
    InvHess,
}

/// <summary>What <see cref="BfgsHessianUpdate"/> does when the curvature condition fails.</summary>
public enum BfgsExceptionStrategy
{
    /// <summary>Skip the update (SciPy's default <c>'skip_update'</c>, min curvature 1e-8).</summary>
    SkipUpdate,
    /// <summary>Powell's damped update, interpolating y towards B s (<c>'damp_update'</c>, min curvature 0.2).</summary>
    DampUpdate,
}

/// <summary>
/// Quasi-Newton approximation of a Hessian (or its inverse) built from steps and gradient
/// differences, as <c>scipy.optimize.HessianUpdateStrategy</c>. <c>trust-constr</c> uses one for the
/// objective and one per nonlinear constraint when no analytic Hessian is supplied.
/// </summary>
/// <remarks>
/// Instances are stateful: <see cref="Initialize"/> resets them, and a solver initialises the
/// instance it is given. Do not hand the same instance to two roles in one solve (the objective and
/// a constraint, say) — the updates would interleave into one meaningless matrix, so
/// <c>trust-constr</c> refuses it.
/// </remarks>
public abstract class HessianUpdateStrategy
{
    /// <summary>Resets the approximation to the identity for an n-dimensional problem.</summary>
    public abstract void Initialize(int n, HessianApproxType approxType);

    /// <summary>Updates with the step <c>s = x₁ − x₀</c> and gradient change <c>y = ∇f(x₁) − ∇f(x₀)</c>.</summary>
    public abstract void Update(double[] deltaX, double[] deltaGrad);

    /// <summary>The product of the current approximation with <paramref name="p"/>.</summary>
    public abstract double[] Dot(double[] p);

    /// <summary>The current approximation as a dense symmetric matrix.</summary>
    public abstract NdArray GetMatrix();
}

/// <summary>
/// Base for the full-matrix updates (<see cref="BfgsHessianUpdate"/>, <see cref="Sr1HessianUpdate"/>):
/// stores the dense n×n approximation and applies SciPy's "auto" initial scaling on the first update.
/// </summary>
public abstract class FullHessianUpdateStrategy : HessianUpdateStrategy
{
    private readonly double? _initScale;
    private bool _firstIteration;

    /// <summary>Dense row-major approximation (B or H depending on <see cref="ApproxType"/>).</summary>
    protected double[] M = [];

    /// <summary>Problem dimension.</summary>
    protected int N { get; private set; }

    /// <summary>Which matrix is approximated.</summary>
    protected HessianApproxType ApproxType { get; private set; }

    /// <summary>
    /// <paramref name="initScale"/> null is SciPy's <c>init_scale='auto'</c>: on the first update the
    /// identity is scaled by <c>yᵀy / |yᵀs|</c> (Nocedal &amp; Wright eq. 6.20), which makes the
    /// initial model the right order of magnitude; a number scales the identity by that value.
    /// </summary>
    protected FullHessianUpdateStrategy(double? initScale) => _initScale = initScale;

    /// <inheritdoc />
    public override void Initialize(int n, HessianApproxType approxType)
    {
        if (n < 1) throw new ArgumentException("n must be positive.");
        N = n; ApproxType = approxType; _firstIteration = true;
        M = new double[n * n];
        for (var i = 0; i < n; i++) M[i * n + i] = 1;
    }

    /// <summary>SciPy's automatic scale <c>yᵀy/|yᵀs|</c> (or its reciprocal for the inverse).</summary>
    protected double AutoScale(double[] dx, double[] dg)
    {
        double ss = 0, yy = 0, ys = 0;
        for (var i = 0; i < N; i++) { ss += dx[i] * dx[i]; yy += dg[i] * dg[i]; ys += dg[i] * dx[i]; }
        ys = Math.Abs(ys);
        if (ys == 0 || yy == 0 || ss == 0) return 1;
        return ApproxType == HessianApproxType.Hess ? yy / ys : ys / yy;
    }

    /// <inheritdoc />
    public override void Update(double[] deltaX, double[] deltaGrad)
    {
        if (M.Length == 0) throw new InvalidOperationException("Call Initialize before Update.");
        if (deltaX.Length != N || deltaGrad.Length != N) throw new ArgumentException($"Update vectors must have length {N}.");
        var allZero = true;
        foreach (var v in deltaX) if (v != 0) { allZero = false; break; }
        if (allZero) return;
        allZero = true;
        foreach (var v in deltaGrad) if (v != 0) { allZero = false; break; }
        // A zero gradient change means the function is linear along s: no curvature information.
        if (allZero) return;
        if (_firstIteration)
        {
            var scale = _initScale ?? AutoScale(deltaX, deltaGrad);
            for (var i = 0; i < M.Length; i++) M[i] *= scale;
            _firstIteration = false;
        }
        UpdateImplementation(deltaX, deltaGrad);
    }

    /// <summary>The method-specific update, after scaling.</summary>
    protected abstract void UpdateImplementation(double[] deltaX, double[] deltaGrad);

    /// <inheritdoc />
    public override double[] Dot(double[] p)
    {
        var r = new double[N];
        OptimizeDense.MatVec(M, N, N, p, r);
        return r;
    }

    /// <summary>Accumulates M·p into <paramref name="acc"/> without allocating.</summary>
    internal void AddDot(ReadOnlySpan<double> p, Span<double> acc)
    {
        for (var i = 0; i < N; i++)
        {
            double s = 0;
            var row = M.AsSpan(i * N, N);
            for (var j = 0; j < N; j++) s += row[j] * p[j];
            acc[i] += s;
        }
    }

    /// <inheritdoc />
    public override NdArray GetMatrix() => new((double[])M.Clone(), N, N);

    /// <summary>M += alpha·x·xᵀ (BLAS <c>syr</c>, applied to the full symmetric matrix).</summary>
    protected void Syr(double alpha, ReadOnlySpan<double> x)
    {
        for (var i = 0; i < N; i++)
        {
            var ax = alpha * x[i];
            for (var j = 0; j < N; j++) M[i * N + j] += ax * x[j];
        }
    }

    /// <summary>M += alpha·(x·yᵀ + y·xᵀ) (BLAS <c>syr2</c>).</summary>
    protected void Syr2(double alpha, ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        for (var i = 0; i < N; i++)
            for (var j = 0; j < N; j++) M[i * N + j] += alpha * (x[i] * y[j] + y[i] * x[j]);
    }

    /// <summary>Resets M to <paramref name="scale"/>·I.</summary>
    protected void ResetToScaledIdentity(double scale)
    {
        Array.Clear(M);
        for (var i = 0; i < N; i++) M[i * N + i] = scale;
    }
}

/// <summary>
/// The BFGS update (<c>scipy.optimize.BFGS</c>). It keeps the approximation positive definite by
/// skipping (or damping) any update whose curvature <c>sᵀy</c> is not sufficiently positive — the
/// right behaviour for an objective, a poor one for an indefinite constraint Hessian (use
/// <see cref="Sr1HessianUpdate"/> there if BFGS stalls).
/// </summary>
public sealed class BfgsHessianUpdate : FullHessianUpdateStrategy
{
    /// <summary>What happens when the curvature condition fails.</summary>
    public BfgsExceptionStrategy ExceptionStrategy { get; }

    /// <summary>The curvature threshold: <c>sᵀy ≤ minCurvature · sᵀBs</c> triggers the exception strategy.</summary>
    public double MinCurvature { get; }

    /// <summary>Creates a BFGS update; <paramref name="minCurvature"/> defaults to 1e-8 (skip) or 0.2 (damp).</summary>
    public BfgsHessianUpdate(BfgsExceptionStrategy exceptionStrategy = BfgsExceptionStrategy.SkipUpdate, double? minCurvature = null, double? initScale = null)
        : base(initScale)
    {
        ExceptionStrategy = exceptionStrategy;
        MinCurvature = minCurvature ?? (exceptionStrategy == BfgsExceptionStrategy.SkipUpdate ? 1e-8 : 0.2);
    }

    /// <inheritdoc />
    protected override void UpdateImplementation(double[] deltaX, double[] deltaGrad)
    {
        var hess = ApproxType == HessianApproxType.Hess;
        var w = hess ? deltaX : deltaGrad;
        var z = (double[])(hess ? deltaGrad : deltaX).Clone();
        var wz = OptimizeDense.Dot(w, z);
        var mw = Dot(w);
        var wmw = OptimizeDense.Dot(mw, w);
        if (wmw <= 0.0)
        {
            // The approximation lost definiteness to rounding: restart from a scaled identity.
            ResetToScaledIdentity(AutoScale(deltaX, deltaGrad));
            mw = Dot(w);
            wmw = OptimizeDense.Dot(mw, w);
        }
        if (wz <= MinCurvature * wmw)
        {
            if (ExceptionStrategy == BfgsExceptionStrategy.SkipUpdate) return;
            var factor = (1 - MinCurvature) / (1 - wz / wmw);
            for (var i = 0; i < z.Length; i++) z[i] = factor * z[i] + (1 - factor) * mw[i];
            wz = OptimizeDense.Dot(w, z);
        }
        if (hess)
        {
            Syr(1.0 / wz, z);
            Syr(-1.0 / wmw, mw);
        }
        else
        {
            Syr2(-1.0 / wz, z, mw);
            Syr((wz + wmw) / (wz * wz), z);
        }
    }
}

/// <summary>
/// The symmetric rank-one update (<c>scipy.optimize.SR1</c>). Unlike BFGS it can represent an
/// indefinite Hessian, which suits constraint Hessians and non-convex Lagrangians; updates whose
/// denominator is tiny relative to the vectors are skipped (Nocedal &amp; Wright eq. 6.26).
/// </summary>
public sealed class Sr1HessianUpdate : FullHessianUpdateStrategy
{
    /// <summary>Relative threshold below which the update is skipped.</summary>
    public double MinDenominator { get; }

    /// <summary>Creates an SR1 update.</summary>
    public Sr1HessianUpdate(double minDenominator = 1e-8, double? initScale = null) : base(initScale) => MinDenominator = minDenominator;

    /// <inheritdoc />
    protected override void UpdateImplementation(double[] deltaX, double[] deltaGrad)
    {
        var hess = ApproxType == HessianApproxType.Hess;
        var w = hess ? deltaX : deltaGrad;
        var z = hess ? deltaGrad : deltaX;
        var mw = Dot(w);
        var r = new double[z.Length];
        for (var i = 0; i < r.Length; i++) r[i] = z[i] - mw[i];
        var den = OptimizeDense.Dot(w, r);
        if (Math.Abs(den) <= MinDenominator * OptimizeDense.Norm2(w) * OptimizeDense.Norm2(r)) return;
        Syr(1 / den, r);
    }
}
