using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>How <see cref="Optimize.Shgo(Func{double[], double}, Bounds, IReadOnlyList{IMinimizeConstraint}?, int?, int?, OptimizeCallback?, ShgoMinimizerKwargs?, ShgoOptions?, ShgoSamplingMethod)"/> places its sampling points.</summary>
public enum ShgoSamplingMethod
{
    /// <summary>Corners and centroid of the box, then uniform simplicial refinement (SciPy's default).</summary>
    Simplicial,
    /// <summary>Unscrambled Sobol' points (n rounded up to a power of two), connected by a Delaunay triangulation.</summary>
    Sobol,
    /// <summary>Unscrambled Halton points, connected by a Delaunay triangulation.</summary>
    Halton,
}

/// <summary>The local minimiser SHGO runs from each pool point (SciPy's <c>minimizer_kwargs</c>).</summary>
public sealed record ShgoMinimizerKwargs
{
    /// <summary>Local method; SLSQP by default. Bounds and constraints are passed only to methods that accept them.</summary>
    public MinimizeMethod Method { get; init; } = MinimizeMethod.Slsqp;

    /// <summary>Gradient of the objective.</summary>
    public Func<double[], double[]>? Jac { get; init; }

    /// <summary>Hessian of the objective, for methods that use one.</summary>
    public Func<double[], NdArray>? Hess { get; init; }

    /// <summary>Hessian-vector product, for methods that use one.</summary>
    public Func<double[], double[], double[]>? Hessp { get; init; }

    /// <summary>Tolerance passed to <c>Minimize</c>.</summary>
    public double? Tol { get; init; }

    /// <summary>
    /// Method options. Null means SciPy's default for SHGO: <c>ftol = 1e-12</c> when the method is
    /// SLSQP (a much tighter tolerance than SLSQP's own 1e-6, so the local minima are polished).
    /// </summary>
    public MinimizeOptions? Options { get; init; }
}

/// <summary>Global options for SHGO (SciPy's <c>options</c> dictionary).</summary>
public sealed record ShgoOptions
{
    /// <summary>Stop after this many iterations (setting any of the five stopping options disables <c>iters</c>).</summary>
    public int? MaxIter { get; init; }

    /// <summary>Stop once this many feasible sampling points have been evaluated.</summary>
    public int? MaxFev { get; init; }

    /// <summary>Stop once this many sampling points (feasible or not) have been generated.</summary>
    public int? MaxEv { get; init; }

    /// <summary>Stop after this many seconds (checked between iterations).</summary>
    public double? MaxTime { get; init; }

    /// <summary>Known global minimum: stop once a value within <see cref="FTol"/> (relative) of it is found.</summary>
    public double? FMin { get; init; }

    /// <summary>Relative tolerance for <see cref="FMin"/> (absolute when FMin is 0).</summary>
    public double FTol { get; init; } = 1e-4;

    /// <summary>Stop when the number of local minima grows by at most this much in an iteration.</summary>
    public int? MinHgrd { get; init; }

    /// <summary>Minimise from the pool after every iteration (true) or only once at the end.</summary>
    public bool MinimizeEveryIter { get; init; } = true;

    /// <summary>Limit the local minimisations per iteration to this many (null: the whole pool).</summary>
    public int? LocalIter { get; init; }

    /// <summary>
    /// Keep infeasible sampling points in the triangulation with f = +∞ (true, SciPy's default) or drop
    /// them before triangulating (Sobol/Halton only).
    /// </summary>
    public bool InfinityConstraints { get; init; } = true;
}

public static partial class Optimize
{
    /// <summary>
    /// Simplicial homology global optimisation (<c>scipy.optimize.shgo</c>; Endres, Sandrock &amp;
    /// Focke 2018). Samples the box, connects the samples into a simplicial complex, takes as
    /// starting points every vertex lower than all its neighbours (the star-domain test that finds
    /// one candidate per basin of the sampled surface), and runs a local minimiser from each.
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="bounds">Search box. Infinite sides are replaced by ±1e50 for sampling, as SciPy does.</param>
    /// <param name="constraints">
    /// Constraints. Inequalities decide which sampling points are feasible; all constraints go to
    /// the local minimiser when it accepts constraints (SLSQP, COBYLA, TrustConstr).
    /// </param>
    /// <param name="n">Points per iteration; 100 means 2^dim + 1 for simplicial sampling (one cube). Null with
    /// simplicial sampling refines every sub-cube completely each iteration.</param>
    /// <param name="iters">Sampling iterations (null: 1).</param>
    /// <param name="callback">Passed to the local minimiser.</param>
    /// <param name="minimizerKwargs">Local minimiser settings.</param>
    /// <param name="options">Stopping rules and other options.</param>
    /// <param name="samplingMethod">Sampling scheme.</param>
    /// <returns>
    /// <see cref="OptimizeResult.X"/>/<see cref="OptimizeResult.Fun"/> the best local minimum,
    /// <see cref="OptimizeResult.Xl"/>/<see cref="OptimizeResult.Funl"/> every local minimum found
    /// (sorted by value; two starting points converging to the same minimum give two rows, as in SciPy),
    /// <see cref="OptimizeResult.Nfev"/> sampling plus local evaluations, <see cref="OptimizeResult.Nlfev"/>
    /// the local part, <see cref="OptimizeResult.Nit"/> iterations, <see cref="OptimizeResult.Tnev"/> points sampled.
    /// </returns>
    public static OptimizeResult Shgo(Func<double[], double> f, Bounds bounds, IReadOnlyList<IMinimizeConstraint>? constraints = null,
        int? n = 100, int? iters = 1, OptimizeCallback? callback = null, ShgoMinimizerKwargs? minimizerKwargs = null,
        ShgoOptions? options = null, ShgoSamplingMethod samplingMethod = ShgoSamplingMethod.Simplicial)
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(bounds);
        return new OptimizeShgo(f, bounds, constraints, n, iters, callback, minimizerKwargs, options, samplingMethod).Run();
    }

    /// <summary>
    /// <see cref="Shgo(Func{double[], double}, Bounds, IReadOnlyList{IMinimizeConstraint}?, int?, int?, OptimizeCallback?, ShgoMinimizerKwargs?, ShgoOptions?, ShgoSamplingMethod)"/>
    /// with SciPy's legacy <c>[(lo, hi), ...]</c> bounds.
    /// </summary>
    public static OptimizeResult Shgo(Func<double[], double> f, (double Lo, double Hi)[] bounds, IReadOnlyList<IMinimizeConstraint>? constraints = null,
        int? n = 100, int? iters = 1, OptimizeCallback? callback = null, ShgoMinimizerKwargs? minimizerKwargs = null,
        ShgoOptions? options = null, ShgoSamplingMethod samplingMethod = ShgoSamplingMethod.Simplicial)
        => Shgo(f, new Bounds(bounds.Select(b => b.Lo).ToArray(), bounds.Select(b => b.Hi).ToArray()), constraints, n, iters, callback,
            minimizerKwargs, options, samplingMethod);
}

internal sealed class OptimizeShgo
{
    private readonly Func<double[], double> _f;
    private readonly int _dim;
    private readonly double[] _lb, _ub;          // sampling box (±1e50 for infinite sides)
    private readonly double[] _userLb, _userUb;  // as given, for the local minimiser
    private readonly IReadOnlyList<IMinimizeConstraint> _cons;
    private readonly OptimizeConstraintSet? _gcons;
    private readonly OptimizeCallback? _callback;
    private readonly ShgoMinimizerKwargs _mk;
    private readonly ShgoOptions _o;
    private readonly ShgoSamplingMethod _sampling;
    private readonly int? _iters;
    private readonly int? _n;
    private readonly System.Diagnostics.Stopwatch _clock = System.Diagnostics.Stopwatch.StartNew();

    private ShgoComplex? _hc;
    private readonly ShgoVertexCache _dv = new();      // Delaunay vertices
    private readonly List<double[]> _points = [];       // Delaunay sample coordinates
    private int _qmcIndex;

    private int _itersDone, _nSampled, _fn, _nfevSampling, _hgr;
    private bool _stopGlobal;
    private readonly Dictionary<double[], OptimizeResult> _lmc = new(ShgoKeyComparer.Instance);
    private readonly List<double[]> _xlMaps = [];
    private readonly List<double> _fMaps = [];
    private int _nlfev, _nljev, _nlhev;

    public OptimizeShgo(Func<double[], double> f, Bounds bounds, IReadOnlyList<IMinimizeConstraint>? constraints, int? n, int? iters,
        OptimizeCallback? callback, ShgoMinimizerKwargs? mk, ShgoOptions? o, ShgoSamplingMethod sampling)
    {
        _f = f; _dim = bounds.Length; _callback = callback; _sampling = sampling;
        if (_dim == 0) throw new ArgumentException("bounds must have at least one variable.");
        _userLb = (double[])bounds.Lb.Clone(); _userUb = (double[])bounds.Ub.Clone();
        _lb = new double[_dim]; _ub = new double[_dim];
        for (var i = 0; i < _dim; i++)
        {
            _lb[i] = double.IsFinite(_userLb[i]) ? _userLb[i] : -1e50;
            _ub[i] = double.IsFinite(_userUb[i]) ? _userUb[i] : 1e50;
        }
        _cons = constraints ?? [];
        foreach (var c in _cons) if (c is null) throw new ArgumentException("constraints contains a null entry.");
        if (_cons.Count > 0)
        {
            var mid = new double[_dim];
            for (var i = 0; i < _dim; i++) mid[i] = 0.5 * (_lb[i] + _ub[i]);
            _gcons = new OptimizeConstraintSet(_cons, mid, 1.4901161193847656e-08);
            if (_gcons.MIn == 0) _gcons = null;
        }
        _mk = mk ?? new ShgoMinimizerKwargs();
        _o = o ?? new ShgoOptions();

        // SciPy's defaulting of n and iters.
        if (n is null && iters is null && sampling == ShgoSamplingMethod.Simplicial) n = (1 << _dim) + 1;
        iters ??= 1;
        if (n is null && sampling != ShgoSamplingMethod.Simplicial) n = 100;
        if (n == 100 && sampling == ShgoSamplingMethod.Simplicial) n = (1 << _dim) + 1;
        if (n is <= 0) throw new ArgumentException("n must be positive.");
        if (iters <= 0) throw new ArgumentException("iters must be positive.");
        if (_o.MaxIter is not null || _o.MaxFev is not null || _o.MaxEv is not null || _o.MinHgrd is not null || _o.FMin is not null)
            iters = null;
        if (sampling == ShgoSamplingMethod.Sobol) n = (int)Math.Pow(2, Math.Ceiling(Math.Log2(n!.Value)));
        _n = n; _iters = iters;
        if (sampling == ShgoSamplingMethod.Simplicial) _hc = new ShgoComplex(_dim, _lb, _ub);
    }

    private ShgoVertexCache Vertices => _hc?.V ?? _dv;

    public OptimizeResult Run()
    {
        while (!_stopGlobal)
        {
            Iterate();
            StoppingCriteria();
        }
        if (!_o.MinimizeEveryIter) FindMinima();

        var res = new OptimizeResult { Nit = _itersDone, Nlfev = _nlfev, Nljev = _nljev, Nhev = _nlhev, Tnev = _nSampled };
        if (_xlMaps.Count == 0)
        {
            var (xLow, fLow) = FindLowestVertex();
            res.Success = false;
            res.Status = 1;
            res.Message = $"Failed to find a feasible minimizer point. Lowest sampling point = {(fLow is null ? "None" : fLow.Value.ToString("R"))}";
            res.X = xLow ?? [];
            res.Fun = fLow ?? double.NaN;
            res.Nfev = _fn;
            return res;
        }
        var order = Enumerable.Range(0, _fMaps.Count).OrderBy(i => _fMaps[i]).ToArray();
        var xl = new double[order.Length * _dim];
        for (var k = 0; k < order.Length; k++) Array.Copy(_xlMaps[order[k]], 0, xl, k * _dim, _dim);
        res.Xl = new NdArray(xl, order.Length, _dim);
        res.Funl = order.Select(i => _fMaps[i]).ToArray();
        res.X = (double[])_xlMaps[order[0]].Clone();
        res.Fun = _fMaps[order[0]];
        res.Nfev = _fn + _nlfev;
        res.Success = true;
        res.Status = 0;
        res.Message = "Optimization terminated successfully.";
        return res;
    }

    private void Iterate()
    {
        if (_sampling == ShgoSamplingMethod.Simplicial) IterateHypercube();
        else IterateDelaunay();
        if (_o.MinimizeEveryIter) FindMinima();
        _itersDone++;
    }

    private void IterateHypercube()
    {
        if (_n is null)
        {
            _hc!.RefineAll();
            _nSampled = _hc.V.Count;
        }
        else
        {
            _hc!.Refine(_n.Value);
            _nSampled += _n.Value;
        }
        ProcessPools(_hc.V);
        _fn = _nfevSampling;
    }

    private void IterateDelaunay()
    {
        var count = _n!.Value;
        var fresh = new List<double[]>();
        var unit = _sampling == ShgoSamplingMethod.Sobol
            ? OptimizeQmc.Sobol(_qmcIndex + count, _dim, new GraviRandom(0), shift: false).Skip(_qmcIndex).ToArray()
            : Enumerable.Range(_qmcIndex, count).Select(HaltonPoint).ToArray();
        _qmcIndex += count;
        foreach (var u in unit)
        {
            var x = new double[_dim];
            for (var i = 0; i < _dim; i++) x[i] = u[i] * (_ub[i] - _lb[i]) + _lb[i];
            fresh.Add(x);
        }
        // Local minima found so far join the triangulation, as SciPy appends them to the samples.
        foreach (var xl in _xlMaps)
            if (!_dv.Contains(xl) && !fresh.Any(p => ShgoKeyComparer.Instance.Equals(p, xl))) fresh.Add((double[])xl.Clone());
        if (!_o.InfinityConstraints && _gcons is not null) fresh.RemoveAll(p => !IsFeasible(p));
        foreach (var p in fresh)
            if (!_dv.Contains(p)) { _points.Add(p); _ = _dv[p]; }
        _nSampled += count;

        if (_dim < 2)
        {
            var sorted = _points.OrderBy(p => p[0]).ToList();
            for (var i = 1; i < sorted.Count; i++) _dv[sorted[i - 1]].Connect(_dv[sorted[i]]);
        }
        else if (_points.Count > _dim + 1)
        {
            // Like SciPy's incremental Qhull, edges are only ever added: an edge that stops being
            // Delaunay after new points arrive stays in the graph.
            foreach (var s in ShgoDelaunay.Triangulate(_points, _dim))
                for (var a = 0; a < s.Length; a++)
                    for (var b = a + 1; b < s.Length; b++) _dv[_points[s[a]]].Connect(_dv[_points[s[b]]]);
        }
        ProcessPools(_dv);
        _fn = _nfevSampling;
    }

    private static readonly int[] HaltonPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];

    private double[] HaltonPoint(int index)
    {
        if (_dim > HaltonPrimes.Length) throw new NotSupportedException($"Halton sampling supports up to {HaltonPrimes.Length} dimensions.");
        var r = new double[_dim];
        for (var j = 0; j < _dim; j++)
        {
            int b = HaltonPrimes[j], i = index;
            double fct = 1, v = 0;
            while (i > 0) { fct /= b; v += fct * (i % b); i /= b; }
            r[j] = v;
        }
        return r;
    }

    private bool IsFeasible(double[] x)
    {
        if (_gcons is null) return true;
        foreach (var g in _gcons.Eval((double[])x.Clone()).In) if (g < 0.0) return false;
        return true;
    }

    private void ProcessPools(ShgoVertexCache cache)
    {
        foreach (var v in cache.Pending)
        {
            v.Evaluated = true;
            v.Feasible = IsFeasible(v.X);
            if (!v.Feasible) { v.F = double.PositiveInfinity; continue; }
            var fv = _f((double[])v.X.Clone());
            _nfevSampling++;
            v.F = double.IsNaN(fv) ? double.PositiveInfinity : fv;
        }
        cache.Pending.Clear();
    }

    private (double[]? X, double? F) FindLowestVertex()
    {
        double fLow = double.PositiveInfinity;
        double[]? xLow = null;
        foreach (var v in Vertices.Ordered)
            if (v.F < fLow) { fLow = v.F; xLow = v.X; }
        for (var k = 0; k < _fMaps.Count; k++)
            if (_fMaps[k] < fLow) { fLow = _fMaps[k]; xLow = _xlMaps[k]; }
        return double.IsPositiveInfinity(fLow) ? (null, null) : ((double[])xLow!.Clone(), fLow);
    }

    private void StoppingCriteria()
    {
        if (_o.MaxIter is { } mi && _itersDone >= mi) _stopGlobal = true;
        if (_iters is { } it && _itersDone >= it) _stopGlobal = true;
        if (_o.MaxFev is { } mf && _fn >= mf) _stopGlobal = true;
        if (_o.MaxEv is { } me && _nSampled >= me) _stopGlobal = true;
        if (_o.MaxTime is { } mt && _clock.Elapsed.TotalSeconds >= mt) _stopGlobal = true;
        if (_o.FMin is { } fmin)
        {
            var (_, fLow) = FindLowestVertex();
            if (fLow is { } fl)
            {
                if (fmin == 0.0) { if (fl <= _o.FTol) _stopGlobal = true; }
                else
                {
                    var pe = (fl - fmin) / Math.Abs(fmin);
                    if (fl <= fmin || pe <= _o.FTol) _stopGlobal = true;
                }
            }
        }
        if (_o.MinHgrd is { } mh && _xlMaps.Count > 0)
        {
            var hgrd = _xlMaps.Count - _hgr;
            _hgr = _xlMaps.Count;
            if (hgrd <= mh) _stopGlobal = true;
        }
    }

    private void FindMinima()
    {
        // Minimiser pool: vertices lower than all their neighbours, excluding local minima already found.
        var pool = new List<ShgoVertex>();
        foreach (var v in Vertices.Ordered)
        {
            var inLmc = false;
            foreach (var xl in _xlMaps) if (ShgoKeyComparer.Instance.Equals(v.X, xl)) { inLmc = true; break; }
            if (inLmc) continue;
            if (v.Minimiser()) pool.Add(v);
        }
        if (pool.Count == 0) return;
        pool = pool.OrderBy(v => v.F).ToList();

        // Lowest first; afterwards always the pool point farthest from the latest local minimum.
        var last = Minimize(pool[0]);
        pool.RemoveAt(0);
        var budget = _o.LocalIter;
        while (true)
        {
            StoppingCriteria();
            if (budget is not null)
            {
                budget--;
                if (budget <= 0) break;
            }
            if (pool.Count == 0) break;
            var ref0 = last.X;
            var far = 0;
            var farD = double.NegativeInfinity;
            for (var k = 0; k < pool.Count; k++)
            {
                double d2 = 0;
                for (var i = 0; i < _dim; i++) { var t = pool[k].X[i] - ref0[i]; d2 += t * t; }
                if (d2 >= farD) { farD = d2; far = k; }
            }
            last = Minimize(pool[far]);
            pool.RemoveAt(far);
        }
    }

    private static bool AcceptsBounds(MinimizeMethod m) => m is MinimizeMethod.NelderMead or MinimizeMethod.Powell or MinimizeMethod.LBfgsB
        or MinimizeMethod.Tnc or MinimizeMethod.Cobyla or MinimizeMethod.Slsqp or MinimizeMethod.TrustConstr;

    private static bool AcceptsConstraints(MinimizeMethod m) => m is MinimizeMethod.Cobyla or MinimizeMethod.Slsqp or MinimizeMethod.TrustConstr;

    private OptimizeResult Minimize(ShgoVertex start)
    {
        if (_lmc.TryGetValue(start.X, out var cached)) return cached;
        var lb = (double[])_lb.Clone();
        var ub = (double[])_ub.Clone();
        if (_sampling == ShgoSamplingMethod.Simplicial)
        {
            // Local bounds: the box spanned by the start vertex's neighbours (SciPy's construct_lcb_simplicial).
            foreach (var vn in start.Nn)
                for (var i = 0; i < _dim; i++)
                {
                    var xi = vn.X[i];
                    if (xi < start.X[i] && xi > lb[i]) lb[i] = xi;
                    if (xi > start.X[i] && xi < ub[i]) ub[i] = xi;
                }
        }
        for (var i = 0; i < _dim; i++)
        {
            if (!double.IsFinite(_userLb[i]) && lb[i] == -1e50) lb[i] = double.NegativeInfinity;
            if (!double.IsFinite(_userUb[i]) && ub[i] == 1e50) ub[i] = double.PositiveInfinity;
        }
        var method = _mk.Method;
        var opts = _mk.Options ?? (method == MinimizeMethod.Slsqp ? new SlsqpOptions { Ftol = 1e-12 } : null);
        var lres = Optimize.Minimize(_f, (double[])start.X.Clone(), method, _mk.Jac, _mk.Hess, _mk.Hessp,
            AcceptsBounds(method) ? new Bounds(lb, ub) : null,
            AcceptsConstraints(method) && _cons.Count > 0 ? _cons : null,
            _mk.Tol, opts, _callback);
        _nlfev += lres.Nfev;
        _nljev += lres.Njev;
        _nlhev += lres.Nhev;
        _lmc[(double[])start.X.Clone()] = lres;
        _xlMaps.Add((double[])lres.X.Clone());
        _fMaps.Add(lres.Fun);
        return lres;
    }
}
