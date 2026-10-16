using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <c>Optimize.Brute</c>.</summary>
/// <param name="X0">Best point (after the finishing local search, when one ran).</param>
/// <param name="Fval">Objective at <paramref name="X0"/>.</param>
/// <param name="Grid">Grid coordinates, shape (n, Ns₁, …, Nsₙ).</param>
/// <param name="Jout">Objective on the grid, shape (Ns₁, …, Nsₙ).</param>
public sealed record BruteResult(double[] X0, double Fval, NdArray Grid, NdArray Jout);

/// <summary>Local minimiser hook for <see cref="Optimize.BasinHopping"/>: minimise f from x.</summary>
public delegate OptimizeResult BasinHoppingMinimizer(Func<double[], double> f, double[] x);

public static partial class Optimize
{
    /// <summary>
    /// Generalized simulated annealing with Tsallis–Stariolo visiting and acceptance distributions
    /// plus local searches (Xiang et al. 1997), <c>scipy.optimize.dual_annealing</c>.
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="bounds">Finite search box.</param>
    /// <param name="maxiter">Global iterations.</param>
    /// <param name="initialTemp">Initial temperature.</param>
    /// <param name="restartTempRatio">Re-annealing starts when the temperature falls to initialTemp·ratio.</param>
    /// <param name="visit">Visiting parameter q_v in (1, 3].</param>
    /// <param name="accept">Acceptance parameter q_a in (−10⁴, −5].</param>
    /// <param name="maxfun">Soft limit on evaluations (local searches included).</param>
    /// <param name="seed">Random seed.</param>
    /// <param name="noLocalSearch">Classic generalized simulated annealing without local searches.</param>
    /// <param name="callback">Called with (x, f, context: 0 annealing, 1 local search, 2 dual annealing); return true to stop.</param>
    /// <param name="x0">Starting point.</param>
    /// <param name="localSearchOptions">Options for the L-BFGS-B local search (default maxiter = min(max(6n, 100), 1000)).</param>
    public static OptimizeResult DualAnnealing(Func<double[], double> f, Bounds bounds, int maxiter = 1000, double initialTemp = 5230.0,
        double restartTempRatio = 2e-5, double visit = 2.62, double accept = -5.0, int maxfun = 10_000_000, int? seed = null,
        bool noLocalSearch = false, Func<double[], double, int, bool>? callback = null, double[]? x0 = null, LBfgsBOptions? localSearchOptions = null)
        => OptimizeDualAnnealing.Run(f, bounds, maxiter, initialTemp, restartTempRatio, visit, accept, maxfun, seed, noLocalSearch, callback, x0, localSearchOptions);

    /// <summary>
    /// Basin hopping (Wales &amp; Doye 1997), <c>scipy.optimize.basinhopping</c>: random perturbation,
    /// local minimisation, Metropolis acceptance at temperature T, adaptive step size.
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="x0">Starting point.</param>
    /// <param name="niter">Basin-hopping iterations.</param>
    /// <param name="t">Metropolis temperature (0: accept only improvements).</param>
    /// <param name="stepsize">Initial uniform displacement half-width.</param>
    /// <param name="minimizer">Local minimiser; default <c>Minimize(f, x)</c> with <paramref name="method"/>.</param>
    /// <param name="method">Local method when <paramref name="minimizer"/> is null (BFGS, or L-BFGS-B when bounds are given).</param>
    /// <param name="bounds">Bounds passed to the default local minimiser.</param>
    /// <param name="takeStep">Custom step (x → new x); disables step adaptation.</param>
    /// <param name="acceptTest">Extra test (fNew, xNew, fOld, xOld) → accept?</param>
    /// <param name="callback">Called with (x, f, accepted) each iteration; return true to stop.</param>
    /// <param name="interval">Step size adaptation interval.</param>
    /// <param name="niterSuccess">Stop after this many iterations without a new global minimum.</param>
    /// <param name="seed">Random seed.</param>
    /// <param name="targetAcceptRate">Target acceptance rate of the adaptation.</param>
    /// <param name="stepwiseFactor">Step size multiplier of the adaptation.</param>
    public static OptimizeResult BasinHopping(Func<double[], double> f, double[] x0, int niter = 100, double t = 1.0, double stepsize = 0.5,
        BasinHoppingMinimizer? minimizer = null, MinimizeMethod? method = null, Bounds? bounds = null, Func<double[], double[]>? takeStep = null,
        Func<double, double[], double, double[], bool>? acceptTest = null, Func<double[], double, bool, bool>? callback = null, int interval = 50,
        int? niterSuccess = null, int? seed = null, double targetAcceptRate = 0.5, double stepwiseFactor = 0.9)
        => OptimizeBasinHopping.Run(f, x0, niter, t, stepsize, minimizer, method, bounds, takeStep, acceptTest, callback, interval, niterSuccess, seed, targetAcceptRate, stepwiseFactor);

    /// <summary>
    /// DIRECT (Jones, Perttunen &amp; Stuckman 1993) and its locally biased DIRECT-L variant
    /// (Gablonsky &amp; Kelley 2001), <c>scipy.optimize.direct</c>: deterministic Lipschitzian
    /// partitioning of the box into hyperrectangles.
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="bounds">Finite box.</param>
    /// <param name="eps">Minimal relative improvement a rectangle must promise to be divided.</param>
    /// <param name="maxfun">Evaluation limit (default 1000 n).</param>
    /// <param name="maxiter">Iteration limit.</param>
    /// <param name="locallyBiased">DIRECT-L (true) or the original DIRECT.</param>
    /// <param name="fGlobal">Known global minimum, if any: stop within <paramref name="fGlobalRelTol"/> of it.</param>
    /// <param name="fGlobalRelTol">Relative tolerance for <paramref name="fGlobal"/>.</param>
    /// <param name="volTol">Stop when the best rectangle's volume (fraction of the box) falls below this.</param>
    /// <param name="lenTol">Stop when half the best rectangle's normalised size measure falls below this.</param>
    /// <param name="callback">Called each iteration with the best point; return true to stop.</param>
    public static OptimizeResult Direct(Func<double[], double> f, Bounds bounds, double eps = 1e-4, int? maxfun = null, int maxiter = 1000,
        bool locallyBiased = true, double fGlobal = double.NegativeInfinity, double fGlobalRelTol = 1e-4, double volTol = 1e-16, double lenTol = 1e-6,
        Func<double[], bool>? callback = null)
        => OptimizeDirect.Run(f, bounds, eps, maxfun ?? 1000 * bounds.Length, maxiter, locallyBiased, fGlobal, fGlobalRelTol, volTol, lenTol, callback);

    /// <summary>
    /// Exhaustive grid search, <c>scipy.optimize.brute</c>: <paramref name="ns"/> points from lo to hi
    /// (inclusive) per variable, then an optional unconstrained Nelder–Mead polish from the best grid
    /// point (SciPy's default <c>finish=fmin</c>).
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="ranges">(lo, hi) per variable.</param>
    /// <param name="ns">Points per axis.</param>
    /// <param name="finish">Polish with Nelder–Mead.</param>
    /// <param name="workers">Parallel evaluations (f must be thread-safe).</param>
    public static BruteResult Brute(Func<double[], double> f, (double Lo, double Hi)[] ranges, int ns = 20, bool finish = true, int workers = 1)
        => OptimizeBrute.Run(f, ranges.Select(r => Enumerable.Range(0, ns).Select(i => ns == 1 ? r.Lo : r.Lo + (r.Hi - r.Lo) * i / (ns - 1)).ToArray()).ToArray(), finish, workers);

    /// <summary>
    /// Grid search over explicit axes (the <c>slice(lo, hi, step)</c> form: the end point is excluded,
    /// as with <c>numpy.arange</c>).
    /// </summary>
    public static BruteResult Brute(Func<double[], double> f, (double Lo, double Hi, double Step)[] slices, bool finish = true, int workers = 1)
        => OptimizeBrute.Run(f, slices.Select(s =>
        {
            if (!(s.Step > 0)) throw new ArgumentException("Slice steps must be positive.");
            var cnt = (int)Math.Ceiling((s.Hi - s.Lo) / s.Step);
            return Enumerable.Range(0, Math.Max(cnt, 0)).Select(i => s.Lo + i * s.Step).ToArray();
        }).ToArray(), finish, workers);
}

/// <summary>Port of SciPy's <c>_dual_annealing.py</c>.</summary>
internal static class OptimizeDualAnnealing
{
    private const double TailLimit = 1e8, MinVisitBound = 1e-10;

    public static OptimizeResult Run(Func<double[], double> func, Bounds bounds, int maxiter, double initialTemp, double restartTempRatio,
        double qv, double qa, int maxfun, int? seed, bool noLocalSearch, Func<double[], double, int, bool>? callback, double[]? x0, LBfgsBOptions? lsOptions)
    {
        ArgumentNullException.ThrowIfNull(func);
        var n = bounds.Length;
        var lower = bounds.Lb; var upper = bounds.Ub;
        for (var i = 0; i < n; i++) if (!double.IsFinite(lower[i]) || !double.IsFinite(upper[i])) throw new ArgumentException("Some bounds values are inf values or nan values");
        if (!(qv > 1 && qv <= 3)) throw new ArgumentException("visit must lie in (1, 3].");
        if (!(qa > -1e4 && qa <= -5)) throw new ArgumentException("accept must lie in (-1e4, -5].");
        var rng = new GraviRandom(seed ?? Environment.TickCount);
        var range = upper.Select((u, i) => u - lower[i]).ToArray();
        var nfev = 0; var njev = 0;
        double Fun(double[] x) { nfev++; return func((double[])x.Clone()); }

        // Visiting distribution constants.
        var factor2 = Math.Exp((4.0 - qv) * Math.Log(qv - 1.0));
        var factor3 = Math.Exp((2.0 - qv) * Math.Log(2.0) / (qv - 1.0));
        var factor4p = Math.Sqrt(Math.PI) * factor2 / (factor3 * (3.0 - qv));
        var factor5 = 1.0 / (qv - 1.0) - 0.5;
        var d1 = 2.0 - factor5;
        var factor6 = Math.PI * (1.0 - factor5) / Math.Sin(Math.PI * (1.0 - factor5)) / Math.Exp(OptimizeGlobalMath.LogGamma(d1));

        double[] VisitFn(double temperature, int dim)
        {
            var factor1 = Math.Exp(Math.Log(temperature) / (qv - 1.0));
            var factor4 = factor4p * factor1;
            var sigmax = Math.Exp(-(qv - 1.0) * Math.Log(factor6 / factor4) / (3.0 - qv));
            var r = new double[dim];
            for (var i = 0; i < dim; i++)
            {
                var xv = rng.Normal() * sigmax;
                var yv = rng.Normal();
                var den = Math.Exp((qv - 1.0) * Math.Log(Math.Abs(yv)) / (3.0 - qv));
                r[i] = xv / den;
            }
            return r;
        }
        double Wrap(double v, int i)
        {
            var a = v - lower[i];
            var b = a % range[i] + range[i];
            var res = b % range[i] + lower[i];
            if (Math.Abs(res - lower[i]) < MinVisitBound) res += MinVisitBound;
            return res;
        }
        double[] Visiting(double[] x, int step, double temperature)
        {
            if (step < n)
            {
                var visits = VisitFn(temperature, n);
                var upS = rng.NextDouble(); var loS = rng.NextDouble();
                var xv = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var v = visits[i];
                    if (v > TailLimit) v = TailLimit * upS;
                    else if (v < -TailLimit) v = -TailLimit * loS;
                    xv[i] = Wrap(v + x[i], i);
                }
                return xv;
            }
            var xc = (double[])x.Clone();
            var vis = VisitFn(temperature, 1)[0];
            if (vis > TailLimit) vis = TailLimit * rng.NextDouble();
            else if (vis < -TailLimit) vis = -TailLimit * rng.NextDouble();
            var idx = step - n;
            xc[idx] = Wrap(vis + x[idx], idx);
            return xc;
        }

        var lsMaxIter = Math.Min(Math.Max(n * 6, 100), 1000);
        var lsOpt = lsOptions ?? new LBfgsBOptions { MaxIter = lsMaxIter };
        (double E, double[] X) LocalSearch(double[] x, double e)
        {
            OptimizeResult mres;
            try { mres = Optimize.Minimize(Fun, x, MinimizeMethod.LBfgsB, bounds: bounds, options: lsOpt); }
            catch (NotSupportedException) { return (e, x); }
            njev += mres.Njev;
            var valid = mres.X.All(double.IsFinite) && double.IsFinite(mres.Fun) && bounds.Contains(mres.X);
            return valid && mres.Fun < e ? (mres.Fun, mres.X) : (e, x);
        }

        // Energy state.
        double[] curX; double curE;
        void Reset(double[]? start)
        {
            curX = start is null ? Enumerable.Range(0, n).Select(i => rng.Uniform(lower[i], upper[i])).ToArray() : (double[])start.Clone();
            for (var tries = 0; ; tries++)
            {
                curE = Fun(curX);
                if (double.IsFinite(curE)) break;
                if (tries >= 1000) throw new ArgumentException("Stopping algorithm because function create NaN or (+/-) infinity values even with trying new random parameters");
                curX = Enumerable.Range(0, n).Select(i => rng.Uniform(lower[i], upper[i])).ToArray();
            }
        }
        curX = []; curE = 0;
        Reset(x0);
        var xbest = (double[])curX.Clone(); var ebest = curE;
        string? stopMsg = null;
        bool UpdateBest(double e, double[] x, int ctx)
        {
            ebest = e; xbest = (double[])x.Clone();
            return callback is not null && callback((double[])x.Clone(), e, ctx);
        }

        // Strategy chain state.
        var emin = curE; var xmin = (double[])curX.Clone();
        var notImproved = 0; var notImprovedMax = 1000;
        var k = 100 * n;
        var temperatureStep = 0.0;
        var improved = false;

        var temperatureRestart = initialTemp * restartTempRatio;
        var t1 = Math.Exp((qv - 1) * Math.Log(2.0)) - 1.0;
        var iteration = 0;
        var success = true;
        var needStop = false;
        var messages = new List<string>();
        while (!needStop)
        {
            for (var i = 0; i < maxiter; i++)
            {
                var s = i + 2.0;
                var t2 = Math.Exp((qv - 1) * Math.Log(s)) - 1.0;
                var temperature = initialTemp * t1 / t2;
                if (iteration >= maxiter) { messages.Add("Maximum number of iteration reached"); needStop = true; break; }
                if (temperature < temperatureRestart) { Reset(null); break; }

                // ---- StrategyChain.run ----
                temperatureStep = temperature / (i + 1.0);
                notImproved++;
                for (var j = 0; j < n * 2; j++)
                {
                    if (j == 0) improved = i == 0;
                    var xv = Visiting(curX, j, temperature);
                    var e = Fun(xv);
                    if (e < curE)
                    {
                        curE = e; curX = xv;
                        if (e < ebest)
                        {
                            if (UpdateBest(e, xv, 0)) { stopMsg = "Callback function requested to stop early by returning True"; break; }
                            improved = true; notImproved = 0;
                        }
                    }
                    else
                    {
                        var r = rng.NextDouble();
                        var pqvTemp = 1.0 - ((1.0 - qa) * (e - curE) / temperatureStep);
                        var pqv = pqvTemp <= 0 ? 0 : Math.Exp(Math.Log(pqvTemp) / (1.0 - qa));
                        if (r <= pqv) { curE = e; curX = xv; xmin = (double[])curX.Clone(); }
                        if (notImproved >= notImprovedMax && (j == 0 || curE < emin)) { emin = curE; xmin = (double[])curX.Clone(); }
                    }
                    if (nfev >= maxfun) { stopMsg = "Maximum number of function call reached during annealing"; break; }
                }
                if (stopMsg is not null) { messages.Add(stopMsg); needStop = true; success = false; break; }

                // ---- StrategyChain.local_search ----
                if (!noLocalSearch)
                {
                    if (improved)
                    {
                        var (e, x) = LocalSearch(xbest, ebest);
                        if (e < ebest)
                        {
                            notImproved = 0;
                            if (UpdateBest(e, x, 1)) stopMsg = "Callback function requested to stop early by returning True";
                            curE = e; curX = (double[])x.Clone();
                        }
                        if (stopMsg is null && nfev >= maxfun) stopMsg = "Maximum number of function call reached during local search";
                    }
                    if (stopMsg is null)
                    {
                        var doLs = false;
                        if (k < 90 * n)
                        {
                            var pls = Math.Exp(k * (ebest - curE) / temperatureStep);
                            if (pls >= rng.NextDouble()) doLs = true;
                        }
                        if (notImproved >= notImprovedMax) doLs = true;
                        if (doLs)
                        {
                            var (e, x) = LocalSearch(xmin, emin);
                            xmin = (double[])x.Clone(); emin = e;
                            notImproved = 0; notImprovedMax = n;
                            if (e < ebest)
                            {
                                if (UpdateBest(emin, xmin, 2)) stopMsg = "Callback function requested to stop early by returning True";
                                curE = e; curX = (double[])x.Clone();
                            }
                            if (stopMsg is null && nfev >= maxfun) stopMsg = "Maximum number of function call reached during dual annealing";
                        }
                    }
                    if (stopMsg is not null) { messages.Add(stopMsg); needStop = true; success = false; break; }
                }
                iteration++;
            }
        }
        return new OptimizeResult
        {
            X = xbest, Fun = ebest, Nit = iteration, Nfev = nfev, Njev = njev, Success = success, Status = success ? 0 : 1,
            Message = string.Join("; ", messages),
        };
    }
}

/// <summary>Port of SciPy's <c>_basinhopping.py</c>.</summary>
internal static class OptimizeBasinHopping
{
    public static OptimizeResult Run(Func<double[], double> func, double[] x0, int niter, double temperature, double stepsize,
        BasinHoppingMinimizer? minimizer, MinimizeMethod? method, Bounds? bounds, Func<double[], double[]>? takeStep,
        Func<double, double[], double, double[], bool>? acceptTest, Func<double[], double, bool, bool>? callback, int interval,
        int? niterSuccess, int? seed, double targetAcceptRate, double stepwiseFactor)
    {
        ArgumentNullException.ThrowIfNull(func);
        ArgumentNullException.ThrowIfNull(x0);
        var rng = new GraviRandom(seed ?? Environment.TickCount);
        var nfev = 0;
        OptimizeResult Minimize(double[] x)
        {
            var r = minimizer is not null ? minimizer(func, (double[])x.Clone())
                                          : Optimize.Minimize(func, x, method ?? (bounds is null ? MinimizeMethod.Bfgs : MinimizeMethod.LBfgsB), bounds: bounds);
            nfev += r.Nfev;
            return r;
        }
        var beta = temperature != 0 ? 1.0 / temperature : double.PositiveInfinity;
        var nstep = 0; var naccept = 0;
        double[] Step(double[] x)
        {
            if (takeStep is not null) return takeStep((double[])x.Clone());
            nstep++;
            if (nstep % interval == 0)
            {
                var rate = (double)naccept / nstep;
                if (rate > targetAcceptRate) stepsize /= stepwiseFactor; else stepsize *= stepwiseFactor;
            }
            var y = (double[])x.Clone();
            for (var i = 0; i < y.Length; i++) y[i] += rng.Uniform(-stepsize, stepsize);
            return y;
        }

        var failures = 0;
        var minres = Minimize(x0);
        if (!minres.Success) failures++;
        var x = (double[])minres.X.Clone();
        var energy = minres.Fun;
        var incumbent = minres;
        var lowest = minres;
        if (callback is not null) callback((double[])minres.X.Clone(), minres.Fun, true);
        var nsucc = niterSuccess ?? niter + 2;
        var count = 0; var i = 0;
        var message = "requested number of basinhopping iterations completed successfully";
        for (i = 0; i < niter; i++)
        {
            var trial = Minimize(Step(x));
            if (!trial.Success) failures++;
            var accept = true;
            if (acceptTest is not null && !acceptTest(trial.Fun, (double[])trial.X.Clone(), energy, (double[])x.Clone())) accept = false;
            // Metropolis.
            var w = Math.Exp(Math.Min(0, -(trial.Fun - incumbent.Fun) * beta));
            var metro = w >= rng.NextDouble() && (trial.Success || !incumbent.Success);
            if (!metro) accept = false;
            if (accept) naccept++;
            var newGlobal = false;
            if (accept)
            {
                energy = trial.Fun; x = (double[])trial.X.Clone(); incumbent = trial;
                if (trial.Success && trial.Fun < lowest.Fun) { lowest = trial; newGlobal = true; }
            }
            if (callback is not null && callback((double[])trial.X.Clone(), trial.Fun, accept))
            {
                message = "callback function requested stop early by returning True";
                break;
            }
            count++;
            if (newGlobal) count = 0;
            else if (count > nsucc) { message = "success condition satisfied"; break; }
        }
        return new OptimizeResult
        {
            X = (double[])lowest.X.Clone(), Fun = lowest.Fun, Nit = Math.Min(i + 1, niter), Nfev = nfev, Success = lowest.Success,
            Message = message, LowestOptimizationResult = lowest, Status = 0, Nlfev = failures,
        };
    }
}

/// <summary>DIRECT / DIRECT-L on the unit cube.</summary>
internal static class OptimizeDirect
{
    private sealed class Rect
    {
        public required double[] C;   // centre in [0,1]^n
        public required int[] Level;  // side length of dimension i is 3^-Level[i]
        public double F;
        public double Size;           // size measure
        public int SizeKey;           // key grouping equal sizes
    }

    public static OptimizeResult Run(Func<double[], double> func, Bounds bounds, double eps, int maxfun, int maxiter, bool locallyBiased,
        double fGlobal, double fGlobalRelTol, double volTol, double lenTol, Func<double[], bool>? callback)
    {
        ArgumentNullException.ThrowIfNull(func);
        var n = bounds.Length;
        var lb = bounds.Lb; var ub = bounds.Ub;
        for (var i = 0; i < n; i++)
        {
            if (!double.IsFinite(lb[i]) || !double.IsFinite(ub[i])) throw new ArgumentException("Bounds must be finite.");
            if (!(ub[i] > lb[i])) throw new ArgumentException("u[i] < l[i] for some i");
        }
        var nfev = 0;
        double F(double[] c)
        {
            nfev++;
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = lb[i] + c[i] * (ub[i] - lb[i]);
            var v = func(x);
            return double.IsNaN(v) || double.IsInfinity(v) ? double.MaxValue : v;
        }
        void Measure(Rect r)
        {
            if (locallyBiased)
            {
                // DIRECT-L: size = half the longest side, i.e. depends on the smallest level only.
                var minL = r.Level.Min();
                r.Size = 0.5 * Math.Pow(3, -minL);
                r.SizeKey = minL;
            }
            else
            {
                double s = 0;
                foreach (var l in r.Level) s += Math.Pow(9, -l);
                r.Size = 0.5 * Math.Sqrt(s);
                // Distinct sizes correspond to distinct sorted level multisets; the sum of levels and the min define it uniquely enough.
                r.SizeKey = r.Level.Sum() * 1000 + r.Level.Min();
            }
        }
        var rects = new List<Rect>();
        var c0 = Enumerable.Repeat(0.5, n).ToArray();
        var root = new Rect { C = c0, Level = new int[n], F = F(c0) };
        Measure(root);
        rects.Add(root);
        var best = root;
        int status = 0; string message = "";
        var nit = 0;
        while (true)
        {
            if (nfev >= maxfun) { status = 1; message = $"Number of function evaluations done is larger than maxfun={maxfun}"; break; }
            if (nit >= maxiter) { status = 2; message = $"Number of iterations is larger than maxiter={maxiter}"; break; }
            nit++;
            // ---- potentially optimal rectangles: lower right convex hull of (size, f) ----
            var groups = rects.GroupBy(r => r.SizeKey).Select(g => g.OrderBy(r => r.F).ToList()).OrderBy(g => g[0].Size).ToList();
            var cand = groups.Select(g => g[0]).ToList();
            var fmin = best.F;
            var selected = new List<Rect>();
            // Hull: keep candidate j if there is K > 0 with f_j − K d_j ≤ f_i − K d_i for all i, and f_j − K d_j ≤ fmin − eps|fmin|.
            for (var j = 0; j < cand.Count; j++)
            {
                var cj = cand[j];
                double kLow = 0, kHigh = double.PositiveInfinity;
                var ok = true;
                for (var i = 0; i < cand.Count && ok; i++)
                {
                    if (i == j) continue;
                    var ci = cand[i];
                    if (ci.Size < cj.Size) kLow = Math.Max(kLow, (cj.F - ci.F) / (cj.Size - ci.Size));
                    else if (ci.Size > cj.Size) kHigh = Math.Min(kHigh, (ci.F - cj.F) / (ci.Size - cj.Size));
                    else if (ci.F < cj.F) ok = false;
                }
                if (!ok || kLow > kHigh + 1e-15 * Math.Max(1, Math.Abs(kHigh))) continue;
                var k = double.IsPositiveInfinity(kHigh) ? kLow : kHigh;
                if (j == cand.Count - 1 || cj.F - k * cj.Size <= fmin - eps * Math.Abs(fmin))
                {
                    if (locallyBiased) selected.Add(cj);
                    else selected.AddRange(groups[j].Where(r => r.F == cj.F));
                }
            }
            if (selected.Count == 0) selected.Add(cand[^1]);

            // ---- divide ----
            foreach (var r in selected)
            {
                if (nfev >= maxfun) break;
                var minL = r.Level.Min();
                var dims = Enumerable.Range(0, n).Where(i => r.Level[i] == minL).ToArray();
                var delta = Math.Pow(3, -(minL + 1));
                var samples = new List<(int Dim, double W, Rect Plus, Rect Minus)>();
                foreach (var i in dims)
                {
                    var cp = (double[])r.C.Clone(); cp[i] += delta;
                    var cm = (double[])r.C.Clone(); cm[i] -= delta;
                    var fp = F(cp); var fm = F(cm);
                    samples.Add((i, Math.Min(fp, fm), new Rect { C = cp, Level = (int[])r.Level.Clone(), F = fp }, new Rect { C = cm, Level = (int[])r.Level.Clone(), F = fm }));
                }
                // Divide along the dimensions in order of increasing w: the best samples get the biggest boxes.
                var level = r.Level;
                foreach (var s in samples.OrderBy(t => t.W).ThenBy(t => t.Dim))
                {
                    level[s.Dim]++;
                    foreach (var child in new[] { s.Plus, s.Minus })
                    {
                        for (var q = 0; q < n; q++) child.Level[q] = level[q];
                        Measure(child);
                        rects.Add(child);
                        if (child.F < best.F) best = child;
                    }
                }
                Measure(r);
            }
            if (callback is not null && callback(ToX(best.C))) { status = 9; message = "Forced stop"; break; }
            if (double.IsFinite(fGlobal))
            {
                var err = fGlobal != 0 ? (best.F - fGlobal) / Math.Abs(fGlobal) : best.F;
                if (err <= fGlobalRelTol) { status = 3; message = $"The best function value found is within a relative error={fGlobalRelTol} of the (known) global optimum f_min"; break; }
            }
            var vol = Math.Pow(3, -best.Level.Sum());
            if (vol <= volTol) { status = 4; message = $"The volume of the hyperrectangle containing the lowest function value found is below vol_tol={volTol}"; break; }
            if (best.Size <= lenTol) { status = 5; message = $"The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={lenTol}"; break; }
        }
        return new OptimizeResult
        {
            X = ToX(best.C), Fun = best.F, Nfev = nfev, Nit = nit, Status = status, Success = status >= 3 && status != 9, Message = message,
        };

        double[] ToX(double[] c) => c.Select((v, i) => lb[i] + v * (ub[i] - lb[i])).ToArray();
    }
}

/// <summary>Grid search behind <see cref="Optimize.Brute(Func{double[], double}, ValueTuple{double, double}[], int, bool, int)"/>.</summary>
internal static class OptimizeBrute
{
    public static BruteResult Run(Func<double[], double> func, double[][] axes, bool finish, int workers)
    {
        ArgumentNullException.ThrowIfNull(func);
        var n = axes.Length;
        if (n == 0) throw new ArgumentException("At least one range is needed.");
        if (axes.Any(a => a.Length == 0)) throw new ArgumentException("Every range must contain at least one grid point.");
        var shape = axes.Select(a => a.Length).ToArray();
        long total = 1;
        foreach (var s in shape) total *= s;
        if (total > int.MaxValue / Math.Max(1, n)) throw new ArgumentException("The grid is too large.");
        var tot = (int)total;
        var jout = new double[tot];
        var grid = new double[n * tot];
        double[] Point(int flat)
        {
            var x = new double[n];
            var rem = flat;
            for (var d = n - 1; d >= 0; d--) { x[d] = axes[d][rem % shape[d]]; rem /= shape[d]; }
            return x;
        }
        if (workers > 1) Parallel.For(0, tot, new ParallelOptions { MaxDegreeOfParallelism = workers }, k => jout[k] = func(Point(k)));
        else for (var k = 0; k < tot; k++) jout[k] = func(Point(k));
        for (var k = 0; k < tot; k++) { var x = Point(k); for (var d = 0; d < n; d++) grid[d * tot + k] = x[d]; }
        var bi = 0;
        for (var k = 1; k < tot; k++) if (jout[k] < jout[bi] || double.IsNaN(jout[bi])) bi = k;
        var xmin = Point(bi);
        var fmin = jout[bi];
        if (finish)
        {
            var r = Optimize.Minimize(func, xmin, MinimizeMethod.NelderMead);
            if (r.Fun < fmin || double.IsNaN(fmin)) { xmin = r.X; fmin = r.Fun; }
        }
        return new BruteResult(xmin, fmin, new NdArray(grid, [n, .. shape]), new NdArray(jout, shape));
    }
}

/// <summary>Numerical helpers for the global optimisers.</summary>
internal static class OptimizeGlobalMath
{
    /// <summary>ln Γ(x) for x &gt; 0 by the Lanczos approximation (g = 7, n = 9), accurate to ~1e-15 relative.</summary>
    public static double LogGamma(double x)
    {
        double[] c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        x -= 1;
        var a = c[0];
        var t = x + 7.5;
        for (var i = 1; i < 9; i++) a += c[i] / (x + i);
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(a);
    }
}
