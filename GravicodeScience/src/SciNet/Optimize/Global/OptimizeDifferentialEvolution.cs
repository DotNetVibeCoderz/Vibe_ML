using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Mutation/crossover strategies of <see cref="Optimize.DifferentialEvolution"/>.</summary>
public enum DifferentialEvolutionStrategy
{
    /// <summary>best + F(r0 − r1), binomial crossover (SciPy's default).</summary>
    Best1Bin,
    /// <summary>best + F(r0 − r1), exponential crossover.</summary>
    Best1Exp,
    /// <summary>r0 + F(r1 − r2), binomial.</summary>
    Rand1Bin,
    /// <summary>r0 + F(r1 − r2), exponential.</summary>
    Rand1Exp,
    /// <summary>r0 + F(r1 + r2 − r3 − r4), binomial.</summary>
    Rand2Bin,
    /// <summary>r0 + F(r1 + r2 − r3 − r4), exponential.</summary>
    Rand2Exp,
    /// <summary>best + F(r0 + r1 − r2 − r3), binomial.</summary>
    Best2Bin,
    /// <summary>best + F(r0 + r1 − r2 − r3), exponential.</summary>
    Best2Exp,
    /// <summary>x + F(best − x + r0 − r1), binomial.</summary>
    CurrentToBest1Bin,
    /// <summary>x + F(best − x + r0 − r1), exponential.</summary>
    CurrentToBest1Exp,
    /// <summary>r0 + F(best − r0) + F(r1 − r2), binomial.</summary>
    RandToBest1Bin,
    /// <summary>r0 + F(best − r0) + F(r1 − r2), exponential.</summary>
    RandToBest1Exp,
}

/// <summary>Initial population of <see cref="Optimize.DifferentialEvolution"/>.</summary>
public enum DifferentialEvolutionInit
{
    /// <summary>Latin hypercube (SciPy's default): one sample per stratum in every coordinate.</summary>
    LatinHypercube,
    /// <summary>Sobol' sequence with a random digital shift; the population is rounded up to a power of two (≤ 16 dimensions).</summary>
    Sobol,
    /// <summary>Halton sequence with a random start.</summary>
    Halton,
    /// <summary>Uniform random.</summary>
    Random,
}

/// <summary>When <see cref="Optimize.DifferentialEvolution"/> replaces population members.</summary>
public enum DifferentialEvolutionUpdating
{
    /// <summary>Each improved trial replaces its parent at once (SciPy's default; faster convergence).</summary>
    Immediate,
    /// <summary>A whole generation is evaluated before any replacement; required for parallel evaluation.</summary>
    Deferred,
}

public static partial class Optimize
{
    /// <summary>
    /// Differential evolution (Storn &amp; Price 1997), <c>scipy.optimize.differential_evolution</c>.
    /// </summary>
    /// <param name="f">Objective; with <paramref name="workers"/> &gt; 1 it is called concurrently and must be thread-safe.</param>
    /// <param name="bounds">Finite bounds for every variable.</param>
    /// <param name="strategy">Mutation strategy.</param>
    /// <param name="maxiter">Maximum generations.</param>
    /// <param name="popsize">Population multiplier: the population has max(5, popsize·n) members.</param>
    /// <param name="tol">Relative convergence tolerance: stop when std(E) ≤ atol + tol·|mean(E)|.</param>
    /// <param name="mutation">Fixed differential weight F; ignored when <paramref name="mutationRange"/> is set.</param>
    /// <param name="mutationRange">Dithering: F drawn uniformly from this range every generation. Default (0.5, 1).</param>
    /// <param name="recombination">Crossover probability CR.</param>
    /// <param name="seed">Random seed.</param>
    /// <param name="callback">Called each generation with (best x, convergence = tol/(std/|mean|)); return true to stop.</param>
    /// <param name="polish">Refine the best member with L-BFGS-B (SLSQP when constrained) at the end.</param>
    /// <param name="init">Population initialisation.</param>
    /// <param name="initialPopulation">Explicit initial population (rows), overriding <paramref name="init"/>.</param>
    /// <param name="atol">Absolute convergence tolerance.</param>
    /// <param name="updating">Immediate or deferred replacement.</param>
    /// <param name="workers">Parallel evaluations per generation (forces deferred updating).</param>
    /// <param name="constraints">Constraints handled by Lampinen's feasibility rules.</param>
    /// <param name="x0">A member placed in the initial population.</param>
    /// <param name="integrality">True for integer variables (rounded before evaluation).</param>
    /// <param name="maxfun">Maximum function evaluations.</param>
    public static OptimizeResult DifferentialEvolution(Func<double[], double> f, Bounds bounds,
        DifferentialEvolutionStrategy strategy = DifferentialEvolutionStrategy.Best1Bin, int maxiter = 1000, int popsize = 15, double tol = 0.01,
        double? mutation = null, (double Min, double Max)? mutationRange = null, double recombination = 0.7, int? seed = null,
        Func<double[], double, bool>? callback = null, bool polish = true, DifferentialEvolutionInit init = DifferentialEvolutionInit.LatinHypercube,
        NdArray? initialPopulation = null, double atol = 0, DifferentialEvolutionUpdating updating = DifferentialEvolutionUpdating.Immediate,
        int workers = 1, IReadOnlyList<IMinimizeConstraint>? constraints = null, double[]? x0 = null, bool[]? integrality = null,
        int maxfun = int.MaxValue)
        => OptimizeDe.Run(f, bounds, strategy, maxiter, popsize, tol, mutation, mutation is null ? mutationRange ?? (0.5, 1.0) : mutationRange,
            recombination, seed, callback, polish, init, initialPopulation, atol, updating, workers, constraints ?? [], x0, integrality, maxfun);
}

/// <summary>A port of SciPy's <c>DifferentialEvolutionSolver</c> (random streams differ, so trajectories do too).</summary>
internal static class OptimizeDe
{
    public static OptimizeResult Run(Func<double[], double> func, Bounds bounds, DifferentialEvolutionStrategy strategy, int maxiter, int popsize, double tol,
        double? mutation, (double Min, double Max)? dither, double cr, int? seed, Func<double[], double, bool>? callback, bool polish,
        DifferentialEvolutionInit init, NdArray? initialPopulation, double atol, DifferentialEvolutionUpdating updating, int workers,
        IReadOnlyList<IMinimizeConstraint> constraints, double[]? x0, bool[]? integrality, int maxfun)
    {
        ArgumentNullException.ThrowIfNull(func);
        ArgumentNullException.ThrowIfNull(bounds);
        var n = bounds.Length;
        for (var i = 0; i < n; i++) if (!double.IsFinite(bounds.Lb[i]) || !double.IsFinite(bounds.Ub[i])) throw new ArgumentException("Differential evolution needs finite bounds.");
        if (dither is { } dd && !(0 <= dd.Min && dd.Min <= dd.Max && dd.Max <= 2)) throw new ArgumentException("The mutation range must lie within [0, 2].");
        if (mutation is { } mm && !(mm >= 0 && mm <= 2)) throw new ArgumentException("The mutation constant must lie within [0, 2].");
        if (!(cr >= 0 && cr <= 1)) throw new ArgumentException("recombination must lie within [0, 1].");
        if (workers > 1) updating = DifferentialEvolutionUpdating.Deferred;
        var rng = new GraviRandom(seed ?? Environment.TickCount);
        var lb = bounds.Lb; var ub = bounds.Ub;
        var integ = integrality ?? new bool[n];
        if (integ.Length != n) throw new ArgumentException("integrality must have one entry per variable.");

        double[] Scale(double[] t)
        {
            var x = new double[n];
            for (var j = 0; j < n; j++)
            {
                x[j] = 0.5 * (lb[j] + ub[j]) + (t[j] - 0.5) * Math.Abs(ub[j] - lb[j]);
                if (integ[j]) x[j] = Math.Clamp(Math.Round(x[j]), Math.Ceiling(lb[j]), Math.Floor(ub[j]));
            }
            return x;
        }
        double[] Unscale(double[] x)
        {
            var t = new double[n];
            for (var j = 0; j < n; j++) t[j] = (x[j] - 0.5 * (lb[j] + ub[j])) / Math.Abs(ub[j] - lb[j]) + 0.5;
            return t;
        }

        // ---- initial population ----
        double[][] pop;
        if (initialPopulation is not null)
        {
            if (initialPopulation.Rank != 2 || initialPopulation.Shape[1] != n || initialPopulation.Shape[0] < 5)
                throw new ArgumentException("The initial population must have shape (S, N) with S >= 5.");
            var d = initialPopulation.AsContiguous().ToArray();
            pop = Enumerable.Range(0, initialPopulation.Shape[0]).Select(i => Unscale(d.AsSpan(i * n, n).ToArray()).Select(v => Math.Clamp(v, 0, 1)).ToArray()).ToArray();
        }
        else
        {
            var np = Math.Max(5, popsize * n);
            pop = init switch
            {
                DifferentialEvolutionInit.Random => Enumerable.Range(0, np).Select(_ => Enumerable.Range(0, n).Select(_ => rng.NextDouble()).ToArray()).ToArray(),
                DifferentialEvolutionInit.Halton => OptimizeQmc.Halton(np, n, rng),
                DifferentialEvolutionInit.Sobol => OptimizeQmc.Sobol(1 << (int)Math.Ceiling(Math.Log2(np)), n, rng),
                _ => Lhs(np, n, rng),
            };
        }
        var npop = pop.Length;
        if (x0 is not null)
        {
            if (x0.Length != n) throw new ArgumentException("x0 must have one entry per variable.");
            pop[0] = Unscale(x0).Select(v => Math.Clamp(v, 0, 1)).ToArray();
        }

        // ---- constraints (Lampinen) ----
        var cset = constraints.Count > 0 ? new OptimizeConstraintSet(constraints, Scale(pop[0]), 1e-8) : null;
        double[] Violations(double[] x)
        {
            if (cset is null) return [];
            var (e, i) = cset.Eval(x);
            return [.. e.Select(Math.Abs), .. i.Select(v => Math.Max(0, -v))];
        }
        static bool Feasible(double[] cv) { foreach (var v in cv) if (v > 0) return false; return true; }
        static bool Accept(double eT, double[] cvT, double eO, double[] cvO)
        {
            bool fT = Feasible(cvT), fO = Feasible(cvO);
            if (fT && fO) return eT <= eO;
            if (fT && !fO) return true;
            if (!fT && fO) return false;
            for (var k = 0; k < cvT.Length; k++) if (cvT[k] > cvO[k]) return false;
            return true;
        }

        var nfev = 0;
        var energies = new double[npop];
        var cvs = new double[npop][];
        double Eval(double[] x) { Interlocked.Increment(ref nfev); var v = func((double[])x.Clone()); return double.IsNaN(v) ? double.PositiveInfinity : v; }

        void EvaluateMany(double[][] trials, double[] outE, double[][] outCv)
        {
            var scaled = trials.Select(Scale).ToArray();
            for (var k = 0; k < trials.Length; k++) outCv[k] = Violations(scaled[k]);
            var idx = Enumerable.Range(0, trials.Length).Where(k => Feasible(outCv[k])).ToArray();
            var budget = Math.Max(0, maxfun - nfev);
            if (idx.Length > budget) idx = idx.Take(budget).ToArray();
            for (var k = 0; k < trials.Length; k++) outE[k] = double.PositiveInfinity;
            if (workers > 1 && idx.Length > 1)
                Parallel.ForEach(idx, new ParallelOptions { MaxDegreeOfParallelism = workers }, k => outE[k] = Eval(scaled[k]));
            else foreach (var k in idx) outE[k] = Eval(scaled[k]);
        }

        void Promote()
        {
            // argmin among feasible members (or the least violation when none is feasible)
            int best;
            var feas = Enumerable.Range(0, npop).Where(k => Feasible(cvs[k])).ToArray();
            if (feas.Length > 0) best = feas.OrderBy(k => energies[k]).First();
            else best = Enumerable.Range(0, npop).OrderBy(k => cvs[k].Sum()).First();
            if (best != 0)
            {
                (pop[0], pop[best]) = (pop[best], pop[0]);
                (energies[0], energies[best]) = (energies[best], energies[0]);
                (cvs[0], cvs[best]) = (cvs[best], cvs[0]);
            }
        }

        EvaluateMany(pop, energies, cvs);
        Promote();

        double scale = mutation ?? 0.5;
        int[] Samples(int candidate, int count)
        {
            var pool = Enumerable.Range(0, npop).ToArray();
            for (var i = npop - 1; i > 0; i--) { var j = rng.Next(i + 1); (pool[i], pool[j]) = (pool[j], pool[i]); }
            return pool.Where(v => v != candidate).Take(count).ToArray();
        }
        var binomial = strategy.ToString().EndsWith("Bin", StringComparison.Ordinal);
        double[] Mutate(int candidate)
        {
            var fill = rng.Next(n);
            var s = Samples(candidate, 5);
            var bp = new double[n];
            for (var j = 0; j < n; j++)
            {
                bp[j] = strategy switch
                {
                    DifferentialEvolutionStrategy.Best1Bin or DifferentialEvolutionStrategy.Best1Exp => pop[0][j] + scale * (pop[s[0]][j] - pop[s[1]][j]),
                    DifferentialEvolutionStrategy.Rand1Bin or DifferentialEvolutionStrategy.Rand1Exp => pop[s[0]][j] + scale * (pop[s[1]][j] - pop[s[2]][j]),
                    DifferentialEvolutionStrategy.Rand2Bin or DifferentialEvolutionStrategy.Rand2Exp => pop[s[0]][j] + scale * (pop[s[1]][j] + pop[s[2]][j] - pop[s[3]][j] - pop[s[4]][j]),
                    DifferentialEvolutionStrategy.Best2Bin or DifferentialEvolutionStrategy.Best2Exp => pop[0][j] + scale * (pop[s[0]][j] + pop[s[1]][j] - pop[s[2]][j] - pop[s[3]][j]),
                    DifferentialEvolutionStrategy.CurrentToBest1Bin or DifferentialEvolutionStrategy.CurrentToBest1Exp => pop[candidate][j] + scale * (pop[0][j] - pop[candidate][j] + pop[s[0]][j] - pop[s[1]][j]),
                    _ => pop[s[0]][j] + scale * (pop[0][j] - pop[s[0]][j]) + scale * (pop[s[1]][j] - pop[s[2]][j]),
                };
            }
            var trial = (double[])pop[candidate].Clone();
            if (binomial)
            {
                for (var j = 0; j < n; j++) if (rng.NextDouble() < cr || j == fill) trial[j] = bp[j];
            }
            else
            {
                var cross = Enumerable.Range(0, n).Select(_ => rng.NextDouble() < cr).ToArray();
                cross[0] = true;
                for (var i = 0; i < n && cross[i]; i++) { trial[fill] = bp[fill]; fill = (fill + 1) % n; }
            }
            for (var j = 0; j < n; j++) if (trial[j] > 1 || trial[j] < 0) trial[j] = rng.NextDouble();
            return trial;
        }

        bool Converged()
        {
            if (energies.Any(double.IsInfinity)) return false;
            var mean = energies.Average();
            var std = Math.Sqrt(energies.Select(e => (e - mean) * (e - mean)).Average());
            return std <= atol + tol * Math.Abs(mean);
        }
        double Convergence()
        {
            if (energies.Any(double.IsInfinity)) return double.PositiveInfinity;
            var mean = energies.Average();
            return Math.Sqrt(energies.Select(e => (e - mean) * (e - mean)).Average()) / (Math.Abs(mean) + 2.220446049250313e-16);
        }

        var warning = false;
        var message = OptimizeMessages.Success;
        var nit = 0;
        for (nit = 1; nit <= maxiter; nit++)
        {
            if (dither is { } dr) scale = rng.Uniform(dr.Min, dr.Max);
            var stop = false;
            if (updating == DifferentialEvolutionUpdating.Immediate)
            {
                for (var cand = 0; cand < npop; cand++)
                {
                    if (nfev >= maxfun) { stop = true; break; }
                    var trial = Mutate(cand);
                    var x = Scale(trial);
                    var cv = Violations(x);
                    var e = Feasible(cv) ? Eval(x) : double.PositiveInfinity;
                    if (Accept(e, cv, energies[cand], cvs[cand]))
                    {
                        pop[cand] = trial; energies[cand] = e; cvs[cand] = cv;
                        if (Accept(e, cv, energies[0], cvs[0])) Promote();
                    }
                }
            }
            else
            {
                if (nfev >= maxfun) stop = true;
                else
                {
                    var trials = Enumerable.Range(0, npop).Select(Mutate).ToArray();
                    var te = new double[npop]; var tcv = new double[npop][];
                    EvaluateMany(trials, te, tcv);
                    for (var k = 0; k < npop; k++)
                        if (Accept(te[k], tcv[k], energies[k], cvs[k])) { pop[k] = trials[k]; energies[k] = te[k]; cvs[k] = tcv[k]; }
                    Promote();
                }
            }
            if (stop)
            {
                warning = true;
                message = nfev > maxfun ? OptimizeMessages.MaxFev : "Maximum number of function evaluations has been reached.";
                break;
            }
            if (callback is not null && callback(Scale(pop[0]), tol / (Convergence() + 2.220446049250313e-16)))
            {
                warning = true; message = "callback function requested stop early"; break;
            }
            if (Converged()) break;
        }
        if (nit > maxiter) { nit = maxiter; warning = true; message = OptimizeMessages.MaxIter; }

        var xbest = Scale(pop[0]);
        var res = new OptimizeResult
        {
            X = xbest, Fun = energies[0], Nit = nit, Success = !warning, Status = warning ? 1 : 0, Message = message,
        };
        var nlfev = 0;
        if (polish && !integ.All(t => t))
        {
            var plb = (double[])lb.Clone(); var pub = (double[])ub.Clone();
            for (var j = 0; j < n; j++) if (integ[j]) { plb[j] = xbest[j]; pub[j] = xbest[j]; }
            try
            {
                var pr = cset is null
                    ? Optimize.Minimize(func, xbest, MinimizeMethod.LBfgsB, bounds: new Bounds(plb, pub))
                    : Optimize.Minimize(func, xbest, MinimizeMethod.Slsqp, bounds: new Bounds(plb, pub), constraints: constraints);
                nlfev = pr.Nfev;
                nfev += pr.Nfev;
                var inside = true;
                for (var j = 0; j < n; j++) if (pr.X[j] < lb[j] || pr.X[j] > ub[j]) inside = false;
                var feasibleOk = cset is null || Feasible(Violations(pr.X)) || (pr.ConstrViolation ?? 0) <= 1e-8;
                if (pr.Fun < res.Fun && pr.Success && inside && feasibleOk)
                {
                    res.X = pr.X; res.Fun = pr.Fun; res.Jac = pr.Jac;
                    pop[0] = Unscale(pr.X); energies[0] = pr.Fun;
                }
            }
            catch (NotSupportedException) { }
        }
        res.Nfev = nfev;
        res.Nlfev = nlfev;
        if (cset is not null)
        {
            var (e, i) = cset.Eval(res.X);
            res.ConstrViolation = OptimizeConstraintSet.Violation(e, i);
            if (res.ConstrViolation > 0) { res.Success = false; res.Message = "The solution does not satisfy the constraints, MAXCV = " + res.ConstrViolation; }
        }
        var popArr = new double[npop * n];
        for (var k = 0; k < npop; k++) Scale(pop[k]).CopyTo(popArr, k * n);
        res.Population = new NdArray(popArr, npop, n);
        res.PopulationEnergies = (double[])energies.Clone();
        return res;
    }

    /// <summary>SciPy's <c>init_population_lhs</c>: stratified samples, independently permuted per coordinate.</summary>
    private static double[][] Lhs(int np, int n, GraviRandom rng)
    {
        var seg = 1.0 / np;
        var samples = new double[np][];
        for (var i = 0; i < np; i++) { samples[i] = new double[n]; for (var j = 0; j < n; j++) samples[i][j] = seg * rng.NextDouble() + (double)i / np; }
        var pop = new double[np][];
        for (var i = 0; i < np; i++) pop[i] = new double[n];
        for (var j = 0; j < n; j++)
        {
            var order = Enumerable.Range(0, np).ToArray();
            for (var i = np - 1; i > 0; i--) { var k = rng.Next(i + 1); (order[i], order[k]) = (order[k], order[i]); }
            for (var i = 0; i < np; i++) pop[i][j] = samples[order[i]][j];
        }
        return pop;
    }
}

/// <summary>Small quasi-random generators for population initialisation.</summary>
internal static class OptimizeQmc
{
    private static readonly int[] Primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113];

    /// <summary>Halton points (radical inverses in prime bases) starting at a random index.</summary>
    public static double[][] Halton(int count, int d, GraviRandom rng)
    {
        if (d > Primes.Length) throw new NotSupportedException($"Halton initialisation supports up to {Primes.Length} dimensions.");
        var start = 1 + rng.Next(1000);
        var r = new double[count][];
        for (var i = 0; i < count; i++)
        {
            r[i] = new double[d];
            for (var j = 0; j < d; j++) r[i][j] = RadicalInverse(start + i, Primes[j]);
        }
        return r;
    }

    private static double RadicalInverse(int i, int b)
    {
        double f = 1, v = 0;
        while (i > 0) { f /= b; v += f * (i % b); i /= b; }
        return v;
    }

    // Joe & Kuo (2008) direction numbers, dimensions 2..16: (s, a, m_1..m_s).
    private static readonly (int S, int A, int[] M)[] JoeKuo =
    [
        (1, 0, [1]), (2, 1, [1, 3]), (3, 1, [1, 3, 1]), (3, 2, [1, 1, 1]), (4, 1, [1, 1, 3, 3]), (4, 4, [1, 3, 5, 13]),
        (5, 2, [1, 1, 5, 5, 17]), (5, 4, [1, 1, 5, 5, 5]), (5, 7, [1, 1, 7, 11, 19]), (5, 11, [1, 1, 5, 1, 1]),
        (5, 13, [1, 1, 1, 3, 11]), (5, 14, [1, 3, 5, 5, 31]), (6, 1, [1, 3, 3, 9, 7, 49]), (6, 13, [1, 1, 1, 15, 21, 21]),
        (6, 16, [1, 3, 1, 13, 27, 49]),
    ];

    /// <summary>Unscrambled Sobol' points 0..count−1 (Gray-code order), digitally shifted by a random mask per dimension.</summary>
    public static double[][] Sobol(int count, int d, GraviRandom rng, bool shift = true)
    {
        if (d > JoeKuo.Length + 1) throw new NotSupportedException($"Sobol initialisation supports up to {JoeKuo.Length + 1} dimensions.");
        const int bits = 32;
        var v = new uint[d][];
        for (var j = 0; j < d; j++)
        {
            v[j] = new uint[bits];
            if (j == 0) { for (var k = 0; k < bits; k++) v[j][k] = 1u << (bits - 1 - k); continue; }
            var (s, a, m) = JoeKuo[j - 1];
            for (var k = 0; k < s && k < bits; k++) v[j][k] = (uint)m[k] << (bits - 1 - k);
            for (var k = s; k < bits; k++)
            {
                var val = v[j][k - s] ^ (v[j][k - s] >> s);
                for (var t = 1; t < s; t++) if (((a >> (s - 1 - t)) & 1) == 1) val ^= v[j][k - t];
                v[j][k] = val;
            }
        }
        var masks = Enumerable.Range(0, d).Select(_ => shift ? (uint)(rng.NextUInt64() >> 32) : 0u).ToArray();
        var x = new uint[d];
        var r = new double[count][];
        for (var i = 0; i < count; i++)
        {
            r[i] = new double[d];
            for (var j = 0; j < d; j++) r[i][j] = (x[j] ^ masks[j]) / 4294967296.0;
            // Gray code: flip the direction number of the lowest zero bit of i.
            var c = System.Numerics.BitOperations.TrailingZeroCount(~(uint)i);
            for (var j = 0; j < d; j++) x[j] ^= v[j][c];
        }
        return r;
    }
}
