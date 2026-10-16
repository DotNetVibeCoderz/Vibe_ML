namespace Gravicode.Science.SciNet;

/// <summary>
/// Kernels behind the goodness-of-fit tests: exact two-sample KS path counting, the Cramér–von
/// Mises distributions, the Anderson–Darling k-sample sums, and Royston's Shapiro–Wilk (AS R94).
/// </summary>
internal static class StatsKsKernels
{
    // ------------------------------------------------------------------ two-sample KS

    /// <summary>scipy's <c>_attempt_exact_2kssamp</c>; false when the counting overflowed.</summary>
    public static bool AttemptExact2Samp(int n1, int n2, int g, double d, Alternative alternative, out double dOut, out double prob)
    {
        var lcm = (long)(n1 / g) * n2;
        var h = (long)Math.Round(d * lcm, MidpointRounding.ToEven);
        dOut = h * 1.0 / lcm;
        prob = double.NaN;
        if (h == 0) { prob = 1.0; return true; }
        if (alternative == Alternative.TwoSided)
            prob = n1 == n2 ? ProbOutsideSquare(n1, h) : OuterProbInside(n1, n2, g, h);
        else if (n1 == n2)
        {
            prob = 1.0;
            for (var j = 0L; j < h; j++) prob *= (n1 - j) / (n1 + j + 1.0);
        }
        else
        {
            var paths = CountPathsOutside(n1, n2, g, h);
            var bin = Special.Binom(n1 + n2, n1);
            if (!double.IsFinite(paths) || paths > bin || double.IsInfinity(bin)) return false;
            prob = paths / bin;
        }
        return prob >= 0 && prob <= 1 && double.IsFinite(prob);
    }

    private static double ProbOutsideSquare(long n, long h)
    {
        var P = 0.0;
        var k = (long)Math.Floor((double)n / h);
        while (k >= 0)
        {
            var p1 = 1.0;
            for (var j = 0L; j < h; j++) p1 = (n - k * h - j) * p1 / (n + k * h + j + 1);
            P = p1 * (1.0 - P);
            k--;
        }
        return 2 * P;
    }

    /// <summary>
    /// Proportion of lattice paths from (0,0) to (m,n) leaving the band |x/m − y/n| &lt; h/lcm, computed
    /// column by column on probabilities rather than counts so nothing overflows (scipy's pythran kernel).
    /// </summary>
    private static double OuterProbInside(int m, int n, int g, long h)
    {
        if (m < n) (m, n) = (n, m);
        long mg = m / g, ng = n / g;
        // A(i, j) = (paths from the origin to (i, j) that stay inside |i·ng − j·mg| < h) / C(i + j, j).
        // Dividing by the binomial at every step turns the path count into a probability, so the
        // recurrence A(i,j) = [i·A(i−1,j) + j·A(i,j−1)]/(i + j) never overflows.
        var prev = new double[n + 1];
        var cur = new double[n + 1];
        var lo = 0;
        var hi = (int)Math.Min((h + mg - 1) / mg, n + 1); // exclusive
        for (var j = lo; j < hi; j++) prev[j] = 1;
        for (var i = 1; i <= m; i++)
        {
            var nlo = (int)Math.Max(FloorDiv(ng * i - h, mg) + 1, 0);
            var nhi = (int)Math.Min(CeilDiv(ng * i + h, mg), n + 1);
            if (nhi <= nlo) return 1.0;
            var left = 0.0;
            for (var j = nlo; j < nhi; j++)
            {
                var below = j >= lo && j < hi ? prev[j] : 0.0;
                left = (below * i + left * j) / (i + j);
                cur[j] = left;
            }
            for (var j = lo; j < hi; j++) prev[j] = 0;
            (prev, cur) = (cur, prev);
            lo = nlo;
            hi = nhi;
        }
        return n >= lo && n < hi ? 1 - prev[n] : 1.0;
    }

    private static long CeilDiv(long a, long b) => -FloorDiv(-a, b);

    private static double CountPathsOutside(int m, int n, int g, long h)
    {
        if (m < n) (m, n) = (n, m);
        long mg = m / g, ng = n / g;
        // Python floor division: (mg - h) // mg rounds towards −∞.
        var lxj = n + FloorDiv(mg - h, mg);
        if (lxj <= 0) return Special.Binom(m + n, n);
        var xj = new long[lxj];
        for (var j = 0; j < lxj; j++) xj[j] = FloorDiv(h + mg * j + ng - 1, ng);
        var B = new double[lxj];
        B[0] = 1;
        for (var j = 1; j < lxj; j++)
        {
            var Bj = Special.Binom(xj[j] + j, j);
            for (var i = 0; i < j; i++) Bj -= Special.Binom(xj[j] - xj[i] + j - i, j - i) * B[i];
            B[j] = Bj;
        }
        var num = 0.0;
        for (var j = 0; j < lxj; j++) num += B[j] * Special.Binom(m - xj[j] + (n - j), n - j);
        return num;
    }

    private static long FloorDiv(long a, long b) => a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);

    // ------------------------------------------------------------------ Cramér–von Mises

    /// <summary>Limiting CvM CDF (Anderson &amp; Darling 1952), scipy's <c>_cdf_cvm_inf</c>.</summary>
    public static double CdfCvmInf(double x)
    {
        var tot = 0.0;
        for (var k = 0; k < 1000; k++)
        {
            var u = Math.Exp(Special.GammaLn(k + 0.5) - Special.GammaLn(k + 1.0)) / (Math.Pow(Math.PI, 1.5) * Math.Sqrt(x));
            double y = 4 * k + 1;
            var q = y * y / (16 * x);
            var z = u * Math.Sqrt(y) * Math.Exp(-q) * Special.Kv(0.25, q);
            tot += z;
            if (!(Math.Abs(z) >= 1e-7)) break;
        }
        return tot;
    }

    private static double Psi1Mod(double x)
    {
        static double Ed2(double y)
        {
            var z = y * y / 4;
            var b = Special.Kv(0.25, z) + Special.Kv(0.75, z);
            return Math.Exp(-z) * Math.Pow(y / 2, 1.5) * b / Math.Sqrt(Math.PI);
        }
        static double Ed3(double y)
        {
            var z = y * y / 4;
            var c = Math.Exp(-z) / Math.Sqrt(Math.PI);
            return c * Math.Pow(y / 2, 2.5) * (2 * Special.Kv(0.25, z) + 3 * Special.Kv(0.75, z) - Special.Kv(1.25, z));
        }
        double Ak(int k)
        {
            double m = 2 * k + 1;
            var sx = 2 * Math.Sqrt(x);
            var y1 = Math.Pow(x, 0.75);
            var y2 = Math.Pow(x, 1.25);
            var g12 = Special.Gamma(k + 0.5);
            var g32 = Special.Gamma(k + 1.5);
            var e1 = m * g12 * Ed2((4 * k + 3) / sx) / (9 * y1);
            var e2 = g12 * Ed3((4 * k + 1) / sx) / (72 * y2);
            var e3 = 2 * (m + 2) * g32 * Ed3((4 * k + 5) / sx) / (12 * y2);
            var e4 = 7 * m * g12 * Ed2((4 * k + 1) / sx) / (144 * y1);
            var e5 = 7 * m * g12 * Ed2((4 * k + 5) / sx) / (144 * y1);
            return e1 + e2 + e3 + e4 + e5;
        }
        var tot = 0.0;
        for (var k = 0; k < 1000; k++)
        {
            var z = -Ak(k) / (Math.PI * Special.Gamma(k + 1.0));
            tot += z;
            if (!(Math.Abs(z) >= 1e-7)) break;
        }
        return tot;
    }

    /// <summary>Finite-n CvM CDF (Csörgő &amp; Faraway 1996), scipy's <c>_cdf_cvm</c>.</summary>
    public static double CdfCvm(double x, int n)
    {
        if (x >= n / 3.0) return 1.0;
        if (!(1.0 / (12 * n) < x)) return 0.0;
        return CdfCvmInf(x) * (1 + 1.0 / (12 * n)) + Psi1Mod(x) / n;
    }

    /// <summary>
    /// Exact two-sample CvM p-value P(U ≥ u) by Xiao, Gordon &amp; Yakovlev's (2007) recursion over the
    /// lattice of merged orderings, tracking the distribution of Σ(a·v − b·u)² per column.
    /// </summary>
    public static double CvmTwoSampleExact(double s, int m, int n)
    {
        long lcm = (long)m / Gcd(m, n) * n;
        long a = lcm / m, b = lcm / n;
        double mn = (double)m * n;
        var zeta = Math.Floor(lcm * (double)lcm * (m + n) * (6 * s - mn * (4 * mn - 1)) / (6 * mn * mn));
        var combinations = Special.Binom(m + n, m);
        var gs = new Dictionary<long, double>[m + 1];
        gs[0] = new Dictionary<long, double> { [0] = 1 };
        for (var v = 1; v <= m; v++) gs[v] = new Dictionary<long, double>();
        for (var u = 0; u <= n; u++)
        {
            var next = new Dictionary<long, double>[m + 1];
            var tmp = new Dictionary<long, double>();
            for (var v = 0; v <= m; v++)
            {
                // tmp (arriving from the left) merged with gs[v] (arriving from below).
                var merged = new Dictionary<long, double>(tmp);
                foreach (var (key, f) in gs[v]) merged[key] = merged.TryGetValue(key, out var e) ? e + f : f;
                var res = (a * v - b * u) * (a * v - b * u);
                tmp = new Dictionary<long, double>(merged.Count);
                foreach (var (key, f) in merged) tmp[key + res] = f;
                next[v] = tmp;
            }
            gs = next;
        }
        var total = 0.0;
        foreach (var (value, f) in gs[m]) if (value >= zeta) total += f;
        return total / combinations;
    }

    private static long Gcd(long a, long b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    // ------------------------------------------------------------------ Anderson–Darling k-sample

    public static double AndersonMidrank(double[][] samples, double[] Z, double[] zstar, int[] n, int N)
    {
        var L = zstar.Length;
        var left = new double[L];
        var lj = new double[L];
        for (var j = 0; j < L; j++)
        {
            left[j] = Stats.LowerBound(Z, zstar[j]);
            lj[j] = N == L ? 1.0 : Stats.UpperBound(Z, zstar[j]) - left[j];
        }
        var A = 0.0;
        for (var i = 0; i < samples.Length; i++)
        {
            var s = StatsTestMath.Sorted(samples[i]);
            var inner = 0.0;
            for (var j = 0; j < L; j++)
            {
                var bj = left[j] + lj[j] / 2.0;
                double r = Stats.UpperBound(s, zstar[j]);
                var fij = r - Stats.LowerBound(s, zstar[j]);
                var mij = r - fij / 2.0;
                var t = N * mij - bj * n[i];
                inner += lj[j] / N * t * t / (bj * (N - bj) - N * lj[j] / 4.0);
            }
            A += inner / n[i];
        }
        return A * (N - 1.0) / N;
    }

    public static double AndersonRight(double[][] samples, double[] Z, double[] zstar, int[] n, int N)
    {
        var L = zstar.Length - 1;
        var lj = new double[L];
        var bj = new double[L];
        var acc = 0.0;
        for (var j = 0; j < L; j++)
        {
            lj[j] = Stats.UpperBound(Z, zstar[j]) - Stats.LowerBound(Z, zstar[j]);
            acc += lj[j];
            bj[j] = acc;
        }
        var A = 0.0;
        for (var i = 0; i < samples.Length; i++)
        {
            var s = StatsTestMath.Sorted(samples[i]);
            var inner = 0.0;
            for (var j = 0; j < L; j++)
            {
                double mij = Stats.UpperBound(s, zstar[j]);
                var t = N * mij - bj[j] * n[i];
                inner += lj[j] / N * t * t / (bj[j] * (N - bj[j]));
            }
            A += inner / n[i];
        }
        return A;
    }

    public static double AndersonContinuous(double[][] samples, double[] Z, int[] n, int N)
    {
        var A = 0.0;
        for (var i = 0; i < samples.Length; i++)
        {
            var s = StatsTestMath.Sorted(samples[i]);
            var inner = 0.0;
            for (var j = 1; j < N; j++)
            {
                double mij = Stats.UpperBound(s, Z[j - 1]);
                var t = N * mij - (double)j * n[i];
                inner += t * t / ((double)j * (N - j));
            }
            A += inner / n[i];
        }
        return A / N;
    }

    /// <summary>Least-squares quadratic fit (numpy's <c>polyfit(x, y, 2)</c>), highest power first.</summary>
    public static double[] PolyFit2(double[] x, double[] y)
    {
        // Normal equations on centred/scaled x keep the 3×3 system well conditioned.
        var n = x.Length;
        double[,] m = new double[3, 3];
        var r = new double[3];
        for (var i = 0; i < n; i++)
        {
            double[] row = [x[i] * x[i], x[i], 1.0];
            for (var a = 0; a < 3; a++)
            {
                r[a] += row[a] * y[i];
                for (var b = 0; b < 3; b++) m[a, b] += row[a] * row[b];
            }
        }
        // Gaussian elimination with partial pivoting.
        for (var c = 0; c < 3; c++)
        {
            var p = c;
            for (var k = c + 1; k < 3; k++) if (Math.Abs(m[k, c]) > Math.Abs(m[p, c])) p = k;
            if (p != c)
            {
                for (var k = 0; k < 3; k++) (m[c, k], m[p, k]) = (m[p, k], m[c, k]);
                (r[c], r[p]) = (r[p], r[c]);
            }
            for (var k = c + 1; k < 3; k++)
            {
                var f = m[k, c] / m[c, c];
                for (var l = c; l < 3; l++) m[k, l] -= f * m[c, l];
                r[k] -= f * r[c];
            }
        }
        var coef = new double[3];
        for (var c = 2; c >= 0; c--)
        {
            var s = r[c];
            for (var l = c + 1; l < 3; l++) s -= m[c, l] * coef[l];
            coef[c] = s / m[c, c];
        }
        return coef;
    }

    // ------------------------------------------------------------------ logistic MLE for Anderson

    /// <summary>Solves the logistic likelihood equations (scipy's rootfunc) by Newton's method.</summary>
    public static (double Loc, double Scale) LogisticFit(double[] x, double a0, double b0)
    {
        double a = a0, b = b0;
        var n = x.Length;
        for (var it = 0; it < 100; it++)
        {
            double f1 = -0.5 * n, f2 = n, j11 = 0, j12 = 0, j21 = 0, j22 = 0;
            foreach (var xi in x)
            {
                var t = (xi - a) / b;
                var s = 1.0 / (1.0 + Math.Exp(t));
                f1 += s;
                f2 += t * (2 * s - 1);
                var ds = s * (1 - s);
                j11 += ds / b;
                j12 += ds * t / b;
                var gp = 2 * s - 1 - 2 * t * ds;
                j21 += -gp / b;
                j22 += -gp * t / b;
            }
            var det = j11 * j22 - j12 * j21;
            var da = (f1 * j22 - j12 * f2) / det;
            var db = (j11 * f2 - j21 * f1) / det;
            a -= da;
            b -= db;
            if (b <= 0) b = b0 * 0.5;
            if (Math.Abs(da) < 1e-14 * Math.Max(1, Math.Abs(a)) && Math.Abs(db) < 1e-14 * Math.Max(1, b)) break;
        }
        return (a, b);
    }

    // ------------------------------------------------------------------ Shapiro–Wilk (AS R94)

    private static double Poly(ReadOnlySpan<double> cc, double x)
    {
        var ret = cc[0];
        var nord = cc.Length;
        if (nord > 1)
        {
            var p = x * cc[nord - 1];
            for (var j = nord - 2; j > 0; j--) p = (p + cc[j]) * x;
            ret += p;
        }
        return ret;
    }

    /// <summary>
    /// Royston (1995), Algorithm AS R94: the W statistic from approximated Shapiro–Wilk coefficients
    /// and its p-value from a normalising transformation of log(1 − W). <paramref name="x"/> is sorted.
    /// </summary>
    public static (double W, double Pw) Swilk(double[] x)
    {
        const double small = 1e-19;
        double[] g = [-2.273, .459];
        double[] c1 = [0.0, .221157, -.147981, -2.07119, 4.434685, -2.706056];
        double[] c2 = [0.0, .042981, -.293762, -1.752461, 5.682633, -3.582633];
        double[] c3 = [.544, -.39978, .025054, -6.714e-4];
        double[] c4 = [1.3822, -.77857, .062767, -.0020322];
        double[] c5 = [-1.5861, -.31082, -.083751, .0038915];
        double[] c6 = [-.4803, -.082676, .0030302];
        var n = x.Length;
        var nn2 = n / 2;
        var a = new double[nn2 + 1]; // 1-based
        double an = n;
        if (n == 3) a[1] = Math.Sqrt(0.5);
        else
        {
            var an25 = an + .25;
            var summ2 = 0.0;
            for (var i = 1; i <= nn2; i++)
            {
                a[i] = Special.Ndtri((i - .375) / an25);
                summ2 += a[i] * a[i];
            }
            summ2 *= 2;
            var ssumm2 = Math.Sqrt(summ2);
            var rsn = 1 / Math.Sqrt(an);
            var a1 = Poly(c1, rsn) - a[1] / ssumm2;
            int i1;
            double fac;
            if (n > 5)
            {
                i1 = 3;
                var a2 = -a[2] / ssumm2 + Poly(c2, rsn);
                fac = Math.Sqrt((summ2 - 2 * (a[1] * a[1]) - 2 * (a[2] * a[2])) / (1 - 2 * (a1 * a1) - 2 * (a2 * a2)));
                a[2] = a2;
            }
            else
            {
                i1 = 2;
                fac = Math.Sqrt((summ2 - 2 * (a[1] * a[1])) / (1 - 2 * (a1 * a1)));
            }
            a[1] = a1;
            for (var i = i1; i <= nn2; i++) a[i] /= -fac;
        }
        var range = x[n - 1] - x[0];
        if (range < small) return (1.0, 1.0);
        var xx = x[0] / range;
        var sx = xx;
        var sa = -a[1];
        for (int i = 1, j = n - 1; i < n; j--)
        {
            var xi = x[i] / range;
            sx += xi;
            i++;
            if (i != j) sa += Math.Sign(i - j) * a[Math.Min(i, j)];
            xx = xi;
        }
        sa /= n;
        sx /= n;
        double ssa = 0, ssx = 0, sax = 0;
        for (int i = 0, j = n - 1; i < n; i++, j--)
        {
            var asa = i != j ? Math.Sign(i - j) * a[1 + Math.Min(i, j)] - sa : -sa;
            var xsx = x[i] / range - sx;
            ssa += asa * asa;
            ssx += xsx * xsx;
            sax += asa * xsx;
        }
        var ssassx = Math.Sqrt(ssa * ssx);
        var w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
        var w = 1 - w1;
        if (n == 3)
        {
            const double pi6 = 1.90985931710274, stqr = 1.04719755119660;
            var pw3 = pi6 * (Math.Asin(Math.Sqrt(w)) - stqr);
            return (w, Math.Max(pw3, 0));
        }
        var y = Math.Log(w1);
        var lxx = Math.Log(an);
        double m, s;
        if (n <= 11)
        {
            var gamma = Poly(g, an);
            if (y >= gamma) return (w, 1e-99);
            y = -Math.Log(gamma - y);
            m = Poly(c3, an);
            s = Math.Exp(Poly(c4, an));
        }
        else
        {
            m = Poly(c5, lxx);
            s = Math.Exp(Poly(c6, lxx));
        }
        return (w, Special.Ndtr(-(y - m) / s));
    }
}
