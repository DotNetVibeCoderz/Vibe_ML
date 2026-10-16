using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Optimize
{
    static partial void RunMinpackLm(MinpackLmProblem problem, ref OptimizeResult? result) => result = OptimizeMinpack.Lm(problem);
}

/// <summary>
/// MINPACK (Moré, Garbow &amp; Hillstrom 1980) translated routine by routine: <c>enorm</c>,
/// <c>qrfac</c>, <c>qform</c>, <c>qrsolv</c>, <c>lmpar</c>, <c>dogleg</c>, <c>r1updt</c>,
/// <c>r1mpyq</c>, and the drivers <c>lmdif</c>/<c>lmder</c> (Levenberg–Marquardt) and
/// <c>hybrd</c>/<c>hybrj</c> (Powell's hybrid method). Indexing stays 1-based and column-major,
/// like the Fortran, so each line can be checked against the original.
/// </summary>
internal static class OptimizeMinpack
{
    private const double Epsmch = 2.220446049250313e-16;
    private const double Dwarf = 2.2250738585072014e-308;

    /// <summary>1-based column-major matrix.</summary>
    internal sealed class FMat(int rows, int cols)
    {
        public readonly double[] A = new double[Math.Max(1, rows * cols)];
        public readonly int Rows = rows, Cols = cols;
        public ref double this[int i, int j] => ref A[(i - 1) + (j - 1) * Rows];
    }

    /// <summary>Euclidean norm of v[off+1 .. off+n] with MINPACK's overflow/underflow guards.</summary>
    public static double Enorm(int n, double[] v, int off = 0)
    {
        const double rdwarf = 3.834e-20, rgiant = 1.304e19;
        double s1 = 0, s2 = 0, s3 = 0, x1max = 0, x3max = 0;
        var agiant = rgiant / n;
        for (var i = 1; i <= n; i++)
        {
            var xabs = Math.Abs(v[off + i]);
            if (xabs > rdwarf && xabs < agiant) s2 += xabs * xabs;
            else if (xabs <= rdwarf)
            {
                if (xabs > x3max) { var r = x3max / xabs; s3 = 1 + s3 * r * r; x3max = xabs; }
                else if (xabs != 0) { var r = xabs / x3max; s3 += r * r; }
            }
            else
            {
                if (xabs > x1max) { var r = x1max / xabs; s1 = 1 + s1 * r * r; x1max = xabs; }
                else { var r = xabs / x1max; s1 += r * r; }
            }
        }
        if (s1 != 0) return x1max * Math.Sqrt(s1 + (s2 / x1max) / x1max);
        if (s2 != 0)
            return s2 >= x3max ? Math.Sqrt(s2 * (1 + (x3max / s2) * (x3max * s3))) : Math.Sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
        return x3max * Math.Sqrt(s3);
    }

    /// <summary>Column j of a as a 1-based vector view helper: Enorm over rows r0..m of column j.</summary>
    private static double EnormCol(FMat a, int j, int r0, int m)
    {
        var tmp = new double[m - r0 + 2];
        for (var i = r0; i <= m; i++) tmp[i - r0 + 1] = a[i, j];
        return Enorm(m - r0 + 1, tmp);
    }

    public static void Qrfac(int m, int n, FMat a, bool pivot, int[] ipvt, double[] rdiag, double[] acnorm, double[] wa)
    {
        for (var j = 1; j <= n; j++)
        {
            acnorm[j] = EnormCol(a, j, 1, m);
            rdiag[j] = acnorm[j];
            wa[j] = rdiag[j];
            if (pivot) ipvt[j] = j;
        }
        var minmn = Math.Min(m, n);
        for (var j = 1; j <= minmn; j++)
        {
            if (pivot)
            {
                var kmax = j;
                for (var k = j; k <= n; k++) if (rdiag[k] > rdiag[kmax]) kmax = k;
                if (kmax != j)
                {
                    for (var i = 1; i <= m; i++) (a[i, j], a[i, kmax]) = (a[i, kmax], a[i, j]);
                    rdiag[kmax] = rdiag[j];
                    wa[kmax] = wa[j];
                    (ipvt[j], ipvt[kmax]) = (ipvt[kmax], ipvt[j]);
                }
            }
            var ajnorm = EnormCol(a, j, j, m);
            if (ajnorm != 0)
            {
                if (a[j, j] < 0) ajnorm = -ajnorm;
                for (var i = j; i <= m; i++) a[i, j] /= ajnorm;
                a[j, j] += 1;
                for (var k = j + 1; k <= n; k++)
                {
                    double sum = 0;
                    for (var i = j; i <= m; i++) sum += a[i, j] * a[i, k];
                    var temp = sum / a[j, j];
                    for (var i = j; i <= m; i++) a[i, k] -= temp * a[i, j];
                    if (pivot && rdiag[k] != 0)
                    {
                        temp = a[j, k] / rdiag[k];
                        rdiag[k] *= Math.Sqrt(Math.Max(0, 1 - temp * temp));
                        var q = rdiag[k] / wa[k];
                        if (0.05 * q * q <= Epsmch)
                        {
                            rdiag[k] = j + 1 <= m ? EnormCol(a, k, j + 1, m) : 0;
                            wa[k] = rdiag[k];
                        }
                    }
                }
            }
            rdiag[j] = -ajnorm;
        }
    }

    public static void Qform(int m, int n, FMat q, double[] wa)
    {
        var minmn = Math.Min(m, n);
        for (var j = 2; j <= minmn; j++) for (var i = 1; i <= j - 1; i++) q[i, j] = 0;
        for (var j = n + 1; j <= m; j++) { for (var i = 1; i <= m; i++) q[i, j] = 0; q[j, j] = 1; }
        for (var l = 1; l <= minmn; l++)
        {
            var k = minmn - l + 1;
            for (var i = k; i <= m; i++) { wa[i] = q[i, k]; q[i, k] = 0; }
            q[k, k] = 1;
            if (wa[k] == 0) continue;
            for (var j = k; j <= m; j++)
            {
                double sum = 0;
                for (var i = k; i <= m; i++) sum += q[i, j] * wa[i];
                var temp = sum / wa[k];
                for (var i = k; i <= m; i++) q[i, j] -= temp * wa[i];
            }
        }
    }

    public static void Qrsolv(int n, FMat r, int[] ipvt, double[] diag, double[] qtb, double[] x, double[] sdiag, double[] wa)
    {
        for (var j = 1; j <= n; j++)
        {
            for (var i = j; i <= n; i++) r[i, j] = r[j, i];
            x[j] = r[j, j];
            wa[j] = qtb[j];
        }
        for (var j = 1; j <= n; j++)
        {
            var l = ipvt[j];
            if (diag[l] != 0)
            {
                for (var k = j; k <= n; k++) sdiag[k] = 0;
                sdiag[j] = diag[l];
                double qtbpj = 0;
                for (var k = j; k <= n; k++)
                {
                    if (sdiag[k] == 0) continue;
                    double sin, cos;
                    if (Math.Abs(r[k, k]) < Math.Abs(sdiag[k]))
                    {
                        var cotan = r[k, k] / sdiag[k];
                        sin = 0.5 / Math.Sqrt(0.25 + 0.25 * cotan * cotan);
                        cos = sin * cotan;
                    }
                    else
                    {
                        var tan = sdiag[k] / r[k, k];
                        cos = 0.5 / Math.Sqrt(0.25 + 0.25 * tan * tan);
                        sin = cos * tan;
                    }
                    r[k, k] = cos * r[k, k] + sin * sdiag[k];
                    var temp = cos * wa[k] + sin * qtbpj;
                    qtbpj = -sin * wa[k] + cos * qtbpj;
                    wa[k] = temp;
                    for (var i = k + 1; i <= n; i++)
                    {
                        temp = cos * r[i, k] + sin * sdiag[i];
                        sdiag[i] = -sin * r[i, k] + cos * sdiag[i];
                        r[i, k] = temp;
                    }
                }
            }
            sdiag[j] = r[j, j];
            r[j, j] = x[j];
        }
        var nsing = n;
        for (var j = 1; j <= n; j++)
        {
            if (sdiag[j] == 0 && nsing == n) nsing = j - 1;
            if (nsing < n) wa[j] = 0;
        }
        for (var k = 1; k <= nsing; k++)
        {
            var j = nsing - k + 1;
            double sum = 0;
            for (var i = j + 1; i <= nsing; i++) sum += r[i, j] * wa[i];
            wa[j] = (wa[j] - sum) / sdiag[j];
        }
        for (var j = 1; j <= n; j++) x[ipvt[j]] = wa[j];
    }

    public static void Lmpar(int n, FMat r, int[] ipvt, double[] diag, double[] qtb, double delta, ref double par,
        double[] x, double[] sdiag, double[] wa1, double[] wa2)
    {
        const double p1 = 0.1, p001 = 0.001;
        var nsing = n;
        for (var j = 1; j <= n; j++)
        {
            wa1[j] = qtb[j];
            if (r[j, j] == 0 && nsing == n) nsing = j - 1;
            if (nsing < n) wa1[j] = 0;
        }
        for (var k = 1; k <= nsing; k++)
        {
            var j = nsing - k + 1;
            wa1[j] /= r[j, j];
            var temp = wa1[j];
            for (var i = 1; i <= j - 1; i++) wa1[i] -= r[i, j] * temp;
        }
        for (var j = 1; j <= n; j++) x[ipvt[j]] = wa1[j];
        var iter = 0;
        for (var j = 1; j <= n; j++) wa2[j] = diag[j] * x[j];
        var dxnorm = Enorm(n, wa2);
        var fp = dxnorm - delta;
        if (fp <= p1 * delta) { par = 0; return; }
        double parl = 0;
        if (nsing >= n)
        {
            for (var j = 1; j <= n; j++) { var l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
            for (var j = 1; j <= n; j++)
            {
                double sum = 0;
                for (var i = 1; i <= j - 1; i++) sum += r[i, j] * wa1[i];
                wa1[j] = (wa1[j] - sum) / r[j, j];
            }
            var temp = Enorm(n, wa1);
            parl = ((fp / delta) / temp) / temp;
        }
        for (var j = 1; j <= n; j++)
        {
            double sum = 0;
            for (var i = 1; i <= j; i++) sum += r[i, j] * qtb[i];
            wa1[j] = sum / diag[ipvt[j]];
        }
        var gnorm = Enorm(n, wa1);
        var paru = gnorm / delta;
        if (paru == 0) paru = Dwarf / Math.Min(delta, p1);
        par = Math.Max(par, parl);
        par = Math.Min(par, paru);
        if (par == 0) par = gnorm / dxnorm;
        while (true)
        {
            iter++;
            if (par == 0) par = Math.Max(Dwarf, p001 * paru);
            var temp = Math.Sqrt(par);
            for (var j = 1; j <= n; j++) wa1[j] = temp * diag[j];
            Qrsolv(n, r, ipvt, wa1, qtb, x, sdiag, wa2);
            for (var j = 1; j <= n; j++) wa2[j] = diag[j] * x[j];
            dxnorm = Enorm(n, wa2);
            temp = fp;
            fp = dxnorm - delta;
            if (Math.Abs(fp) <= p1 * delta || (parl == 0 && fp <= temp && temp < 0) || iter == 10) break;
            for (var j = 1; j <= n; j++) { var l = ipvt[j]; wa1[j] = diag[l] * (wa2[l] / dxnorm); }
            for (var j = 1; j <= n; j++)
            {
                wa1[j] /= sdiag[j];
                temp = wa1[j];
                for (var i = j + 1; i <= n; i++) wa1[i] -= r[i, j] * temp;
            }
            temp = Enorm(n, wa1);
            var parc = ((fp / delta) / temp) / temp;
            if (fp > 0) parl = Math.Max(parl, par);
            if (fp < 0) paru = Math.Min(paru, par);
            par = Math.Max(parl, par + parc);
        }
    }

    /// <summary>lmdif / lmder. Returns the MINPACK result with X, residuals, final Jacobian and cov_x in HessInv.</summary>
    public static OptimizeResult Lm(MinpackLmProblem p)
    {
        var n = p.X0.Length;
        var x = new double[n + 1];
        for (var i = 1; i <= n; i++) x[i] = p.X0[i - 1];
        var nfev = 0; var njev = 0;
        double[] F(double[] x1)
        {
            nfev++;
            var xs = new double[n];
            Array.Copy(x1, 1, xs, 0, n);
            var v = p.Fun(xs);
            var r = new double[v.Length + 1];
            Array.Copy(v, 0, r, 1, v.Length);
            return r;
        }
        var fvec = F(x);
        var m = fvec.Length - 1;
        if (n > m) throw new ArgumentException($"Improper input: func input vector length N={n} must not exceed func output vector length M={m}.");
        var maxfev = p.MaxFev > 0 ? p.MaxFev : (p.Jac is null ? 200 * (n + 1) : 100 * (n + 1));
        var mode = p.Diag is null ? 1 : 2;
        var diag = new double[n + 1];
        if (p.Diag is not null) { if (p.Diag.Length != n) throw new ArgumentException("diag must have one entry per variable."); for (var i = 1; i <= n; i++) diag[i] = p.Diag[i - 1]; }
        var epsfcn = p.Epsfcn ?? Epsmch;
        double ftol = p.Ftol, xtol = p.Xtol, gtol = p.Gtol, factor = p.Factor;
        var info = 0;
        var fjac = new FMat(m, n);
        var ipvt = new int[n + 1];
        var iter = 1;
        if (ftol < 0 || xtol < 0 || gtol < 0 || factor <= 0 || (mode == 2 && diag.Skip(1).Any(d => d <= 0)))
            return Finish(0);

        var qtf = new double[n + 1];
        double[] wa1 = new double[n + 1], wa2 = new double[n + 1], wa3 = new double[n + 1], wa4 = new double[m + 1];
        var fnorm = Enorm(m, fvec);
        double par = 0, delta = 0, xnorm = 0;

        while (info == 0)
        {
            // Jacobian.
            if (p.Jac is not null)
            {
                njev++;
                var xs = new double[n];
                Array.Copy(x, 1, xs, 0, n);
                var jm = p.Jac(xs);
                if (jm.Rank != 2 || jm.Shape[0] != m || jm.Shape[1] != n) throw new ArgumentException($"The Jacobian must have shape ({m}, {n}).");
                var js = jm.AsContiguous().ToArray();
                for (var i = 1; i <= m; i++) for (var j = 1; j <= n; j++) fjac[i, j] = js[(i - 1) * n + (j - 1)];
            }
            else
            {
                njev++;
                var eps = Math.Sqrt(Math.Max(epsfcn, Epsmch));
                for (var j = 1; j <= n; j++)
                {
                    var temp = x[j];
                    var h = eps * Math.Abs(temp);
                    if (h == 0) h = eps;
                    x[j] = temp + h;
                    var wa = F(x);
                    x[j] = temp;
                    for (var i = 1; i <= m; i++) fjac[i, j] = (wa[i] - fvec[i]) / h;
                }
            }
            Qrfac(m, n, fjac, true, ipvt, wa1, wa2, wa3);
            if (iter == 1)
            {
                if (mode != 2) for (var j = 1; j <= n; j++) { diag[j] = wa2[j]; if (wa2[j] == 0) diag[j] = 1; }
                for (var j = 1; j <= n; j++) wa3[j] = diag[j] * x[j];
                xnorm = Enorm(n, wa3);
                delta = factor * xnorm;
                if (delta == 0) delta = factor;
            }
            for (var i = 1; i <= m; i++) wa4[i] = fvec[i];
            for (var j = 1; j <= n; j++)
            {
                if (fjac[j, j] != 0)
                {
                    double sum = 0;
                    for (var i = j; i <= m; i++) sum += fjac[i, j] * wa4[i];
                    var temp = -sum / fjac[j, j];
                    for (var i = j; i <= m; i++) wa4[i] += fjac[i, j] * temp;
                }
                fjac[j, j] = wa1[j];
                qtf[j] = wa4[j];
            }
            double gnorm = 0;
            if (fnorm != 0)
                for (var j = 1; j <= n; j++)
                {
                    var l = ipvt[j];
                    if (wa2[l] == 0) continue;
                    double sum = 0;
                    for (var i = 1; i <= j; i++) sum += fjac[i, j] * (qtf[i] / fnorm);
                    gnorm = Math.Max(gnorm, Math.Abs(sum / wa2[l]));
                }
            if (gnorm <= gtol) { info = 4; break; }
            if (mode != 2) for (var j = 1; j <= n; j++) diag[j] = Math.Max(diag[j], wa2[j]);

            double ratio;
            do
            {
                Lmpar(n, fjac, ipvt, diag, qtf, delta, ref par, wa1, wa2, wa3, wa4);
                for (var j = 1; j <= n; j++) { wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j]; }
                var pnorm = Enorm(n, wa3);
                if (iter == 1) delta = Math.Min(delta, pnorm);
                var fnew = F(wa2);
                Array.Copy(fnew, wa4, m + 1);
                var fnorm1 = Enorm(m, wa4);
                double actred = -1;
                if (0.1 * fnorm1 < fnorm) { var q = fnorm1 / fnorm; actred = 1 - q * q; }
                for (var j = 1; j <= n; j++)
                {
                    wa3[j] = 0;
                    var temp = wa1[ipvt[j]];
                    for (var i = 1; i <= j; i++) wa3[i] += fjac[i, j] * temp;
                }
                var temp1 = Enorm(n, wa3) / fnorm;
                var temp2 = (Math.Sqrt(par) * pnorm) / fnorm;
                var prered = temp1 * temp1 + temp2 * temp2 / 0.5;
                var dirder = -(temp1 * temp1 + temp2 * temp2);
                ratio = prered != 0 ? actred / prered : 0;
                if (ratio <= 0.25)
                {
                    var temp = actred >= 0 ? 0.5 : 0.5 * dirder / (dirder + 0.5 * actred);
                    if (0.1 * fnorm1 >= fnorm || temp < 0.1) temp = 0.1;
                    delta = temp * Math.Min(delta, pnorm / 0.1);
                    par /= temp;
                }
                else if (par == 0 || ratio >= 0.75)
                {
                    delta = pnorm / 0.5;
                    par *= 0.5;
                }
                if (ratio >= 1e-4)
                {
                    for (var j = 1; j <= n; j++) { x[j] = wa2[j]; wa2[j] = diag[j] * x[j]; }
                    Array.Copy(wa4, fvec, m + 1);
                    xnorm = Enorm(n, wa2);
                    fnorm = fnorm1;
                    iter++;
                }
                if (Math.Abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1) info = 1;
                if (delta <= xtol * xnorm) info = 2;
                if (Math.Abs(actred) <= ftol && prered <= ftol && 0.5 * ratio <= 1 && info == 2) info = 3;
                if (info != 0) break;
                if (nfev >= maxfev) info = 5;
                if (Math.Abs(actred) <= Epsmch && prered <= Epsmch && 0.5 * ratio <= 1) info = 6;
                if (delta <= Epsmch * xnorm) info = 7;
                if (gnorm <= Epsmch) info = 8;
                if (info != 0) break;
            } while (ratio < 1e-4);
        }
        return Finish(info);

        OptimizeResult Finish(int inf)
        {
            var xs = new double[n];
            Array.Copy(x, 1, xs, 0, n);
            var fv = new double[m > 0 ? m : 0];
            if (m > 0) Array.Copy(fvec, 1, fv, 0, m);
            double ss = 0;
            foreach (var v in fv) ss += v * v;
            string[] msgs =
            [
                "Improper input parameters.",
                $"Both actual and predicted relative reductions in the sum of squares\n  are at most {ftol:F6}",
                $"The relative error between two consecutive iterates is at most {xtol:F6}",
                $"Both actual and predicted relative reductions in the sum of squares\n  are at most {ftol:F6} and the relative error between two consecutive iterates is at \n  most {xtol:F6}",
                $"The cosine of the angle between func(x) and any column of the\n  Jacobian is at most {gtol:F6} in absolute value",
                $"Number of calls to function has reached maxfev = {maxfev}.",
                $"ftol={ftol:F6} is too small, no further reduction in the sum of squares\n  is possible.",
                $"xtol={xtol:F6} is too small, no further improvement in the approximate\n  solution is possible.",
                $"gtol={gtol:F6} is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.",
            ];
            NdArray? cov = null;
            if (inf is >= 1 and <= 4) cov = CovFromR(fjac, ipvt, n);
            return new OptimizeResult
            {
                X = xs, FunVector = fv, Fun = 0.5 * ss, Nfev = nfev, Njev = njev, Status = inf, Success = inf is >= 1 and <= 4,
                Message = msgs[inf], HessInv = cov, Nit = iter - 1,
            };
        }
    }

    /// <summary>cov_x = (RᵀR)⁻¹ with the column permutation undone, as SciPy's <c>leastsq</c>; null when R is singular.</summary>
    private static NdArray? CovFromR(FMat fjac, int[] ipvt, int n)
    {
        // R is the upper triangle of the first n rows of fjac (after qrfac; diagonal restored by lmdif).
        var r = new double[n * n];
        for (var i = 1; i <= n; i++) for (var j = i; j <= n; j++) r[(i - 1) * n + (j - 1)] = fjac[i, j];
        for (var i = 0; i < n; i++) if (r[i * n + i] == 0) return null;
        // Invert the upper-triangular R column by column.
        var inv = new double[n * n];
        for (var c = 0; c < n; c++)
        {
            for (var i = n - 1; i >= 0; i--)
            {
                var s = i == c ? 1.0 : 0.0;
                for (var k = i + 1; k < n; k++) s -= r[i * n + k] * inv[k * n + c];
                inv[i * n + c] = s / r[i * n + i];
            }
        }
        var cov = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                double s = 0;
                for (var k = 0; k < n; k++) s += inv[i * n + k] * inv[j * n + k];
                cov[(ipvt[i + 1] - 1) * n + (ipvt[j + 1] - 1)] = s;
            }
        foreach (var v in cov) if (!double.IsFinite(v)) return null;
        return new NdArray(cov, n, n);
    }

    // ----- hybrd / hybrj -----

    public static void Dogleg(int n, double[] r, double[] diag, double[] qtb, double delta, double[] x, double[] wa1, double[] wa2)
    {
        var jj = n * (n + 1) / 2 + 1;
        for (var k = 1; k <= n; k++)
        {
            var j = n - k + 1;
            jj -= k;
            var l = jj + 1;
            double sum = 0;
            for (var i = j + 1; i <= n; i++) { sum += r[l] * x[i]; l++; }
            var temp = r[jj];
            if (temp == 0)
            {
                l = j;
                for (var i = 1; i <= j; i++) { temp = Math.Max(temp, Math.Abs(r[l])); l = l + n - i; }
                temp = Epsmch * temp;
                if (temp == 0) temp = Epsmch;
            }
            x[j] = (qtb[j] - sum) / temp;
        }
        for (var j = 1; j <= n; j++) { wa1[j] = 0; wa2[j] = diag[j] * x[j]; }
        var qnorm = Enorm(n, wa2);
        if (qnorm <= delta) return;
        {
            var l = 1;
            for (var j = 1; j <= n; j++)
            {
                var temp = qtb[j];
                for (var i = j; i <= n; i++) { wa1[i] += r[l] * temp; l++; }
                wa1[j] /= diag[j];
            }
        }
        var gnorm = Enorm(n, wa1);
        double sgnorm = 0;
        var alpha = delta / qnorm;
        if (gnorm != 0)
        {
            for (var j = 1; j <= n; j++) wa1[j] = (wa1[j] / gnorm) / diag[j];
            var l = 1;
            for (var j = 1; j <= n; j++)
            {
                double sum = 0;
                for (var i = j; i <= n; i++) { sum += r[l] * wa1[i]; l++; }
                wa2[j] = sum;
            }
            var temp = Enorm(n, wa2);
            sgnorm = (gnorm / temp) / temp;
            alpha = 0;
            if (sgnorm < delta)
            {
                var bnorm = Enorm(n, qtb);
                temp = (bnorm / gnorm) * (bnorm / qnorm) * (sgnorm / delta);
                var dq = delta / qnorm; var sd = sgnorm / delta;
                temp = temp - dq * sd * sd + Math.Sqrt((temp - dq) * (temp - dq) + (1 - dq * dq) * (1 - sd * sd));
                alpha = (dq * (1 - sd * sd)) / temp;
            }
        }
        var t2 = (1 - alpha) * Math.Min(sgnorm, delta);
        for (var j = 1; j <= n; j++) x[j] = t2 * wa1[j] + alpha * x[j];
    }

    public static bool R1updt(int m, int n, double[] s, double[] u, double[] v, double[] w)
    {
        var giant = double.MaxValue;
        var jj = (n * (2 * m - n + 1)) / 2 - (m - n);
        var l = jj;
        for (var i = n; i <= m; i++) { w[i] = s[l]; l++; }
        for (var nmj = 1; nmj <= n - 1; nmj++)
        {
            var j = n - nmj;
            jj -= m - j + 1;
            w[j] = 0;
            if (v[j] == 0) continue;
            double sin, cos, tau;
            if (Math.Abs(v[n]) < Math.Abs(v[j]))
            {
                var cotan = v[n] / v[j];
                sin = 0.5 / Math.Sqrt(0.25 + 0.25 * cotan * cotan);
                cos = sin * cotan;
                tau = 1;
                if (Math.Abs(cos) * giant > 1) tau = 1 / cos;
            }
            else
            {
                var tan = v[j] / v[n];
                cos = 0.5 / Math.Sqrt(0.25 + 0.25 * tan * tan);
                sin = cos * tan;
                tau = sin;
            }
            v[n] = sin * v[j] + cos * v[n];
            v[j] = tau;
            l = jj;
            for (var i = j; i <= m; i++)
            {
                var temp = cos * s[l] - sin * w[i];
                w[i] = sin * s[l] + cos * w[i];
                s[l] = temp;
                l++;
            }
        }
        for (var i = 1; i <= m; i++) w[i] += v[n] * u[i];
        var sing = false;
        for (var j = 1; j <= n - 1; j++)
        {
            if (w[j] != 0)
            {
                double sin, cos, tau;
                if (Math.Abs(s[jj]) < Math.Abs(w[j]))
                {
                    var cotan = s[jj] / w[j];
                    sin = 0.5 / Math.Sqrt(0.25 + 0.25 * cotan * cotan);
                    cos = sin * cotan;
                    tau = 1;
                    if (Math.Abs(cos) * giant > 1) tau = 1 / cos;
                }
                else
                {
                    var tan = w[j] / s[jj];
                    cos = 0.5 / Math.Sqrt(0.25 + 0.25 * tan * tan);
                    sin = cos * tan;
                    tau = sin;
                }
                l = jj;
                for (var i = j; i <= m; i++)
                {
                    var temp = cos * s[l] + sin * w[i];
                    w[i] = -sin * s[l] + cos * w[i];
                    s[l] = temp;
                    l++;
                }
                w[j] = tau;
            }
            if (s[jj] == 0) sing = true;
            jj += m - j + 1;
        }
        l = jj;
        for (var i = n; i <= m; i++) { s[l] = w[i]; l++; }
        if (s[jj] == 0) sing = true;
        return sing;
    }

    public static void R1mpyq(int m, int n, FMat a, double[] v, double[] w)
    {
        for (var nmj = 1; nmj <= n - 1; nmj++)
        {
            var j = n - nmj;
            double cos, sin;
            if (Math.Abs(v[j]) > 1) { cos = 1 / v[j]; sin = Math.Sqrt(1 - cos * cos); }
            else { sin = v[j]; cos = Math.Sqrt(1 - sin * sin); }
            for (var i = 1; i <= m; i++)
            {
                var temp = cos * a[i, j] - sin * a[i, n];
                a[i, n] = sin * a[i, j] + cos * a[i, n];
                a[i, j] = temp;
            }
        }
        for (var j = 1; j <= n - 1; j++)
        {
            double cos, sin;
            if (Math.Abs(w[j]) > 1) { cos = 1 / w[j]; sin = Math.Sqrt(1 - cos * cos); }
            else { sin = w[j]; cos = Math.Sqrt(1 - sin * sin); }
            for (var i = 1; i <= m; i++)
            {
                var temp = cos * a[i, j] + sin * a[i, n];
                a[i, n] = -sin * a[i, j] + cos * a[i, n];
                a[i, j] = temp;
            }
        }
    }

    /// <summary>Result of <see cref="Hybrd"/>.</summary>
    public sealed record HybrdResult(double[] X, double[] Fvec, int Info, int Nfev, int Njev, double[] Q, double[] R);

    /// <summary>hybrd (Jacobian by forward differences) / hybrj (analytic Jacobian).</summary>
    public static HybrdResult Hybrd(Func<double[], double[]> fun, Func<double[], NdArray>? jac, double[] x0, double xtol, int maxfev,
        double? epsfcn, double factor, double[]? diagIn)
    {
        var n = x0.Length;
        var x = new double[n + 1];
        for (var i = 1; i <= n; i++) x[i] = x0[i - 1];
        int nfev = 0, njev = 0;
        void F(double[] x1, double[] outv)
        {
            nfev++;
            var xs = new double[n];
            Array.Copy(x1, 1, xs, 0, n);
            var v = fun(xs);
            if (v.Length != n) throw new ArgumentException($"Root needs as many equations as unknowns: got {v.Length} for {n}.");
            Array.Copy(v, 0, outv, 1, n);
        }
        if (maxfev <= 0) maxfev = jac is null ? 200 * (n + 1) : 100 * (n + 1);
        var mode = diagIn is null ? 1 : 2;
        var diag = new double[n + 1];
        if (diagIn is not null) for (var i = 1; i <= n; i++) diag[i] = diagIn[i - 1];
        var fvec = new double[n + 1];
        var fjac = new FMat(n, n);
        var lr = n * (n + 1) / 2;
        var r = new double[lr + 1];
        var qtf = new double[n + 1];
        double[] wa1 = new double[n + 1], wa2 = new double[n + 1], wa3 = new double[n + 1], wa4 = new double[n + 1];
        var info = 0;
        if (xtol < 0 || factor <= 0 || (mode == 2 && diag.Skip(1).Any(d => d <= 0))) return Done();

        F(x, fvec);
        var fnorm = Enorm(n, fvec);
        var iter = 1; int ncsuc = 0, ncfail = 0, nslow1 = 0, nslow2 = 0;
        double delta = 0, xnorm = 0;
        var eps = Math.Sqrt(Math.Max(epsfcn ?? Epsmch, Epsmch));
        while (true)
        {
            var jeval = true;
            if (jac is not null)
            {
                njev++;
                var xs = new double[n];
                Array.Copy(x, 1, xs, 0, n);
                var jm = jac(xs);
                if (jm.Rank != 2 || jm.Shape[0] != n || jm.Shape[1] != n) throw new ArgumentException($"The Jacobian must have shape ({n}, {n}).");
                var js = jm.AsContiguous().ToArray();
                for (var i = 1; i <= n; i++) for (var j = 1; j <= n; j++) fjac[i, j] = js[(i - 1) * n + (j - 1)];
            }
            else
            {
                njev++;
                for (var j = 1; j <= n; j++)
                {
                    var temp = x[j];
                    var h = eps * Math.Abs(temp);
                    if (h == 0) h = eps;
                    x[j] = temp + h;
                    F(x, wa1);
                    x[j] = temp;
                    for (var i = 1; i <= n; i++) fjac[i, j] = (wa1[i] - fvec[i]) / h;
                }
            }
            var ipvtDummy = new int[n + 1];
            Qrfac(n, n, fjac, false, ipvtDummy, wa1, wa2, wa3);
            if (iter == 1)
            {
                if (mode != 2) for (var j = 1; j <= n; j++) { diag[j] = wa2[j]; if (wa2[j] == 0) diag[j] = 1; }
                for (var j = 1; j <= n; j++) wa3[j] = diag[j] * x[j];
                xnorm = Enorm(n, wa3);
                delta = factor * xnorm;
                if (delta == 0) delta = factor;
            }
            for (var i = 1; i <= n; i++) qtf[i] = fvec[i];
            for (var j = 1; j <= n; j++)
            {
                if (fjac[j, j] == 0) continue;
                double sum = 0;
                for (var i = j; i <= n; i++) sum += fjac[i, j] * qtf[i];
                var temp = -sum / fjac[j, j];
                for (var i = j; i <= n; i++) qtf[i] += fjac[i, j] * temp;
            }
            for (var j = 1; j <= n; j++)
            {
                var l = j;
                for (var i = 1; i <= j - 1; i++) { r[l] = fjac[i, j]; l = l + n - i; }
                r[l] = wa1[j];
            }
            Qform(n, n, fjac, wa1);
            if (mode != 2) for (var j = 1; j <= n; j++) diag[j] = Math.Max(diag[j], wa2[j]);

            while (true)
            {
                Dogleg(n, r, diag, qtf, delta, wa1, wa2, wa3);
                for (var j = 1; j <= n; j++) { wa1[j] = -wa1[j]; wa2[j] = x[j] + wa1[j]; wa3[j] = diag[j] * wa1[j]; }
                var pnorm = Enorm(n, wa3);
                if (iter == 1) delta = Math.Min(delta, pnorm);
                F(wa2, wa4);
                var fnorm1 = Enorm(n, wa4);
                double actred = -1;
                if (fnorm1 < fnorm) { var q = fnorm1 / fnorm; actred = 1 - q * q; }
                var lp = 1;
                for (var i = 1; i <= n; i++)
                {
                    double sum = 0;
                    for (var j = i; j <= n; j++) { sum += r[lp] * wa1[j]; lp++; }
                    wa3[i] = qtf[i] + sum;
                }
                var tmp = Enorm(n, wa3);
                double prered = 0;
                if (tmp < fnorm) { var q = tmp / fnorm; prered = 1 - q * q; }
                var ratio = prered > 0 ? actred / prered : 0;
                if (ratio < 0.1) { ncsuc = 0; ncfail++; delta *= 0.5; }
                else
                {
                    ncfail = 0; ncsuc++;
                    if (ratio >= 0.5 || ncsuc > 1) delta = Math.Max(delta, pnorm / 0.5);
                    if (Math.Abs(ratio - 1) <= 0.1) delta = pnorm / 0.5;
                }
                if (ratio >= 1e-4)
                {
                    for (var j = 1; j <= n; j++) { x[j] = wa2[j]; wa2[j] = diag[j] * x[j]; fvec[j] = wa4[j]; }
                    xnorm = Enorm(n, wa2);
                    fnorm = fnorm1;
                    iter++;
                }
                nslow1++;
                if (actred >= 0.001) nslow1 = 0;
                if (jeval) nslow2++;
                if (actred >= 0.1) nslow2 = 0;
                if (delta <= xtol * xnorm || fnorm == 0) info = 1;
                if (info != 0) return Done();
                if (nfev >= maxfev) info = 2;
                if (0.1 * Math.Max(0.1 * delta, pnorm) <= Epsmch * xnorm) info = 3;
                if (nslow2 == 5) info = 4;
                if (nslow1 == 10) info = 5;
                if (info != 0) return Done();
                if (ncfail == 2) break;
                for (var j = 1; j <= n; j++)
                {
                    double sum = 0;
                    for (var i = 1; i <= n; i++) sum += fjac[i, j] * wa4[i];
                    wa2[j] = (sum - wa3[j]) / pnorm;
                    wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm);
                    if (ratio >= 1e-4) qtf[j] = sum;
                }
                R1updt(n, n, r, wa1, wa2, wa3);
                R1mpyq(n, n, fjac, wa2, wa3);
                var qt = new FMat(1, n);
                for (var j = 1; j <= n; j++) qt[1, j] = qtf[j];
                R1mpyq(1, n, qt, wa2, wa3);
                for (var j = 1; j <= n; j++) qtf[j] = qt[1, j];
                jeval = false;
            }
        }

        HybrdResult Done()
        {
            var xs = new double[n]; var fs = new double[n];
            Array.Copy(x, 1, xs, 0, n); Array.Copy(fvec, 1, fs, 0, n);
            var q = new double[n * n];
            for (var i = 1; i <= n; i++) for (var j = 1; j <= n; j++) q[(i - 1) * n + j - 1] = fjac[i, j];
            var rr = new double[n * n];
            var l = 1;
            for (var i = 1; i <= n; i++) for (var j = i; j <= n; j++) { rr[(i - 1) * n + j - 1] = r[l]; l++; }
            return new HybrdResult(xs, fs, info, nfev, njev, q, rr);
        }
    }
}
