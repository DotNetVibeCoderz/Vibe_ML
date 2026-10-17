namespace Gravicode.Science.SciNet;

/// <summary>A minimal double-double (unevaluated sum hi + lo) for series that cancel.</summary>
internal readonly struct SpecialDoubleDouble(double hi, double lo = 0)
{
    public readonly double Hi = hi, Lo = lo;

    private static SpecialDoubleDouble TwoSum(double a, double b)
    {
        var s = a + b;
        var bb = s - a;
        return new(s, (a - (s - bb)) + (b - bb));
    }

    private static SpecialDoubleDouble QuickTwoSum(double a, double b)
    {
        var s = a + b;
        return new(s, b - (s - a));
    }

    public static SpecialDoubleDouble operator +(SpecialDoubleDouble a, SpecialDoubleDouble b)
    {
        var s = TwoSum(a.Hi, b.Hi);
        var t = TwoSum(a.Lo, b.Lo);
        var r = QuickTwoSum(s.Hi, s.Lo + t.Hi);
        return QuickTwoSum(r.Hi, r.Lo + t.Lo);
    }

    public static SpecialDoubleDouble operator *(SpecialDoubleDouble a, SpecialDoubleDouble b)
    {
        var p = a.Hi * b.Hi;
        var e = Math.FusedMultiplyAdd(a.Hi, b.Hi, -p);
        e += a.Hi * b.Lo + a.Lo * b.Hi;
        return QuickTwoSum(p, e);
    }

    public static SpecialDoubleDouble operator /(SpecialDoubleDouble a, SpecialDoubleDouble b)
    {
        // Long division: q1 = a/b, r = a − q1 b, q2 = r/b, refine once more.
        var q1 = a.Hi / b.Hi;
        var r = a + new SpecialDoubleDouble(-1) * (b * new SpecialDoubleDouble(q1));
        var q2 = r.Hi / b.Hi;
        r += new SpecialDoubleDouble(-1) * (b * new SpecialDoubleDouble(q2));
        var q3 = r.Hi / b.Hi;
        var q = QuickTwoSum(q1, q2);
        return q + new SpecialDoubleDouble(q3);
    }

    public double Value => Hi + Lo;
}

public static partial class Special
{
    private const int StruveMaxIter = 10000;
    private const double StruveSumEps = 1e-16, StruveSumTiny = 1e-100, StruveGoodEps = 1e-12, StruveAcceptableEps = 1e-7, StruveAcceptableAtol = 1e-300;

    /// <summary>
    /// Struve function H_v(x) (scipy.special.struve), DLMF 11.2.1. Negative x is allowed for integer v.
    /// </summary>
    /// <remarks>
    /// A port of scipy's (Pauli Virtanen's) scheme: the large-x asymptotic expansion (DLMF 11.6.1),
    /// the power series summed in double-double, and the Bessel-function series (DLMF 11.4.19) each
    /// return an error estimate, and the first with relative error below 1e-12 — otherwise the best
    /// below 1e-7 — is returned; NaN when none is acceptable.
    /// </remarks>
    public static double StruveH(double v, double x) => StruveHL(v, x, isH: true);

    /// <summary>Modified Struve function L_v(x) (scipy.special.modstruve), DLMF 11.2.2.</summary>
    /// <remarks>Same three-method scheme as <see cref="StruveH"/>.</remarks>
    public static double StruveL(double v, double x) => StruveHL(v, x, isH: false);

    private static double StruveHL(double v, double z, bool isH)
    {
        if (double.IsNaN(v) || double.IsNaN(z)) return double.NaN;
        int n;
        if (z < 0)
        {
            n = (int)v;
            if (v == n) return (n % 2 == 0 ? -1 : 1) * StruveHL(v, -z, isH);
            return double.NaN;
        }
        if (z == 0)
        {
            if (v < -1) return GammaSgn(v + 1.5) * double.PositiveInfinity;
            if (v == -1) return 2 / Math.Sqrt(Math.PI) * RGamma(0.5);
            return 0;
        }
        var nn = -v - 0.5;
        if (nn == Math.Floor(nn) && nn > 0 && nn < int.MaxValue)
        {
            n = (int)nn;
            return isH ? (n % 2 == 0 ? 1 : -1) * Jv(n + 0.5, z) : Iv(n + 0.5, z);
        }

        Span<double> value = stackalloc double[3];
        Span<double> err = stackalloc double[3];
        if (z >= 0.7 * v + 12)
        {
            value[0] = StruveAsympLargeZ(v, z, isH, out err[0]);
            if (err[0] < StruveGoodEps * Math.Abs(value[0])) return value[0];
        }
        else err[0] = double.PositiveInfinity;

        value[1] = StruvePowerSeries(v, z, isH, out err[1]);
        if (err[1] < StruveGoodEps * Math.Abs(value[1])) return value[1];

        if (Math.Abs(z) < Math.Abs(v) + 20)
        {
            value[2] = StruveBesselSeries(v, z, isH, out err[2]);
            if (err[2] < StruveGoodEps * Math.Abs(value[2])) return value[2];
        }
        else err[2] = double.PositiveInfinity;

        n = 0;
        if (err[1] < err[n]) n = 1;
        if (err[2] < err[n]) n = 2;
        if (err[n] < StruveAcceptableEps * Math.Abs(value[n]) || err[n] < StruveAcceptableAtol) return value[n];

        var tmp = -GammaLn(v + 1.5) + (v + 1) * Math.Log(z / 2);
        if (!isH) tmp = Math.Abs(tmp);
        if (tmp > 700) return double.PositiveInfinity * GammaSgn(v + 1.5);
        return double.NaN;
    }

    private static double StruveAsympLargeZ(double v, double z, bool isH, out double err)
    {
        var sgn = isH ? -1 : 1;
        var m = z / 2;
        var maxiter = m <= 0 ? 0 : m > StruveMaxIter ? StruveMaxIter : (int)m;
        if (maxiter == 0 || z < v) { err = double.PositiveInfinity; return double.NaN; }
        var term = -sgn / Math.Sqrt(Math.PI) * Math.Exp(-GammaLn(v + 0.5) + (v - 1) * Math.Log(z / 2)) * GammaSgn(v + 0.5);
        var sum = term;
        var maxterm = 0.0;
        for (var n = 0; n < maxiter; n++)
        {
            term *= sgn * (1 + 2 * n) * (1 + 2 * n - 2 * v) / (z * z);
            sum += term;
            if (Math.Abs(term) > maxterm) maxterm = Math.Abs(term);
            if (Math.Abs(term) < StruveSumEps * Math.Abs(sum) || term == 0 || !double.IsFinite(sum)) break;
        }
        sum += isH ? Yv(v, z) : Iv(v, z);
        err = Math.Abs(term) + Math.Abs(maxterm) * StruveSumEps;
        return sum;
    }

    private static double StruvePowerSeries(double v, double z, bool isH, out double err)
    {
        var sgn = isH ? -1 : 1;
        var tmp = -GammaLn(v + 1.5) + (v + 1) * Math.Log(z / 2);
        double scaleexp;
        if (tmp < -600 || tmp > 600) { scaleexp = tmp / 2; tmp -= scaleexp; }
        else scaleexp = 0;
        var term = 2 / Math.Sqrt(Math.PI) * Math.Exp(tmp) * GammaSgn(v + 1.5);
        var sum = term;
        var maxterm = 0.0;
        var cterm = new SpecialDoubleDouble(term);
        var csum = new SpecialDoubleDouble(sum);
        var z2 = new SpecialDoubleDouble(sgn * z * z);
        var c2v = new SpecialDoubleDouble(2 * v);
        for (var n = 0; n < StruveMaxIter; n++)
        {
            var cdiv = new SpecialDoubleDouble(3 + 2 * n) * (new SpecialDoubleDouble(3 + 2 * n) + c2v);
            cterm = cterm * z2 / cdiv;
            csum += cterm;
            term = cterm.Value;
            sum = csum.Value;
            if (Math.Abs(term) > maxterm) maxterm = Math.Abs(term);
            if (Math.Abs(term) < StruveSumTiny * Math.Abs(sum) || term == 0 || !double.IsFinite(sum)) break;
        }
        err = Math.Abs(term) + Math.Abs(maxterm) * 1e-22;
        if (scaleexp != 0)
        {
            sum *= Math.Exp(scaleexp);
            err *= Math.Exp(scaleexp);
        }
        if (sum == 0 && term == 0 && v < 0 && !isH) { err = double.PositiveInfinity; return double.NaN; }
        return sum;
    }

    private static double StruveBesselSeries(double v, double z, bool isH, out double err)
    {
        if (isH && v < 0) { err = double.PositiveInfinity; return double.NaN; }
        double sum = 0, maxterm = 0, term = 0;
        var cterm = Math.Sqrt(z / (2 * Math.PI));
        for (var n = 0; n < StruveMaxIter; n++)
        {
            if (isH)
            {
                term = cterm * Jv(n + v + 0.5, z) / (n + 0.5);
                cterm *= z / 2 / (n + 1);
            }
            else
            {
                term = cterm * Iv(n + v + 0.5, z) / (n + 0.5);
                cterm *= -z / 2 / (n + 1);
            }
            sum += term;
            if (Math.Abs(term) > maxterm) maxterm = Math.Abs(term);
            if (Math.Abs(term) < StruveSumEps * Math.Abs(sum) || term == 0 || !double.IsFinite(sum)) break;
        }
        err = Math.Abs(term) + Math.Abs(maxterm) * 1e-16 + 1e-300 * Math.Abs(cterm);
        return sum;
    }
}
