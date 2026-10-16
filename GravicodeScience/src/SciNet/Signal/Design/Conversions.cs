using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// One or two critical frequencies of a filter (scipy's <c>Wn</c>): a scalar for low/high-pass,
/// a pair for band-pass/stop. Converts implicitly from <c>double</c>, <c>double[]</c> and a tuple,
/// and can be written as a collection expression: <c>Signal.Butter(4, [5.0, 50.0], FilterType.Bandpass)</c>.
/// </summary>
/// <param name="Values">The frequencies, one or two.</param>
[System.Runtime.CompilerServices.CollectionBuilder(typeof(CriticalFrequencies), nameof(Create))]
public readonly record struct CriticalFrequencies(double[] Values)
{
    /// <summary>Builds the value from a collection expression such as <c>[5.0, 50.0]</c>.</summary>
    public static CriticalFrequencies Create(ReadOnlySpan<double> values) => new(values.ToArray());

    /// <summary>Enumerates the frequencies (required for collection-expression support).</summary>
    public IEnumerator<double> GetEnumerator() => ((IEnumerable<double>)(Values ?? [])).GetEnumerator();

    /// <summary>A single cutoff.</summary>
    public static implicit operator CriticalFrequencies(double w) => new([w]);

    /// <summary>One or two cutoffs.</summary>
    public static implicit operator CriticalFrequencies(double[] w) => new(w);

    /// <summary>A band edge pair.</summary>
    public static implicit operator CriticalFrequencies((double Low, double High) w) => new([w.Low, w.High]);

    /// <summary>Number of frequencies.</summary>
    public int Length => Values?.Length ?? 0;
}

public static partial class Signal
{
    // ------------------------------------------------------------------ normalize / tf <-> zpk

    /// <summary>
    /// Normalises a transfer function so the leading denominator coefficient is one, as
    /// <c>scipy.signal.normalize</c>: leading zeros of <paramref name="a"/> are trimmed and leading
    /// numerator coefficients with magnitude ≤ 1e-14 are dropped (scipy warns "badly conditioned").
    /// </summary>
    public static FilterBA Normalize(ReadOnlySpan<double> b, ReadOnlySpan<double> a)
    {
        var den = TrimLeadingZeros(a);
        if (den.Length == 0) throw new ArgumentException("Denominator must have at least one nonzero element.", nameof(a));
        if (b.Length == 0) throw new ArgumentException("Numerator must not be empty.", nameof(b));
        var a0 = den[0];
        var outA = new double[den.Length];
        for (var i = 0; i < den.Length; i++) outA[i] = den[i] / a0;
        var num = new double[b.Length];
        for (var i = 0; i < b.Length; i++) num[i] = b[i] / a0;
        var lead = 0;
        while (lead < num.Length && Math.Abs(num[lead]) <= 1e-14) lead++;
        if (lead == num.Length) lead--;
        return new FilterBA(lead > 0 ? num[lead..] : num, outA);
    }

    /// <summary>Zeros, poles and gain of a transfer function (<c>scipy.signal.tf2zpk</c>).</summary>
    /// <remarks>Roots come from <see cref="Roots"/>; their order may differ from numpy's, so compare sorted.</remarks>
    public static FilterZpk Tf2Zpk(ReadOnlySpan<double> b, ReadOnlySpan<double> a)
    {
        var (nb, na) = Normalize(b, a);
        var k = nb[0];
        var bn = new double[nb.Length];
        for (var i = 0; i < nb.Length; i++) bn[i] = nb[i] / k;
        return new FilterZpk(Roots(bn), Roots(na), k);
    }

    /// <summary>Transfer-function coefficients from zeros, poles and gain (<c>scipy.signal.zpk2tf</c>).</summary>
    public static FilterBA Zpk2Tf(ReadOnlySpan<Complex> z, ReadOnlySpan<Complex> p, double k)
    {
        var b = Poly(z);
        for (var i = 0; i < b.Length; i++) b[i] *= k;
        return new FilterBA(b, Poly(p));
    }

    /// <inheritdoc cref="Zpk2Tf(ReadOnlySpan{Complex}, ReadOnlySpan{Complex}, double)"/>
    public static FilterBA Zpk2Tf(FilterZpk zpk)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        return Zpk2Tf(zpk.Z, zpk.P, zpk.K);
    }

    // ------------------------------------------------------------------ SOS

    /// <summary>Second-order sections from a transfer function (<c>scipy.signal.tf2sos</c>).</summary>
    public static NdArray Tf2Sos(ReadOnlySpan<double> b, ReadOnlySpan<double> a, SosPairing? pairing = null, bool analog = false)
        => Zpk2Sos(Tf2Zpk(b, a), pairing, analog);

    /// <summary>Transfer function of a cascade of second-order sections (<c>scipy.signal.sos2tf</c>).</summary>
    public static FilterBA Sos2Tf(NdArray sos)
    {
        var s = ValidateSos(sos, out var n);
        double[] b = [1.0], a = [1.0];
        for (var i = 0; i < n; i++)
        {
            b = Polymul(b, s.Slice(i * 6, 3));
            a = Polymul(a, s.Slice(i * 6 + 3, 3));
        }
        return new FilterBA(b, a);
    }

    /// <summary>
    /// Zeros, poles and gain of a cascade of second-order sections (<c>scipy.signal.sos2zpk</c>).
    /// Like scipy, each section contributes two zeros and two poles, padded with zeros at the origin.
    /// </summary>
    public static FilterZpk Sos2Zpk(NdArray sos)
    {
        var s = ValidateSos(sos, out var n);
        var z = new Complex[2 * n];
        var p = new Complex[2 * n];
        var k = 1.0;
        for (var i = 0; i < n; i++)
        {
            var zpk = Tf2Zpk(s.Slice(i * 6, 3), s.Slice(i * 6 + 3, 3));
            Array.Copy(zpk.Z, 0, z, 2 * i, zpk.Z.Length);
            Array.Copy(zpk.P, 0, p, 2 * i, zpk.P.Length);
            k *= zpk.K;
        }
        return new FilterZpk(z, p, k);
    }

    private static ReadOnlySpan<double> ValidateSos(NdArray sos, out int sections)
    {
        ArgumentNullException.ThrowIfNull(sos);
        if (sos.Rank != 2 || sos.Shape[1] != 6) throw new ArgumentException("sos array must be shape (n_sections, 6).", nameof(sos));
        sections = sos.Shape[0];
        var s = sos.AsContiguous().AsSpan();
        for (var i = 0; i < sections; i++)
            if (s[i * 6 + 3] != 1.0) throw new ArgumentException("sos[:, 3] should be all ones.", nameof(sos));
        return s;
    }

    /// <summary>
    /// Second-order sections from zeros, poles and gain (<c>scipy.signal.zpk2sos</c>), following
    /// scipy's pairing algorithm exactly: poles closest to the unit circle (the imaginary axis when
    /// analog) are placed in the <b>last</b> section, each paired with its nearest zero, and the
    /// gain goes into the first section.
    /// </summary>
    /// <param name="zpk">The filter.</param>
    /// <param name="pairing">Pairing rule; default <see cref="SosPairing.Nearest"/> (digital) or <see cref="SosPairing.Minimal"/> (analog).</param>
    /// <param name="analog">Whether the zpk describes an analog filter.</param>
    public static NdArray Zpk2Sos(FilterZpk zpk, SosPairing? pairing = null, bool analog = false)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var pair = pairing ?? (analog ? SosPairing.Minimal : SosPairing.Nearest);
        if (analog && pair != SosPairing.Minimal) throw new ArgumentException("For analog zpk2sos conversion, pairing must be Minimal.", nameof(pairing));
        var z = zpk.Z.ToList();
        var p = zpk.P.ToList();
        var k = zpk.K;
        if (z.Count == 0 && p.Count == 0)
            return analog ? new NdArray([0, 0, k, 0, 0, 1], 1, 6) : new NdArray([k, 0, 0, 1, 0, 0], 1, 6);

        int nSections;
        if (pair != SosPairing.Minimal)
        {
            while (p.Count < z.Count) p.Add(Complex.Zero);
            while (z.Count < p.Count) z.Add(Complex.Zero);
            nSections = (Math.Max(p.Count, z.Count) + 1) / 2;
            if (p.Count % 2 == 1 && pair == SosPairing.Nearest) { p.Add(Complex.Zero); z.Add(Complex.Zero); }
        }
        else
        {
            if (p.Count < z.Count) throw new ArgumentException("For analog zpk2sos conversion, must have len(p) >= len(z).");
            nSections = (p.Count + 1) / 2;
        }

        z = CplxRealConcat(z.ToArray());
        p = CplxRealConcat(p.ToArray());

        int IdxWorst(List<Complex> v)
        {
            var best = 0;
            var bv = double.PositiveInfinity;
            for (var i = 0; i < v.Count; i++)
            {
                var d = analog ? Math.Abs(v[i].Real) : Math.Abs(1 - Complex.Abs(v[i]));
                if (d < bv) { bv = d; best = i; }
            }
            return best;
        }
        static int CountReal(List<Complex> v)
        {
            var c = 0;
            foreach (var x in v) if (x.Imaginary == 0) c++;
            return c;
        }
        static int Nearest(List<Complex> fro, Complex to, int which) // 0 any, 1 real, 2 complex
        {
            var order = Enumerable.Range(0, fro.Count).OrderBy(i => Complex.Abs(fro[i] - to)).ToArray();
            foreach (var i in order)
            {
                var isReal = fro[i].Imaginary == 0;
                if (which == 0 || (which == 1 && isReal) || (which == 2 && !isReal)) return i;
            }
            throw new InvalidOperationException("No matching zero found.");
        }
        static Complex Take(List<Complex> v, int i)
        {
            var x = v[i];
            v.RemoveAt(i);
            return x;
        }

        var sos = new double[nSections * 6];
        void Single(int si, Complex[] zz, Complex[] pp)
        {
            var (b, a) = Zpk2Tf(zz, pp, 1.0);
            for (var i = 0; i < b.Length; i++) sos[si * 6 + 3 - b.Length + i] = b[i];
            for (var i = 0; i < a.Length; i++) sos[si * 6 + 6 - a.Length + i] = a[i];
        }

        for (var si = nSections - 1; si >= 0; si--)
        {
            var p1 = Take(p, IdxWorst(p));
            var p1Real = p1.Imaginary == 0;
            if (p1Real && CountReal(p) == 0)
            {
                if (pair != SosPairing.Minimal)
                {
                    var z1 = Take(z, Nearest(z, p1, 1));
                    Single(si, [z1, 0], [p1, 0]);
                }
                else if (z.Count > 0)
                {
                    var z1 = Take(z, Nearest(z, p1, 1));
                    Single(si, [z1], [p1]);
                }
                else Single(si, [], [p1]);
            }
            else if (p.Count + 1 == z.Count && !p1Real && CountReal(p) == 1 && CountReal(z) == 1)
            {
                var z1 = Take(z, Nearest(z, p1, 2));
                Single(si, [z1, Complex.Conjugate(z1)], [p1, Complex.Conjugate(p1)]);
            }
            else
            {
                Complex p2;
                if (p1Real)
                {
                    var realIdx = new List<int>();
                    for (var i = 0; i < p.Count; i++) if (p[i].Imaginary == 0) realIdx.Add(i);
                    var sub = realIdx.Select(i => p[i]).ToList();
                    p2 = Take(p, realIdx[IdxWorst(sub)]);
                }
                else p2 = Complex.Conjugate(p1);

                if (z.Count > 0)
                {
                    var z1 = Take(z, Nearest(z, p1, 0));
                    if (z1.Imaginary != 0) Single(si, [z1, Complex.Conjugate(z1)], [p1, p2]);
                    else if (z.Count > 0)
                    {
                        var z2 = Take(z, Nearest(z, p1, 1));
                        Single(si, [z1, z2], [p1, p2]);
                    }
                    else Single(si, [z1], [p1, p2]);
                }
                else Single(si, [], [p1, p2]);
            }
        }
        if (p.Count != 0 || z.Count != 0) throw new InvalidOperationException("zpk2sos pairing left unpaired roots.");
        for (var i = 0; i < 3; i++) sos[i] *= k;
        return new NdArray(sos, nSections, 6);
    }

    /// <summary>
    /// scipy's <c>_cplxreal</c>: complex values with positive imaginary part (one of each conjugate
    /// pair, averaged with its partner) followed by the real values, each group sorted by real part.
    /// </summary>
    private static List<Complex> CplxRealConcat(Complex[] z)
    {
        var (zc, zr) = CplxReal(z);
        var r = new List<Complex>(zc.Length + zr.Length);
        r.AddRange(zc);
        foreach (var v in zr) r.Add(new Complex(v, 0));
        return r;
    }

    internal static (Complex[] Complex, double[] Real) CplxReal(Complex[] zIn, double? tolerance = null)
    {
        if (zIn.Length == 0) return ([], []);
        var tol = tolerance ?? 100 * 2.220446049250313e-16;
        // lexsort((|imag|, real)): primary real, secondary |imag|; stable.
        var z = zIn.Select((v, i) => (v, i)).OrderBy(t => t.v.Real).ThenBy(t => Math.Abs(t.v.Imaginary)).ThenBy(t => t.i).Select(t => t.v).ToArray();
        var zr = new List<double>();
        var rest = new List<Complex>();
        foreach (var v in z)
        {
            if (Math.Abs(v.Imaginary) <= tol * Complex.Abs(v)) zr.Add(v.Real);
            else rest.Add(v);
        }
        if (rest.Count == 0) return ([], zr.ToArray());
        var zp = rest.Where(v => v.Imaginary > 0).ToArray();
        var zn = rest.Where(v => v.Imaginary < 0).ToArray();
        if (zp.Length != zn.Length) throw new ArgumentException("Array contains complex value with no matching conjugate.");

        // Within runs of equal real part, order by |imag| so conjugates line up.
        var start = 0;
        while (start < zp.Length)
        {
            var stop = start + 1;
            while (stop < zp.Length && zp[stop].Real - zp[stop - 1].Real <= tol * Complex.Abs(zp[stop - 1])) stop++;
            if (stop - start > 1)
            {
                Array.Sort(zp, start, stop - start, Comparer<Complex>.Create((x, y) => Math.Abs(x.Imaginary).CompareTo(Math.Abs(y.Imaginary))));
                Array.Sort(zn, start, stop - start, Comparer<Complex>.Create((x, y) => Math.Abs(x.Imaginary).CompareTo(Math.Abs(y.Imaginary))));
            }
            start = stop;
        }
        var zc = new Complex[zp.Length];
        for (var i = 0; i < zp.Length; i++)
        {
            if (Complex.Abs(zp[i] - Complex.Conjugate(zn[i])) > tol * Complex.Abs(zn[i]))
                throw new ArgumentException("Array contains complex value with no matching conjugate.");
            zc[i] = (zp[i] + Complex.Conjugate(zn[i])) / 2;
        }
        return (zc, zr.ToArray());
    }

    // ------------------------------------------------------------------ frequency transforms (zpk)

    private static int RelativeDegree(int nz, int np)
    {
        var d = np - nz;
        if (d < 0) throw new ArgumentException("Improper transfer function. Must have at least as many poles as zeros.");
        return d;
    }

    /// <summary>Complex square root following C99 <c>csqrt</c>, honouring the sign of a zero imaginary part as numpy does.</summary>
    private static Complex CSqrt(Complex z)
    {
        double x = z.Real, y = z.Imaginary;
        if (x == 0 && y == 0) return new Complex(0, y);
        var t = Math.Sqrt((Math.Abs(x) + Complex.Abs(z)) / 2);
        if (x >= 0) return new Complex(t, y / (2 * t));
        return new Complex(Math.Abs(y) / (2 * t), double.CopySign(t, y));
    }

    private static Complex Prod(ReadOnlySpan<Complex> v, Func<Complex, Complex> f)
    {
        var r = Complex.One;
        foreach (var x in v) r *= f(x);
        return r;
    }

    /// <summary>Scales a low-pass prototype to cutoff <paramref name="wo"/> (<c>lp2lp_zpk</c>).</summary>
    public static FilterZpk Lp2LpZpk(FilterZpk zpk, double wo = 1.0)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var degree = RelativeDegree(zpk.Z.Length, zpk.P.Length);
        return new FilterZpk(zpk.Z.Select(v => wo * v).ToArray(), zpk.P.Select(v => wo * v).ToArray(), zpk.K * Math.Pow(wo, degree));
    }

    /// <summary>Low-pass prototype to high-pass with cutoff <paramref name="wo"/> (<c>lp2hp_zpk</c>).</summary>
    public static FilterZpk Lp2HpZpk(FilterZpk zpk, double wo = 1.0)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var degree = RelativeDegree(zpk.Z.Length, zpk.P.Length);
        var z = new Complex[zpk.Z.Length + degree];
        for (var i = 0; i < zpk.Z.Length; i++) z[i] = wo / zpk.Z[i];
        var p = zpk.P.Select(v => wo / v).ToArray();
        var k = zpk.K * (Prod(zpk.Z, v => -v) / Prod(zpk.P, v => -v)).Real;
        return new FilterZpk(z, p, k);
    }

    /// <summary>Low-pass prototype to band-pass centred on <paramref name="wo"/> with width <paramref name="bw"/> (<c>lp2bp_zpk</c>).</summary>
    public static FilterZpk Lp2BpZpk(FilterZpk zpk, double wo = 1.0, double bw = 1.0)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var degree = RelativeDegree(zpk.Z.Length, zpk.P.Length);
        Complex[] Map(Complex[] r)
        {
            var o = new Complex[2 * r.Length];
            for (var i = 0; i < r.Length; i++)
            {
                var lp = r[i] * bw / 2;
                var s = CSqrt(lp * lp - wo * wo);
                o[i] = lp + s;
                o[i + r.Length] = lp - s;
            }
            return o;
        }
        var z = Map(zpk.Z);
        Array.Resize(ref z, z.Length + degree);
        return new FilterZpk(z, Map(zpk.P), zpk.K * Math.Pow(bw, degree));
    }

    /// <summary>Low-pass prototype to band-stop centred on <paramref name="wo"/> with width <paramref name="bw"/> (<c>lp2bs_zpk</c>).</summary>
    public static FilterZpk Lp2BsZpk(FilterZpk zpk, double wo = 1.0, double bw = 1.0)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var degree = RelativeDegree(zpk.Z.Length, zpk.P.Length);
        Complex[] Map(Complex[] r)
        {
            var o = new Complex[2 * r.Length];
            for (var i = 0; i < r.Length; i++)
            {
                var hp = bw / 2 / r[i];
                var s = CSqrt(hp * hp - wo * wo);
                o[i] = hp + s;
                o[i + r.Length] = hp - s;
            }
            return o;
        }
        var zm = Map(zpk.Z);
        var z = new Complex[zm.Length + 2 * degree];
        Array.Copy(zm, z, zm.Length);
        for (var i = 0; i < degree; i++)
        {
            z[zm.Length + i] = new Complex(0, wo);
            z[zm.Length + degree + i] = new Complex(0, -wo);
        }
        var k = zpk.K * (Prod(zpk.Z, v => -v) / Prod(zpk.P, v => -v)).Real;
        return new FilterZpk(z, Map(zpk.P), k);
    }

    /// <summary>
    /// Analog to digital by the bilinear transform <c>s = 2fs (z−1)/(z+1)</c> (<c>bilinear_zpk</c>);
    /// zeros at infinity map to Nyquist (z = −1).
    /// </summary>
    public static FilterZpk BilinearZpk(FilterZpk zpk, double fs)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        if (!(fs > 0) || double.IsInfinity(fs)) throw new ArgumentException("fs must be a positive finite number.", nameof(fs));
        var degree = RelativeDegree(zpk.Z.Length, zpk.P.Length);
        var fs2 = 2.0 * fs;
        var z = new Complex[zpk.Z.Length + degree];
        for (var i = 0; i < zpk.Z.Length; i++) z[i] = (fs2 + zpk.Z[i]) / (fs2 - zpk.Z[i]);
        for (var i = 0; i < degree; i++) z[zpk.Z.Length + i] = -1.0;
        var p = zpk.P.Select(v => (fs2 + v) / (fs2 - v)).ToArray();
        var k = zpk.K * (Prod(zpk.Z, v => fs2 - v) / Prod(zpk.P, v => fs2 - v)).Real;
        return new FilterZpk(z, p, k);
    }

    // ------------------------------------------------------------------ frequency transforms (ba)

    /// <summary>Scales a low-pass prototype transfer function to cutoff <paramref name="wo"/> (<c>lp2lp</c>).</summary>
    public static FilterBA Lp2Lp(ReadOnlySpan<double> b, ReadOnlySpan<double> a, double wo = 1.0)
    {
        var d = a.Length;
        var n = b.Length;
        var m = Math.Max(d, n);
        var pwo = new double[m];
        for (var i = 0; i < m; i++) pwo[i] = Math.Pow(wo, m - 1 - i);
        var start1 = Math.Max(n - d, 0);
        var start2 = Math.Max(d - n, 0);
        var nb = new double[n];
        for (var i = 0; i < n; i++) nb[i] = b[i] * pwo[start1] / pwo[start2 + i];
        var na = new double[d];
        for (var i = 0; i < d; i++) na[i] = a[i] * pwo[start1] / pwo[start1 + i];
        return Normalize(nb, na);
    }

    /// <summary>Low-pass prototype transfer function to high-pass (<c>lp2hp</c>).</summary>
    public static FilterBA Lp2Hp(ReadOnlySpan<double> b, ReadOnlySpan<double> a, double wo = 1.0)
    {
        var d = a.Length;
        var n = b.Length;
        var m = Math.Max(d, n);
        var pwo = new double[m];
        for (var i = 0; i < m; i++) pwo[i] = wo != 1 ? Math.Pow(wo, i) : 1.0;
        double[] outA, outB;
        if (d >= n)
        {
            outA = new double[d];
            for (var i = 0; i < d; i++) outA[i] = a[d - 1 - i] * pwo[i];
            outB = new double[d];
            for (var i = 0; i < n; i++) outB[i] = b[n - 1 - i] * pwo[i];
        }
        else
        {
            outB = new double[n];
            for (var i = 0; i < n; i++) outB[i] = b[n - 1 - i] * pwo[i];
            outA = new double[n];
            for (var i = 0; i < d; i++) outA[i] = a[d - 1 - i] * pwo[i];
        }
        return Normalize(outB, outA);
    }

    private static double Comb(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        double r = 1;
        for (var i = 1; i <= k; i++) r = r * (n - k + i) / i;
        return Math.Round(r);
    }

    /// <summary>Low-pass prototype transfer function to band-pass (<c>lp2bp</c>).</summary>
    public static FilterBA Lp2Bp(ReadOnlySpan<double> b, ReadOnlySpan<double> a, double wo = 1.0, double bw = 1.0)
    {
        var dd = a.Length - 1;
        var nn = b.Length - 1;
        var ma = Math.Max(nn, dd);
        var np = nn + ma;
        var dp = dd + ma;
        var wosq = wo * wo;
        var bprime = new double[np + 1];
        var aprime = new double[dp + 1];
        for (var j = 0; j <= np; j++)
        {
            var val = 0.0;
            for (var i = 0; i <= nn; i++)
                for (var k = 0; k <= i; k++)
                    if (ma - i + 2 * k == j) val += Comb(i, k) * b[nn - i] * Math.Pow(wosq, i - k) / Math.Pow(bw, i);
            bprime[np - j] = val;
        }
        for (var j = 0; j <= dp; j++)
        {
            var val = 0.0;
            for (var i = 0; i <= dd; i++)
                for (var k = 0; k <= i; k++)
                    if (ma - i + 2 * k == j) val += Comb(i, k) * a[dd - i] * Math.Pow(wosq, i - k) / Math.Pow(bw, i);
            aprime[dp - j] = val;
        }
        return Normalize(bprime, aprime);
    }

    /// <summary>Low-pass prototype transfer function to band-stop (<c>lp2bs</c>).</summary>
    public static FilterBA Lp2Bs(ReadOnlySpan<double> b, ReadOnlySpan<double> a, double wo = 1.0, double bw = 1.0)
    {
        var dd = a.Length - 1;
        var nn = b.Length - 1;
        var m = Math.Max(nn, dd);
        var np = 2 * m;
        var dp = 2 * m;
        var wosq = wo * wo;
        var bprime = new double[np + 1];
        var aprime = new double[dp + 1];
        for (var j = 0; j <= np; j++)
        {
            var val = 0.0;
            for (var i = 0; i <= nn; i++)
                for (var k = 0; k <= m - i; k++)
                    if (i + 2 * k == j) val += Comb(m - i, k) * b[nn - i] * Math.Pow(wosq, m - i - k) * Math.Pow(bw, i);
            bprime[np - j] = val;
        }
        for (var j = 0; j <= dp; j++)
        {
            var val = 0.0;
            for (var i = 0; i <= dd; i++)
                for (var k = 0; k <= m - i; k++)
                    if (i + 2 * k == j) val += Comb(m - i, k) * a[dd - i] * Math.Pow(wosq, m - i - k) * Math.Pow(bw, i);
            aprime[dp - j] = val;
        }
        return Normalize(bprime, aprime);
    }

    /// <summary>Bilinear transform of an analog transfer function to digital (<c>scipy.signal.bilinear</c>).</summary>
    public static FilterBA Bilinear(ReadOnlySpan<double> b, ReadOnlySpan<double> a, double fs = 1.0)
    {
        if (!(fs > 0) || double.IsInfinity(fs)) throw new ArgumentException("fs must be a positive finite number.", nameof(fs));
        b = TrimLeadingZeros(b);
        a = TrimLeadingZeros(a);
        var fac = Math.Sqrt(fs * 2);
        // Polynomials in z, lowest power first.
        double[] zp1 = [1 / fac, 1 / fac];
        double[] zm1 = [-fac, fac];
        var n = Math.Max(a.Length, b.Length) - 1;
        var powP = new double[n + 1][];
        var powM = new double[n + 1][];
        powP[0] = [1.0];
        powM[0] = [1.0];
        for (var i = 1; i <= n; i++) { powP[i] = MulLow(powP[i - 1], zp1); powM[i] = MulLow(powM[i - 1], zm1); }
        double[] Build(ReadOnlySpan<double> c)
        {
            var acc = new double[n + 1];
            for (var q = 0; q < c.Length; q++)
            {
                var coef = c[c.Length - 1 - q];
                var term = MulLow(powP[n - q], powM[q]);
                for (var i = 0; i < term.Length; i++) acc[i] += coef * term[i];
            }
            Array.Reverse(acc);
            return acc;
        }
        return Normalize(Build(b), Build(a));
    }

    private static double[] MulLow(double[] x, double[] y)
    {
        var r = new double[x.Length + y.Length - 1];
        for (var i = 0; i < x.Length; i++)
            for (var j = 0; j < y.Length; j++)
                r[i + j] += x[i] * y[j];
        return r;
    }
}
