using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>How <see cref="Signal.UniqueRoots"/> and the residue functions merge roots closer than the tolerance.</summary>
public enum ResidueRootType
{
    /// <summary>Smallest (numpy order: by real part, then imaginary).</summary>
    Min,
    /// <summary>Largest (numpy order: by real part, then imaginary).</summary>
    Max,
    /// <summary>Mean of the group (scipy's default for the residue functions).</summary>
    Avg,
}

public static partial class Signal
{
    /// <summary>
    /// Groups roots closer than <paramref name="tol"/> (Euclidean distance in the complex plane) into
    /// one root with a multiplicity — scipy's <c>unique_roots</c>. Groups are formed in input order,
    /// each from the not-yet-used roots within tol of its first member.
    /// </summary>
    public static (Complex[] Unique, int[] Multiplicity) UniqueRoots(Complex[] p, double tol = 1e-3, ResidueRootType rtype = ResidueRootType.Min)
    {
        ArgumentNullException.ThrowIfNull(p);
        var used = new bool[p.Length];
        var u = new List<Complex>(); var m = new List<int>();
        for (var i = 0; i < p.Length; i++)
        {
            if (used[i]) continue;
            var group = new List<Complex>();
            for (var j = 0; j < p.Length; j++)
                if (!used[j] && (p[j] - p[i]).Magnitude <= tol) { group.Add(p[j]); used[j] = true; }
            u.Add(Reduce(group, rtype));
            m.Add(group.Count);
        }
        return (u.ToArray(), m.ToArray());
    }

    /// <summary>
    /// Partial-fraction expansion of <c>b(s)/a(s)</c> (coefficients highest power first) —
    /// scipy's <c>residue</c>: <c>H(s) = Σ r_i/(s − p_i)^j + k(s)</c>, where a pole of multiplicity m
    /// contributes terms with j = 1..m in that order.
    /// </summary>
    /// <remarks>
    /// Poles closer than <paramref name="tol"/> are treated as one repeated pole (replaced by their
    /// <paramref name="rtype"/>-reduction), and poles are returned sorted by magnitude. Repeated
    /// roots are ill-conditioned: an m-fold pole is only known to about ε^(1/m), which is why the
    /// grouping exists. Among poles of equal magnitude (a conjugate pair) the one with positive
    /// imaginary part comes first, as LAPACK orders them for numpy; any other tie keeps root order,
    /// which can differ from numpy's.
    /// </remarks>
    /// <param name="b">Numerator coefficients.</param>
    /// <param name="a">Denominator coefficients.</param>
    /// <param name="tol">Distance below which poles are merged.</param>
    /// <param name="rtype">How merged poles are combined.</param>
    /// <returns>Residues, poles and the direct polynomial term k (highest power first).</returns>
    public static (Complex[] R, Complex[] P, double[] K) Residue(double[] b, double[] a, double tol = 1e-3, ResidueRootType rtype = ResidueRootType.Avg)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        b = TrimFront(b); a = TrimFront(a);
        if (a.Length == 0) throw new ArgumentException("Denominator `a` is zero.", nameof(a));
        var poles = Roots(a);
        if (b.Length == 0) return (new Complex[poles.Length], SortByMagnitude(poles).Sorted, []);
        double[] k = [];
        var num = b;
        if (b.Length >= a.Length)
        {
            var (q, r) = PolyDivReal(b, a);
            k = q; num = r;
        }
        var (unique, mult) = UniqueRoots(poles, tol, rtype);
        var (su, order) = SortByMagnitude(unique);
        var sm = order.Select(i => mult[i]).ToArray();
        var res = ComputeResidues(su, sm, num.Select(v => new Complex(v, 0)).ToArray());
        var idx = 0;
        for (var i = 0; i < su.Length; i++) for (var j = 0; j < sm[i]; j++) poles[idx++] = su[i];
        for (var i = 0; i < res.Length; i++) res[i] /= a[0];
        return (res, poles, k);
    }

    /// <summary>
    /// Partial-fraction expansion of <c>b(z)/a(z)</c> in powers of z⁻¹ (coefficients of z⁰, z⁻¹, …) —
    /// scipy's <c>residuez</c>: <c>H(z) = Σ r_i/(1 − p_i z⁻¹)^j + k(z⁻¹)</c>.
    /// </summary>
    /// <param name="b">Numerator coefficients in ascending powers of z⁻¹.</param>
    /// <param name="a">Denominator coefficients in ascending powers of z⁻¹; a[0] ≠ 0.</param>
    /// <param name="tol">Distance below which poles are merged.</param>
    /// <param name="rtype">How merged poles are combined.</param>
    public static (Complex[] R, Complex[] P, double[] K) Residuez(double[] b, double[] a, double tol = 1e-3, ResidueRootType rtype = ResidueRootType.Avg)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        b = TrimBack(b); a = TrimBack(a);
        if (a.Length == 0) throw new ArgumentException("Denominator `a` is zero.", nameof(a));
        if (a[0] == 0) throw new ArgumentException("First coefficient of determinant `a` must be non-zero.", nameof(a));
        var poles = Roots(a);
        if (b.Length == 0) return (new Complex[poles.Length], SortByMagnitude(poles).Sorted, []);
        var bRev = b.Reverse().ToArray();
        var aRev = a.Reverse().ToArray();
        double[] kRev = [];
        if (bRev.Length >= aRev.Length)
        {
            var (q, r) = PolyDivReal(bRev, aRev);
            kRev = q; bRev = r;
        }
        var (unique, mult) = UniqueRoots(poles, tol, rtype);
        var (su, order) = SortByMagnitude(unique);
        var sm = order.Select(i => mult[i]).ToArray();
        var inv = su.Select(v => 1 / v).ToArray();
        var res = ComputeResidues(inv, sm, bRev.Select(v => new Complex(v, 0)).ToArray());
        var idx = 0;
        var powers = new int[res.Length];
        for (var i = 0; i < su.Length; i++)
            for (var j = 0; j < sm[i]; j++) { poles[idx] = su[i]; powers[idx] = j + 1; idx++; }
        for (var i = 0; i < res.Length; i++) res[i] *= Complex.Pow(-poles[i], powers[i]) / aRev[0];
        return (res, poles, kRev.Reverse().ToArray());
    }

    /// <summary>
    /// Rebuilds <c>b(s)/a(s)</c> from a partial-fraction expansion — scipy's <c>invres</c>, the
    /// inverse of <see cref="Residue"/>. Consecutive poles within <paramref name="tol"/> of the first
    /// of their run are one repeated pole (the residues of a repeated pole are in increasing power).
    /// </summary>
    /// <returns>Numerator and denominator, highest power first (complex; imaginary parts vanish for conjugate-symmetric input).</returns>
    public static (Complex[] B, Complex[] A) Invres(Complex[] r, Complex[] p, double[] k, double tol = 1e-3, ResidueRootType rtype = ResidueRootType.Avg)
    {
        ArgumentNullException.ThrowIfNull(r);
        ArgumentNullException.ThrowIfNull(p);
        k = TrimFront(k ?? []);
        var (unique, mult) = GroupPoles(p, tol, rtype);
        var (factors, den) = ComputeFactors(unique, mult, includePowers: true);
        Complex[] numer = k.Length == 0 ? [Complex.Zero] : CMul(k.Select(v => new Complex(v, 0)).ToArray(), den);
        for (var i = 0; i < r.Length && i < factors.Count; i++) numer = CAdd(numer, CScale(factors[i], r[i]));
        return (numer, den);
    }

    /// <summary>
    /// Rebuilds <c>b(z)/a(z)</c> (ascending powers of z⁻¹) from a partial-fraction expansion —
    /// scipy's <c>invresz</c>, the inverse of <see cref="Residuez"/>.
    /// </summary>
    public static (Complex[] B, Complex[] A) Invresz(Complex[] r, Complex[] p, double[] k, double tol = 1e-3, ResidueRootType rtype = ResidueRootType.Avg)
    {
        ArgumentNullException.ThrowIfNull(r);
        ArgumentNullException.ThrowIfNull(p);
        k = TrimBack(k ?? []);
        var (unique, mult) = GroupPoles(p, tol, rtype);
        var (factors, den) = ComputeFactors(unique, mult, includePowers: true);
        Complex[] numer = k.Length == 0 ? [Complex.Zero]
            : CMul(k.Reverse().Select(v => new Complex(v, 0)).ToArray(), den.Reverse().ToArray());
        for (var i = 0; i < r.Length && i < factors.Count; i++) numer = CAdd(numer, CScale(factors[i].Reverse().ToArray(), r[i]));
        return (numer.Reverse().ToArray(), den);
    }

    // ------------------------------------------------------------------ helpers (numpy semantics)

    private static Complex Reduce(List<Complex> g, ResidueRootType rtype)
    {
        switch (rtype)
        {
            case ResidueRootType.Avg:
            {
                var s = Complex.Zero;
                foreach (var v in g) s += v;
                return s / g.Count;
            }
            default:
            {
                var best = g[0];
                foreach (var v in g)
                {
                    var c = v.Real != best.Real ? v.Real.CompareTo(best.Real) : v.Imaginary.CompareTo(best.Imaginary);
                    if (rtype == ResidueRootType.Max ? c > 0 : c < 0) best = v;
                }
                return best;
            }
        }
    }

    private static (Complex[] Unique, int[] Mult) GroupPoles(Complex[] poles, double tol, ResidueRootType rtype)
    {
        if (poles.Length == 0) return ([], []);
        var u = new List<Complex>(); var m = new List<int>();
        var pole = poles[0];
        var block = new List<Complex> { pole };
        for (var i = 1; i < poles.Length; i++)
        {
            // scipy appends the run's first pole, not poles[i], so a run reduces to its first member.
            if ((poles[i] - pole).Magnitude <= tol) block.Add(pole);
            else
            {
                u.Add(Reduce(block, rtype)); m.Add(block.Count);
                pole = poles[i];
                block = [pole];
            }
        }
        u.Add(Reduce(block, rtype)); m.Add(block.Count);
        return (u.ToArray(), m.ToArray());
    }

    private static (Complex[] Sorted, int[] Order) SortByMagnitude(Complex[] p)
    {
        var order = Enumerable.Range(0, p.Length).ToArray();
        // Stable by |p|; a conjugate pair (equal magnitude) puts +imag first, as LAPACK returns them.
        Array.Sort(order, (i, j) =>
        {
            var c = Complex.Abs(p[i]).CompareTo(Complex.Abs(p[j]));
            if (c != 0) return c;
            if (p[i].Real == p[j].Real && p[i].Imaginary == -p[j].Imaginary && p[i].Imaginary != 0) return p[j].Imaginary.CompareTo(p[i].Imaginary);
            return i.CompareTo(j);
        });
        return (order.Select(i => p[i]).ToArray(), order);
    }

    private static (List<Complex[]> Factors, Complex[] Denominator) ComputeFactors(Complex[] roots, int[] mult, bool includePowers)
    {
        Complex[] current = [Complex.One];
        var suffixes = new List<Complex[]> { current };
        for (var i = roots.Length - 1; i >= 1; i--)
        {
            Complex[] mono = [Complex.One, -roots[i]];
            for (var j = 0; j < mult[i]; j++) current = CMul(current, mono);
            suffixes.Add(current);
        }
        suffixes.Reverse();
        var factors = new List<Complex[]>();
        current = [Complex.One];
        for (var i = 0; i < roots.Length; i++)
        {
            Complex[] mono = [Complex.One, -roots[i]];
            var block = new List<Complex[]>();
            for (var j = 0; j < mult[i]; j++)
            {
                if (j == 0 || includePowers) block.Add(CMul(current, suffixes[i]));
                current = CMul(current, mono);
            }
            block.Reverse();
            factors.AddRange(block);
        }
        return (factors, current);
    }

    private static Complex[] ComputeResidues(Complex[] poles, int[] mult, Complex[] numerator)
    {
        var (factors, _) = ComputeFactors(poles, mult, includePowers: false);
        var res = new List<Complex>();
        for (var i = 0; i < poles.Length; i++)
        {
            var pole = poles[i];
            var factor = factors[i];
            if (mult[i] == 1)
            {
                res.Add(CPolyval(numerator, pole) / CPolyval(factor, pole));
                continue;
            }
            var numer = (Complex[])numerator.Clone();
            Complex[] mono = [Complex.One, -pole];
            var (f2, d) = CPolyDiv(factor, mono);
            var block = new List<Complex>();
            for (var j = 0; j < mult[i]; j++)
            {
                var (q, n) = CPolyDiv(numer, mono);
                var r = n[0] / d[0];
                numer = CSub(q, CScale(f2, r));
                block.Add(r);
            }
            block.Reverse();
            res.AddRange(block);
        }
        return res.ToArray();
    }

    private static double[] TrimFront(double[] a) { var i = 0; while (i < a.Length && a[i] == 0) i++; return a[i..]; }

    private static double[] TrimBack(double[] a) { var e = a.Length; while (e > 0 && a[e - 1] == 0) e--; return a[..e]; }

    private static Complex CPolyval(Complex[] p, Complex x)
    {
        var y = Complex.Zero;
        foreach (var c in p) y = y * x + c;
        return y;
    }

    private static Complex[] CMul(Complex[] a, Complex[] b)
    {
        var r = new Complex[a.Length + b.Length - 1];
        for (var i = 0; i < a.Length; i++) for (var j = 0; j < b.Length; j++) r[i + j] += a[i] * b[j];
        return r;
    }

    private static Complex[] CAdd(Complex[] a, Complex[] b)
    {
        var n = Math.Max(a.Length, b.Length);
        var r = new Complex[n];
        for (var i = 0; i < a.Length; i++) r[n - a.Length + i] += a[i];
        for (var i = 0; i < b.Length; i++) r[n - b.Length + i] += b[i];
        return r;
    }

    private static Complex[] CSub(Complex[] a, Complex[] b) => CAdd(a, CScale(b, -1));

    private static Complex[] CScale(Complex[] a, Complex s) => a.Select(v => v * s).ToArray();

    /// <summary>numpy's <c>polydiv</c>, including its trap: leading remainder terms with
    /// |r₀| ≤ 1e-8 (allclose's default atol) are dropped.</summary>
    private static (Complex[] Q, Complex[] R) CPolyDiv(Complex[] u, Complex[] v)
    {
        var m = u.Length - 1; var n = v.Length - 1;
        var scale = 1 / v[0];
        var q = new Complex[Math.Max(m - n + 1, 1)];
        var r = (Complex[])u.Clone();
        for (var k = 0; k < m - n + 1; k++)
        {
            var d = scale * r[k];
            q[k] = d;
            for (var j = 0; j <= n; j++) r[k + j] -= d * v[j];
        }
        var s = 0;
        while (r.Length - s > 1 && r[s].Magnitude <= 1e-8) s++;
        return (q, r[s..]);
    }

    private static (double[] Q, double[] R) PolyDivReal(double[] u, double[] v)
    {
        var m = u.Length - 1; var n = v.Length - 1;
        var scale = 1 / v[0];
        var q = new double[Math.Max(m - n + 1, 1)];
        var r = (double[])u.Clone();
        for (var k = 0; k < m - n + 1; k++)
        {
            var d = scale * r[k];
            q[k] = d;
            for (var j = 0; j <= n; j++) r[k + j] -= d * v[j];
        }
        var s = 0;
        while (r.Length - s > 1 && Math.Abs(r[s]) <= 1e-8) s++;
        return (q, r[s..]);
    }
}
