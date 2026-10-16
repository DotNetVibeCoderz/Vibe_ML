using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Small span-based linear solvers private to the integrators. The implicit ODE methods factor
/// an n x n matrix many thousands of times for small n, where an allocation per factorisation
/// (as GraviNum's Decomposition does) would dominate; so these work in caller-owned buffers.
/// A zero pivot is not an exception here: like LAPACK's dgetrf it is reported and the factors
/// are left as they are, so a Newton iteration sees non-finite values and shrinks the step,
/// which is exactly what scipy's solvers do on a singular iteration matrix.
/// </summary>
internal static class IntegrateLinear
{
    /// <summary>In-place LU with partial pivoting of a row-major n x n matrix. Returns false on a zero pivot.</summary>
    internal static bool LuFactor(Span<double> a, int n, Span<int> piv)
    {
        var ok = true;
        for (var k = 0; k < n; k++)
        {
            var p = k;
            var max = Math.Abs(a[k * n + k]);
            for (var i = k + 1; i < n; i++)
            {
                var v = Math.Abs(a[i * n + k]);
                if (v > max) { max = v; p = i; }
            }
            piv[k] = p;
            if (p != k)
            {
                var rk = a.Slice(k * n, n);
                var rp = a.Slice(p * n, n);
                for (var j = 0; j < n; j++) (rk[j], rp[j]) = (rp[j], rk[j]);
            }
            var d = a[k * n + k];
            if (d == 0.0) { ok = false; continue; }
            var inv = 1.0 / d;
            var rowK = a.Slice(k * n, n);
            for (var i = k + 1; i < n; i++)
            {
                var row = a.Slice(i * n, n);
                var l = row[k] * inv;
                row[k] = l;
                if (l == 0.0) continue;
                for (var j = k + 1; j < n; j++) row[j] -= l * rowK[j];
            }
        }
        return ok;
    }

    /// <summary>Solves LU x = b in place using factors from LuFactor.</summary>
    internal static void LuSolve(ReadOnlySpan<double> lu, int n, ReadOnlySpan<int> piv, Span<double> b)
    {
        for (var k = 0; k < n; k++)
        {
            var p = piv[k];
            if (p != k) (b[k], b[p]) = (b[p], b[k]);
        }
        for (var i = 1; i < n; i++)
        {
            var s = b[i];
            var row = lu.Slice(i * n, i);
            for (var j = 0; j < i; j++) s -= row[j] * b[j];
            b[i] = s;
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var j = i + 1; j < n; j++) s -= lu[i * n + j] * b[j];
            b[i] = s / lu[i * n + i];
        }
    }

    /// <summary>Complex LU with partial pivoting (pivot chosen on |re| + |im|, as LAPACK's zgetrf does via cabs1).</summary>
    internal static bool LuFactor(Span<Complex> a, int n, Span<int> piv)
    {
        var ok = true;
        for (var k = 0; k < n; k++)
        {
            var p = k;
            var max = Abs1(a[k * n + k]);
            for (var i = k + 1; i < n; i++)
            {
                var v = Abs1(a[i * n + k]);
                if (v > max) { max = v; p = i; }
            }
            piv[k] = p;
            if (p != k)
                for (var j = 0; j < n; j++) (a[k * n + j], a[p * n + j]) = (a[p * n + j], a[k * n + j]);
            var d = a[k * n + k];
            if (d == Complex.Zero) { ok = false; continue; }
            var inv = 1.0 / d;
            for (var i = k + 1; i < n; i++)
            {
                var l = a[i * n + k] * inv;
                a[i * n + k] = l;
                if (l == Complex.Zero) continue;
                for (var j = k + 1; j < n; j++) a[i * n + j] -= l * a[k * n + j];
            }
        }
        return ok;
    }

    /// <summary>Complex counterpart of <see cref="LuSolve(ReadOnlySpan{double}, int, ReadOnlySpan{int}, Span{double})"/>.</summary>
    internal static void LuSolve(ReadOnlySpan<Complex> lu, int n, ReadOnlySpan<int> piv, Span<Complex> b)
    {
        for (var k = 0; k < n; k++)
        {
            var p = piv[k];
            if (p != k) (b[k], b[p]) = (b[p], b[k]);
        }
        for (var i = 1; i < n; i++)
        {
            var s = b[i];
            for (var j = 0; j < i; j++) s -= lu[i * n + j] * b[j];
            b[i] = s;
        }
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var j = i + 1; j < n; j++) s -= lu[i * n + j] * b[j];
            b[i] = s / lu[i * n + i];
        }
    }

    private static double Abs1(Complex z) => Math.Abs(z.Real) + Math.Abs(z.Imaginary);

    /// <summary>Dense solve with a throwaway copy; for small one-off systems (Newton-Cotes, tridiagonal moments).</summary>
    internal static double[] SolveDense(double[] a, double[] b, int n)
    {
        var lu = (double[])a.Clone();
        var piv = new int[n];
        var x = (double[])b.Clone();
        LuFactor(lu, n, piv);
        LuSolve(lu, n, piv, x);
        return x;
    }
}

/// <summary>
/// Banded LU with partial pivoting, the dgbtrf/dgbtrs scheme on row storage. Each row keeps a
/// window of 2kl + ku + 1 columns starting at <c>row - kl</c>, wide enough to absorb the fill-in
/// that row interchanges create. Multipliers stay at the position where they were computed and
/// the solve replays interchanges and eliminations column by column, which is what makes the
/// position-indexed storage correct (moving them with later swaps would misattribute them).
/// </summary>
internal sealed class IntegrateBandedLu
{
    private readonly int _n, _kl, _ku, _w;
    private readonly double[] _a;
    private readonly int[] _piv;

    internal IntegrateBandedLu(int n, int kl, int ku)
    {
        _n = n; _kl = kl; _ku = ku; _w = 2 * kl + ku + 1;
        _a = new double[(long)n * _w > int.MaxValue ? throw new ArgumentException("Banded system too large.") : n * _w];
        _piv = new int[n];
    }

    internal int Size => _n;

    /// <summary>Clears the matrix before reassembly.</summary>
    internal void Clear() => Array.Clear(_a);

    /// <summary>Adds <paramref name="v"/> at (r, c); the entry must lie in the original band.</summary>
    internal void Add(int r, int c, double v)
    {
        var off = c - (r - _kl);
        if (off < 0 || off > _kl + _ku) throw new InvalidOperationException($"Entry ({r},{c}) outside the band.");
        _a[r * _w + off] += v;
    }

    private ref double At(int r, int c) => ref _a[r * _w + (c - r + _kl)];

    /// <summary>Factorises in place. Returns false when a pivot is exactly zero (singular).</summary>
    internal bool Factor()
    {
        var n = _n;
        var reach = _kl + _ku;
        for (var j = 0; j < n; j++)
        {
            var last = Math.Min(n - 1, j + _kl);
            var p = j;
            var max = Math.Abs(At(j, j));
            for (var r = j + 1; r <= last; r++)
            {
                var v = Math.Abs(At(r, j));
                if (v > max) { max = v; p = r; }
            }
            _piv[j] = p;
            if (max == 0.0) return false;
            var cEnd = Math.Min(n - 1, j + reach);
            if (p != j)
                for (var c = j; c <= cEnd; c++)
                {
                    ref var x = ref At(j, c);
                    ref var y = ref At(p, c);
                    (x, y) = (y, x);
                }
            var inv = 1.0 / At(j, j);
            for (var r = j + 1; r <= last; r++)
            {
                ref var lr = ref At(r, j);
                if (lr == 0.0) continue;
                var l = lr * inv;
                lr = l;
                var rowR = r * _w - r + _kl;
                var rowJ = j * _w - j + _kl;
                for (var c = j + 1; c <= cEnd; c++) _a[rowR + c] -= l * _a[rowJ + c];
            }
        }
        return true;
    }

    /// <summary>Solves in place with the factors.</summary>
    internal void Solve(Span<double> b)
    {
        var n = _n;
        for (var j = 0; j < n; j++)
        {
            var p = _piv[j];
            if (p != j) (b[j], b[p]) = (b[p], b[j]);
            var bj = b[j];
            if (bj == 0.0) continue;
            var last = Math.Min(n - 1, j + _kl);
            for (var r = j + 1; r <= last; r++) b[r] -= At(r, j) * bj;
        }
        var reach = _kl + _ku;
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            var cEnd = Math.Min(n - 1, i + reach);
            var row = i * _w - i + _kl;
            for (var c = i + 1; c <= cEnd; c++) s -= _a[row + c] * b[c];
            b[i] = s / _a[row + i];
        }
    }
}
