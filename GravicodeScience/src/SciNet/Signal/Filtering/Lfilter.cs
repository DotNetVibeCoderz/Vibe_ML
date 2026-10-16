using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    // ------------------------------------------------------------------ lfilter

    /// <summary>
    /// The allocation-free kernel of <c>lfilter</c>: Direct Form II transposed, exactly scipy's
    /// recurrence, normalising by <c>a[0]</c>. <paramref name="z"/> holds the
    /// <c>max(len a, len b) − 1</c> delay states on entry (initial conditions) and on exit (final
    /// conditions), so consecutive blocks of a stream can be filtered by repeated calls.
    /// </summary>
    /// <param name="b">Numerator.</param>
    /// <param name="a">Denominator; <c>a[0]</c> must be non-zero.</param>
    /// <param name="x">Input.</param>
    /// <param name="y">Output, same length as <paramref name="x"/> (may alias it).</param>
    /// <param name="z">Delay states, length <c>max(len a, len b) − 1</c>.</param>
    public static void Lfilter(ReadOnlySpan<double> b, ReadOnlySpan<double> a, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        if (b.Length == 0) throw new ArgumentException("Parameter b must be a non-empty array.", nameof(b));
        if (a.Length == 0) throw new ArgumentException("Parameter a must be a non-empty array.", nameof(a));
        if (a[0] == 0) throw new ArgumentException("First `a` filter coefficient must be non-zero.", nameof(a));
        if (y.Length != x.Length) throw new ArgumentException("y must have the same length as x.", nameof(y));
        var n = Math.Max(a.Length, b.Length);
        if (z.Length != n - 1) throw new ArgumentException($"z must have length {n - 1}.", nameof(z));

        // Normalised, zero-padded coefficients. Small orders live on the stack.
        Span<double> bn = n <= 64 ? stackalloc double[n] : new double[n];
        Span<double> an = n <= 64 ? stackalloc double[n] : new double[n];
        bn.Clear(); an.Clear();
        var a0 = a[0];
        for (var i = 0; i < b.Length; i++) bn[i] = b[i] / a0;
        for (var i = 0; i < a.Length; i++) an[i] = a[i] / a0;

        if (a.Length == 1 && n >= 8 && !x.Overlaps(y))
        {
            FirDirect(bn[..b.Length], x, y, z);
            return;
        }
        if (n == 1)
        {
            var g = bn[0];
            for (var i = 0; i < x.Length; i++) y[i] = g * x[i];
            return;
        }
        if (n == 2)
        {
            double b0 = bn[0], b1 = bn[1], a1 = an[1], z0 = z[0];
            for (var i = 0; i < x.Length; i++)
            {
                var xi = x[i];
                var yi = z0 + b0 * xi;
                z0 = xi * b1 - yi * a1;
                y[i] = yi;
            }
            z[0] = z0;
            return;
        }
        if (n == 3)
        {
            double b0 = bn[0], b1 = bn[1], b2 = bn[2], a1 = an[1], a2 = an[2], z0 = z[0], z1 = z[1];
            for (var i = 0; i < x.Length; i++)
            {
                var xi = x[i];
                var yi = z0 + b0 * xi;
                z0 = z1 + xi * b1 - yi * a1;
                z1 = xi * b2 - yi * a2;
                y[i] = yi;
            }
            z[0] = z0; z[1] = z1;
            return;
        }
        Df2tLoop(bn, an, x, y, z);
    }

    /// <summary>The general DF2T recurrence on normalised, equal-length coefficients — the scalar reference every fast path is tested against.</summary>
    internal static void Df2tLoop(ReadOnlySpan<double> bn, ReadOnlySpan<double> an, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        var last = bn.Length - 2;
        for (var i = 0; i < x.Length; i++)
        {
            var xi = x[i];
            var yi = z[0] + bn[0] * xi;
            for (var k = 1; k <= last; k++) z[k - 1] = z[k] + xi * bn[k] - yi * an[k];
            z[last] = xi * bn[last + 1] - yi * an[last + 1];
            y[i] = yi;
        }
    }

    /// <summary>
    /// FIR filtering as a direct convolution with SIMD dot products over the reversed taps, plus
    /// the DF2T initial state (which, for an FIR, adds z[i] to output i) and the final state.
    /// The recurrence form serialises on the state update; the convolution does not.
    /// </summary>
    private static void FirDirect(ReadOnlySpan<double> b, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        var nt = b.Length;
        var len = x.Length;
        var br = new double[nt];
        for (var k = 0; k < nt; k++) br[k] = b[nt - 1 - k];
        var brs = (ReadOnlySpan<double>)br;
        // Head: windows that run off the start of x.
        var head = Math.Min(nt - 1, len);
        for (var i = 0; i < head; i++)
        {
            var s = 0.0;
            for (var k = 0; k <= i; k++) s += b[k] * x[i - k];
            y[i] = s + z[i];
        }
        var vw = System.Numerics.Vector<double>.Count;
        for (var i = nt - 1; i < len; i++)
        {
            var win = x.Slice(i - nt + 1, nt);
            var acc = System.Numerics.Vector<double>.Zero;
            var k = 0;
            for (; k <= nt - vw; k += vw)
                acc += new System.Numerics.Vector<double>(brs.Slice(k, vw)) * new System.Numerics.Vector<double>(win.Slice(k, vw));
            var s = System.Numerics.Vector.Sum(acc);
            for (; k < nt; k++) s += brs[k] * win[k];
            y[i] = s;
        }
        // Final state: zf[j] = Σ_{k>j} b[k] x[len-1-(k-j-1)], plus initial state still in flight when len is short.
        Span<double> zf = nt <= 64 ? stackalloc double[nt - 1] : new double[nt - 1];
        for (var j = 0; j < nt - 1; j++)
        {
            var s = j + len < nt - 1 ? z[j + len] : 0.0;
            for (var k = j + 1; k < nt; k++)
            {
                var idx = len - 1 - (k - j - 1);
                if (idx >= 0) s += b[k] * x[idx];
            }
            zf[j] = s;
        }
        zf.CopyTo(z);
    }

    /// <summary>
    /// Filters <paramref name="x"/> with the IIR/FIR filter <c>b/a</c> from rest
    /// (<c>scipy.signal.lfilter</c>).
    /// </summary>
    public static double[] Lfilter(double[] b, double[] a, double[] x)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(x);
        var y = new double[x.Length];
        var z = new double[Math.Max(a.Length, b.Length) - 1];
        Lfilter(b, a, x, y, z);
        return y;
    }

    /// <summary>Filters with initial conditions <paramref name="zi"/>; returns the output and the final conditions.</summary>
    public static (double[] Y, double[] Zf) Lfilter(double[] b, double[] a, double[] x, double[] zi)
    {
        ArgumentNullException.ThrowIfNull(zi);
        var y = new double[x.Length];
        var z = zi.ToArray();
        Lfilter(b, a, x, y, z);
        return (y, z);
    }

    /// <summary>
    /// Filters every 1-D line of <paramref name="x"/> along <paramref name="axis"/> (default last).
    /// Lines are independent, so large inputs run in parallel.
    /// </summary>
    public static NdArray Lfilter(double[] b, double[] a, NdArray x, int axis = -1)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        var nz = Math.Max(a.Length, b.Length) - 1;
        return ApplyAlongAxis(x, axis, (src, dst) =>
        {
            Span<double> z = nz <= 64 ? stackalloc double[nz] : new double[nz];
            z.Clear();
            Lfilter(b, a, src, dst, z);
        });
    }

    /// <summary>
    /// Steady-state initial conditions for a step response (<c>scipy.signal.lfilter_zi</c>):
    /// multiply by the first input sample to start a filter as if that value had been held forever.
    /// </summary>
    public static double[] LfilterZi(double[] b, double[] a)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        var ai = 0;
        while (a.Length - ai > 1 && a[ai] == 0.0) ai++;
        if (a.Length - ai < 1) throw new ArgumentException("There must be at least one nonzero `a` coefficient.");
        var aa = a[ai..].ToArray();
        var bb = b.ToArray();
        if (aa[0] != 1.0)
        {
            var a0 = aa[0];
            for (var i = 0; i < bb.Length; i++) bb[i] /= a0;
            for (var i = 0; i < aa.Length; i++) aa[i] /= a0;
        }
        var n = Math.Max(aa.Length, bb.Length);
        Array.Resize(ref aa, n);
        Array.Resize(ref bb, n);
        var m = n - 1;
        if (m == 0) return [];
        // (I - companion(a)^T) zi = b[1:] - a[1:] b[0]; companion's first row is -a[1:].
        var mat = new double[m * m];
        for (var i = 0; i < m; i++)
        {
            mat[i * m + i] += 1.0;
            mat[i * m + 0] -= -aa[i + 1];          // C^T[i,0] = C[0,i] = -a[i+1]
            if (i + 1 < m) mat[i * m + i + 1] -= 1.0; // C^T[i,i+1] = C[i+1,i] = 1
        }
        var rhs = new double[m];
        for (var i = 0; i < m; i++) rhs[i] = bb[i + 1] - aa[i + 1] * bb[0];
        return SolveDense(mat, rhs, m);
    }

    /// <summary>Gaussian elimination with partial pivoting for small dense systems.</summary>
    internal static double[] SolveDense(double[] aIn, double[] bIn, int n)
    {
        var a = aIn.ToArray();
        var b = bIn.ToArray();
        for (var k = 0; k < n; k++)
        {
            var p = k;
            for (var i = k + 1; i < n; i++) if (Math.Abs(a[i * n + k]) > Math.Abs(a[p * n + k])) p = i;
            if (a[p * n + k] == 0) throw new ArgumentException("Singular matrix.");
            if (p != k)
            {
                for (var j = 0; j < n; j++) (a[k * n + j], a[p * n + j]) = (a[p * n + j], a[k * n + j]);
                (b[k], b[p]) = (b[p], b[k]);
            }
            for (var i = k + 1; i < n; i++)
            {
                var f = a[i * n + k] / a[k * n + k];
                if (f == 0) continue;
                for (var j = k; j < n; j++) a[i * n + j] -= f * a[k * n + j];
                b[i] -= f * b[k];
            }
        }
        var x = new double[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var j = i + 1; j < n; j++) s -= a[i * n + j] * x[j];
            x[i] = s / a[i * n + i];
        }
        return x;
    }

    /// <summary>
    /// Initial conditions that reproduce given past outputs (and inputs) (<c>scipy.signal.lfiltic</c>).
    /// </summary>
    /// <param name="b">Numerator.</param>
    /// <param name="a">Denominator.</param>
    /// <param name="y">Past outputs, most recent first: <c>y[−1], y[−2], …</c>.</param>
    /// <param name="x">Past inputs, most recent first (default zeros).</param>
    public static double[] Lfiltic(double[] b, double[] a, double[] y, double[]? x = null)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(y);
        var n = a.Length - 1;
        var m = b.Length - 1;
        if (n < 0) throw new ArgumentException("There must be at least one `a` coefficient.");
        var k = Math.Max(m, n);
        var xx = new double[Math.Max(m, x?.Length ?? 0)];
        if (x is not null) Array.Copy(x, xx, x.Length);
        var yy = new double[Math.Max(n, y.Length)];
        Array.Copy(y, yy, y.Length);
        var zi = new double[k];
        for (var i = 0; i < m; i++)
        {
            var s = 0.0;
            for (var j = 0; j < m - i; j++) s += b[i + 1 + j] * xx[j];
            zi[i] = s;
        }
        for (var i = 0; i < n; i++)
        {
            var s = 0.0;
            for (var j = 0; j < n - i; j++) s += a[i + 1 + j] * yy[j];
            zi[i] -= s;
        }
        if (a[0] != 1.0)
        {
            if (a[0] == 0.0) throw new ArgumentException("First `a` filter coefficient must be non-zero.");
            for (var i = 0; i < k; i++) zi[i] /= a[0];
        }
        return zi;
    }

    // ------------------------------------------------------------------ sosfilt

    /// <summary>
    /// The allocation-free kernel of <c>sosfilt</c>: a cascade of DF2T biquads, sample by sample
    /// through all sections as scipy's C loop does. <paramref name="sos"/> is the flattened
    /// <c>n × 6</c> array (each <c>a0</c> must be 1); <paramref name="zi"/> holds <c>n × 2</c> states
    /// in and out.
    /// </summary>
    public static void Sosfilt(ReadOnlySpan<double> sos, ReadOnlySpan<double> x, Span<double> y, Span<double> zi)
    {
        if (sos.Length % 6 != 0) throw new ArgumentException("sos must have 6 coefficients per section.", nameof(sos));
        var ns = sos.Length / 6;
        if (zi.Length != 2 * ns) throw new ArgumentException($"zi must have length {2 * ns}.", nameof(zi));
        if (y.Length != x.Length) throw new ArgumentException("y must have the same length as x.", nameof(y));
        // In-place use (y aliasing x exactly) is safe: each stage reads x[i] before writing y[i].
        // A partial overlap is not, so it takes the reference path.
        if (x.Overlaps(y, out var offset) && offset != 0)
        {
            SosfiltReference(sos, x, y, zi);
            return;
        }
        // A biquad is latency-bound: each output waits on the previous state update. Running up
        // to four sections sample-major with their states in locals lets the CPU overlap section
        // s+1 at sample i with section s at sample i+1 (scipy's order gets the same ILP but keeps
        // state in memory); groups are chained through y over cache-sized blocks. The arithmetic
        // per section is unchanged, so results are bit-identical to the reference.
        const int block = 4096;
        for (var start = 0; start < x.Length; start += block)
        {
            var len = Math.Min(block, x.Length - start);
            var src = x.Slice(start, len);
            var dst = y.Slice(start, len);
            var s0 = 0;
            while (s0 < ns)
            {
                var g = ns - s0 >= 4 ? 4 : ns - s0 >= 2 ? 2 : 1;
                if (g == 4) SosGroup4(sos.Slice(s0 * 6, 24), src, dst, zi.Slice(2 * s0, 8));
                else if (g == 2) SosGroup2(sos.Slice(s0 * 6, 12), src, dst, zi.Slice(2 * s0, 4));
                else SosGroup1(sos.Slice(s0 * 6, 6), src, dst, zi.Slice(2 * s0, 2));
                src = dst;
                s0 += g;
            }
        }
    }

    private static void SosGroup1(ReadOnlySpan<double> c, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        double b0 = c[0], b1 = c[1], b2 = c[2], a1 = c[4], a2 = c[5], z0 = z[0], z1 = z[1];
        for (var i = 0; i < x.Length; i++)
        {
            var xc = x[i];
            var xn = b0 * xc + z0;
            z0 = b1 * xc - a1 * xn + z1;
            z1 = b2 * xc - a2 * xn;
            y[i] = xn;
        }
        z[0] = z0; z[1] = z1;
    }

    private static void SosGroup2(ReadOnlySpan<double> c, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        double b00 = c[0], b01 = c[1], b02 = c[2], a01 = c[4], a02 = c[5];
        double b10 = c[6], b11 = c[7], b12 = c[8], a11 = c[10], a12 = c[11];
        double z00 = z[0], z01 = z[1], z10 = z[2], z11 = z[3];
        for (var i = 0; i < x.Length; i++)
        {
            var x0 = x[i];
            var x1 = b00 * x0 + z00;
            z00 = b01 * x0 - a01 * x1 + z01;
            z01 = b02 * x0 - a02 * x1;
            var x2 = b10 * x1 + z10;
            z10 = b11 * x1 - a11 * x2 + z11;
            z11 = b12 * x1 - a12 * x2;
            y[i] = x2;
        }
        z[0] = z00; z[1] = z01; z[2] = z10; z[3] = z11;
    }

    private static void SosGroup4(ReadOnlySpan<double> c, ReadOnlySpan<double> x, Span<double> y, Span<double> z)
    {
        double b00 = c[0], b01 = c[1], b02 = c[2], a01 = c[4], a02 = c[5];
        double b10 = c[6], b11 = c[7], b12 = c[8], a11 = c[10], a12 = c[11];
        double b20 = c[12], b21 = c[13], b22 = c[14], a21 = c[16], a22 = c[17];
        double b30 = c[18], b31 = c[19], b32 = c[20], a31 = c[22], a32 = c[23];
        double z00 = z[0], z01 = z[1], z10 = z[2], z11 = z[3], z20 = z[4], z21 = z[5], z30 = z[6], z31 = z[7];
        for (var i = 0; i < x.Length; i++)
        {
            var x0 = x[i];
            var x1 = b00 * x0 + z00;
            z00 = b01 * x0 - a01 * x1 + z01;
            z01 = b02 * x0 - a02 * x1;
            var x2 = b10 * x1 + z10;
            z10 = b11 * x1 - a11 * x2 + z11;
            z11 = b12 * x1 - a12 * x2;
            var x3 = b20 * x2 + z20;
            z20 = b21 * x2 - a21 * x3 + z21;
            z21 = b22 * x2 - a22 * x3;
            var x4 = b30 * x3 + z30;
            z30 = b31 * x3 - a31 * x4 + z31;
            z31 = b32 * x3 - a32 * x4;
            y[i] = x4;
        }
        z[0] = z00; z[1] = z01; z[2] = z10; z[3] = z11; z[4] = z20; z[5] = z21; z[6] = z30; z[7] = z31;
    }

    /// <summary>scipy's sample-major cascade, kept as the reference the blocked kernel is tested against.</summary>
    internal static void SosfiltReference(ReadOnlySpan<double> sos, ReadOnlySpan<double> x, Span<double> y, Span<double> zi)
    {
        var ns = sos.Length / 6;
        for (var i = 0; i < x.Length; i++)
        {
            var xn = x[i];
            for (var s = 0; s < ns; s++)
            {
                var o = s * 6;
                var xc = xn;
                xn = sos[o] * xc + zi[2 * s];
                zi[2 * s] = sos[o + 1] * xc - sos[o + 4] * xn + zi[2 * s + 1];
                zi[2 * s + 1] = sos[o + 2] * xc - sos[o + 5] * xn;
            }
            y[i] = xn;
        }
    }

    /// <summary>Filters with second-order sections from rest (<c>scipy.signal.sosfilt</c>).</summary>
    public static double[] Sosfilt(NdArray sos, double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var s = ValidateSos(sos, out var n);
        var y = new double[x.Length];
        Sosfilt(s, x, y, new double[2 * n]);
        return y;
    }

    /// <summary>Filters with second-order sections and initial states <paramref name="zi"/> (<c>n × 2</c>); returns output and final states.</summary>
    public static (double[] Y, NdArray Zf) Sosfilt(NdArray sos, double[] x, NdArray zi)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(zi);
        var s = ValidateSos(sos, out var n);
        if (zi.Size != 2 * n) throw new ArgumentException($"Invalid zi shape; must be ({n}, 2).", nameof(zi));
        var z = zi.AsContiguous().ToArray();
        var y = new double[x.Length];
        Sosfilt(s, x, y, z);
        return (y, new NdArray(z, n, 2));
    }

    /// <summary>Filters every line of <paramref name="x"/> along <paramref name="axis"/> with second-order sections.</summary>
    public static NdArray Sosfilt(NdArray sos, NdArray x, int axis = -1)
    {
        var s = ValidateSos(sos, out var n).ToArray();
        return ApplyAlongAxis(x, axis, (src, dst) =>
        {
            Span<double> z = 2 * n <= 128 ? stackalloc double[2 * n] : new double[2 * n];
            z.Clear();
            Sosfilt(s, src, dst, z);
        });
    }

    /// <summary>Steady-state step-response initial states for each section (<c>sosfilt_zi</c>), shape <c>n × 2</c>.</summary>
    public static NdArray SosfiltZi(NdArray sos)
    {
        var s = ValidateSos(sos, out var n);
        var zi = new double[2 * n];
        var scale = 1.0;
        for (var i = 0; i < n; i++)
        {
            var b = s.Slice(i * 6, 3).ToArray();
            var a = s.Slice(i * 6 + 3, 3).ToArray();
            var z = LfilterZi(b, a);
            zi[2 * i] = scale * z[0];
            zi[2 * i + 1] = scale * z[1];
            scale *= (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]);
        }
        return new NdArray(zi, n, 2);
    }

    // ------------------------------------------------------------------ axis helper

    internal delegate void SignalLineKernel(ReadOnlySpan<double> src, Span<double> dst);

    /// <summary>
    /// Applies a same-length line transform along one axis of an n-D array. The axis is moved
    /// last into a contiguous scratch buffer, so the kernel always sees contiguous spans; lines
    /// run in parallel once there is enough work to pay for it.
    /// </summary>
    internal static NdArray ApplyAlongAxis(NdArray x, int axis, SignalLineKernel kernel)
    {
        ArgumentNullException.ThrowIfNull(x);
        var rank = x.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentException("axis out of range.", nameof(axis));
        var shape = x.Shape.ToArray();
        var len = shape[axis];
        var src = x.AsContiguous().ToArray();
        var outer = 1;
        for (var i = 0; i < axis; i++) outer *= shape[i];
        var inner = 1;
        for (var i = axis + 1; i < rank; i++) inner *= shape[i];
        var dst = new double[src.Length];
        var lines = outer * inner;
        void Line(int li)
        {
            var o = li / inner;
            var iIn = li % inner;
            var baseIdx = o * len * inner + iIn;
            if (inner == 1)
            {
                kernel(src.AsSpan(baseIdx, len), dst.AsSpan(baseIdx, len));
                return;
            }
            var tmpIn = new double[len];
            var tmpOut = new double[len];
            for (var k = 0; k < len; k++) tmpIn[k] = src[baseIdx + k * inner];
            kernel(tmpIn, tmpOut);
            for (var k = 0; k < len; k++) dst[baseIdx + k * inner] = tmpOut[k];
        }
        if (lines > 1 && (long)lines * len >= 1 << 16) Parallel.For(0, lines, Line);
        else for (var li = 0; li < lines; li++) Line(li);
        return new NdArray(dst, shape);
    }
}
