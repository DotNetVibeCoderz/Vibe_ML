using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Fft
{
    // ---------------------------------------------------------------- public API, 1-D arrays

    /// <summary>
    /// Discrete cosine transform of type 1–4 — scipy's <c>dct(x, type, n, norm, orthogonalize)</c>.
    /// </summary>
    /// <remarks>
    /// <para>Unnormalised definitions (scipy's, <see cref="FftNorm.Backward"/>), for N points:</para>
    /// <list type="bullet">
    /// <item>I: <c>y_k = x_0 + (−1)^k x_{N−1} + 2 Σ_{n=1}^{N−2} x_n cos(π k n/(N−1))</c> (N ≥ 2)</item>
    /// <item>II: <c>y_k = 2 Σ x_n cos(π k (2n+1)/(2N))</c></item>
    /// <item>III: <c>y_k = x_0 + 2 Σ_{n≥1} x_n cos(π n (2k+1)/(2N))</c></item>
    /// <item>IV: <c>y_k = 2 Σ x_n cos(π (2k+1)(2n+1)/(4N))</c></item>
    /// </list>
    /// <para>
    /// <see cref="FftNorm.Forward"/> divides by 2(N−1) for type I and 2N otherwise; Ortho by the
    /// square root. <paramref name="orthogonalize"/> (default: true exactly when norm is Ortho)
    /// applies the √2 corrections to the first/last terms that make the Ortho matrices orthogonal.
    /// </para>
    /// <para>
    /// All four types run in O(N log N) through the real FFT: type II by Makhoul's even/odd
    /// reordering into one real transform of length N, type III by inverting that, type IV by a
    /// complex transform of length N/2 (2N for odd N), type I by the even extension of length
    /// 2(N−1).
    /// </para>
    /// </remarks>
    /// <param name="x">Real input.</param>
    /// <param name="type">Transform type, 1–4.</param>
    /// <param name="n">Length (truncate / zero-pad the input).</param>
    /// <param name="norm">Normalisation.</param>
    /// <param name="orthogonalize">Use the orthogonalised variant; defaults to <c>norm == Ortho</c>.</param>
    public static double[] Dct(double[] x, int type = 2, int? n = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null)
        => Trig1D(x, sine: false, type, n, norm, orthogonalize, inverse: false);

    /// <summary>
    /// Inverse DCT — scipy's <c>idct</c>: the DCT of the inverse type (II ↔ III, I and IV are their
    /// own inverses) with the normalisation direction swapped, so <c>Idct(Dct(x, t), t) = x</c>
    /// for every norm.
    /// </summary>
    public static double[] Idct(double[] x, int type = 2, int? n = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null)
        => Trig1D(x, sine: false, type, n, norm, orthogonalize, inverse: true);

    /// <summary>
    /// Discrete sine transform of type 1–4 — scipy's <c>dst</c>.
    /// </summary>
    /// <remarks>
    /// <para>Unnormalised definitions, for N points:</para>
    /// <list type="bullet">
    /// <item>I: <c>y_k = 2 Σ x_n sin(π (k+1)(n+1)/(N+1))</c></item>
    /// <item>II: <c>y_k = 2 Σ x_n sin(π (k+1)(2n+1)/(2N))</c></item>
    /// <item>III: <c>y_k = (−1)^k x_{N−1} + 2 Σ_{n=0}^{N−2} x_n sin(π (2k+1)(n+1)/(2N))</c></item>
    /// <item>IV: <c>y_k = 2 Σ x_n sin(π (2k+1)(2n+1)/(4N))</c></item>
    /// </list>
    /// <para>
    /// Types II–IV are computed from the matching DCT by the index identities
    /// <c>DST-II(x)_k = DCT-II((−1)^n x_n)_{N−1−k}</c> and
    /// <c>DST-III/IV(x)_k = (−1)^k DCT-III/IV(reversed x)_k</c>; type I uses the odd extension of
    /// length 2(N+1). The Forward normaliser is 2(N+1) for type I and 2N otherwise.
    /// </para>
    /// </remarks>
    public static double[] Dst(double[] x, int type = 2, int? n = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null)
        => Trig1D(x, sine: true, type, n, norm, orthogonalize, inverse: false);

    /// <summary>Inverse DST — scipy's <c>idst</c>.</summary>
    public static double[] Idst(double[] x, int type = 2, int? n = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null)
        => Trig1D(x, sine: true, type, n, norm, orthogonalize, inverse: true);

    // ---------------------------------------------------------------- along an axis

    /// <summary>DCT along one axis of an n-D array — scipy's <c>dct(x, type, n, axis, ...)</c>.</summary>
    /// <param name="x">Real input array.</param>
    /// <param name="type">Transform type, 1–4.</param>
    /// <param name="n">Length along the axis (truncate / zero-pad).</param>
    /// <param name="axis">Axis to transform.</param>
    /// <param name="norm">Normalisation.</param>
    /// <param name="orthogonalize">Orthogonalised variant; defaults to <c>norm == Ortho</c>.</param>
    /// <param name="workers">Thread cap; null = all cores when worthwhile.</param>
    public static NdArray Dct(NdArray x, int type = 2, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigAxisPublic(x, false, type, n, axis, norm, orthogonalize, false, workers);

    /// <summary>Inverse DCT along one axis — scipy's <c>idct(x, type, n, axis, ...)</c>.</summary>
    public static NdArray Idct(NdArray x, int type = 2, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigAxisPublic(x, false, type, n, axis, norm, orthogonalize, true, workers);

    /// <summary>DST along one axis — scipy's <c>dst(x, type, n, axis, ...)</c>.</summary>
    public static NdArray Dst(NdArray x, int type = 2, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigAxisPublic(x, true, type, n, axis, norm, orthogonalize, false, workers);

    /// <summary>Inverse DST along one axis — scipy's <c>idst(x, type, n, axis, ...)</c>.</summary>
    public static NdArray Idst(NdArray x, int type = 2, int? n = null, int axis = -1, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigAxisPublic(x, true, type, n, axis, norm, orthogonalize, true, workers);

    // ---------------------------------------------------------------- n-D

    /// <summary>n-D DCT over <paramref name="axes"/> (default all) — scipy's <c>dctn</c>.</summary>
    /// <param name="x">Real input array.</param>
    /// <param name="type">Transform type, 1–4, used on every axis.</param>
    /// <param name="s">Lengths along the transformed axes (truncate / zero-pad).</param>
    /// <param name="axes">Axes to transform; defaults to the last <c>s.Length</c>, or all.</param>
    /// <param name="norm">Normalisation, per axis.</param>
    /// <param name="orthogonalize">Orthogonalised variant; defaults to <c>norm == Ortho</c>.</param>
    /// <param name="workers">Thread cap; null = all cores when worthwhile.</param>
    public static NdArray DctN(NdArray x, int type = 2, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigN(x, false, type, s, axes, norm, orthogonalize, false, workers);

    /// <summary>n-D inverse DCT — scipy's <c>idctn</c>.</summary>
    public static NdArray IdctN(NdArray x, int type = 2, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigN(x, false, type, s, axes, norm, orthogonalize, true, workers);

    /// <summary>n-D DST — scipy's <c>dstn</c>.</summary>
    public static NdArray DstN(NdArray x, int type = 2, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigN(x, true, type, s, axes, norm, orthogonalize, false, workers);

    /// <summary>n-D inverse DST — scipy's <c>idstn</c>.</summary>
    public static NdArray IdstN(NdArray x, int type = 2, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, bool? orthogonalize = null, int? workers = null)
        => TrigN(x, true, type, s, axes, norm, orthogonalize, true, workers);

    // ---------------------------------------------------------------- plumbing

    private static double[] Trig1D(double[] x, bool sine, int type, int? n, FftNorm norm, bool? orthogonalize, bool inverse)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = new[] { x.Length };
        return TrigAxis(x, shape, 0, sine, type, n, norm, orthogonalize, inverse, 1);
    }

    private static NdArray TrigAxisPublic(NdArray x, bool sine, int type, int? n, int axis, FftNorm norm, bool? orthogonalize, bool inverse, int? workers)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var data = TrigAxis(x.ToArray(), shape, axis, sine, type, n, norm, orthogonalize, inverse, FftAxes.ResolveWorkers(workers));
        return new NdArray(data, shape);
    }

    private static NdArray TrigN(NdArray x, bool sine, int type, int[]? s, int[]? axes, FftNorm norm, bool? orthogonalize, bool inverse, int? workers)
    {
        ArgumentNullException.ThrowIfNull(x);
        var shape = x.Shape.ToArray();
        var (ax, sizes) = FftAxes.ResolveShapeAndAxes(shape, s, axes);
        var w = FftAxes.ResolveWorkers(workers);
        var data = x.ToArray();
        for (var i = 0; i < ax.Length; i++)
            data = TrigAxis(data, shape, ax[i], sine, type, sizes[i], norm, orthogonalize, inverse, w);
        return new NdArray(data, shape);
    }

    private static double[] TrigAxis(double[] data, int[] shape, int axis, bool sine, int type, int? n, FftNorm norm, bool? orthogonalize, bool inverse, int workers)
    {
        if (type is < 1 or > 4) throw new ArgumentException($"Transform type must be 1, 2, 3 or 4; got {type}.", nameof(type));
        var ax = FftAxes.NormalizeAxis(axis, shape.Length);
        var len = ResolveLength(n, shape[ax]);
        if (!sine && type == 1 && len < 2)
            throw new ArgumentException("DCT-I is not defined for fewer than 2 points.", nameof(n));

        // The inverse is the transform of the inverse type with the norm direction swapped.
        var orth = orthogonalize ?? norm == FftNorm.Ortho;
        var effectiveType = inverse ? type switch { 2 => 3, 3 => 2, _ => type } : type;
        var effectiveNorm = inverse ? SwapDirection(norm) : norm;

        data = FftAxes.Resize(data, shape, ax, len);
        var (outer, inner) = FftAxes.Split(shape, ax);
        var result = new double[data.Length];
        FftAxes.MapLines<double, double>(data, result, outer, len, len, inner, workers,
            (line, output) => FftTrig.Transform(line, output, sine, effectiveType, effectiveNorm, orth));
        return result;
    }
}

/// <summary>The DCT/DST kernels behind <see cref="Fft.Dct(double[], int, int?, FftNorm, bool?)"/> and friends.</summary>
internal static class FftTrig
{
    private static readonly ConcurrentDictionary<(int Kind, int N), Complex[]> Tables = new();

    private static readonly double Sqrt2 = Math.Sqrt(2.0);

    /// <summary>One line, with orthogonalisation and normalisation applied.</summary>
    public static void Transform(ReadOnlySpan<double> x, Span<double> y, bool sine, int type, FftNorm norm, bool orth)
    {
        var n = x.Length;
        double[]? rented = null;
        try
        {
            // Pre-scaling of the input (orthogonalised I and III) needs a private copy.
            var needsPre = orth && (type == 1 || type == 3);
            if (needsPre)
            {
                rented = ArrayPool<double>.Shared.Rent(n);
                var copy = rented.AsSpan(0, n);
                x.CopyTo(copy);
                if (type == 1 && !sine) { copy[0] *= Sqrt2; copy[n - 1] *= Sqrt2; }
                else if (type == 3 && !sine) copy[0] *= Sqrt2;
                else if (type == 3 && sine) copy[n - 1] *= Sqrt2;
                x = copy;
            }

            if (sine) Dst(x, y, type);
            else Dct(x, y, type);

            if (orth)
            {
                if (!sine && type == 1) { y[0] /= Sqrt2; y[n - 1] /= Sqrt2; }
                else if (!sine && type == 2) y[0] /= Sqrt2;
                else if (sine && type == 2) y[n - 1] /= Sqrt2;
            }

            var factor = !sine && type == 1 ? 2.0 * (n - 1) : sine && type == 1 ? 2.0 * (n + 1) : 2.0 * n;
            var scale = norm switch
            {
                FftNorm.Backward => 1.0,
                FftNorm.Ortho => 1.0 / Math.Sqrt(factor),
                _ => 1.0 / factor,
            };
            FftComplexOps.Scale(y, scale);
        }
        finally
        {
            if (rented is not null) ArrayPool<double>.Shared.Return(rented);
        }
    }

    private static Complex[] Table(int kind, int n) => Tables.GetOrAdd((kind, n), static key =>
    {
        var (kind, n) = key;
        return kind switch
        {
            // exp(-iπ k/(2N)), k < N
            0 => Build(n, k => FftMath.Twiddle(k, 4L * n)),
            // DCT-IV even N: exp(-iπ(4m+1)/(4N)), m < N/2
            1 => Build(n / 2, m => FftMath.Twiddle(4L * m + 1, 8L * n)),
            // DCT-IV even N: exp(-iπ p/N), p < N/2
            2 => Build(n / 2, p => FftMath.Twiddle(p, 2L * n)),
            // DCT-IV odd N: exp(-iπ(2k+1)/(4N)), k < N
            _ => Build(n, k => FftMath.Twiddle(2L * k + 1, 8L * n)),
        };

        static Complex[] Build(int count, Func<int, Complex> f)
        {
            var t = new Complex[count];
            for (var i = 0; i < count; i++) t[i] = f(i);
            return t;
        }
    });

    /// <summary>Unnormalised DCT of type 1–4; <paramref name="y"/> must not alias <paramref name="x"/>.</summary>
    public static void Dct(ReadOnlySpan<double> x, Span<double> y, int type)
    {
        switch (type)
        {
            case 1: Dct1(x, y); break;
            case 2: Dct2(x, y); break;
            case 3: Dct3(x, y); break;
            default: Dct4(x, y); break;
        }
    }

    /// <summary>Unnormalised DST of type 1–4 via the DCT index identities.</summary>
    public static void Dst(ReadOnlySpan<double> x, Span<double> y, int type)
    {
        var n = x.Length;
        if (type == 1) { Dst1(x, y); return; }

        var rented = ArrayPool<double>.Shared.Rent(2 * n);
        try
        {
            var t = rented.AsSpan(0, n);
            var u = rented.AsSpan(n, n);
            if (type == 2)
            {
                for (var i = 0; i < n; i++) t[i] = (i & 1) == 0 ? x[i] : -x[i];
                Dct2(t, u);
                for (var k = 0; k < n; k++) y[k] = u[n - 1 - k];
            }
            else
            {
                for (var i = 0; i < n; i++) t[i] = x[n - 1 - i];
                if (type == 3) Dct3(t, u); else Dct4(t, u);
                for (var k = 0; k < n; k++) y[k] = (k & 1) == 0 ? u[k] : -u[k];
            }
        }
        finally { ArrayPool<double>.Shared.Return(rented); }
    }

    /// <summary>Even extension of length 2(N−1); the real parts of its rfft are the DCT-I.</summary>
    private static void Dct1(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        var m = 2 * (n - 1);
        var e = ArrayPool<double>.Shared.Rent(m);
        var spec = ArrayPool<Complex>.Shared.Rent(n);
        try
        {
            for (var i = 0; i < n; i++) e[i] = x[i];
            for (var i = 1; i < n - 1; i++) e[m - i] = x[i];
            RfftPlan.Get(m).Forward(e.AsSpan(0, m), spec.AsSpan(0, n));
            for (var k = 0; k < n; k++) y[k] = spec[k].Real;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(e);
            ArrayPool<Complex>.Shared.Return(spec);
        }
    }

    /// <summary>Odd extension <c>[0, x, 0, −reverse(x)]</c> of length 2(N+1); DST-I is minus its imaginary part.</summary>
    private static void Dst1(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        var m = 2 * (n + 1);
        var e = ArrayPool<double>.Shared.Rent(m);
        var spec = ArrayPool<Complex>.Shared.Rent(n + 2);
        try
        {
            e[0] = 0;
            e[n + 1] = 0;
            for (var i = 0; i < n; i++) { e[i + 1] = x[i]; e[m - 1 - i] = -x[i]; }
            RfftPlan.Get(m).Forward(e.AsSpan(0, m), spec.AsSpan(0, n + 2));
            for (var k = 0; k < n; k++) y[k] = -spec[k + 1].Imaginary;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(e);
            ArrayPool<Complex>.Shared.Return(spec);
        }
    }

    /// <summary>
    /// Makhoul (1980): reorder to <c>v = [x_0, x_2, …, x_3, x_1]</c>, take one real FFT of length
    /// N, and rotate: <c>y_k = 2 Re(exp(−iπk/2N) V_k)</c>.
    /// </summary>
    private static void Dct2(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        var h = n / 2;
        var v = ArrayPool<double>.Shared.Rent(n);
        var spec = ArrayPool<Complex>.Shared.Rent(h + 1);
        try
        {
            for (var i = 0; 2 * i < n; i++) v[i] = x[2 * i];
            for (var i = 0; 2 * i + 1 < n; i++) v[n - 1 - i] = x[2 * i + 1];
            RfftPlan.Get(n).Forward(v.AsSpan(0, n), spec.AsSpan(0, h + 1));
            var tw = Table(0, n);
            for (var k = 0; k <= h; k++)
            {
                var a = spec[k];
                y[k] = 2.0 * (tw[k].Real * a.Real - tw[k].Imaginary * a.Imaginary);
            }
            for (var k = h + 1; k < n; k++)
            {
                var a = spec[n - k]; // V_k = conj(V_{N-k})
                y[k] = 2.0 * (tw[k].Real * a.Real + tw[k].Imaginary * a.Imaginary);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(v);
            ArrayPool<Complex>.Shared.Return(spec);
        }
    }

    /// <summary>
    /// The inverse of Makhoul's reordering: <c>V_k = exp(iπk/2N)(x_k − i x_{N−k})</c> is Hermitian,
    /// so one inverse real FFT of length N gives the reordered output.
    /// </summary>
    private static void Dct3(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        var h = n / 2;
        var v = ArrayPool<double>.Shared.Rent(n);
        var spec = ArrayPool<Complex>.Shared.Rent(h + 1);
        try
        {
            var tw = Table(0, n);
            spec[0] = new Complex(x[0], 0);
            for (var k = 1; k <= h; k++)
            {
                var c = Complex.Conjugate(tw[k]);
                var a = new Complex(x[k], -x[n - k]);
                spec[k] = c * a;
            }
            RfftPlan.Get(n).Inverse(spec.AsSpan(0, h + 1), v.AsSpan(0, n));
            for (var i = 0; 2 * i < n; i++) y[2 * i] = v[i];
            for (var i = 0; 2 * i + 1 < n; i++) y[2 * i + 1] = v[n - 1 - i];
        }
        finally
        {
            ArrayPool<double>.Shared.Return(v);
            ArrayPool<Complex>.Shared.Return(spec);
        }
    }

    /// <summary>
    /// Even N: pair <c>x_{2m}</c> with <c>x_{N−1−2m}</c> into one complex sequence, pre-rotate by
    /// <c>exp(−iπ(4m+1)/4N)</c>, one complex FFT of length N/2, post-rotate by <c>exp(−iπp/N)</c>:
    /// the real parts are the even outputs and minus the imaginary parts the odd ones, reversed.
    /// Odd N: <c>y_k = 2 Re(exp(−iπ(2k+1)/4N) · FFT_{2N}(x_n exp(−iπn/2N))_k)</c>.
    /// </summary>
    private static void Dct4(ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        if ((n & 1) == 0)
        {
            var m = n / 2;
            var u = ArrayPool<Complex>.Shared.Rent(m);
            try
            {
                var pre = Table(1, n);
                var post = Table(2, n);
                for (var j = 0; j < m; j++) u[j] = new Complex(x[2 * j], x[n - 1 - 2 * j]) * pre[j];
                var span = u.AsSpan(0, m);
                FftPlan.Get(m).Execute(span, span, inverse: false);
                for (var p = 0; p < m; p++)
                {
                    var c = post[p] * u[p];
                    y[2 * p] = 2.0 * c.Real;
                    y[n - 1 - 2 * p] = -2.0 * c.Imaginary;
                }
            }
            finally { ArrayPool<Complex>.Shared.Return(u); }
            return;
        }

        var b = ArrayPool<Complex>.Shared.Rent(2 * n);
        try
        {
            var pre = Table(0, n);
            var post = Table(3, n);
            for (var j = 0; j < n; j++) b[j] = x[j] * pre[j];
            var span = b.AsSpan(0, 2 * n);
            span[n..].Clear();
            FftPlan.Get(2 * n).Execute(span, span, inverse: false);
            for (var k = 0; k < n; k++)
            {
                var c = post[k] * b[k];
                y[k] = 2.0 * c.Real;
            }
        }
        finally { ArrayPool<Complex>.Shared.Return(b); }
    }
}
