using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Gravicode.Science.SciNet;

/// <summary>Marker type: forward transform, kernel <c>exp(-2πi jk/n)</c>.</summary>
internal struct FftForwardDirection;

/// <summary>Marker type: backward transform, kernel <c>exp(+2πi jk/n)</c>.</summary>
internal struct FftBackwardDirection;

/// <summary>Marker type: use the FMA / AVX kernels where the CPU has them.</summary>
internal struct FftSimd;

/// <summary>Marker type: scalar complex arithmetic only — the reference path the tests compare against.</summary>
internal struct FftScalar;

/// <summary>
/// The butterfly passes of the mixed-radix transform, on raw interleaved re/im buffers.
/// </summary>
/// <remarks>
/// <para>
/// The structure is FFTPACK's (as modernised by pocketfft, which is what numpy and scipy run): a
/// radix-<c>ip</c> pass reads <c>CC(i, j, k) = cc[i + ido (j + ip k)]</c> and writes
/// <c>CH(i, k, j) = ch[i + ido (k + l1 j)]</c>, ping-ponging between two buffers, so the result
/// comes out in natural order with no bit-reversal pass (a Stockham autosort). The twiddle for
/// output <c>j</c> of column <c>i</c> is <c>exp(-2πi j l1 i / n)</c>, stored for every
/// <c>i</c> including 0 so the inner loop has no special case and can run two columns per
/// 256-bit register.
/// </para>
/// <para>
/// The direction is a generic struct parameter rather than a <c>bool</c>: the JIT specialises each
/// kernel per direction, so <c>typeof(D) == typeof(FftForwardDirection)</c> folds to a constant and
/// neither direction pays for the other's branch.
/// </para>
/// <para>
/// Each complex number lives in one <see cref="Vector128{T}"/>: add and subtract are one
/// instruction, and a complex multiply is <c>movddup</c>/<c>unpckhpd</c>/<c>mul</c>/
/// <c>fmaddsub</c>. Radix-2 and radix-4 additionally have a 256-bit path that handles two columns
/// at once. Without FMA every multiply falls back to the scalar formula, which is also what the
/// <see cref="FftScalar"/> instantiation runs unconditionally, as the reference for the tests.
/// </para>
/// </remarks>
internal static unsafe class FftKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool UseFma<S>() where S : struct => typeof(S) == typeof(FftSimd) && Fma.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool UseAvx<S>() where S : struct => typeof(S) == typeof(FftSimd) && Avx.IsSupported && Fma.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsForward<D>() where D : struct => typeof(D) == typeof(FftForwardDirection);

    // ---------------------------------------------------------------- 128-bit helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Ld(double* p) => Vector128.Load(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void St(double* p, Vector128<double> v) => v.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Swap(Vector128<double> a)
    {
        if (Avx.IsSupported) return Avx.Permute(a, 0b01);
        if (Sse2.IsSupported) return Sse2.Shuffle(a, a, 0b01);
        return Vector128.Create(a.GetElement(1), a.GetElement(0));
    }

    /// <summary>Multiplies by <c>-i</c> (forward) or <c>+i</c> (backward).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Rot<D>(Vector128<double> a) where D : struct
    {
        // -i (r, i) = (i, -r); +i (r, i) = (-i, r). Swap, then flip one sign bit.
        var s = Swap(a);
        return IsForward<D>()
            ? Vector128.Xor(s, Vector128.Create(0.0, -0.0))
            : Vector128.Xor(s, Vector128.Create(-0.0, 0.0));
    }

    /// <summary><c>a * w</c> forward, <c>a * conj(w)</c> backward.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Tw<D, S>(Vector128<double> a, Vector128<double> w) where D : struct where S : struct
    {
        if (UseFma<S>())
        {
            var wr = Sse3.MoveAndDuplicate(w);
            var wi = Sse2.UnpackHigh(w, w);
            var t = Sse2.Multiply(Swap(a), wi);
            // fmaddsub: even lane a*b - c, odd lane a*b + c.
            return IsForward<D>() ? Fma.MultiplyAddSubtract(a, wr, t) : Fma.MultiplySubtractAdd(a, wr, t);
        }
        double ar = a.GetElement(0), ai = a.GetElement(1), xr = w.GetElement(0), xi = w.GetElement(1);
        if (!IsForward<D>()) xi = -xi;
        return Vector128.Create(ar * xr - ai * xi, ar * xi + ai * xr);
    }

    // ---------------------------------------------------------------- 256-bit helpers (two columns)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Ld2(double* p) => Vector256.Load(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void St2(double* p, Vector256<double> v) => v.Store(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Rot2<D>(Vector256<double> a) where D : struct
    {
        var s = Avx.Permute(a, 0b0101);
        return IsForward<D>()
            ? Vector256.Xor(s, Vector256.Create(0.0, -0.0, 0.0, -0.0))
            : Vector256.Xor(s, Vector256.Create(-0.0, 0.0, -0.0, 0.0));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Tw2<D>(Vector256<double> a, Vector256<double> w) where D : struct
    {
        var wr = Avx.DuplicateEvenIndexed(w);
        var wi = Avx.Permute(w, 0b1111);
        var t = Avx.Multiply(Avx.Permute(a, 0b0101), wi);
        return IsForward<D>() ? Fma.MultiplyAddSubtract(a, wr, t) : Fma.MultiplySubtractAdd(a, wr, t);
    }

    // ---------------------------------------------------------------- radix 2

    public static void Pass2<D, S>(int ido, int l1, double* cc, double* ch, double* wa) where D : struct where S : struct
    {
        const int cdim = 2;
        if (ido == 1)
        {
            for (var k = 0; k < l1; k++)
            {
                var a = Ld(cc + 2 * (cdim * k));
                var b = Ld(cc + 2 * (1 + cdim * k));
                St(ch + 2 * k, a + b);
                St(ch + 2 * (k + l1), a - b);
            }
            return;
        }

        for (var k = 0; k < l1; k++)
        {
            var src0 = cc + 2 * (ido * (cdim * k));
            var src1 = cc + 2 * (ido * (1 + cdim * k));
            var dst0 = ch + 2 * (ido * k);
            var dst1 = ch + 2 * (ido * (k + l1));
            var i = 0;
            if (UseAvx<S>())
            {
                for (; i + 1 < ido; i += 2)
                {
                    var a = Ld2(src0 + 2 * i);
                    var b = Ld2(src1 + 2 * i);
                    St2(dst0 + 2 * i, a + b);
                    St2(dst1 + 2 * i, Tw2<D>(a - b, Ld2(wa + 2 * i)));
                }
            }
            for (; i < ido; i++)
            {
                var a = Ld(src0 + 2 * i);
                var b = Ld(src1 + 2 * i);
                St(dst0 + 2 * i, a + b);
                St(dst1 + 2 * i, Tw<D, S>(a - b, Ld(wa + 2 * i)));
            }
        }
    }

    // ---------------------------------------------------------------- radix 4

    public static void Pass4<D, S>(int ido, int l1, double* cc, double* ch, double* wa) where D : struct where S : struct
    {
        const int cdim = 4;
        if (ido == 1)
        {
            for (var k = 0; k < l1; k++)
            {
                var src = cc + 2 * (cdim * k);
                var c0 = Ld(src); var c1 = Ld(src + 2); var c2 = Ld(src + 4); var c3 = Ld(src + 6);
                var t2 = c0 + c2; var t1 = c0 - c2;
                var t3 = c1 + c3; var t4 = Rot<D>(c1 - c3);
                St(ch + 2 * k, t2 + t3);
                St(ch + 2 * (k + l1), t1 + t4);
                St(ch + 2 * (k + 2 * l1), t2 - t3);
                St(ch + 2 * (k + 3 * l1), t1 - t4);
            }
            return;
        }

        var w1 = wa;
        var w2 = wa + 2 * ido;
        var w3 = wa + 4 * ido;
        var stride = 2 * ido * l1; // between output blocks j and j+1
        for (var k = 0; k < l1; k++)
        {
            var s0 = cc + 2 * (ido * (cdim * k));
            var s1 = s0 + 2 * ido;
            var s2 = s1 + 2 * ido;
            var s3 = s2 + 2 * ido;
            var d0 = ch + 2 * (ido * k);
            var d1 = d0 + stride;
            var d2 = d1 + stride;
            var d3 = d2 + stride;
            var i = 0;
            if (UseAvx<S>())
            {
                for (; i + 1 < ido; i += 2)
                {
                    var o = 2 * i;
                    var c0 = Ld2(s0 + o); var c1 = Ld2(s1 + o); var c2 = Ld2(s2 + o); var c3 = Ld2(s3 + o);
                    var t2 = c0 + c2; var t1 = c0 - c2;
                    var t3 = c1 + c3; var t4 = Rot2<D>(c1 - c3);
                    St2(d0 + o, t2 + t3);
                    St2(d1 + o, Tw2<D>(t1 + t4, Ld2(w1 + o)));
                    St2(d2 + o, Tw2<D>(t2 - t3, Ld2(w2 + o)));
                    St2(d3 + o, Tw2<D>(t1 - t4, Ld2(w3 + o)));
                }
            }
            for (; i < ido; i++)
            {
                var o = 2 * i;
                var c0 = Ld(s0 + o); var c1 = Ld(s1 + o); var c2 = Ld(s2 + o); var c3 = Ld(s3 + o);
                var t2 = c0 + c2; var t1 = c0 - c2;
                var t3 = c1 + c3; var t4 = Rot<D>(c1 - c3);
                St(d0 + o, t2 + t3);
                St(d1 + o, Tw<D, S>(t1 + t4, Ld(w1 + o)));
                St(d2 + o, Tw<D, S>(t2 - t3, Ld(w2 + o)));
                St(d3 + o, Tw<D, S>(t1 - t4, Ld(w3 + o)));
            }
        }
    }

    // ---------------------------------------------------------------- radix 3

    public static void Pass3<D, S>(int ido, int l1, double* cc, double* ch, double* wa) where D : struct where S : struct
    {
        const int cdim = 3;
        const double tw1r = -0.5;
        var tw1i = (IsForward<D>() ? -1.0 : 1.0) * 0.86602540378443864676;
        // i * tw1i * t = swap(t) * (-tw1i, tw1i)
        var rotScale = Vector128.Create(-tw1i, tw1i);
        var half = Vector128.Create(tw1r);

        for (var k = 0; k < l1; k++)
        {
            var s0 = cc + 2 * (ido * (cdim * k));
            var s1 = s0 + 2 * ido;
            var s2 = s1 + 2 * ido;
            var d0 = ch + 2 * (ido * k);
            var d1 = ch + 2 * (ido * (k + l1));
            var d2 = ch + 2 * (ido * (k + 2 * l1));
            for (var i = 0; i < ido; i++)
            {
                var o = 2 * i;
                var t0 = Ld(s0 + o);
                var c1 = Ld(s1 + o); var c2 = Ld(s2 + o);
                var t1 = c1 + c2; var t2 = c1 - c2;
                var ca = t0 + half * t1;
                var cb = Swap(t2) * rotScale;
                St(d0 + o, t0 + t1);
                if (ido == 1)
                {
                    St(d1 + o, ca + cb);
                    St(d2 + o, ca - cb);
                }
                else
                {
                    St(d1 + o, Tw<D, S>(ca + cb, Ld(wa + o)));
                    St(d2 + o, Tw<D, S>(ca - cb, Ld(wa + 2 * ido + o)));
                }
            }
        }
    }

    // ---------------------------------------------------------------- radix 5

    public static void Pass5<D, S>(int ido, int l1, double* cc, double* ch, double* wa) where D : struct where S : struct
    {
        const int cdim = 5;
        var sign = IsForward<D>() ? -1.0 : 1.0;
        var tw1r = Vector128.Create(0.3090169943749474241);
        var tw2r = Vector128.Create(-0.8090169943749474241);
        var tw1i = sign * 0.95105651629515357212;
        var tw2i = sign * 0.58778525229247312917;
        var r11 = Vector128.Create(-tw1i, tw1i);
        var r12 = Vector128.Create(-tw2i, tw2i);
        var r22 = Vector128.Create(-tw2i, tw2i);
        var r21 = Vector128.Create(tw1i, -tw1i); // -tw1i

        for (var k = 0; k < l1; k++)
        {
            var s0 = cc + 2 * (ido * (cdim * k));
            var d0 = ch + 2 * (ido * k);
            var dstride = 2 * ido * l1;
            for (var i = 0; i < ido; i++)
            {
                var o = 2 * i;
                var t0 = Ld(s0 + o);
                var c1 = Ld(s0 + 2 * ido + o);
                var c2 = Ld(s0 + 4 * ido + o);
                var c3 = Ld(s0 + 6 * ido + o);
                var c4 = Ld(s0 + 8 * ido + o);
                var t1 = c1 + c4; var t4 = c1 - c4;
                var t2 = c2 + c3; var t3 = c2 - c3;
                St(d0 + o, t0 + t1 + t2);

                var sw4 = Swap(t4); var sw3 = Swap(t3);
                // u = 1, 4
                var ca1 = t0 + tw1r * t1 + tw2r * t2;
                var cb1 = sw4 * r11 + sw3 * r12;
                // u = 2, 3
                var ca2 = t0 + tw2r * t1 + tw1r * t2;
                var cb2 = sw4 * r22 + sw3 * r21;

                var y1 = ca1 + cb1; var y4 = ca1 - cb1;
                var y2 = ca2 + cb2; var y3 = ca2 - cb2;
                if (ido == 1)
                {
                    St(d0 + o + dstride, y1);
                    St(d0 + o + 2 * dstride, y2);
                    St(d0 + o + 3 * dstride, y3);
                    St(d0 + o + 4 * dstride, y4);
                }
                else
                {
                    St(d0 + o + dstride, Tw<D, S>(y1, Ld(wa + o)));
                    St(d0 + o + 2 * dstride, Tw<D, S>(y2, Ld(wa + 2 * ido + o)));
                    St(d0 + o + 3 * dstride, Tw<D, S>(y3, Ld(wa + 4 * ido + o)));
                    St(d0 + o + 4 * dstride, Tw<D, S>(y4, Ld(wa + 6 * ido + o)));
                }
            }
        }
    }

    // ---------------------------------------------------------------- generic odd radix

    /// <summary>
    /// Any odd radix, by the direct DFT of size <paramref name="ip"/> with the usual
    /// symmetric/antisymmetric split (halves the multiplies).
    /// </summary>
    /// <param name="ido">Columns per block.</param>
    /// <param name="l1">Product of the radices already processed.</param>
    /// <param name="ip">The radix (odd).</param>
    /// <param name="cc">Input buffer.</param>
    /// <param name="ch">Output buffer.</param>
    /// <param name="wa">Twiddles for this pass.</param>
    /// <param name="cs">cos and sin of <c>2π m/ip</c>, interleaved, for <c>m &lt; ip</c>.</param>
    public static void PassGeneric<D, S>(int ido, int l1, int ip, double* cc, double* ch, double* wa, double* cs)
        where D : struct where S : struct
    {
        var half = (ip - 1) / 2;
        var s = stackalloc Vector128<double>[half + 1];
        var d = stackalloc Vector128<double>[half + 1];
        var dstride = 2 * ido * l1;

        for (var k = 0; k < l1; k++)
        {
            var src = cc + 2 * (ido * (ip * k));
            var dst = ch + 2 * (ido * k);
            for (var i = 0; i < ido; i++)
            {
                var o = 2 * i;
                var c0 = Ld(src + o);
                var y0 = c0;
                for (var j = 1; j <= half; j++)
                {
                    var a = Ld(src + 2 * ido * j + o);
                    var b = Ld(src + 2 * ido * (ip - j) + o);
                    s[j] = a + b;
                    d[j] = a - b;
                    y0 += s[j];
                }
                St(dst + o, y0);

                for (var m = 1; m <= half; m++)
                {
                    var A = c0;
                    var B = Vector128<double>.Zero;
                    var idx = 0;
                    for (var j = 1; j <= half; j++)
                    {
                        idx += m;
                        if (idx >= ip) idx -= ip;
                        A += s[j] * Vector128.Create(cs[2 * idx]);
                        B += d[j] * Vector128.Create(cs[2 * idx + 1]);
                    }
                    // forward: y_m = A - iB; backward: y_m = A + iB. Rot<D> is -i / +i.
                    var rb = Rot<D>(B);
                    var ym = A + rb;
                    var yp = A - rb;
                    if (ido == 1)
                    {
                        St(dst + o + m * dstride, ym);
                        St(dst + o + (ip - m) * dstride, yp);
                    }
                    else
                    {
                        St(dst + o + m * dstride, Tw<D, S>(ym, Ld(wa + 2 * ((m - 1) * ido) + o)));
                        St(dst + o + (ip - m) * dstride, Tw<D, S>(yp, Ld(wa + 2 * ((ip - m - 1) * ido) + o)));
                    }
                }
            }
        }
    }
}
