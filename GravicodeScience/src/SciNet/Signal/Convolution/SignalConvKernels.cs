using System.Numerics;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Allocation-free inner loops shared by the convolution, resampling and filtering code. Every
/// direct sum is written as a dot product of contiguous spans (the kernel is reversed once up
/// front) so it runs on <see cref="Vector{T}"/> lanes.
/// </summary>
internal static class SignalConvKernels
{
    /// <summary>When false the scalar reference paths run; the tests compare the two.</summary>
    internal static bool SimdEnabled = true;

    /// <summary>Σ a[i]·b[i] with two vector accumulators.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var n = a.Length;
        var i = 0;
        double sum = 0;
        if (SimdEnabled && Vector.IsHardwareAccelerated && n >= 2 * Vector<double>.Count)
        {
            var w = Vector<double>.Count;
            ref var ra = ref MemoryMarshal.GetReference(a);
            ref var rb = ref MemoryMarshal.GetReference(b);
            var acc0 = Vector<double>.Zero;
            var acc1 = Vector<double>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                acc0 += Vector.LoadUnsafe(ref ra, (nuint)i) * Vector.LoadUnsafe(ref rb, (nuint)i);
                acc1 += Vector.LoadUnsafe(ref ra, (nuint)(i + w)) * Vector.LoadUnsafe(ref rb, (nuint)(i + w));
            }
            sum = Vector.Sum(acc0 + acc1);
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>
    /// Writes (or adds, when <paramref name="accumulate"/>) <c>y[t] = full[start + t]</c> where
    /// <c>full = x ∗ h</c> is the full linear convolution and <paramref name="hRev"/> is h reversed.
    /// </summary>
    public static void ConvolveRange(ReadOnlySpan<double> x, ReadOnlySpan<double> hRev, int start, Span<double> y, bool accumulate)
    {
        var n = x.Length;
        var m = hRev.Length;
        for (var t = 0; t < y.Length; t++)
        {
            var k = start + t;
            var lo = Math.Max(0, k - m + 1);
            var hi = Math.Min(n - 1, k);
            var v = lo > hi ? 0.0 : Dot(x.Slice(lo, hi - lo + 1), hRev.Slice(lo + m - 1 - k, hi - lo + 1));
            if (accumulate) y[t] += v; else y[t] = v;
        }
    }

    /// <summary>
    /// <see cref="ConvolveRange"/> over arrays, split into parallel chunks once the work
    /// (outputs × kernel length) is large enough to pay for the scheduling.
    /// </summary>
    public static void ConvolveRangeParallel(double[] x, double[] hRev, int start, double[] y, int yOffset, int count)
    {
        var work = (long)count * Math.Min(x.Length, hRev.Length);
        if (work < 1 << 18 || count < 64)
        {
            ConvolveRange(x, hRev, start, y.AsSpan(yOffset, count), false);
            return;
        }
        var chunks = Math.Min(Environment.ProcessorCount * 4, count / 32);
        var per = (count + chunks - 1) / chunks;
        Parallel.For(0, chunks, c =>
        {
            var s = c * per;
            var e = Math.Min(count, s + per);
            if (s < e) ConvolveRange(x, hRev, start + s, y.AsSpan(yOffset + s, e - s), false);
        });
    }

    /// <summary>
    /// The smallest <em>even</em> 5-smooth length ≥ <paramref name="target"/>. Odd real FFT lengths
    /// lose the half-length complex trick and cost about twice as much, and <c>next_fast_len</c>
    /// happily returns odd ones (10125 for 10007), so convolution pads to an even length instead.
    /// </summary>
    public static int EvenFastLength(int target)
        => target <= 2 ? 2 : 2 * Fft.NextFastLen((target + 1) / 2, real: true);

    /// <summary>
    /// Full linear convolution through the real FFT at an even fast length; the two forward
    /// transforms run concurrently once they are large enough to pay for it.
    /// </summary>
    public static double[] FftConvolveFull(double[] a, double[] b)
    {
        var outLen = a.Length + b.Length - 1;
        var size = EvenFastLength(outLen);
        var plan = RfftPlan.Get(size);
        var bins = plan.SpectrumLength;
        var fa = new Complex[bins];
        var fb = new Complex[bins];
        var bufA = new double[size];
        var bufB = new double[size];
        a.CopyTo(bufA, 0);
        b.CopyTo(bufB, 0);
        if (size >= 1 << 15)
            Parallel.Invoke(() => plan.Forward(bufA, fa), () => plan.Forward(bufB, fb));
        else
        {
            plan.Forward(bufA, fa);
            plan.Forward(bufB, fb);
        }
        var scale = 1.0 / size;
        for (var k = 0; k < bins; k++) fa[k] *= fb[k] * scale;
        plan.Inverse(fa, bufA);
        if (outLen == size) return bufA;
        return bufA.AsSpan(0, outLen).ToArray();
    }

    /// <summary>Reversed copy.</summary>
    public static double[] Reversed(ReadOnlySpan<double> a)
    {
        var r = a.ToArray();
        Array.Reverse(r);
        return r;
    }
}
