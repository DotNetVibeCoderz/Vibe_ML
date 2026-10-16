using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A precomputed complex FFT of one length: factorisation, twiddle tables and (for awkward
/// lengths) a Bluestein convolution, reusable from any number of threads.
/// </summary>
/// <remarks>
/// <para>
/// <b>Algorithm.</b> Lengths whose prime factors are small are transformed directly by a
/// mixed-radix Stockham pass sequence (radix 4, 2, 3, 5 with hand-written butterflies, any other
/// odd prime with a generic one), so 1000, 1536, 2187 and 3125 cost about what a power of two
/// does. A length with a large prime factor is instead re-expressed as a convolution (Bluestein's
/// chirp-z identity <c>jk = (j² + k² − (k−j)²)/2</c>) of a 5-smooth length ≥ 2n − 1, whose
/// plan is cached like any other. The choice uses an operation-count model, so a length such as
/// 7·11·13 stays direct while 10007 goes through Bluestein.
/// </para>
/// <para>
/// <b>Normalisation.</b> Plans are <em>unnormalised</em> in both directions: a forward then a
/// backward execution multiplies by <see cref="Length"/>. The <see cref="Fft"/> facade applies
/// <see cref="FftNorm"/>; the plan stays the raw kernel so callers composing transforms (Bluestein,
/// the real transform, the DCTs) do not pay for scalings that cancel.
/// </para>
/// <para>
/// <b>Threading.</b> A plan is immutable once built. <see cref="Get"/> caches plans per length in
/// a concurrent dictionary; scratch memory comes from <see cref="ArrayPool{T}.Shared"/> unless the
/// caller supplies it, so steady-state execution allocates nothing.
/// </para>
/// </remarks>
public sealed class FftPlan
{
    private const int CacheLimit = 512;
    private static readonly ConcurrentDictionary<int, FftPlan> Cache = new();

    private readonly int[] _factors = [];
    private readonly double[][] _twiddles = [];
    private readonly double[]?[] _genericTables = [];

    // Bluestein
    private readonly FftPlan? _inner;
    private readonly Complex[]? _chirp;
    private readonly Complex[]? _kernel;

    /// <summary>
    /// Diagnostics switch: when false, every plan runs its butterflies with scalar complex
    /// arithmetic instead of the FMA/AVX kernels. Both paths compute the same thing; the scalar one
    /// is kept as the reference the SIMD path is tested against. Process-wide; default true.
    /// </summary>
    public static bool SimdEnabled { get; set; } = true;

    /// <summary>Transform length.</summary>
    public int Length { get; }

    /// <summary>
    /// Number of <see cref="Complex"/> elements of scratch an execution needs when the caller
    /// passes it explicitly.
    /// </summary>
    public int ScratchLength { get; }

    /// <summary>True when the plan evaluates the transform as a Bluestein convolution.</summary>
    public bool UsesBluestein => _inner is not null;

    /// <summary>The radices of the direct transform, in pass order (empty for Bluestein and n = 1).</summary>
    public IReadOnlyList<int> Factors => _factors;

    /// <summary>Builds a plan without consulting or filling the cache.</summary>
    /// <param name="length">Transform length, at least 1.</param>
    public FftPlan(int length)
    {
        if (length < 1) throw new ArgumentException($"FFT length must be positive, got {length}.", nameof(length));
        Length = length;
        if (length == 1) return;

        if (PreferBluestein(length))
        {
            var m = (int)FftMath.NextSmooth(2L * length - 1, 5);
            _inner = Get(m);
            _chirp = new Complex[length];
            var twoN = 2L * length;
            for (long k = 0; k < length; k++)
                _chirp[k] = FftMath.Twiddle(k * k % twoN, twoN); // exp(-iπ k²/n), with k² reduced mod 2n
            // Kernel h_j = conj(chirp_j), circularly symmetric; transformed once and pre-scaled by
            // 1/m so the inner inverse transform needs no normalisation pass.
            var h = new Complex[m];
            h[0] = Complex.Conjugate(_chirp[0]);
            for (var j = 1; j < length; j++) h[j] = h[m - j] = Complex.Conjugate(_chirp[j]);
            _inner.Execute(h, h, inverse: false);
            for (var j = 0; j < m; j++) h[j] /= m;
            _kernel = h;
            ScratchLength = m + _inner.ScratchLength;
            return;
        }

        _factors = Factorise(length);
        _twiddles = new double[_factors.Length][];
        _genericTables = new double[]?[_factors.Length];
        var l1 = 1;
        for (var p = 0; p < _factors.Length; p++)
        {
            var ip = _factors[p];
            var ido = length / (l1 * ip);
            var tw = new double[2 * (ip - 1) * ido];
            for (var j = 1; j < ip; j++)
            for (var i = 0; i < ido; i++)
            {
                var w = FftMath.Twiddle((long)j * l1 * i, length);
                var at = 2 * ((j - 1) * ido + i);
                tw[at] = w.Real;
                tw[at + 1] = w.Imaginary;
            }
            _twiddles[p] = tw;
            if (ip > 5)
            {
                // cos(2πm/ip), sin(2πm/ip) — positive sine; the kernel applies the direction.
                var cs = new double[2 * ip];
                for (var m = 0; m < ip; m++)
                {
                    var w = FftMath.Twiddle(m, ip);
                    cs[2 * m] = w.Real;
                    cs[2 * m + 1] = -w.Imaginary;
                }
                _genericTables[p] = cs;
            }
            l1 *= ip;
        }
        ScratchLength = length;
    }

    /// <summary>
    /// The cached plan for <paramref name="length"/>, building it on first use. Thread-safe.
    /// </summary>
    /// <remarks>
    /// The cache is cleared wholesale once it holds 512 lengths, which bounds memory for callers
    /// that sweep many lengths without making the common case pay for an LRU.
    /// </remarks>
    public static FftPlan Get(int length)
    {
        if (Cache.TryGetValue(length, out var plan)) return plan;
        if (Cache.Count >= CacheLimit) Cache.Clear();
        return Cache.GetOrAdd(length, static n => new FftPlan(n));
    }

    /// <summary>
    /// Bluestein costs three transforms of about 2n, so it pays only when the direct factorisation
    /// contains a large prime. The 1.5 fudge factor is pocketfft's, and it agrees with measurement
    /// here: with it, 7·11·13 = 1001 stays direct and every prime above ~100 goes to Bluestein.
    /// </summary>
    private static bool PreferBluestein(int n)
    {
        if (n < 50) return false;
        var largest = FftMath.LargestPrimeFactor(n);
        if (largest <= 13) return false;
        var direct = FftMath.CostGuess(n);
        var m = (int)FftMath.NextSmooth(2L * n - 1, 5);
        var blue = 2 * FftMath.CostGuess(m) * 1.5;
        return blue < direct;
    }

    /// <summary>Pass order: one radix 2 (if the 2-power is odd) first, then 4s, then odd primes ascending.</summary>
    private static int[] Factorise(int n)
    {
        var result = new List<int>();
        while (n % 4 == 0) { result.Add(4); n /= 4; }
        if (n % 2 == 0) { result.Insert(0, 2); n /= 2; }
        foreach (var p in FftMath.PrimeFactors(n)) result.Add(p);
        return [.. result];
    }

    // ---------------------------------------------------------------- execution

    /// <summary>
    /// Unnormalised transform of <paramref name="input"/> into <paramref name="output"/>, renting
    /// scratch from the shared pool.
    /// </summary>
    /// <param name="input">Exactly <see cref="Length"/> values.</param>
    /// <param name="output">Exactly <see cref="Length"/> values; may be the same memory as
    /// <paramref name="input"/> (in place) but must not partially overlap it.</param>
    /// <param name="inverse">False for <c>exp(-2πi jk/n)</c>, true for <c>exp(+2πi jk/n)</c>.</param>
    public void Execute(ReadOnlySpan<Complex> input, Span<Complex> output, bool inverse = false)
    {
        if (ScratchLength == 0)
        {
            Execute(input, output, Span<Complex>.Empty, inverse);
            return;
        }
        var scratch = ArrayPool<Complex>.Shared.Rent(ScratchLength);
        try { Execute(input, output, scratch.AsSpan(0, ScratchLength), inverse); }
        finally { ArrayPool<Complex>.Shared.Return(scratch); }
    }

    /// <summary>
    /// Unnormalised transform with caller-supplied scratch of at least <see cref="ScratchLength"/>
    /// elements — the fully allocation-free entry point.
    /// </summary>
    public unsafe void Execute(ReadOnlySpan<Complex> input, Span<Complex> output, Span<Complex> scratch, bool inverse = false)
    {
        var n = Length;
        if (input.Length != n) throw new ArgumentException($"Input has {input.Length} values; the plan is for {n}.", nameof(input));
        if (output.Length != n) throw new ArgumentException($"Output has {output.Length} values; the plan is for {n}.", nameof(output));
        if (scratch.Length < ScratchLength) throw new ArgumentException($"Scratch needs {ScratchLength} values.", nameof(scratch));
        if (input.Overlaps(output, out var offset) && offset != 0)
            throw new ArgumentException("Input and output must be the same span or not overlap at all.");

        if (n == 1) { output[0] = input[0]; return; }
        if (_inner is not null) { Bluestein(input, output, scratch, inverse); return; }

        fixed (Complex* pin = input)
        fixed (Complex* pout = output)
        fixed (Complex* pscr = scratch)
        {
            var inPtr = (double*)pin;
            var outPtr = (double*)pout;
            var scrPtr = (double*)pscr;
            var passes = _factors.Length;

            // The last pass must land in output. Pass p writes to output when (passes-1-p) is even.
            // In place with an odd pass count, pass 0 would read and write the same buffer, so the
            // input is first moved into scratch.
            double* src = inPtr;
            if (inPtr == outPtr && (passes & 1) == 1)
            {
                Buffer.MemoryCopy(inPtr, scrPtr, (long)n * 16, (long)n * 16);
                src = scrPtr;
            }

            var l1 = 1;
            for (var p = 0; p < passes; p++)
            {
                var dst = ((passes - 1 - p) & 1) == 0 ? outPtr : scrPtr;
                var ip = _factors[p];
                var ido = n / (l1 * ip);
                fixed (double* tw = _twiddles[p])
                {
                    if (SimdEnabled)
                    {
                        if (inverse) RunPass<FftBackwardDirection, FftSimd>(p, ip, ido, l1, src, dst, tw);
                        else RunPass<FftForwardDirection, FftSimd>(p, ip, ido, l1, src, dst, tw);
                    }
                    else
                    {
                        if (inverse) RunPass<FftBackwardDirection, FftScalar>(p, ip, ido, l1, src, dst, tw);
                        else RunPass<FftForwardDirection, FftScalar>(p, ip, ido, l1, src, dst, tw);
                    }
                }
                src = dst;
                l1 *= ip;
            }
            if (src != outPtr) Buffer.MemoryCopy(src, outPtr, (long)n * 16, (long)n * 16);
        }
    }

    /// <summary>In-place unnormalised transform.</summary>
    public void ExecuteInPlace(Span<Complex> data, bool inverse = false) => Execute(data, data, inverse);

    private unsafe void RunPass<D, S>(int p, int ip, int ido, int l1, double* src, double* dst, double* tw)
        where D : struct where S : struct
    {
        switch (ip)
        {
            case 4: FftKernels.Pass4<D, S>(ido, l1, src, dst, tw); break;
            case 2: FftKernels.Pass2<D, S>(ido, l1, src, dst, tw); break;
            case 3: FftKernels.Pass3<D, S>(ido, l1, src, dst, tw); break;
            case 5: FftKernels.Pass5<D, S>(ido, l1, src, dst, tw); break;
            default:
                fixed (double* cs = _genericTables[p])
                    FftKernels.PassGeneric<D, S>(ido, l1, ip, src, dst, tw, cs);
                break;
        }
    }

    /// <summary>
    /// <c>X_k = w_k Σ_j (x_j w_j) conj(w_{k−j})</c> with <c>w_k = exp(−iπ k²/n)</c>; the sum is a
    /// circular convolution of length m ≥ 2n − 1, evaluated with the cached inner plan. The
    /// backward transform is the conjugate of the forward transform of the conjugate.
    /// </summary>
    private void Bluestein(ReadOnlySpan<Complex> input, Span<Complex> output, Span<Complex> scratch, bool inverse)
    {
        var n = Length;
        var inner = _inner!;
        var m = inner.Length;
        var a = scratch[..m];
        var innerScratch = scratch.Slice(m, inner.ScratchLength);
        var chirp = _chirp.AsSpan();
        var kernel = _kernel.AsSpan();

        if (inverse)
            for (var k = 0; k < n; k++) a[k] = Complex.Conjugate(input[k]) * chirp[k];
        else
            for (var k = 0; k < n; k++) a[k] = input[k] * chirp[k];
        a[n..].Clear();

        inner.Execute(a, a, innerScratch, inverse: false);
        FftComplexOps.MultiplyInPlace(a, kernel);
        inner.Execute(a, a, innerScratch, inverse: true);

        if (inverse)
            for (var k = 0; k < n; k++) output[k] = Complex.Conjugate(a[k] * chirp[k]);
        else
            for (var k = 0; k < n; k++) output[k] = a[k] * chirp[k];
    }
}

/// <summary>Element-wise complex helpers used by the FFT module.</summary>
internal static unsafe class FftComplexOps
{
    /// <summary><c>a[i] *= b[i]</c>.</summary>
    public static void MultiplyInPlace(Span<Complex> a, ReadOnlySpan<Complex> b)
    {
        for (var i = 0; i < a.Length; i++) a[i] *= b[i];
    }

    /// <summary>Multiplies every element by a real factor (skipped when the factor is 1).</summary>
    public static void Scale(Span<Complex> a, double factor)
    {
        if (factor == 1.0) return;
        var d = MemoryMarshal.Cast<Complex, double>(a);
        for (var i = 0; i < d.Length; i++) d[i] *= factor;
    }

    /// <summary>Multiplies every element by a real factor (skipped when the factor is 1).</summary>
    public static void Scale(Span<double> a, double factor)
    {
        if (factor == 1.0) return;
        for (var i = 0; i < a.Length; i++) a[i] *= factor;
    }

    /// <summary>Conjugates in place.</summary>
    public static void Conjugate(Span<Complex> a)
    {
        var d = MemoryMarshal.Cast<Complex, double>(a);
        for (var i = 1; i < d.Length; i += 2) d[i] = -d[i];
    }
}
