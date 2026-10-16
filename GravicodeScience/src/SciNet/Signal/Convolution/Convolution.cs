using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>
    /// Turns the vectorised direct-sum kernels (convolution, upfirdn, savgol) on or off. Off runs
    /// the scalar reference loops, which exist so the two can be tested against each other.
    /// </summary>
    public static bool DirectKernelSimd
    {
        get => SignalConvKernels.SimdEnabled;
        set => SignalConvKernels.SimdEnabled = value;
    }

    // Cost model for method = Auto, fitted to Release timings on a 4-core laptop (see
    // ChooseConvMethod remarks). Seconds; only the ratios matter.
    private const double DirectSerialPerOutput = 45e-9, DirectSerialPerMac = 0.9e-9;
    private const double DirectParallelPerOutput = 6e-9, DirectParallelPerMac = 0.07e-9;
    private const double FftPerNLogN = 2.8e-9, FftFixed = 150e-6;
    private const double OaSmallBlockPerNLogN = 0.3e-9, OaLargeBlockPerNLogN = 1.5e-9, OaPerBlock = 3e-6;

    // ================================================================== 1-D convolve/correlate

    /// <summary>
    /// Convolution of two 1-D sequences (<c>scipy.signal.convolve</c>).
    /// </summary>
    /// <remarks>
    /// <para><see cref="ConvolveMode.Same"/> is centred with respect to the full output, i.e. it
    /// starts at <c>(m − 1)/2</c> (integer division); <see cref="ConvolveMode.Valid"/> swaps the
    /// inputs when the second is longer, as scipy does.</para>
    /// <para>The direct path is a SIMD dot product per output, parallel above ~2.6·10⁵ multiply-adds;
    /// the FFT path pads to the next 5-smooth length. With <see cref="ConvolveMethod.Auto"/> a
    /// NaN or infinity anywhere forces the direct path — through an FFT it would contaminate
    /// every output, which scipy only warns about.</para>
    /// </remarks>
    /// <param name="in1">First input.</param>
    /// <param name="in2">Second input.</param>
    /// <param name="mode">Output size.</param>
    /// <param name="method">Direct, FFT, or chosen automatically.</param>
    public static double[] Convolve(double[] in1, double[] in2, ConvolveMode mode = ConvolveMode.Full, ConvolveMethod method = ConvolveMethod.Auto)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Length == 0 || in2.Length == 0) return [];
        if (mode == ConvolveMode.Valid && in2.Length > in1.Length) (in1, in2) = (in2, in1);
        if (method == ConvolveMethod.Auto) method = ChooseConvMethod(in1, in2, mode);
        var n = in1.Length;
        var m = in2.Length;
        var (start, count) = ConvRange1D(n, m, mode);
        if (method == ConvolveMethod.Fft)
        {
            // Overlap-add whenever it is the cheaper FFT strategy: identical up to rounding,
            // and its small blocks stay in cache where one huge transform does not.
            var full = OaCost(n, m) < PlainFftCost((long)n + m - 1) ? OverlapAddFull(in1, in2) : SignalConvKernels.FftConvolveFull(in1, in2);
            return start == 0 && count == full.Length ? full : full.AsSpan(start, count).ToArray();
        }
        var y = new double[count];
        SignalConvKernels.ConvolveRangeParallel(in1, SignalConvKernels.Reversed(in2), start, y, 0, count);
        return y;
    }

    /// <summary>
    /// Cross-correlation of two 1-D sequences (<c>scipy.signal.correlate</c>):
    /// <c>z[k] = Σ_l in1[l] · in2[l − k + N − 1]</c>, which is the convolution of
    /// <paramref name="in1"/> with <paramref name="in2"/> reversed — modes and lags follow from that.
    /// </summary>
    /// <param name="in1">First input.</param>
    /// <param name="in2">Second input.</param>
    /// <param name="mode">Output size; see <see cref="CorrelationLags"/> for the matching lags.</param>
    /// <param name="method">Direct, FFT, or chosen automatically.</param>
    public static double[] Correlate(double[] in1, double[] in2, ConvolveMode mode = ConvolveMode.Full, ConvolveMethod method = ConvolveMethod.Auto)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Length == 0 || in2.Length == 0) return [];
        return Convolve(in1, SignalConvKernels.Reversed(in2), mode, method);
    }

    /// <summary>
    /// The lag index for each output of <see cref="Correlate(double[], double[], ConvolveMode, ConvolveMethod)"/>
    /// (<c>scipy.signal.correlation_lags</c>).
    /// </summary>
    /// <param name="in1Len">Length of the first input.</param>
    /// <param name="in2Len">Length of the second input.</param>
    /// <param name="mode">The mode the correlation was computed with.</param>
    public static int[] CorrelationLags(int in1Len, int in2Len, ConvolveMode mode = ConvolveMode.Full)
    {
        if (in1Len < 1 || in2Len < 1) throw new ArgumentException("Lengths must be positive.");
        switch (mode)
        {
            case ConvolveMode.Full:
                return Range(-in2Len + 1, in1Len);
            case ConvolveMode.Same:
                {
                    var lags = Range(-in2Len + 1, in1Len);
                    var mid = lags.Length / 2;
                    var bound = in1Len / 2;
                    var lo = mid - bound;
                    var hi = in1Len % 2 == 0 ? mid + bound : mid + bound + 1;
                    lo = Math.Clamp(lo, 0, lags.Length);
                    hi = Math.Clamp(hi, lo, lags.Length);
                    return lags[lo..hi];
                }
            default:
                {
                    var bound = in1Len - in2Len;
                    return bound >= 0 ? Range(0, bound + 1) : Range(bound, 1);
                }
        }

        static int[] Range(int from, int to)
        {
            var r = new int[Math.Max(0, to - from)];
            for (var i = 0; i < r.Length; i++) r[i] = from + i;
            return r;
        }
    }

    /// <summary>
    /// Which method <see cref="ConvolveMethod.Auto"/> would use (<c>scipy.signal.choose_conv_method</c>
    /// without <c>measure</c>).
    /// </summary>
    /// <remarks>
    /// <para>scipy's constants were fitted to numpy's scalar <c>convolve</c> and pocketfft; this
    /// engine's direct path is a vectorised, parallel dot product and its FFT a different
    /// implementation, so the model was re-fitted to Release timings (4-core laptop): direct
    /// ≈ 45 ns per output + 0.9 ns per multiply-add serially, ≈ 6 ns + 0.07 ns once it runs in
    /// parallel; one whole-output FFT ≈ 2.8 ns per <c>N log₂ N</c> + 150 µs (cold-cache cost of
    /// a single call, well above a tight loop's); overlap-add (1-D, short kernel) ≈ 3 µs + 0.3 ns
    /// per <c>L log₂ L</c> per block while blocks stay ≤ 8192. A 1-D kernel of ≤ 32 taps is always
    /// direct. It is a
    /// rough model — the FFT engine's speed varies by a factor of two between neighbouring
    /// 5-smooth lengths — so near the crossover either choice is within ~2× of the best. The FFT branch
    /// costs the cheaper of whole-output and overlap-add, and <see cref="ConvolveMethod.Fft"/>
    /// then runs that one.</para>
    /// <para>Non-finite inputs always go direct, because an FFT spreads one NaN over the whole
    /// output.</para>
    /// </remarks>
    public static ConvolveMethod ChooseConvMethod(double[] in1, double[] in2, ConvolveMode mode = ConvolveMode.Full)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (!AllFinite(in1) || !AllFinite(in2)) return ConvolveMethod.Direct;
        return FftFaster([in1.Length], [in2.Length], mode) ? ConvolveMethod.Fft : ConvolveMethod.Direct;
    }

    /// <summary>n-D form of <see cref="ChooseConvMethod(double[], double[], ConvolveMode)"/>.</summary>
    public static ConvolveMethod ChooseConvMethod(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (!AllFinite(in1.AsContiguous().AsSpan()) || !AllFinite(in2.AsContiguous().AsSpan())) return ConvolveMethod.Direct;
        return FftFaster(in1.Shape.ToArray(), in2.Shape.ToArray(), mode) ? ConvolveMethod.Fft : ConvolveMethod.Direct;
    }

    /// <summary>
    /// FFT convolution (<c>scipy.signal.fftconvolve</c>) of 1-D sequences: one real FFT of the
    /// whole padded output, at an even 5-smooth length. <see cref="Convolve(double[], double[], ConvolveMode, ConvolveMethod)"/>
    /// with <see cref="ConvolveMethod.Fft"/> switches to overlap-add when that is cheaper.
    /// </summary>
    public static double[] Fftconvolve(double[] in1, double[] in2, ConvolveMode mode = ConvolveMode.Full)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Length == 0 || in2.Length == 0) return [];
        if (mode == ConvolveMode.Valid && in2.Length > in1.Length) (in1, in2) = (in2, in1);
        var (start, count) = ConvRange1D(in1.Length, in2.Length, mode);
        var full = SignalConvKernels.FftConvolveFull(in1, in2);
        return start == 0 && count == full.Length ? full : full.AsSpan(start, count).ToArray();
    }

    /// <summary>
    /// Overlap-add convolution (<c>scipy.signal.oaconvolve</c>): the long input is cut into blocks
    /// whose FFT length minimises total work for the short one (scipy's Lambert-W optimum), each
    /// block is convolved through a cached real-FFT plan, and the tails are added. Equal to
    /// <see cref="Fftconvolve(double[], double[], ConvolveMode)"/> up to rounding; much faster when
    /// one input is far longer than the other. Blocks run in parallel (even blocks, then odd, so
    /// no two writers ever share an output sample).
    /// </summary>
    public static double[] Oaconvolve(double[] in1, double[] in2, ConvolveMode mode = ConvolveMode.Full)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Length == 0 || in2.Length == 0) return [];
        if (mode == ConvolveMode.Valid && in2.Length > in1.Length) (in1, in2) = (in2, in1);
        var (start, count) = ConvRange1D(in1.Length, in2.Length, mode);
        var full = OverlapAddFull(in1, in2);
        return start == 0 && count == full.Length ? full : full.AsSpan(start, count).ToArray();
    }

    private static double[] OverlapAddFull(double[] a, double[] b)
    {
        var (x, h) = a.Length >= b.Length ? (a, b) : (b, a);
        var n = x.Length;
        var m = h.Length;
        if (m == n || m == 1 || m >= n / 2.0) return SignalConvKernels.FftConvolveFull(a, b);
        var overlap = m - 1;
        var opt = -overlap * Special.LambertW(-1.0 / (2 * Math.E * overlap), -1);
        var block = SignalConvKernels.EvenFastLength((int)Math.Ceiling(opt));
        if (block >= n) return SignalConvKernels.FftConvolveFull(a, b);
        var step = block - m + 1;
        var outLen = n + m - 1;
        var y = new double[outLen];
        var plan = RfftPlan.Get(block);
        var bins = plan.SpectrumLength;
        var hBuf = new double[block];
        h.CopyTo(hBuf, 0);
        var hf = new Complex[bins];
        plan.Forward(hBuf, hf);
        var scale = 1.0 / block;
        for (var k = 0; k < bins; k++) hf[k] *= scale;
        var nBlocks = (n + step - 1) / step;

        void Block(int bi, double[] buf, Complex[] spec, Complex[] scratch)
        {
            var s = bi * step;
            var len = Math.Min(step, n - s);
            Array.Clear(buf);
            Array.Copy(x, s, buf, 0, len);
            plan.Forward(buf, spec, scratch);
            for (var k = 0; k < spec.Length; k++) spec[k] *= hf[k];
            plan.Inverse(spec, buf, scratch);
            var outN = Math.Min(len + m - 1, outLen - s);
            for (var i = 0; i < outN; i++) y[s + i] += buf[i];
        }

        if ((long)n * Math.Log2(block) < 1 << 20 || nBlocks < 4)
        {
            var buf = new double[block];
            var spec = new Complex[bins];
            var scratch = new Complex[Math.Max(1, plan.ScratchLength)];
            for (var bi = 0; bi < nBlocks; bi++) Block(bi, buf, spec, scratch);
            return y;
        }
        for (var parity = 0; parity < 2; parity++)
        {
            var count = (nBlocks - parity + 1) / 2;
            Parallel.For(0, count,
                () => (new double[block], new Complex[bins], new Complex[Math.Max(1, plan.ScratchLength)]),
                (j, _, st) => { Block(2 * j + parity, st.Item1, st.Item2, st.Item3); return st; },
                _ => { });
        }
        return y;
    }

    /// <summary>
    /// Divides <paramref name="signal"/> by <paramref name="divisor"/> as polynomials — inverse
    /// filtering (<c>scipy.signal.deconvolve</c>). <c>signal = convolve(divisor, quotient) + remainder</c>.
    /// </summary>
    /// <remarks>The quotient comes from running <c>lfilter(signal, divisor, impulse)</c>, so a
    /// divisor whose roots lie outside the unit circle makes it grow geometrically — that is the
    /// mathematics, not a bug.</remarks>
    public static (double[] Quotient, double[] Remainder) Deconvolve(double[] signal, double[] divisor)
    {
        ArgumentNullException.ThrowIfNull(signal);
        ArgumentNullException.ThrowIfNull(divisor);
        if (signal.Length == 0) throw new ArgumentException("Parameter signal must be a non-empty 1-D array.", nameof(signal));
        if (divisor.Length == 0) throw new ArgumentException("Parameter divisor must be a non-empty 1-D array.", nameof(divisor));
        if (divisor[0] == 0) throw new ArgumentException("The leading divisor coefficient must be non-zero.", nameof(divisor));
        var n = signal.Length;
        var d = divisor.Length;
        if (d > n) return ([], (double[])signal.Clone());
        var input = new double[n - d + 1];
        input[0] = 1;
        var quot = Lfilter(signal, divisor, input);
        var conv = Convolve(divisor, quot, ConvolveMode.Full, ConvolveMethod.Direct);
        var rem = new double[n];
        for (var i = 0; i < n; i++) rem[i] = signal[i] - conv[i];
        return (quot, rem);
    }

    // ================================================================== n-D convolve/correlate

    /// <summary>
    /// n-D convolution (<c>scipy.signal.convolve</c>); both inputs must have the same rank.
    /// </summary>
    public static NdArray Convolve(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full, ConvolveMethod method = ConvolveMethod.Auto)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Rank != in2.Rank) throw new ArgumentException("volume and kernel should have the same dimensionality.");
        if (in1.Size == 0 || in2.Size == 0) return new NdArray(Array.Empty<double>(), 0);
        if (SwapNeeded(mode, in1.Shape, in2.Shape)) (in1, in2) = (in2, in1);
        if (method == ConvolveMethod.Auto) method = ChooseConvMethod(in1, in2, mode);
        var s1 = in1.Shape.ToArray();
        var s2 = in2.Shape.ToArray();
        NdArray full;
        if (method == ConvolveMethod.Fft)
        {
            if (in1.Rank == 1)
                full = new NdArray(SignalConvKernels.FftConvolveFull(in1.AsContiguous().ToArray(), in2.AsContiguous().ToArray()), s1[0] + s2[0] - 1);
            else full = Fft.Convolve(in1, in2);
        }
        else full = DirectFullNd(in1, in2);
        return ApplyConvMode(full, s1, s2, mode, correlate2dSame: false);
    }

    /// <summary>
    /// n-D cross-correlation (<c>scipy.signal.correlate</c>): convolution with the second input
    /// flipped along every axis.
    /// </summary>
    public static NdArray Correlate(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full, ConvolveMethod method = ConvolveMethod.Auto)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        return Convolve(in1, FlipAll(in2), mode, method);
    }

    /// <summary>
    /// n-D FFT convolution over <paramref name="axes"/> (default all) — <c>scipy.signal.fftconvolve</c>.
    /// Axes that are not convolved must have equal lengths in both inputs (no broadcasting).
    /// </summary>
    public static NdArray Fftconvolve(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Rank != in2.Rank) throw new ArgumentException("in1 and in2 should have the same dimensionality.");
        if (in1.Size == 0 || in2.Size == 0) return new NdArray(Array.Empty<double>(), 0);
        var rank = in1.Rank;
        var ax = (axes ?? Enumerable.Range(0, rank).ToArray()).Select(a => a < 0 ? a + rank : a).ToArray();
        if (axes is not null && ax.Length == 0) throw new ArgumentException("When provided, axes cannot be empty.", nameof(axes));
        if (SwapNeeded(mode, in1.Shape, in2.Shape, ax)) (in1, in2) = (in2, in1);
        var s1 = in1.Shape.ToArray();
        var s2 = in2.Shape.ToArray();
        var full = Fft.Convolve(in1, in2, ax);
        return ApplyConvMode(full, s1, s2, mode, false, ax);
    }

    /// <summary>
    /// n-D overlap-add convolution (<c>scipy.signal.oaconvolve</c>). A 1-D input (or an array
    /// convolved along a single axis) uses true overlap-add; otherwise this computes the same
    /// result through <see cref="Fftconvolve(NdArray, NdArray, ConvolveMode, int[])"/> — identical
    /// up to rounding, without the multi-axis block bookkeeping.
    /// </summary>
    public static NdArray Oaconvolve(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full, int[]? axes = null)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Rank == 1 && in2.Rank == 1)
        {
            var r = Oaconvolve(in1.AsContiguous().ToArray(), in2.AsContiguous().ToArray(), mode);
            return new NdArray(r, r.Length);
        }
        return Fftconvolve(in1, in2, mode, axes);
    }

    // ================================================================== 2-D with boundaries

    /// <summary>
    /// 2-D convolution with an edge rule (<c>scipy.signal.convolve2d</c>). The first input is
    /// extended by <paramref name="boundary"/>, so with <see cref="Convolve2dBoundary.Fill"/> and a
    /// non-zero <paramref name="fillvalue"/> the result differs from zero-padded convolution.
    /// </summary>
    /// <param name="in1">First 2-D input.</param>
    /// <param name="in2">Second 2-D input (the kernel).</param>
    /// <param name="mode">Output size.</param>
    /// <param name="boundary">Edge extension of <paramref name="in1"/>.</param>
    /// <param name="fillvalue">Pad value for <see cref="Convolve2dBoundary.Fill"/>.</param>
    public static NdArray Convolve2d(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full,
        Convolve2dBoundary boundary = Convolve2dBoundary.Fill, double fillvalue = 0)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Rank != 2 || in2.Rank != 2) throw new ArgumentException("convolve2d inputs must both be 2-D arrays.");
        if (SwapNeeded(mode, in1.Shape, in2.Shape)) (in1, in2) = (in2, in1);
        var h = in2.AsContiguous().ToArray();
        return Conv2dCore(in1, h, in2.Shape[0], in2.Shape[1], mode, boundary, fillvalue, sameFloor: true);
    }

    /// <summary>
    /// 2-D cross-correlation with an edge rule (<c>scipy.signal.correlate2d</c>).
    /// </summary>
    /// <remarks>
    /// For an even-sized kernel, <see cref="ConvolveMode.Same"/> here is <em>not</em> the same as
    /// <see cref="Correlate(NdArray, NdArray, ConvolveMode, ConvolveMethod)"/>: scipy's 2-D routine
    /// centres the window at <c>K/2</c>, the n-D one at <c>(K − 1)/2</c>. Both are reproduced.
    /// </remarks>
    public static NdArray Correlate2d(NdArray in1, NdArray in2, ConvolveMode mode = ConvolveMode.Full,
        Convolve2dBoundary boundary = Convolve2dBoundary.Fill, double fillvalue = 0)
    {
        ArgumentNullException.ThrowIfNull(in1);
        ArgumentNullException.ThrowIfNull(in2);
        if (in1.Rank != 2 || in2.Rank != 2) throw new ArgumentException("correlate2d inputs must both be 2-D arrays.");
        var swapped = SwapNeeded(mode, in1.Shape, in2.Shape);
        if (swapped) (in1, in2) = (in2, in1);
        var h = in2.AsContiguous().ToArray();
        Array.Reverse(h); // flipping both axes of a row-major matrix reverses its buffer
        var r = Conv2dCore(in1, h, in2.Shape[0], in2.Shape[1], mode, boundary, fillvalue, sameFloor: false);
        if (swapped)
        {
            var d = r.AsContiguous().ToArray();
            Array.Reverse(d);
            r = new NdArray(d, r.Shape[0], r.Shape[1]);
        }
        return r;
    }

    private static NdArray Conv2dCore(NdArray in1, double[] h, int kr, int kc, ConvolveMode mode, Convolve2dBoundary boundary, double fill, bool sameFloor)
    {
        var mr = in1.Shape[0];
        var mc = in1.Shape[1];
        var a = in1.AsContiguous().ToArray();
        if (a.Length == 0 || h.Length == 0) return new NdArray(Array.Empty<double>(), 0, 0);
        int r0, c0, rows, cols;
        switch (mode)
        {
            case ConvolveMode.Full: r0 = 0; c0 = 0; rows = mr + kr - 1; cols = mc + kc - 1; break;
            case ConvolveMode.Same:
                r0 = sameFloor ? (kr - 1) / 2 : kr / 2;
                c0 = sameFloor ? (kc - 1) / 2 : kc / 2;
                rows = mr; cols = mc; break;
            default: r0 = kr - 1; c0 = kc - 1; rows = mr - kr + 1; cols = mc - kc + 1; break;
        }
        // Padded window P[a, b] = X(a + r0 − kr + 1, b + c0 − kc + 1); output = valid correlation
        // of P with the reversed kernel, one SIMD dot product per kernel row.
        var pr = rows + kr - 1;
        var pc = cols + kc - 1;
        var p = new double[(long)pr * pc];
        for (var i = 0; i < pr; i++)
        {
            var si = i + r0 - kr + 1;
            var ri = MapIndex(si, mr, boundary);
            for (var j = 0; j < pc; j++)
            {
                var sj = j + c0 - kc + 1;
                var rj = MapIndex(sj, mc, boundary);
                p[(long)i * pc + j] = ri < 0 || rj < 0 ? fill : a[ri * mc + rj];
            }
        }
        var hr = (double[])h.Clone();
        Array.Reverse(hr); // correlation with the reversed kernel
        var y = new double[rows * cols];
        void Row(int i)
        {
            var yr = y.AsSpan(i * cols, cols);
            for (var q = 0; q < kr; q++)
            {
                var krow = hr.AsSpan(q * kc, kc);
                var prow = p.AsSpan((i + q) * pc, pc);
                for (var j = 0; j < cols; j++) yr[j] += SignalConvKernels.Dot(prow.Slice(j, kc), krow);
            }
        }
        if ((long)rows * cols * kr * kc > 1 << 18 && rows > 1) Parallel.For(0, rows, Row);
        else for (var i = 0; i < rows; i++) Row(i);
        return new NdArray(y, rows, cols);
    }

    private static int MapIndex(int i, int n, Convolve2dBoundary boundary)
    {
        if (i >= 0 && i < n) return i;
        switch (boundary)
        {
            case Convolve2dBoundary.Wrap:
                return ((i % n) + n) % n;
            case Convolve2dBoundary.Symm:
                {
                    var p = 2 * n;
                    var k = ((i % p) + p) % p;
                    return k < n ? k : p - 1 - k;
                }
            default:
                return -1;
        }
    }

    // ================================================================== helpers

    private static (int Start, int Count) ConvRange1D(int n, int m, ConvolveMode mode) => mode switch
    {
        ConvolveMode.Full => (0, n + m - 1),
        ConvolveMode.Same => ((m - 1) / 2, n),
        ConvolveMode.Valid => (m - 1, n - m + 1),
        _ => throw new ArgumentException("Acceptable mode flags are Valid, Same, or Full.", nameof(mode)),
    };

    private static bool SwapNeeded(ConvolveMode mode, ReadOnlySpan<int> s1, ReadOnlySpan<int> s2, int[]? axes = null)
    {
        if (mode != ConvolveMode.Valid || s1.Length == 0) return false;
        bool ok1 = true, ok2 = true;
        for (var i = 0; i < s1.Length; i++)
        {
            if (axes is not null && Array.IndexOf(axes, i) < 0) continue;
            ok1 &= s1[i] >= s2[i];
            ok2 &= s2[i] >= s1[i];
        }
        if (!(ok1 || ok2))
            throw new ArgumentException("For 'valid' mode, one must be at least as large as the other in every dimension.");
        return !ok1;
    }

    private static bool FftFaster(int[] s1, int[] s2, ConvolveMode mode)
    {
        double directOps, outputs = 1;
        long fullN = 1;
        for (var i = 0; i < s1.Length; i++) fullN *= s1[i] + s2[i] - 1;
        if (s1.Length == 1)
        {
            double a = s1[0], b = s2[0];
            directOps = mode switch
            {
                ConvolveMode.Full => a * b,
                ConvolveMode.Valid => b >= a ? (b - a + 1) * a : (a - b + 1) * b,
                _ => a < b ? a * b : a * b - Math.Floor(b / 2) * Math.Floor((b + 1) / 2),
            };
            outputs = mode switch
            {
                ConvolveMode.Full => a + b - 1,
                ConvolveMode.Valid => Math.Abs(a - b) + 1,
                _ => a,
            };
            // A kernel this short is a handful of SIMD lanes per output; no FFT beats it.
            if (Math.Min(a, b) <= 32) return false;
            var fft = Math.Min(PlainFftCost(fullN), OaCost(Math.Max(s1[0], s2[0]), Math.Min(s1[0], s2[0])));
            return fft < DirectCost(outputs, directOps);
        }
        double p1 = 1, p2 = 1;
        for (var i = 0; i < s1.Length; i++)
        {
            p1 *= s1[i];
            p2 *= s2[i];
            outputs *= mode switch
            {
                ConvolveMode.Full => s1[i] + s2[i] - 1,
                ConvolveMode.Valid => Math.Abs(s1[i] - s2[i]) + 1,
                _ => s1[i],
            };
        }
        directOps = mode == ConvolveMode.Same ? p1 * p2 : Math.Min(p1, p2) * outputs;
        // n-D FFTs are the generic multi-axis path (no overlap-add); the row-wise direct path
        // only parallelises inside each row pair, so cost it serially.
        return PlainFftCost(fullN) < outputs * DirectSerialPerOutput + directOps * DirectSerialPerMac;
    }

    private static double DirectCost(double outputs, double macs)
        => macs < 1 << 18
            ? outputs * DirectSerialPerOutput + macs * DirectSerialPerMac
            : outputs * DirectParallelPerOutput + macs * DirectParallelPerMac;

    private static double PlainFftCost(long fullN)
    {
        var n = (double)Math.Max(2, fullN);
        return FftPerNLogN * n * Math.Log2(n) + FftFixed;
    }

    /// <summary>Estimated overlap-add time for a long input of n and a short one of m (∞ when OA does not apply).</summary>
    private static double OaCost(int n, int m)
    {
        if (m <= 1 || m == n || m >= n / 2.0) return double.PositiveInfinity;
        var overlap = m - 1;
        var opt = -overlap * Special.LambertW(-1.0 / (2 * Math.E * overlap), -1);
        if (!double.IsFinite(opt)) return double.PositiveInfinity;
        double block = SignalConvKernels.EvenFastLength((int)Math.Min(int.MaxValue / 4, Math.Ceiling(opt)));
        if (block >= n) return double.PositiveInfinity;
        var step = block - m + 1;
        var per = block <= 8192 ? OaSmallBlockPerNLogN : OaLargeBlockPerNLogN;
        return Math.Ceiling(n / step) * (block * Math.Log2(block) * per + OaPerBlock) + FftFixed;
    }

    private static bool AllFinite(ReadOnlySpan<double> a)
    {
        foreach (var v in a) if (!double.IsFinite(v)) return false;
        return true;
    }

    private static NdArray FlipAll(NdArray a)
    {
        var d = a.AsContiguous().ToArray();
        Array.Reverse(d); // reversing a row-major buffer flips every axis at once
        return new NdArray(d, a.Shape.ToArray());
    }

    /// <summary>Full n-D direct convolution, accumulated one last-axis row pair at a time.</summary>
    private static NdArray DirectFullNd(NdArray in1, NdArray in2)
    {
        var rank = in1.Rank;
        var s1 = in1.Shape.ToArray();
        var s2 = in2.Shape.ToArray();
        var so = new int[rank];
        for (var i = 0; i < rank; i++) so[i] = s1[i] + s2[i] - 1;
        var a = in1.AsContiguous().ToArray();
        var b = in2.AsContiguous().ToArray();
        var outSize = 1;
        foreach (var v in so) outSize *= v;
        var y = new double[outSize];
        var n1 = s1[rank - 1];
        var n2 = s2[rank - 1];
        var no = so[rank - 1];
        var rows1 = a.Length / n1;
        var rows2 = b.Length / n2;
        // Strides of the leading (row-index) dims.
        var lead = rank - 1;
        var st1 = RowStrides(s1, lead);
        var st2 = RowStrides(s2, lead);
        var sto = RowStrides(so, lead);
        var tmp = new double[no];
        for (var r2 = 0; r2 < rows2; r2++)
        {
            var hRow = b.AsSpan(r2 * n2, n2);
            var hRev = SignalConvKernels.Reversed(hRow);
            for (var r1 = 0; r1 < rows1; r1++)
            {
                var ro = 0;
                for (var d = 0; d < lead; d++)
                    ro += ((r1 / st1[d]) % s1[d] + (r2 / st2[d]) % s2[d]) * sto[d];
                SignalConvKernels.ConvolveRange(a.AsSpan(r1 * n1, n1), hRev, 0, y.AsSpan(ro * no, no), accumulate: true);
            }
        }
        return new NdArray(y, so);
    }

    private static int[] RowStrides(int[] shape, int lead)
    {
        var st = new int[lead];
        var acc = 1;
        for (var d = lead - 1; d >= 0; d--) { st[d] = acc; acc *= shape[d]; }
        return st;
    }

    private static NdArray ApplyConvMode(NdArray full, int[] s1, int[] s2, ConvolveMode mode, bool correlate2dSame, int[]? axes = null)
    {
        if (mode == ConvolveMode.Full) return full;
        var rank = s1.Length;
        var shape = full.Shape.ToArray();
        var target = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            var conv = axes is null || Array.IndexOf(axes, i) >= 0;
            target[i] = mode == ConvolveMode.Same ? s1[i] : conv ? s1[i] - s2[i] + 1 : shape[i];
        }
        var start = new int[rank];
        for (var i = 0; i < rank; i++) start[i] = (shape[i] - target[i]) / 2;
        return CropNd(full, start, target);
    }

    private static NdArray CropNd(NdArray src, int[] start, int[] size)
    {
        var shape = src.Shape.ToArray();
        var data = src.AsContiguous().ToArray();
        var rank = shape.Length;
        var total = 1;
        foreach (var v in size) total *= v;
        var dst = new double[total];
        if (total == 0) return new NdArray(dst, size);
        var srcStride = new int[rank];
        var acc = 1;
        for (var d = rank - 1; d >= 0; d--) { srcStride[d] = acc; acc *= shape[d]; }
        var last = size[rank - 1];
        var rows = total / last;
        var idx = new int[rank];
        for (var r = 0; r < rows; r++)
        {
            var rem = r;
            var off = start[rank - 1];
            for (var d = rank - 2; d >= 0; d--)
            {
                idx[d] = rem % size[d];
                rem /= size[d];
                off += (idx[d] + start[d]) * srcStride[d];
            }
            Array.Copy(data, off, dst, r * last, last);
        }
        return new NdArray(dst, size);
    }
}
