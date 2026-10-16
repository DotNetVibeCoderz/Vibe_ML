using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A precomputed real-input FFT of one length: real signal of <see cref="Length"/> samples to its
/// <c>n/2 + 1</c> non-redundant bins, and back.
/// </summary>
/// <remarks>
/// <para>
/// For even <c>n</c> the signal is <em>reinterpreted</em> (not copied) as <c>n/2</c> complex
/// numbers <c>z_t = x_{2t} + i x_{2t+1}</c>, transformed with a half-length complex plan, and the
/// two interleaved spectra are separated with one O(n) pass:
/// <c>X_k = E_k + w^k O_k</c>, <c>X_{h−k} = conj(E_k − w^k O_k)</c> where
/// <c>E_k = (Z_k + conj Z_{h−k})/2</c>, <c>O_k = −i (Z_k − conj Z_{h−k})/2</c>. That is roughly
/// half the work of promoting the signal to complex. The inverse runs the same algebra backwards
/// and writes straight into the output buffer, so it needs no temporary of its own.
/// </para>
/// <para>
/// Odd lengths have no such split and use a full-length complex plan internally. Plans are
/// unnormalised: <c>Inverse(Forward(x)) = n·x</c>.
/// </para>
/// </remarks>
public sealed class RfftPlan
{
    private const int CacheLimit = 512;
    private static readonly ConcurrentDictionary<int, RfftPlan> Cache = new();

    private readonly FftPlan _complex;
    private readonly Complex[]? _w; // exp(-2πik/n), k = 0..h/2 (even n only)

    /// <summary>Number of real samples.</summary>
    public int Length { get; }

    /// <summary>Number of spectrum bins, <c>n/2 + 1</c>.</summary>
    public int SpectrumLength => Length / 2 + 1;

    /// <summary>Complex scratch elements an execution needs when the caller supplies it.</summary>
    public int ScratchLength { get; }

    /// <summary>Builds a plan without using the cache.</summary>
    public RfftPlan(int length)
    {
        if (length < 1) throw new ArgumentException($"FFT length must be positive, got {length}.", nameof(length));
        Length = length;
        if ((length & 1) == 0)
        {
            var h = length / 2;
            _complex = FftPlan.Get(h);
            _w = new Complex[h / 2 + 1];
            for (var k = 0; k < _w.Length; k++) _w[k] = FftMath.Twiddle(k, length);
            ScratchLength = _complex.ScratchLength;
        }
        else
        {
            _complex = FftPlan.Get(length);
            ScratchLength = length + _complex.ScratchLength;
        }
    }

    /// <summary>The cached plan for a length; thread-safe.</summary>
    public static RfftPlan Get(int length)
    {
        if (Cache.TryGetValue(length, out var plan)) return plan;
        if (Cache.Count >= CacheLimit) Cache.Clear();
        return Cache.GetOrAdd(length, static n => new RfftPlan(n));
    }

    /// <summary>Forward transform, renting scratch from the shared pool.</summary>
    /// <param name="input"><see cref="Length"/> real samples.</param>
    /// <param name="output"><see cref="SpectrumLength"/> bins.</param>
    public void Forward(ReadOnlySpan<double> input, Span<Complex> output)
    {
        var scratch = ScratchLength == 0 ? null : ArrayPool<Complex>.Shared.Rent(ScratchLength);
        try { Forward(input, output, scratch.AsSpan(0, ScratchLength)); }
        finally { if (scratch is not null) ArrayPool<Complex>.Shared.Return(scratch); }
    }

    /// <summary>Forward transform with caller-supplied scratch (allocation-free).</summary>
    public void Forward(ReadOnlySpan<double> input, Span<Complex> output, Span<Complex> scratch)
    {
        var n = Length;
        if (input.Length != n) throw new ArgumentException($"Input has {input.Length} samples; the plan is for {n}.", nameof(input));
        if (output.Length != SpectrumLength) throw new ArgumentException($"Output needs {SpectrumLength} bins.", nameof(output));
        if (scratch.Length < ScratchLength) throw new ArgumentException($"Scratch needs {ScratchLength} values.", nameof(scratch));

        if (n == 1) { output[0] = new Complex(input[0], 0); return; }

        if ((n & 1) == 1)
        {
            var buffer = scratch[..n];
            for (var i = 0; i < n; i++) buffer[i] = new Complex(input[i], 0);
            _complex.Execute(buffer, buffer, scratch.Slice(n, _complex.ScratchLength), inverse: false);
            buffer[..output.Length].CopyTo(output);
            return;
        }

        var h = n / 2;
        var z = MemoryMarshal.Cast<double, Complex>(input);
        _complex.Execute(z, output[..h], scratch[.._complex.ScratchLength], inverse: false);

        var z0 = output[0];
        output[0] = new Complex(z0.Real + z0.Imaginary, 0);
        output[h] = new Complex(z0.Real - z0.Imaginary, 0);
        var w = _w!;
        int k;
        for (k = 1; k < h - k; k++)
        {
            var zk = output[k];
            var zm = Complex.Conjugate(output[h - k]);
            var e = (zk + zm) * 0.5;
            var d = (zk - zm) * 0.5;
            var o = new Complex(d.Imaginary, -d.Real); // -i d
            var wo = w[k] * o;
            output[k] = e + wo;
            output[h - k] = Complex.Conjugate(e - wo);
        }
        if (k == h - k) output[k] = Complex.Conjugate(output[k]);
    }

    /// <summary>Unnormalised inverse transform (<c>n·x</c>), renting scratch from the shared pool.</summary>
    /// <param name="input"><see cref="SpectrumLength"/> bins. The imaginary parts of bin 0 and
    /// (for even n) bin n/2 are ignored, as numpy and scipy do: a real signal cannot produce them.</param>
    /// <param name="output"><see cref="Length"/> real samples.</param>
    public void Inverse(ReadOnlySpan<Complex> input, Span<double> output)
    {
        var scratch = ScratchLength == 0 ? null : ArrayPool<Complex>.Shared.Rent(ScratchLength);
        try { Inverse(input, output, scratch.AsSpan(0, ScratchLength)); }
        finally { if (scratch is not null) ArrayPool<Complex>.Shared.Return(scratch); }
    }

    /// <summary>Unnormalised inverse with caller-supplied scratch (allocation-free).</summary>
    public void Inverse(ReadOnlySpan<Complex> input, Span<double> output, Span<Complex> scratch)
    {
        var n = Length;
        if (input.Length != SpectrumLength) throw new ArgumentException($"Input needs {SpectrumLength} bins.", nameof(input));
        if (output.Length != n) throw new ArgumentException($"Output needs {n} samples.", nameof(output));
        if (scratch.Length < ScratchLength) throw new ArgumentException($"Scratch needs {ScratchLength} values.", nameof(scratch));

        if (n == 1) { output[0] = input[0].Real; return; }

        if ((n & 1) == 1)
        {
            var buffer = scratch[..n];
            buffer[0] = new Complex(input[0].Real, 0);
            for (var j = 1; j < input.Length; j++)
            {
                buffer[j] = input[j];
                buffer[n - j] = Complex.Conjugate(input[j]);
            }
            _complex.Execute(buffer, buffer, scratch.Slice(n, _complex.ScratchLength), inverse: true);
            for (var i = 0; i < n; i++) output[i] = buffer[i].Real;
            return;
        }

        var h = n / 2;
        var zs = MemoryMarshal.Cast<double, Complex>(output);
        var x0 = input[0].Real;
        var xh = input[h].Real;
        zs[0] = new Complex(x0 + xh, x0 - xh);
        var w = _w!;
        int k;
        for (k = 1; k < h - k; k++)
        {
            var xk = input[k];
            var xm = Complex.Conjugate(input[h - k]);
            var s = xk + xm;
            var d = xk - xm;
            var t = Complex.Conjugate(w[k]) * d;
            t = new Complex(-t.Imaginary, t.Real); // i * conj(w^k) * d
            zs[k] = s + t;
            zs[h - k] = Complex.Conjugate(s - t);
        }
        if (k == h - k) zs[k] = 2.0 * Complex.Conjugate(input[k]);

        _complex.Execute(zs, zs, scratch[.._complex.ScratchLength], inverse: true);
    }
}
