using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Fft
{
    // ---------------------------------------------------------------- FFTLog (Hamilton 2000)

    /// <summary>
    /// Fast Hankel transform of a logarithmically spaced sequence — scipy's <c>fht</c>, Hamilton's
    /// FFTLog: <c>A(k) = ∫ a(r) J_μ(kr) k dr</c> for a periodic sequence a_j = a(r_j),
    /// r_j = r_c exp((j − j_c) dln), returning A at k_j = k_c exp((j − j_c) dln), where
    /// j_c = (n − 1)/2 and k_c r_c = exp(<paramref name="offset"/>).
    /// </summary>
    /// <remarks>
    /// The transform is a convolution in log r, done with one real FFT, a multiplication by the
    /// coefficients u_m = 2^(2q)·Γ((μ+1+q+2πi m/(n dln))/2)/Γ((μ+1−q−…)/2)·(2/k_c r_c)^(2πi m/(n dln)),
    /// and an inverse FFT. A non-zero <paramref name="bias"/> q applies the power-law bias of
    /// Hamilton's eq. (B5). Choose <paramref name="offset"/> with <see cref="Fhtoffset"/> to avoid
    /// ringing. The sequence is treated as periodic, so the input should decay at both ends.
    /// </remarks>
    /// <param name="a">Real periodic input on a log-spaced grid.</param>
    /// <param name="dln">Uniform logarithmic spacing.</param>
    /// <param name="mu">Order of the Hankel transform (any real).</param>
    /// <param name="offset">ln(k_c r_c).</param>
    /// <param name="bias">Power-law bias q.</param>
    public static double[] Fht(double[] a, double dln, double mu, double offset = 0.0, double bias = 0.0)
    {
        ArgumentNullException.ThrowIfNull(a);
        var n = a.Length;
        if (n == 0) return [];
        var x = (double[])a.Clone();
        var jc = (n - 1) / 2.0;
        if (bias != 0) for (var j = 0; j < n; j++) x[j] *= Math.Exp(-bias * (j - jc) * dln);
        var u = FhtCoefficients(n, dln, mu, offset, bias, inverse: false);
        var res = Fhtq(x, u, inverse: false);
        if (bias != 0) for (var j = 0; j < n; j++) res[j] *= Math.Exp(-bias * ((j - jc) * dln + offset));
        return res;
    }

    /// <summary>
    /// Inverse fast Hankel transform — scipy's <c>ifht</c>: recovers a from
    /// <see cref="Fht"/>(a, dln, mu, offset, bias) up to the periodicity of the sequence.
    /// </summary>
    public static double[] Ifht(double[] A, double dln, double mu, double offset = 0.0, double bias = 0.0)
    {
        ArgumentNullException.ThrowIfNull(A);
        var n = A.Length;
        if (n == 0) return [];
        var x = (double[])A.Clone();
        var jc = (n - 1) / 2.0;
        if (bias != 0) for (var j = 0; j < n; j++) x[j] *= Math.Exp(bias * ((j - jc) * dln + offset));
        var u = FhtCoefficients(n, dln, mu, offset, bias, inverse: true);
        var res = Fhtq(x, u, inverse: true);
        if (bias != 0) for (var j = 0; j < n; j++) res[j] /= Math.Exp(-bias * (j - jc) * dln);
        return res;
    }

    /// <summary>
    /// The optimal offset ln(k_c r_c) for <see cref="Fht"/> — scipy's <c>fhtoffset</c>: the value
    /// nearest <paramref name="initial"/> that makes the transform coefficients real at the Nyquist
    /// frequency (Hamilton 2000, eq. 3.4), which minimises ringing.
    /// </summary>
    public static double Fhtoffset(double dln, double mu, double initial = 0.0, double bias = 0.0)
    {
        var lnkr = initial;
        var q = bias;
        var xp = (mu + 1 + q) / 2;
        var xm = (mu + 1 - q) / 2;
        var y = Math.PI / (2 * dln);
        var zp = FftLogGamma(new Complex(xp, y));
        var zm = FftLogGamma(new Complex(xm, y));
        var arg = (Math.Log(2) - lnkr) / dln + (zp.Imaginary + zm.Imaginary) / Math.PI;
        return lnkr + (arg - Math.Round(arg, MidpointRounding.ToEven)) * dln;
    }

    private static Complex[] FhtCoefficients(int n, double dln, double mu, double offset, double bias, bool inverse)
    {
        var lnkr = offset;
        var q = bias;
        var xp = (mu + 1 + q) / 2;
        var xm = (mu + 1 - q) / 2;
        var m = n / 2 + 1;
        var u = new Complex[m];
        var ln2 = Math.Log(2);
        for (var i = 0; i < m; i++)
        {
            var y = m == 1 ? 0 : Math.PI * (n / 2) / (n * dln) * i / (m - 1);
            var v = FftLogGamma(new Complex(xm, y));
            var w = FftLogGamma(new Complex(xp, y));
            var re = w.Real - v.Real + ln2 * q;
            var im = w.Imaginary + v.Imaginary + y * 2 * (ln2 - lnkr);
            u[i] = Complex.Exp(new Complex(re, im));
        }
        if (n % 2 == 0) u[m - 1] = new Complex(u[m - 1].Real, 0);
        if (!double.IsFinite(u[0].Real) || !double.IsFinite(u[0].Imaginary) || double.IsNaN(u[0].Real))
            u[0] = Math.Pow(2, q) * Special.Poch(xm, xp - xm);
        // A singular forward transform is regularised to 0 and a singular inverse to ∞ (scipy warns).
        if (!inverse && double.IsInfinity(u[0].Real)) u[0] = 0;
        else if (inverse && u[0] == Complex.Zero) u[0] = double.PositiveInfinity;
        return u;
    }

    private static double[] Fhtq(double[] a, Complex[] u, bool inverse)
    {
        var n = a.Length;
        var spec = Rfft(a);
        for (var i = 0; i < spec.Length; i++)
            spec[i] = inverse ? spec[i] / Complex.Conjugate(u[i]) : spec[i] * u[i];
        var r = Irfft(spec, n);
        Array.Reverse(r);
        return r;
    }

    /// <summary>
    /// Principal-branch log Γ(z) for complex z (scipy's <c>loggamma</c>): analytic off the negative
    /// real axis, imaginary part continuous (not reduced mod 2π). Stirling's series at |w| ≥ 20,
    /// reached by the recurrence log Γ(z) = log Γ(z + m) − Σ log(z + k).
    /// </summary>
    internal static Complex FftLogGamma(Complex z)
    {
        if (z.Imaginary == 0 && z.Real <= 0 && z.Real == Math.Floor(z.Real)) return new Complex(double.PositiveInfinity, 0);
        var shift = Complex.Zero;
        var w = z;
        while (w.Real < 20)
        {
            shift += Complex.Log(w);
            w += 1;
        }
        // Stirling: (w − ½) log w − w + ½ log 2π + Σ B₂ₖ / (2k(2k−1) w^(2k−1)).
        ReadOnlySpan<double> c = [1.0 / 12, -1.0 / 360, 1.0 / 1260, -1.0 / 1680, 1.0 / 1188, -691.0 / 360360, 1.0 / 156, -3617.0 / 122400];
        var inv = 1 / w;
        var inv2 = inv * inv;
        var series = Complex.Zero;
        var p = inv;
        foreach (var ck in c) { series += ck * p; p *= inv2; }
        var lg = (w - 0.5) * Complex.Log(w) - w + 0.5 * Math.Log(2 * Math.PI) + series;
        return lg - shift;
    }

    // ---------------------------------------------------------------- Hermitian n-D

    /// <summary>
    /// 2-D FFT of a Hermitian-symmetric signal — scipy's <c>hfft2</c>: the real spectrum of a
    /// signal given by its half-plane, <c>irfft2(conj(x))</c> with the normalisation direction swapped.
    /// </summary>
    public static NdArray Hfft2(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => HfftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>Inverse of <see cref="Hfft2"/> — scipy's <c>ihfft2</c>.</summary>
    public static ComplexNdArray Ihfft2(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
        => IhfftN(x, s, axes ?? [-2, -1], norm, workers);

    /// <summary>
    /// n-D Hermitian FFT — scipy's <c>hfftn</c>: the last transformed axis is the half-length
    /// Hermitian one (output length 2(m − 1) unless <paramref name="s"/> says otherwise).
    /// </summary>
    public static NdArray HfftN(ComplexNdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var data = x.ToArray();
        FftComplexOps.Conjugate(data);
        return IrfftN(new ComplexNdArray(data, x.Shape.ToArray()), s, axes, SwapDirection(norm), workers);
    }

    /// <summary>Inverse n-D Hermitian FFT — scipy's <c>ihfftn</c>: <c>conj(rfftn(x))</c> with the normalisation direction swapped.</summary>
    public static ComplexNdArray IhfftN(NdArray x, int[]? s = null, int[]? axes = null, FftNorm norm = FftNorm.Backward, int? workers = null)
    {
        var r = RfftN(x, s, axes, SwapDirection(norm), workers);
        var data = r.ToArray();
        FftComplexOps.Conjugate(data);
        return new ComplexNdArray(data, r.Shape.ToArray());
    }
}
