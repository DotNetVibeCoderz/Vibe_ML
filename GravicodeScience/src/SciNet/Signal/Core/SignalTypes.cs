using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>Band type of a frequency-selective filter (scipy's <c>btype</c>).</summary>
public enum FilterType
{
    /// <summary>Passes frequencies below the cutoff.</summary>
    Lowpass,
    /// <summary>Passes frequencies above the cutoff.</summary>
    Highpass,
    /// <summary>Passes frequencies between two cutoffs.</summary>
    Bandpass,
    /// <summary>Rejects frequencies between two cutoffs.</summary>
    Bandstop,
}

/// <summary>Representation an IIR design returns (scipy's <c>output</c>).</summary>
public enum FilterOutput
{
    /// <summary>Numerator/denominator polynomials, <see cref="FilterBA"/>.</summary>
    Ba,
    /// <summary>Zeros, poles and gain, <see cref="FilterZpk"/>.</summary>
    Zpk,
    /// <summary>Second-order sections, an <c>n × 6</c> array.</summary>
    Sos,
}

/// <summary>IIR prototype family (scipy's <c>ftype</c>).</summary>
public enum IirDesign
{
    /// <summary>Butterworth: maximally flat passband.</summary>
    Butter,
    /// <summary>Chebyshev type I: equiripple passband.</summary>
    Cheby1,
    /// <summary>Chebyshev type II: equiripple stopband.</summary>
    Cheby2,
    /// <summary>Elliptic (Cauer): equiripple in both bands.</summary>
    Ellip,
    /// <summary>Bessel/Thomson: maximally flat group delay.</summary>
    Bessel,
}

/// <summary>Normalisation of a Bessel prototype (scipy's <c>norm</c>).</summary>
public enum BesselNorm
{
    /// <summary>Phase response reaches its midpoint at the cutoff (scipy's default).</summary>
    Phase,
    /// <summary>Group delay at DC is <c>1/ωc</c>.</summary>
    Delay,
    /// <summary>Gain is −3 dB at the cutoff.</summary>
    Mag,
}

/// <summary>How <see cref="Signal.Zpk2Sos(FilterZpk, SosPairing?, bool)"/> pairs poles and zeros.</summary>
public enum SosPairing
{
    /// <summary>Pair each pole with its nearest zero; an odd filter gets a first-order final section.</summary>
    Nearest,
    /// <summary>Like <see cref="Nearest"/> but the odd section keeps a second-order shape.</summary>
    KeepOdd,
    /// <summary>Fewest sections; no zero is added that the filter did not have.</summary>
    Minimal,
}

/// <summary>Padding used by <see cref="Signal.Filtfilt(double[], double[], double[], FiltfiltPad, int?, FiltfiltMethod, int?)"/>.</summary>
public enum FiltfiltPad
{
    /// <summary>Odd extension (scipy's default): reflect and negate about the end point.</summary>
    Odd,
    /// <summary>Even extension: mirror about the end point.</summary>
    Even,
    /// <summary>Repeat the end value.</summary>
    Constant,
    /// <summary>No padding.</summary>
    None,
}

/// <summary>Edge handling of <c>filtfilt</c>.</summary>
public enum FiltfiltMethod
{
    /// <summary>Pad the signal and start each pass from the steady state.</summary>
    Pad,
    /// <summary>Gustafsson's method: choose initial states so forward-backward equals backward-forward.</summary>
    Gust,
}

/// <summary>Remez filter type (scipy's <c>type</c>).</summary>
public enum RemezType
{
    /// <summary>Ordinary multiband filter.</summary>
    Bandpass,
    /// <summary>Differentiator: odd symmetry, weight divided by frequency in the passband.</summary>
    Differentiator,
    /// <summary>Hilbert transformer: odd symmetry.</summary>
    Hilbert,
}

/// <summary>Normalisation of a DPSS (Slepian) window.</summary>
public enum DpssNorm
{
    /// <summary>Unit energy (L2 norm one).</summary>
    L2,
    /// <summary>Peak one, with the even-length correction <c>M²/(M²+NW)</c>.</summary>
    Approximate,
    /// <summary>Peak one, with the even-length correction computed by sub-sample shifting.</summary>
    Subsample,
}

/// <summary>Method of <see cref="Signal.MinimumPhase"/>.</summary>
public enum MinimumPhaseMethod
{
    /// <summary>Cepstral (homomorphic) method; the result matches the square root of the magnitude.</summary>
    Homomorphic,
    /// <summary>Hilbert-transform method; needs an equiripple linear-phase input.</summary>
    Hilbert,
}

/// <summary>Frequency sweep law of <see cref="Signal.Chirp(ReadOnlySpan{double}, double, double, double, ChirpMethod, double, bool)"/>.</summary>
public enum ChirpMethod
{
    /// <summary>Frequency changes linearly with time.</summary>
    Linear,
    /// <summary>Frequency follows a parabola.</summary>
    Quadratic,
    /// <summary>Frequency changes geometrically.</summary>
    Logarithmic,
    /// <summary>Frequency follows a hyperbola.</summary>
    Hyperbolic,
}

/// <summary>Discretisation method of <see cref="Signal.Cont2Discrete(StateSpace, double, DiscretizationMethod, double)"/>.</summary>
public enum DiscretizationMethod
{
    /// <summary>Zero-order hold.</summary>
    Zoh,
    /// <summary>First-order (triangle) hold.</summary>
    Foh,
    /// <summary>Tustin's bilinear transform, <c>gbt</c> with α = 0.5.</summary>
    Bilinear,
    /// <summary>Forward Euler, <c>gbt</c> with α = 0.</summary>
    Euler,
    /// <summary>Backward difference, <c>gbt</c> with α = 1.</summary>
    BackwardDiff,
    /// <summary>Impulse invariance.</summary>
    Impulse,
    /// <summary>Generalised bilinear transform with the given α.</summary>
    Gbt,
}

/// <summary>A transfer function as numerator and denominator coefficients, highest power first.</summary>
/// <param name="B">Numerator coefficients.</param>
/// <param name="A">Denominator coefficients.</param>
public sealed record FilterBA(double[] B, double[] A)
{
    /// <summary>Deconstructs into <c>(b, a)</c>.</summary>
    public void Deconstruct(out double[] b, out double[] a) { b = B; a = A; }
}

/// <summary>A filter as zeros, poles and system gain.</summary>
/// <param name="Z">Zeros.</param>
/// <param name="P">Poles.</param>
/// <param name="K">Gain.</param>
public sealed record FilterZpk(Complex[] Z, Complex[] P, double K)
{
    /// <summary>Deconstructs into <c>(z, p, k)</c>.</summary>
    public void Deconstruct(out Complex[] z, out Complex[] p, out double k) { z = Z; p = P; k = K; }
}

/// <summary>
/// The result of an IIR design: exactly one of <see cref="Ba"/>, <see cref="Zpk"/>, <see cref="Sos"/>
/// is set, matching the requested <see cref="FilterOutput"/>.
/// </summary>
public sealed class IirFilterResult
{
    internal IirFilterResult(FilterBA? ba, FilterZpk? zpk, Gravicode.Science.GraviNum.NdArray? sos)
    {
        Ba = ba; Zpk = zpk; Sos = sos;
    }

    /// <summary>Numerator/denominator, when the output was <see cref="FilterOutput.Ba"/>.</summary>
    public FilterBA? Ba { get; }

    /// <summary>Zeros/poles/gain, when the output was <see cref="FilterOutput.Zpk"/>.</summary>
    public FilterZpk? Zpk { get; }

    /// <summary>Second-order sections (<c>n × 6</c>), when the output was <see cref="FilterOutput.Sos"/>.</summary>
    public Gravicode.Science.GraviNum.NdArray? Sos { get; }
}

/// <summary>
/// A window name plus its parameters, the analogue of scipy's <c>('kaiser', 8.6)</c> tuples.
/// Converts implicitly from a bare name and from <c>(name, parameter)</c> tuples.
/// </summary>
/// <param name="Name">Window name or alias as scipy spells it (<c>"hann"</c>, <c>"kaiser"</c>, <c>"general_gaussian"</c>...).</param>
/// <param name="Parameters">Positional parameters after the length.</param>
public sealed record WindowSpec(string Name, params double[] Parameters)
{
    /// <summary>A window with no parameters.</summary>
    public static implicit operator WindowSpec(string name) => new(name);

    /// <summary>A window with one parameter, e.g. <c>("kaiser", 8.6)</c>.</summary>
    public static implicit operator WindowSpec((string Name, double P) spec) => new(spec.Name, spec.P);

    /// <summary>A window with two parameters, e.g. <c>("general_gaussian", 1.5, 7)</c>.</summary>
    public static implicit operator WindowSpec((string Name, double P1, double P2) spec) => new(spec.Name, spec.P1, spec.P2);

    /// <summary>A Kaiser window with the given β, as scipy treats a bare float.</summary>
    public static WindowSpec Kaiser(double beta) => new("kaiser", beta);
}
