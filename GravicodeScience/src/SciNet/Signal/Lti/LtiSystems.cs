using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A linear time-invariant system, continuous (<see cref="Dt"/> null, scipy's <c>lti</c>) or
/// discrete with sample time <see cref="Dt"/> (scipy's <c>dlti</c>). Concrete forms are
/// <see cref="TransferFunction"/>, <see cref="ZerosPolesGain"/> and <see cref="StateSpace"/>, each
/// convertible to the others with scipy's conversions.
/// </summary>
public abstract class LtiSystem
{
    /// <summary>Creates a system with the given sample time (null = continuous).</summary>
    protected LtiSystem(double? dt)
    {
        if (dt is double d && !(d > 0)) throw new ArgumentException("dt must be positive (or null for a continuous system).", nameof(dt));
        Dt = dt;
    }

    /// <summary>Sample time of a discrete system; null for a continuous one.</summary>
    public double? Dt { get; }

    /// <summary>True for a discrete-time system.</summary>
    public bool IsDiscrete => Dt is not null;

    /// <summary>The system as a transfer function (SISO only).</summary>
    public abstract TransferFunction ToTf();

    /// <summary>The system as zeros, poles and gain (SISO only).</summary>
    public abstract ZerosPolesGain ToZpk();

    /// <summary>The system in state-space form.</summary>
    public abstract StateSpace ToSs();

    /// <summary>Poles of the system.</summary>
    public virtual Complex[] Poles => ToZpk().P;

    /// <summary>Zeros of the system.</summary>
    public virtual Complex[] Zeros => ToZpk().Z;

    /// <summary>
    /// Discretises a continuous system with sample time <paramref name="dt"/>
    /// (<c>lti.to_discrete</c>), keeping the same representation.
    /// </summary>
    public abstract LtiSystem ToDiscrete(double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN);
}

/// <summary>Transfer function <c>num(s)/den(s)</c> (or in z for a discrete system), normalised as scipy does.</summary>
public sealed class TransferFunction : LtiSystem
{
    /// <summary>Creates a transfer function; coefficients are highest power first and get normalised (<c>den[0] = 1</c>).</summary>
    public TransferFunction(double[] num, double[] den, double? dt = null) : base(dt)
    {
        ArgumentNullException.ThrowIfNull(num);
        ArgumentNullException.ThrowIfNull(den);
        var (b, a) = Signal.Normalize(num, den);
        Num = b;
        Den = a;
    }

    /// <summary>Numerator coefficients.</summary>
    public double[] Num { get; }

    /// <summary>Denominator coefficients (monic).</summary>
    public double[] Den { get; }

    /// <inheritdoc/>
    public override TransferFunction ToTf() => this;

    /// <inheritdoc/>
    public override ZerosPolesGain ToZpk()
    {
        var zpk = Signal.Tf2Zpk(Num, Den);
        return new ZerosPolesGain(zpk.Z, zpk.P, zpk.K, Dt);
    }

    /// <inheritdoc/>
    public override StateSpace ToSs()
    {
        var (a, b, c, d) = Signal.Tf2Ss(Num, Den);
        return new StateSpace(a, b, c, d, Dt);
    }

    /// <inheritdoc/>
    public override LtiSystem ToDiscrete(double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN)
    {
        var sd = Signal.Cont2Discrete(ToSs(), dt, method, alpha);
        var (num, den) = Signal.Ss2Tf(sd.A, sd.B, sd.C, sd.D);
        return new TransferFunction(num.ToArray(), den, dt);
    }
}

/// <summary>A system as zeros, poles and gain.</summary>
public sealed class ZerosPolesGain : LtiSystem
{
    /// <summary>Creates a zpk system.</summary>
    public ZerosPolesGain(Complex[] zeros, Complex[] poles, double gain, double? dt = null) : base(dt)
    {
        ArgumentNullException.ThrowIfNull(zeros);
        ArgumentNullException.ThrowIfNull(poles);
        Z = zeros.ToArray();
        P = poles.ToArray();
        K = gain;
    }

    /// <summary>Zeros.</summary>
    public Complex[] Z { get; }

    /// <summary>Poles.</summary>
    public Complex[] P { get; }

    /// <summary>Gain.</summary>
    public double K { get; }

    /// <inheritdoc/>
    public override Complex[] Poles => P;

    /// <inheritdoc/>
    public override Complex[] Zeros => Z;

    /// <summary>As the plain <see cref="FilterZpk"/> record.</summary>
    public FilterZpk ToFilterZpk() => new(Z, P, K);

    /// <inheritdoc/>
    public override TransferFunction ToTf()
    {
        var (b, a) = Signal.Zpk2Tf(Z, P, K);
        return new TransferFunction(b, a, Dt);
    }

    /// <inheritdoc/>
    public override ZerosPolesGain ToZpk() => this;

    /// <inheritdoc/>
    public override StateSpace ToSs() => ToTf().ToSs();

    /// <inheritdoc/>
    public override LtiSystem ToDiscrete(double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN)
    {
        var sd = Signal.Cont2Discrete(ToSs(), dt, method, alpha);
        var zpk = Signal.Ss2Zpk(sd.A, sd.B, sd.C, sd.D);
        return new ZerosPolesGain(zpk.Z, zpk.P, zpk.K, dt);
    }
}

/// <summary>
/// State-space system <c>x' = Ax + Bu, y = Cx + Du</c> (or <c>x[k+1] = ...</c> when discrete).
/// Multiple inputs and outputs are allowed; tf/zpk conversions use input 0 as scipy does.
/// </summary>
public sealed class StateSpace : LtiSystem
{
    /// <summary>Creates a state-space system; empty matrices are sized from the others (scipy's <c>abcd_normalize</c>).</summary>
    public StateSpace(NdArray a, NdArray b, NdArray c, NdArray d, double? dt = null) : base(dt)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        ArgumentNullException.ThrowIfNull(d);
        A = As2D(a); B = As2D(b); C = As2D(c); D = As2D(d);
        var n = A.Shape[0];
        if (A.Shape[1] != n) throw new ArgumentException("A must be square.", nameof(a));
        if (B.Shape[0] != n) throw new ArgumentException($"B must have {n} rows.", nameof(b));
        if (C.Shape[1] != n) throw new ArgumentException($"C must have {n} columns.", nameof(c));
        if (D.Shape[0] != C.Shape[0] || D.Shape[1] != B.Shape[1]) throw new ArgumentException($"D must be {C.Shape[0]}x{B.Shape[1]}.", nameof(d));
    }

    private static NdArray As2D(NdArray m) => m.Rank == 2 ? m.AsContiguous() : m.Rank == 1 ? new NdArray(m.ToArray(), 1, m.Size) : new NdArray([m.ToArray()[0]], 1, 1);

    /// <summary>State matrix.</summary>
    public NdArray A { get; }

    /// <summary>Input matrix.</summary>
    public NdArray B { get; }

    /// <summary>Output matrix.</summary>
    public NdArray C { get; }

    /// <summary>Feedthrough matrix.</summary>
    public NdArray D { get; }

    /// <summary>Number of inputs.</summary>
    public int Inputs => B.Shape[1];

    /// <summary>Number of outputs.</summary>
    public int Outputs => C.Shape[0];

    /// <summary>Number of states.</summary>
    public int States => A.Shape[0];

    /// <inheritdoc/>
    public override TransferFunction ToTf()
    {
        if (Inputs != 1 || Outputs != 1) throw new InvalidOperationException("Transfer-function form requires a SISO system.");
        var (num, den) = Signal.Ss2Tf(A, B, C, D);
        return new TransferFunction(num.ToArray(), den, Dt);
    }

    /// <inheritdoc/>
    public override ZerosPolesGain ToZpk()
    {
        if (Inputs != 1 || Outputs != 1) throw new InvalidOperationException("Zpk form requires a SISO system.");
        var zpk = Signal.Ss2Zpk(A, B, C, D);
        return new ZerosPolesGain(zpk.Z, zpk.P, zpk.K, Dt);
    }

    /// <inheritdoc/>
    public override StateSpace ToSs() => this;

    /// <summary>Poles: the eigenvalues of <see cref="A"/> (valid for MIMO systems too).</summary>
    public override Complex[] Poles => States == 0 ? [] : Linalg.Eigvals(A);

    /// <inheritdoc/>
    public override LtiSystem ToDiscrete(double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN)
        => Signal.Cont2Discrete(this, dt, method, alpha);
}
