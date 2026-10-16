using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalWaveformTests
{
    private static readonly Dictionary<string, double[]> R = SignalWaveRef.D;

    [Fact]
    public void PeriodicWavesMatchScipy()
    {
        var t = R["t"];
        SignalAssert.Close(R["saw"], Signal.Sawtooth(t.Select(v => 2 * Math.PI * 0.7 * v).ToArray()), 1e-12, 1e-12, "saw");
        SignalAssert.Close(R["saw_tri"], Signal.Sawtooth(t, 0.5), 1e-12, 1e-12, "triangle");
        SignalAssert.Close(R["saw_w0"], Signal.Sawtooth(t, 0.0), 1e-12, 1e-12, "width 0");
        SignalAssert.Close(R["saw_w3"], Signal.Sawtooth(t, 0.3), 1e-12, 1e-12, "width .3");
        SignalAssert.Close(R["sq"], Signal.Square(t), 0, 0, "square");
        SignalAssert.Close(R["sq_d"], Signal.Square(t, 0.2), 0, 0, "duty");
        Assert.True(double.IsNaN(Signal.Square([1.0], 1.5)[0]));
    }

    [Fact]
    public void ChirpsMatchScipy()
    {
        var t = R["tc"];
        SignalAssert.Close(R["chirp_linear"], Signal.Chirp(t, 5, 2, 20, ChirpMethod.Linear, 30), 1e-11, 1e-12, "linear");
        SignalAssert.Close(R["chirp_quadratic"], Signal.Chirp(t, 5, 2, 20, ChirpMethod.Quadratic, 30), 1e-11, 1e-12, "quadratic");
        SignalAssert.Close(R["chirp_logarithmic"], Signal.Chirp(t, 5, 2, 20, ChirpMethod.Logarithmic, 30), 1e-11, 1e-12, "log");
        SignalAssert.Close(R["chirp_hyperbolic"], Signal.Chirp(t, 5, 2, 20, ChirpMethod.Hyperbolic, 30), 1e-11, 1e-12, "hyperbolic");
        SignalAssert.Close(R["chirp_quad_v"], Signal.Chirp(t, 5, 2, 20, ChirpMethod.Quadratic, vertexZero: false), 1e-11, 1e-12, "quad vertex");
        SignalAssert.Close(R["chirp_hyp_down"], Signal.Chirp(t, 20, 2, 5, ChirpMethod.Hyperbolic), 1e-11, 1e-12, "hyp down");
        SignalAssert.Close(R["chirp_cplx"], Signal.ChirpComplex(t, 5, 2, 20), 1e-11, 1e-12, "complex");
        SignalAssert.Close(R["sweep"], Signal.SweepPoly(t, [0.5, -1, 2, 3], 10), 1e-11, 1e-12, "sweep_poly");
        Assert.Throws<ArgumentException>(() => Signal.Chirp(t, 0, 2, 20, ChirpMethod.Logarithmic));
    }

    [Fact]
    public void GaussPulseMatchesScipy()
    {
        var (i, q, e) = Signal.GaussPulseFull(R["tg"], 5000, 0.4);
        SignalAssert.Close(R["gp_i"], i, 1e-11, 1e-14, "I");
        SignalAssert.Close(R["gp_q"], q, 1e-11, 1e-14, "Q");
        SignalAssert.Close(R["gp_e"], e, 1e-12, 1e-14, "env");
        SignalAssert.Close(R["gp_i"], Signal.GaussPulse(R["tg"], 5000, 0.4), 1e-11, 1e-14, "I only");
        SignalAssert.Close(R["gp_cut"][0], Signal.GaussPulseCutoff(5000, 0.4, -6, -40), 1e-14, 0, "cutoff");
    }

    [Fact]
    public void MaxLenSeqMatchesScipy()
    {
        var (s5, st5) = Signal.MaxLenSeq(5);
        SignalAssert.Close(R["mls5"], s5.Select(v => (double)v).ToArray(), 0, 0, "mls5");
        SignalAssert.Close(R["mls5_state"], st5.Select(v => (double)v).ToArray(), 0, 0, "mls5 state");
        var (s8, st8) = Signal.MaxLenSeq(8, length: 40);
        SignalAssert.Close(R["mls8"], s8.Select(v => (double)v).ToArray(), 0, 0, "mls8");
        SignalAssert.Close(R["mls8_state"], st8.Select(v => (double)v).ToArray(), 0, 0, "mls8 state");
        var (sc, stc) = Signal.MaxLenSeq(4, [1, 0, 0, 1], 20, [3, 1]);
        SignalAssert.Close(R["mls_custom"], sc.Select(v => (double)v).ToArray(), 0, 0, "custom");
        SignalAssert.Close(R["mls_custom_state"], stc.Select(v => (double)v).ToArray(), 0, 0, "custom state");
        // Continuing from the returned state gives the next part of the same sequence.
        var (first, mid) = Signal.MaxLenSeq(10, length: 300);
        var (second, _) = Signal.MaxLenSeq(10, mid, 300);
        var (whole, _) = Signal.MaxLenSeq(10, length: 600);
        Assert.Equal(whole, first.Concat(second).ToArray());
        // Defining property: the ±1 sequence's circular autocorrelation is N at lag 0 and −1 elsewhere.
        var (m, _) = Signal.MaxLenSeq(7);
        var pm = m.Select(v => v == 1 ? 1.0 : -1.0).ToArray();
        for (var lag = 0; lag < 127; lag += 9)
        {
            var s = 0.0;
            for (var k = 0; k < 127; k++) s += pm[k] * pm[(k + lag) % 127];
            Assert.Equal(lag == 0 ? 127.0 : -1.0, s);
        }
    }

    [Fact]
    public void UnitImpulse()
    {
        Assert.Equal([0.0, 0, 1, 0, 0], Signal.UnitImpulse(5, 2));
        Assert.Equal([0.0, 0, 0, 1], Signal.UnitImpulse(4, -1));
        var u = Signal.UnitImpulse([3, 4], mid: true).ToArray();
        Assert.Equal(1.0, u[1 * 4 + 2]);
        Assert.Equal(1.0, u.Sum());
    }
}
