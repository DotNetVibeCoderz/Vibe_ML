using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

// Expected values: scipy 1.17.1 interp1d / numpy 2.4.4 interp. X is deliberately unsorted.
public class Interp1dTests
{
    private static readonly double[] X = [3.0, 0.0, 1.0, 2.5, 4.5, 6.0];
    private static readonly double[] Y = [0.2, 0.3, 1.1, -0.4, 1.7, 0.9];
    private static readonly double[] Xe = [0.0, 0.4, 0.5, 1.0, 1.75, 2.2, 3.3, 4.5, 5.9, 6.0];

    [Fact]
    public void Kinds_MatchScipy()
    {
        AllClose([0.3, 0.6200000000000001, 0.7, 1.1, 0.3500000000000001, -0.10000000000000009, 0.49999999999999983, 1.7, 0.9533333333333331, 0.9], new Interp1d(X, Y, Interp1dKind.Linear).Evaluate(Xe), 1e-12, "linear");
        AllClose([0.3, 0.3, 0.3, 1.1, 1.1, -0.4, 0.2, 1.7, 0.9, 0.9], new Interp1d(X, Y, Interp1dKind.Nearest).Evaluate(Xe), 1e-12, "nearest");
        AllClose([0.3, 0.3, 1.1, 1.1, -0.4, -0.4, 0.2, 1.7, 0.9, 0.9], new Interp1d(X, Y, Interp1dKind.NearestUp).Evaluate(Xe), 1e-12, "nearest-up");
        AllClose([0.3, 0.3, 0.3, 1.1, 1.1, 1.1, 0.2, 1.7, 1.7, 0.9], new Interp1d(X, Y, Interp1dKind.Zero).Evaluate(Xe), 1e-12, "zero");
        AllClose([0.3, 0.6200000000000001, 0.7000000000000001, 1.1, 0.35000000000000003, -0.10000000000000017, 0.4999999999999998, 1.7, 0.9533333333333331, 0.9], new Interp1d(X, Y, Interp1dKind.Slinear).Evaluate(Xe), 1e-12, "slinear");
        AllClose([0.3, 0.8981770859913245, 0.9897677979076295, 1.1, 0.17871906098494528, -0.4072018372033683, 0.5980280683847917, 1.6999999999999997, 1.0323185619914372, 0.9], new Interp1d(X, Y, Interp1dKind.Quadratic).Evaluate(Xe), 1e-12, "quadratic");
        AllClose([0.3, 1.1454040000000003, 1.2298055555555554, 1.1, 0.06632031249999984, -0.39489200000000013, 0.6207179999999997, 1.7, 1.0513214938271598, 0.9], new Interp1d(X, Y, Interp1dKind.Cubic).Evaluate(Xe), 1e-12, "cubic");
        AllClose([0.3, 0.3, 0.3, 1.1, 1.1, 1.1, 0.2, 1.7, 1.7, 0.9], new Interp1d(X, Y, Interp1dKind.Previous).Evaluate(Xe), 1e-12, "previous");
        AllClose([0.3, 1.1, 1.1, 1.1, -0.4, -0.4, 1.7, 1.7, 0.9, 0.9], new Interp1d(X, Y, Interp1dKind.Next).Evaluate(Xe), 1e-12, "next");
        AllClose([0.3, 1.6043697777777781, 1.6604938271604943, 1.1000000000000003, -0.2462456597222233, -0.5171707936507933, 0.6812160000000004, 1.7000000000000006, 0.7216591604938271, 0.9], new Interp1d(X, Y, 5).Evaluate(Xe), 1e-10, "order 5");
    }

    [Fact]
    public void FillValues_MatchScipy()
    {
        double[] xo = [-1.0, 0.4, 6.5];
        AllClose([-0.5, 0.6200000000000001, 0.6333333333333333], new Interp1d(X, Y, Interp1dKind.Linear, fillValue: Interp1dFill.ExtrapolateFill).Evaluate(xo), 1e-12, "linear extrapolate");
        AllClose([-6.837277777777777, 1.1454040000000003, -0.10564506172839505], new Interp1d(X, Y, Interp1dKind.Cubic, fillValue: Interp1dFill.ExtrapolateFill).Evaluate(xo), 1e-12, "cubic extrapolate");
        AllClose([0.3, 0.3, 0.9], new Interp1d(X, Y, Interp1dKind.Nearest, fillValue: Interp1dFill.ExtrapolateFill).Evaluate(xo), 1e-12, "nearest extrapolate");
        AllClose([double.NaN, 0.3, 0.9], new Interp1d(X, Y, Interp1dKind.Previous, fillValue: Interp1dFill.ExtrapolateFill).Evaluate(xo), 1e-12, "previous extrapolate");
        AllClose([0.3, 1.1, double.NaN], new Interp1d(X, Y, Interp1dKind.Next, fillValue: Interp1dFill.ExtrapolateFill).Evaluate(xo), 1e-12, "next extrapolate");
        AllClose([-7.0, 0.6200000000000001, 8.0], new Interp1d(X, Y, boundsError: false, fillValue: Interp1dFill.BelowAbove(-7, 8)).Evaluate(xo), 0, "below/above");
        AllClose([double.NaN, 0.6200000000000001, double.NaN], new Interp1d(X, Y, boundsError: false).Evaluate(xo), 0, "nan fill");
        Assert.Throws<ArgumentException>(() => new Interp1d(X, Y).Evaluate(xo));
    }

    [Fact]
    public void VectorValued_LastAxisDefault_MatchesScipy()
    {
        var y2 = new double[18];
        for (var i = 0; i < 6; i++) { y2[i] = Y[i]; y2[6 + i] = 2 * Y[i]; y2[12 + i] = Y[i] * Y[i]; }
        var f = new Interp1d(X, new NdArray(y2, 3, 6), Interp1dKind.Cubic);
        var r = f.Evaluate(new NdArray([0.5, 1.5, 2.0, 5.0], 2, 2));
        Assert.Equal([3, 2, 2], r.Shape.ToArray());
        AllClose([1.2298055555555554, 0.4352916666666665, -0.23961111111111127, 1.7539938271604938, 2.4596111111111107, 0.870583333333333, -0.47922222222222255, 3.5079876543209876, 0.976352777777778, 1.0074708333333335, 0.5852944444444443, 3.424008024691357], r.ToArray(), 1e-12, "values");
    }

    [Fact]
    public void NumpyInterp_MatchesNumpy()
    {
        double[] xp = [0.0, 1.0, 2.5, 3.0, 4.5, 6.0];
        double[] fp = [0.3, 1.1, -0.4, 0.2, 1.7, 0.9];
        double[] q = [-1.0, 0.0, 0.3, 1.0, 2.9, 6.0, 6.5, double.NaN];
        AllClose([0.3, 0.3, 0.54, 1.1, 0.07999999999999996, 0.9, 0.9, double.NaN], Interpolate.Interp(q, xp, fp), 0, "default");
        AllClose([-5.0, 0.3, 0.54, 1.1, 0.07999999999999996, 0.9, 5.0, double.NaN], Interpolate.Interp(q, xp, fp, -5, 5), 0, "left/right");
        AllClose([1.2333333333333334, 0.9600000000000001, 0.6000000000000001, 1.1], Interpolate.Interp([-1, 0.3, 7.5, 13], xp, fp, period: 6), 1e-15, "period");
        Assert.Equal(2.0, Interpolate.Interp(0.25, [1.0], [2.0]));
    }

    [Fact]
    public void NumpyInterp_FastPathsMatchReference()
    {
        var rng = new Random(3);
        var xp = new double[500];
        var fp = new double[500];
        for (var i = 1; i < 500; i++) xp[i] = xp[i - 1] + rng.NextDouble() + 1e-3;
        for (var i = 0; i < 500; i++) fp[i] = Math.Sin(xp[i]);
        var sorted = new double[20_000];
        for (var i = 0; i < sorted.Length; i++) sorted[i] = -5 + (xp[^1] + 10) * i / sorted.Length;
        var shuffled = sorted.OrderBy(_ => rng.Next()).ToArray();
        foreach (var q in new[] { sorted, shuffled })
        {
            // Naive reference: linear scan for the interval, numpy's formula.
            var expected = new double[q.Length];
            for (var i = 0; i < q.Length; i++)
            {
                var v = q[i];
                if (v < xp[0]) { expected[i] = fp[0]; continue; }
                if (v >= xp[^1]) { expected[i] = fp[^1]; continue; }
                var j = 0;
                while (xp[j + 1] <= v) j++;
                expected[i] = xp[j] == v ? fp[j] : (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (v - xp[j]) + fp[j];
            }
            Assert.Equal(expected, Interpolate.Interp(q, xp, fp));
        }
    }
}
