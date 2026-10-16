using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

public class CubicSplineTests
{
    private static readonly double[] X = [0, 1, 2.5, 3, 4.5, 6];
    private static readonly double[] Y = [0.3, 1.1, -0.4, 0.2, 1.7, 0.9];
    private static readonly double[] Xe = [-0.7, 0, 0.4, 1, 2.2, 3.3, 4.5, 5.9, 6, 7.2];

    // Every expected array below: scipy 1.17.1, CubicSpline(X, Y, bc_type=...) at Xe.
    [Fact]
    public void NotAKnot_MatchesScipy_ValuesDerivativesIntegralRoots()
    {
        var cs = new CubicSpline(X, Y);
        AllClose([-3.7809191111111096, 0.3, 1.1454040000000003, 1.1, -0.394892, 0.6207179999999998, 1.7, 1.05132149382716, 0.8999999999999996, -2.2819480000000003], cs.Evaluate(Xe), 1e-13, "v");
        AllClose([8.733550555555553, 3.269027777777777, 1.0699299999999998, -0.9694166666666664, -0.5146299999999995, 1.3252327777777781, 0.37981944444444427, -1.4341087037037044, -1.5929722222222222, -3.8040255555555555], cs.Evaluate(Xe, 1), 1e-13, "d1");
        AllClose([-9.275644444444442, -6.337277777777776, -4.658211111111109, -2.139611111111112, 2.8975888888888903, -0.5534666666666668, -1.0222222222222221, -1.5691037037037037, -1.6081666666666665, -2.076922222222222], cs.Evaluate(Xe, 2), 1e-13, "d2");
        Close(4.989917052469137, cs.Integrate(-0.5, 6.5), 1e-13, "integral");
        AllClose([-0.08468851845871597, 1.7985001095859983, 2.864418326572718, 6.4560986995204175], cs.Roots(), 1e-12, "roots");
        AllClose([0.699611111111111, 0.6996111111111113, -1.5310555555555556, -0.06510493827160493, -0.06510493827160493, -3.168638888888888, -1.069805555555556, 2.0784444444444445, -0.218138888888889, -0.5111111111111111, 3.269027777777777, -0.9694166666666664, 0.5435416666666668, 1.4736944444444446, 0.37981944444444427, 0.3, 1.1, -0.4, 0.2, 1.7], cs.C.ToArray(), 1e-13, "c");
    }

    [Fact]
    public void Clamped_Natural_Custom_MatchScipy()
    {
        var cl = new CubicSpline(X, Y, SplineBoundaryCondition.Clamped);
        AllClose([1.9869142857142856, 0.3, 0.5772338192419825, 1.1, -0.3133551020408165, 0.663815743440233, 1.7, 0.9098950869236584, 0.9000000000000004, 3.1725784256559777], cl.Evaluate(Xe), 1e-13, "clamped");
        Close(5.530433403520139, cl.Integrate(-0.5, 6.5), 1e-13, "clamped int");
        AllClose([1.9366291269892408, 2.872976405484815], cl.Roots(), 1e-12, "clamped roots");

        var na = new CubicSpline(X, Y, SplineBoundaryCondition.Natural);
        AllClose([-0.4720427934621103, 0.3, 0.8195696879643388, 1.1, -0.34875839524517094, 0.6411756315007426, 1.7, 0.9893685983159978, 0.8999999999999999, 0.10363268945022241], na.Evaluate(Xe), 1e-13, "natural");
        AllClose([2.4946210995542337, 0, -1.4254977711738488, -3.5637444279346218, 3.0305695889054, -0.5631302625061912, -1.4478454680534916, -0.09652303120356587, 0, 1.158276374442793], na.Evaluate(Xe, 2), 1e-12, "natural d2");
        AllClose([-1.4102261687177453, -0.2197353068666241, 1.8820772669644077, 2.8689907779319936], na.Roots(), 1e-12, "natural roots");

        var cu = new CubicSpline(X, Y, (SplineBoundaryCondition.Derivative(1, 0.5), SplineBoundaryCondition.Derivative(2, -1.0)));
        AllClose([1.1073438860971525, 0.3, 0.6644381909547739, 1.1, -0.32900100502512597, 0.635907537688442, 1.7, 1.027377256653638, 0.9000000000000001, -1.3775477386934678], cu.Evaluate(Xe), 1e-13, "custom");
        Close(0.5, cu.Evaluate(0.0, 1), 1e-14, "custom start slope");
        Close(-1.0, cu.Evaluate(6.0, 2), 1e-13, "custom end curvature");
        AllClose([1.9153874741477646, 2.869537597990652, 6.562653175126491, 36.07889867099362], cu.Roots(), 1e-11, "custom roots");
    }

    [Fact]
    public void Periodic_MatchesScipy_WithPeriodicExtrapolationAndIntegral()
    {
        var yp = (double[])Y.Clone();
        yp[^1] = yp[0];
        var cs = new CubicSpline(X, yp, SplineBoundaryCondition.Periodic);
        Assert.Equal(InterpolateExtrapolate.Periodic, cs.Extrapolate);
        AllClose([0.8058251851851854, 0.3, 0.6208228571428571, 1.1, -0.31504, 0.691588571428571, 1.7, 0.293131851851852, 0.3, 1.0144330158730157], cs.Evaluate(Xe), 1e-13, "v");
        AllClose([-1.3291555555555554, 0.25333333333333347, 1.1307428571428573, -0.028571428571428567, -0.8358857142857126, 1.6040761904761904, -0.42380952380952375, -0.10843174603174432, 0.25333333333333347, -0.7856000000000006], cs.Evaluate(Xe, 1), 1e-12, "d1");
        Close(12.020265386243388, cs.Integrate(-2, 15.3), 1e-13, "int over periods");
        Close(0.25627524867724866, cs.Integrate(5.5, 6.2), 1e-13, "int across the seam");
        Assert.Throws<ArgumentException>(() => new CubicSpline(X, Y, SplineBoundaryCondition.Periodic));
    }

    [Fact]
    public void SmallN_SpecialCases_MatchScipy()
    {
        AllClose([0.30000000000000004, 1.0, 1.4, 2.0, 3.2, 4.3, 5.5, 6.9, 7.0, 8.2], new CubicSpline([0, 2.0], [1, 3.0]).Evaluate(Xe), 1e-14, "n2");
        AllClose([-1.3916666666666666, 1.0, 2.0000000000000004, 3.0, 3.2, 1.2750000000000004, -3.124999999999999, -11.29166666666667, -12.0, -21.799999999999997], new CubicSpline([0, 1, 3.0], [1, 3.0, 2]).Evaluate(Xe), 1e-13, "n3");
        AllClose([1.4270000000000005, 1.0, 1.7519999999999998, 3.0, 1.6079999999999997, 1.5159999999999996, 2.875, 0.9290000000000003, 1.0, 3.088], new CubicSpline([0, 1, 3.0], [1, 3.0, 1], SplineBoundaryCondition.Periodic).Evaluate(Xe), 1e-13, "n3 periodic");
        AllClose([3.8542499999999995, 1.0, 1.536, 3.0, 2.6879999999999997, 2.164812499999999, 7.9765625, 32.43368750000001, 35.1875, 81.6005], new CubicSpline([0, 1, 3.0], [1, 3.0, 2], SplineBoundaryCondition.Clamped).Evaluate(Xe), 1e-13, "n3 clamped");
    }

    [Fact]
    public void VectorValued_AxisPlacementMatchesScipy()
    {
        var yv = new double[12];
        for (var i = 0; i < 6; i++) { yv[2 * i] = Y[i]; yv[2 * i + 1] = Y[i] * Y[i]; }
        var cs = new CubicSpline(X, new NdArray(yv, 6, 2));
        var r = cs.Evaluate(new NdArray([0.5, 2.0, 5.0], 3));
        Assert.Equal([3, 2], r.Shape.ToArray());
        AllClose([1.2298055555555556, 0.9763527777777782, -0.23961111111111089, 0.5852944444444443, 1.7539938271604938, 3.4240080246913585], r.ToArray(), 1e-13, "vec");

        var yt = new NdArray(yv, 6, 2).Transpose(1, 0).Copy();
        var ct = new CubicSpline(X, yt, axis: 1);
        var rt = ct.Evaluate(new NdArray([0.5, 2.0, 5.0], 3));
        Assert.Equal([2, 3], rt.Shape.ToArray());
        AllClose([1.2298055555555556, -0.23961111111111089, 1.7539938271604938, 0.9763527777777782, 0.5852944444444443, 3.4240080246913585], rt.ToArray(), 1e-13, "vecT");
        Assert.Throws<InvalidOperationException>(() => cs.Evaluate(0.5));
    }

    [Fact]
    public void Pchip_Akima_Makima_Hermite_MatchScipy()
    {
        var p = new PchipInterpolator(X, Y);
        AllClose([-1.0501599999999998, 0.3, 0.8004800000000001, 1.1, -0.24400000000000022, 0.5686769230769229, 1.7, 1.0234370370370365, 0.8999999999999999, -1.7992000000000008], p.Evaluate(Xe), 1e-13, "pchip");
        AllClose([2.2984, 1.52, 0.9696, 0.0, -0.96, 1.3144615384615386, 0.0, -1.1697777777777782, -1.3000000000000003, -3.3480000000000008], p.Evaluate(Xe, 1), 1e-13, "pchip d1");

        var ak = new Akima1DInterpolator(X, Y);
        AllClose([double.NaN, 0.3, 0.8273600000000001, 1.1, -0.4326400000000006, 0.5242637362637361, 1.7, 1.0285584045584044, 0.9000000000000004, double.NaN], ak.Evaluate(Xe), 1e-13, "akima");
        AllClose([double.NaN, 1.7000000000000002, 0.9511999999999999, -0.010000000000000009, -0.6435999999999984, 1.075824175824176, 0.8230769230769229, -1.2685470085470083, -1.2999999999999998, double.NaN], ak.Evaluate(Xe, 1), 1e-13, "akima d1");
        AllClose([double.NaN, 0.3, 0.7926758620689655, 1.1, -0.3165379310344829, 0.5398303543098887, 1.7, 0.9983474313022697, 0.9000000000000002, double.NaN], new Akima1DInterpolator(X, Y, AkimaMethod.Makima).Evaluate(Xe), 1e-13, "makima");
        AllClose([-1.40597, 0.3, 0.8273600000000001, 1.1, -0.4326400000000006, 0.5242637362637361, 1.7, 1.0285584045584044, 0.9000000000000004, -0.3769230769230756], new Akima1DInterpolator(X, Y, extrapolate: InterpolateExtrapolate.Extrapolate).Evaluate(Xe), 1e-13, "akima extrapolated");

        // A step: all three shape-preserving methods must stay flat on the flat parts.
        double[] step = [0, 0, 0, 1, 1, 1.0];
        AllClose([double.NaN, 0, 0, 0, 0, 1, 1, 1, 1, double.NaN], new Akima1DInterpolator(X, step, AkimaMethod.Makima).Evaluate(Xe), 1e-14, "makima step");
        AllClose([double.NaN, 0, 0, 0, 0, 1, 1, 1, 1, double.NaN], new Akima1DInterpolator(X, step).Evaluate(Xe), 1e-14, "akima step");
        AllClose([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], new PchipInterpolator(X, step).Evaluate(Xe), 1e-14, "pchip step");

        var h = new CubicHermiteSpline(X, Y, [1, 0, -1, 2, 0.5, -0.3]);
        AllClose([0.0018000000000000516, 0.3, 0.7256, 1.1, -0.052000000000000046, 0.7159999999999997, 1.7, 0.9394370370370371, 0.9000000000000006, 2.9528000000000016], h.Evaluate(Xe), 1e-13, "hermite");
        Close(4.026596666666666, h.Integrate(0.2, 5.1), 1e-13, "hermite integral");
    }

    [Fact]
    public void Spline_ReproducesData_AndIsC2()
    {
        var cs = new CubicSpline(X, Y, SplineBoundaryCondition.Natural);
        AllClose(Y, cs.Evaluate(X), 1e-14, "interpolates");
        for (var i = 1; i < X.Length - 1; i++)
        {
            var e = 1e-9;
            for (var nu = 0; nu <= 2; nu++)
                Close(cs.Evaluate(X[i] - e, nu), cs.Evaluate(X[i] + e, nu), 1e-6, $"C{nu} at knot {i}");
        }
    }

    [Fact]
    public void LargeInput_ParallelEvaluationMatchesSerial()
    {
        var n = 2000;
        var x = new double[n];
        var y = new double[n];
        for (var i = 0; i < n; i++) { x[i] = i * 0.01; y[i] = Math.Sin(x[i]); }
        var cs = new CubicSpline(x, y);
        var q = new double[300_000];
        var rng = new Random(7);
        for (var i = 0; i < q.Length; i++) q[i] = rng.NextDouble() * 20;
        var par = cs.Evaluate(q);
        var ser = new double[q.Length];
        cs.Evaluate(q.AsSpan(), ser);
        Assert.Equal(ser, par);
        for (var i = 0; i < 1000; i++) Close(Math.Sin(q[i]), par[i], 1e-9, "sin");
    }
}
