using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

// Expected values generated with scipy 1.17.1 (make_interp_spline, make_lsq_spline,
// make_smoothing_spline, BSpline methods).
public class BSplineTests
{
    private static readonly double[] X = [0.0, 0.344828, 0.689655, 1.034483, 1.37931, 1.724138, 2.068966, 2.413793, 2.758621, 3.103448, 3.448276, 3.793103, 4.137931, 4.482759, 4.827586, 5.172414, 5.517241, 5.862069, 6.206897, 6.551724, 6.896552, 7.241379, 7.586207, 7.931034, 8.275862, 8.62069, 8.965517, 9.310345, 9.655172, 10.0];
    private static readonly double[] Y = [0.025146, 0.311614, 0.764356, 0.880578, 0.874589, 1.060585, 1.139259, 0.854645, 0.232931, -0.214949, -0.426553, -0.598123, -1.304493, -1.017509, -1.242554, -1.042495, -0.80207, -0.47204, 0.006112, 0.473825, 0.549917, 1.091447, 0.831323, 1.067335, 1.093015, 0.739001, 0.294586, -0.070162, -0.319906, -0.499982];
    private static readonly double[] Xe = [-0.5, 0.0, 1.3, 4.4, 7.77, 10.0, 10.6];

    [Fact]
    public void MakeInterpSpline_MatchesScipy()
    {
        var s3 = Interpolate.MakeInterpSpline(X, Y);
        AllClose([0.0, 0.0, 0.0, 0.0, 0.689655, 1.034483, 1.37931, 1.724138, 2.068966, 2.413793, 2.758621, 3.103448, 3.448276, 3.793103, 4.137931, 4.482759, 4.827586, 5.172414, 5.517241, 5.862069, 6.206897, 6.551724, 6.896552, 7.241379, 7.586207, 7.931034, 8.275862, 8.62069, 8.965517, 9.310345, 10.0, 10.0, 10.0, 10.0], s3.T, 0, "t3");
        AllClose([0.025146, 0.010496694523232552, 0.8307092102987702, 0.9050543158426292, 0.813955308400295, 1.0866591530473342, 1.202918230640274, 0.9372221731479088, 0.17606059519531445, -0.24387569960891198, -0.49025340701906944, -0.35442840514606266, -1.6807751316219715, -0.7494312507997936, -1.4265553798386028, -0.9996718345950552, -0.8297287241123811, -0.4938317218087565, -0.027184515044427555, 0.6392385926412335, 0.31318096703054094, 1.4075376825120125, 0.605351009000959, 1.1589968821233023, 1.1626728129210975, 0.7484022702071372, 0.27772664533130814, -0.2149689625926011, -0.3897920706310283, -0.499982], s3.C.ToArray(), 1e-10, "c3");
        AllClose([1.2857900939552265, 0.025146, 0.8622915788925177, -1.0854317932835646, 0.9095797551937839, -0.499982, -0.8259337880298809], s3.Evaluate(Xe), 1e-10, "v3");
        AllClose(Y, s3.Evaluate(X), 1e-12, "interpolates");
        AllClose([3.0904156520985597, 0.025146, 0.8605390862178294, -1.0884830397526422, 0.8883150033492307, -0.499982, -0.08532332836283274], Interpolate.MakeInterpSpline(X, Y, 5).Evaluate(Xe), 1e-9, "k5");
        AllClose([0.06633570765002883, 0.025146, 0.8619161527406622, -1.0700111178906053, 0.9256710619577899, -0.499982, -0.6668097322250386], Interpolate.MakeInterpSpline(X, Y, 2).Evaluate(Xe), 1e-10, "k2");
        AllClose([1.124866468628696, 0.025146, 0.862258913486534, -1.0854317930575563, 0.9095697343496396, -0.499982, 0.96582193305861], Interpolate.MakeInterpSpline(X, Y, bcType: SplineBoundaryCondition.Clamped).Evaluate(Xe), 1e-10, "clamped");
        AllClose([-0.49465872370176556, 0.02514600000000002, 0.8619301709056907, -1.0854317905885982, 0.9095799221275936, -0.499982, -0.855782031080345], Interpolate.MakeInterpSpline(X, Y, bcType: SplineBoundaryCondition.Natural).Evaluate(Xe), 1e-10, "natural");
        var bc = (SplineBoundaryCondition.FromDerivatives((1, 0.5), (2, -1.0)), SplineBoundaryCondition.FromDerivatives((1, 0.0), (2, 0.3)));
        var s5d = Interpolate.MakeInterpSpline(X, Y, 5, bcType: bc);
        AllClose([-2.249410020280803, 0.02514599999999999, 0.8592910113272637, -1.0884824285523942, 0.8880337874965675, -0.499982, -9.59998432517186], s5d.Evaluate(Xe), 1e-9, "k5 derivative bcs");
        Close(0.5, s5d.Evaluate(0.0, 1), 1e-9, "k5 first derivative at 0");
        Close(0.3, s5d.Evaluate(10.0, 2), 1e-9, "k5 second derivative at 10");
        // CubicSpline and make_interp_spline build the same not-a-knot spline.
        AllClose(new CubicSpline(X, Y).Evaluate(Xe), s3.Evaluate(Xe), 1e-10, "cubic spline agreement");
    }

    [Fact]
    public void MakeInterpSpline_Periodic_MatchesScipy()
    {
        var yp = (double[])Y.Clone();
        yp[^1] = yp[0];
        var p3 = Interpolate.MakeInterpSpline(X, yp, bcType: SplineBoundaryCondition.Periodic);
        Assert.Equal(InterpolateExtrapolate.Periodic, p3.Extrapolate);
        AllClose([-1.0344829999999998, -0.6896550000000001, -0.3448279999999997, 0.0, 0.344828, 0.689655, 1.034483, 1.37931, 1.724138, 2.068966, 2.413793, 2.758621, 3.103448, 3.448276, 3.793103, 4.137931, 4.482759, 4.827586, 5.172414, 5.517241, 5.862069, 6.206897, 6.551724, 6.896552, 7.241379, 7.586207, 7.931034, 8.275862, 8.62069, 8.965517, 9.310345, 9.655172, 10.0, 10.344828, 10.689655, 11.034483], p3.T, 1e-14, "t");
        AllClose([-0.4940498270997541, 0.10500868177966897, 0.22489156318407702, 0.8651071126920581, 0.9008176195366343, 0.8150905300349557, 1.086354970926852, 1.202999735723288, 0.937200333881176, 0.17606644699640228, -0.2438772675955195, -0.49025298688685365, -0.3544285176918354, -1.680775101571901, -0.7494312584541288, -1.426555379271225, -0.9996718292101997, -0.829728746219125, -0.4938316387663465, -0.027184825106841674, 0.6392397498462642, 0.3131766482631447, 1.407553800340412, 0.6052908563198262, 1.159221374515862, 1.1618349931257308, 0.7515290521366962, 0.2660573597520674, -0.048244423931507796, -0.4940498270997541, 0.10500868177966897, 0.22489156318407702], p3.C.ToArray(), 1e-10, "c");
        AllClose([-0.2664615642816389, 0.025146, 0.8617245422819599, -1.0854317889441814, 0.909643452257404, 0.025146, 0.6593485038382683], p3.Evaluate(Xe), 1e-10, "v");
        AllClose([-0.261232972402754, 0.02514600000000004, 0.8585214020805525, -1.085553100295591, 0.8964961612712748, 0.02514600000000004, 0.6512879324091416], Interpolate.MakeInterpSpline(X, yp, 4, bcType: SplineBoundaryCondition.Periodic).Evaluate(Xe), 1e-10, "k4");
        // 200 points goes through the banded + Woodbury path.
        var xl = new double[200];
        var yl = new double[200];
        for (var i = 0; i < 200; i++) { xl[i] = 7.0 * i / 199; yl[i] = Math.Cos(xl[i]) * Math.Exp(-0.1 * xl[i]); }
        yl[^1] = yl[0];
        AllClose([1.0652164022254664, -0.7099226419444394, 0.8451397127664726, 0.40634688908651473], Interpolate.MakeInterpSpline(xl, yl, bcType: SplineBoundaryCondition.Periodic).Evaluate([0.013, 3.3, 6.99, 8.1]), 1e-10, "large periodic");
        // n <= k goes through the dense formulation.
        AllClose([0.512, -0.11644444444444435, 1.0284444444444443, -0.2640000000000004], Interpolate.MakeInterpSpline([0, 1, 2.5], [1, -1, 1.0], bcType: SplineBoundaryCondition.Periodic).Evaluate([0.3, 1.7, 2.4, 3.1]), 1e-10, "n<=k periodic");
        AllClose(new CubicSpline(X, yp, SplineBoundaryCondition.Periodic).Evaluate(Xe), p3.Evaluate(Xe), 1e-10, "matches CubicSpline periodic");
    }

    [Fact]
    public void MakeLsqSpline_MatchesScipy()
    {
        double[] t = [0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0];
        AllClose([0.043891908177194086, 0.589240105388282, 2.075846785497855, -1.9201900174663133, -0.4996925384160316, 2.1841299871414006, -0.3438710048464065, -0.5040623301331313], Interpolate.MakeLsqSpline(X, Y, t).C.ToArray(), 1e-10, "c");
        var w = new double[30];
        for (var i = 0; i < 30; i++) w[i] = 1 + 2.0 * i / 29;
        AllClose([0.06067575126927935, 0.5459485857719979, 2.1218780735019678, -1.95408081976552, -0.4729508617181328, 2.161586407956106, -0.33223023787040407, -0.5057105977541194], Interpolate.MakeLsqSpline(X, Y, t, w: w).C.ToArray(), 1e-10, "weighted c");
        // Schoenberg-Whitney violated: every interior knot in one data gap.
        Assert.ThrowsAny<ArgumentException>(() => Interpolate.MakeLsqSpline(X, Y, [0, 0, 0, 0, 0.1, 0.12, 0.14, 0.16, 0.18, 10, 10, 10, 10]));
    }

    [Fact]
    public void MakeSmoothingSpline_GcvLambdaAndValues_MatchScipy()
    {
        var s = Interpolate.MakeSmoothingSpline(X, Y, null, null, out var lam);
        Close(0.06732778263081324, lam, 1e-6, "GCV lambda");
        AllClose([-0.384188859043236, 0.06241180785399053, 0.9756813898920506, -1.1401042014150886, 1.0363364209579706, -0.5560540205771412, -1.055274096092365], s.Evaluate(Xe), 1e-7, "values");
        AllClose([-0.0679318160983956, 0.24484605947322458, 0.8848584370009305, -0.9962377634355158, 0.9203436079213188, -0.4958848915507479, -1.026454674968365], Interpolate.MakeSmoothingSpline(X, Y, lam: 0.5).Evaluate(Xe), 1e-10, "lam = 0.5");
        var w = new double[30];
        for (var i = 0; i < 30; i++) w[i] = 1 + 2.0 * i / 29;
        Interpolate.MakeSmoothingSpline(X, Y, w, null, out var lamw);
        Close(0.1156724700500316, lamw, 1e-6, "weighted GCV lambda");
        // lam = 0 interpolates (natural cubic spline).
        AllClose(Y, Interpolate.MakeSmoothingSpline(X, Y, lam: 0.0).Evaluate(X), 1e-9, "lam 0 interpolates");
    }

    [Fact]
    public void BSpline_Methods_MatchScipy()
    {
        double[] t = [0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 4.0, 4.0, 4.0, 4.0];
        var b = new BSpline(t, [1, -1, 2, 0.5, 3, -2], 3);
        double[] xb = [-0.5, 0.0, 0.3, 1.0, 2.0, 3.9, 4.0, 4.5];
        AllClose([6.87875, 1.0, -0.03941000000000008, 0.77, 1.2785185185185184, -1.0812129629629623, -2.0, -9.34837962962963], b.Evaluate(xb), 1e-13, "v");
        AllClose([-18.4725, -6.0, -1.2740999999999996, 1.71, -0.004444444444444362, -8.396944444444443, -10.0, -19.923611111111107], b.Evaluate(xb, 1), 1e-13, "d1");
        AllClose([-22.979999999999997, -22.979999999999997, -22.979999999999997, 4.131111111111111, 4.131111111111111, -12.722222222222221, -12.722222222222221, -12.722222222222221], b.Evaluate(xb, 3), 1e-13, "d3");
        AllClose([-6.0, 3.6, -1.125, 2.5, -10.0, 0.0, 0.0, 0.0], b.Derivative().C.ToArray(), 1e-13, "derivative c");
        AllClose([0.30598437500000003, 0.0, 0.024014655, 0.10850000000000001, 0.8979259259259259, 6.166598282407406, 6.5375, 7.8449508101851855], b.Antiderivative(2).Evaluate(xb), 1e-13, "antiderivative 2");
        AllClose(b.Evaluate(xb), b.Antiderivative(2).Derivative(2).Evaluate(xb), 1e-12, "d2(anti2) = id");
        Close(3.598935185185185, b.Integrate(0.5, 3.5), 1e-13, "integrate");
        Close(1.4746296296296304, b.Integrate(-1, 5), 1e-13, "integrate extrapolated");
        Close(3.625, b.Integrate(-1, 5, InterpolateExtrapolate.None), 1e-13, "integrate clipped");
        var bp = new BSpline(t, [1, -1, 2, 0.5, 3, -2], 3, InterpolateExtrapolate.Periodic);
        AllClose([1.1817129629629628, 1.0, -0.03941000000000008, 0.77, 1.2785185185185184, -1.0812129629629623, 1.0, -0.07874999999999996], bp.Evaluate(xb), 1e-13, "periodic");
        Close(8.775572916666667, bp.Integrate(-1, 9.5), 1e-12, "periodic integral");
        AllClose([0.5, 0.125, 0.7916666666666666, 0.16666666666666666, 0.041666666666666664], BSpline.BasisElement([0, 1, 2, 4]).Evaluate([-1, 0.5, 1.5, 3, 4.5]), 1e-14, "basis element");
        var dm = BSpline.DesignMatrix([0.2, 1.5, 3.9], t, 3);
        AllClose([0.5120000000000001, 0.44448, 0.04272, 0.0008, 0.0, 0.0, 0.0, 0.10666666666666666, 0.5836111111111111, 0.30046296296296293, 0.009259259259259259, 0.0, 0.0, 0.0, 5.55555555555557e-05, 0.006388888888888899, 0.18051851851851863, 0.8130370370370369], dm.ToDense().ToArray(), 1e-14, "design matrix");
        var bi = b.InsertKnot(1.7, 2);
        AllClose([0.0, 0.0, 0.0, 0.0, 1.0, 1.7, 1.7, 2.5, 4.0, 4.0, 4.0, 4.0], bi.T, 0, "inserted knots");
        AllClose([1.0, -1.0, 1.0399999999999998, 1.2592999999999996, 1.297361111111111, 1.0833333333333333, 3.0, -2.0], bi.C.ToArray(), 1e-13, "inserted coefficients");
        AllClose(b.Evaluate(xb), b.ToPPoly().Evaluate(xb), 1e-12, "PPoly agreement");
    }
}
