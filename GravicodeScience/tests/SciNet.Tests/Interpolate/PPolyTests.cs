using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

public class PPolyTests
{
    private static readonly double[,] Coef = { { 1, 2, -1 }, { 0, 1, 0.5 }, { -1, 0, 2 } };
    private static readonly double[] Bp = [0, 1, 3, 4];
    private static readonly double[] Xe = [-1, 0, 0.5, 1, 2, 3, 3.5, 4, 5];

    // Expected values: scipy 1.17.1, PPoly(Coef, Bp) and friends.
    [Fact]
    public void Evaluate_Derivatives_Antiderivative_MatchScipy()
    {
        var p = new PPoly(Coef, Bp);
        AllClose([0.0, -1.0, -0.75, 0.0, 3.0, 2.0, 2.0, 1.5, -1.0], p.Evaluate(Xe), 1e-15, "v");
        AllClose([-2.0, 0.0, 1.0, 1.0, 5.0, 0.5, -0.5, -1.5, -3.5], p.Evaluate(Xe, 1), 1e-15, "d1");
        AllClose([2.0, 2.0, 2.0, 4.0, 4.0, -2.0, -2.0, -2.0, -2.0], p.Evaluate(Xe, 2), 1e-15, "d2");
        AllClose([2.0, 4.0, -2.0, 0.0, 1.0, 0.5], p.Derivative().C.ToArray(), 1e-15, "derivative c");
        var a2 = p.Antiderivative(2);
        AllClose([-0.4166666666666667, 0.0, -0.11979166666666667, -0.4166666666666667, -0.7500000000000002, 2.2499999999999996, 5.838541666666666, 9.916666666666666, 18.916666666666668], a2.Evaluate(Xe), 1e-14, "anti2");
        AllClose([0.08333333333333333, 0.16666666666666666, -0.08333333333333333, 0.0, 0.16666666666666666, 0.08333333333333333, -0.5, 0.0, 1.0, 0.0, -0.6666666666666667, 6.666666666666666, 0.0, -0.4166666666666667, 2.2499999999999996], a2.C.ToArray(), 1e-14, "anti2 c");
        // The derivative of the antiderivative is the original.
        AllClose(p.Evaluate(Xe), a2.Derivative(2).Evaluate(Xe), 1e-13, "d(anti) = p");
    }

    [Fact]
    public void Integrate_Roots_Solve_Periodic_MatchScipy()
    {
        var p = new PPoly(Coef, Bp);
        Close(8.333333333333334, p.Integrate(-1, 5), 1e-14, "int");
        Close(-8.333333333333334, p.Integrate(5, -1), 1e-14, "int reversed");
        Close(0.958333333333333, p.Integrate(0.5, 2, InterpolateExtrapolate.None), 1e-14, "int inside");
        Assert.True(double.IsNaN(p.Integrate(-1, 2, InterpolateExtrapolate.None)), "out of range without extrapolation is NaN");
        AllClose([-1.0, 1.0, 4.686140661634507], p.Roots(), 1e-14, "roots");
        AllClose([1.0], p.Roots(extrapolate: false), 1e-14, "roots no extrapolation");
        AllClose([-1.224744871391589, 1.3090169943749475, 4.5], p.Solve(0.5), 1e-14, "solve 0.5");

        var pp = new PPoly(Coef, Bp, InterpolateExtrapolate.Periodic);
        AllClose([2.0, -1.0, -0.75, 0.0, 3.0, 2.0, 2.0, -1.0, 0.0], pp.Evaluate(Xe), 1e-15, "periodic");
        Close(18.625, pp.Integrate(-1, 9.5), 1e-13, "periodic integral");
    }

    [Fact]
    public void DescendingBreakpoints_MatchScipy()
    {
        var p = new PPoly(Coef, [4, 3, 1, 0]);
        AllClose([-3.0, 0.5, 1.5, 2.0, 1.0, 0.0, -0.75, -1.0, 0.0], p.Evaluate(Xe), 1e-15, "v");
        AllClose([4.5, 2.5, 1.5, 0.5, -3.0, 1.0, -1.0, 0.0, 2.0], p.Evaluate(Xe, 1), 1e-15, "d1");
        Close(2.333333333333333, p.Integrate(-1, 5), 1e-14, "int");
        // scipy reports each interval's roots in ascending value order and only drops a root
        // equal to the one just before it, hence 3.0 twice.
        AllClose([3.0, 5.0, 2.5, 3.0, -0.18614066163450715], p.Roots(), 1e-14, "roots");
    }

    [Fact]
    public void Roots_DiscontinuityAndIdenticallyZeroPieces()
    {
        var step = new PPoly(new double[,] { { 0, 0, 0 }, { 1, -1, 1 } }, [0, 1, 2, 3]);
        AllClose([1, 2], step.Roots(), 0, "sign changes at jumps");
        Assert.Empty(step.Roots(discontinuity: false));
        var zero = new PPoly(new double[,] { { 0, 1 }, { 0, -1 } }, [0, 1, 2]);
        AllClose([0, double.NaN, 2], zero.Roots(), 0, "identically zero piece (scipy: [0, nan, 2])");
        var dbl = new PPoly(new double[,] { { 1 }, { -2 }, { 1 } }, [0, 3]);
        AllClose([1], dbl.Roots(), 1e-15, "double root reported once");
        // x(x^2-1)(x^2-4) goes through the companion-matrix path.
        var quintic = new PPoly(new double[,] { { 1 }, { 0 }, { -5 }, { 0 }, { 4 }, { 0 } }, [0, 3]); // local variable s = x here
        AllClose([-2, -1, 0, 1, 2], quintic.Roots(), 1e-13, "quintic");
    }

    [Fact]
    public void Extend_MatchesScipy()
    {
        var p = new PPoly(Coef, Bp);
        p.Extend(new double[,] { { 1, 0 }, { 2, 1 } }, [4, 6]);
        AllClose([0.0, 1.0, 3.0, 4.0, 4.0, 6.0], p.X, 0, "x");
        AllClose([1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0, -1.0, 0.0, 2.0, 2.0, 1.0], p.C.ToArray(), 0, "c");
        AllClose([1.0, 1.0, 1.0, 0.0], p.Evaluate([4.5, 5, 7, -1]), 1e-15, "v");
        Assert.Throws<ArgumentException>(() => p.Extend(new double[,] { { 1 } }, [2]));
    }

    [Fact]
    public void FromSpline_MatchesScipy()
    {
        double[] t = [0, 0, 0, 0, 1, 2.5, 4, 4, 4, 4];
        double[] c = [1, -1, 2, 0.5, 3, -2];
        var ps = PPoly.FromSpline(t, c, 3);
        AllClose(t, ps.X, 0, "x");
        AllClose([-3.83, -3.83, -3.83, -3.83, 0.6885185185185185, -2.1203703703703707, -2.1203703703703707, -2.1203703703703707, -2.1203703703703707, 9.6, 9.6, 9.6, 9.6, -1.89, 1.2083333333333333, -8.333333333333334, -8.333333333333334, -8.333333333333334, -6.0, -6.0, -6.0, -6.0, 1.7100000000000006, 0.6875, -10.0, -10.0, -10.0, 1.0, 1.0, 1.0, 1.0, 0.77, 1.40625, -2.0, -2.0, -2.0], ps.C.ToArray(), 1e-13, "c");
        AllClose([6.87875, -0.039409999999999834, 0.77, 1.278518518518519, -1.0812129629629634, -2.0, -9.34837962962963], ps.Evaluate([-0.5, 0.3, 1, 2, 3.9, 4, 4.5]), 1e-13, "v");
    }

    [Fact]
    public void BPoly_MatchesScipy()
    {
        var b = new BPoly(new double[,] { { 1, 2 }, { 3, -1 }, { 0.5, 2 } }, [0, 1, 3]);
        double[] xb = [-0.5, 0, 0.3, 1, 2, 3, 3.5];
        AllClose([-2.125, 1.0, 1.7949999999999997, 2.0, 0.5, 2.0, 3.875], b.Evaluate(xb), 1e-14, "v");
        AllClose([8.5, 4.0, 1.2999999999999998, -3.0, 0.0, 3.0, 4.5], b.Evaluate(xb, 1), 1e-14, "d1");
        AllClose([-9.0, -9.0, -9.0, 3.0, 3.0, 3.0, 3.0], b.Evaluate(xb, 2), 1e-14, "d2");
        AllClose([0.1875, 0.0, 0.43949999999999995, 1.5, 2.5, 3.5, 4.9375], b.Antiderivative().Evaluate(xb), 1e-14, "anti");
        AllClose([0.018229166666666664, 0.0, 0.0599625, 0.7916666666666666, 2.9166666666666665, 5.791666666666667, 7.861979166666667], b.Antiderivative(2).Evaluate(xb), 1e-14, "anti2");
        Close(4.75, b.Integrate(-0.5, 3.5), 1e-14, "int");
        var bp = new BPoly(new double[,] { { 1, 2 }, { 3, -1 }, { 0.5, 2 } }, [0, 1, 3], InterpolateExtrapolate.Periodic);
        Close(9.666, bp.Integrate(-0.5, 7.3), 1e-13, "periodic int");

        AllClose([-4.5, 1.5, 4.0, -3.0, 1.0, 2.0], PPoly.FromBernsteinBasis(b).C.ToArray(), 1e-14, "to power");
        var p = new PPoly(Coef, Bp);
        var tb = BPoly.FromPowerBasis(p);
        AllClose([-1.0, 0.0, 2.0, -1.0, 1.0, 2.25, 0.0, 10.0, 1.5], tb.C.ToArray(), 1e-14, "to bernstein");
        AllClose(p.Evaluate(Xe), tb.Evaluate(Xe), 1e-13, "round trip values");

        var be = new BPoly(new double[,] { { 1, 2 }, { 3, -1 }, { 0.5, 2 } }, [0, 1, 3]);
        be.Extend(new double[,] { { 1 }, { 0 } }, [4]);
        Assert.Equal([3, 3], be.C.Shape.ToArray());
        AllClose([0.5, 0.0, 1.875], be.Evaluate([3.5, 4, 0.5]), 1e-14, "extend");
    }

    [Fact]
    public void BPoly_FromDerivatives_MatchesScipy()
    {
        double[][] yi = [[1, 0], [2, 1, -1], [0, 3]];
        var fd = BPoly.FromDerivatives([0, 1, 3], yi);
        Assert.Equal([5, 2], fd.C.Shape.ToArray());
        AllClose([1.0, 2.0, 1.0, 2.5, 1.4166666666666665, 2.6666666666666665, 1.75, -1.5, 2.0, 0.0], fd.C.ToArray(), 1e-14, "c");
        AllClose([1.9062499999999991, 1.0, 1.17505, 2.0, 1.375, 0.0, 4.3046875], fd.Evaluate([-0.5, 0, 0.3, 1, 2, 3, 3.5]), 1e-13, "v");
        // The prescribed derivatives hold at the nodes.
        Close(1, fd.Evaluate(1.0, 1), 1e-13, "f'(1)");
        Close(3, fd.Evaluate(3.0, 1), 1e-13, "f'(3)");
        Close(-1, fd.Evaluate(1.0, 2), 1e-12, "f''(1)");
        var fo = BPoly.FromDerivatives([0, 1, 3], yi, orders: [3, 2]);
        AllClose([1.0, 2.0, 1.0, -0.5, 1.3333333333333335, -1.6666666666666667, 1.75, -1.5, 2.0, 0.0], fo.C.ToArray(), 1e-14, "orders");
    }
}
