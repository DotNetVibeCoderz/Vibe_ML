using System.Collections.Concurrent;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Quadrature node tables. The Gauss-Kronrod tables are QUADPACK's (Piessens et al. 1983),
/// checked here for exactness on monomials up to degree 3n+1; Gauss-Legendre rules of any
/// order are computed by Newton iteration on the three-term recurrence and cached.
/// </summary>
internal static class IntegrateRules
{
    // 21-point Kronrod abscissae (positive half, descending, centre last); odd 0-based
    // indices 1,3,5,7,9 are the 10-point Gauss nodes.
    internal static readonly double[] Xgk21 =
    [
        0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
        0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
        0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
        0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
        0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0,
    ];

    internal static readonly double[] Wgk21 =
    [
        0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
        0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
        0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
        0.123491976262065851077208465292940, 0.134709217311473325928054001771707,
        0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
    ];

    internal static readonly double[] Wg10 =
    [
        0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
        0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
    ];

    // 15-point Kronrod; odd indices 1,3,5 plus the centre are the 7-point Gauss nodes.
    internal static readonly double[] Xgk15 =
    [
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0.0,
    ];

    internal static readonly double[] Wgk15 =
    [
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
    ];

    /// <summary>7-point Gauss weights for nodes Xgk15[1], [3], [5] and the centre.</summary>
    internal static readonly double[] Wg7 =
    [
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
    ];

    /// <summary>cos(k pi / 24), k = 1..11: the Clenshaw-Curtis nodes of QUADPACK's 25-point rules.</summary>
    internal static readonly double[] Xcc =
    [
        0.991444861373810411144557526928563, 0.965925826289068286749743199728897,
        0.923879532511286756128183189396788, 0.866025403784438646763723170752936,
        0.793353340291235164579776961501299, 0.707106781186547524400844362104849,
        0.608761429008720639416097542898164, 0.500000000000000000000000000000000,
        0.382683432365089771728459984030399, 0.258819045102520762348898837624048,
        0.130526192220051591548406227895489,
    ];

    private static readonly ConcurrentDictionary<int, (double[] X, double[] W)> GaussCache = new();

    /// <summary>
    /// n-point Gauss-Legendre nodes (ascending) and weights on [-1, 1]. Newton on P_n from
    /// Tricomi's initial guess converges in a handful of iterations for any n; the weights use
    /// the derivative at the converged node, 2 / ((1 - x^2) P_n'(x)^2).
    /// </summary>
    internal static (double[] X, double[] W) GaussLegendre(int n)
    {
        if (n < 1) throw new ArgumentException("Gauss-Legendre order must be at least 1.", nameof(n));
        return GaussCache.GetOrAdd(n, static n =>
        {
            var x = new double[n];
            var w = new double[n];
            var m = (n + 1) / 2;
            for (var i = 0; i < m; i++)
            {
                var z = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5));
                double dp = 0;
                for (var it = 0; it < 100; it++)
                {
                    var (p, d) = Legendre(n, z);
                    dp = d;
                    var dz = p / d;
                    z -= dz;
                    if (Math.Abs(dz) <= 1e-16 * Math.Max(1.0, Math.Abs(z))) break;
                }
                dp = Legendre(n, z).Dp;
                x[i] = -z;
                x[n - 1 - i] = z;
                var wi = 2.0 / ((1 - z * z) * dp * dp);
                w[i] = wi;
                w[n - 1 - i] = wi;
            }
            if (n % 2 == 1) x[n / 2] = 0.0;
            return (x, w);
        });
    }

    private static (double P, double Dp) Legendre(int n, double x)
    {
        double p0 = 1, p1 = x;
        if (n == 0) return (1, 0);
        for (var k = 2; k <= n; k++)
        {
            var p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        var dp = n * (x * p1 - p0) / (x * x - 1);
        return (p1, dp);
    }
}
