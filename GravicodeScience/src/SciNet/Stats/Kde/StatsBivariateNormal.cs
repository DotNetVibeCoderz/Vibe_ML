namespace Gravicode.Science.SciNet;

/// <summary>
/// Bivariate standard normal upper-orthant probabilities, Genz's BVNU (Drezner &amp; Wesolowsky 1990
/// as refined by Genz 2004): Gauss–Legendre quadrature of Plackett's identity for |ρ| &lt; 0.925 and a
/// series expansion around ρ = ±1 beyond, accurate to about 1e-15.
/// </summary>
internal static class StatsBivariateNormal
{
    private static readonly double[] W3 = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904];
    private static readonly double[] X3 = [-0.9324695142031522, -0.6612093864662647, -0.2386191860831970];
    private static readonly double[] W6 = [.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659, 0.2334925365383547, 0.2491470458134029];
    private static readonly double[] X6 = [-0.9815606342467191, -0.9041172563704750, -0.7699026741943050, -0.5873179542866171, -0.3678314989981802, -0.1252334085114692];
    private static readonly double[] W10 = [.01761400713915212, .04060142980038694, .06267204833410906, .08327674157670475, 0.1019301198172404,
        0.1181945319615184, 0.1316886384491766, 0.1420961093183821, 0.1491729864726037, 0.1527533871307259];
    private static readonly double[] X10 = [-0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188, -0.7463319064601508,
        -0.6360536807265150, -0.5108670019508271, -0.3737060887154196, -0.2277858511416451, -0.07652652113349733];

    /// <summary>P(X &gt; h, Y &gt; k) for standard normals with correlation <paramref name="r"/>.</summary>
    public static double Upper(double h, double k, double r)
    {
        if (double.IsPositiveInfinity(h) || double.IsPositiveInfinity(k)) return 0.0;
        if (double.IsNegativeInfinity(h) && double.IsNegativeInfinity(k)) return 1.0;
        if (double.IsNegativeInfinity(h)) return Special.Ndtr(-k);
        if (double.IsNegativeInfinity(k)) return Special.Ndtr(-h);
        double[] w, x;
        var ar = Math.Abs(r);
        if (ar < 0.3) { w = W3; x = X3; }
        else if (ar < 0.75) { w = W6; x = X6; }
        else { w = W10; x = X10; }
        var lg = w.Length;
        var hk = h * k;
        double bvn = 0;
        if (ar < 0.925)
        {
            var hs = (h * h + k * k) / 2;
            var asr = Math.Asin(r);
            for (var i = 0; i < lg; i++)
            {
                var sn = Math.Sin(asr * (1 - x[i]) / 2);
                bvn += w[i] * Math.Exp((sn * hk - hs) / (1 - sn * sn));
                sn = Math.Sin(asr * (1 + x[i]) / 2);
                bvn += w[i] * Math.Exp((sn * hk - hs) / (1 - sn * sn));
            }
            return bvn * asr / (4 * Math.PI) + Special.Ndtr(-h) * Special.Ndtr(-k);
        }
        if (r < 0) { k = -k; hk = -hk; }
        if (ar < 1)
        {
            var aS = (1 - r) * (1 + r);
            var a = Math.Sqrt(aS);
            var bs = (h - k) * (h - k);
            var c = (4 - hk) / 8;
            var d = (12 - hk) / 16;
            bvn = a * Math.Exp(-(bs / aS + hk) / 2) * (1 - c * (bs - aS) * (1 - d * bs / 5) / 3 + c * d * aS * aS / 5);
            if (hk > -160)
            {
                var b = Math.Sqrt(bs);
                bvn -= Math.Exp(-hk / 2) * Math.Sqrt(2 * Math.PI) * Special.Ndtr(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3);
            }
            a /= 2;
            for (var i = 0; i < lg; i++)
                for (var sgn = -1; sgn <= 1; sgn += 2)
                {
                    var xs = a * (sgn * x[i] + 1);
                    xs *= xs;
                    var rs = Math.Sqrt(1 - xs);
                    bvn += a * w[i] * (Math.Exp(-bs / (2 * xs) - hk / (1 + rs)) / rs - Math.Exp(-(bs / xs + hk) / 2) * (1 + c * xs * (1 + d * xs)));
                }
            bvn = -bvn / (2 * Math.PI);
        }
        if (r > 0) bvn += Special.Ndtr(-Math.Max(h, k));
        else
        {
            bvn = -bvn;
            if (k > h) bvn += Special.Ndtr(k) - Special.Ndtr(h);
        }
        return bvn;
    }
}
