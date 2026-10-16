namespace Gravicode.Science.SciNet;

/// <summary>Implicit QL iteration on a symmetric tridiagonal matrix (EISPACK <c>tql2</c>).</summary>
/// <remarks>
/// GraviNum has the same kernel but keeps it internal, and it walks eigenvectors by column of a
/// row-major buffer. Here the vectors are accumulated as <em>rows</em> of <c>vt</c>, so each plane
/// rotation updates two contiguous rows. Eigenvalues come out unsorted: blocks deflate as they
/// converge, in no relation to magnitude — callers sort and carry the rows.
/// </remarks>
internal static class LinalgTridiagonal
{
    /// <param name="d">Diagonal in, eigenvalues out.</param>
    /// <param name="e">Off-diagonal in <c>e[0..n-2]</c> (<c>e[i] = T[i, i+1]</c>); destroyed.</param>
    /// <param name="vt">Null, or the identity in and eigenvectors as rows out.</param>
    /// <param name="n">Order.</param>
    public static void Tql2(double[] d, double[] e, double[]? vt, int n)
    {
        if (n == 0) return;
        e[n - 1] = 0.0;
        var f = 0.0;
        var tst1 = 0.0;
        const double eps = LinalgKernels.Eps;
        for (var l = 0; l < n; l++)
        {
            tst1 = Math.Max(tst1, Math.Abs(d[l]) + Math.Abs(e[l]));
            var m = l;
            while (m < n)
            {
                if (Math.Abs(e[m]) <= eps * tst1) break;
                m++;
            }
            if (m == n) m = n - 1;

            if (m > l)
            {
                var iter = 0;
                do
                {
                    if (++iter > 300) throw new LinalgException("EighTridiagonal: QL iteration did not converge.");
                    var g = d[l];
                    var p = (d[l + 1] - g) / (2.0 * e[l]);
                    var r = Hypot(p, 1.0);
                    if (p < 0) r = -r;
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    var dl1 = d[l + 1];
                    var h = g - d[l];
                    for (var i = l + 2; i < n; i++) d[i] -= h;
                    f += h;

                    p = d[m];
                    var c = 1.0;
                    var c2 = c;
                    var c3 = c;
                    var el1 = e[l + 1];
                    var s = 0.0;
                    var s2 = 0.0;
                    for (var i = m - 1; i >= l; i--)
                    {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        if (vt is not null)
                        {
                            var ri = vt.AsSpan(i * n, n);
                            var ri1 = vt.AsSpan((i + 1) * n, n);
                            for (var k = 0; k < n; k++)
                            {
                                var hk = ri1[k];
                                ri1[k] = s * ri[k] + c * hk;
                                ri[k] = c * ri[k] - s * hk;
                            }
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                } while (Math.Abs(e[l]) > eps * tst1);
            }
            d[l] += f;
            e[l] = 0.0;
        }
    }

    private static double Hypot(double a, double b)
    {
        var x = Math.Abs(a);
        var y = Math.Abs(b);
        var w = Math.Max(x, y);
        var z = Math.Min(x, y);
        if (z == 0.0) return w;
        var r = z / w;
        return w * Math.Sqrt(1.0 + r * r);
    }
}
