using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SparseMatrixFunctionTests
{
    /// <summary>exp(tA) for symmetric A from its eigendecomposition: an independent reference.</summary>
    private static NdArray SymmetricExpm(NdArray a, double t)
    {
        var e = Decomposition.SymmetricEigen(a);
        var n = a.Shape[0];
        var v = e.Vectors.AsContiguous().ToArray();
        var l = e.Values.ToArray();
        var r = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < n; k++) s += v[i * n + k] * Math.Exp(t * l[k]) * v[j * n + k];
                r[i * n + j] = s;
            }
        return new NdArray(r, n, n);
    }

    [Fact]
    public void ExpmMultiply_Symmetric_MatchesEigendecomposition()
    {
        var a = Poisson2D(7, 6).Scale(-0.4); // heat-equation generator
        var n = a.Rows;
        var b = Enumerable.Range(0, n).Select(i => Math.Sin(0.3 * i) + 0.1).ToArray();
        foreach (var t in new[] { 0.1, 1.0, 25.0 })
        {
            var reference = LinAlg.Dot(SymmetricExpm(a.ToDense(), t), new NdArray(b, n)).ToArray();
            var x = SparseLinalg.ExpmMultiply(a, b, t);
            var scale = reference.Max(Math.Abs);
            Close(reference, x, 1e-12 * Math.Max(1, scale), $"t={t}");
        }
    }

    [Fact]
    public void ExpmMultiply_NonSymmetric_MatchesDenseExpm()
    {
        var a = ConvectionDiffusion(6, 8).Scale(-0.2);
        var b = new NdArray(Enumerable.Range(0, 72).Select(i => (double)(i % 7) - 3).ToArray(), 36, 2);
        var reference = LinAlg.Dot(Linalg.Expm(a.ToDense().Copy()), b);
        Close(reference, SparseLinalg.ExpmMultiply(a, b), 1e-12, "block");
        // A large-norm generator takes the estimated-power branch of fragment 3.1.
        var big = Poisson2D(5, 5).Scale(-6.0).Add(SparseMatrices.Random(25, 25, 0.2, 3));
        var v = Enumerable.Range(0, 25).Select(i => 1.0 / (i + 1)).ToArray();
        var refBig = LinAlg.Dot(Linalg.Expm(big.ToDense().Copy()), new NdArray(v, 25)).ToArray();
        Close(refBig, SparseLinalg.ExpmMultiply(big, v), 1e-11, "large norm");
        Close(refBig, SparseLinalg.ExpmMultiply(big.AsLinearOperator(), v, traceA: big.Trace()), 1e-11, "operator form");
        Close(refBig, SparseLinalg.ExpmMultiply(big.AsLinearOperator(), v), 1e-11, "operator form, no shift");
    }

    [Fact]
    public void ExpmMultiply_Interval_MatchesPointwise()
    {
        var a = Poisson2D(4, 4).Scale(-1.0);
        var b = Enumerable.Range(0, 16).Select(i => (double)i).ToArray();
        var grid = SparseLinalg.ExpmMultiply(a, b, start: 0, stop: 2, num: 5);
        Assert.Equal([5, 16], grid.Shape.ToArray());
        for (var q = 0; q < 5; q++)
        {
            var t = 0.5 * q;
            var expected = LinAlg.Dot(SymmetricExpm(a.ToDense(), t), new NdArray(b, 16)).ToArray();
            Close(expected, grid.Row(q).ToArray(), 1e-11, $"t={t}");
        }
    }

    [Fact]
    public void Expm_MatchesDenseExpm_AcrossPadeDegrees()
    {
        foreach (var scale in new[] { 0.001, 0.05, 0.3, 0.8, 3.0, 40.0 })
        {
            var a = ConvectionDiffusion(4, 3).Scale(-scale / 4).Add(SparseMatrices.Random(16, 16, 0.1, 7).Scale(scale / 10));
            var expected = Linalg.Expm(a.ToDense().Copy());
            var actual = SparseLinalg.Expm(a).ToDense();
            var tol = 1e-12 * Math.Max(1, MaxAbs(expected));
            Close(expected, actual, tol, $"scale {scale}");
        }
        // Block diagonal stays block diagonal: the case where sparse expm is worth it.
        var blocks = SparseMatrices.BlockDiag(Laplacian1D(3), Laplacian1D(2));
        var e = SparseLinalg.Expm(blocks);
        Assert.Equal(13, e.Nnz);
    }

    [Fact]
    public void Onenormest_IsALowerBound_AndUsuallyExact()
    {
        var exactHits = 0;
        for (var seed = 0; seed < 10; seed++)
        {
            var a = SparseMatrices.Random(60, 60, 0.08, seed + 200).Add(SparseMatrices.Eye(60).Scale(-0.3));
            var exact = a.Norm(SparseNormKind.One);
            var est = SparseLinalg.Onenormest(a);
            Assert.True(est <= exact * (1 + 1e-14), $"estimate {est} exceeds exact {exact}");
            Assert.True(est >= exact / 3, $"estimate {est} far below exact {exact}");
            if (Math.Abs(est - exact) <= 1e-12 * exact) exactHits++;
        }
        Assert.True(exactHits >= 7, $"only {exactHits}/10 exact");
        var small = CsrMatrix.FromDense(new NdArray([1.0, -2, 3, 4], 2, 2));
        Assert.Equal(6.0, SparseLinalg.Onenormest(small));
    }

    [Fact]
    public void MatrixPower_MatchesRepeatedProduct()
    {
        var a = CsrMatrix.FromDense(RandomDense(6, 6, 0.4, 111));
        var d = a.ToDense();
        var expected = NdArray.Eye(6);
        for (var p = 0; p <= 5; p++)
        {
            Close(expected, SparseLinalg.MatrixPower(a, p).ToDense(), 1e-12, $"p={p}");
            expected = NaiveMatMul(expected, d);
        }
    }
}
