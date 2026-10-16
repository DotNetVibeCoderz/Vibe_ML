using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Quasi-Monte Carlo engines and discrepancies against scipy 1.17.1.</summary>
public class StatsQmcTests
{
    private static double[] Columns(NdArray a, int[] cols)
    {
        var n = a.Shape[0];
        var d = a.Shape[1];
        var x = Flat(a);
        var r = new double[n * cols.Length];
        for (var i = 0; i < n; i++)
            for (var c = 0; c < cols.Length; c++) r[i * cols.Length + c] = x[i * d + cols[c]];
        return r;
    }

    [Fact]
    public void UnscrambledSobolIsBitIdenticalToScipy()
    {
        var e = new SobolEngine(10, scramble: false);
        Close("sobol10", 0, 0, Flat(e.Random(64)));
        Close("sobol10_next", 0, 0, Flat(e.Random(5)));
        var big = new SobolEngine(4096, scramble: false);
        Close("sobol4096", 0, 0, Columns(big.Random(256), [0, 1, 2, 9, 100, 1110, 2047, 4095]));
        Close("sobol_bits64", 0, 0, Flat(new SobolEngine(3, false, bits: 64).Random(16)));
        Close("sobol_bits10", 0, 0, Flat(new SobolEngine(3, false, bits: 10).Random(8)));
        var ff = new SobolEngine(5, false);
        ff.FastForward(37);
        Close("sobol_ff", 0, 0, Flat(ff.Random(9)));
        var b2 = new SobolEngine(5, false);
        Close("sobol_b2", 0, 0, [.. Flat(b2.RandomBase2(3)), .. Flat(b2.RandomBase2(3))]);
        Assert.Throws<ArgumentException>(() => b2.RandomBase2(2)); // 16 + 4 is not a power of two
        var rs = new SobolEngine(6, false);
        rs.Random(3);
        rs.Reset();
        Close("sobol_reset", 0, 0, Flat(rs.Random(4)));
        var it = new SobolEngine(2, false).Integers([0, 5], [10, 8], 8);
        var flat = new double[16];
        for (var i = 0; i < 8; i++) { flat[2 * i] = it[i, 0]; flat[2 * i + 1] = it[i, 1]; }
        Close("sobol_int", 0, 0, flat);
    }

    [Fact]
    public void ParallelSobolDrawEqualsSequentialDraws()
    {
        // A big draw goes through the chunked Gray-code path; many small draws through one chain.
        var big = Flat(new SobolEngine(7, false).Random(1 << 16));
        var small = new SobolEngine(7, false);
        for (var c = 0; c < 64; c++)
        {
            var part = Flat(small.Random(1024));
            for (var i = 0; i < part.Length; i++)
                Assert.True(part[i] == big[c * 1024 * 7 + i], $"chunk {c} element {i}");
        }
    }

    [Fact]
    public void ScrambledSobolKeepsTheNetStructure()
    {
        // For 2^m points of a (t, m, s)-net in base 2, every 1-D projection has exactly one point in each
        // interval [k/2^m, (k+1)/2^m), and for dims 0 and 1 (t = 0) every 2-D elementary box of
        // volume 2^-m holds exactly one point. Scrambling must preserve both.
        const int m = 10;
        var e = new SobolEngine(8, scramble: true, seed: 12345);
        var x = Flat(e.RandomBase2(m));
        var n = 1 << m;
        for (var j = 0; j < 8; j++)
        {
            var seen = new bool[n];
            for (var i = 0; i < n; i++)
            {
                var k = (int)(x[i * 8 + j] * n);
                Assert.False(seen[k], $"dim {j}: two points in stratum {k}");
                seen[k] = true;
            }
        }
        for (var a = 0; a <= m; a++)
        {
            var b = m - a;
            var seen = new HashSet<(int, int)>();
            for (var i = 0; i < n; i++)
                Assert.True(seen.Add(((int)(x[i * 8] * (1 << a)), (int)(x[i * 8 + 1] * (1 << b)))), $"box {a}x{b} hit twice");
        }
        // Reset reproduces the same scrambling; a different seed gives different points.
        var again = Flat(e.Reset().Random(4));
        Assert.Equal(x[..32], again);
        Assert.NotEqual(x[..32], Flat(new SobolEngine(8, seed: 1).Random(4)));
        // The scrambled mean of a smooth integrand converges much faster than Monte Carlo's 1/√n.
        var est = 0.0;
        for (var i = 0; i < n; i++) est += Math.Exp(x[i * 8] + x[i * 8 + 1]);
        est /= n;
        var exact = Math.Pow(Math.E - 1, 2);
        Assert.True(Math.Abs(est - exact) < 2e-4, $"QMC estimate {est} vs {exact}");
    }

    [Fact]
    public void HaltonMatchesScipyAndScrambledIsBalanced()
    {
        var h = new HaltonEngine(6, scramble: false);
        Close("halton", 1e-15, 0, Flat(h.Random(40)));
        Close("halton_next", 1e-15, 0, Flat(h.Random(3)));
        // Scrambled: base-2 coordinate of 2^k points still has one point per dyadic stratum.
        var s = Flat(new HaltonEngine(3, seed: 5).Random(64));
        var seen = new bool[64];
        for (var i = 0; i < 64; i++)
        {
            var k = (int)(s[i * 3] * 64);
            Assert.False(seen[k]);
            seen[k] = true;
        }
        // Base 3, 27 points: one per ternary stratum.
        var seen3 = new bool[27];
        var s3 = Flat(new HaltonEngine(2, seed: 9).Random(27));
        for (var i = 0; i < 27; i++) seen3[(int)(s3[i * 2 + 1] * 27)] = true;
        Assert.All(seen3, Assert.True);
    }

    [Fact]
    public void LatinHypercubeStratifiesEveryAxis()
    {
        foreach (var opt in new string?[] { null, "random-cd" })
        {
            const int n = 50, d = 4;
            var e = new LatinHypercubeEngine(d, optimization: opt, seed: 3);
            var x = Flat(e.Random(n));
            for (var j = 0; j < d; j++)
            {
                var strata = Enumerable.Range(0, n).Select(i => (int)(x[i * d + j] * n)).OrderBy(v => v);
                Assert.Equal(Enumerable.Range(0, n), strata);
            }
            if (opt is not null)
            {
                var plain = Qmc.Discrepancy(new LatinHypercubeEngine(d, seed: 3).Random(n));
                var optimised = Qmc.Discrepancy(new NdArray(x, n, d));
                Assert.True(optimised < plain, $"random-cd did not reduce CD: {optimised} vs {plain}");
            }
        }
        var centred = Flat(new LatinHypercubeEngine(2, scramble: false, seed: 1).Random(5));
        Assert.All(centred, v => Assert.Equal(0.5, v * 5 - Math.Floor(v * 5), 12));
        // Strength 2: n = p², every pair of columns is balanced on the p × p grid.
        var oa = Flat(new LatinHypercubeEngine(4, strength: 2, seed: 2).Random(49));
        for (var a = 0; a < 4; a++)
            for (var b = a + 1; b < 4; b++)
            {
                var cells = new HashSet<(int, int)>();
                for (var i = 0; i < 49; i++) cells.Add(((int)(oa[i * 4 + a] * 7), (int)(oa[i * 4 + b] * 7)));
                Assert.Equal(49, cells.Count);
            }
        Assert.Throws<ArgumentException>(() => new LatinHypercubeEngine(3, strength: 2).Random(10));
    }

    [Fact]
    public void DiscrepanciesMatchScipy()
    {
        var s = new NdArray(R.DSAMP, 30, 3);
        foreach (var m in new[] { "CD", "WD", "MD", "L2-star" })
            Close($"disc/{m}", 1e-12, 1e-15, Qmc.Discrepancy(s, method: m), Qmc.Discrepancy(s, true, m));
        var init = Qmc.Discrepancy(s, iterative: true);
        var appended = new NdArray([.. R.DSAMP, .. R.XNEW], 31, 3);
        Close("disc_update", 1e-12, 1e-15, Qmc.UpdateDiscrepancy(R.XNEW, s, init), Qmc.Discrepancy(appended));
        var head = new NdArray(R.DSAMP[..12], 4, 3);
        var scaled = Qmc.Scale(head, [-1.0, 0, 2], [1.0, 10, 2.5]);
        Close("scale", 1e-14, 1e-15, [.. Flat(scaled), .. Flat(Qmc.Scale(scaled, [-1.0, 0, 2], [1.0, 10, 2.5], reverse: true))]);
        Assert.Throws<ArgumentException>(() => Qmc.Scale(scaled, [0.0, 0, 0], [1.0, 1, 1]));
    }

    [Fact]
    public void QmcSamplersMatchScipyWithUnscrambledEngines()
    {
        var cov = new NdArray([2.0, 0.3, 0.1, 0.3, 1.0, -0.2, 0.1, -0.2, 0.5], 3, 3);
        var mv = new MultivariateNormalQmc([1.0, -2.0, 0.5], cov, engine: new SobolEngine(3, false));
        Close("mvnqmc", 1e-12, 1e-12, Flat(mv.Random(16)));
        var bm = new MultivariateNormalQmc([1.0, -2.0, 0.5], cov, invTransform: false, engine: new SobolEngine(4, false));
        Close("mvnqmc_bm", 1e-12, 1e-12, Flat(bm.Random(15))[3..]);
        // A singular covariance goes through the eigen root; the sign of the eigenvector is arbitrary,
        // so check the defining property (both coordinates equal) and the magnitudes.
        var sg = Flat(new MultivariateNormalQmc([0.0, 0.0], new NdArray([1.0, 1, 1, 1], 2, 2), engine: new SobolEngine(2, false)).Random(8))[2..];
        var expected = StatsTestsRef.Expected["mvnqmc_singular"];
        for (var i = 0; i < sg.Length; i++) Near(Math.Abs(expected[i]), Math.Abs(sg[i]), 1e-12, 1e-12, "singular |x|");
        for (var i = 0; i < sg.Length; i += 2) Near(sg[i], sg[i + 1], 1e-12, 1e-12, "x1 == x2");
        var mq = new MultinomialQmc([0.2, 0.3, 0.5], 10, new SobolEngine(1, false));
        Close("multinomialqmc", 0, 0, Flat(mq.Random(4)));
        // Default engines: scrambled, so check moments instead.
        var big = Flat(new MultivariateNormalQmc([1.0, -2.0, 0.5], cov, seed: 4).Random(4096));
        var m0 = Enumerable.Range(0, 4096).Average(i => big[i * 3]);
        var v0 = Enumerable.Range(0, 4096).Average(i => (big[i * 3] - m0) * (big[i * 3] - m0));
        Assert.True(Math.Abs(m0 - 1.0) < 0.01 && Math.Abs(v0 - 2.0) < 0.02, $"mean {m0} var {v0}");
    }

    [Fact]
    public void PoissonDiskRespectsTheRadius()
    {
        var e = new PoissonDiskEngine(2, radius: 0.1, seed: 7);
        var x = Flat(e.FillSpace());
        var n = x.Length / 2;
        Assert.True(n > 40, $"only {n} points");
        for (var i = 0; i < n; i++)
        {
            Assert.InRange(x[2 * i], 0, 1);
            for (var j = i + 1; j < n; j++)
            {
                var d2 = Math.Pow(x[2 * i] - x[2 * j], 2) + Math.Pow(x[2 * i + 1] - x[2 * j + 1], 2);
                Assert.True(d2 >= 0.01 - 1e-12, $"points {i}, {j} at distance {Math.Sqrt(d2)}");
            }
        }
        Assert.Equal(n, e.NumGenerated);
        Assert.Equal(0, e.Random(5).Shape[0]); // the space is full
        Assert.Equal(5, e.Reset().Random(5).Shape[0]);
    }
}
