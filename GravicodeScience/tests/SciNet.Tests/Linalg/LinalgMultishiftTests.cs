using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The LAPACK-route Schur kernel — blocked Hessenberg (dgehrd/dlahr2/dorghr) and small-bulge
/// multishift QR with aggressive early deflation (dhseqr/dlaqr0/dlaqr3/dlaqr5, dlahqr, dtrexc) —
/// pinned by the factorisation itself (A = Z T Zᵀ to ~n·eps, Z orthogonal, T quasi-triangular with
/// standardised blocks), against the double-shift hqr2 kernel it replaced, against closed-form
/// spectra, and against SciPy's eigenvalues (tools/verify/schur_fixtures.py) on hard matrices.
/// </summary>
public class LinalgMultishiftTests
{
    private const double Eps = 2.220446049250313e-16;

    // Frank12: scipy 1.17.1 schur
    private static readonly double[] Frank12Re = [0.03102806513242027, 0.04950741950335119, 0.08122766609270644, 0.1436465179421115, 0.2847497207326018, 0.6435053189990054, 1.5539887091321494, 3.511855948580762, 6.961533085567125, 12.311077400868514, 20.198988645877062, 32.228891501572086];
    private static readonly double[] Frank12Im = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Grcar20: scipy 1.17.1 schur
    private static readonly double[] Grcar20Re = [0.10801684437642312, 0.10801684437642312, 0.21881797343738316, 0.21881797343738316, 0.3999261466160602, 0.3999261466160602, 0.6465390740767214, 0.6465390740767214, 0.9519546531741725, 0.9519546531741725, 1.2998608352340768, 1.2998608352340768, 1.5677604977038788, 1.5677604977038788, 1.5820703766821222, 1.5820703766821222, 1.610100748548235, 1.610100748548235, 1.6149528501509371, 1.6149528501509371];
    private static readonly double[] Grcar20Im = [-2.225250547862933, 2.225250547862933, -2.113193786792485, 2.113193786792485, -1.9353473706609081, 1.9353473706609081, -1.7067066189257805, 1.7067066189257805, -1.4536209175235724, 1.4536209175235724, -1.2261005395039521, 1.2261005395039521, -0.2212804174428313, 0.2212804174428313, -0.6436899439832878, 0.6436899439832878, -1.1215670734772019, 1.1215670734772019, -0.9906460496756546, 0.9906460496756546];
    // Graded16: scipy 1.17.1 schur
    private static readonly double[] Graded16Re = [-4.399443143844303, -0.005279638778833019, -0.005279638778833019, -0.00016730502942653936, -1.1564497162758798e-21, -4.503057279833973e-24, 5.00822270483529e-24, 2.255095028254763e-26, 2.205275191459589e-25, 4.158624801903425e-05, 8.856307264392622e-05, 8.856307264392622e-05, 0.001246769250964933, 0.22052810428108544, 0.3104391923975401, 1.0999600441699935];
    private static readonly double[] Graded16Im = [0.0, -0.007587160903945033, 0.007587160903945033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00018123394628472752, 0.00018123394628472752, 0.0, 0.0, 0.0, 0.0];
    // Hashed40: scipy 1.17.1 schur
    private static readonly double[] Hashed40Re = [-3.4379995213182095, -3.4379995213182095, -2.892014938871802, -2.892014938871802, -2.871298802147132, -2.871298802147132, -2.1727695547405252, -2.1727695547405252, -1.7323557213313905, -1.7323557213313905, -1.487838477078956, -1.2594520730834726, -1.2594520730834726, -1.1894121761317067, -0.3122958572514188, -0.30401982037683606, -0.30401982037683606, -0.12179917470606638, -0.12179917470606638, -0.03352540653838615, -0.03352540653838615, 0.1624950407144053, 0.1624950407144053, 0.5108430009379447, 0.5108430009379447, 0.951824914068508, 0.951824914068508, 1.6273222943015178, 1.6273222943015178, 1.64979670196945, 1.64979670196945, 1.7434491691558414, 1.7434491691558414, 2.885215508117066, 2.885215508117066, 3.0194834498104717, 3.0194834498104717, 3.0205383360491522, 3.205941220465296, 3.772676822024864];
    private static readonly double[] Hashed40Im = [-0.7089376082663946, 0.7089376082663946, -0.3369607049275596, 0.3369607049275596, -2.425429322384368, 2.425429322384368, -1.3438342953348077, 1.3438342953348077, -2.306051682545559, 2.306051682545559, 0.0, -0.46151794722043304, 0.46151794722043304, 0.0, 0.0, -2.8339696444979148, 2.8339696444979148, -3.3851365247092944, 3.3851365247092944, -2.2144455109642767, 2.2144455109642767, -1.2007005975567595, 1.2007005975567595, -2.838379814600482, 2.838379814600482, -1.7344053740448822, 1.7344053740448822, -1.336609763082924, 1.336609763082924, -0.2038840529763193, 0.2038840529763193, -2.4899486232903127, 2.4899486232903127, -1.2015920780209546, 1.2015920780209546, -1.9537344052221337, 1.9537344052221337, 0.0, 0.0, 0.0];
    // Hashed120: scipy 1.17.1 schur
    private static readonly double[] Hashed120Re = [-6.340899100427389, -6.340899100427389, -6.022035710064398, -6.022035710064398, -5.843709238545786, -5.843709238545786, -5.786398037700533, -5.786398037700533, -5.405351250655554, -5.405351250655554, -5.286939147648008, -5.286939147648008, -5.019789636101606, -5.019789636101606, -4.267678248167947, -4.267678248167947, -4.001439286543013, -4.001439286543013, -3.738548218707587, -3.738548218707587, -3.3966605716827245, -3.3966605716827245, -3.1485371025463604, -3.1485371025463604, -2.940876713083819, -2.940876713083819, -2.5606144237028348, -2.5606144237028348, -2.50385005720942, -2.50385005720942, -1.9512104117727027, -1.9512104117727027, -1.929919715048579, -1.8668786784175213, -1.8668786784175213, -1.7233874947925623, -1.7233874947925623, -0.9959402237119795, -0.9959402237119795, -0.9119715866658651, -0.9119715866658651, -0.7931554572278336, -0.7931554572278336, -0.7226453144904205, -0.7226453144904205, -0.46757209580211045, -0.46757209580211045, -0.420944618801526, -0.420944618801526, -0.06692427713027138, -0.06692427713027138, -1.663028793051686e-16, -7.024336988254301e-16, 3.0750268092786567e-16, 1.3674583694950906e-15, 1.3223447311784823e-16, -7.013920891934974e-16, -3.6249106442865103e-16, 1.957298673557162e-16, -1.0309753949591718e-15, 7.779182945231135e-16, -1.0309753949591718e-15, 1.957298673557162e-16, -3.6249106442865103e-16, -7.013920891934974e-16, 1.3223447311784823e-16, 1.3674583694950906e-15, 3.0750268092786567e-16, -7.024336988254301e-16, -1.663028793051686e-16, 0.036188511458431924, 0.036188511458431924, 0.06893628277214604, 0.06893628277214604, 0.7573536934251018, 0.7573536934251018, 0.8219501465863464, 0.8219501465863464, 1.0858807838479525, 1.0858807838479525, 1.3084055088521467, 1.3084055088521467, 1.5322509200207712, 1.5322509200207712, 1.7617163659563335, 1.7617163659563335, 1.9553521010925756, 1.9553521010925756, 2.1069803758973142, 2.1069803758973142, 2.2082082588831655, 2.2082082588831655, 2.7244116164595393, 2.7244116164595393, 3.18557397997142, 3.3469088120382775, 3.3469088120382775, 3.3683939837144057, 3.3683939837144057, 3.891539447556233, 3.891539447556233, 3.9196481654077484, 4.377264451744706, 4.377264451744706, 4.614071604795728, 4.614071604795728, 4.760707880451152, 4.760707880451152, 5.183172104655725, 5.183172104655725, 5.239356818292579, 5.239356818292579, 5.816921864107254, 5.816921864107254, 6.5144433517447435, 6.5144433517447435, 6.523778372422381, 6.523778372422381, 7.552112429659513, 7.552112429659513];
    private static readonly double[] Hashed120Im = [-6.471694656815122, 6.471694656815122, -2.9056242546203173, 2.9056242546203173, -4.641801960804572, 4.641801960804572, -0.3464516356080239, 0.3464516356080239, -1.6571110773254243, 1.6571110773254243, -0.7657511371739733, 0.7657511371739733, -3.18611253010995, 3.18611253010995, -2.0101643358130157, 2.0101643358130157, -0.08907146883544698, 0.08907146883544698, -4.581637382596187, 4.581637382596187, -4.192462022546358, 4.192462022546358, -0.2541123409030347, 0.2541123409030347, -3.0673915840275012, 3.0673915840275012, -5.655181748229685, 5.655181748229685, -2.2504862569974593, 2.2504862569974593, -0.41807202309357455, 0.41807202309357455, 0.0, -5.10894605746494, 5.10894605746494, -2.15033602054258, 2.15033602054258, -4.654085453417793, 4.654085453417793, -4.93900935442313, 4.93900935442313, -0.46495694639291846, 0.46495694639291846, -3.634236153665586, 3.634236153665586, -1.525235377807448, 1.525235377807448, -3.3454806422090635, 3.3454806422090635, -5.140460299338903, 5.140460299338903, -1.352453232663e-15, -8.113193200085429e-16, -7.09020980850726e-16, -2.845740851931683e-16, -2.7089324284896323e-16, -2.5338744966920753e-16, -2.4644793381417173e-16, -2.3444535134235485e-16, -1.7817888796083457e-16, 0.0, 1.7817888796083457e-16, 2.3444535134235485e-16, 2.4644793381417173e-16, 2.5338744966920753e-16, 2.7089324284896323e-16, 2.845740851931683e-16, 7.09020980850726e-16, 8.113193200085429e-16, 1.352453232663e-15, -0.48526990297976275, 0.48526990297976275, -6.183816692403604, 6.183816692403604, -4.127012687998289, 4.127012687998289, -2.7696661284649227, 2.7696661284649227, -6.442877331067168, 6.442877331067168, -1.5874125507321961, 1.5874125507321961, -0.2847782479201593, 0.2847782479201593, -3.44646546884138, 3.44646546884138, -5.7581760192310405, 5.7581760192310405, -3.8744192179258223, 3.8744192179258223, -2.087062477655366, 2.087062477655366, -0.2505302622091154, 0.2505302622091154, 0.0, -2.9693768121119155, 2.9693768121119155, -5.26448998527875, 5.26448998527875, -1.854858789321579, 1.854858789321579, 0.0, -0.3164976513475474, 0.3164976513475474, -4.246652766803356, 4.246652766803356, -3.4089221717094156, 3.4089221717094156, -2.1926313224757292, 2.1926313224757292, -5.133955887189366, 5.133955887189366, -1.779464475539004, 1.779464475539004, -4.602115618368741, 4.602115618368741, -0.8113732623026351, 0.8113732623026351, -0.8227278621136469, 0.8227278621136469];

    // ------------------------------------------------------------------ helpers

    private static double[] Rand(int n, int seed, double scale = 1.0)
    {
        var r = new GraviRandom(seed);
        var a = new double[n * n];
        for (var i = 0; i < a.Length; i++) a[i] = (r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static double[] Mul(double[] a, double[] b, int n) => LinalgGemm.Multiply(a, b, n, n, n);

    private static double[] Transpose(double[] a, int n)
    {
        var t = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) t[j * n + i] = a[i * n + j];
        return t;
    }

    private static double Fro(double[] a) => Math.Sqrt(a.Sum(x => x * x));

    /// <summary>A random orthogonal matrix: the Q of the Hessenberg reduction of a random matrix.</summary>
    private static double[] RandomOrthogonal(int n, int seed)
    {
        var h = Rand(n, seed);
        var q = new double[n * n];
        LinalgSchurKernel.Hessenberg(h, q, n);
        return q;
    }

    private static (double[] T, double[] Z, Complex[] W) Multishift(double[] a, int n)
    {
        var t = (double[])a.Clone();
        var z = new double[n * n];
        var wr = new double[n];
        var wi = new double[n];
        LinalgSchurKernel.RealSchurMultishift(t, z, n, wr, wi);
        return (t, z, wr.Zip(wi, (x, y) => new Complex(x, y)).ToArray());
    }

    private static Complex[] DoubleShift(double[] a, int n)
    {
        var t = (double[])a.Clone();
        var wr = new double[n];
        var wi = new double[n];
        LinalgSchurKernel.RealSchurDoubleShift(t, new double[n * n], n, wr, wi);
        return wr.Zip(wi, (x, y) => new Complex(x, y)).ToArray();
    }

    /// <summary>
    /// The real Schur form contract: ‖A − Z T Zᵀ‖/‖A‖ and ‖ZᵀZ − I‖ of order n·eps, nothing below
    /// the subdiagonal, no two adjacent non-zero subdiagonals, each 2×2 block standardised
    /// (equal diagonal, off-diagonals of opposite sign) and the eigenvalues read off the blocks.
    /// </summary>
    private static void AssertSchurForm(double[] a, double[] t, double[] z, Complex[] w, int n, string what)
    {
        var tol = 10.0 * Math.Max(n, 10) * Eps;
        var rec = Mul(Mul(z, t, n), Transpose(z, n), n);
        var normA = Math.Max(Fro(a), 1.0);
        var res = Fro(rec.Zip(a, (x, y) => x - y).ToArray()) / normA;
        Assert.True(res <= tol, $"{what}: ‖A − Z T Zᵀ‖/‖A‖ = {res:E2} > {tol:E2}");
        var ztz = Mul(Transpose(z, n), z, n);
        for (var i = 0; i < n; i++) ztz[i * n + i] -= 1.0;
        var orth = Fro(ztz);
        Assert.True(orth <= tol, $"{what}: ‖ZᵀZ − I‖ = {orth:E2} > {tol:E2}");

        for (var i = 2; i < n; i++)
            for (var j = 0; j < i - 1; j++) Assert.True(t[i * n + j] == 0.0, $"{what}: T[{i},{j}] = {t[i * n + j]}");
        for (var i = 1; i + 1 < n; i++)
            Assert.False(t[i * n + i - 1] != 0.0 && t[(i + 1) * n + i] != 0.0, $"{what}: adjacent subdiagonals at {i}");
        for (var k = 0; k < n; k++)
        {
            if (k + 1 < n && t[(k + 1) * n + k] != 0.0)
            {
                double a11 = t[k * n + k], b = t[k * n + k + 1], c = t[(k + 1) * n + k], d = t[(k + 1) * n + k + 1];
                Assert.True(a11 == d, $"{what}: block {k} diagonal {a11} vs {d}");
                Assert.True(b * c < 0, $"{what}: block {k} off-diagonals {b}, {c} not of opposite sign");
                var im = Math.Sqrt(Math.Abs(b)) * Math.Sqrt(Math.Abs(c));
                Assert.Equal(new Complex(a11, im), w[k]);
                Assert.Equal(new Complex(d, -im), w[k + 1]);
                k++;
            }
            else Assert.Equal(new Complex(t[k * n + k], 0.0), w[k]);
        }
    }

    /// <summary>Greedy nearest matching of two spectra; returns the largest distance.</summary>
    private static double SpectrumDistance(IEnumerable<Complex> expected, IEnumerable<Complex> actual)
    {
        var pool = actual.ToList();
        var worst = 0.0;
        foreach (var e in expected.OrderByDescending(x => x.Magnitude))
        {
            var best = pool.MinBy(x => (x - e).Magnitude);
            worst = Math.Max(worst, (best - e).Magnitude);
            pool.Remove(best);
        }
        return worst;
    }

    private static Complex[] Fixture(double[] re, double[] im) => re.Zip(im, (x, y) => new Complex(x, y)).ToArray();

    private static double[] Frank(int n)
    {
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) a[i * n + j] = j >= i - 1 ? n - Math.Max(i, j) : 0.0;
        return a;
    }

    private static double[] Grcar(int n, int k = 3)
    {
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                a[i * n + j] = j == i - 1 ? -1.0 : (j >= i && j <= i + k ? 1.0 : 0.0);
        return a;
    }

    private static double[] Graded(int n)
    {
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) a[i * n + j] = ((7 * i + 3 * j) % 11 - 5 + 0.5) * Math.Pow(2.0, -(i + j));
        return a;
    }

    private static double[] Hashed(int n)
    {
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) a[i * n + j] = (i + 1) * (j + 3) * (i + j + 5) % 101 / 50.0 - 1.0;
        return a;
    }

    // ------------------------------------------------------------------ blocked Hessenberg

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(6)]
    [InlineData(40)]
    [InlineData(64)]   // all unblocked (dgehd2) + backward Q
    [InlineData(65)]   // one panel
    [InlineData(97)]   // ragged last panel
    [InlineData(257)]  // several panels, parallel products
    public void BlockedHessenberg_ReconstructsAndMatchesUnblocked(int n)
    {
        var a = Rand(n, 500 + n);
        var h = (double[])a.Clone();
        var q = new double[n * n];
        LinalgHessenbergBlocked.Reduce(h, q, n);
        var tol = 10.0 * Math.Max(n, 10) * Eps;

        var rec = Mul(Mul(q, h, n), Transpose(q, n), n);
        Assert.True(Fro(rec.Zip(a, (x, y) => x - y).ToArray()) / Fro(a) <= tol, "Q H Qᵀ − A");
        var qtq = Mul(Transpose(q, n), q, n);
        for (var i = 0; i < n; i++) qtq[i * n + i] -= 1.0;
        Assert.True(Fro(qtq) <= tol, "QᵀQ − I");
        for (var i = 2; i < n; i++)
            for (var j = 0; j < i - 1; j++) Assert.Equal(0.0, h[i * n + j]);

        // Same reflectors as the unblocked reduction (same sign convention), so the same H and Q.
        var h0 = (double[])a.Clone();
        var q0 = new double[n * n];
        LinalgSchurKernel.Hessenberg(h0, q0, n);
        Assert.True(Fro(h.Zip(h0, (x, y) => x - y).ToArray()) <= 1e-10 * Fro(a), "H vs unblocked");
        Assert.True(Fro(q.Zip(q0, (x, y) => x - y).ToArray()) <= 1e-10 * n, "Q vs unblocked");

        // Without Q the same H.
        var h1 = (double[])a.Clone();
        LinalgHessenbergBlocked.Reduce(h1, null, n);
        Assert.Equal(h, h1);
    }

    [Fact]
    public void Hessenberg_PublicRoute_ReconstructsAtBlockedSize()
    {
        const int n = 150;
        var a = Rand(n, 77);
        var r = Linalg.Hessenberg(new NdArray(a, n, n), calcQ: true);
        var q = r.Q!.ToArray();
        var rec = Mul(Mul(q, r.H.ToArray(), n), Transpose(q, n), n);
        Assert.True(Fro(rec.Zip(a, (x, y) => x - y).ToArray()) / Fro(a) <= 10.0 * n * Eps, "Q H Qᵀ − A");
    }

    // ------------------------------------------------------------------ multishift vs double shift

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(15)]   // dlahqr (dhseqr below 75)
    [InlineData(16)]
    [InlineData(74)]
    [InlineData(75)]   // dlaqr0 from here
    [InlineData(76)]
    [InlineData(100)]
    [InlineData(151)]  // 20 shifts
    [InlineData(300)]  // deflation windows above 75 recurse into dlaqr4
    public void Multishift_RandomMatrix_SchurFormAndSameSpectrumAsDoubleShift(int n)
    {
        var a = Rand(n, 900 + n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, $"random {n}");
        var d = SpectrumDistance(DoubleShift(a, n), w);
        Assert.True(d <= 1e-11 * Math.Max(1, n / 10), $"spectrum vs hqr2: {d:E2}");
    }

    [Fact]
    public void Schur_PublicRoute_LargeMatrixTakesMultishiftPath()
    {
        const int n = 220;
        var a = Rand(n, 4242);
        var s = Linalg.Schur(new NdArray(a, n, n));
        var t = s.T.ToArray();
        var z = s.Z.ToArray();
        // The same T and Z as calling the kernel directly: no native LAPACK is routed at this order.
        var (t2, z2, w) = Multishift(a, n);
        Assert.Equal(t2, t);
        Assert.Equal(z2, z);
        AssertSchurForm(a, t, z, w, n, "Linalg.Schur 220");
    }

    [Fact]
    public void Eigvals_EigenvaluesOnlyPath_MatchesFullSchur()
    {
        const int n = 160;
        var a = Rand(n, 31);
        var t = (double[])a.Clone();
        var wr = new double[n];
        var wi = new double[n];
        LinalgSchurKernel.RealSchurMultishift(t, null, n, wr, wi);
        var (_, _, w) = Multishift(a, n);
        var d = SpectrumDistance(w, wr.Zip(wi, (x, y) => new Complex(x, y)));
        Assert.True(d <= 1e-11, $"wantt = false vs true: {d:E2}");
    }

    // ------------------------------------------------------------------ SciPy references

    [Fact]
    public void Frank12_MatchesScipy()
    {
        const int n = 12;
        var a = Frank(n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "Frank 12");
        // The small eigenvalues of the Frank matrix are ill-conditioned: SciPy's balanced eigvals
        // differs from its (and this) unbalanced Schur form by ~5e-8 in them.
        var dist = SpectrumDistance(Fixture(Frank12Re, Frank12Im), w);
        Assert.True(dist <= 1e-11, $"Frank 12 vs scipy: {dist:E2}");
    }

    [Fact]
    public void Grcar20_MatchesScipy()
    {
        const int n = 20;
        var a = Grcar(n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "Grcar 20");
        Assert.True(SpectrumDistance(Fixture(Grcar20Re, Grcar20Im), w) <= 1e-10, "Grcar 20 vs scipy");
    }

    [Fact]
    public void Graded16_MatchesScipyToAbsoluteAccuracy()
    {
        const int n = 16;
        var a = Graded(n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "graded 16");
        Assert.True(SpectrumDistance(Fixture(Graded16Re, Graded16Im), w) <= 1e-13 * Fro(a), "graded 16 vs scipy");
    }

    [Fact]
    public void Hashed40_MatchesScipy()
    {
        const int n = 40;
        var a = Hashed(n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "hashed 40");
        Assert.True(SpectrumDistance(Fixture(Hashed40Re, Hashed40Im), w) <= 1e-12, "hashed 40 vs scipy");
    }

    [Fact]
    public void Hashed120_MultishiftPath_MatchesScipy()
    {
        const int n = 120;
        var a = Hashed(n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "hashed 120");
        var dist = SpectrumDistance(Fixture(Hashed120Re, Hashed120Im), w);
        Assert.True(dist <= 1e-11, $"hashed 120 vs scipy: {dist:E2}");
        // Eigvals balances and takes the eigenvalues-only path; same spectrum.
        var ev = Linalg.Eigvals(new NdArray(a, n, n));
        Assert.True(SpectrumDistance(Fixture(Hashed120Re, Hashed120Im), ev) <= 1e-11, "Eigvals 120 vs scipy");
    }

    // ------------------------------------------------------------------ hard cases, closed forms

    [Theory]
    [InlineData(100)]
    [InlineData(257)]
    public void Frank_And_Grcar_Large_SchurForm(int n)
    {
        foreach (var (name, a) in new[] { ("Frank", Frank(n)), ("Grcar", Grcar(n)) })
        {
            var (t, z, w) = Multishift(a, n);
            AssertSchurForm(a, t, z, w, n, $"{name} {n}");
            // Grcar's spectrum is extremely sensitive and Frank's small eigenvalues are lost by
            // any backward-stable method, so only the trace is compared.
            var trace = Enumerable.Range(0, n).Sum(i => a[i * n + i]);
            Assert.True(Math.Abs(w.Sum(x => x.Real) - trace) <= 1e-10 * Math.Max(1, Math.Abs(trace)), $"{name} trace");
        }
    }

    [Fact]
    public void Companion_OfXnMinusOne_GivesRootsOfUnity()
    {
        const int n = 120;
        var a = new double[n * n];
        for (var i = 1; i < n; i++) a[i * n + i - 1] = 1.0;
        a[n - 1] = 1.0; // x^n = 1: the last column of the companion matrix
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "companion");
        var exact = Enumerable.Range(0, n).Select(k => Complex.FromPolarCoordinates(1.0, 2 * Math.PI * k / n));
        Assert.True(SpectrumDistance(exact, w) <= 1e-12, "roots of unity");
    }

    [Fact]
    public void NonsymmetricTridiagonalToeplitz_ClosedFormSpectrum()
    {
        // diag 1, super 1, sub 0.95: λ_k = 1 + 2·sqrt(0.95)·cos(kπ/(n+1)).
        const int n = 100;
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
        {
            a[i * n + i] = 1.0;
            if (i + 1 < n) a[i * n + i + 1] = 1.0;
            if (i > 0) a[i * n + i - 1] = 0.95;
        }
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "Toeplitz");
        var exact = Enumerable.Range(1, n).Select(k => new Complex(1 + 2 * Math.Sqrt(0.95) * Math.Cos(k * Math.PI / (n + 1)), 0));
        Assert.True(SpectrumDistance(exact, w) <= 1e-11, "Toeplitz spectrum");
    }

    [Fact]
    public void WilkinsonW101_MatchesSymmetricSolver()
    {
        // W+_n: diagonal |(n-1)/2 - i|, unit off-diagonals; eigenvalues in nearly coincident pairs.
        const int n = 101;
        var a = new double[n * n];
        for (var i = 0; i < n; i++)
        {
            a[i * n + i] = Math.Abs((n - 1) / 2.0 - i);
            if (i + 1 < n) a[i * n + i + 1] = a[(i + 1) * n + i] = 1.0;
        }
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "Wilkinson");
        var sym = Linalg.Eigvalsh(new NdArray(a, n, n)).Select(x => new Complex(x, 0));
        Assert.True(SpectrumDistance(sym, w) <= 1e-12 * n, "W+ vs eigh");
    }

    [Fact]
    public void ClusteredNormalMatrix_EigenvaluesToMachineAccuracy()
    {
        // Q diag(1 + 1e-10 k) Qᵀ: a tight cluster, perfectly conditioned (normal).
        const int n = 120;
        var q = RandomOrthogonal(n, 5);
        var d = new double[n * n];
        for (var i = 0; i < n; i++) d[i * n + i] = 1.0 + 1e-10 * i;
        var a = Mul(Mul(q, d, n), Transpose(q, n), n);
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "cluster");
        var exact = Enumerable.Range(0, n).Select(i => new Complex(1.0 + 1e-10 * i, 0));
        Assert.True(SpectrumDistance(exact, w) <= 1e-13, "clustered eigenvalues");
    }

    [Fact]
    public void DefectiveJordanBlocks_ClusterMeansAreAccurate()
    {
        // Jordan blocks J_8(2) and J_8(-1), and a defective complex pair 0.5 ± 1.5i of
        // multiplicity 4, conjugated by an orthogonal Q. A perturbation of eps splits a size-k
        // block by ~eps^(1/k), but each cluster's mean stays accurate to ~eps·‖A‖ (it is a trace).
        const int n = 96;
        var j = new double[n * n];
        var p = 0;
        foreach (var lambda in new[] { 2.0, -1.0 })
        {
            for (var i = 0; i < 8; i++)
            {
                j[(p + i) * n + p + i] = lambda;
                if (i + 1 < 8) j[(p + i) * n + p + i + 1] = 1.0;
            }
            p += 8;
        }
        for (var b = 0; b < 4; b++)
        {
            var o = p + 2 * b;
            j[o * n + o] = 0.5; j[o * n + o + 1] = 1.5; j[(o + 1) * n + o] = -1.5; j[(o + 1) * n + o + 1] = 0.5;
            if (b < 3) { j[o * n + o + 2] = 1.0; j[(o + 1) * n + o + 3] = 1.0; }
        }
        p += 8;
        for (var i = p; i < n; i++) j[i * n + i] = 5.0 + i; // distinct, well separated
        var q = RandomOrthogonal(n, 6);
        var a = Mul(Mul(q, j, n), Transpose(q, n), n);

        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "Jordan");

        foreach (var (lambda, size) in new[] { (new Complex(2, 0), 8), (new Complex(-1, 0), 8), (new Complex(0.5, 1.5), 4), (new Complex(0.5, -1.5), 4) })
        {
            var cluster = w.OrderBy(x => (x - lambda).Magnitude).Take(size).ToArray();
            Assert.True(cluster.All(x => (x - lambda).Magnitude <= 0.05), $"cluster at {lambda}");
            var mean = cluster.Aggregate(Complex.Zero, (s, x) => s + x) / size;
            Assert.True((mean - lambda).Magnitude <= 1e-10, $"cluster mean at {lambda}: {mean}");
        }
        for (var i = p; i < n; i++)
            Assert.True(w.Min(x => (x - (5.0 + i)).Magnitude) <= 1e-11, $"simple eigenvalue {5.0 + i}");
    }

    [Fact]
    public void AlreadyTriangular_IsReturnedUnchanged()
    {
        const int n = 90;
        var a = Rand(n, 8);
        for (var i = 1; i < n; i++)
            for (var c = 0; c < i; c++) a[i * n + c] = 0.0;
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "triangular");
        Assert.Equal(a, t);
        for (var i = 0; i < n; i++) Assert.Equal(a[i * n + i], w[i].Real);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void ZeroAndIdentity(double diag)
    {
        const int n = 80;
        var a = new double[n * n];
        for (var i = 0; i < n; i++) a[i * n + i] = diag;
        var (t, z, w) = Multishift(a, n);
        AssertSchurForm(a, t, z, w, n, "trivial");
        Assert.All(w, x => Assert.Equal(new Complex(diag, 0), x));
    }

    [Fact]
    public void TinyOrders_MatchClosedForms()
    {
        // n = 1.
        var (t1, _, w1) = Multishift([5.0], 1);
        Assert.Equal(5.0, t1[0]);
        Assert.Equal(new Complex(5, 0), w1[0]);

        // n = 2 rotation: eigenvalues ±i, standardised block.
        double[] rot = [0, -1, 1, 0];
        var (t2, z2, w2) = Multishift(rot, 2);
        AssertSchurForm(rot, t2, z2, w2, 2, "rotation");
        Assert.True(SpectrumDistance([Complex.ImaginaryOne, -Complex.ImaginaryOne], w2) <= 1e-15, "±i");

        // n = 2 real: [[4, 1], [2, 3]] has eigenvalues 5 and 2.
        double[] re2 = [4, 1, 2, 3];
        var (t3, z3, w3) = Multishift(re2, 2);
        AssertSchurForm(re2, t3, z3, w3, 2, "real 2x2");
        Assert.True(SpectrumDistance([5, 2], w3) <= 1e-14, "5, 2");

        // n = 3: a quarter turn about z with a scale of 2 along z: eigenvalues ±i and 2.
        double[] r3 = [0, -1, 0, 1, 0, 0, 0, 0, 2];
        var (t4, z4, w4) = Multishift(r3, 3);
        AssertSchurForm(r3, t4, z4, w4, 3, "3x3");
        Assert.True(SpectrumDistance([Complex.ImaginaryOne, -Complex.ImaginaryOne, 2], w4) <= 1e-14, "±i, 2");
    }

    // ------------------------------------------------------------------ dtrexc

    [Fact]
    public unsafe void Trexc_MovesTheFirstBlockToTheEndAndKeepsTheFactorisation()
    {
        const int n = 14;
        var a = Rand(n, 12);
        var (t, z, w) = Multishift(a, n);
        var first = w[0];
        var firstIsPair = first.Imaginary != 0;
        var work = new double[n];
        int ifst = 1, ilst = n;
        int info;
        fixed (double* pt = t, pz = z, pw = work)
            info = LinalgMultishiftQr.Trexc(true, n, pt, n, pz, n, ref ifst, ref ilst, pw);
        Assert.Equal(0, info);

        var rec = Mul(Mul(z, t, n), Transpose(z, n), n);
        Assert.True(Fro(rec.Zip(a, (x, y) => x - y).ToArray()) <= 1e-13 * Fro(a), "A = Z T Zᵀ after the swaps");
        var last = firstIsPair
            ? new Complex(t[(n - 2) * n + n - 2], Math.Sqrt(Math.Abs(t[(n - 2) * n + n - 1])) * Math.Sqrt(Math.Abs(t[(n - 1) * n + n - 2])))
            : new Complex(t[(n - 1) * n + n - 1], 0);
        Assert.True((last - first).Magnitude <= 1e-12, $"moved block {last} vs {first}");
        for (var i = 2; i < n; i++)
            for (var c = 0; c < i - 1; c++) Assert.Equal(0.0, t[i * n + c]);
    }
}
