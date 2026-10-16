using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.NdImageAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Binary/grey morphology and distance transforms against scipy 1.17.1 (values in <see cref="NdImageExpectedMorphology"/>).</summary>
public class NdImageMorphologyTests
{
    private static double[] E(string key) => NdImageExpectedMorphology.Values[key];

    /// <summary>9×11 binary image: ((37i + 17j) mod 23) > 9.</summary>
    internal static NdArray B2()
    {
        var d = new double[99];
        for (var i = 0; i < 9; i++)
            for (var j = 0; j < 11; j++)
                d[i * 11 + j] = (i * 37 + j * 17) % 23 > 9 ? 1 : 0;
        return new NdArray(d, 9, 11);
    }

    /// <summary>4×5×6 binary volume: ((31i + 13j + 7k) mod 19) > 8.</summary>
    internal static NdArray B3()
    {
        var d = new double[120];
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 5; j++)
                for (var k = 0; k < 6; k++)
                    d[(i * 5 + j) * 6 + k] = (i * 31 + j * 13 + k * 7) % 19 > 8 ? 1 : 0;
        return new NdArray(d, 4, 5, 6);
    }

    private static NdArray Mask()
    {
        var d = new double[99];
        for (var i = 0; i < 9; i++)
            for (var j = 0; j < 11; j++)
                d[i * 11 + j] = (i + 2 * j) % 5 != 0 ? 1 : 0;
        return new NdArray(d, 9, 11);
    }

    private static NdArray S8() => NdImage.GenerateBinaryStructure(2, 2);
    private static NdArray S23() => Arr(new double[,] { { 1, 1, 0 }, { 0, 1, 1 } });
    private static NdArray NonFlat() => Arr(new double[,] { { 0.5, 1.0, 0.0 }, { 2.0, 0.25, 1.5 }, { 0.0, 1.0, 0.75 } });

    [Fact]
    public void BinaryErosionAndDilation_MatchScipy()
    {
        var b = B2();
        Close(E("ero"), NdImage.BinaryErosion(b), 0, "erosion");
        Close(E("ero_s8_it2"), NdImage.BinaryErosion(b, S8(), 2), 0, "s8 x2");
        Close(E("ero_border"), NdImage.BinaryErosion(b, borderValue: true), 0, "border 1");
        Close(E("ero_s23_origin"), NdImage.BinaryErosion(b, S23(), origin: [0, -1], borderValue: true), 0, "2x3 origin");
        Close(E("ero_stable"), NdImage.BinaryErosion(b, iterations: 0, borderValue: true), 0, "until stable");
        Close(E("ero_mask_it3"), NdImage.BinaryErosion(b, S8(), 3, Mask()), 0, "mask x3");
        Close(E("dil"), NdImage.BinaryDilation(b), 0, "dilation");
        Close(E("dil_s8_it2"), NdImage.BinaryDilation(b, S8(), 2), 0, "dil s8 x2");
        Close(E("dil_s23_origin"), NdImage.BinaryDilation(b, S23(), origin: [0, -1]), 0, "dil 2x3 origin");
        Close(E("dil_mask_stable"), NdImage.BinaryDilation(b, iterations: 0, mask: Mask()), 0, "dil mask stable");
        Close(E("dil_border"), NdImage.BinaryDilation(b, borderValue: true), 0, "dil border");
        Close(E("ero3d"), NdImage.BinaryErosion(B3(), NdImage.GenerateBinaryStructure(3, 3), borderValue: true), 0, "3-D erosion");
        Close(E("dil3d"), NdImage.BinaryDilation(B3(), iterations: 2), 0, "3-D dilation");
    }

    [Fact]
    public void CompoundBinaryOperations_MatchScipy()
    {
        var b = B2();
        Close(E("open"), NdImage.BinaryOpening(b), 0, "opening");
        Close(E("open_s8"), NdImage.BinaryOpening(b, S8(), 2), 0, "opening s8 x2");
        Close(E("close"), NdImage.BinaryClosing(b), 0, "closing");
        Close(E("close_s23"), NdImage.BinaryClosing(b, S23()), 0, "closing 2x3");
        var seed = new double[99];
        seed[4 * 11 + 5] = 1;
        Close(E("prop"), NdImage.BinaryPropagation(new NdArray(seed, 9, 11), mask: b), 0, "propagation");
        Close(E("prop_s8"), NdImage.BinaryPropagation(new NdArray(seed, 9, 11), S8(), b), 0, "propagation s8");
        Close(E("fill"), NdImage.BinaryFillHoles(b), 0, "fill holes");
        Close(E("fill_s8"), NdImage.BinaryFillHoles(b, S8()), 0, "fill holes s8");
        Close(E("fill3d"), NdImage.BinaryFillHoles(B3()), 0, "fill holes 3-D");
        Close(E("hom"), NdImage.BinaryHitOrMiss(b, Arr(new double[,] { { 0, 1, 0 }, { 1, 1, 1 }, { 0, 0, 0 } })), 0, "hit or miss");
        Close(E("hom2"), NdImage.BinaryHitOrMiss(b, Arr(new double[,] { { 0, 1, 0 }, { 1, 1, 0 }, { 0, 0, 0 } }),
            Arr(new double[,] { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } }), origin1: [0, 1]), 0, "hit or miss 2");
    }

    [Fact]
    public void Structures_MatchScipy()
    {
        Close(E("iter"), NdImage.IterateStructure(NdImage.GenerateBinaryStructure(2, 1), 3), 0, "iterate");
        Close(E("gbs32"), NdImage.GenerateBinaryStructure(3, 2), 0, "generate 3,2");
    }

    [Fact]
    public void PropagationShortcut_EqualsBruteForce()
    {
        var img = NdImageFilterTests.RandomImage(40, 50, 4);
        var bin = new NdArray(img.ToArray().Select(v => v > 12 ? 1.0 : 0.0).ToArray(), 40, 50);
        var mask = new NdArray(img.ToArray().Select((v, i) => i % 7 == 0 ? 0.0 : 1.0).ToArray(), 40, 50);
        foreach (var it in new[] { 2, 3, 5, 0 })
            foreach (var bv in new[] { false, true })
                foreach (var s in new[] { NdImage.GenerateBinaryStructure(2, 1), S8(), S23() })
                {
                    var fast = NdImage.BinaryErosion(bin, s, it, mask, bv);
                    var brute = NdImage.BinaryErosion(bin, s, it, mask, bv, bruteForce: true);
                    Close(brute.ToArray(), fast, 0, $"erosion it={it} border={bv}");
                    var fastD = NdImage.BinaryDilation(bin, s, it, mask, bv);
                    var bruteD = NdImage.BinaryDilation(bin, s, it, mask, bv, bruteForce: true);
                    Close(bruteD.ToArray(), fastD, 0, $"dilation it={it} border={bv}");
                }
    }

    [Fact]
    public void MorphologyIdentities_Hold()
    {
        var b = B2();
        var s = S8();
        // Opening and closing are idempotent.
        var open = NdImage.BinaryOpening(b, s);
        Close(open.ToArray(), NdImage.BinaryOpening(open, s), 0, "opening idempotent");
        var close = NdImage.BinaryClosing(b, s, borderValue: false);
        var closeClose = NdImage.BinaryClosing(close, s);
        // closing can differ near the border because the default border is 0 for the erosion step; check the interior.
        var cc = closeClose.ToArray();
        var c1 = close.ToArray();
        for (var i = 1; i < 8; i++)
            for (var j = 1; j < 10; j++)
                Assert.Equal(c1[i * 11 + j], cc[i * 11 + j]);
        // Duality: dilation of the complement is the complement of erosion (border values swapped).
        var comp = new NdArray(b.ToArray().Select(v => 1 - v).ToArray(), 9, 11);
        var ero = NdImage.BinaryErosion(b, S23(), borderValue: true).ToArray();
        // scipy's dilation reflects its structure (and shifts even axes by one), so passing the reversed S with origin
        // -1 on the even axis makes the dilation use exactly S: erosion(X, S) = NOT dilation(NOT X, reflect(S)).
        var flat = new double[] { 1, 1, 0, 0, 1, 1 }.Reverse().ToArray();
        var dil = NdImage.BinaryDilation(comp, new NdArray(flat, 2, 3), borderValue: false, origin: [-1, 0]).ToArray();
        for (var i = 0; i < ero.Length; i++) Assert.Equal(ero[i], 1 - dil[i]);
        // Grey: erosion ≤ input ≤ dilation, and opening ≤ input ≤ closing.
        var a = A2().ToArray();
        var ge = NdImage.GreyErosion(A2(), [3, 3]).ToArray();
        var gd = NdImage.GreyDilation(A2(), [3, 3]).ToArray();
        var go = NdImage.GreyOpening(A2(), [3, 3]).ToArray();
        var gc = NdImage.GreyClosing(A2(), [3, 3]).ToArray();
        for (var i = 0; i < a.Length; i++)
        {
            Assert.True(ge[i] <= a[i] && a[i] <= gd[i]);
            Assert.True(go[i] <= a[i] && a[i] <= gc[i]);
        }
        Close(go, NdImage.GreyOpening(new NdArray(go, 7, 9), [3, 3]), 0, "grey opening idempotent");
    }

    public static TheoryData<NdImageMode> Modes() => NdImageAssert.Modes();

    [Theory]
    [MemberData(nameof(Modes))]
    public void GreyMorphology_MatchesScipy(NdImageMode m)
    {
        var n = Name(m);
        var fpDil = Arr(new double[,] { { 0, 1, 1 }, { 1, 1, 1 }, { 1, 1, 0 } });
        Close(E($"gero_{n}"), NdImage.GreyErosion(A2(), [3, 2], mode: m, cval: 1.5), 0, "erosion");
        Close(E($"gdil_{n}"), NdImage.GreyDilation(A2(), [3, 2], mode: m, cval: 1.5), 0, "dilation");
        Close(E($"gdilfp_{n}"), NdImage.GreyDilation(A2(), footprint: S23(), mode: m, cval: 1.5), 0, "dilation footprint");
        Close(E($"gerost_{n}"), NdImage.GreyErosion(A2(), structure: NonFlat(), mode: m, cval: 1.5), 1e-15, "erosion structure");
        Close(E($"gdilst_{n}"), NdImage.GreyDilation(A2(), footprint: fpDil, structure: NonFlat(), mode: m, cval: 1.5), 1e-15, "dilation structure");
        Close(E($"gopen_{n}"), NdImage.GreyOpening(A2(), [2, 3], mode: m, cval: 1.5), 0, "opening");
        Close(E($"gclose_{n}"), NdImage.GreyClosing(A2(), footprint: NdImage.GenerateBinaryStructure(2, 1), mode: m, cval: 1.5), 0, "closing");
        Close(E($"mgrad_{n}"), NdImage.MorphologicalGradient(A2(), [3], mode: m, cval: 1.5), 1e-15, "gradient");
        Close(E($"mlap_{n}"), NdImage.MorphologicalLaplace(A2(), structure: NonFlat(), mode: m, cval: 1.5), 1e-14, "laplace");
        Close(E($"wth_{n}"), NdImage.WhiteTophat(A2(), [3, 3], mode: m, cval: 1.5), 1e-15, "white tophat");
        Close(E($"bth_{n}"), NdImage.BlackTophat(A2(), footprint: S23(), mode: m, cval: 1.5), 1e-15, "black tophat");
        Close(E($"gdil3d_{n}"), NdImage.GreyDilation(A3(), [2, 3, 2], mode: m, cval: 1.5), 0, "3-D dilation");
    }

    [Fact]
    public void DistanceTransforms_MatchScipyIncludingIndices()
    {
        var r = NdImage.DistanceTransformEdt(B2(), null, true, true);
        Close(E("edt"), r.Distances!, 1e-14, "edt");
        Close(E("edt_idx"), r.Indices!, 0, "edt indices");
        r = NdImage.DistanceTransformEdt(B2(), [1.5, 0.7], true, true);
        Close(E("edt_s"), r.Distances!, 1e-14, "edt sampled");
        Close(E("edt_s_idx"), r.Indices!, 0, "edt sampled indices");
        r = NdImage.DistanceTransformEdt(B3(), [2.0, 1.0, 0.5], true, true);
        Close(E("edt3"), r.Distances!, 1e-14, "edt 3-D");
        Close(E("edt3_idx"), r.Indices!, 0, "edt 3-D indices");
        Close(E("edt_none"), NdImage.DistanceTransformEdt(NdArray.Ones(2, 3)), 1e-14, "no background");
        foreach (var (met, name) in new[] { (NdImageDistanceMetric.Taxicab, "taxicab"), (NdImageDistanceMetric.Chessboard, "chessboard") })
        {
            var c = NdImage.DistanceTransformCdt(B2(), met, true, true);
            Close(E($"cdt_{name}"), c.Distances!, 0, $"cdt {name}");
            Close(E($"cdt_{name}_idx"), c.Indices!, 0, $"cdt {name} indices");
            var bf = NdImage.DistanceTransformBf(B2(), met, null, true, true);
            Close(E($"bf_{name}"), bf.Distances!, 0, $"bf {name}");
            Close(E($"bf_{name}_idx"), bf.Indices!, 0, $"bf {name} indices");
        }
        var be = NdImage.DistanceTransformBf(B2(), NdImageDistanceMetric.Euclidean, [1.5, 0.7], true, true);
        Close(E("bf_euc"), be.Distances!, 1e-14, "bf euclidean");
        Close(E("bf_euc_idx"), be.Indices!, 0, "bf euclidean indices");
        var c3 = NdImage.DistanceTransformCdt(B3(), NdImageDistanceMetric.Taxicab, true, true);
        Close(E("cdt3"), c3.Distances!, 0, "cdt 3-D");
        Close(E("cdt3_idx"), c3.Indices!, 0, "cdt 3-D indices");
    }

    [Fact]
    public void Edt_EqualsBruteForce_WithAnisotropicSampling()
    {
        var img = NdImageFilterTests.RandomImage(60, 45, 8);
        var bin = new NdArray(img.ToArray().Select(v => v > 3 ? 1.0 : 0.0).ToArray(), 60, 45);
        double[] samp = [0.6, 1.7];
        var edt = NdImage.DistanceTransformEdt(bin, samp).ToArray();
        var bf = NdImage.DistanceTransformBf(bin, NdImageDistanceMetric.Euclidean, samp).ToArray();
        Close(bf, edt, 1e-12, "edt vs bf");
        // and against a naive all-pairs search written here
        var b = bin.ToArray();
        for (var x = 0; x < b.Length; x += 13)
        {
            var best = double.MaxValue;
            for (var y = 0; y < b.Length; y++)
            {
                if (b[y] != 0) continue;
                var di = (x / 45 - y / 45) * samp[0];
                var dj = (x % 45 - y % 45) * samp[1];
                best = Math.Min(best, di * di + dj * dj);
            }
            Assert.True(Math.Abs((b[x] == 0 ? 0 : Math.Sqrt(best)) - edt[x]) < 1e-12, $"element {x}");
        }
    }
}
