using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpectralAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SignalResamplingTests
{
    [Fact]
    public void Upfirdn_RatesMatchScipy()
    {
        foreach (var (up, down) in new[] { (1, 1), (3, 1), (1, 3), (3, 2), (2, 5), (4, 4) })
            Close($"upfirdn_{up}_{down}", Signal.Upfirdn(R("h7"), R("x30"), up, down), 1e-13);
    }

    [Fact]
    public void Upfirdn_ExtensionModesMatchScipy()
    {
        foreach (var mode in new[] { UpfirdnPad.Constant, UpfirdnPad.Wrap, UpfirdnPad.Edge, UpfirdnPad.Smooth, UpfirdnPad.Symmetric,
                     UpfirdnPad.Reflect, UpfirdnPad.Antisymmetric, UpfirdnPad.Antireflect, UpfirdnPad.Line })
        {
            var name = mode.ToString().ToLowerInvariant();
            Close($"upfirdn_mode_{name}", Signal.Upfirdn(R("h7"), R("x30"), 2, 3, mode, mode == UpfirdnPad.Constant ? 0.7 : 0), 1e-13);
            // Filter far longer than the signal: multiple reflections.
            Close($"upfirdn_long_{name}", Signal.Upfirdn(R("h20"), R("x6"), 1, 1, mode), 1e-13);
        }
    }

    [Fact]
    public void Upfirdn_LargeParallelPath_MatchesDirectDefinition()
    {
        var rng = new Random(2);
        var x = Enumerable.Range(0, 20_000).Select(_ => rng.NextDouble()).ToArray();
        var h = Enumerable.Range(0, 61).Select(_ => rng.NextDouble()).ToArray();
        var y = Signal.Upfirdn(h, x, 3, 2);
        // y[i] = Σ_k h[k] · xup[2i − k], xup the zero-stuffed signal.
        foreach (var i in new[] { 0, 5, 100, 15_000, y.Length - 1 })
        {
            double s = 0;
            for (var k = 0; k < h.Length; k++)
            {
                var j = 2L * i - k;
                if (j >= 0 && j % 3 == 0 && j / 3 < x.Length) s += h[k] * x[j / 3];
            }
            Assert.True(Math.Abs(s - y[i]) < 1e-12, $"y[{i}]");
        }
        Assert.Equal((int)Math.Ceiling(((x.Length - 1) * 3 + h.Length) / 2.0), y.Length);
    }

    [Fact]
    public void Upfirdn_AlongAxis_MatchesRows()
    {
        var a = Nd("a2");
        var y = Signal.Upfirdn(R("h7"), a, 2, 3, axis: 1);
        var rows = a.Shape[0];
        var cols = y.Shape[1];
        var data = y.AsContiguous().ToArray();
        var src = a.AsContiguous().ToArray();
        for (var r = 0; r < rows; r++)
        {
            var row = Signal.Upfirdn(R("h7"), src.AsSpan(r * 7, 7).ToArray(), 2, 3);
            Close(row, data.AsSpan(r * cols, cols), 1e-14, $"row {r}");
        }
    }

    [Fact]
    public void Resample_MatchesScipy()
    {
        var x = R("x50");
        foreach (var num in new[] { 37, 80, 25, 50, 75 })
            Close($"resample_{num}", Signal.Resample(x, num), 1e-12);
        Close("resample_37_hann", Signal.Resample(x, 37, (WindowSpec)"hann"), 1e-12);
        Close("resample_80_kaiser", Signal.Resample(x, 80, new WindowSpec("kaiser", 4.0)), 1e-12);
        Close("resample51_40", Signal.Resample(R("x51"), 40), 1e-12);
        Close("resample51_77", Signal.Resample(R("x51"), 77), 1e-12);
        var t = Enumerable.Range(0, 50).Select(i => i / 49.0).ToArray();
        Close("resample_t", Signal.Resample(x, 37, t, null).T, 1e-13);
    }

    [Fact]
    public void Resample_ComplexMatchesScipy()
    {
        var xc = C("xc20");
        Close("resample_c_14", Signal.Resample(xc, 14), 1e-12);
        Close("resample_c_31", Signal.Resample(xc, 31), 1e-12);
    }

    [Fact]
    public void ResamplePoly_MatchesScipy()
    {
        var x = R("x100");
        foreach (var (up, down) in new[] { (3, 2), (2, 3), (1, 4), (5, 1), (7, 3) })
            Close($"rpoly_{up}_{down}", Signal.ResamplePoly(x, up, down), 1e-12);
        foreach (var (name, pad) in new[] { ("line", UpfirdnPad.Line), ("mean", UpfirdnPad.Mean), ("median", UpfirdnPad.Median), ("edge", UpfirdnPad.Edge),
                     ("smooth", UpfirdnPad.Smooth), ("maximum", UpfirdnPad.Maximum), ("minimum", UpfirdnPad.Minimum), ("reflect", UpfirdnPad.Reflect),
                     ("antisymmetric", UpfirdnPad.Antisymmetric) })
            Close($"rpoly_pad_{name}", Signal.ResamplePoly(x, 3, 2, padtype: pad), 1e-12);
        Close("rpoly_cval", Signal.ResamplePoly(x, 3, 2, cval: 2.5), 1e-12);
        Close("rpoly_hamming", Signal.ResamplePoly(x, 2, 3, (WindowSpec)"hamming"), 1e-12);
        Close("rpoly_arraywin", Signal.ResamplePoly(x, 2, 3, R("h20")), 1e-12);
    }

    [Fact]
    public void ResamplePoly_NdAlongAxis0_MatchesColumns()
    {
        var a = Nd("a2"); // 6 × 7
        var y = Signal.ResamplePoly(a, 3, 2, axis: 0);
        Assert.Equal(new[] { 9, 7 }, y.Shape.ToArray());
        var src = a.AsContiguous().ToArray();
        var col = Enumerable.Range(0, 6).Select(i => src[i * 7 + 2]).ToArray();
        var expect = Signal.ResamplePoly(col, 3, 2);
        var got = Enumerable.Range(0, 9).Select(i => y.AsContiguous().AsSpan()[i * 7 + 2]).ToArray();
        Close(expect, got, 1e-14, "column 2");
    }

    [Fact]
    public void Decimate_MatchesScipy()
    {
        var x = R("x200");
        Close("dec_iir3", Signal.Decimate(x, 3), 1e-10);
        Close("dec_iir3_nozp", Signal.Decimate(x, 3, zeroPhase: false), 1e-10);
        Close("dec_fir4", Signal.Decimate(x, 4, ftype: DecimateFilter.Fir), 1e-12);
        Close("dec_fir4_nozp", Signal.Decimate(x, 4, ftype: DecimateFilter.Fir, zeroPhase: false), 1e-12);
        Close("dec_iir2_n4", Signal.Decimate(x, 2, n: 4), 1e-10);
        Close("dec_fir3_n10", Signal.Decimate(x, 3, n: 10, ftype: DecimateFilter.Fir), 1e-12);
    }
}
