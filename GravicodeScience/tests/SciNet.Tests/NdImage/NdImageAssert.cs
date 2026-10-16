using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Shared inputs and comparisons for the ndimage tests.</summary>
internal static class NdImageAssert
{
    public static readonly NdImageMode[] AllModes =
    [
        NdImageMode.Reflect, NdImageMode.Mirror, NdImageMode.Nearest, NdImageMode.Constant,
        NdImageMode.Wrap, NdImageMode.GridMirror, NdImageMode.GridConstant, NdImageMode.GridWrap,
    ];

    public static TheoryData<NdImageMode> Modes()
    {
        var d = new TheoryData<NdImageMode>();
        foreach (var m in AllModes) d.Add(m);
        return d;
    }

    /// <summary>The 7×9 test image the scipy reference values were computed on: ((37i + 17j) mod 23)/4 − 2.</summary>
    public static NdArray A2()
    {
        var d = new double[63];
        for (var i = 0; i < 7; i++)
            for (var j = 0; j < 9; j++)
                d[i * 9 + j] = ((i * 37 + j * 17) % 23) / 4.0 - 2;
        return new NdArray(d, 7, 9);
    }

    /// <summary>The 4×5×6 test volume: ((31i + 13j + 7k) mod 19)/2 − 3.</summary>
    public static NdArray A3()
    {
        var d = new double[120];
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 5; j++)
                for (var k = 0; k < 6; k++)
                    d[(i * 5 + j) * 6 + k] = ((i * 31 + j * 13 + k * 7) % 19) / 2.0 - 3;
        return new NdArray(d, 4, 5, 6);
    }

    public static string Name(NdImageMode m) => m.ToString();

    public static NdArray Arr(double[,] v) => NdArray.FromArray(v);

    public static void Close(double[] expected, NdArray actual, double tol = 1e-12, string what = "")
        => Close(expected, actual.ToArray(), tol, what);

    public static void Close(double[] expected, double[] actual, double tol = 1e-12, string what = "")
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (var i = 0; i < expected.Length; i++)
        {
            var e = expected[i];
            var a = actual[i];
            if (double.IsNaN(e) && double.IsNaN(a)) continue;
            var scale = Math.Max(1.0, Math.Abs(e));
            Assert.True(Math.Abs(e - a) <= tol * scale, $"{what}: element {i} is {a:R}, scipy gives {e:R}");
        }
    }
}
