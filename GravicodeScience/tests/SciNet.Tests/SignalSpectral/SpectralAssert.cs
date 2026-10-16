using System.Numerics;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance-explicit comparisons against the scipy 1.17.1 references in <see cref="SignalSpectralRef"/>.</summary>
internal static class SpectralAssert
{
    public static double[] R(string key) => SignalSpectralRef.D.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException(key);

    public static int[] I(string key) => R(key).Select(v => (int)v).ToArray();

    public static Complex[] C(string key)
    {
        var v = R(key);
        var c = new Complex[v.Length / 2];
        for (var i = 0; i < c.Length; i++) c[i] = new Complex(v[2 * i], v[2 * i + 1]);
        return c;
    }

    public static int[] Shape(string key) => R(key + "#shape").Select(v => (int)v).ToArray();

    public static NdArray Nd(string key) => new((double[])R(key).Clone(), Shape(key));

    /// <summary>|a − e| ≤ atol + rtol·max|e| (scale-relative: FFT rounding is relative to the largest value).</summary>
    public static void Close(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double rtol = 1e-10, string what = "", double atol = 1e-13)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length} != expected {expected.Length}");
        double scale = 0;
        foreach (var e in expected) if (double.IsFinite(e)) scale = Math.Max(scale, Math.Abs(e));
        for (var i = 0; i < expected.Length; i++)
        {
            var e = expected[i];
            var a = actual[i];
            if (double.IsNaN(e)) { Assert.True(double.IsNaN(a), $"{what}[{i}]: expected NaN, got {a}"); continue; }
            Assert.True(Math.Abs(a - e) <= atol + rtol * scale, $"{what}[{i}]: {a:R} vs expected {e:R} (diff {Math.Abs(a - e):E2}, scale {scale:E2})");
        }
    }

    public static void Close(string key, ReadOnlySpan<double> actual, double rtol = 1e-10) => Close(R(key), actual, rtol, key);

    public static void Close(string key, NdArray actual, double rtol = 1e-10)
    {
        if (SignalSpectralRef.D.ContainsKey(key + "#shape"))
            Assert.True(Shape(key).SequenceEqual(actual.Shape.ToArray()), $"{key}: shape [{string.Join(",", actual.Shape.ToArray())}] != [{string.Join(",", Shape(key))}]");
        Close(R(key), actual.AsContiguous().AsSpan(), rtol, key);
    }

    public static void Close(string key, ReadOnlySpan<Complex> actual, double rtol = 1e-10)
    {
        var e = C(key);
        Assert.True(e.Length == actual.Length, $"{key}: length {actual.Length} != expected {e.Length}");
        double scale = 0;
        foreach (var v in e) scale = Math.Max(scale, v.Magnitude);
        for (var i = 0; i < e.Length; i++)
            Assert.True((actual[i] - e[i]).Magnitude <= 1e-13 + rtol * scale, $"{key}[{i}]: {actual[i]} vs expected {e[i]}");
    }

    public static void Close(string key, ComplexNdArray actual, double rtol = 1e-10)
    {
        if (SignalSpectralRef.D.ContainsKey(key + "#shape"))
            Assert.True(Shape(key).SequenceEqual(actual.Shape), $"{key}: shape [{string.Join(",", actual.Shape)}] != [{string.Join(",", Shape(key))}]");
        Close(key, (ReadOnlySpan<Complex>)actual.Data, rtol);
    }

    public static void Equal(string key, int[] actual)
    {
        var e = I(key);
        Assert.True(e.SequenceEqual(actual), $"{key}: [{string.Join(",", actual)}] != expected [{string.Join(",", e)}]");
    }
}
