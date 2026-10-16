using System.Numerics;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Fixture lookup and tolerance assertions shared by the IO tests.</summary>
internal static class IOAssert
{
    private static readonly Lazy<string> FixtureRoot = new(() =>
    {
        // Walk up from the test binaries; fall back to this source file's location for builds
        // whose output directory lives outside the repository.
        var root = FindSolution(AppContext.BaseDirectory) ?? FindSolution(SourceDirectory())
            ?? throw new InvalidOperationException("Could not find Gravicode.Science.sln above " + AppContext.BaseDirectory);
        return Path.Combine(root, "tests", "SciNet.Tests", "IO", "Fixtures");
    });

    private static string? FindSolution(string? start)
    {
        var dir = start is null ? null : new DirectoryInfo(start);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Gravicode.Science.sln"))) dir = dir.Parent;
        return dir?.FullName;
    }

    private static string? SourceDirectory([System.Runtime.CompilerServices.CallerFilePath] string path = "") => Path.GetDirectoryName(path);

    /// <summary>Path of a fixture file (relative to tests/SciNet.Tests/IO/Fixtures).</summary>
    public static string Fixture(string name) => Path.Combine(FixtureRoot.Value, name);

    /// <summary>A fresh temporary file path; the caller deletes it.</summary>
    public static string TempFile(string extension) => Path.Combine(Path.GetTempPath(), "scinet_io_" + Guid.NewGuid().ToString("N") + extension);

    public static void Close(double expected, double actual, double tol, string what = "")
        => Assert.True(expected.Equals(actual) || Math.Abs(expected - actual) <= tol, $"{what}: expected {expected:R}, got {actual:R} (tol {tol:G3})");

    public static void Close(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tol, string what = "")
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (var i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]) || Math.Abs(expected[i] - actual[i]) <= tol,
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R} (tol {tol:G3})");
    }

    public static void Close(NdArray expected, NdArray actual, double tol, string what = "")
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        Close(expected.ToArray(), actual.ToArray(), tol, what);
    }

    public static void Close(Complex expected, Complex actual, double tol, string what = "")
        => Assert.True(Complex.Abs(expected - actual) <= tol, $"{what}: expected {expected}, got {actual} (tol {tol:G3})");

    public static void Strings(string?[] expected, IEnumerable<string?> actual, string what = "")
    {
        var a = actual.ToArray();
        Assert.True(expected.SequenceEqual(a), $"{what}: [{string.Join(", ", a.Select(x => x ?? "null"))}], expected [{string.Join(", ", expected.Select(x => x ?? "null"))}]");
    }

    public static void Shape(int[] expected, IReadOnlyList<int> actual, string what = "")
        => Assert.True(expected.SequenceEqual(actual), $"{what}: shape ({string.Join(", ", actual)}), expected ({string.Join(", ", expected)})");
}
