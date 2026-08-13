using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the optional native BLAS bridge.
/// </summary>
/// <remarks>
/// The library ships no native binary, so on most machines there is nothing to bind to and the
/// managed kernels are used. These tests are written to be meaningful either way: the ones that
/// need a library skip cleanly without it, and the ones that describe the fallback behaviour run
/// everywhere.
/// </remarks>
public class NativeBlasTests
{
    /// <summary>Runs <paramref name="body"/> with the native path forced on, then restores it.</summary>
    private static void WithNative(bool enabled, Action body)
    {
        var previous = NativeBlas.Enabled;
        NativeBlas.Enabled = enabled;
        try { body(); }
        finally { NativeBlas.Enabled = previous; }
    }

    [Fact]
    public void WithoutALibraryTheManagedKernelsAreUsed()
    {
        // The point of the design: absence of a BLAS is the normal case, not a failure.
        if (NativeBlas.IsAvailable) return;

        Assert.False(NativeBlas.ShouldUse(512, 512, 512));
        Assert.Contains("none found", NativeBlas.Describe(), StringComparison.OrdinalIgnoreCase);

        // And the product still works.
        var rng = new GraviRandom(2);
        var a = rng.StandardNormal(40, 30);
        var b = rng.StandardNormal(30, 20);
        Assert.Equal([40, 20], LinAlg.Dot(a, b).Shape.ToArray());
    }

    [Fact]
    public void SmallProductsStayOnTheManagedPath()
    {
        // Below the threshold the call overhead and the threads a tuned BLAS spins up cost more
        // than the arithmetic saved, so the decision must not depend only on availability.
        Assert.False(NativeBlas.ShouldUse(8, 8, 8));
        Assert.False(NativeBlas.ShouldUse(32, 32, 32));
    }

    [Fact]
    public void DisablingItForcesTheManagedPath()
    {
        WithNative(false, () => Assert.False(NativeBlas.ShouldUse(1024, 1024, 1024)));
    }

    [Theory]
    [InlineData(64, 64, 64)]
    [InlineData(128, 96, 77)]
    [InlineData(200, 300, 129)]
    public void NativeAndManagedProductsAgree(int m, int k, int n)
    {
        if (!NativeBlas.IsAvailable) return;   // nothing to compare against on this machine

        var rng = new GraviRandom(11);
        var a = rng.StandardNormal(m, k);
        var b = rng.StandardNormal(k, n);

        NdArray managed = null!;
        NdArray native = null!;

        WithNative(false, () => managed = LinAlg.Dot(a, b));
        WithNative(true, () => native = LinAlg.Dot(a, b));

        // Both sum k products in different orders, so they agree to rounding, not bit for bit.
        Assert.True(UFunc.AllClose(native, managed, 1e-9),
            $"native and managed products differ at {m}x{k}x{n}");
    }

    /// <summary>Runs <paramref name="body"/> with the packed kernel forced on or off.</summary>
    private static void WithPacked(bool enabled, Action body)
    {
        var previous = PackedMatMul.Enabled;
        PackedMatMul.Enabled = enabled;
        try { body(); }
        finally { PackedMatMul.Enabled = previous; }
    }

    [Theory]
    [InlineData(200, 200, 200)]
    [InlineData(256, 300, 129)]
    [InlineData(129, 260, 257)]
    public void PackedAndSimpleKernelsAgree(int m, int k, int n)
    {
        // The packed kernel rewrites both operands before touching them and pads ragged tiles with
        // zeros. Shapes that do not divide evenly by the tile size are where that goes wrong.
        var rng = new GraviRandom(13);
        var a = rng.StandardNormal(m, k);
        var b = rng.StandardNormal(k, n);

        NdArray simple = null!;
        NdArray packed = null!;

        WithNative(false, () =>
        {
            WithPacked(false, () => simple = LinAlg.Dot(a, b));
            WithPacked(true, () => packed = LinAlg.Dot(a, b));
        });

        Assert.True(UFunc.AllClose(packed, simple, 1e-9),
            $"packed and simple kernels differ at {m}x{k}x{n}");
    }

    [Fact]
    public void PackedKernelMatchesANaiveTripleLoop()
    {
        // Against the definition, not against the other fast kernel: two fast kernels agreeing
        // proves only that they share an assumption.
        const int m = 224, k = 200, n = 224;
        var rng = new GraviRandom(19);
        var a = rng.StandardNormal(m, k);
        var b = rng.StandardNormal(k, n);

        NdArray got = null!;
        WithNative(false, () => WithPacked(true, () => got = LinAlg.Dot(a, b)));

        var worst = 0.0;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var acc = 0.0;
                for (var p = 0; p < k; p++) acc += a[i, p] * b[p, j];
                worst = Math.Max(worst, Math.Abs(acc - got[i, j]));
            }

        Assert.True(worst < 1e-9, $"max difference from the naive product was {worst:E2}");
    }

    [Fact]
    public void SmallProductsSkipThePackedKernelToo()
    {
        Assert.False(PackedMatMul.ShouldUse(64, 64, 64));
        Assert.True(PackedMatMul.ShouldUse(512, 512, 512));
    }

    [Fact]
    public void ADetectedLibraryReportsAConsistentIntegerWidth()
    {
        if (!NativeBlas.IsAvailable) return;

        // LP64 and ILP64 builds are not interchangeable: calling one through the other's signature
        // reads the wrong bytes as a dimension. Exactly one width must be in play.
        Assert.Contains(NativeBlas.UsesWideIntegers ? "ILP64" : "LP64", NativeBlas.Describe());
        Assert.NotNull(NativeBlas.LibraryPath);
    }
}
