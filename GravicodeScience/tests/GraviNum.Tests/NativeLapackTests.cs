using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the optional native LAPACK bridge.
/// </summary>
/// <remarks>
/// Correctness is judged by each factorisation's defining property — <c>A = QR</c>, <c>A V = V L</c>
/// — rather than by agreement with the managed routine. Two implementations agreeing shows only
/// that they share an assumption; the defining property is what makes either of them right.
/// </remarks>
public class NativeLapackTests
{
    private static void WithNative(bool enabled, Action body)
    {
        var previous = NativeLapack.Enabled;
        NativeLapack.Enabled = enabled;
        try { body(); }
        finally { NativeLapack.Enabled = previous; }
    }

    private static double MaxDiff(NdArray a, NdArray b)
    {
        var worst = 0.0;
        for (var i = 0; i < a.Size; i++) worst = Math.Max(worst, Math.Abs(a.At(i) - b.At(i)));
        return worst;
    }

    [Fact]
    public void WithoutALibraryTheManagedFactorisationsAreUsed()
    {
        if (NativeLapack.IsAvailable) return;

        Assert.False(NativeLapack.ShouldUse(512));
        Assert.Contains("none found", NativeLapack.Describe(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SmallProblemsStayManaged()
    {
        // Marshalling and the threads a tuned LAPACK starts cost more than the arithmetic saved,
        // and the managed routines are already quick at this size.
        Assert.False(NativeLapack.ShouldUse(8));
        Assert.False(NativeLapack.ShouldUse(63));
    }

    [Fact]
    public void DisablingItForcesTheManagedPath()
    {
        WithNative(false, () => Assert.False(NativeLapack.ShouldUse(1024)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SolveRecoversAKnownSolution(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int n = 80;
        var rng = new GraviRandom(3);
        var a = rng.StandardNormal(n, n);
        for (var i = 0; i < n; i++) a[i, i] += n;      // keep it well conditioned

        var x = rng.StandardNormal(n);
        var b = LinAlg.Dot(a, x);

        NdArray solved = null!;
        WithNative(native, () => solved = LinAlg.Solve(a, b));

        Assert.True(MaxDiff(solved, x) < 1e-8, $"native={native}, error {MaxDiff(solved, x):E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QrFactorsReconstructTheMatrixAndQIsOrthonormal(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int m = 100, n = 70;
        var rng = new GraviRandom(7);
        var a = rng.StandardNormal(m, n);

        QrResult qr = null!;
        WithNative(native, () => qr = Decomposition.Qr(a));

        Assert.True(MaxDiff(LinAlg.Dot(qr.Q, qr.R), a) < 1e-9,
            $"native={native}, A = QR error {MaxDiff(LinAlg.Dot(qr.Q, qr.R), a):E2}");

        var qtq = LinAlg.Dot(qr.Q.T, qr.Q);
        Assert.True(MaxDiff(qtq, NdArray.Eye(n)) < 1e-9, $"native={native}, Q is not orthonormal");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SvdReconstructsTheMatrixWithDescendingValues(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int m = 90, n = 70;
        var rng = new GraviRandom(11);
        var a = rng.StandardNormal(m, n);

        SvdResult svd = null!;
        WithNative(native, () => svd = Decomposition.Svd(a));

        Assert.True(MaxDiff(svd.Reconstruct(), a) < 1e-9,
            $"native={native}, reconstruction error {MaxDiff(svd.Reconstruct(), a):E2}");

        for (var i = 1; i < svd.SingularValues.Size; i++)
            Assert.True(svd.SingularValues.At(i - 1) >= svd.SingularValues.At(i) - 1e-12);
    }

    [Fact]
    public void NativeAndManagedSingularValuesAgree()
    {
        if (!NativeLapack.IsAvailable) return;

        var rng = new GraviRandom(13);
        var a = rng.StandardNormal(120, 80);

        NdArray managed = null!;
        NdArray native = null!;
        WithNative(false, () => managed = Decomposition.SingularValues(a));
        WithNative(true, () => native = Decomposition.SingularValues(a));

        Assert.True(UFunc.AllClose(native, managed, 1e-9));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SymmetricEigenSatisfiesTheEigenEquationAndSortsDescending(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int n = 80;
        var rng = new GraviRandom(17);
        var m = rng.StandardNormal(n, n);
        var symmetric = (m + m.T) * 0.5;

        EigenResult eigen = null!;
        WithNative(native, () => eigen = Decomposition.SymmetricEigen(symmetric));

        var av = LinAlg.Dot(symmetric, eigen.Vectors);
        var worst = 0.0;
        for (var j = 0; j < n; j++)
            for (var i = 0; i < n; i++)
                worst = Math.Max(worst, Math.Abs(av[i, j] - eigen.Vectors[i, j] * eigen.Values.At(j)));

        Assert.True(worst < 1e-9, $"native={native}, A V = V L error {worst:E2}");

        for (var i = 1; i < n; i++)
            Assert.True(eigen.Values.At(i - 1) >= eigen.Values.At(i) - 1e-12,
                $"native={native}, eigenvalues are not descending");
    }

    [Fact]
    public void NativeAndManagedEigenvaluesAgree()
    {
        // The two order and sign-fix their results independently, so this checks the wrapper
        // makes them interchangeable, not just that LAPACK works.
        if (!NativeLapack.IsAvailable) return;

        var rng = new GraviRandom(19);
        var m = rng.StandardNormal(100, 100);
        var symmetric = (m + m.T) * 0.5;

        EigenResult managed = null!;
        EigenResult native = null!;
        WithNative(false, () => managed = Decomposition.SymmetricEigen(symmetric));
        WithNative(true, () => native = Decomposition.SymmetricEigen(symmetric));

        Assert.True(UFunc.AllClose(native.Values, managed.Values, 1e-9));
        Assert.True(UFunc.AllClose(UFunc.Abs(native.Vectors), UFunc.Abs(managed.Vectors), 1e-7));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LuFactorsSatisfyPAEqualsLU(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int n = 100;
        var rng = new GraviRandom(23);
        var a = rng.StandardNormal(n, n);

        LuResult lu = null!;
        WithNative(native, () => lu = Decomposition.Lu(a));

        var pa = LinAlg.Dot(lu.PermutationMatrix(), a);
        Assert.True(MaxDiff(LinAlg.Dot(lu.Lower, lu.Upper), pa) < 1e-9,
            $"native={native}, P A = L U error {MaxDiff(LinAlg.Dot(lu.Lower, lu.Upper), pa):E2}");

        // LAPACK reports a sequence of row swaps, not a finished permutation. Reading one as the
        // other yields something that still looks like a valid L and U but reconstructs the wrong
        // matrix, so the result being a genuine bijection is worth asserting on its own.
        var seen = new bool[n];
        foreach (var p in lu.Pivot)
        {
            Assert.InRange(p, 0, n - 1);
            Assert.False(seen[p], $"native={native}, row {p} appears twice in the permutation");
            seen[p] = true;
        }

        Assert.True(lu.Lower.At(0) == 1.0, "L must have a unit diagonal");
    }

    [Fact]
    public void NativeAndManagedLuAgreeOnTheDeterminant()
    {
        // The determinant folds the pivot sign and U's diagonal together, so it catches a sign
        // convention mismatch that P A = L U alone would not.
        if (!NativeLapack.IsAvailable) return;

        var rng = new GraviRandom(29);
        var a = rng.StandardNormal(90, 90);

        double managedSign = 0, nativeSign = 0;
        double managedLog = 0, nativeLog = 0;

        WithNative(false, () => (managedLog, managedSign) = LinAlg.SlogDet(a));
        WithNative(true, () => (nativeLog, nativeSign) = LinAlg.SlogDet(a));

        Assert.Equal(managedSign, nativeSign);
        Assert.True(Math.Abs(managedLog - nativeLog) < 1e-8, $"{managedLog:F8} vs {nativeLog:F8}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CholeskyReconstructsAPositiveDefiniteMatrix(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        const int n = 90;
        var rng = new GraviRandom(31);
        var m = rng.StandardNormal(n, n);
        var spd = LinAlg.Dot(m, m.T) + NdArray.Eye(n) * n;

        NdArray l = null!;
        WithNative(native, () => l = Decomposition.Cholesky(spd));

        Assert.True(MaxDiff(LinAlg.Dot(l, l.T), spd) < 1e-8,
            $"native={native}, A = L L^T error {MaxDiff(LinAlg.Dot(l, l.T), spd):E2}");

        // LAPACK leaves the untouched triangle holding the input, so it has to be cleared. If it
        // were not, this would be the original matrix's upper half rather than zeros.
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
                Assert.Equal(0.0, l[i, j], 12);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CholeskyRejectsANonPositiveDefiniteMatrix(bool native)
    {
        if (native && !NativeLapack.IsAvailable) return;

        // Large enough to cross the native threshold, and indefinite by construction.
        const int n = 80;
        var rng = new GraviRandom(37);
        var m = rng.StandardNormal(n, n);
        var symmetric = (m + m.T) * 0.5;      // symmetric, but not positive definite

        WithNative(native, () =>
            Assert.Throws<InvalidOperationException>(() => Decomposition.Cholesky(symmetric)));
    }

    [Fact]
    public void ADetectedLibraryReportsAConsistentIntegerWidth()
    {
        if (!NativeLapack.IsAvailable) return;

        Assert.Contains(NativeLapack.UsesWideIntegers ? "ILP64" : "LP64", NativeLapack.Describe());
        Assert.NotNull(NativeLapack.LibraryPath);
    }
}
