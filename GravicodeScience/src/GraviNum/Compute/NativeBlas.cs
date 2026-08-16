using System.Runtime.InteropServices;

namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// An optional bridge to a native BLAS, used for the matrix product when one is present.
/// </summary>
/// <remarks>
/// <para>
/// This library ships no native binary. A tuned BLAS is a large, platform-specific artifact, and
/// bundling one per runtime identifier would multiply the package size for users who never touch
/// dense linear algebra. So this looks for a BLAS the machine already has, and silently keeps to
/// the managed kernels when it finds none — the same shape as the GPU backend, and for the same
/// reason: a dependency the user did not ask for should never be the difference between working
/// and not working.
/// </para>
/// <para>
/// Point it at a specific build with the <c>GRAVICODE_BLAS</c> environment variable, either a full
/// path or a library name. Otherwise the usual names are tried in order.
/// </para>
/// <para>
/// <b>Both integer widths are handled.</b> A stock OpenBLAS uses 32-bit indices and exports
/// <c>cblas_dgemm</c>; an ILP64 build uses 64-bit indices and exports <c>cblas_dgemm64_</c>. They
/// are not interchangeable — calling one through the other's signature reads the wrong bytes as a
/// dimension and corrupts memory — so the width is detected from which symbol resolves and the
/// matching signature is used.
/// </para>
/// </remarks>
public static class NativeBlas
{
    private const int RowMajor = 101;
    private const int NoTranspose = 111;

    /// <summary>Library names tried when <c>GRAVICODE_BLAS</c> is not set.</summary>
    private static readonly string[] Candidates =
    [
        "openblas", "libopenblas", "openblas64_", "libopenblas64_",
        "mkl_rt", "mkl_rt.2",
        "blas", "libblas",
        "Accelerate",
    ];

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void DgemmNarrow(int layout, int transA, int transB, int m, int n, int k,
        double alpha, nint a, int lda, nint b, int ldb, double beta, nint c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void DgemmWide(long layout, long transA, long transB, long m, long n, long k,
        double alpha, nint a, long lda, nint b, long ldb, double beta, nint c, long ldc);

    private static readonly DgemmNarrow? Narrow;
    private static readonly DgemmWide? Wide;

    static NativeBlas()
    {
        try { (Narrow, Wide, LibraryPath) = Probe(); }
        catch { /* A missing or unusable BLAS is not an error; it is the common case. */ }
    }

    /// <summary>True when a usable native <c>dgemm</c> was found.</summary>
    public static bool IsAvailable => Narrow is not null || Wide is not null;

    /// <summary>Path or name of the library that was loaded, when one was.</summary>
    public static string? LibraryPath { get; }

    /// <summary>True when the loaded build uses 64-bit indices (ILP64).</summary>
    public static bool UsesWideIntegers => Wide is not null;

    /// <summary>
    /// Whether the native path is preferred over the managed kernels.
    /// </summary>
    /// <remarks>
    /// On by default when a BLAS is present, because unlike the GPU backend it is faster at every
    /// size that matters. Set it to false to compare against the managed kernels, which stay in
    /// place as the reference the tests check against.
    /// </remarks>
    public static bool Enabled { get; set; } = true;

    /// <summary>Whether a product of this shape should go to the native library.</summary>
    /// <remarks>
    /// Small products are left to the managed kernel: the call overhead and the threading a tuned
    /// BLAS spins up cost more than the arithmetic saved.
    /// </remarks>
    public static bool ShouldUse(int m, int n, int k)
        => Enabled && IsAvailable && (long)m * n * k >= 64L * 64 * 64;

    /// <summary>
    /// Row-major <c>C = A B</c>, with <c>A</c> being <c>m x k</c> and <c>B</c> being <c>k x n</c>.
    /// </summary>
    /// <remarks>Buffers must be contiguous and correctly sized; the caller guarantees that.</remarks>
    public static unsafe void Multiply(double[] a, double[] b, double[] c, int m, int n, int k)
    {
        if (!IsAvailable) throw new InvalidOperationException("No native BLAS is loaded.");

        fixed (double* pa = a, pb = b, pc = c)
        {
            if (Wide is not null)
                Wide(RowMajor, NoTranspose, NoTranspose, m, n, k, 1.0,
                    (nint)pa, k, (nint)pb, n, 0.0, (nint)pc, n);
            else
                Narrow!(RowMajor, NoTranspose, NoTranspose, m, n, k, 1.0,
                    (nint)pa, k, (nint)pb, n, 0.0, (nint)pc, n);
        }
    }

    private static (DgemmNarrow?, DgemmWide?, string?) Probe()
    {
        var explicitPath = Environment.GetEnvironmentVariable("GRAVICODE_BLAS");
        var names = string.IsNullOrWhiteSpace(explicitPath)
            ? Candidates
            : [explicitPath, .. Candidates];

        foreach (var name in names)
        {
            if (!NativeLibrary.TryLoad(name, out var handle)) continue;

            // Narrow symbols are tried first: a library exporting both is an LP64 build with
            // compatibility aliases, and 32-bit indices are the safer default.
            //
            // The scipy_ prefix is not exotic. numpy and scipy bundle their own OpenBLAS with
            // every symbol renamed, so that loading both cannot collide — and on a machine with
            // numpy installed that is usually the only BLAS present.
            foreach (var symbol in new[] { "cblas_dgemm", "scipy_cblas_dgemm" })
                if (NativeLibrary.TryGetExport(handle, symbol, out var narrow))
                    return (Marshal.GetDelegateForFunctionPointer<DgemmNarrow>(narrow), null, name);

            foreach (var symbol in new[] { "cblas_dgemm64_", "scipy_cblas_dgemm64_" })
                if (NativeLibrary.TryGetExport(handle, symbol, out var wide))
                    return (null, Marshal.GetDelegateForFunctionPointer<DgemmWide>(wide), name);

            // Loaded, but exports no CBLAS interface — a Fortran-only build. Its column-major
            // convention is workable but a different code path, so it is left unsupported rather
            // than guessed at.
            NativeLibrary.Free(handle);
        }

        return (null, null, null);
    }

    /// <summary>A one-line description of what was found, for diagnostics.</summary>
    public static string Describe() => IsAvailable
        ? $"native BLAS: {LibraryPath} ({(UsesWideIntegers ? "ILP64" : "LP64")}), enabled={Enabled}"
        : "native BLAS: none found, using managed kernels";
}
