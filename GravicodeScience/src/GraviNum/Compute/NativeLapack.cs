using System.Runtime.InteropServices;

namespace Gravicode.Science.GraviNum.Compute;

/// <summary>
/// An optional bridge to a native LAPACK, used for the factorisations when one is present.
/// </summary>
/// <remarks>
/// <para>
/// The companion to <see cref="NativeBlas"/>, and it follows the same rules: nothing is bundled,
/// a library the machine already has is used if there is one, and the managed implementations stay
/// as both the fallback and the reference the tests compare against.
/// </para>
/// <para>
/// This binds the <b>LAPACKE</b> interface rather than the Fortran one. LAPACKE takes a layout
/// argument, so row-major matrices can be passed straight through; the Fortran entry points are
/// column-major only, and every call would need a transpose in and another out â€” two copies that
/// would eat much of what the native routine saves.
/// </para>
/// <para>
/// Both integer widths are handled, for the reason set out in <see cref="NativeBlas"/>: an ILP64
/// build takes 64-bit dimensions and calling it through a 32-bit signature reads the wrong bytes.
/// Note that the layout argument stays a 32-bit <c>int</c> in both, because LAPACKE declares it as
/// a plain <c>int</c> rather than a <c>lapack_int</c> â€” a detail that silently corrupts the stack
/// if it is widened along with everything else.
/// </para>
/// </remarks>
public static class NativeLapack
{
    /// <summary>LAPACKE's row-major layout constant.</summary>
    private const int RowMajor = 101;

    private const byte NoVectors = (byte)'N';
    private const byte AllVectors = (byte)'A';
    private const byte WithVectors = (byte)'V';
    private const byte Upper = (byte)'U';
    private const byte Lower = (byte)'L';

    // ---- narrow (LP64) signatures ----

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GesvNarrow(int layout, int n, int nrhs, nint a, int lda, nint ipiv, nint b, int ldb);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GeqrfNarrow(int layout, int m, int n, nint a, int lda, nint tau);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OrgqrNarrow(int layout, int m, int n, int k, nint a, int lda, nint tau);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GesvdNarrow(int layout, byte jobu, byte jobvt, int m, int n, nint a, int lda,
        nint s, nint u, int ldu, nint vt, int ldvt, nint superb);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SyevNarrow(int layout, byte jobz, byte uplo, int n, nint a, int lda, nint w);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetrfNarrow(int layout, int m, int n, nint a, int lda, nint ipiv);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PotrfNarrow(int layout, byte uplo, int n, nint a, int lda);

    // ---- wide (ILP64) signatures ----

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long GesvWide(int layout, long n, long nrhs, nint a, long lda, nint ipiv, nint b, long ldb);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long GeqrfWide(int layout, long m, long n, nint a, long lda, nint tau);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long OrgqrWide(int layout, long m, long n, long k, nint a, long lda, nint tau);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long GesvdWide(int layout, byte jobu, byte jobvt, long m, long n, nint a, long lda,
        nint s, nint u, long ldu, nint vt, long ldvt, nint superb);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long SyevWide(int layout, byte jobz, byte uplo, long n, nint a, long lda, nint w);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long GetrfWide(int layout, long m, long n, nint a, long lda, nint ipiv);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long PotrfWide(int layout, byte uplo, long n, nint a, long lda);

    private static GesvNarrow? _gesvN;
    private static GeqrfNarrow? _geqrfN;
    private static OrgqrNarrow? _orgqrN;
    private static GesvdNarrow? _gesvdN;
    private static SyevNarrow? _syevN;
    private static GetrfNarrow? _getrfN;
    private static PotrfNarrow? _potrfN;

    private static GesvWide? _gesvW;
    private static GeqrfWide? _geqrfW;
    private static OrgqrWide? _orgqrW;
    private static GesvdWide? _gesvdW;
    private static SyevWide? _syevW;
    private static GetrfWide? _getrfW;
    private static PotrfWide? _potrfW;

    static NativeLapack()
    {
        try { Bind(); }
        catch { /* A missing LAPACK is the common case, not an error. */ }
    }

    /// <summary>True when the four factorisation routines all resolved.</summary>
    public static bool IsAvailable { get; private set; }

    /// <summary>True when the loaded build uses 64-bit indices.</summary>
    public static bool UsesWideIntegers { get; private set; }

    /// <summary>Path or name of the library that was loaded.</summary>
    public static string? LibraryPath { get; private set; }

    /// <summary>Whether the native routines are preferred over the managed ones.</summary>
    public static bool Enabled { get; set; } = true;

    /// <summary>Whether a problem of this size should go native.</summary>
    /// <remarks>
    /// Small factorisations stay managed: the marshalling and the threading a tuned LAPACK starts
    /// cost more than the arithmetic saved, and the managed routines are already fast there.
    /// </remarks>
    public static bool ShouldUse(int n) => Enabled && IsAvailable && n >= 64;

    /// <summary>A one-line description of what was found.</summary>
    public static string Describe() => IsAvailable
        ? $"native LAPACK: {LibraryPath} ({(UsesWideIntegers ? "ILP64" : "LP64")}), enabled={Enabled}"
        : "native LAPACK: none found, using managed factorisations";

    private static void Bind()
    {
        var explicitPath = Environment.GetEnvironmentVariable("GRAVICODE_BLAS");
        string[] names = string.IsNullOrWhiteSpace(explicitPath)
            ? ["openblas", "libopenblas", "openblas64_", "libopenblas64_", "mkl_rt", "mkl_rt.2", "lapack", "liblapack"]
            : [explicitPath, "openblas", "libopenblas", "mkl_rt", "lapack"];

        foreach (var name in names)
        {
            if (!NativeLibrary.TryLoad(name, out var handle)) continue;

            // Narrow first, for the reason given in NativeBlas: a library exporting both is LP64
            // with aliases, and 32-bit indices are the safer default.
            if (TryBindNarrow(handle) || TryBindWide(handle))
            {
                LibraryPath = name;
                IsAvailable = true;
                return;
            }

            NativeLibrary.Free(handle);
        }
    }

    private static bool TryBindNarrow(nint handle)
    {
        if (!TryGet<GesvNarrow>(handle, "LAPACKE_dgesv", out var gesv)
            || !TryGet<GeqrfNarrow>(handle, "LAPACKE_dgeqrf", out var geqrf)
            || !TryGet<OrgqrNarrow>(handle, "LAPACKE_dorgqr", out var orgqr)
            || !TryGet<GesvdNarrow>(handle, "LAPACKE_dgesvd", out var gesvd)
            || !TryGet<SyevNarrow>(handle, "LAPACKE_dsyev", out var syev)
            || !TryGet<GetrfNarrow>(handle, "LAPACKE_dgetrf", out var getrf)
            || !TryGet<PotrfNarrow>(handle, "LAPACKE_dpotrf", out var potrf))
            return false;

        (_gesvN, _geqrfN, _orgqrN, _gesvdN, _syevN) = (gesv, geqrf, orgqr, gesvd, syev);
        (_getrfN, _potrfN) = (getrf, potrf);
        UsesWideIntegers = false;
        return true;
    }

    private static bool TryBindWide(nint handle)
    {
        if (!TryGet<GesvWide>(handle, "LAPACKE_dgesv64_", out var gesv)
            || !TryGet<GeqrfWide>(handle, "LAPACKE_dgeqrf64_", out var geqrf)
            || !TryGet<OrgqrWide>(handle, "LAPACKE_dorgqr64_", out var orgqr)
            || !TryGet<GesvdWide>(handle, "LAPACKE_dgesvd64_", out var gesvd)
            || !TryGet<SyevWide>(handle, "LAPACKE_dsyev64_", out var syev)
            || !TryGet<GetrfWide>(handle, "LAPACKE_dgetrf64_", out var getrf)
            || !TryGet<PotrfWide>(handle, "LAPACKE_dpotrf64_", out var potrf))
            return false;

        (_gesvW, _geqrfW, _orgqrW, _gesvdW, _syevW) = (gesv, geqrf, orgqr, gesvd, syev);
        (_getrfW, _potrfW) = (getrf, potrf);
        UsesWideIntegers = true;
        return true;
    }

    /// <summary>Resolves a symbol, trying the bare name and the scipy-renamed one.</summary>
    /// <remarks>
    /// numpy and scipy bundle their own OpenBLAS with every symbol prefixed, so that loading both
    /// cannot collide â€” and on a machine with numpy installed that is usually the only one there.
    /// </remarks>
    private static bool TryGet<T>(nint handle, string name, out T? result) where T : Delegate
    {
        foreach (var candidate in new[] { name, "scipy_" + name })
            if (NativeLibrary.TryGetExport(handle, candidate, out var address))
            {
                result = Marshal.GetDelegateForFunctionPointer<T>(address);
                return true;
            }

        result = null;
        return false;
    }

    // ================================================================ routines

    /// <summary>
    /// LU factorisation with partial pivoting, in place.
    /// </summary>
    /// <param name="a">
    /// On entry the matrix; on exit L below the diagonal (its unit diagonal implied) and U on and
    /// above it, packed into the one array as LAPACK returns them.
    /// </param>
    /// <param name="n">Matrix order.</param>
    /// <param name="permutation">
    /// The row permutation in this library's convention: entry <c>i</c> is the source row of
    /// output row <c>i</c>.
    /// </param>
    /// <param name="sign">Determinant sign contributed by the row swaps.</param>
    /// <remarks>
    /// The pivot conversion is the reason this routine was not bound alongside the others. LAPACK
    /// reports a <em>sequence of swaps</em> — at step <c>i</c>, row <c>i</c> was exchanged with row
    /// <c>ipiv[i]</c>, one-based — whereas <see cref="LuResult.Pivot"/> is a finished permutation.
    /// Reading one as the other produces a factorisation that reconstructs to the wrong matrix
    /// while still looking like a valid L and U, so the swaps are replayed rather than copied.
    /// </remarks>
    public static unsafe int Lu(double[] a, int n, out int[] permutation, out double sign)
    {
        permutation = new int[n];
        for (var i = 0; i < n; i++) permutation[i] = i;
        sign = 1.0;

        int info;
        var swaps = new int[n];

        fixed (double* pa = a)
        {
            if (_getrfW is not null)
            {
                var wide = new long[n];
                fixed (long* pp = wide)
                    info = (int)_getrfW(RowMajor, n, n, (nint)pa, n, (nint)pp);

                for (var i = 0; i < n; i++) swaps[i] = (int)wide[i];
            }
            else
            {
                fixed (int* pp = swaps)
                    info = _getrfN!(RowMajor, n, n, (nint)pa, n, (nint)pp);
            }
        }

        // A positive info means U has an exact zero on the diagonal — singular, but the
        // factorisation itself is still complete and usable, so it is not treated as failure here.
        if (info < 0) return info;

        for (var i = 0; i < n; i++)
        {
            var target = swaps[i] - 1;      // LAPACK counts from one
            if (target == i || target < 0 || target >= n) continue;

            (permutation[i], permutation[target]) = (permutation[target], permutation[i]);
            sign = -sign;
        }

        return info;
    }

    /// <summary>
    /// Cholesky factorisation, returning the lower-triangular factor in place.
    /// </summary>
    /// <remarks>
    /// LAPACK writes only the triangle it was asked for and leaves the other holding whatever the
    /// input had, so the caller must clear it. That is why this returns nothing useful in the
    /// upper triangle rather than zeros.
    /// </remarks>
    public static unsafe int Cholesky(double[] a, int n)
    {
        fixed (double* pa = a)
        {
            return _potrfW is not null
                ? (int)_potrfW(RowMajor, Lower, n, (nint)pa, n)
                : _potrfN!(RowMajor, Lower, n, (nint)pa, n);
        }
    }

    /// <summary>Solves <c>A X = B</c> in place; <paramref name="a"/> and <paramref name="b"/> are destroyed.</summary>
    /// <returns>LAPACK's info code: 0 on success, positive when the matrix is singular.</returns>
    public static unsafe int Solve(double[] a, double[] b, int n, int nrhs)
    {
        fixed (double* pa = a, pb = b)
        {
            if (_gesvW is not null)
            {
                var pivots = new long[n];
                fixed (long* pp = pivots)
                    return (int)_gesvW(RowMajor, n, nrhs, (nint)pa, n, (nint)pp, (nint)pb, nrhs);
            }

            var narrowPivots = new int[n];
            fixed (int* pp = narrowPivots)
                return _gesvN!(RowMajor, n, nrhs, (nint)pa, n, (nint)pp, (nint)pb, nrhs);
        }
    }

    /// <summary>
    /// QR factorisation of an <c>m x n</c> matrix, returning the reduced factors.
    /// </summary>
    /// <remarks>
    /// LAPACK returns Q implicitly, as the Householder reflectors that generate it, because that
    /// form is what the solve routines actually want. <c>dorgqr</c> expands it, which is a second
    /// call and real work â€” so this is only worth doing when Q is genuinely needed.
    /// </remarks>
    public static unsafe int Qr(double[] a, int m, int n, out double[] q, out double[] r)
    {
        var k = Math.Min(m, n);
        var tau = new double[k];
        q = [];
        r = [];

        int info;
        fixed (double* pa = a, pt = tau)
        {
            info = _geqrfW is not null
                ? (int)_geqrfW(RowMajor, m, n, (nint)pa, n, (nint)pt)
                : _geqrfN!(RowMajor, m, n, (nint)pa, n, (nint)pt);
        }
        if (info != 0) return info;

        // R is the upper triangle of the overwritten matrix, before Q consumes the rest.
        r = new double[(long)k * n];
        for (var i = 0; i < k; i++)
            for (var j = i; j < n; j++)
                r[(long)i * n + j] = a[(long)i * n + j];

        fixed (double* pa = a, pt = tau)
        {
            info = _orgqrW is not null
                ? (int)_orgqrW(RowMajor, m, k, k, (nint)pa, n, (nint)pt)
                : _orgqrN!(RowMajor, m, k, k, (nint)pa, n, (nint)pt);
        }
        if (info != 0) return info;

        // dorgqr writes Q into the first k columns of an n-wide buffer; compact it.
        q = new double[(long)m * k];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < k; j++)
                q[(long)i * k + j] = a[(long)i * n + j];

        return 0;
    }

    /// <summary>Singular value decomposition, returning <c>U</c>, the values, and <c>V^T</c>.</summary>
    public static unsafe int Svd(double[] a, int m, int n, out double[] u, out double[] s, out double[] vt)
    {
        var k = Math.Min(m, n);
        u = new double[(long)m * m];
        s = new double[k];
        vt = new double[(long)n * n];
        var superb = new double[Math.Max(1, k - 1)];

        fixed (double* pa = a, pu = u, ps = s, pv = vt, pb = superb)
        {
            return _gesvdW is not null
                ? (int)_gesvdW(RowMajor, AllVectors, AllVectors, m, n, (nint)pa, n,
                    (nint)ps, (nint)pu, m, (nint)pv, n, (nint)pb)
                : _gesvdN!(RowMajor, AllVectors, AllVectors, m, n, (nint)pa, n,
                    (nint)ps, (nint)pu, m, (nint)pv, n, (nint)pb);
        }
    }

    /// <summary>Singular values only â€” much cheaper, since neither factor is accumulated.</summary>
    public static unsafe int SingularValues(double[] a, int m, int n, out double[] s)
    {
        var k = Math.Min(m, n);
        s = new double[k];
        var superb = new double[Math.Max(1, k - 1)];

        fixed (double* pa = a, ps = s, pb = superb)
        {
            return _gesvdW is not null
                ? (int)_gesvdW(RowMajor, NoVectors, NoVectors, m, n, (nint)pa, n,
                    (nint)ps, 0, m, 0, n, (nint)pb)
                : _gesvdN!(RowMajor, NoVectors, NoVectors, m, n, (nint)pa, n,
                    (nint)ps, 0, m, 0, n, (nint)pb);
        }
    }

    /// <summary>
    /// Eigenvalues and eigenvectors of a symmetric matrix.
    /// </summary>
    /// <remarks>
    /// On return <paramref name="a"/> holds the eigenvectors and <paramref name="w"/> the
    /// eigenvalues, <b>ascending</b> â€” LAPACK's convention, and the opposite of this library's, so
    /// the caller reverses them.
    /// </remarks>
    public static unsafe int SymmetricEigen(double[] a, int n, out double[] w)
    {
        w = new double[n];
        fixed (double* pa = a, pw = w)
        {
            return _syevW is not null
                ? (int)_syevW(RowMajor, WithVectors, Upper, n, (nint)pa, n, (nint)pw)
                : _syevN!(RowMajor, WithVectors, Upper, n, (nint)pa, n, (nint)pw);
        }
    }
}

