using System.Runtime.InteropServices;
using Gravicode.Science.GraviNum.Compute;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The LAPACKE routines SciNet needs beyond what GraviNum's <see cref="NativeLapack"/> binds:
/// <c>dgees</c> (real Schur), <c>dgeev</c> (eigenvectors), <c>dgehrd</c>/<c>dorghr</c> (Hessenberg).
/// </summary>
/// <remarks>
/// <para>
/// It binds against the very library GraviNum found — same path, same integer width — so the
/// two can never disagree about which LAPACK is in use, and it is enabled exactly when
/// <see cref="NativeLapack.ShouldUse"/> says so (which honours <see cref="NativeLapack.Enabled"/>
/// and the 64-row floor). The managed kernels remain the default and the reference.
/// </para>
/// <para>
/// Conventions, all following GraviNum's: LAPACKE row-major layout (101), so nothing is
/// transposed here; the layout argument stays a 32-bit <c>int</c> in the ILP64 signatures while
/// every <c>lapack_int</c> is widened; symbols are looked up bare and with numpy/scipy's
/// <c>scipy_</c> prefix, with the <c>64_</c> suffix for ILP64 builds.
/// </para>
/// </remarks>
internal static unsafe class LinalgNativeLapack
{
    private const int RowMajor = 101;

    private static readonly bool Bound;
    private static readonly bool Wide;
    private static readonly nint GeesP, GeevP, GehrdP, OrghrP;

    static LinalgNativeLapack()
    {
        try
        {
            if (!NativeLapack.IsAvailable || NativeLapack.LibraryPath is null) return;
            if (!NativeLibrary.TryLoad(NativeLapack.LibraryPath, out var handle)) return;
            Wide = NativeLapack.UsesWideIntegers;
            var suffix = Wide ? "64_" : "";
            if (Find(handle, "LAPACKE_dgees" + suffix, out GeesP)
                & Find(handle, "LAPACKE_dgeev" + suffix, out GeevP)
                & Find(handle, "LAPACKE_dgehrd" + suffix, out GehrdP)
                & Find(handle, "LAPACKE_dorghr" + suffix, out OrghrP))
                Bound = true;
        }
        catch
        {
            Bound = false; // a missing or odd LAPACK is the common case, not an error
        }
    }

    private static bool Find(nint handle, string name, out nint address)
    {
        foreach (var candidate in new[] { name, "scipy_" + name })
            if (NativeLibrary.TryGetExport(handle, candidate, out address)) return true;
        address = 0;
        return false;
    }

    /// <summary>True when GraviNum found a LAPACK and it exports all four routines.</summary>
    public static bool IsAvailable => Bound;

    /// <summary>Whether an order-n problem should go native (GraviNum's policy, plus these bindings).</summary>
    public static bool ShouldUse(int n) => Bound && NativeLapack.ShouldUse(n);

    /// <summary>
    /// <c>dgees</c>: real Schur form in place in <paramref name="a"/>, Schur vectors into
    /// <paramref name="vs"/> when not null, eigenvalues into <paramref name="wr"/>/<paramref name="wi"/>.
    /// Returns LAPACK's info (0 on success, positive when the QR iteration failed).
    /// </summary>
    public static int Gees(double[] a, int n, double[]? vs, double[] wr, double[] wi)
    {
        var jobvs = vs is null ? (byte)'N' : (byte)'V';
        var scratch = vs ?? new double[1];
        fixed (double* pa = a, pv = scratch, pr = wr, pi = wi)
        {
            if (Wide)
            {
                long sdim = 0;
                var f = (delegate* unmanaged[Cdecl]<int, byte, byte, nint, long, double*, long, long*, double*, double*, double*, long, long>)GeesP;
                return (int)f(RowMajor, jobvs, (byte)'N', 0, n, pa, n, &sdim, pr, pi, pv, Math.Max(1, n));
            }
            else
            {
                int sdim = 0;
                var f = (delegate* unmanaged[Cdecl]<int, byte, byte, nint, int, double*, int, int*, double*, double*, double*, int, int>)GeesP;
                return f(RowMajor, jobvs, (byte)'N', 0, n, pa, n, &sdim, pr, pi, pv, Math.Max(1, n));
            }
        }
    }

    /// <summary>
    /// <c>dgeev</c> with right eigenvectors only: <paramref name="vr"/> receives them in LAPACK's
    /// packed real form (a complex pair as the real and imaginary parts in adjacent columns).
    /// <paramref name="a"/> is destroyed.
    /// </summary>
    public static int Geev(double[] a, int n, double[] wr, double[] wi, double[] vr)
    {
        var vl = new double[1];
        fixed (double* pa = a, pr = wr, pi = wi, pvr = vr, pvl = vl)
        {
            if (Wide)
            {
                var f = (delegate* unmanaged[Cdecl]<int, byte, byte, long, double*, long, double*, double*, double*, long, double*, long, long>)GeevP;
                return (int)f(RowMajor, (byte)'N', (byte)'V', n, pa, n, pr, pi, pvl, Math.Max(1, n), pvr, Math.Max(1, n));
            }
            else
            {
                var f = (delegate* unmanaged[Cdecl]<int, byte, byte, int, double*, int, double*, double*, double*, int, double*, int, int>)GeevP;
                return f(RowMajor, (byte)'N', (byte)'V', n, pa, n, pr, pi, pvl, Math.Max(1, n), pvr, Math.Max(1, n));
            }
        }
    }

    /// <summary>
    /// <c>dgehrd</c> (+ <c>dorghr</c> into <paramref name="q"/> when not null): Hessenberg form in
    /// place, with everything below the first subdiagonal cleared — LAPACK leaves the reflectors there.
    /// </summary>
    public static int Hessenberg(double[] h, int n, double[]? q)
    {
        var tau = new double[Math.Max(1, n - 1)];
        int info;
        fixed (double* ph = h, pt = tau)
        {
            info = Wide
                ? (int)((delegate* unmanaged[Cdecl]<int, long, long, long, double*, long, double*, long>)GehrdP)(RowMajor, n, 1, n, ph, n, pt)
                : ((delegate* unmanaged[Cdecl]<int, int, int, int, double*, int, double*, int>)GehrdP)(RowMajor, n, 1, n, ph, n, pt);
        }
        if (info != 0) return info;
        if (q is not null)
        {
            Array.Copy(h, q, h.Length);
            fixed (double* pq = q, pt = tau)
            {
                info = Wide
                    ? (int)((delegate* unmanaged[Cdecl]<int, long, long, long, double*, long, double*, long>)OrghrP)(RowMajor, n, 1, n, pq, n, pt)
                    : ((delegate* unmanaged[Cdecl]<int, int, int, int, double*, int, double*, int>)OrghrP)(RowMajor, n, 1, n, pq, n, pt);
            }
            if (info != 0) return info;
        }
        for (var i = 2; i < n; i++)
            for (var j = 0; j < i - 1; j++) h[i * n + j] = 0.0;
        return 0;
    }
}
