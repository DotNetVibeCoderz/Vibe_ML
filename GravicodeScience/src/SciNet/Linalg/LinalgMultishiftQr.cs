using System.Runtime.Intrinsics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The small-bulge multishift QR algorithm with aggressive early deflation (Braman, Byers and
/// Mathias 2002), as LAPACK 3.12 implements it: <c>dhseqr</c> → <c>dlaqr0</c> (driver),
/// <c>dlaqr3</c>/<c>dlaqr2</c> (aggressive early deflation), <c>dlaqr5</c> (a chain of tightly
/// packed bulges chased together, their reflectors accumulated into a small orthogonal U and
/// applied off the diagonal as matrix products), <c>dlahqr</c> (double-shift QR for small
/// blocks), and <c>dtrexc</c>/<c>dlaexc</c>/<c>dlasy2</c> (swapping diagonal blocks of a real
/// Schur form, which deflation needs to move undeflatable eigenvalues out of the way).
/// </summary>
/// <remarks>
/// <para>
/// Ported line by line from the reference Fortran, with LAPACK's tuning (<c>iparmq</c>: 75 as the
/// switch from <c>dlahqr</c>, the shift count and window size by order, accumulation from 14
/// shifts, nibble 14). The ports keep LAPACK's 1-based indices: each routine offsets its
/// pointers so that <c>h[i * ldh + j]</c> is <c>H(i, j)</c>. Storage is row-major, so a row
/// operation in the Fortran (stride <c>LDH</c>) is contiguous here and a column operation has
/// stride <c>ldh</c>; the arithmetic is unchanged.
/// </para>
/// <para>
/// Two departures, neither of which changes the arithmetic: the workspaces are separate arrays
/// rather than the unused lower triangle of H, and the off-diagonal products are one call over
/// the whole range instead of LAPACK's <c>NV</c>/<c>NH</c> chunks (which exist only to bound
/// that workspace).
/// </para>
/// </remarks>
internal static unsafe class LinalgMultishiftQr
{
    /// <summary>LAPACK <c>dlamch('P')</c>: relative machine precision times the base.</summary>
    private const double Ulp = 2.220446049250313e-16;

    /// <summary>LAPACK <c>dlamch('S')</c>: the safe minimum.</summary>
    private const double SafeMin = 2.2250738585072014e-308;

    /// <summary>
    /// <c>iparmq</c>'s NMIN: <c>dhseqr</c> uses <c>dlahqr</c> below this order, and the
    /// deflation window recurses into the multishift code above it.
    /// </summary>
    internal const int NMin = 75;

    private const int NTiny = 15;
    private const int NL = 49;
    private const int KExNw = 5;
    private const int KExSh = 6;
    private const int Nibble = 14;
    private const double Wilk1 = 0.75;
    private const double Wilk2 = -0.4375;

    // ================================================================ dhseqr

    /// <summary>
    /// LAPACK <c>dhseqr</c> on a whole upper Hessenberg matrix (ilo = 1, ihi = n): eigenvalues into
    /// <paramref name="wr"/>/<paramref name="wi"/>, the Schur form in place when
    /// <paramref name="wantt"/>, and <c>Z ← Z Q</c> when <paramref name="wantz"/> (pass Z holding
    /// the Hessenberg reduction's Q to get the Schur vectors of the original matrix).
    /// Everything below the subdiagonal is cleared when <paramref name="wantt"/>.
    /// </summary>
    /// <returns>0, or LAPACK's positive info when the iteration failed to converge.</returns>
    public static int Hseqr(bool wantt, bool wantz, int n, double* H, int ldh, double* wr, double* wi, double* Z, int ldz)
    {
        if (n == 0) return 0;
        if (n == 1)
        {
            wr[0] = H[0];
            wi[0] = 0.0;
            return 0;
        }
        int info;
        if (n >= NMin)
            info = Laqr0(wantt, wantz, n, 1, n, H, ldh, wr, wi, 1, n, Z, ldz, recursive: true);
        else
        {
            info = Lahqr(wantt, wantz, n, 1, n, H, ldh, wr, wi, 1, n, Z, ldz);
            if (info > 0)
            {
                // Rare dlahqr failure: dhseqr retries with the multishift code from kbot = info,
                // padding to NL rows when the matrix is smaller than that.
                var kbot = info;
                if (n >= NL)
                    info = Laqr0(wantt, wantz, n, 1, kbot, H, ldh, wr, wi, 1, n, Z, ldz, recursive: true);
                else
                {
                    var hl = new double[NL * NL];
                    for (var i = 0; i < n; i++)
                        new Span<double>(H + (long)i * ldh, n).CopyTo(hl.AsSpan(i * NL, n));
                    fixed (double* phl = hl)
                        info = Laqr0(wantt, wantz, NL, 1, kbot, phl, NL, wr, wi, 1, n, Z, ldz, recursive: true, lwork: NL);
                    if (wantt || info != 0)
                        for (var i = 0; i < n; i++)
                            hl.AsSpan(i * NL, n).CopyTo(new Span<double>(H + (long)i * ldh, n));
                }
            }
        }
        if ((wantt || info != 0) && n > 2)
            for (var i = 2; i < n; i++)
                new Span<double>(H + (long)i * ldh, i - 1).Clear();
        return info;
    }

    // ================================================================ iparmq

    /// <summary><c>iparmq</c> ISPEC = 15: the number of simultaneous shifts for an active block of order nh.</summary>
    internal static int ShiftCount(int nh)
    {
        var ns = 2;
        if (nh >= 30) ns = 4;
        if (nh >= 60) ns = 10;
        if (nh >= 150) ns = Math.Max(10, nh / (int)Math.Round(Math.Log(nh) / Math.Log(2.0), MidpointRounding.AwayFromZero));
        if (nh >= 590) ns = 64;
        if (nh >= 3000) ns = 128;
        if (nh >= 6000) ns = 256;
        return Math.Max(2, ns - ns % 2);
    }

    /// <summary><c>iparmq</c> ISPEC = 13: the deflation window size.</summary>
    private static int WindowSize(int nh) => nh <= 500 ? ShiftCount(nh) : 3 * ShiftCount(nh) / 2;

    /// <summary>
    /// <c>iparmq</c> ISPEC = 16, deliberately not LAPACK's value: LAPACK accumulates the bulge
    /// reflectors into U only from 14 shifts, and with 10 (orders 60–149) applies each one to the
    /// rest of H and to Z directly. Here that direct update walks three columns of every row of a
    /// row-major buffer per reflector; accumulating measured 1.2–1.3x faster at n = 76–100.
    /// </summary>
    private static int Kacc22(int nh) => 2;

    // ================================================================ dlaqr0 / dlaqr4

    /// <summary>
    /// LAPACK <c>dlaqr0</c> (<paramref name="recursive"/>) or <c>dlaqr4</c> (not): the multishift
    /// driver. Indices are 1-based; <paramref name="wrp"/>/<paramref name="wip"/> point at WR(1).
    /// </summary>
    internal static int Laqr0(bool wantt, bool wantz, int n, int ilo, int ihi, double* H, int ldh,
        double* wrp, double* wip, int iloz, int ihiz, double* Z, int ldz, bool recursive, int lwork = int.MaxValue)
    {
        if (n == 0) return 0;
        if (n <= NTiny) return Lahqr(wantt, wantz, n, ilo, ihi, H, ldh, wrp, wip, iloz, ihiz, Z, ldz);

        var h = H - ldh - 1;
        var wr = wrp - 1;
        var wi = wip - 1;

        var nhAll = ihi - ilo + 1;
        var nwr = Math.Max(2, WindowSize(nhAll));
        nwr = Math.Min(Math.Min(ihi - ilo + 1, (n - 1) / 3), nwr);
        var nsr = ShiftCount(nhAll);
        nsr = Math.Min(Math.Min(nsr, (n - 3) / 6), ihi - ilo);
        nsr = Math.Max(2, nsr - nsr % 2);

        var nmin = Math.Max(NTiny, NMin);
        var kacc22 = Math.Min(2, Math.Max(0, Kacc22(nhAll)));
        var nwmax = Math.Min((n - 1) / 3, lwork / 2);
        var nw = nwmax;
        var nsmax = Math.Min((n - 3) / 6, (int)Math.Min(int.MaxValue, 2L * lwork / 3));
        nsmax -= nsmax % 2;
        var ndfl = 1;
        var itmax = Math.Max(30, 2 * KExSh) * Math.Max(10, ihi - ilo + 1);
        var kbot = ihi;
        var ndec = -1;

        // Workspaces: deflation window (T, V, WV, WH), shift block, bulge chase (V, U, WV, WH).
        var aedT = new double[nwmax * nwmax];
        var aedV = new double[nwmax * nwmax];
        var aedWv = new double[n * nwmax];
        var aedWh = new double[nwmax * n];
        var aedWork = new double[Math.Max(2 * nwmax, n) + 1];
        var shiftH = new double[Math.Max(1, nsmax * nsmax)];
        var nbmpsMax = Math.Max(1, nsmax / 2);
        var kduMax = 4 * nbmpsMax;
        var bV = new double[3 * nbmpsMax];
        var bU = new double[kduMax * kduMax];
        var bWv = new double[n * kduMax];
        var bWh = new double[kduMax * n];

        fixed (double* pT = aedT, pV = aedV, pWv = aedWv, pWh = aedWh, pWork = aedWork, pS = shiftH,
               qV = bV, qU = bU, qWv = bWv, qWh = bWh)
        {
            for (var it = 1; it <= itmax; it++)
            {
                if (kbot < ilo) return 0;

                // Locate the active block.
                int k;
                for (k = kbot; k >= ilo + 1; k--)
                    if (h[k * ldh + k - 1] == 0.0) break;
                var ktop = k;

                // Deflation window size: grow it after nondeflating steps, then shrink it once.
                var nh = kbot - ktop + 1;
                var nwupbd = Math.Min(nh, nwmax);
                nw = ndfl < KExNw ? Math.Min(nwupbd, nwr) : Math.Min(nwupbd, 2 * nw);
                if (nw < nwmax)
                {
                    if (nw >= nh - 1) nw = nh;
                    else
                    {
                        var kwtop = kbot - nw + 1;
                        if (Math.Abs(h[kwtop * ldh + kwtop - 1]) > Math.Abs(h[(kwtop - 1) * ldh + kwtop - 2])) nw++;
                    }
                }
                if (ndfl < KExNw) ndec = -1;
                else if (ndec >= 0 || nw >= nwupbd)
                {
                    ndec++;
                    if (nw - ndec < 2) ndec = 0;
                    nw -= ndec;
                }

                // Aggressive early deflation.
                Aed(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz, ihiz, Z, ldz, out var ls, out var ld,
                    wrp, wip, pT, nwmax, pV, nwmax, pWv, nwmax, pWh, n, pWork, recursive);

                kbot -= ld;
                var ks = kbot - ls + 1;

                // Skip the sweep when deflation did enough on its own (more than nibble percent).
                if (ld == 0 || (100 * ld <= nw * Nibble && kbot - ktop + 1 > Math.Min(nmin, nwmax)))
                {
                    var ns = Math.Min(Math.Min(nsmax, nsr), Math.Max(2, kbot - ktop));
                    ns -= ns % 2;

                    if (ndfl % KExSh == 0)
                    {
                        // Exceptional shifts.
                        ks = kbot - ns + 1;
                        for (var i = kbot; i >= Math.Max(ks + 1, ktop + 2); i -= 2)
                        {
                            var ss = Math.Abs(h[i * ldh + i - 1]) + Math.Abs(h[(i - 1) * ldh + i - 2]);
                            var aa = Wilk1 * ss + h[i * ldh + i];
                            var bb = ss;
                            var cc = Wilk2 * ss;
                            var dd = aa;
                            Lanv2Full(ref aa, ref bb, ref cc, ref dd, out wr[i - 1], out wi[i - 1], out wr[i], out wi[i], out _, out _);
                        }
                        if (ks == ktop)
                        {
                            wr[ks + 1] = h[(ks + 1) * ldh + ks + 1];
                            wi[ks + 1] = 0.0;
                            wr[ks] = wr[ks + 1];
                            wi[ks] = wi[ks + 1];
                        }
                    }
                    else
                    {
                        // Too few shifts from deflation: take the eigenvalues of the trailing block.
                        if (kbot - ks + 1 <= ns / 2)
                        {
                            ks = kbot - ns + 1;
                            for (var r = 0; r < ns; r++)
                                for (var c = 0; c < ns; c++)
                                    pS[r * ns + c] = h[(ks + r) * ldh + ks + c];
                            int inf;
                            if (ns > nmin && recursive)
                                inf = Laqr0(false, false, ns, 1, ns, pS, ns, wr + ks, wi + ks, 1, 1, null, 1, recursive: false, lwork);
                            else
                                inf = Lahqr(false, false, ns, 1, ns, pS, ns, wr + ks, wi + ks, 1, 1, null, 1);
                            ks += inf;

                            // dlahqr failed completely: use the trailing 2×2's eigenvalues.
                            if (ks >= kbot)
                            {
                                var aa = h[(kbot - 1) * ldh + kbot - 1];
                                var cc = h[kbot * ldh + kbot - 1];
                                var bb = h[(kbot - 1) * ldh + kbot];
                                var dd = h[kbot * ldh + kbot];
                                Lanv2Full(ref aa, ref bb, ref cc, ref dd, out wr[kbot - 1], out wi[kbot - 1], out wr[kbot], out wi[kbot], out _, out _);
                                ks = kbot - 1;
                            }
                        }

                        if (kbot - ks + 1 > ns)
                        {
                            // Sort by decreasing magnitude so the smallest ones are used.
                            var sorted = false;
                            for (k = kbot; k >= ks + 1; k--)
                            {
                                if (sorted) break;
                                sorted = true;
                                for (var i = ks; i <= k - 1; i++)
                                {
                                    if (Math.Abs(wr[i]) + Math.Abs(wi[i]) < Math.Abs(wr[i + 1]) + Math.Abs(wi[i + 1]))
                                    {
                                        sorted = false;
                                        (wr[i], wr[i + 1]) = (wr[i + 1], wr[i]);
                                        (wi[i], wi[i + 1]) = (wi[i + 1], wi[i]);
                                    }
                                }
                            }
                        }

                        // Shuffle so complex conjugate pairs stay together.
                        for (var i = kbot; i >= ks + 2; i -= 2)
                        {
                            if (wi[i] != -wi[i - 1])
                            {
                                var swap = wr[i]; wr[i] = wr[i - 1]; wr[i - 1] = wr[i - 2]; wr[i - 2] = swap;
                                swap = wi[i]; wi[i] = wi[i - 1]; wi[i - 1] = wi[i - 2]; wi[i - 2] = swap;
                            }
                        }
                    }

                    // Two real shifts: use only the one closer to H(kbot, kbot), twice.
                    if (kbot - ks + 1 == 2 && wi[kbot] == 0.0)
                    {
                        if (Math.Abs(wr[kbot] - h[kbot * ldh + kbot]) < Math.Abs(wr[kbot - 1] - h[kbot * ldh + kbot]))
                            wr[kbot - 1] = wr[kbot];
                        else
                            wr[kbot] = wr[kbot - 1];
                    }

                    ns = Math.Min(ns, kbot - ks + 1);
                    ns -= ns % 2;
                    ks = kbot - ns + 1;

                    Laqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, wr + ks, wi + ks, H, ldh, iloz, ihiz, Z, ldz,
                        qV, qU, kduMax, qWv, kduMax, qWh, n);
                }

                ndfl = ld > 0 ? 1 : ndfl + 1;
            }
            return kbot;
        }
    }

    // ================================================================ dlaqr3 / dlaqr2

    /// <summary>
    /// LAPACK <c>dlaqr3</c> (<paramref name="recursive"/>: windows above <see cref="NMin"/> use the
    /// multishift code) or <c>dlaqr2</c> (always <c>dlahqr</c>). Aggressive early deflation on the
    /// trailing nw×nw window: Schur-factor the window, test the spike for negligible entries, move
    /// the undeflatable eigenvalues up with <c>dtrexc</c>, and return the rest as shifts.
    /// </summary>
    private static void Aed(bool wantt, bool wantz, int n, int ktop, int kbot, int nw, double* H, int ldh,
        int iloz, int ihiz, double* Z, int ldz, out int ns, out int nd, double* srp, double* sip,
        double* Tp, int ldt, double* Vp, int ldv, double* WVp, int ldwv, double* WHp, int ldwh, double* work, bool recursive)
    {
        var h = H - ldh - 1;
        var z = Z - ldz - 1;
        var sr = srp - 1;
        var si = sip - 1;
        var t = Tp - ldt - 1;
        var v = Vp - ldv - 1;

        ns = 0;
        nd = 0;
        if (ktop > kbot || nw < 1) return;
        var jw = Math.Min(nw, kbot - ktop + 1);

        var smlnum = SafeMin * (n / Ulp);
        var kwtop = kbot - jw + 1;
        var s = kwtop == ktop ? 0.0 : h[kwtop * ldh + kwtop - 1];

        if (kbot == kwtop)
        {
            // 1×1 window.
            sr[kwtop] = h[kwtop * ldh + kwtop];
            si[kwtop] = 0.0;
            ns = 1;
            nd = 0;
            if (Math.Abs(s) <= Math.Max(smlnum, Ulp * Math.Abs(h[kwtop * ldh + kwtop])))
            {
                ns = 0;
                nd = 1;
                if (kwtop > ktop) h[kwtop * ldh + kwtop - 1] = 0.0;
            }
            return;
        }

        // T = the window (upper Hessenberg part only), V = I.
        for (var i = 1; i <= jw; i++)
        {
            new Span<double>(Tp + (long)(i - 1) * ldt, jw).Clear();
            new Span<double>(Vp + (long)(i - 1) * ldv, jw).Clear();
            v[i * ldv + i] = 1.0;
            for (var j = Math.Max(1, i - 1); j <= jw; j++)
                t[i * ldt + j] = h[(kwtop + i - 1) * ldh + kwtop + j - 1];
        }

        int infqr;
        if (recursive && jw > NMin)
            infqr = Laqr0(true, true, jw, 1, jw, Tp, ldt, sr + kwtop, si + kwtop, 1, jw, Vp, ldv, recursive: false);
        else
            infqr = Lahqr(true, true, jw, 1, jw, Tp, ldt, sr + kwtop, si + kwtop, 1, jw, Vp, ldv);

        // dtrexc needs a clean subdiagonal band.
        for (var j = 1; j <= jw - 3; j++)
        {
            t[(j + 2) * ldt + j] = 0.0;
            t[(j + 3) * ldt + j] = 0.0;
        }
        if (jw > 2) t[jw * ldt + jw - 2] = 0.0;

        // Deflation detection loop.
        ns = jw;
        var ilst = infqr + 1;
        while (ilst <= ns)
        {
            var bulge = ns != 1 && t[ns * ldt + ns - 1] != 0.0;
            if (!bulge)
            {
                // Real eigenvalue.
                var foo = Math.Abs(t[ns * ldt + ns]);
                if (foo == 0.0) foo = Math.Abs(s);
                if (Math.Abs(s * v[1 * ldv + ns]) <= Math.Max(smlnum, Ulp * foo))
                    ns -= 1; // deflatable
                else
                {
                    // Undeflatable: move it up out of the way.
                    var ifst = ns;
                    Trexc(true, jw, Tp, ldt, Vp, ldv, ref ifst, ref ilst, work);
                    ilst += 1;
                }
            }
            else
            {
                // Complex conjugate pair.
                var foo = Math.Abs(t[ns * ldt + ns]) + Math.Sqrt(Math.Abs(t[ns * ldt + ns - 1])) * Math.Sqrt(Math.Abs(t[(ns - 1) * ldt + ns]));
                if (foo == 0.0) foo = Math.Abs(s);
                if (Math.Max(Math.Abs(s * v[1 * ldv + ns]), Math.Abs(s * v[1 * ldv + ns - 1])) <= Math.Max(smlnum, Ulp * foo))
                    ns -= 2;
                else
                {
                    var ifst = ns;
                    Trexc(true, jw, Tp, ldt, Vp, ldv, ref ifst, ref ilst, work);
                    ilst += 2;
                }
            }
        }

        if (ns == 0) s = 0.0;

        if (ns < jw)
        {
            // Sort the diagonal of T by decreasing magnitude (deflated part, for better shifts).
            var sorted = false;
            var i = ns + 1;
            while (!sorted)
            {
                sorted = true;
                var kend = i - 1;
                i = infqr + 1;
                int k;
                if (i == ns) k = i + 1;
                else if (t[(i + 1) * ldt + i] == 0.0) k = i + 1;
                else k = i + 2;
                while (k <= kend)
                {
                    double evi, evk;
                    if (k == i + 1) evi = Math.Abs(t[i * ldt + i]);
                    else evi = Math.Abs(t[i * ldt + i]) + Math.Sqrt(Math.Abs(t[(i + 1) * ldt + i])) * Math.Sqrt(Math.Abs(t[i * ldt + i + 1]));

                    if (k == kend) evk = Math.Abs(t[k * ldt + k]);
                    else if (t[(k + 1) * ldt + k] == 0.0) evk = Math.Abs(t[k * ldt + k]);
                    else evk = Math.Abs(t[k * ldt + k]) + Math.Sqrt(Math.Abs(t[(k + 1) * ldt + k])) * Math.Sqrt(Math.Abs(t[k * ldt + k + 1]));

                    if (evi >= evk) i = k;
                    else
                    {
                        sorted = false;
                        var ifst = i;
                        var il = k;
                        var info = Trexc(true, jw, Tp, ldt, Vp, ldv, ref ifst, ref il, work);
                        i = info == 0 ? il : k;
                    }
                    if (i == kend) k = i + 1;
                    else if (t[(i + 1) * ldt + i] == 0.0) k = i + 1;
                    else k = i + 2;
                }
            }
        }

        // Restore shift/eigenvalue arrays.
        {
            var i = jw;
            while (i >= infqr + 1)
            {
                if (i == infqr + 1 || t[i * ldt + i - 1] == 0.0)
                {
                    sr[kwtop + i - 1] = t[i * ldt + i];
                    si[kwtop + i - 1] = 0.0;
                    i -= 1;
                }
                else
                {
                    var aa = t[(i - 1) * ldt + i - 1];
                    var cc = t[i * ldt + i - 1];
                    var bb = t[(i - 1) * ldt + i];
                    var dd = t[i * ldt + i];
                    Lanv2Full(ref aa, ref bb, ref cc, ref dd, out sr[kwtop + i - 2], out si[kwtop + i - 2],
                        out sr[kwtop + i - 1], out si[kwtop + i - 1], out _, out _);
                    i -= 2;
                }
            }
        }

        if (ns < jw || s == 0.0)
        {
            var tau = work + jw; // jw entries for the window's Hessenberg reflectors
            var scratch = new double[Math.Max(jw, n)];
            fixed (double* psc = scratch)
            {
                if (ns > 1 && s != 0.0)
                {
                    // Reflection to reduce the spike to a single entry, then back to Hessenberg.
                    for (var j = 0; j < ns; j++) work[j] = v[1 * ldv + j + 1];
                    var beta = work[0];
                    Larfg(ns, ref beta, work + 1, 1, out var tauS);
                    for (var i = 3; i <= jw; i++)
                        for (var j = 1; j <= i - 2; j++) t[i * ldt + j] = 0.0;
                    ApplyLeft(ns, jw, work, 1, tauS, Tp, ldt, psc);
                    ApplyRight(ns, ns, work, 1, tauS, Tp, ldt, psc);
                    ApplyRight(jw, ns, work, 1, tauS, Vp, ldv, psc);
                    LinalgHessenbergBlocked.Gehd2(jw, 1, ns, Tp, ldt, tau);
                }

                // Copy the updated window back into H.
                if (kwtop > 1) h[kwtop * ldh + kwtop - 1] = s * v[1 * ldv + 1];
                for (var i = 1; i <= jw; i++)
                    for (var j = i; j <= jw; j++)
                        h[(kwtop + i - 1) * ldh + kwtop + j - 1] = t[i * ldt + j];
                for (var i = 1; i <= jw - 1; i++)
                    h[(kwtop + i) * ldh + kwtop + i - 1] = t[(i + 1) * ldt + i];

                // Accumulate the window Hessenberg reflectors into V (dormhr, right, no transpose).
                if (ns > 1 && s != 0.0)
                    for (var i = 1; i <= ns - 1; i++)
                        ApplyRight(jw, ns - i, Tp + (long)i * ldt + (i - 1), ldt, tau[i - 1], Vp + i, ldv, psc);
            }

            // Off-diagonal updates as matrix products.
            var ltop = wantt ? 1 : ktop;
            var rows = kwtop - ltop;
            if (rows > 0)
            {
                var blk = H + (long)(ltop - 1) * ldh + (kwtop - 1);
                Gemm(false, false, rows, jw, jw, 1.0, blk, ldh, Vp, ldv, 0.0, WVp, ldwv);
                CopyBlock(rows, jw, WVp, ldwv, blk, ldh);
            }
            if (wantt && kbot < n)
            {
                var cols = n - kbot;
                var blk = H + (long)(kwtop - 1) * ldh + kbot;
                Gemm(true, false, jw, cols, jw, 1.0, Vp, ldv, blk, ldh, 0.0, WHp, ldwh);
                CopyBlock(jw, cols, WHp, ldwh, blk, ldh);
            }
            if (wantz)
            {
                var zrows = ihiz - iloz + 1;
                var blk = Z + (long)(iloz - 1) * ldz + (kwtop - 1);
                Gemm(false, false, zrows, jw, jw, 1.0, blk, ldz, Vp, ldv, 0.0, WVp, ldwv);
                CopyBlock(zrows, jw, WVp, ldwv, blk, ldz);
            }
        }

        nd = jw - ns;
        ns -= infqr;
        _ = z;
    }

    // ================================================================ dlaqr5

    /// <summary>
    /// LAPACK <c>dlaqr5</c> (3.10 form): one multishift QR sweep with <paramref name="nshfts"/>
    /// shifts as a chain of 3×3 bulges chased two columns at a time. With
    /// <paramref name="kacc22"/> ≥ 1 the reflectors of each chain position are accumulated into U
    /// and applied to the rest of H and to Z as matrix products.
    /// </summary>
    private static void Laqr5(bool wantt, bool wantz, int kacc22, int n, int ktop, int kbot, int nshfts,
        double* srp, double* sip, double* H, int ldh, int iloz, int ihiz, double* Z, int ldz,
        double* Vbuf, double* Up, int ldu, double* WVp, int ldwv, double* WHp, int ldwh)
    {
        if (nshfts < 2 || ktop >= kbot) return;
        var h = H - ldh - 1;
        var z = Z - ldz - 1;
        var sr = srp - 1;
        var si = sip - 1;
        var u = Up - ldu - 1;
        // V(i, m) = vb[3 * m + i] (1-based i = 1..3, m = 1..nbmps): column m contiguous.
        var vb = Vbuf - 4;

        // Shuffle shifts into pairs of real shifts and conjugate pairs.
        for (var i = 1; i <= nshfts - 2; i += 2)
        {
            if (si[i] != -si[i + 1])
            {
                var swap = sr[i]; sr[i] = sr[i + 1]; sr[i + 1] = sr[i + 2]; sr[i + 2] = swap;
                swap = si[i]; si[i] = si[i + 1]; si[i + 1] = si[i + 2]; si[i + 2] = swap;
            }
        }

        var ns = nshfts - nshfts % 2;
        var smlnum = SafeMin * (n / Ulp);
        var accum = kacc22 == 1 || kacc22 == 2;

        if (ktop + 2 <= kbot) h[(ktop + 2) * ldh + ktop] = 0.0;

        var nbmps = ns / 2;
        var kdu = 4 * nbmps;
        Span<double> vt = stackalloc double[3];

        for (var incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps)
        {
            int jtop;
            if (accum) jtop = Math.Max(ktop, incol);
            else if (wantt) jtop = 1;
            else jtop = ktop;

            var ndcol = incol + kdu;
            if (accum)
            {
                for (var r = 0; r < kdu; r++)
                {
                    new Span<double>(Up + (long)r * ldu, kdu).Clear();
                    Up[(long)r * ldu + r] = 1.0;
                }
            }

            for (var krcol = incol; krcol <= Math.Min(incol + 2 * nbmps - 1, kbot - 2); krcol++)
            {
                var mtop = Math.Max(1, (ktop - krcol) / 2 + 1);
                var mbot = Math.Min(nbmps, (kbot - krcol - 1) / 2);
                var m22 = mbot + 1;
                var bmp22 = mbot < nbmps && krcol + 2 * (m22 - 1) == kbot - 2;

                if (bmp22)
                {
                    // Special case: a 2×2 reflection at the bottom.
                    var k = krcol + 2 * (m22 - 1);
                    var vm = vb + 3 * m22;
                    if (k == ktop - 1)
                    {
                        Laqr1(2, H + (long)k * ldh + k, ldh, sr[2 * m22 - 1], si[2 * m22 - 1], sr[2 * m22], si[2 * m22], vm + 1);
                        var beta = vm[1];
                        Larfg(2, ref beta, vm + 2, 1, out vm[1]);
                    }
                    else
                    {
                        var beta = h[(k + 1) * ldh + k];
                        vm[2] = h[(k + 2) * ldh + k];
                        Larfg(2, ref beta, vm + 2, 1, out vm[1]);
                        h[(k + 1) * ldh + k] = beta;
                        h[(k + 2) * ldh + k] = 0.0;
                    }

                    var t1 = vm[1];
                    var t2 = t1 * vm[2];
                    var v2 = vm[2];
                    for (var j = jtop; j <= Math.Min(kbot, k + 3); j++)
                    {
                        var row = h + j * ldh;
                        var refsum = row[k + 1] + v2 * row[k + 2];
                        row[k + 1] -= refsum * t1;
                        row[k + 2] -= refsum * t2;
                    }

                    int jbot;
                    if (accum) jbot = Math.Min(ndcol, kbot);
                    else if (wantt) jbot = n;
                    else jbot = kbot;
                    {
                        var r1 = h + (k + 1) * ldh;
                        var r2 = h + (k + 2) * ldh;
                        for (var j = k + 1; j <= jbot; j++)
                        {
                            var refsum = r1[j] + v2 * r2[j];
                            r1[j] -= refsum * t1;
                            r2[j] -= refsum * t2;
                        }
                    }

                    // Convergence test, a la Ahues & Tisseur (LAWN 122, 1997).
                    if (k >= ktop) TestDeflate(h, ldh, k, ktop, kbot, smlnum);

                    if (accum)
                    {
                        var kms = k - incol;
                        for (var j = Math.Max(1, ktop - incol); j <= kdu; j++)
                        {
                            var row = u + j * ldu;
                            var refsum = row[kms + 1] + v2 * row[kms + 2];
                            row[kms + 1] -= refsum * t1;
                            row[kms + 2] -= refsum * t2;
                        }
                    }
                    else if (wantz)
                    {
                        for (var j = iloz; j <= ihiz; j++)
                        {
                            var row = z + j * ldz;
                            var refsum = row[k + 1] + v2 * row[k + 2];
                            row[k + 1] -= refsum * t1;
                            row[k + 2] -= refsum * t2;
                        }
                    }
                }

                // Normal case: chain of 3×3 reflections.
                for (var m = mbot; m >= mtop; m--)
                {
                    var k = krcol + 2 * (m - 1);
                    var vm = vb + 3 * m;
                    if (k == ktop - 1)
                    {
                        Laqr1(3, H + (long)(ktop - 1) * ldh + (ktop - 1), ldh, sr[2 * m - 1], si[2 * m - 1], sr[2 * m], si[2 * m], vm + 1);
                        var alpha = vm[1];
                        Larfg(3, ref alpha, vm + 2, 1, out vm[1]);
                    }
                    else
                    {
                        // Perform delayed transformation of row below the mth bulge.
                        var t1d = vm[1];
                        var t2d = t1d * vm[2];
                        var t3d = t1d * vm[3];
                        var refsumD = vm[3] * h[(k + 3) * ldh + k + 2];
                        h[(k + 3) * ldh + k] = -refsumD * t1d;
                        h[(k + 3) * ldh + k + 1] = -refsumD * t2d;
                        h[(k + 3) * ldh + k + 2] -= refsumD * t3d;

                        // Calculate reflection to move the mth bulge one step.
                        var beta = h[(k + 1) * ldh + k];
                        vm[2] = h[(k + 2) * ldh + k];
                        vm[3] = h[(k + 3) * ldh + k];
                        Larfg(3, ref beta, vm + 2, 1, out vm[1]);

                        if (h[(k + 3) * ldh + k] != 0.0 || h[(k + 3) * ldh + k + 1] != 0.0 || h[(k + 3) * ldh + k + 2] == 0.0)
                        {
                            // Typical case: not collapsed (yet).
                            h[(k + 1) * ldh + k] = beta;
                            h[(k + 2) * ldh + k] = 0.0;
                            h[(k + 3) * ldh + k] = 0.0;
                        }
                        else
                        {
                            // Atypical case: collapsed bulge. Try a reflector built from the
                            // shifts and keep it only if it introduces negligible fill-in.
                            fixed (double* pvt = vt)
                            {
                                Laqr1(3, H + (long)k * ldh + k, ldh, sr[2 * m - 1], si[2 * m - 1], sr[2 * m], si[2 * m], pvt);
                                var alpha = pvt[0];
                                Larfg(3, ref alpha, pvt + 1, 1, out pvt[0]);
                                var t1 = pvt[0];
                                var t2 = t1 * pvt[1];
                                var t3 = t1 * pvt[2];
                                var refsum = h[(k + 1) * ldh + k] + pvt[1] * h[(k + 2) * ldh + k];
                                if (Math.Abs(h[(k + 2) * ldh + k] - refsum * t2) + Math.Abs(refsum * t3) >
                                    Ulp * (Math.Abs(h[k * ldh + k]) + Math.Abs(h[(k + 1) * ldh + k + 1]) + Math.Abs(h[(k + 2) * ldh + k + 2])))
                                {
                                    // Starting a new bulge here would create non-negligible fill-in.
                                    h[(k + 1) * ldh + k] = beta;
                                    h[(k + 2) * ldh + k] = 0.0;
                                    h[(k + 3) * ldh + k] = 0.0;
                                }
                                else
                                {
                                    h[(k + 1) * ldh + k] -= refsum * t1;
                                    h[(k + 2) * ldh + k] = 0.0;
                                    h[(k + 3) * ldh + k] = 0.0;
                                    vm[1] = pvt[0];
                                    vm[2] = pvt[1];
                                    vm[3] = pvt[2];
                                }
                            }
                        }
                    }

                    // Apply reflection from the right and the first column of update from the left.
                    {
                        var t1 = vm[1];
                        var t2 = t1 * vm[2];
                        var t3 = t1 * vm[3];
                        var v2 = vm[2];
                        var v3 = vm[3];
                        for (var j = jtop; j <= Math.Min(kbot, k + 3); j++)
                        {
                            var row = h + j * ldh;
                            var refsum = row[k + 1] + v2 * row[k + 2] + v3 * row[k + 3];
                            row[k + 1] -= refsum * t1;
                            row[k + 2] -= refsum * t2;
                            row[k + 3] -= refsum * t3;
                        }
                        var rs = h[(k + 1) * ldh + k + 1] + v2 * h[(k + 2) * ldh + k + 1] + v3 * h[(k + 3) * ldh + k + 1];
                        h[(k + 1) * ldh + k + 1] -= rs * t1;
                        h[(k + 2) * ldh + k + 1] -= rs * t2;
                        h[(k + 3) * ldh + k + 1] -= rs * t3;
                    }

                    if (k < ktop) continue;
                    TestDeflate(h, ldh, k, ktop, kbot, smlnum);
                }

                // Multiply H by the reflections from the left.
                {
                    int jbot;
                    if (accum) jbot = Math.Min(ndcol, kbot);
                    else if (wantt) jbot = n;
                    else jbot = kbot;
                    for (var m = mbot; m >= mtop; m--)
                    {
                        var k = krcol + 2 * (m - 1);
                        var vm = vb + 3 * m;
                        var j0 = Math.Max(ktop, krcol + 2 * m);
                        if (j0 <= jbot)
                            LeftReflect3(h + (k + 1) * ldh, ldh, j0, jbot, vm[1], vm[2], vm[3]);
                    }
                }

                // Accumulate orthogonal transformations.
                if (accum)
                {
                    for (var m = mbot; m >= mtop; m--)
                    {
                        var k = krcol + 2 * (m - 1);
                        var kms = k - incol;
                        var i2 = Math.Max(1, ktop - incol);
                        i2 = Math.Max(i2, kms - (krcol - incol) + 1);
                        var i4 = Math.Min(kdu, krcol + 2 * (mbot - 1) - incol + 5);
                        var vm = vb + 3 * m;
                        var t1 = vm[1];
                        var t2 = t1 * vm[2];
                        var t3 = t1 * vm[3];
                        var v2 = vm[2];
                        var v3 = vm[3];
                        for (var j = i2; j <= i4; j++)
                        {
                            var row = u + j * ldu;
                            var refsum = row[kms + 1] + v2 * row[kms + 2] + v3 * row[kms + 3];
                            row[kms + 1] -= refsum * t1;
                            row[kms + 2] -= refsum * t2;
                            row[kms + 3] -= refsum * t3;
                        }
                    }
                }
                else if (wantz)
                {
                    for (var m = mbot; m >= mtop; m--)
                    {
                        var k = krcol + 2 * (m - 1);
                        var vm = vb + 3 * m;
                        var t1 = vm[1];
                        var t2 = t1 * vm[2];
                        var t3 = t1 * vm[3];
                        var v2 = vm[2];
                        var v3 = vm[3];
                        for (var j = iloz; j <= ihiz; j++)
                        {
                            var row = z + j * ldz;
                            var refsum = row[k + 1] + v2 * row[k + 2] + v3 * row[k + 3];
                            row[k + 1] -= refsum * t1;
                            row[k + 2] -= refsum * t2;
                            row[k + 3] -= refsum * t3;
                        }
                    }
                }
            }

            // Off-diagonal updates with the accumulated U, as matrix products.
            if (accum)
            {
                int jtopA, jbotA;
                if (wantt) { jtopA = 1; jbotA = n; }
                else { jtopA = ktop; jbotA = kbot; }
                var k1 = Math.Max(1, ktop - incol);
                var nu = (kdu - Math.Max(0, ndcol - kbot)) - k1 + 1;
                var uk = Up + (long)(k1 - 1) * ldu + (k1 - 1);

                // Horizontal multiply.
                var jcol = Math.Min(ndcol, kbot) + 1;
                var jlen = jbotA - jcol + 1;
                if (jlen > 0)
                {
                    var blk = H + (long)(incol + k1 - 1) * ldh + (jcol - 1);
                    Gemm(true, false, nu, jlen, nu, 1.0, uk, ldu, blk, ldh, 0.0, WHp, ldwh);
                    CopyBlock(nu, jlen, WHp, ldwh, blk, ldh);
                }

                // Vertical multiply.
                var rows = Math.Max(ktop, incol) - jtopA;
                if (rows > 0)
                {
                    var blk = H + (long)(jtopA - 1) * ldh + (incol + k1 - 1);
                    Gemm(false, false, rows, nu, nu, 1.0, blk, ldh, uk, ldu, 0.0, WVp, ldwv);
                    CopyBlock(rows, nu, WVp, ldwv, blk, ldh);
                }

                // Z multiply.
                if (wantz)
                {
                    var zrows = ihiz - iloz + 1;
                    var blk = Z + (long)(iloz - 1) * ldz + (incol + k1 - 1);
                    Gemm(false, false, zrows, nu, nu, 1.0, blk, ldz, uk, ldu, 0.0, WVp, ldwv);
                    CopyBlock(zrows, nu, WVp, ldwv, blk, ldz);
                }
            }
        }
    }

    /// <summary>
    /// dlaqr5's deflation test on H(k+1, k) (Ahues and Tisseur): sets it to zero when negligible.
    /// </summary>
    private static void TestDeflate(double* h, int ldh, int k, int ktop, int kbot, double smlnum)
    {
        var hk1k = h[(k + 1) * ldh + k];
        if (hk1k == 0.0) return;
        var tst1 = Math.Abs(h[k * ldh + k]) + Math.Abs(h[(k + 1) * ldh + k + 1]);
        if (tst1 == 0.0)
        {
            if (k >= ktop + 1) tst1 += Math.Abs(h[k * ldh + k - 1]);
            if (k >= ktop + 2) tst1 += Math.Abs(h[k * ldh + k - 2]);
            if (k >= ktop + 3) tst1 += Math.Abs(h[k * ldh + k - 3]);
            if (k <= kbot - 2) tst1 += Math.Abs(h[(k + 2) * ldh + k + 1]);
            if (k <= kbot - 3) tst1 += Math.Abs(h[(k + 3) * ldh + k + 1]);
            if (k <= kbot - 4) tst1 += Math.Abs(h[(k + 4) * ldh + k + 1]);
        }
        if (Math.Abs(hk1k) <= Math.Max(smlnum, Ulp * tst1))
        {
            var hkk1 = h[k * ldh + k + 1];
            var h12 = Math.Max(Math.Abs(hk1k), Math.Abs(hkk1));
            var h21 = Math.Min(Math.Abs(hk1k), Math.Abs(hkk1));
            var diff = Math.Abs(h[k * ldh + k] - h[(k + 1) * ldh + k + 1]);
            var h11 = Math.Max(Math.Abs(h[(k + 1) * ldh + k + 1]), diff);
            var h22 = Math.Min(Math.Abs(h[(k + 1) * ldh + k + 1]), diff);
            var scl = h11 + h12;
            var tst2 = h22 * (h11 / scl);
            if (tst2 == 0.0 || h21 * (h12 / scl) <= Math.Max(smlnum, Ulp * tst2))
                h[(k + 1) * ldh + k] = 0.0;
        }
    }

    /// <summary>
    /// Rows r0, r0+ld, r0+2·ld (1-based columns j0..j1) ← (I − τ v vᵀ) rows, v = (1, v2, v3),
    /// with dlaqr5's arithmetic (refsum, then three subtractions), four columns at a time.
    /// </summary>
    private static void LeftReflect3(double* r0, int ld, int j0, int j1, double tau, double v2, double v3)
    {
        var r1 = r0 + ld;
        var r2 = r1 + ld;
        var t1 = tau;
        var t2 = tau * v2;
        var t3 = tau * v3;
        var j = j0;
        if (Vector256.IsHardwareAccelerated)
        {
            var vv2 = Vector256.Create(v2); var vv3 = Vector256.Create(v3);
            var vt1 = Vector256.Create(t1); var vt2 = Vector256.Create(t2); var vt3 = Vector256.Create(t3);
            for (; j <= j1 - 3; j += 4)
            {
                var a = Vector256.Load(r0 + j);
                var b = Vector256.Load(r1 + j);
                var c = Vector256.Load(r2 + j);
                var refsum = a + vv2 * b + vv3 * c;
                Vector256.Store(a - refsum * vt1, r0 + j);
                Vector256.Store(b - refsum * vt2, r1 + j);
                Vector256.Store(c - refsum * vt3, r2 + j);
            }
        }
        for (; j <= j1; j++)
        {
            var refsum = r0[j] + v2 * r1[j] + v3 * r2[j];
            r0[j] -= refsum * t1;
            r1[j] -= refsum * t2;
            r2[j] -= refsum * t3;
        }
    }

    // ================================================================ dlahqr

    /// <summary>
    /// LAPACK <c>dlahqr</c>: the double-shift QR iteration for small Hessenberg blocks, with the
    /// Ahues–Tisseur deflation test and standardised 2×2 blocks. 1-based ilo/ihi/iloz/ihiz;
    /// <paramref name="wrp"/>/<paramref name="wip"/> point at WR(1). Returns 0 or the failing index.
    /// </summary>
    internal static int Lahqr(bool wantt, bool wantz, int n, int ilo, int ihi, double* H, int ldh,
        double* wrp, double* wip, int iloz, int ihiz, double* Z, int ldz)
    {
        if (n == 0) return 0;
        var h = H - ldh - 1;
        var z = Z - ldz - 1;
        var wr = wrp - 1;
        var wi = wip - 1;

        if (ilo == ihi)
        {
            wr[ilo] = h[ilo * ldh + ilo];
            wi[ilo] = 0.0;
            return 0;
        }

        // Clear out the trash.
        for (var j = ilo; j <= ihi - 3; j++)
        {
            h[(j + 2) * ldh + j] = 0.0;
            h[(j + 3) * ldh + j] = 0.0;
        }
        if (ilo <= ihi - 2) h[ihi * ldh + ihi - 2] = 0.0;

        var nh = ihi - ilo + 1;
        var nz = ihiz - iloz + 1;
        var smlnum = SafeMin * (nh / Ulp);

        int i1 = 0, i2 = 0;
        if (wantt)
        {
            i1 = 1;
            i2 = n;
        }

        var itmax = 30 * Math.Max(10, nh);
        var kdefl = 0;
        var i = ihi;
        Span<double> vs = stackalloc double[3];

        while (true)
        {
            var l = ilo;
            if (i < ilo) return 0;

            var converged = false;
            for (var its = 0; its <= itmax; its++)
            {
                // Look for a single small subdiagonal element.
                int k;
                for (k = i; k >= l + 1; k--)
                {
                    var hkk1 = Math.Abs(h[k * ldh + k - 1]);
                    if (hkk1 <= smlnum) break;
                    var tst = Math.Abs(h[(k - 1) * ldh + k - 1]) + Math.Abs(h[k * ldh + k]);
                    if (tst == 0.0)
                    {
                        if (k - 2 >= ilo) tst += Math.Abs(h[(k - 1) * ldh + k - 2]);
                        if (k + 1 <= ihi) tst += Math.Abs(h[(k + 1) * ldh + k]);
                    }
                    if (hkk1 <= Ulp * tst)
                    {
                        var hk1k = Math.Abs(h[(k - 1) * ldh + k]);
                        var ab = Math.Max(hkk1, hk1k);
                        var ba = Math.Min(hkk1, hk1k);
                        var diff = Math.Abs(h[(k - 1) * ldh + k - 1] - h[k * ldh + k]);
                        var aa = Math.Max(Math.Abs(h[k * ldh + k]), diff);
                        var bb = Math.Min(Math.Abs(h[k * ldh + k]), diff);
                        var s0 = aa + ab;
                        if (ba * (ab / s0) <= Math.Max(smlnum, Ulp * (bb * (aa / s0)))) break;
                    }
                }
                l = k;
                if (l > ilo) h[l * ldh + l - 1] = 0.0; // H(L, L-1) is negligible

                // Exit when a 1×1 or 2×2 block splits off at the bottom.
                if (l >= i - 1)
                {
                    converged = true;
                    break;
                }
                kdefl++;

                // The active submatrix is rows/columns l..i; i1..i2 are the columns updated.
                if (!wantt)
                {
                    i1 = l;
                    i2 = i;
                }

                double h11, h12, h21, h22, s;
                if (kdefl % (2 * 10) == 0)
                {
                    // Exceptional shift.
                    s = Math.Abs(h[i * ldh + i - 1]) + Math.Abs(h[(i - 1) * ldh + i - 2]);
                    h11 = 0.75 * s + h[i * ldh + i];
                    h12 = -0.4375 * s;
                    h21 = s;
                    h22 = h11;
                }
                else if (kdefl % 10 == 0)
                {
                    s = Math.Abs(h[(l + 1) * ldh + l]) + Math.Abs(h[(l + 2) * ldh + l + 1]);
                    h11 = 0.75 * s + h[l * ldh + l];
                    h12 = -0.4375 * s;
                    h21 = s;
                    h22 = h11;
                }
                else
                {
                    // Wilkinson's double shift (the eigenvalues of the trailing 2×2).
                    h11 = h[(i - 1) * ldh + i - 1];
                    h21 = h[i * ldh + i - 1];
                    h12 = h[(i - 1) * ldh + i];
                    h22 = h[i * ldh + i];
                }
                s = Math.Abs(h11) + Math.Abs(h12) + Math.Abs(h21) + Math.Abs(h22);
                double rt1r, rt1i, rt2r, rt2i;
                if (s == 0.0)
                {
                    rt1r = rt1i = rt2r = rt2i = 0.0;
                }
                else
                {
                    h11 /= s;
                    h21 /= s;
                    h12 /= s;
                    h22 /= s;
                    var tr = (h11 + h22) / 2.0;
                    var det = (h11 - tr) * (h22 - tr) - h12 * h21;
                    var rtdisc = Math.Sqrt(Math.Abs(det));
                    if (det >= 0.0)
                    {
                        // Complex conjugate shifts.
                        rt1r = tr * s;
                        rt2r = rt1r;
                        rt1i = rtdisc * s;
                        rt2i = -rt1i;
                    }
                    else
                    {
                        // Real shifts (use only one of them).
                        rt1r = tr + rtdisc;
                        rt2r = tr - rtdisc;
                        if (Math.Abs(rt1r - h22) <= Math.Abs(rt2r - h22))
                        {
                            rt1r *= s;
                            rt2r = rt1r;
                        }
                        else
                        {
                            rt2r *= s;
                            rt1r = rt2r;
                        }
                        rt1i = rt2i = 0.0;
                    }
                }

                // Look for two consecutive small subdiagonal elements.
                int m;
                for (m = i - 2; m >= l; m--)
                {
                    var h21s = h[(m + 1) * ldh + m];
                    s = Math.Abs(h[m * ldh + m] - rt2r) + Math.Abs(rt2i) + Math.Abs(h21s);
                    h21s = h[(m + 1) * ldh + m] / s;
                    vs[0] = h21s * h[m * ldh + m + 1] + (h[m * ldh + m] - rt1r) * ((h[m * ldh + m] - rt2r) / s) - rt1i * (rt2i / s);
                    vs[1] = h21s * (h[m * ldh + m] + h[(m + 1) * ldh + m + 1] - rt1r - rt2r);
                    vs[2] = h21s * h[(m + 2) * ldh + m + 1];
                    s = Math.Abs(vs[0]) + Math.Abs(vs[1]) + Math.Abs(vs[2]);
                    vs[0] /= s;
                    vs[1] /= s;
                    vs[2] /= s;
                    if (m == l) break;
                    var h00 = Math.Abs(h[m * ldh + m - 1]) * (Math.Abs(vs[1]) + Math.Abs(vs[2]));
                    var h01 = Ulp * Math.Abs(vs[0]) * (Math.Abs(h[(m - 1) * ldh + m - 1]) + Math.Abs(h[m * ldh + m]) + Math.Abs(h[(m + 1) * ldh + m + 1]));
                    if (h00 <= h01) break;
                }

                // Double-shift QR step.
                fixed (double* v = vs)
                    for (var kk = m; kk <= i - 1; kk++)
                    {
                        // The first iteration determines the reflector from the shifts; later
                        // ones restore H to Hessenberg form (the bulge chase).
                        var nr = Math.Min(3, i - kk + 1);
                        if (kk > m)
                            for (var q = 0; q < nr; q++) v[q] = h[(kk + q) * ldh + kk - 1];
                        Larfg(nr, ref v[0], v + 1, 1, out var t1);
                        if (kk > m)
                        {
                            h[kk * ldh + kk - 1] = v[0];
                            h[(kk + 1) * ldh + kk - 1] = 0.0;
                            if (kk < i - 1) h[(kk + 2) * ldh + kk - 1] = 0.0;
                        }
                        else if (m > l)
                        {
                            // Use the following instead of H(K, K-1) = -H(K, K-1) to avoid a bug
                            // when v(2) and v(3) underflow.
                            h[kk * ldh + kk - 1] *= 1.0 - t1;
                        }
                        var v2 = v[1];
                        var t2 = t1 * v2;
                        if (nr == 3)
                        {
                            var v3 = v[2];
                            var t3 = t1 * v3;
                            // Row change from the left.
                            if (kk <= i2)
                                LeftReflect3(h + kk * ldh, ldh, kk, i2, t1, v2, v3);
                            // Column change from the right.
                            for (var j = i1; j <= Math.Min(kk + 3, i); j++)
                            {
                                var row = h + j * ldh;
                                var sum = row[kk] + v2 * row[kk + 1] + v3 * row[kk + 2];
                                row[kk] -= sum * t1;
                                row[kk + 1] -= sum * t2;
                                row[kk + 2] -= sum * t3;
                            }
                            if (wantz)
                                for (var j = iloz; j <= ihiz; j++)
                                {
                                    var row = z + j * ldz;
                                    var sum = row[kk] + v2 * row[kk + 1] + v3 * row[kk + 2];
                                    row[kk] -= sum * t1;
                                    row[kk + 1] -= sum * t2;
                                    row[kk + 2] -= sum * t3;
                                }
                        }
                        else if (nr == 2)
                        {
                            var r0 = h + kk * ldh;
                            var r1 = r0 + ldh;
                            for (var j = kk; j <= i2; j++)
                            {
                                var sum = r0[j] + v2 * r1[j];
                                r0[j] -= sum * t1;
                                r1[j] -= sum * t2;
                            }
                            for (var j = i1; j <= i; j++)
                            {
                                var row = h + j * ldh;
                                var sum = row[kk] + v2 * row[kk + 1];
                                row[kk] -= sum * t1;
                                row[kk + 1] -= sum * t2;
                            }
                            if (wantz)
                                for (var j = iloz; j <= ihiz; j++)
                                {
                                    var row = z + j * ldz;
                                    var sum = row[kk] + v2 * row[kk + 1];
                                    row[kk] -= sum * t1;
                                    row[kk + 1] -= sum * t2;
                                }
                        }
                    }
            }

            if (!converged) return i; // failure to converge in the remaining block

            if (l == i)
            {
                // H(I, I-1) is negligible: one eigenvalue has converged.
                wr[i] = h[i * ldh + i];
                wi[i] = 0.0;
            }
            else if (l == i - 1)
            {
                // A pair: standardise the 2×2 block and apply the rotation to the rest.
                Lanv2Full(ref h[(i - 1) * ldh + i - 1], ref h[(i - 1) * ldh + i], ref h[i * ldh + i - 1], ref h[i * ldh + i],
                    out wr[i - 1], out wi[i - 1], out wr[i], out wi[i], out var cs, out var sn);
                if (wantt)
                {
                    if (i2 > i) Rot(i2 - i, h + (i - 1) * ldh + i + 1, 1, h + i * ldh + i + 1, 1, cs, sn);
                    Rot(i - i1 - 1, h + i1 * ldh + i - 1, ldh, h + i1 * ldh + i, ldh, cs, sn);
                }
                if (wantz) Rot(nz, z + iloz * ldz + i - 1, ldz, z + iloz * ldz + i, ldz, cs, sn);
            }
            kdefl = 0;
            i = l - 1;
        }
    }

    /// <summary>
    /// LAPACK <c>dlaqr1</c>: a scalar multiple of the first column of (H − s1 I)(H − s2 I) for a
    /// 2×2 or 3×3 H (1-based at <paramref name="H"/>), with s1, s2 real or a conjugate pair.
    /// </summary>
    private static void Laqr1(int n, double* H, int ldh, double sr1, double si1, double sr2, double si2, double* v)
    {
        double A(int i, int j) => H[(i - 1) * ldh + (j - 1)];
        if (n != 2 && n != 3) return;
        if (n == 2)
        {
            var s = Math.Abs(A(1, 1) - sr2) + Math.Abs(si2) + Math.Abs(A(2, 1));
            if (s == 0.0)
            {
                v[0] = 0.0;
                v[1] = 0.0;
            }
            else
            {
                var h21s = A(2, 1) / s;
                v[0] = h21s * A(1, 2) + (A(1, 1) - sr1) * ((A(1, 1) - sr2) / s) - si1 * (si2 / s);
                v[1] = h21s * (A(1, 1) + A(2, 2) - sr1 - sr2);
            }
        }
        else
        {
            var s = Math.Abs(A(1, 1) - sr2) + Math.Abs(si2) + Math.Abs(A(2, 1)) + Math.Abs(A(3, 1));
            if (s == 0.0)
            {
                v[0] = 0.0;
                v[1] = 0.0;
                v[2] = 0.0;
            }
            else
            {
                var h21s = A(2, 1) / s;
                var h31s = A(3, 1) / s;
                v[0] = (A(1, 1) - sr1) * ((A(1, 1) - sr2) / s) - si1 * (si2 / s) + A(1, 2) * h21s + A(1, 3) * h31s;
                v[1] = h21s * (A(1, 1) + A(2, 2) - sr1 - sr2) + A(2, 3) * h31s;
                v[2] = h31s * (A(1, 1) + A(3, 3) - sr1 - sr2) + h21s * A(3, 2);
            }
        }
    }

    // ================================================================ dtrexc / dlaexc / dlasy2

    /// <summary>
    /// LAPACK <c>dtrexc</c>: reorders the real Schur form T (n×n at <paramref name="T"/>) so the
    /// block at <paramref name="ifst"/> moves to <paramref name="ilst"/> (1-based), updating Q.
    /// Returns 0, or 1 when a swap was refused as too ill-conditioned (ilst then holds where the
    /// block stopped).
    /// </summary>
    internal static int Trexc(bool wantq, int n, double* T, int ldt, double* Q, int ldq, ref int ifst, ref int ilst, double* work)
    {
        if (n <= 1) return 0;
        var t = T - ldt - 1;

        // Determine the first row of the specified block and whether it is 1×1 or 2×2.
        if (ifst > 1 && t[ifst * ldt + ifst - 1] != 0.0) ifst--;
        var nbf = 1;
        if (ifst < n && t[(ifst + 1) * ldt + ifst] != 0.0) nbf = 2;

        if (ilst > 1 && t[ilst * ldt + ilst - 1] != 0.0) ilst--;
        var nbl = 1;
        if (ilst < n && t[(ilst + 1) * ldt + ilst] != 0.0) nbl = 2;

        if (ifst == ilst) return 0;

        int here, nbnext, info;
        if (ifst < ilst)
        {
            // Move the block down.
            if (nbf == 2 && nbl == 1) ilst--;
            if (nbf == 1 && nbl == 2) ilst++;
            here = ifst;
            do
            {
                if (nbf == 1 || nbf == 2)
                {
                    // Current block either 1×1 or 2×2.
                    nbnext = 1;
                    if (here + nbf + 1 <= n && t[(here + nbf + 1) * ldt + here + nbf] != 0.0) nbnext = 2;
                    info = Laexc(wantq, n, T, ldt, Q, ldq, here, nbf, nbnext, work);
                    if (info != 0)
                    {
                        ilst = here;
                        return info;
                    }
                    here += nbnext;
                    // Test if a 2×2 block breaks into two 1×1 blocks.
                    if (nbf == 2 && t[(here + 1) * ldt + here] == 0.0) nbf = 3;
                }
                else
                {
                    // Current block consists of two 1×1 blocks each of which must be swapped
                    // individually.
                    nbnext = 1;
                    if (here + 3 <= n && t[(here + 3) * ldt + here + 2] != 0.0) nbnext = 2;
                    info = Laexc(wantq, n, T, ldt, Q, ldq, here + 1, 1, nbnext, work);
                    if (info != 0)
                    {
                        ilst = here;
                        return info;
                    }
                    if (nbnext == 1)
                    {
                        // Swap two 1×1 blocks, no problems possible.
                        Laexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work);
                        here++;
                    }
                    else
                    {
                        // Recompute NBNEXT in case the 2×2 split.
                        if (t[(here + 2) * ldt + here + 1] == 0.0) nbnext = 1;
                        if (nbnext == 2)
                        {
                            info = Laexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work);
                            if (info != 0)
                            {
                                ilst = here;
                                return info;
                            }
                            here += 2;
                        }
                        else
                        {
                            Laexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work);
                            Laexc(wantq, n, T, ldt, Q, ldq, here + 1, 1, 1, work);
                            here += 2;
                        }
                    }
                }
            } while (here < ilst);
        }
        else
        {
            // Move the block up.
            here = ifst;
            do
            {
                if (nbf == 1 || nbf == 2)
                {
                    nbnext = 1;
                    if (here >= 3 && t[(here - 1) * ldt + here - 2] != 0.0) nbnext = 2;
                    info = Laexc(wantq, n, T, ldt, Q, ldq, here - nbnext, nbnext, nbf, work);
                    if (info != 0)
                    {
                        ilst = here;
                        return info;
                    }
                    here -= nbnext;
                    if (nbf == 2 && t[(here + 1) * ldt + here] == 0.0) nbf = 3;
                }
                else
                {
                    nbnext = 1;
                    if (here >= 3 && t[(here - 1) * ldt + here - 2] != 0.0) nbnext = 2;
                    info = Laexc(wantq, n, T, ldt, Q, ldq, here - nbnext, nbnext, 1, work);
                    if (info != 0)
                    {
                        ilst = here;
                        return info;
                    }
                    if (nbnext == 1)
                    {
                        Laexc(wantq, n, T, ldt, Q, ldq, here, nbnext, 1, work);
                        here--;
                    }
                    else
                    {
                        if (t[here * ldt + here - 1] == 0.0) nbnext = 1;
                        if (nbnext == 2)
                        {
                            info = Laexc(wantq, n, T, ldt, Q, ldq, here - 1, 2, 1, work);
                            if (info != 0)
                            {
                                ilst = here;
                                return info;
                            }
                            here -= 2;
                        }
                        else
                        {
                            Laexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work);
                            Laexc(wantq, n, T, ldt, Q, ldq, here - 1, 1, 1, work);
                            here -= 2;
                        }
                    }
                }
            } while (here > ilst);
        }
        ilst = here;
        return 0;
    }

    /// <summary>
    /// LAPACK <c>dlaexc</c>: swaps the adjacent diagonal blocks T11 (n1×n1, at j1) and T22
    /// (n2×n2) of a real Schur form by an orthogonal similarity, updating Q. Returns 1 when the
    /// swap would perturb T too much (the blocks are too close) and leaves T unchanged.
    /// </summary>
    private static int Laexc(bool wantq, int n, double* T, int ldt, double* Q, int ldq, int j1, int n1, int n2, double* work)
    {
        if (n == 0 || n1 == 0 || n2 == 0) return 0;
        if (j1 + n1 > n) return 0;
        var t = T - ldt - 1;
        var q = Q - ldq - 1;
        var j2 = j1 + 1;
        var j3 = j1 + 2;
        var j4 = j1 + 3;

        double* T_(int i, int j) => t + (long)i * ldt + j;
        double* Q_(int i, int j) => q + (long)i * ldq + j;

        if (n1 == 1 && n2 == 1)
        {
            // Swap two 1×1 blocks.
            var t11 = *T_(j1, j1);
            var t22 = *T_(j2, j2);
            Lartg(*T_(j1, j2), t22 - t11, out var cs, out var sn, out _);
            if (j3 <= n) Rot(n - j1 - 1, T_(j1, j3), 1, T_(j2, j3), 1, cs, sn);
            Rot(j1 - 1, T_(1, j1), ldt, T_(1, j2), ldt, cs, sn);
            *T_(j1, j1) = t22;
            *T_(j2, j2) = t11;
            if (wantq) Rot(n, Q_(1, j1), ldq, Q_(1, j2), ldq, cs, sn);
            return 0;
        }

        // Swapping involves at least one 2×2 block: copy the diagonal block of order n1 + n2 to D.
        var nd = n1 + n2;
        var dbuf = stackalloc double[16];
        var d = dbuf - 4 - 1; // D(i, j) = d[i * 4 + j], ldd = 4
        var dnorm = 0.0;
        for (var i = 1; i <= nd; i++)
            for (var j = 1; j <= nd; j++)
            {
                d[i * 4 + j] = *T_(j1 + i - 1, j1 + j - 1);
                dnorm = Math.Max(dnorm, Math.Abs(d[i * 4 + j]));
            }
        var eps = Ulp;
        var smlnum = SafeMin / eps;
        var thresh = Math.Max(10.0 * eps * dnorm, smlnum);

        // Solve T11 X - X T22 = scale * T12 for X.
        var xbuf = stackalloc double[4];
        Lasy2(-1, n1, n2, dbuf, 4, dbuf + n1 * 4 + n1, 4, dbuf + n1, 4, out var scale, xbuf, 2, out _);
        double X(int i, int j) => xbuf[(i - 1) * 2 + (j - 1)];

        var uu = stackalloc double[3];
        var u1 = stackalloc double[3];
        var u2 = stackalloc double[3];
        var k = n1 + n1 + n2 - 3;
        switch (k)
        {
            case 1:
            {
                // N1 = 1, N2 = 2: generate elementary reflector H so that ( scale, X11, X12 ) H = ( 0, 0, * ).
                uu[0] = scale;
                uu[1] = X(1, 1);
                uu[2] = X(1, 2);
                Larfg(3, ref uu[2], uu, 1, out var tau);
                uu[2] = 1.0;
                var t11 = *T_(j1, j1);
                // Perform swap tentatively on the diagonal block.
                Larfx(true, 3, 3, uu, tau, dbuf, 4);
                Larfx(false, 3, 3, uu, tau, dbuf, 4);
                // Test whether to reject the swap.
                if (Math.Max(Math.Max(Math.Abs(d[3 * 4 + 1]), Math.Abs(d[3 * 4 + 2])), Math.Abs(d[3 * 4 + 3] - t11)) > thresh) return 1;
                // Accept swap: apply transformation to the entire matrix T.
                Larfx(true, 3, n - j1 + 1, uu, tau, T_(j1, j1), ldt);
                Larfx(false, j2, 3, uu, tau, T_(1, j1), ldt);
                *T_(j3, j1) = 0.0;
                *T_(j3, j2) = 0.0;
                *T_(j3, j3) = t11;
                if (wantq) Larfx(false, n, 3, uu, tau, Q_(1, j1), ldq);
                break;
            }
            case 2:
            {
                // N1 = 2, N2 = 1: generate H so that H ( -X11, -X21, scale )ᵀ = ( *, 0, 0 )ᵀ.
                uu[0] = -X(1, 1);
                uu[1] = -X(2, 1);
                uu[2] = scale;
                Larfg(3, ref uu[0], uu + 1, 1, out var tau);
                uu[0] = 1.0;
                var t33 = *T_(j3, j3);
                Larfx(true, 3, 3, uu, tau, dbuf, 4);
                Larfx(false, 3, 3, uu, tau, dbuf, 4);
                if (Math.Max(Math.Max(Math.Abs(d[2 * 4 + 1]), Math.Abs(d[3 * 4 + 1])), Math.Abs(d[1 * 4 + 1] - t33)) > thresh) return 1;
                Larfx(false, j3, 3, uu, tau, T_(1, j1), ldt);
                Larfx(true, 3, n - j1, uu, tau, T_(j1, j2), ldt);
                *T_(j1, j1) = t33;
                *T_(j2, j1) = 0.0;
                *T_(j3, j1) = 0.0;
                if (wantq) Larfx(false, n, 3, uu, tau, Q_(1, j1), ldq);
                break;
            }
            case 3:
            {
                // N1 = 2, N2 = 2: two reflectors.
                u1[0] = -X(1, 1);
                u1[1] = -X(2, 1);
                u1[2] = scale;
                Larfg(3, ref u1[0], u1 + 1, 1, out var tau1);
                u1[0] = 1.0;
                var temp = -tau1 * (X(1, 2) + u1[1] * X(2, 2));
                u2[0] = -temp * u1[1] - X(2, 2);
                u2[1] = -temp * u1[2];
                u2[2] = scale;
                Larfg(3, ref u2[0], u2 + 1, 1, out var tau2);
                u2[0] = 1.0;
                Larfx(true, 3, 4, u1, tau1, dbuf, 4);
                Larfx(false, 4, 3, u1, tau1, dbuf, 4);
                Larfx(true, 3, 4, u2, tau2, dbuf + 4, 4);
                Larfx(false, 4, 3, u2, tau2, dbuf + 1, 4);
                if (Math.Max(Math.Max(Math.Abs(d[3 * 4 + 1]), Math.Abs(d[3 * 4 + 2])), Math.Max(Math.Abs(d[4 * 4 + 1]), Math.Abs(d[4 * 4 + 2]))) > thresh) return 1;
                Larfx(true, 3, n - j1 + 1, u1, tau1, T_(j1, j1), ldt);
                Larfx(false, j4, 3, u1, tau1, T_(1, j1), ldt);
                Larfx(true, 3, n - j1 + 1, u2, tau2, T_(j2, j1), ldt);
                Larfx(false, j4, 3, u2, tau2, T_(1, j2), ldt);
                *T_(j3, j1) = 0.0;
                *T_(j3, j2) = 0.0;
                *T_(j4, j1) = 0.0;
                *T_(j4, j2) = 0.0;
                if (wantq)
                {
                    Larfx(false, n, 3, u1, tau1, Q_(1, j1), ldq);
                    Larfx(false, n, 3, u2, tau2, Q_(1, j2), ldq);
                }
                break;
            }
        }

        if (n2 == 2)
        {
            // Standardise the new 2×2 block T11.
            Lanv2Full(ref *T_(j1, j1), ref *T_(j1, j2), ref *T_(j2, j1), ref *T_(j2, j2), out _, out _, out _, out _, out var cs, out var sn);
            Rot(n - j1 - 1, T_(j1, j1 + 2), 1, T_(j2, j1 + 2), 1, cs, sn);
            Rot(j1 - 1, T_(1, j1), ldt, T_(1, j2), ldt, cs, sn);
            if (wantq) Rot(n, Q_(1, j1), ldq, Q_(1, j2), ldq, cs, sn);
        }
        if (n1 == 2)
        {
            // Standardise the new 2×2 block T22.
            j3 = j1 + n2;
            j4 = j3 + 1;
            Lanv2Full(ref *T_(j3, j3), ref *T_(j3, j4), ref *T_(j4, j3), ref *T_(j4, j4), out _, out _, out _, out _, out var cs, out var sn);
            if (j3 + 2 <= n) Rot(n - j3 - 1, T_(j3, j3 + 2), 1, T_(j4, j3 + 2), 1, cs, sn);
            Rot(j3 - 1, T_(1, j3), ldt, T_(1, j4), ldt, cs, sn);
            if (wantq) Rot(n, Q_(1, j3), ldq, Q_(1, j4), ldq, cs, sn);
        }
        _ = work;
        return 0;
    }

    /// <summary>
    /// LAPACK <c>dlasy2</c> without transposes: solves <c>TL X + isgn X TR = scale B</c> for the
    /// n1×n2 X (n1, n2 ∈ {1, 2}), all operands row-major 1-based at the given pointers.
    /// Returns 1 when TL and −isgn·TR have close eigenvalues (perturbed to proceed).
    /// </summary>
    private static int Lasy2(int isgn, int n1, int n2, double* TL, int ldtl, double* TR, int ldtr, double* B, int ldb,
        out double scale, double* X, int ldx, out double xnorm)
    {
        double tl(int i, int j) => TL[(i - 1) * ldtl + (j - 1)];
        double tr(int i, int j) => TR[(i - 1) * ldtr + (j - 1)];
        double b(int i, int j) => B[(i - 1) * ldb + (j - 1)];
        ref double x(int i, int j) => ref X[(i - 1) * ldx + (j - 1)];

        var info = 0;
        scale = 1.0;
        xnorm = 0.0;
        if (n1 == 0 || n2 == 0) return 0;

        var eps = Ulp;
        var smlnum = SafeMin / eps;
        double sgn = isgn;
        var k = n1 + n1 + n2 - 2;

        if (k == 1)
        {
            // 1×1: TL11 X11 + SGN X11 TR11 = B11.
            var tau1 = tl(1, 1) + sgn * tr(1, 1);
            var bet = Math.Abs(tau1);
            if (bet <= smlnum)
            {
                tau1 = smlnum;
                bet = smlnum;
                info = 1;
            }
            scale = 1.0;
            var gam = Math.Abs(b(1, 1));
            if (smlnum * gam > bet) scale = 1.0 / gam;
            x(1, 1) = b(1, 1) * scale / tau1;
            xnorm = Math.Abs(x(1, 1));
            return info;
        }

        if (k == 2 || k == 3)
        {
            // 1×2 or 2×1: a 2×2 system with complete pivoting.
            Span<double> tmp = stackalloc double[4];
            Span<double> btmp = stackalloc double[2];
            double smin;
            if (k == 2)
            {
                // 1×2: TL11 [X11 X12] + ISGN [X11 X12] op[TR11 TR12; TR21 TR22] = [B11 B12].
                smin = Math.Max(eps * Max(Math.Abs(tl(1, 1)), Math.Abs(tr(1, 1)), Math.Abs(tr(1, 2)), Math.Abs(tr(2, 1)), Math.Abs(tr(2, 2))), smlnum);
                tmp[0] = tl(1, 1) + sgn * tr(1, 1);
                tmp[3] = tl(1, 1) + sgn * tr(2, 2);
                tmp[1] = sgn * tr(1, 2);
                tmp[2] = sgn * tr(2, 1);
                btmp[0] = b(1, 1);
                btmp[1] = b(1, 2);
            }
            else
            {
                // 2×1: op[TL11 TL12; TL21 TL22] [X11; X21] + ISGN [X11; X21] TR11 = [B11; B21].
                smin = Math.Max(eps * Max(Math.Abs(tr(1, 1)), Math.Abs(tl(1, 1)), Math.Abs(tl(1, 2)), Math.Abs(tl(2, 1)), Math.Abs(tl(2, 2))), smlnum);
                tmp[0] = tl(1, 1) + sgn * tr(1, 1);
                tmp[3] = tl(2, 2) + sgn * tr(1, 1);
                tmp[1] = tl(2, 1);
                tmp[2] = tl(1, 2);
                btmp[0] = b(1, 1);
                btmp[1] = b(2, 1);
            }

            // Solve the 2×2 system with complete pivoting (column-major TMP as in LAPACK).
            ReadOnlySpan<int> locu12 = [3, 4, 1, 2];
            ReadOnlySpan<int> locl21 = [2, 1, 4, 3];
            ReadOnlySpan<int> locu22 = [4, 3, 2, 1];
            ReadOnlySpan<bool> xswpiv = [false, false, true, true];
            ReadOnlySpan<bool> bswpiv = [false, true, false, true];
            var ipiv = 0;
            for (var q = 1; q < 4; q++)
                if (Math.Abs(tmp[q]) > Math.Abs(tmp[ipiv])) ipiv = q;
            var u11 = tmp[ipiv];
            if (Math.Abs(u11) <= smin)
            {
                info = 1;
                u11 = smin;
            }
            var u12 = tmp[locu12[ipiv] - 1];
            var l21 = tmp[locl21[ipiv] - 1] / u11;
            var u22 = tmp[locu22[ipiv] - 1] - u12 * l21;
            var xswap = xswpiv[ipiv];
            var bswap = bswpiv[ipiv];
            if (Math.Abs(u22) <= smin)
            {
                info = 1;
                u22 = smin;
            }
            if (bswap)
            {
                var temp = btmp[1];
                btmp[1] = btmp[0] - l21 * temp;
                btmp[0] = temp;
            }
            else
            {
                btmp[1] -= l21 * btmp[0];
            }
            scale = 1.0;
            if (2.0 * smlnum * Math.Abs(btmp[1]) > Math.Abs(u22) || 2.0 * smlnum * Math.Abs(btmp[0]) > Math.Abs(u11))
            {
                scale = 0.5 / Math.Max(Math.Abs(btmp[0]), Math.Abs(btmp[1]));
                btmp[0] *= scale;
                btmp[1] *= scale;
            }
            var x22 = btmp[1] / u22;
            var x21 = btmp[0] / u11 - u12 / u11 * x22;
            if (xswap) (x21, x22) = (x22, x21);
            x(1, 1) = x21;
            if (n1 == 1)
            {
                x(1, 2) = x22;
                xnorm = Math.Abs(x(1, 1)) + Math.Abs(x(1, 2));
            }
            else
            {
                x(2, 1) = x22;
                xnorm = Math.Max(Math.Abs(x(1, 1)), Math.Abs(x(2, 1)));
            }
            return info;
        }

        // 2×2: a 4×4 system with complete pivoting (T16 column-major in LAPACK; indices kept).
        {
            var smin = Max(Math.Abs(tr(1, 1)), Math.Abs(tr(1, 2)), Math.Abs(tr(2, 1)), Math.Abs(tr(2, 2)));
            smin = Math.Max(smin, Max(Math.Abs(tl(1, 1)), Math.Abs(tl(1, 2)), Math.Abs(tl(2, 1)), Math.Abs(tl(2, 2))));
            smin = Math.Max(eps * smin, smlnum);
            var t16 = stackalloc double[16];
            double* T16(int i, int j) => t16 + (i - 1) * 4 + (j - 1);
            for (var q = 0; q < 16; q++) t16[q] = 0.0;
            *T16(1, 1) = tl(1, 1) + sgn * tr(1, 1);
            *T16(2, 2) = tl(2, 2) + sgn * tr(1, 1);
            *T16(3, 3) = tl(1, 1) + sgn * tr(2, 2);
            *T16(4, 4) = tl(2, 2) + sgn * tr(2, 2);
            *T16(1, 2) = tl(1, 2);
            *T16(2, 1) = tl(2, 1);
            *T16(3, 4) = tl(1, 2);
            *T16(4, 3) = tl(2, 1);
            *T16(1, 3) = sgn * tr(2, 1);
            *T16(2, 4) = sgn * tr(2, 1);
            *T16(3, 1) = sgn * tr(1, 2);
            *T16(4, 2) = sgn * tr(1, 2);
            Span<double> btmp = [b(1, 1), b(2, 1), b(1, 2), b(2, 2)];
            Span<double> tmp = stackalloc double[4];
            Span<int> jpiv = stackalloc int[4];

            // Gaussian elimination with complete pivoting.
            for (var i = 1; i <= 3; i++)
            {
                var xmax = 0.0;
                int ipsv = i, jpsv = i;
                for (var ip = i; ip <= 4; ip++)
                    for (var jp = i; jp <= 4; jp++)
                        if (Math.Abs(*T16(ip, jp)) >= xmax)
                        {
                            xmax = Math.Abs(*T16(ip, jp));
                            ipsv = ip;
                            jpsv = jp;
                        }
                if (ipsv != i)
                {
                    for (var c = 1; c <= 4; c++) (*T16(ipsv, c), *T16(i, c)) = (*T16(i, c), *T16(ipsv, c));
                    (btmp[i - 1], btmp[ipsv - 1]) = (btmp[ipsv - 1], btmp[i - 1]);
                }
                if (jpsv != i)
                    for (var r = 1; r <= 4; r++) (*T16(r, jpsv), *T16(r, i)) = (*T16(r, i), *T16(r, jpsv));
                jpiv[i - 1] = jpsv;
                if (Math.Abs(*T16(i, i)) < smin)
                {
                    info = 1;
                    *T16(i, i) = smin;
                }
                for (var j = i + 1; j <= 4; j++)
                {
                    *T16(j, i) /= *T16(i, i);
                    btmp[j - 1] -= *T16(j, i) * btmp[i - 1];
                    for (var kk = i + 1; kk <= 4; kk++) *T16(j, kk) -= *T16(j, i) * *T16(i, kk);
                }
            }
            if (Math.Abs(*T16(4, 4)) < smin)
            {
                info = 1;
                *T16(4, 4) = smin;
            }
            scale = 1.0;
            if (8.0 * smlnum * Math.Abs(btmp[0]) > Math.Abs(*T16(1, 1)) ||
                8.0 * smlnum * Math.Abs(btmp[1]) > Math.Abs(*T16(2, 2)) ||
                8.0 * smlnum * Math.Abs(btmp[2]) > Math.Abs(*T16(3, 3)) ||
                8.0 * smlnum * Math.Abs(btmp[3]) > Math.Abs(*T16(4, 4)))
            {
                scale = 1.0 / 8.0 / Max(Math.Abs(btmp[0]), Math.Abs(btmp[1]), Math.Abs(btmp[2]), Math.Abs(btmp[3]));
                for (var q = 0; q < 4; q++) btmp[q] *= scale;
            }
            for (var i = 1; i <= 4; i++)
            {
                var kk = 5 - i;
                var temp = 1.0 / *T16(kk, kk);
                tmp[kk - 1] = btmp[kk - 1] * temp;
                for (var j = kk + 1; j <= 4; j++) tmp[kk - 1] -= temp * *T16(kk, j) * tmp[j - 1];
            }
            for (var i = 1; i <= 3; i++)
            {
                var p = jpiv[4 - i - 1];
                if (p != 4 - i) (tmp[4 - i - 1], tmp[p - 1]) = (tmp[p - 1], tmp[4 - i - 1]);
            }
            x(1, 1) = tmp[0];
            x(2, 1) = tmp[1];
            x(1, 2) = tmp[2];
            x(2, 2) = tmp[3];
            xnorm = Math.Max(Math.Abs(tmp[0]) + Math.Abs(tmp[2]), Math.Abs(tmp[1]) + Math.Abs(tmp[3]));
            return info;
        }
    }

    private static double Max(double a, double b, double c, double d) => Math.Max(Math.Max(a, b), Math.Max(c, d));

    private static double Max(double a, double b, double c, double d, double e) => Math.Max(Max(a, b, c, d), e);

    // ================================================================ BLAS-like helpers

    /// <summary>
    /// <c>C ← alpha·op(A)·op(B) + beta·C</c> on row-major operands (leading dimension = row stride),
    /// through GraviNum's packed kernel, which reads either orientation through its strides.
    /// </summary>
    internal static void Gemm(bool ta, bool tb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
    {
        if (m <= 0 || n <= 0) return;
        long rsA = ta ? 1 : lda, csA = ta ? lda : 1;
        long rsB = tb ? 1 : ldb, csB = tb ? ldb : 1;
        PackedMatMul.Gemm(m, n, k, alpha, a, rsA, csA, b, rsB, csB, beta, c, ldc);
    }

    private static void CopyBlock(int rows, int cols, double* src, int lds, double* dst, int ldd)
    {
        for (var r = 0; r < rows; r++)
            new Span<double>(src + (long)r * lds, cols).CopyTo(new Span<double>(dst + (long)r * ldd, cols));
    }

    /// <summary>BLAS <c>drot</c>: <c>x ← c x + s y, y ← c y − s x</c>.</summary>
    private static void Rot(int count, double* x, int incx, double* y, int incy, double c, double s)
    {
        for (var i = 0; i < count; i++)
        {
            var px = x + (long)i * incx;
            var py = y + (long)i * incy;
            var tx = *px;
            var ty = *py;
            *px = c * tx + s * ty;
            *py = c * ty - s * tx;
        }
    }

    /// <summary>
    /// LAPACK <c>dlarfg</c>: the elementary reflector H with <c>H (alpha, x)ᵀ = (beta, 0)ᵀ</c>,
    /// <c>H = I − τ (1, v)(1, v)ᵀ</c>; alpha is overwritten by beta and x by v.
    /// </summary>
    internal static void Larfg(int n, ref double alpha, double* x, int incx, out double tau)
    {
        if (n <= 1)
        {
            tau = 0.0;
            return;
        }
        var xnorm = Nrm2(n - 1, x, incx);
        if (xnorm == 0.0)
        {
            tau = 0.0;
            return;
        }
        var beta = -Math.CopySign(Lapy2(alpha, xnorm), alpha);
        const double safmin = SafeMin / (Ulp / 2);
        var knt = 0;
        if (Math.Abs(beta) < safmin)
        {
            // xnorm, beta may be inaccurate; scale x and recompute them.
            const double rsafmn = 1.0 / safmin;
            do
            {
                knt++;
                for (var i = 0; i < n - 1; i++) x[(long)i * incx] *= rsafmn;
                beta *= rsafmn;
                alpha *= rsafmn;
            } while (Math.Abs(beta) < safmin && knt < 20);
            xnorm = Nrm2(n - 1, x, incx);
            beta = -Math.CopySign(Lapy2(alpha, xnorm), alpha);
        }
        tau = (beta - alpha) / beta;
        var scal = 1.0 / (alpha - beta);
        for (var i = 0; i < n - 1; i++) x[(long)i * incx] *= scal;
        for (var j = 0; j < knt; j++) beta *= safmin;
        alpha = beta;
    }

    private static double Nrm2(int n, double* x, long incx)
    {
        var ss = 0.0;
        for (var i = 0; i < n; i++)
        {
            var v = x[i * incx];
            ss += v * v;
        }
        // Fast path unless the sum of squares is near under- or overflow.
        if (ss > 1e-280 && ss < 1e280) return Math.Sqrt(ss);
        var scale = 0.0;
        for (var i = 0; i < n; i++) scale = Math.Max(scale, Math.Abs(x[i * incx]));
        if (scale == 0.0 || double.IsInfinity(scale)) return scale;
        ss = 0.0;
        for (var i = 0; i < n; i++)
        {
            var v = x[i * incx] / scale;
            ss += v * v;
        }
        return scale * Math.Sqrt(ss);
    }

    private static double Lapy2(double x, double y)
    {
        var ax = Math.Abs(x);
        var ay = Math.Abs(y);
        var w = Math.Max(ax, ay);
        var z = Math.Min(ax, ay);
        if (z == 0.0 || double.IsNaN(w) || w > double.MaxValue) return w;
        var r = z / w;
        return w * Math.Sqrt(1.0 + r * r);
    }

    /// <summary>
    /// LAPACK <c>dlartg</c> (3.10): plane rotation with <c>c f + s g = r</c>, <c>−s f + c g = 0</c>,
    /// r carrying the sign of f.
    /// </summary>
    private static void Lartg(double f, double g, out double c, out double s, out double r)
    {
        if (g == 0.0)
        {
            c = 1.0;
            s = 0.0;
            r = f;
        }
        else if (f == 0.0)
        {
            c = 0.0;
            s = Math.CopySign(1.0, g);
            r = Math.Abs(g);
        }
        else
        {
            var d = Lapy2(f, g);
            c = Math.Abs(f) / d;
            r = Math.CopySign(d, f);
            s = g / r;
        }
    }

    /// <summary>
    /// <c>C ← (I − τ v vᵀ) C</c> for the m×ncols block C, with v(1) = 1 implicit (LAPACK
    /// <c>dlarf1f</c>, left). <paramref name="work"/> needs ncols entries.
    /// </summary>
    internal static void ApplyLeft(int m, int ncols, double* v, int incv, double tau, double* c, int ldc, double* work)
    {
        if (tau == 0.0 || m <= 0 || ncols <= 0) return;
        new Span<double>(c, ncols).CopyTo(new Span<double>(work, ncols));
        for (var i = 1; i < m; i++)
        {
            var vi = v[(long)i * incv];
            if (vi != 0.0) LinalgBlocked.Axpy(vi, c + (long)i * ldc, work, ncols);
        }
        LinalgBlocked.Axpy(-tau, work, c, ncols);
        for (var i = 1; i < m; i++)
        {
            var vi = v[(long)i * incv];
            if (vi != 0.0) LinalgBlocked.Axpy(-tau * vi, work, c + (long)i * ldc, ncols);
        }
    }

    /// <summary>
    /// <c>C ← C (I − τ v vᵀ)</c> for the mrows×ncols block C, with v(1) = 1 implicit (LAPACK
    /// <c>dlarf1f</c>, right). <paramref name="work"/> needs ncols entries.
    /// </summary>
    internal static void ApplyRight(int mrows, int ncols, double* v, int incv, double tau, double* c, int ldc, double* work)
    {
        if (tau == 0.0 || mrows <= 0 || ncols <= 0) return;
        work[0] = 1.0;
        for (var j = 1; j < ncols; j++) work[j] = v[(long)j * incv];
        for (var r = 0; r < mrows; r++)
        {
            var row = c + (long)r * ldc;
            var s = 0.0;
            for (var j = 0; j < ncols; j++) s += row[j] * work[j];
            if (s != 0.0) LinalgBlocked.Axpy(-tau * s, work, row, ncols);
        }
    }

    /// <summary>
    /// LAPACK <c>dlarfx</c> with an explicit vector v (no implicit unit): left
    /// <c>C ← (I − τ v vᵀ) C</c> (m×ncols) or right <c>C ← C (I − τ v vᵀ)</c>.
    /// </summary>
    private static void Larfx(bool left, int m, int ncols, double* v, double tau, double* c, int ldc)
    {
        if (tau == 0.0) return;
        if (left)
        {
            for (var j = 0; j < ncols; j++)
            {
                var s = 0.0;
                for (var i = 0; i < m; i++) s += v[i] * c[(long)i * ldc + j];
                s *= tau;
                for (var i = 0; i < m; i++) c[(long)i * ldc + j] -= s * v[i];
            }
        }
        else
        {
            for (var i = 0; i < m; i++)
            {
                var row = c + (long)i * ldc;
                var s = 0.0;
                for (var j = 0; j < ncols; j++) s += row[j] * v[j];
                s *= tau;
                for (var j = 0; j < ncols; j++) row[j] -= s * v[j];
            }
        }
    }

    /// <summary>
    /// LAPACK <c>dlanv2</c> with its eigenvalue outputs: the standardisation is
    /// <see cref="LinalgSchurKernel.Lanv2"/>, shared with the double-shift kernel.
    /// </summary>
    internal static void Lanv2Full(ref double a, ref double b, ref double c, ref double d,
        out double rt1r, out double rt1i, out double rt2r, out double rt2i, out double cs, out double sn)
    {
        LinalgSchurKernel.Lanv2(ref a, ref b, ref c, ref d, out cs, out sn);
        rt1r = a;
        rt2r = d;
        if (c == 0.0)
        {
            rt1i = 0.0;
            rt2i = 0.0;
        }
        else
        {
            rt1i = Math.Sqrt(Math.Abs(b)) * Math.Sqrt(Math.Abs(c));
            rt2i = -rt1i;
        }
    }
}
