namespace Gravicode.Science.SciNet;

/// <summary>
/// A transcription of the QUADPACK routines scipy.integrate.quad calls (Piessens, de Doncker-Kapenga,
/// Überhuber and Kahaner, 1983): dqagse, dqagie, dqagpe, dqawoe, dqawfe, dqawse, dqawce and their
/// kernels. Arrays are 1-based (index 0 unused) so that the control flow can follow the Fortran
/// line by line; the error-flag bookkeeping is where most of QUADPACK's behaviour lives, and a
/// "cleaned up" version would return different error estimates.
/// One deliberate difference: the Chebyshev coefficients of the 25-point Clenshaw-Curtis rules
/// are computed by a direct cosine sum instead of dqcheb's hand-unrolled FFT, which changes
/// nothing but rounding.
/// </summary>
internal static class Quadpack
{
    internal const double Epmach = 2.220446049250313e-16;
    internal const double Uflow = 2.2250738585072014e-308;
    internal const double Oflow = double.MaxValue;

    internal sealed class Work
    {
        internal readonly double[] Alist, Blist, Rlist, Elist;
        internal readonly int[] Iord, Level;
        internal int Last;

        internal Work(int limit)
        {
            var n = Math.Max(limit, 2) + 2;
            Alist = new double[n]; Blist = new double[n]; Rlist = new double[n]; Elist = new double[n];
            Iord = new int[n]; Level = new int[n];
        }
    }

    internal struct Output
    {
        internal double Result, AbsErr;
        internal int Neval, Ier;
    }

    // ---------------------------------------------------------------- kernels

    /// <summary>dqk21: 21-point Gauss-Kronrod on [a, b].</summary>
    internal static void Qk21(Func<double, double> f, double a, double b,
        out double result, out double abserr, out double resabs, out double resasc)
    {
        var xgk = IntegrateRules.Xgk21; var wgk = IntegrateRules.Wgk21; var wg = IntegrateRules.Wg10;
        Span<double> fv1 = stackalloc double[10];
        Span<double> fv2 = stackalloc double[10];
        var centr = 0.5 * (a + b);
        var hlgth = 0.5 * (b - a);
        var dhlgth = Math.Abs(hlgth);
        var resg = 0.0;
        var fc = f(centr);
        var resk = wgk[10] * fc;
        resabs = Math.Abs(resk);
        for (var j = 0; j < 5; j++)
        {
            var jtw = 2 * j + 1;
            var absc = hlgth * xgk[jtw];
            var f1 = f(centr - absc);
            var f2 = f(centr + absc);
            fv1[jtw] = f1; fv2[jtw] = f2;
            var fsum = f1 + f2;
            resg += wg[j] * fsum;
            resk += wgk[jtw] * fsum;
            resabs += wgk[jtw] * (Math.Abs(f1) + Math.Abs(f2));
        }
        for (var j = 0; j < 5; j++)
        {
            var jtwm1 = 2 * j;
            var absc = hlgth * xgk[jtwm1];
            var f1 = f(centr - absc);
            var f2 = f(centr + absc);
            fv1[jtwm1] = f1; fv2[jtwm1] = f2;
            var fsum = f1 + f2;
            resk += wgk[jtwm1] * fsum;
            resabs += wgk[jtwm1] * (Math.Abs(f1) + Math.Abs(f2));
        }
        var reskh = resk * 0.5;
        resasc = wgk[10] * Math.Abs(fc - reskh);
        for (var j = 0; j < 10; j++) resasc += wgk[j] * (Math.Abs(fv1[j] - reskh) + Math.Abs(fv2[j] - reskh));
        result = resk * hlgth;
        resabs *= dhlgth;
        resasc *= dhlgth;
        abserr = Math.Abs((resk - resg) * hlgth);
        FinishError(ref abserr, resabs, resasc);
    }

    private static void FinishError(ref double abserr, double resabs, double resasc)
    {
        if (resasc != 0.0 && abserr != 0.0) abserr = resasc * Math.Min(1.0, Math.Pow(200.0 * abserr / resasc, 1.5));
        if (resabs > Uflow / (50.0 * Epmach)) abserr = Math.Max(Epmach * 50.0 * resabs, abserr);
    }

    /// <summary>dqk15i: 15-point Kronrod on the (0,1] image of an infinite range.</summary>
    internal static void Qk15i(Func<double, double> f, double boun, int inf, double a, double b,
        out double result, out double abserr, out double resabs, out double resasc)
    {
        var xgk = IntegrateRules.Xgk15; var wgk = IntegrateRules.Wgk15;
        // wg of dqk15i has zeros at the Kronrod-only nodes.
        Span<double> wg = [0, IntegrateRules.Wg7[0], 0, IntegrateRules.Wg7[1], 0, IntegrateRules.Wg7[2], 0, IntegrateRules.Wg7[3]];
        Span<double> fv1 = stackalloc double[7];
        Span<double> fv2 = stackalloc double[7];
        double dinf = Math.Min(1, inf);
        var centr = 0.5 * (a + b);
        var hlgth = 0.5 * (b - a);
        var tabsc1 = boun + dinf * (1.0 - centr) / centr;
        var fval1 = f(tabsc1);
        if (inf == 2) fval1 += f(-tabsc1);
        var fc = fval1 / centr / centr;
        var resg = wg[7] * fc;
        var resk = wgk[7] * fc;
        resabs = Math.Abs(resk);
        for (var j = 0; j < 7; j++)
        {
            var absc = hlgth * xgk[j];
            var absc1 = centr - absc;
            var absc2 = centr + absc;
            var t1 = boun + dinf * (1.0 - absc1) / absc1;
            var t2 = boun + dinf * (1.0 - absc2) / absc2;
            var f1 = f(t1);
            var f2 = f(t2);
            if (inf == 2) { f1 += f(-t1); f2 += f(-t2); }
            f1 = f1 / absc1 / absc1;
            f2 = f2 / absc2 / absc2;
            fv1[j] = f1; fv2[j] = f2;
            var fsum = f1 + f2;
            resg += wg[j] * fsum;
            resk += wgk[j] * fsum;
            resabs += wgk[j] * (Math.Abs(f1) + Math.Abs(f2));
        }
        var reskh = resk * 0.5;
        resasc = wgk[7] * Math.Abs(fc - reskh);
        for (var j = 0; j < 7; j++) resasc += wgk[j] * (Math.Abs(fv1[j] - reskh) + Math.Abs(fv2[j] - reskh));
        result = resk * hlgth;
        resasc *= hlgth;
        resabs *= hlgth;
        abserr = Math.Abs((resk - resg) * hlgth);
        FinishError(ref abserr, resabs, resasc);
    }

    /// <summary>dqk15w: 15-point Kronrod for f(x) * w(x).</summary>
    internal static void Qk15w(Func<double, double> f, Func<double, double> w, double a, double b,
        out double result, out double abserr, out double resabs, out double resasc)
    {
        var xgk = IntegrateRules.Xgk15; var wgk = IntegrateRules.Wgk15; var wg = IntegrateRules.Wg7;
        Span<double> fv1 = stackalloc double[7];
        Span<double> fv2 = stackalloc double[7];
        var centr = 0.5 * (a + b);
        var hlgth = 0.5 * (b - a);
        var dhlgth = Math.Abs(hlgth);
        var fc = f(centr) * w(centr);
        var resg = wg[3] * fc;
        var resk = wgk[7] * fc;
        resabs = Math.Abs(resk);
        for (var j = 0; j < 3; j++)
        {
            var jtw = 2 * j + 1;
            var absc = hlgth * xgk[jtw];
            var x1 = centr - absc; var x2 = centr + absc;
            var f1 = f(x1) * w(x1);
            var f2 = f(x2) * w(x2);
            fv1[jtw] = f1; fv2[jtw] = f2;
            var fsum = f1 + f2;
            resg += wg[j] * fsum;
            resk += wgk[jtw] * fsum;
            resabs += wgk[jtw] * (Math.Abs(f1) + Math.Abs(f2));
        }
        for (var j = 0; j < 4; j++)
        {
            var jtwm1 = 2 * j;
            var absc = hlgth * xgk[jtwm1];
            var x1 = centr - absc; var x2 = centr + absc;
            var f1 = f(x1) * w(x1);
            var f2 = f(x2) * w(x2);
            fv1[jtwm1] = f1; fv2[jtwm1] = f2;
            var fsum = f1 + f2;
            resk += wgk[jtwm1] * fsum;
            resabs += wgk[jtwm1] * (Math.Abs(f1) + Math.Abs(f2));
        }
        var reskh = resk * 0.5;
        resasc = wgk[7] * Math.Abs(fc - reskh);
        for (var j = 0; j < 7; j++) resasc += wgk[j] * (Math.Abs(fv1[j] - reskh) + Math.Abs(fv2[j] - reskh));
        result = resk * hlgth;
        resabs *= dhlgth;
        resasc *= dhlgth;
        abserr = Math.Abs((resk - resg) * hlgth);
        FinishError(ref abserr, resabs, resasc);
    }

    /// <summary>
    /// Chebyshev coefficients of degree 12 and 24 from the 25 values fval[0..24] at
    /// cos(j pi / 24) (fval[0] and fval[24] already halved, as QUADPACK passes them).
    /// cheb24[k] is the coefficient of T_k with QUADPACK's end-term halving folded in.
    /// </summary>
    internal static void Cheb(ReadOnlySpan<double> fval, Span<double> cheb12, Span<double> cheb24)
    {
        for (var k = 0; k <= 24; k++)
        {
            var s = 0.0;
            for (var j = 0; j <= 24; j++) s += fval[j] * CosTable[(k * j) % 48];
            var c = s / 12.0;
            if (k == 0 || k == 24) c *= 0.5;
            cheb24[k] = c;
        }
        for (var k = 0; k <= 12; k++)
        {
            var s = 0.0;
            for (var j = 0; j <= 12; j++) s += fval[2 * j] * CosTable[(2 * k * j) % 48];
            var c = s / 6.0;
            if (k == 0 || k == 12) c *= 0.5;
            cheb12[k] = c;
        }
    }

    private static readonly double[] CosTable = BuildCos();

    private static double[] BuildCos()
    {
        var t = new double[48];
        for (var i = 0; i < 48; i++) t[i] = Math.Cos(i * Math.PI / 24.0);
        // Exact values where cos has them, so symmetric sums cancel cleanly.
        t[0] = 1; t[12] = 0; t[24] = -1; t[36] = 0; t[8] = 0.5; t[16] = -0.5; t[32] = -0.5; t[40] = 0.5;
        return t;
    }

    private static void FillCcValues(Func<double, double> f, double centr, double hlgth, Span<double> fval)
    {
        var x = IntegrateRules.Xcc;
        fval[0] = 0.5 * f(hlgth + centr);
        fval[12] = f(centr);
        fval[24] = 0.5 * f(centr - hlgth);
        for (var i = 1; i < 12; i++)
        {
            fval[i] = f(hlgth * x[i - 1] + centr);
            fval[24 - i] = f(centr - hlgth * x[i - 1]);
        }
    }

    // ---------------------------------------------------------------- dqpsrt / dqelg

    internal static void Qpsrt(int limit, int last, ref int maxerr, out double ermax, double[] elist, int[] iord, ref int nrmax)
    {
        if (last <= 2)
        {
            iord[1] = 1; iord[2] = 2;
            goto L90;
        }
        var errmax = elist[maxerr];
        if (nrmax != 1)
        {
            var ido = nrmax - 1;
            for (var i = 1; i <= ido; i++)
            {
                var isucc = iord[nrmax - 1];
                if (errmax <= elist[isucc]) break;
                iord[nrmax] = isucc;
                nrmax--;
            }
        }
        var jupbn = last;
        if (last > limit / 2 + 2) jupbn = limit + 3 - last;
        var errmin = elist[last];
        var jbnd = jupbn - 1;
        var ibeg = nrmax + 1;
        var ifound = -1;
        if (ibeg <= jbnd)
        {
            for (var i = ibeg; i <= jbnd; i++)
            {
                var isucc = iord[i];
                if (errmax >= elist[isucc]) { ifound = i; break; }
                iord[i - 1] = isucc;
            }
        }
        if (ifound < 0)
        {
            iord[jbnd] = maxerr;
            iord[jupbn] = last;
            goto L90;
        }
        {
            var i = ifound;
            iord[i - 1] = maxerr;
            var k = jbnd;
            var placed = false;
            for (var j = i; j <= jbnd; j++)
            {
                var isucc = iord[k];
                if (errmin < elist[isucc]) { iord[k + 1] = last; placed = true; break; }
                iord[k + 1] = isucc;
                k--;
            }
            if (!placed) iord[i] = last;
        }
    L90:
        maxerr = iord[nrmax];
        ermax = elist[maxerr];
    }

    /// <summary>Wynn's epsilon algorithm, dqelg. epstab is 1-based with 52 + 1 usable slots.</summary>
    internal static void Qelg(ref int n, double[] epstab, out double result, out double abserr, double[] res3la, ref int nres)
    {
        nres++;
        abserr = Oflow;
        result = epstab[n];
        if (n < 3) goto L100;
        const int limexp = 50;
        epstab[n + 2] = epstab[n];
        var newelm = (n - 1) / 2;
        epstab[n] = Oflow;
        var num = n;
        var k1 = n;
        for (var i = 1; i <= newelm; i++)
        {
            var k2 = k1 - 1;
            var k3 = k1 - 2;
            var res = epstab[k1 + 2];
            var e0 = epstab[k3];
            var e1 = epstab[k2];
            var e2 = res;
            var e1abs = Math.Abs(e1);
            var delta2 = e2 - e1;
            var err2 = Math.Abs(delta2);
            var tol2 = Math.Max(Math.Abs(e2), e1abs) * Epmach;
            var delta3 = e1 - e0;
            var err3 = Math.Abs(delta3);
            var tol3 = Math.Max(e1abs, Math.Abs(e0)) * Epmach;
            if (err2 <= tol2 && err3 <= tol3)
            {
                result = res;
                abserr = err2 + err3;
                goto L100;
            }
            var e3 = epstab[k1];
            epstab[k1] = e1;
            var delta1 = e1 - e3;
            var err1 = Math.Abs(delta1);
            var tol1 = Math.Max(e1abs, Math.Abs(e3)) * Epmach;
            if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) { n = i + i - 1; break; }
            var ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
            var epsinf = Math.Abs(ss * e1);
            if (!(epsinf > 1e-4)) { n = i + i - 1; break; }
            res = e1 + 1.0 / ss;
            epstab[k1] = res;
            k1 -= 2;
            var error = err2 + Math.Abs(res - e2) + err3;
            if (error > abserr) continue;
            abserr = error;
            result = res;
        }
        if (n == limexp) n = 2 * (limexp / 2) - 1;
        var ib = num % 2 == 0 ? 2 : 1;
        var ie = newelm + 1;
        for (var i = 1; i <= ie; i++)
        {
            var ib2 = ib + 2;
            epstab[ib] = epstab[ib2];
            ib = ib2;
        }
        if (num != n)
        {
            var indx = num - n + 1;
            for (var i = 1; i <= n; i++)
            {
                epstab[i] = epstab[indx];
                indx++;
            }
        }
        if (nres < 4)
        {
            res3la[nres] = result;
            abserr = Oflow;
        }
        else
        {
            abserr = Math.Abs(result - res3la[3]) + Math.Abs(result - res3la[2]) + Math.Abs(result - res3la[1]);
            res3la[1] = res3la[2];
            res3la[2] = res3la[3];
            res3la[3] = result;
        }
    L100:
        abserr = Math.Max(abserr, 5.0 * Epmach * Math.Abs(result));
    }

    // ---------------------------------------------------------------- dqagse / dqagie

    internal delegate void Rule(double a, double b, out double result, out double abserr, out double resabs, out double resasc);

    /// <summary>dqagse (rule = dqk21 on [a, b]) and dqagie (rule = dqk15i on [0, 1]) share this body.</summary>
    internal static Output Qags(Rule rule, double a, double b, double epsabs, double epsrel, int limit, Work w, int evalsPerRule, double smallInit)
    {
        var o = new Output();
        var alist = w.Alist; var blist = w.Blist; var rlist = w.Rlist; var elist = w.Elist; var iord = w.Iord;
        alist[1] = a; blist[1] = b; rlist[1] = 0; elist[1] = 0;
        w.Last = 0;
        if (epsabs <= 0 && epsrel < Math.Max(50 * Epmach, 0.5e-28)) { o.Ier = 6; return o; }
        var ierro = 0;
        rule(a, b, out var result, out var abserr, out var defabs, out var resabs);
        var dres = Math.Abs(result);
        var errbnd = Math.Max(epsabs, epsrel * dres);
        var last = 1;
        rlist[1] = result; elist[1] = abserr; iord[1] = 1;
        var ier = 0;
        if (abserr <= 100 * Epmach * defabs && abserr > errbnd) ier = 2;
        if (limit == 1) ier = 1;
        double area = result, errsum = abserr;
        if (ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0.0) goto L140;

        var rlist2 = new double[56];
        var res3la = new double[4];
        rlist2[1] = result;
        var errmax = abserr;
        var maxerr = 1;
        abserr = Oflow;
        var nrmax = 1;
        var nres = 0;
        var numrl2 = 2;
        var ktmin = 0;
        var extrap = false;
        var noext = false;
        int iroff1 = 0, iroff2 = 0, iroff3 = 0;
        var ksgn = -1;
        if (dres >= (1.0 - 50 * Epmach) * defabs) ksgn = 1;
        double small = 0, erlarg = 0, ertest = 0, correc = 0;

        for (last = 2; last <= limit; last++)
        {
            var a1 = alist[maxerr];
            var b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
            var a2 = b1;
            var b2 = blist[maxerr];
            var erlast = errmax;
            rule(a1, b1, out var area1, out var error1, out _, out var defab1);
            rule(a2, b2, out var area2, out var error2, out _, out var defab2);
            var area12 = area1 + area2;
            var erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - rlist[maxerr];
            if (defab1 != error1 && defab2 != error2)
            {
                if (Math.Abs(rlist[maxerr] - area12) <= 1e-5 * Math.Abs(area12) && erro12 >= 0.99 * errmax)
                {
                    if (extrap) iroff2++; else iroff1++;
                }
                if (last > 10 && erro12 > errmax) iroff3++;
            }
            rlist[maxerr] = area1;
            rlist[last] = area2;
            errbnd = Math.Max(epsabs, epsrel * Math.Abs(area));
            if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
            if (iroff2 >= 5) ierro = 3;
            if (last == limit) ier = 1;
            if (Math.Max(Math.Abs(a1), Math.Abs(b2)) <= (1.0 + 100 * Epmach) * (Math.Abs(a2) + 1000 * Uflow)) ier = 4;
            if (error2 > error1)
            {
                alist[maxerr] = a2; alist[last] = a1; blist[last] = b1;
                rlist[maxerr] = area2; rlist[last] = area1;
                elist[maxerr] = error2; elist[last] = error1;
            }
            else
            {
                alist[last] = a2; blist[maxerr] = b1; blist[last] = b2;
                elist[maxerr] = error1; elist[last] = error2;
            }
            Qpsrt(limit, last, ref maxerr, out errmax, elist, iord, ref nrmax);
            if (errsum <= errbnd) goto L115;
            if (ier != 0) goto L100;
            if (last == 2)
            {
                small = smallInit;
                erlarg = errsum;
                ertest = errbnd;
                rlist2[2] = area;
                continue;
            }
            if (noext) continue;
            erlarg -= erlast;
            if (Math.Abs(b1 - a1) > small) erlarg += erro12;
            if (!extrap)
            {
                if (Math.Abs(blist[maxerr] - alist[maxerr]) > small) continue;
                extrap = true;
                nrmax = 2;
            }
            if (ierro != 3 && erlarg > ertest)
            {
                var id = nrmax;
                var jupbnd = last;
                if (last > 2 + limit / 2) jupbnd = limit + 3 - last;
                var jumped = false;
                for (var k = id; k <= jupbnd; k++)
                {
                    maxerr = iord[nrmax];
                    errmax = elist[maxerr];
                    if (Math.Abs(blist[maxerr] - alist[maxerr]) > small) { jumped = true; break; }
                    nrmax++;
                }
                if (jumped) continue;
            }
            numrl2++;
            rlist2[numrl2] = area;
            Qelg(ref numrl2, rlist2, out var reseps, out var abseps, res3la, ref nres);
            ktmin++;
            if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
            if (abseps < abserr)
            {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = Math.Max(epsabs, epsrel * Math.Abs(reseps));
                if (abserr <= ertest) goto L100;
            }
            if (numrl2 == 1) noext = true;
            if (ier == 5) goto L100;
            maxerr = iord[1];
            errmax = elist[maxerr];
            nrmax = 1;
            extrap = false;
            small *= 0.5;
            erlarg = errsum;
        }
        if (last > limit) last = limit;

    L100:
        if (abserr == Oflow) goto L115;
        if (ier + ierro != 0)
        {
            if (ierro == 3) abserr += correc;
            if (ier == 0) ier = 3;
            if (result != 0.0 && area != 0.0)
            {
                if (abserr / Math.Abs(result) > errsum / Math.Abs(area)) goto L115;
            }
            else if (abserr > errsum) goto L115;
            else if (area == 0.0) goto L130;
        }
        if (ksgn == -1 && Math.Max(Math.Abs(result), Math.Abs(area)) <= defabs * 0.01) goto L130;
        if (0.01 > result / area || result / area > 100 || errsum > Math.Abs(area)) ier = 6;
        goto L130;
    L115:
        result = 0;
        for (var k = 1; k <= last; k++) result += rlist[k];
        abserr = errsum;
    L130:
        if (ier > 2) ier--;
    L140:
        o.Result = result;
        o.AbsErr = abserr;
        o.Ier = ier;
        o.Neval = evalsPerRule * (2 * last - 1);
        w.Last = last;
        return o;
    }

    internal static Output Qagse(Func<double, double> f, double a, double b, double epsabs, double epsrel, int limit, Work w)
    {
        Rule r = (double x0, double x1, out double res, out double err, out double ra, out double rasc) => Qk21(f, x0, x1, out res, out err, out ra, out rasc);
        return Qags(r, a, b, epsabs, epsrel, limit, w, 21, Math.Abs(b - a) * 0.375);
    }

    /// <summary>dqagie. inf = 1: (bound, +inf); -1: (-inf, bound); 2: (-inf, +inf).</summary>
    internal static Output Qagie(Func<double, double> f, double bound, int inf, double epsabs, double epsrel, int limit, Work w)
    {
        var boun = inf == 2 ? 0.0 : bound;
        Rule r = (double x0, double x1, out double res, out double err, out double ra, out double rasc) => Qk15i(f, boun, inf, x0, x1, out res, out err, out ra, out rasc);
        var o = Qags(r, 0.0, 1.0, epsabs, epsrel, limit, w, 15, 0.375);
        if (inf == 2) o.Neval *= 2;
        return o;
    }

    // ---------------------------------------------------------------- dqagpe

    internal static Output Qagpe(Func<double, double> f, double a, double b, double[] points, double epsabs, double epsrel, int limit, Work w)
    {
        var o = new Output();
        var npts = points.Length;
        var npts2 = npts + 2;
        var alist = w.Alist; var blist = w.Blist; var rlist = w.Rlist; var elist = w.Elist; var iord = w.Iord; var level = w.Level;
        w.Last = 0;
        if (npts2 < 2 || limit <= npts || (epsabs <= 0 && epsrel < Math.Max(50 * Epmach, 0.5e-28))) { o.Ier = 6; return o; }
        var sign = a > b ? -1.0 : 1.0;
        var pts = new double[npts2 + 1];
        var ndin = new int[npts2 + 1];
        pts[1] = Math.Min(a, b);
        for (var i = 1; i <= npts; i++) pts[i + 1] = points[i - 1];
        pts[npts + 2] = Math.Max(a, b);
        var nint = npts + 1;
        var a1 = pts[1];
        if (npts != 0)
        {
            var nintp1 = nint + 1;
            for (var i = 1; i <= nint; i++)
                for (var j = i + 1; j <= nintp1; j++)
                    if (pts[i] > pts[j]) (pts[i], pts[j]) = (pts[j], pts[i]);
            if (pts[1] != Math.Min(a, b) || pts[nintp1] != Math.Max(a, b)) { o.Ier = 6; return o; }
        }
        double resabs = 0, result = 0, abserr = 0;
        for (var i = 1; i <= nint; i++)
        {
            var b1 = pts[i + 1];
            Qk21(f, a1, b1, out var area1, out var error1, out var defabs, out var resa);
            abserr += error1;
            result += area1;
            ndin[i] = error1 == resa && error1 != 0.0 ? 1 : 0;
            resabs += defabs;
            level[i] = 0;
            elist[i] = error1; alist[i] = a1; blist[i] = b1; rlist[i] = area1; iord[i] = i;
            a1 = b1;
        }
        var errsum = 0.0;
        for (var i = 1; i <= nint; i++)
        {
            if (ndin[i] == 1) elist[i] = abserr;
            errsum += elist[i];
        }
        var last = nint;
        var neval = 21 * nint;
        var dres = Math.Abs(result);
        var errbnd = Math.Max(epsabs, epsrel * dres);
        var ier = 0;
        if (abserr <= 100 * Epmach * resabs && abserr > errbnd) ier = 2;
        if (nint != 1)
        {
            for (var i = 1; i <= npts; i++)
            {
                var jlow = i + 1;
                var ind1 = iord[i];
                var k = 0;
                for (var j = jlow; j <= nint; j++)
                {
                    var ind2 = iord[j];
                    if (elist[ind1] > elist[ind2]) continue;
                    ind1 = ind2;
                    k = j;
                }
                if (ind1 == iord[i]) continue;
                iord[k] = iord[i];
                iord[i] = ind1;
            }
            if (limit < npts2) ier = 1;
        }
        double area = result;
        if (ier != 0 || abserr <= errbnd) goto L210;

        var rlist2 = new double[56];
        var res3la = new double[4];
        rlist2[1] = result;
        var maxerr = iord[1];
        var errmax = elist[maxerr];
        var nrmax = 1;
        var nres = 0;
        var numrl2 = 1;
        var ktmin = 0;
        var extrap = false;
        var noext = false;
        var erlarg = errsum;
        var ertest = errbnd;
        var levmax = 1;
        int iroff1 = 0, iroff2 = 0, iroff3 = 0, ierro = 0;
        abserr = Oflow;
        var ksgn = dres >= (1.0 - 50 * Epmach) * resabs ? 1 : -1;
        double correc = 0;

        for (last = npts2; last <= limit; last++)
        {
            var levcur = level[maxerr] + 1;
            var aa1 = alist[maxerr];
            var b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
            var a2 = b1;
            var b2 = blist[maxerr];
            var erlast = errmax;
            Qk21(f, aa1, b1, out var area1, out var error1, out _, out var defab1);
            Qk21(f, a2, b2, out var area2, out var error2, out _, out var defab2);
            neval += 42;
            var area12 = area1 + area2;
            var erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - rlist[maxerr];
            if (defab1 != error1 && defab2 != error2)
            {
                if (Math.Abs(rlist[maxerr] - area12) <= 1e-5 * Math.Abs(area12) && erro12 >= 0.99 * errmax)
                {
                    if (extrap) iroff2++; else iroff1++;
                }
                if (last > 10 && erro12 > errmax) iroff3++;
            }
            level[maxerr] = levcur;
            level[last] = levcur;
            rlist[maxerr] = area1;
            rlist[last] = area2;
            errbnd = Math.Max(epsabs, epsrel * Math.Abs(area));
            if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
            if (iroff2 >= 5) ierro = 3;
            if (last == limit) ier = 1;
            if (Math.Max(Math.Abs(aa1), Math.Abs(b2)) <= (1.0 + 100 * Epmach) * (Math.Abs(a2) + 1000 * Uflow)) ier = 4;
            if (error2 > error1)
            {
                alist[maxerr] = a2; alist[last] = aa1; blist[last] = b1;
                rlist[maxerr] = area2; rlist[last] = area1;
                elist[maxerr] = error2; elist[last] = error1;
            }
            else
            {
                alist[last] = a2; blist[maxerr] = b1; blist[last] = b2;
                elist[maxerr] = error1; elist[last] = error2;
            }
            Qpsrt(limit, last, ref maxerr, out errmax, elist, iord, ref nrmax);
            if (errsum <= errbnd) goto L190;
            if (ier != 0) goto L170;
            if (noext) continue;
            erlarg -= erlast;
            if (levcur + 1 <= levmax) erlarg += erro12;
            if (!extrap)
            {
                if (level[maxerr] + 1 <= levmax) continue;
                extrap = true;
                nrmax = 2;
            }
            if (ierro != 3 && erlarg > ertest)
            {
                var id = nrmax;
                var jupbnd = last;
                if (last > 2 + limit / 2) jupbnd = limit + 3 - last;
                var jumped = false;
                for (var k = id; k <= jupbnd; k++)
                {
                    maxerr = iord[nrmax];
                    errmax = elist[maxerr];
                    if (level[maxerr] + 1 <= levmax) { jumped = true; break; }
                    nrmax++;
                }
                if (jumped) continue;
            }
            numrl2++;
            rlist2[numrl2] = area;
            if (numrl2 > 2)
            {
                Qelg(ref numrl2, rlist2, out var reseps, out var abseps, res3la, ref nres);
                ktmin++;
                if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
                if (abseps < abserr)
                {
                    ktmin = 0;
                    abserr = abseps;
                    result = reseps;
                    correc = erlarg;
                    ertest = Math.Max(epsabs, epsrel * Math.Abs(reseps));
                    if (abserr < ertest) goto L170;
                }
                if (numrl2 == 1) noext = true;
                if (ier >= 5) goto L170;
            }
            maxerr = iord[1];
            errmax = elist[maxerr];
            nrmax = 1;
            extrap = false;
            levmax++;
            erlarg = errsum;
        }
        if (last > limit) last = limit;

    L170:
        if (abserr == Oflow) goto L190;
        if (ier + ierro != 0)
        {
            if (ierro == 3) abserr += correc;
            if (ier == 0) ier = 3;
            if (result != 0.0 && area != 0.0)
            {
                if (abserr / Math.Abs(result) > errsum / Math.Abs(area)) goto L190;
            }
            else if (abserr > errsum) goto L190;
            else if (area == 0.0) goto L210;
        }
        if (ksgn == -1 && Math.Max(Math.Abs(result), Math.Abs(area)) <= resabs * 0.01) goto L210;
        if (0.01 > result / area || result / area > 100 || errsum > Math.Abs(area)) ier = 6;
        goto L210;
    L190:
        result = 0;
        for (var k = 1; k <= last; k++) result += rlist[k];
        abserr = errsum;
    L210:
        if (ier > 2) ier--;
        o.Result = result * sign;
        o.AbsErr = abserr;
        o.Ier = ier;
        o.Neval = neval;
        w.Last = last;
        return o;
    }

    // ---------------------------------------------------------------- dqawoe / dqc25f

    /// <summary>Chebyshev moment storage shared across dqawoe calls (dqawfe reuses it per cycle).</summary>
    internal sealed class MomentCache
    {
        internal readonly int Maxp1;
        internal readonly double[,] Chebmo;
        internal int Momcom;
        internal int M; // QUADPACK's saved local "m"

        internal MomentCache(int maxp1)
        {
            Maxp1 = maxp1;
            Chebmo = new double[maxp1 + 1, 26];
        }
    }

    /// <summary>
    /// The Chebyshev moments of cos/sin(parint x) on [-1, 1] for T_0..T_24, QUADPACK's way:
    /// a tridiagonal boundary-value solve for |parint| &lt;= 24 (forward recursion is unstable
    /// there) and forward recursion above. v[1..25] cos moments at odd 1-based indices, sin at even.
    /// </summary>
    internal static void ComputeOscillatoryMoments(double parint, Span<double> row /* 1..25 */)
    {
        Span<double> v = stackalloc double[29];
        Span<double> d = stackalloc double[26];
        Span<double> d1 = stackalloc double[26];
        Span<double> d2 = stackalloc double[26];
        var par2 = parint * parint;
        var par22 = par2 + 2.0;
        var sinpar = Math.Sin(parint);
        var cospar = Math.Cos(parint);
        const int noequ = 25;
        const int noeq1 = 24;

        v[1] = 2.0 * sinpar / parint;
        v[2] = (8.0 * cospar + (par2 + par2 - 8.0) * sinpar / parint) / par2;
        v[3] = (32.0 * (par2 - 12.0) * cospar + (2.0 * ((par2 - 80.0) * par2 + 192.0) * sinpar) / parint) / (par2 * par2);
        var ac = 8.0 * cospar;
        var @as = 24.0 * parint * sinpar;
        if (Math.Abs(parint) <= 24.0)
        {
            var an = 6.0;
            for (var k = 1; k <= noeq1; k++)
            {
                var an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 3] = @as - (an2 - 4.0) * ac;
                an += 2.0;
            }
            {
                var an2 = an * an;
                d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                v[noequ + 3] = @as - (an2 - 4.0) * ac;
                v[4] -= 56.0 * par2 * v[3];
                var ass = parint * sinpar;
                var asap = (((((210.0 * par2 - 1.0) * cospar - (105.0 * par2 - 63.0) * ass) / an2
                    - (1.0 - 15.0 * par2) * cospar + 15.0 * ass) / an2 - cospar + 3.0 * ass) / an2 - cospar) / an2;
                v[noequ + 3] -= 2.0 * asap * par2 * (an - 1.0) * (an - 2.0);
            }
            Tridiagonal(noequ, d1, d, d2, v.Slice(4));
        }
        else
        {
            var an = 4.0;
            for (var i = 4; i <= 13; i++)
            {
                var an2 = an * an;
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] - ac) + @as - par2 * (an + 1.0) * (an + 2.0) * v[i - 2])
                    / (par2 * (an - 1.0) * (an - 2.0));
                an += 2.0;
            }
        }
        for (var j = 1; j <= 13; j++) row[2 * j - 1] = v[j];

        v[1] = 2.0 * (sinpar - parint * cospar) / par2;
        v[2] = (18.0 - 48.0 / par2) * sinpar / par2 + (-2.0 + 48.0 / par2) * cospar / parint;
        ac = -24.0 * parint * cospar;
        @as = -8.0 * sinpar;
        if (Math.Abs(parint) <= 24.0)
        {
            d.Clear(); d1.Clear(); d2.Clear();
            var an = 5.0;
            for (var k = 1; k <= noeq1; k++)
            {
                var an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 2] = ac + (an2 - 4.0) * @as;
                an += 2.0;
            }
            {
                var an2 = an * an;
                d[noequ] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                v[noequ + 2] = ac + (an2 - 4.0) * @as;
                v[3] -= 42.0 * par2 * v[2];
                var ass = parint * cospar;
                var asap = (((((105.0 * par2 - 63.0) * ass + (210.0 * par2 - 1.0) * sinpar) / an2
                    + (15.0 * par2 - 1.0) * sinpar - 15.0 * ass) / an2 - 3.0 * ass - sinpar) / an2 - sinpar) / an2;
                v[noequ + 2] -= 2.0 * asap * par2 * (an - 1.0) * (an - 2.0);
            }
            Tridiagonal(noequ, d1, d, d2, v.Slice(3));
        }
        else
        {
            var an = 3.0;
            for (var i = 3; i <= 12; i++)
            {
                var an2 = an * an;
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] + @as) + ac - par2 * (an + 1.0) * (an + 2.0) * v[i - 2])
                    / (par2 * (an - 1.0) * (an - 2.0));
                an += 2.0;
            }
        }
        for (var j = 1; j <= 12; j++) row[2 * j] = v[j];
    }

    /// <summary>LINPACK dgtsl: tridiagonal solve with partial pivoting. c = sub (2..n), d = diag, e = super (1..n-1); b is 0-based.</summary>
    private static void Tridiagonal(int n, Span<double> c, Span<double> d, Span<double> e, Span<double> b0)
    {
        // Work 1-based on copies so b can be indexed like the Fortran.
        Span<double> cc = stackalloc double[n + 2];
        Span<double> dd = stackalloc double[n + 2];
        Span<double> ee = stackalloc double[n + 2];
        Span<double> b = stackalloc double[n + 2];
        for (var i = 1; i <= n; i++) { cc[i] = c[i]; dd[i] = d[i]; ee[i] = e[i]; b[i] = b0[i - 1]; }
        cc[1] = dd[1];
        var nm1 = n - 1;
        if (nm1 >= 1)
        {
            dd[1] = ee[1];
            ee[1] = 0.0;
            ee[n] = 0.0;
            for (var k = 1; k <= nm1; k++)
            {
                var kp1 = k + 1;
                if (Math.Abs(cc[kp1]) >= Math.Abs(cc[k]))
                {
                    (cc[kp1], cc[k]) = (cc[k], cc[kp1]);
                    (dd[kp1], dd[k]) = (dd[k], dd[kp1]);
                    (ee[kp1], ee[k]) = (ee[k], ee[kp1]);
                    (b[kp1], b[k]) = (b[k], b[kp1]);
                }
                if (cc[k] == 0.0) return;
                var t = -cc[kp1] / cc[k];
                cc[kp1] = dd[kp1] + t * dd[k];
                dd[kp1] = ee[kp1] + t * ee[k];
                ee[kp1] = 0.0;
                b[kp1] += t * b[k];
            }
        }
        if (cc[n] == 0.0) return;
        var nm2 = n - 2;
        b[n] /= cc[n];
        if (n > 1)
        {
            b[nm1] = (b[nm1] - dd[nm1] * b[n]) / cc[nm1];
            for (var kb = 1; kb <= nm2; kb++)
            {
                var k = nm2 - kb + 1;
                b[k] = (b[k] - dd[k] * b[k + 1] - ee[k] * b[k + 2]) / cc[k];
            }
        }
        for (var i = 1; i <= n; i++) b0[i - 1] = b[i];
    }

    /// <summary>dqc25f. integr 1 = cos, 2 = sin.</summary>
    private static void Qc25f(Func<double, double> f, double a, double b, double omega, int integr, int nrmom, MomentCache mc, int ksave,
        out double result, out double abserr, out int neval, out double resabs, out double resasc)
    {
        var centr = 0.5 * (b + a);
        var hlgth = 0.5 * (b - a);
        var parint = omega * hlgth;
        if (Math.Abs(parint) <= 2.0)
        {
            Func<double, double> w = integr == 1 ? x => Math.Cos(omega * x) : x => Math.Sin(omega * x);
            Qk15w(f, w, a, b, out result, out abserr, out resabs, out resasc);
            neval = 15;
            return;
        }
        var conc = hlgth * Math.Cos(centr * omega);
        var cons = hlgth * Math.Sin(centr * omega);
        resasc = Oflow;
        neval = 25;
        if (!(nrmom < mc.Momcom || ksave == 1))
        {
            mc.M = mc.Momcom + 1;
            Span<double> row = stackalloc double[26];
            ComputeOscillatoryMoments(parint, row);
            for (var j = 1; j <= 25; j++) mc.Chebmo[mc.M, j] = row[j];
        }
        if (nrmom < mc.Momcom) mc.M = nrmom + 1;
        if (mc.Momcom < mc.Maxp1 - 1 && nrmom >= mc.Momcom) mc.Momcom++;
        var m = mc.M;

        Span<double> fval = stackalloc double[25];
        Span<double> cheb12 = stackalloc double[13];
        Span<double> cheb24 = stackalloc double[25];
        FillCcValues(f, centr, hlgth, fval);
        Cheb(fval, cheb12, cheb24);
        // 1-based moments: chebmo[m, k] pairs with Chebyshev coefficient k-1.
        var resc12 = cheb12[12] * mc.Chebmo[m, 13];
        var ress12 = 0.0;
        var k = 11;
        for (var j = 1; j <= 6; j++)
        {
            resc12 += cheb12[k - 1] * mc.Chebmo[m, k];
            ress12 += cheb12[k] * mc.Chebmo[m, k + 1];
            k -= 2;
        }
        var resc24 = cheb24[24] * mc.Chebmo[m, 25];
        var ress24 = 0.0;
        resabs = Math.Abs(cheb24[24]);
        k = 23;
        for (var j = 1; j <= 12; j++)
        {
            resc24 += cheb24[k - 1] * mc.Chebmo[m, k];
            ress24 += cheb24[k] * mc.Chebmo[m, k + 1];
            resabs += Math.Abs(cheb24[k - 1]) + Math.Abs(cheb24[k]);
            k -= 2;
        }
        var estc = Math.Abs(resc24 - resc12);
        var ests = Math.Abs(ress24 - ress12);
        resabs *= Math.Abs(hlgth);
        if (integr == 2)
        {
            result = conc * ress24 + cons * resc24;
            abserr = Math.Abs(conc * ests) + Math.Abs(cons * estc);
        }
        else
        {
            result = conc * resc24 - cons * ress24;
            abserr = Math.Abs(conc * estc) + Math.Abs(cons * ests);
        }
    }

    internal static Output Qawoe(Func<double, double> f, double a, double b, double omega, int integr, double epsabs, double epsrel,
        int limit, int icall, MomentCache mc, Work w)
    {
        var o = new Output();
        var alist = w.Alist; var blist = w.Blist; var rlist = w.Rlist; var elist = w.Elist; var iord = w.Iord; var nnlog = w.Level;
        alist[1] = a; blist[1] = b; rlist[1] = 0; elist[1] = 0; iord[1] = 0; nnlog[1] = 0;
        w.Last = 0;
        if ((integr != 1 && integr != 2) || (epsabs <= 0 && epsrel < Math.Max(50 * Epmach, 0.5e-28)) || icall < 1 || mc.Maxp1 < 1)
        { o.Ier = 6; return o; }
        var domega = Math.Abs(omega);
        var nrmom = 0;
        if (icall <= 1) mc.Momcom = 0;
        Qc25f(f, a, b, domega, integr, nrmom, mc, 0, out var result, out var abserr, out var neval, out var defabs, out var resabs);
        var dres = Math.Abs(result);
        var errbnd = Math.Max(epsabs, epsrel * dres);
        rlist[1] = result; elist[1] = abserr; iord[1] = 1;
        var ier = 0;
        var last = 1;
        if (abserr <= 100 * Epmach * defabs && abserr > errbnd) ier = 2;
        if (limit == 1) ier = 1;
        double area = result, errsum = abserr;
        var nres = 0;
        if (ier != 0 || abserr <= errbnd) goto L200;

        var rlist2 = new double[56];
        var res3la = new double[4];
        var errmax = abserr;
        var maxerr = 1;
        abserr = Oflow;
        var nrmax = 1;
        var extrap = false;
        var noext = false;
        int ierro = 0, iroff1 = 0, iroff2 = 0, iroff3 = 0, ktmin = 0;
        var small = Math.Abs(b - a) * 0.75;
        var numrl2 = 0;
        var extall = false;
        if (0.5 * Math.Abs(b - a) * domega <= 2.0)
        {
            numrl2 = 1;
            extall = true;
            rlist2[1] = result;
        }
        if (0.25 * Math.Abs(b - a) * domega <= 2.0) extall = true;
        var ksgn = dres >= (1.0 - 50 * Epmach) * defabs ? 1 : -1;
        double erlarg = 0, ertest = 0, correc = 0;

        for (last = 2; last <= limit; last++)
        {
            nrmom = nnlog[maxerr] + 1;
            var a1 = alist[maxerr];
            var b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
            var a2 = b1;
            var b2 = blist[maxerr];
            var erlast = errmax;
            Qc25f(f, a1, b1, domega, integr, nrmom, mc, 0, out var area1, out var error1, out var nev, out _, out var defab1);
            neval += nev;
            Qc25f(f, a2, b2, domega, integr, nrmom, mc, 1, out var area2, out var error2, out nev, out _, out var defab2);
            neval += nev;
            var area12 = area1 + area2;
            var erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - rlist[maxerr];
            if (defab1 != error1 && defab2 != error2)
            {
                if (Math.Abs(rlist[maxerr] - area12) <= 1e-5 * Math.Abs(area12) && erro12 >= 0.99 * errmax)
                {
                    if (extrap) iroff2++; else iroff1++;
                }
                if (last > 10 && erro12 > errmax) iroff3++;
            }
            rlist[maxerr] = area1;
            rlist[last] = area2;
            nnlog[maxerr] = nrmom;
            nnlog[last] = nrmom;
            errbnd = Math.Max(epsabs, epsrel * Math.Abs(area));
            if (iroff1 + iroff2 >= 10 || iroff3 >= 20) ier = 2;
            if (iroff2 >= 5) ierro = 3;
            if (last == limit) ier = 1;
            if (Math.Max(Math.Abs(a1), Math.Abs(b2)) <= (1.0 + 100 * Epmach) * (Math.Abs(a2) + 1000 * Uflow)) ier = 4;
            if (error2 > error1)
            {
                alist[maxerr] = a2; alist[last] = a1; blist[last] = b1;
                rlist[maxerr] = area2; rlist[last] = area1;
                elist[maxerr] = error2; elist[last] = error1;
            }
            else
            {
                alist[last] = a2; blist[maxerr] = b1; blist[last] = b2;
                elist[maxerr] = error1; elist[last] = error2;
            }
            Qpsrt(limit, last, ref maxerr, out errmax, elist, iord, ref nrmax);
            if (errsum <= errbnd) goto L170;
            if (ier != 0) goto L150;
            if (last == 2 && extall)
            {
                small *= 0.5;
                numrl2++;
                rlist2[numrl2] = area;
                ertest = errbnd;
                erlarg = errsum;
                continue;
            }
            if (noext) continue;
            var goto70 = false;
            if (extall)
            {
                erlarg -= erlast;
                if (Math.Abs(b1 - a1) > small) erlarg += erro12;
                if (extrap) goto70 = true;
            }
            if (!goto70)
            {
                var width = Math.Abs(blist[maxerr] - alist[maxerr]);
                if (width > small) continue;
                if (!extall)
                {
                    small *= 0.5;
                    if (0.25 * width * domega > 2.0) continue;
                    extall = true;
                    ertest = errbnd;
                    erlarg = errsum;
                    continue;
                }
                extrap = true;
                nrmax = 2;
            }
            if (ierro != 3 && erlarg > ertest)
            {
                var jupbnd = last;
                if (last > limit / 2 + 2) jupbnd = limit + 3 - last;
                var id = nrmax;
                var jumped = false;
                for (var k = id; k <= jupbnd; k++)
                {
                    maxerr = iord[nrmax];
                    errmax = elist[maxerr];
                    if (Math.Abs(blist[maxerr] - alist[maxerr]) > small) { jumped = true; break; }
                    nrmax++;
                }
                if (jumped) continue;
            }
            numrl2++;
            rlist2[numrl2] = area;
            if (numrl2 >= 3)
            {
                Qelg(ref numrl2, rlist2, out var reseps, out var abseps, res3la, ref nres);
                ktmin++;
                if (ktmin > 5 && abserr < 1e-3 * errsum) ier = 5;
                if (abseps < abserr)
                {
                    ktmin = 0;
                    abserr = abseps;
                    result = reseps;
                    correc = erlarg;
                    ertest = Math.Max(epsabs, epsrel * Math.Abs(reseps));
                    if (abserr <= ertest) goto L150;
                }
                if (numrl2 == 1) noext = true;
                if (ier == 5) goto L150;
            }
            maxerr = iord[1];
            errmax = elist[maxerr];
            nrmax = 1;
            extrap = false;
            small *= 0.5;
            erlarg = errsum;
        }
        if (last > limit) last = limit;

    L150:
        if (abserr == Oflow || nres == 0) goto L170;
        if (ier + ierro != 0)
        {
            if (ierro == 3) abserr += correc;
            if (ier == 0) ier = 3;
            if (result != 0.0 && area != 0.0)
            {
                if (abserr / Math.Abs(result) > errsum / Math.Abs(area)) goto L170;
            }
            else if (abserr > errsum) goto L170;
            else if (area == 0.0) goto L190;
        }
        if (ksgn == -1 && Math.Max(Math.Abs(result), Math.Abs(area)) <= defabs * 0.01) goto L190;
        if (0.01 > result / area || result / area > 100 || errsum >= Math.Abs(area)) ier = 6;
        goto L190;
    L170:
        result = 0;
        for (var k = 1; k <= last; k++) result += rlist[k];
        abserr = errsum;
    L190:
        if (ier > 2) ier--;
    L200:
        if (integr == 2 && omega < 0.0) result = -result;
        o.Result = result;
        o.AbsErr = abserr;
        o.Ier = ier;
        o.Neval = neval;
        w.Last = last;
        return o;
    }

    /// <summary>dqawfe result plus the per-cycle lists scipy reports in full output.</summary>
    internal struct FourierOutput
    {
        internal Output Main;
        internal int Lst;
        internal double[] Rslst, Erlst;
        internal int[] Ierlst;
    }

    internal static FourierOutput Qawfe(Func<double, double> f, double a, double omega, int integr, double epsabs, int limlst, int limit, int maxp1)
    {
        var fo = new FourierOutput { Rslst = new double[limlst + 1], Erlst = new double[limlst + 1], Ierlst = new int[limlst + 1] };
        var o = new Output();
        if ((integr != 1 && integr != 2) || epsabs <= 0 || limlst < 3) { o.Ier = 6; fo.Main = o; return fo; }
        var w = new Work(limit);
        if (omega == 0.0)
        {
            if (integr == 1) o = Qagie(f, a, 1, epsabs, 0.0, limit, w);
            fo.Rslst[1] = o.Result; fo.Erlst[1] = o.AbsErr; fo.Ierlst[1] = o.Ier; fo.Lst = 1;
            fo.Main = o;
            return fo;
        }
        const double p = 0.9;
        var l = (int)Math.Abs(omega);
        var dl = 2 * l + 1;
        var cycle = dl * Math.PI / Math.Abs(omega);
        var ier = 0;
        var ktmin = 0;
        var neval = 0;
        var numrl2 = 0;
        var nres = 0;
        var c1 = a;
        var c2 = cycle + a;
        var p1 = 1.0 - p;
        var eps = epsabs;
        if (epsabs > Uflow / p1) eps = epsabs * p1;
        var ep = eps;
        var fact = 1.0;
        var correc = 0.0;
        var abserr = 0.0;
        var errsum = 0.0;
        double result = 0, drl = 0;
        var psum = new double[56];
        var res3la = new double[4];
        var mc = new MomentCache(maxp1);
        var ll = 0;
        int lst;
        var rslst = fo.Rslst; var erlst = fo.Erlst; var ierlst = fo.Ierlst;
        for (lst = 1; lst <= limlst; lst++)
        {
            var epsa = eps * fact;
            var oo = Qawoe(f, c1, c2, omega, integr, epsa, 0.0, limit, lst, mc, w);
            rslst[lst] = oo.Result; erlst[lst] = oo.AbsErr; ierlst[lst] = oo.Ier;
            neval += oo.Neval;
            fact *= p;
            errsum += erlst[lst];
            drl = 50.0 * Math.Abs(rslst[lst]);
            if (errsum + drl <= epsabs && lst >= 6) goto L80;
            correc = Math.Max(correc, erlst[lst]);
            if (ierlst[lst] != 0) { eps = Math.Max(ep, correc * p1); ier = 7; }
            if (ier == 7 && errsum + drl <= correc * 10.0 && lst > 5) goto L80;
            numrl2++;
            if (lst > 1)
            {
                psum[numrl2] = psum[ll] + rslst[lst];
                if (lst != 2)
                {
                    if (lst == limlst) ier = 1;
                    Qelg(ref numrl2, psum, out var reseps, out var abseps, res3la, ref nres);
                    ktmin++;
                    if (ktmin >= 15 && abserr <= 1e-3 * (errsum + drl)) ier = 4;
                    if (!(abseps > abserr && lst != 3))
                    {
                        abserr = abseps;
                        result = reseps;
                        ktmin = 0;
                        if (abserr + 10.0 * correc <= epsabs || (abserr <= epsabs && 10.0 * correc >= epsabs)) goto L60;
                    }
                    if (ier != 0 && ier != 7) goto L60;
                }
            }
            else psum[1] = rslst[1];
            ll = numrl2;
            c1 = c2;
            c2 += cycle;
        }
        if (lst > limlst) lst = limlst;
    L60:
        abserr += 10.0 * correc;
        if (ier == 0) goto L999;
        if (result != 0.0 && psum[numrl2] != 0.0)
        {
            if (abserr / Math.Abs(result) > (errsum + drl) / Math.Abs(psum[numrl2])) goto L80;
            if (ier >= 1 && ier != 7) abserr += drl;
            goto L999;
        }
        if (abserr > errsum) goto L80;
        if (psum[numrl2] == 0.0) goto L999;
        if (abserr / Math.Abs(result) > (errsum + drl) / Math.Abs(psum[numrl2])) goto L80;
        if (ier >= 1 && ier != 7) abserr += drl;
        goto L999;
    L80:
        result = psum[numrl2];
        abserr = errsum + drl;
    L999:
        o.Result = result; o.AbsErr = abserr; o.Neval = neval; o.Ier = ier;
        fo.Main = o;
        fo.Lst = lst;
        return fo;
    }

    // ---------------------------------------------------------------- dqawse

    /// <summary>dqmomo: modified Chebyshev moments of the algebraico-logarithmic weights, 1-based [1..25].</summary>
    internal static void Qmomo(double alfa, double beta, double[] ri, double[] rj, double[] rg, double[] rh, int integr)
    {
        var alfp1 = alfa + 1.0; var betp1 = beta + 1.0;
        var alfp2 = alfa + 2.0; var betp2 = beta + 2.0;
        var ralf = Math.Pow(2.0, alfp1);
        var rbet = Math.Pow(2.0, betp1);
        ri[1] = ralf / alfp1;
        rj[1] = rbet / betp1;
        ri[2] = ri[1] * alfa / alfp2;
        rj[2] = rj[1] * beta / betp2;
        var an = 2.0; var anm1 = 1.0;
        for (var i = 3; i <= 25; i++)
        {
            ri[i] = -(ralf + an * (an - alfp2) * ri[i - 1]) / (anm1 * (an + alfp1));
            rj[i] = -(rbet + an * (an - betp2) * rj[i - 1]) / (anm1 * (an + betp1));
            anm1 = an;
            an += 1.0;
        }
        if (integr != 1)
        {
            if (integr != 3)
            {
                rg[1] = -ri[1] / alfp1;
                rg[2] = -(ralf + ralf) / (alfp2 * alfp2) - rg[1];
                an = 2.0; anm1 = 1.0;
                var im1 = 2;
                for (var i = 3; i <= 25; i++)
                {
                    rg[i] = -(an * (an - alfp2) * rg[im1] - an * ri[im1] + anm1 * ri[i]) / (anm1 * (an + alfp1));
                    anm1 = an;
                    an += 1.0;
                    im1 = i;
                }
            }
            if (integr != 2)
            {
                rh[1] = -rj[1] / betp1;
                rh[2] = -(rbet + rbet) / (betp2 * betp2) - rh[1];
                an = 2.0; anm1 = 1.0;
                var im1 = 2;
                for (var i = 3; i <= 25; i++)
                {
                    rh[i] = -(an * (an - betp2) * rh[im1] - an * rj[im1] + anm1 * rj[i]) / (anm1 * (an + betp1));
                    anm1 = an;
                    an += 1.0;
                    im1 = i;
                }
                for (var i = 2; i <= 25; i += 2) rh[i] = -rh[i];
            }
        }
        for (var i = 2; i <= 25; i += 2) rj[i] = -rj[i];
    }

    private static void Qc25s(Func<double, double> f, double a, double b, double bl, double br, double alfa, double beta,
        double[] ri, double[] rj, double[] rg, double[] rh, int integr, out double result, out double abserr, out double resasc, out int nev)
    {
        nev = 25;
        if (bl == a && (alfa != 0.0 || integr == 2 || integr == 4))
        {
            CcEndpoint(f, bl, br, b - 0.5 * (br + bl), beta, -1, integr > 2, ri, rg, integr == 2 || integr == 4, alfa, out result, out abserr);
            resasc = 0; // unused by caller in this branch beyond equality test
            resasc = double.NaN;
            return;
        }
        if (br == b && (beta != 0.0 || integr == 3 || integr == 4))
        {
            CcEndpoint(f, bl, br, 0.5 * (br + bl) - a, alfa, +1, integr == 2 || integr == 4, rj, rh, integr == 3 || integr == 4, beta, out result, out abserr);
            resasc = double.NaN;
            return;
        }
        Func<double, double> w = x =>
        {
            var xma = x - a;
            var bmx = b - x;
            var v = Math.Pow(xma, alfa) * Math.Pow(bmx, beta);
            return integr switch
            {
                2 => v * Math.Log(xma),
                3 => v * Math.Log(bmx),
                4 => v * Math.Log(xma) * Math.Log(bmx),
                _ => v,
            };
        };
        Qk15w(f, w, bl, br, out result, out abserr, out _, out resasc);
        nev = 15;
    }

    /// <summary>
    /// The two Clenshaw-Curtis branches of dqc25s, folded into one. The interval touches the singular
    /// end; the smooth factor (distance to the far end)^expo, optionally times its log, is folded into
    /// the function values, and the singular factor is handled by the moments (rMom, and rLog for the log).
    /// sideSign = -1: interval starts at a, far factor is (fix - u) at node +u; +1: interval ends at b.
    /// </summary>
    private static void CcEndpoint(Func<double, double> f, double bl, double br, double fix, double expo, int sideSign,
        bool logFar, double[] rMom, double[] rLog, bool logNear, double nearExpo, out double result, out double abserr)
    {
        var x = IntegrateRules.Xcc;
        var hlgth = 0.5 * (br - bl);
        var centr = 0.5 * (br + bl);
        Span<double> fval = stackalloc double[25];
        // Factor at the node centr + u: (fix - u) for the a-end, (fix + u) for the b-end.
        double Far(double u) => sideSign < 0 ? fix - u : fix + u;
        double G(double u)
        {
            var d = Far(u);
            var v = Math.Pow(d, expo);
            return logFar ? v * Math.Log(d) : v;
        }
        fval[0] = 0.5 * f(hlgth + centr) * G(hlgth);
        fval[12] = f(centr) * G(0);
        fval[24] = 0.5 * f(centr - hlgth) * G(-hlgth);
        for (var i = 1; i < 12; i++)
        {
            var u = hlgth * x[i - 1];
            fval[i] = f(u + centr) * G(u);
            fval[24 - i] = f(centr - u) * G(-u);
        }
        Span<double> cheb12 = stackalloc double[13];
        Span<double> cheb24 = stackalloc double[25];
        Cheb(fval, cheb12, cheb24);
        var factor = Math.Pow(hlgth, nearExpo + 1.0);
        double res12 = 0, res24 = 0;
        result = 0; abserr = 0;
        for (var i = 0; i < 13; i++) { res12 += cheb12[i] * rMom[i + 1]; res24 += cheb24[i] * rMom[i + 1]; }
        for (var i = 13; i < 25; i++) res24 += cheb24[i] * rMom[i + 1];
        if (logNear)
        {
            var dc = Math.Log(br - bl);
            result = res24 * dc;
            abserr = Math.Abs((res24 - res12) * dc);
            res12 = 0; res24 = 0;
            for (var i = 0; i < 13; i++) { res12 += cheb12[i] * rLog[i + 1]; res24 += cheb24[i] * rLog[i + 1]; }
            for (var i = 13; i < 25; i++) res24 += cheb24[i] * rLog[i + 1];
        }
        result = (result + res24) * factor;
        abserr = (abserr + Math.Abs(res24 - res12)) * factor;
    }

    internal static Output Qawse(Func<double, double> f, double a, double b, double alfa, double beta, int integr, double epsabs, double epsrel, int limit, Work w)
    {
        var o = new Output { Ier = 6 };
        var alist = w.Alist; var blist = w.Blist; var rlist = w.Rlist; var elist = w.Elist; var iord = w.Iord;
        rlist[1] = 0; elist[1] = 0; iord[1] = 0;
        w.Last = 0;
        if (b <= a || (epsabs == 0 && epsrel < Math.Max(50 * Epmach, 0.5e-28)) || alfa <= -1 || beta <= -1 || integr < 1 || integr > 4 || limit < 2)
            return o;
        var ier = 0;
        double[] ri = new double[26], rj = new double[26], rg = new double[26], rh = new double[26];
        Qmomo(alfa, beta, ri, rj, rg, rh, integr);
        var centre = 0.5 * (b + a);
        Qc25s(f, a, b, a, centre, alfa, beta, ri, rj, rg, rh, integr, out var area1, out var error1, out var resas1, out var nev);
        var neval = nev;
        Qc25s(f, a, b, centre, b, alfa, beta, ri, rj, rg, rh, integr, out var area2, out var error2, out var resas2, out nev);
        var last = 2;
        neval += nev;
        var result = area1 + area2;
        var abserr = error1 + error2;
        var errbnd = Math.Max(epsabs, epsrel * Math.Abs(result));
        if (error2 > error1)
        {
            alist[1] = centre; alist[2] = a; blist[1] = b; blist[2] = centre;
            rlist[1] = area2; rlist[2] = area1; elist[1] = error2; elist[2] = error1;
        }
        else
        {
            alist[1] = a; alist[2] = centre; blist[1] = centre; blist[2] = b;
            rlist[1] = area1; rlist[2] = area2; elist[1] = error1; elist[2] = error2;
        }
        iord[1] = 1; iord[2] = 2;
        if (limit == 2) ier = 1;
        if (abserr <= errbnd || ier == 1) goto L999;
        var errmax = elist[1];
        var maxerr = 1;
        var nrmax = 1;
        var area = result;
        var errsum = abserr;
        int iroff1 = 0, iroff2 = 0;
        for (last = 3; last <= limit; last++)
        {
            var a1 = alist[maxerr];
            var b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
            var a2 = b1;
            var b2 = blist[maxerr];
            Qc25s(f, a, b, a1, b1, alfa, beta, ri, rj, rg, rh, integr, out area1, out error1, out resas1, out nev);
            neval += nev;
            Qc25s(f, a, b, a2, b2, alfa, beta, ri, rj, rg, rh, integr, out area2, out error2, out resas2, out nev);
            neval += nev;
            var area12 = area1 + area2;
            var erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - rlist[maxerr];
            if (a != a1 && b != b2 && resas1 != error1 && resas2 != error2)
            {
                if (Math.Abs(rlist[maxerr] - area12) < 1e-5 * Math.Abs(area12) && erro12 >= 0.99 * errmax) iroff1++;
                if (last > 10 && erro12 > errmax) iroff2++;
            }
            rlist[maxerr] = area1;
            rlist[last] = area2;
            errbnd = Math.Max(epsabs, epsrel * Math.Abs(area));
            if (errsum > errbnd)
            {
                if (last == limit) ier = 1;
                if (iroff1 >= 6 || iroff2 >= 20) ier = 2;
                if (Math.Max(Math.Abs(a1), Math.Abs(b2)) <= (1.0 + 100 * Epmach) * (Math.Abs(a2) + 1000 * Uflow)) ier = 3;
            }
            if (error2 > error1)
            {
                alist[maxerr] = a2; alist[last] = a1; blist[last] = b1;
                rlist[maxerr] = area2; rlist[last] = area1;
                elist[maxerr] = error2; elist[last] = error1;
            }
            else
            {
                alist[last] = a2; blist[maxerr] = b1; blist[last] = b2;
                elist[maxerr] = error1; elist[last] = error2;
            }
            Qpsrt(limit, last, ref maxerr, out errmax, elist, iord, ref nrmax);
            if (ier != 0 || errsum <= errbnd) break;
        }
        if (last > limit) last = limit;
        result = 0;
        for (var k = 1; k <= last; k++) result += rlist[k];
        abserr = errsum;
    L999:
        o.Result = result; o.AbsErr = abserr; o.Ier = ier; o.Neval = neval;
        w.Last = last;
        return o;
    }

    // ---------------------------------------------------------------- dqawce

    private static void Qc25c(Func<double, double> f, double a, double b, double c, out double result, out double abserr, ref int krul, out int neval)
    {
        var cc = (2.0 * c - b - a) / (b - a);
        if (Math.Abs(cc) >= 1.1)
        {
            krul--;
            Qk15w(f, x => 1.0 / (x - c), a, b, out result, out abserr, out _, out var resasc);
            neval = 15;
            if (resasc == abserr) krul++;
            return;
        }
        var hlgth = 0.5 * (b - a);
        var centr = 0.5 * (b + a);
        neval = 25;
        Span<double> fval = stackalloc double[25];
        Span<double> cheb12 = stackalloc double[13];
        Span<double> cheb24 = stackalloc double[25];
        FillCcValues(f, centr, hlgth, fval);
        Cheb(fval, cheb12, cheb24);
        var amom0 = Math.Log(Math.Abs((1.0 - cc) / (1.0 + cc)));
        var amom1 = 2.0 + cc * amom0;
        var res12 = cheb12[0] * amom0 + cheb12[1] * amom1;
        var res24 = cheb24[0] * amom0 + cheb24[1] * amom1;
        for (var k = 3; k <= 25; k++)
        {
            var amom2 = 2.0 * cc * amom1 - amom0;
            var ak22 = (double)(k - 2) * (k - 2);
            if (k % 2 == 0) amom2 -= 4.0 / (ak22 - 1.0);
            if (k <= 13) res12 += cheb12[k - 1] * amom2;
            res24 += cheb24[k - 1] * amom2;
            amom0 = amom1;
            amom1 = amom2;
        }
        result = res24;
        abserr = Math.Abs(res24 - res12);
    }

    internal static Output Qawce(Func<double, double> f, double a, double b, double c, double epsabs, double epsrel, int limit, Work w)
    {
        var o = new Output { Ier = 6 };
        var alist = w.Alist; var blist = w.Blist; var rlist = w.Rlist; var elist = w.Elist; var iord = w.Iord;
        alist[1] = a; blist[1] = b; rlist[1] = 0; elist[1] = 0; iord[1] = 0;
        w.Last = 0;
        if (c == a || c == b || (epsabs <= 0 && epsrel < Math.Max(50 * Epmach, 0.5e-28))) return o;
        double aa = a, bb = b;
        if (a > b) { aa = b; bb = a; }
        var ier = 0;
        var krule = 1;
        Qc25c(f, aa, bb, c, out var result, out var abserr, ref krule, out var neval);
        var last = 1;
        rlist[1] = result; elist[1] = abserr; iord[1] = 1; alist[1] = a; blist[1] = b;
        var errbnd = Math.Max(epsabs, epsrel * Math.Abs(result));
        if (limit == 1) ier = 1;
        if (abserr < Math.Min(0.01 * Math.Abs(result), errbnd) || ier == 1) goto L70;
        alist[1] = aa; blist[1] = bb; rlist[1] = result;
        var errmax = abserr;
        var maxerr = 1;
        var area = result;
        var errsum = abserr;
        var nrmax = 1;
        int iroff1 = 0, iroff2 = 0;
        for (last = 2; last <= limit; last++)
        {
            var a1 = alist[maxerr];
            var b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
            var b2 = blist[maxerr];
            if (c <= b1 && c > a1) b1 = 0.5 * (c + b2);
            if (c > b1 && c < b2) b1 = 0.5 * (a1 + c);
            var a2 = b1;
            krule = 2;
            Qc25c(f, a1, b1, c, out var area1, out var error1, ref krule, out var nev);
            neval += nev;
            Qc25c(f, a2, b2, c, out var area2, out var error2, ref krule, out nev);
            neval += nev;
            var area12 = area1 + area2;
            var erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - rlist[maxerr];
            if (Math.Abs(rlist[maxerr] - area12) < 1e-5 * Math.Abs(area12) && erro12 >= 0.99 * errmax && krule == 0) iroff1++;
            if (last > 10 && erro12 > errmax && krule == 0) iroff2++;
            rlist[maxerr] = area1;
            rlist[last] = area2;
            errbnd = Math.Max(epsabs, epsrel * Math.Abs(area));
            if (errsum > errbnd)
            {
                if (iroff1 >= 6 && iroff2 > 20) ier = 2;
                if (last == limit) ier = 1;
                if (Math.Max(Math.Abs(a1), Math.Abs(b2)) <= (1.0 + 100 * Epmach) * (Math.Abs(a2) + 1000 * Uflow)) ier = 3;
            }
            if (error2 > error1)
            {
                alist[maxerr] = a2; alist[last] = a1; blist[last] = b1;
                rlist[maxerr] = area2; rlist[last] = area1;
                elist[maxerr] = error2; elist[last] = error1;
            }
            else
            {
                alist[last] = a2; blist[maxerr] = b1; blist[last] = b2;
                elist[maxerr] = error1; elist[last] = error2;
            }
            Qpsrt(limit, last, ref maxerr, out errmax, elist, iord, ref nrmax);
            if (ier != 0 || errsum <= errbnd) break;
        }
        if (last > limit) last = limit;
        result = 0;
        for (var k = 1; k <= last; k++) result += rlist[k];
        abserr = errsum;
    L70:
        if (aa == b) result = -result;
        o.Result = result; o.AbsErr = abserr; o.Ier = ier; o.Neval = neval;
        w.Last = last;
        return o;
    }
}
