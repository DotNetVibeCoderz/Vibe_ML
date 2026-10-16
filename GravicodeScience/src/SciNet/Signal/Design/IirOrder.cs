namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    private static void ValidateGpassGstop(double gpass, double gstop)
    {
        if (gpass <= 0.0) throw new ArgumentException("gpass should be larger than 0.0", nameof(gpass));
        if (gstop <= 0.0) throw new ArgumentException("gstop should be larger than 0.0", nameof(gstop));
        if (gpass > gstop) throw new ArgumentException("gpass should be smaller than gstop", nameof(gpass));
    }

    private enum OrdKind { Butter, Cheby, Ellip }

    /// <summary>
    /// Shared front half of every <c>*ord</c>: validation, scaling by fs, pre-warping and the
    /// natural-frequency ratio. Band-stop runs scipy's bounded Brent search (<c>fminbound</c>,
    /// xtol 1e-5) to move the passband edges, and the returned Wn depends on where it lands —
    /// so the minimiser is ported step for step rather than replaced by a "better" one.
    /// </summary>
    private static (double[] Wp, double[] Ws, int Type, double[] Passb, double[] Stopb, double Nat) OrdSetup(
        CriticalFrequencies wpIn, CriticalFrequencies wsIn, double gpass, double gstop, bool analog, double? fs, OrdKind kind)
    {
        ValidateGpassGstop(gpass, gstop);
        var wp = wpIn.Values.ToArray();
        var ws = wsIn.Values.ToArray();
        if (wp.Length != ws.Length || wp.Length is < 1 or > 2) throw new ArgumentException("wp and ws must have one or two elements each, and the same shape.");
        if (fs is double f)
        {
            if (analog) throw new ArgumentException("fs cannot be specified for an analog filter.", nameof(fs));
            for (var i = 0; i < wp.Length; i++) { wp[i] = 2 * wp[i] / f; ws[i] = 2 * ws[i] / f; }
        }
        var type = 2 * (wp.Length - 1) + 1;
        if (wp[0] >= ws[0]) type++;
        double[] passb, stopb;
        if (!analog)
        {
            passb = wp.Select(v => Math.Tan(Math.PI * v / 2.0)).ToArray();
            stopb = ws.Select(v => Math.Tan(Math.PI * v / 2.0)).ToArray();
        }
        else { passb = wp.ToArray(); stopb = ws.ToArray(); }

        double[] nat;
        switch (type)
        {
            case 1: nat = stopb.Select((s, i) => s / passb[i]).ToArray(); break;
            case 2: nat = stopb.Select((s, i) => passb[i] / s).ToArray(); break;
            case 3:
                {
                    var pb = passb;
                    var sb = stopb;
                    var wp0 = FminBound(w => BandStopObj(w, 0, pb, sb, gpass, gstop, kind), passb[0], stopb[0] - 1e-12);
                    var wp1 = FminBound(w => BandStopObj(w, 1, pb, sb, gpass, gstop, kind), stopb[1] + 1e-12, passb[1]);
                    passb = [wp0, wp1];
                    nat = stopb.Select(s => s * (passb[0] - passb[1]) / (s * s - passb[0] * passb[1])).ToArray();
                    break;
                }
            default:
                nat = stopb.Select(s => (s * s - passb[0] * passb[1]) / (s * (passb[0] - passb[1]))).ToArray();
                break;
        }
        var natMin = nat.Select(Math.Abs).Min();
        return (wp, ws, type, passb, stopb, natMin);
    }

    /// <summary>scipy's <c>band_stop_obj</c>: the order needed if passband edge <paramref name="ind"/> moved to <paramref name="wp"/>.</summary>
    private static double BandStopObj(double wp, int ind, double[] passb, double[] stopb, double gpass, double gstop, OrdKind kind)
    {
        var pc = passb.ToArray();
        pc[ind] = wp;
        var nat = double.PositiveInfinity;
        foreach (var s in stopb) nat = Math.Min(nat, Math.Abs(s * (pc[0] - pc[1]) / (s * s - pc[0] * pc[1])));
        switch (kind)
        {
            case OrdKind.Butter:
                {
                    var gs = Math.Pow(10, 0.1 * Math.Abs(gstop));
                    var gp = Math.Pow(10, 0.1 * Math.Abs(gpass));
                    return Math.Log10((gs - 1.0) / (gp - 1.0)) / (2 * Math.Log10(nat));
                }
            case OrdKind.Cheby:
                {
                    var gs = Math.Pow(10, 0.1 * Math.Abs(gstop));
                    var gp = Math.Pow(10, 0.1 * Math.Abs(gpass));
                    return Math.Acosh(Math.Sqrt((gs - 1.0) / (gp - 1.0))) / Math.Acosh(nat);
                }
            default:
                {
                    var gs = Math.Pow(10, 0.1 * gstop);
                    var gp = Math.Pow(10, 0.1 * gpass);
                    var arg1 = Math.Sqrt((gp - 1.0) / (gs - 1.0));
                    var arg0 = 1.0 / nat;
                    var d00 = SignalElliptic.K(arg0 * arg0);
                    var d01 = SignalElliptic.K(1 - arg0 * arg0);
                    var d10 = SignalElliptic.K(arg1 * arg1);
                    var d11 = SignalElliptic.K(1 - arg1 * arg1);
                    return d00 * d11 / (d01 * d10);
                }
        }
    }

    /// <summary>Port of scipy's <c>_minimize_scalar_bounded</c> (Brent's bounded method), xatol 1e-5, 500 evaluations.</summary>
    private static double FminBound(Func<double, double> func, double x1, double x2, double xatol = 1e-5, int maxfun = 500)
    {
        if (x1 > x2) throw new ArgumentException("The lower bound exceeds the upper bound.");
        var sqrtEps = Math.Sqrt(2.2e-16);
        var goldenMean = 0.5 * (3.0 - Math.Sqrt(5.0));
        double a = x1, b = x2;
        var fulc = a + goldenMean * (b - a);
        double nfc = fulc, xf = fulc;
        double rat = 0.0, e = 0.0;
        var x = xf;
        var fx = func(x);
        var num = 1;
        double ffulc = fx, fnfc = fx;
        var xm = 0.5 * (a + b);
        var tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
        var tol2 = 2.0 * tol1;
        while (Math.Abs(xf - xm) > tol2 - 0.5 * (b - a))
        {
            var golden = true;
            if (Math.Abs(e) > tol1)
            {
                golden = false;
                var r = (xf - nfc) * (fx - ffulc);
                var q = (xf - fulc) * (fx - fnfc);
                var p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = Math.Abs(q);
                r = e;
                e = rat;
                if (Math.Abs(p) < Math.Abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf))
                {
                    rat = (p + 0.0) / q;
                    x = xf + rat;
                    if (x - a < tol2 || b - x < tol2)
                    {
                        var si0 = Math.Sign(xm - xf) + (xm - xf == 0 ? 1 : 0);
                        rat = tol1 * si0;
                    }
                }
                else golden = true;
            }
            if (golden)
            {
                e = xf >= xm ? a - xf : b - xf;
                rat = goldenMean * e;
            }
            var si = Math.Sign(rat) + (rat == 0 ? 1 : 0);
            x = xf + si * Math.Max(Math.Abs(rat), tol1);
            var fu = func(x);
            num++;
            if (fu <= fx)
            {
                if (x >= xf) a = xf; else b = xf;
                fulc = nfc; ffulc = fnfc;
                nfc = xf; fnfc = fx;
                xf = x; fx = fu;
            }
            else
            {
                if (x < xf) a = x; else b = x;
                if (fu <= fnfc || nfc == xf)
                {
                    fulc = nfc; ffulc = fnfc;
                    nfc = x; fnfc = fu;
                }
                else if (fu <= ffulc || fulc == xf || fulc == nfc)
                {
                    fulc = x; ffulc = fu;
                }
            }
            xm = 0.5 * (a + b);
            tol1 = sqrtEps * Math.Abs(xf) + xatol / 3.0;
            tol2 = 2.0 * tol1;
            if (num >= maxfun) break;
        }
        return xf;
    }

    private static double[] PostprocessWn(double[] wnIn, bool analog, double? fs)
    {
        var wn = analog ? wnIn.ToArray() : wnIn.Select(v => Math.Atan(v) * 2.0 / Math.PI).ToArray();
        if (fs is double f) for (var i = 0; i < wn.Length; i++) wn[i] = wn[i] * f / 2;
        return wn;
    }

    /// <summary>
    /// Lowest Butterworth order meeting the specification, and its natural frequency
    /// (<c>scipy.signal.buttord</c>). Wn has one element for low/high-pass and two for band filters.
    /// </summary>
    /// <param name="wp">Passband edge(s) (normalised to Nyquist unless <paramref name="fs"/>; rad/s when analog).</param>
    /// <param name="ws">Stopband edge(s).</param>
    /// <param name="gpass">Maximum passband loss (dB).</param>
    /// <param name="gstop">Minimum stopband attenuation (dB).</param>
    /// <param name="analog">Analog specification.</param>
    /// <param name="fs">Sampling frequency.</param>
    public static (int N, double[] Wn) Buttord(CriticalFrequencies wp, CriticalFrequencies ws, double gpass, double gstop, bool analog = false, double? fs = null)
    {
        var (_, _, type, passb, _, nat) = OrdSetup(wp, ws, gpass, gstop, analog, fs, OrdKind.Butter);
        var gs = Math.Pow(10, 0.1 * Math.Abs(gstop));
        var gp = Math.Pow(10, 0.1 * Math.Abs(gpass));
        var ord = (int)Math.Ceiling(Math.Log10((gs - 1.0) / (gp - 1.0)) / (2 * Math.Log10(nat)));
        var w0 = ord == 0 ? 1.0 : Math.Pow(gp - 1.0, -1.0 / (2.0 * ord));
        double[] wn;
        switch (type)
        {
            case 1: wn = passb.Select(v => w0 * v).ToArray(); break;
            case 2: wn = passb.Select(v => v / w0).ToArray(); break;
            case 3:
                {
                    var discr = Math.Sqrt(Math.Pow(passb[1] - passb[0], 2) + 4 * w0 * w0 * passb[0] * passb[1]);
                    var a = Math.Abs((passb[1] - passb[0] + discr) / (2 * w0));
                    var b = Math.Abs((passb[1] - passb[0] - discr) / (2 * w0));
                    wn = [Math.Min(a, b), Math.Max(a, b)];
                    break;
                }
            default:
                {
                    var d = passb[1] - passb[0];
                    double Wn(double w) => Math.Abs(-w * d / 2.0 + Math.Sqrt(w * w / 4.0 * d * d + passb[0] * passb[1]));
                    var a = Wn(-w0);
                    var b = Wn(w0);
                    wn = [Math.Min(a, b), Math.Max(a, b)];
                    break;
                }
        }
        return (ord, PostprocessWn(wn, analog, fs));
    }

    /// <summary>Lowest Chebyshev type I order meeting the specification (<c>scipy.signal.cheb1ord</c>).</summary>
    /// <inheritdoc cref="Buttord" path="/param"/>
    public static (int N, double[] Wn) Cheb1ord(CriticalFrequencies wp, CriticalFrequencies ws, double gpass, double gstop, bool analog = false, double? fs = null)
    {
        var (_, _, _, passb, _, nat) = OrdSetup(wp, ws, gpass, gstop, analog, fs, OrdKind.Cheby);
        var gs = Math.Pow(10, 0.1 * Math.Abs(gstop));
        var gp = Math.Pow(10, 0.1 * Math.Abs(gpass));
        var v = Math.Acosh(Math.Sqrt((gs - 1.0) / (gp - 1.0)));
        var ord = (int)Math.Ceiling(v / Math.Acosh(nat));
        return (ord, PostprocessWn(passb, analog, fs));
    }

    /// <summary>Lowest Chebyshev type II order meeting the specification (<c>scipy.signal.cheb2ord</c>).</summary>
    /// <inheritdoc cref="Buttord" path="/param"/>
    public static (int N, double[] Wn) Cheb2ord(CriticalFrequencies wp, CriticalFrequencies ws, double gpass, double gstop, bool analog = false, double? fs = null)
    {
        var (_, _, type, passb, _, nat) = OrdSetup(wp, ws, gpass, gstop, analog, fs, OrdKind.Cheby);
        var gs = Math.Pow(10, 0.1 * Math.Abs(gstop));
        var gp = Math.Pow(10, 0.1 * Math.Abs(gpass));
        var v = Math.Acosh(Math.Sqrt((gs - 1.0) / (gp - 1.0)));
        var ord = (int)Math.Ceiling(v / Math.Acosh(nat));
        var newFreq = 1.0 / Math.Cosh(1.0 / ord * v);
        double[] natv;
        switch (type)
        {
            case 1: natv = passb.Select(x => x / newFreq).ToArray(); break;
            case 2: natv = passb.Select(x => x * newFreq).ToArray(); break;
            case 3:
                {
                    var n0 = newFreq / 2.0 * (passb[0] - passb[1]) + Math.Sqrt(newFreq * newFreq * Math.Pow(passb[1] - passb[0], 2) / 4.0 + passb[1] * passb[0]);
                    natv = [n0, passb[1] * passb[0] / n0];
                    break;
                }
            default:
                {
                    var n0 = 1.0 / (2.0 * newFreq) * (passb[0] - passb[1]) + Math.Sqrt(Math.Pow(passb[1] - passb[0], 2) / (4.0 * newFreq * newFreq) + passb[1] * passb[0]);
                    natv = [n0, passb[0] * passb[1] / n0];
                    break;
                }
        }
        return (ord, PostprocessWn(natv, analog, fs));
    }

    /// <summary>Lowest elliptic order meeting the specification (<c>scipy.signal.ellipord</c>).</summary>
    /// <inheritdoc cref="Buttord" path="/param"/>
    public static (int N, double[] Wn) Ellipord(CriticalFrequencies wp, CriticalFrequencies ws, double gpass, double gstop, bool analog = false, double? fs = null)
    {
        var (_, _, _, passb, _, nat) = OrdSetup(wp, ws, gpass, gstop, analog, fs, OrdKind.Ellip);
        var arg1Sq = ExpM1(Math.Log(10) * (0.1 * gpass)) / ExpM1(Math.Log(10) * (0.1 * gstop));
        var arg0 = 1.0 / nat;
        var d00 = SignalElliptic.K(arg0 * arg0);
        var d01 = SignalElliptic.Km1(arg0 * arg0);
        var d10 = SignalElliptic.K(arg1Sq);
        var d11 = SignalElliptic.Km1(arg1Sq);
        var ord = (int)Math.Ceiling(d00 * d11 / (d01 * d10));
        return (ord, PostprocessWn(passb, analog, fs));
    }
}
