namespace Gravicode.Science.SciNet;

/// <summary>
/// A sample in which some observations are only known to lie in a range, scipy's
/// <c>stats.CensoredData</c>: uncensored values, left-censored values (the true value is ≤ the
/// recorded one), right-censored values (≥ the recorded one) and interval-censored pairs [low, high].
/// </summary>
/// <remarks>
/// As in scipy, rows of <c>interval</c> are normalised on construction: a row with equal ends is an
/// uncensored value, a row whose low end is −∞ is left-censored at its high end, and a row whose high
/// end is +∞ is right-censored at its low end. Only genuinely finite, non-degenerate rows remain in
/// <see cref="Interval"/>.
/// </remarks>
public sealed class CensoredData
{
    /// <summary>Builds censored data from its four kinds of observation (any may be null or empty).</summary>
    public CensoredData(double[]? uncensored = null, double[]? left = null, double[]? right = null, double[,]? interval = null)
    {
        Check(uncensored, nameof(uncensored));
        Check(left, nameof(left));
        Check(right, nameof(right));
        var u = new List<double>(uncensored ?? []);
        var l = new List<double>(left ?? []);
        var r = new List<double>(right ?? []);
        var iv = new List<(double, double)>();
        if (interval != null)
        {
            if (interval.GetLength(1) != 2) throw new ArgumentException("interval must have shape (m, 2).", nameof(interval));
            for (var i = 0; i < interval.GetLength(0); i++)
            {
                double a = interval[i, 0], b = interval[i, 1];
                if (double.IsNaN(a) || double.IsNaN(b)) throw new ArgumentException("interval must not contain nan.", nameof(interval));
                if (double.IsInfinity(a) && double.IsInfinity(b)) throw new ArgumentException("In each row in interval, both values must not be infinite.", nameof(interval));
                if (a > b) throw new ArgumentException("In each row of interval, the left value must not exceed the right value.", nameof(interval));
                if (a == b) u.Add(a);
                else if (double.IsInfinity(a)) l.Add(b);
                else if (double.IsInfinity(b)) r.Add(a);
                else iv.Add((a, b));
            }
        }
        Uncensored = [.. u];
        Left = [.. l];
        Right = [.. r];
        Interval = new double[iv.Count, 2];
        for (var i = 0; i < iv.Count; i++) { Interval[i, 0] = iv[i].Item1; Interval[i, 1] = iv[i].Item2; }
    }

    private static void Check(double[]? a, string name)
    {
        if (a == null) return;
        foreach (var x in a)
            if (!double.IsFinite(x)) throw new ArgumentException($"{name} must contain only finite values.", name);
    }

    /// <summary>Exactly observed values.</summary>
    public double[] Uncensored { get; }

    /// <summary>Left-censored values: the observation is at most this.</summary>
    public double[] Left { get; }

    /// <summary>Right-censored values: the observation is at least this.</summary>
    public double[] Right { get; }

    /// <summary>Interval-censored observations, one [low, high] row each (finite, low &lt; high).</summary>
    public double[,] Interval { get; }

    /// <summary>Total number of observations, scipy's <c>len(data)</c>.</summary>
    public int Count => Uncensored.Length + Left.Length + Right.Length + Interval.GetLength(0);

    /// <summary>Number of censored observations (left, right and interval).</summary>
    public int NumCensored() => Left.Length + Right.Length + Interval.GetLength(0);

    /// <summary>Right-censored data from values and flags, scipy's <c>CensoredData.right_censored</c>.</summary>
    public static CensoredData RightCensored(double[] x, bool[] censored)
    {
        var (u, c) = Split(x, censored);
        return new CensoredData(uncensored: u, right: c);
    }

    /// <summary>Left-censored data from values and flags, scipy's <c>CensoredData.left_censored</c>.</summary>
    public static CensoredData LeftCensored(double[] x, bool[] censored)
    {
        var (u, c) = Split(x, censored);
        return new CensoredData(uncensored: u, left: c);
    }

    /// <summary>Interval-censored data from paired bounds (±∞ allowed), scipy's <c>CensoredData.interval_censored</c>.</summary>
    public static CensoredData IntervalCensored(double[] low, double[] high)
    {
        ArgumentNullException.ThrowIfNull(low);
        ArgumentNullException.ThrowIfNull(high);
        if (low.Length != high.Length) throw new ArgumentException("low and high must have the same length.", nameof(high));
        var iv = new double[low.Length, 2];
        for (var i = 0; i < low.Length; i++) { iv[i, 0] = low[i]; iv[i, 1] = high[i]; }
        return new CensoredData(interval: iv);
    }

    private static (double[] U, double[] C) Split(double[] x, bool[] censored)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(censored);
        if (x.Length != censored.Length) throw new ArgumentException("x and censored must have the same length.", nameof(censored));
        foreach (var v in x) if (!double.IsFinite(v)) throw new ArgumentException("x must not contain nan or inf.", nameof(x));
        var u = new List<double>();
        var c = new List<double>();
        for (var i = 0; i < x.Length; i++) (censored[i] ? c : u).Add(x[i]);
        return ([.. u], [.. c]);
    }

    /// <summary>
    /// scipy's <c>_uncensor</c>: every observation as one number (censored ones at their recorded end,
    /// intervals at their midpoint) — only good as a starting point for a fit.
    /// </summary>
    public double[] Uncensor()
    {
        var r = new double[Count];
        var o = 0;
        foreach (var x in Uncensored) r[o++] = x;
        foreach (var x in Left) r[o++] = x;
        foreach (var x in Right) r[o++] = x;
        for (var i = 0; i < Interval.GetLength(0); i++) r[o++] = 0.5 * (Interval[i, 0] + Interval[i, 1]);
        return r;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var parts = new List<string> { $"{Uncensored.Length} not censored" };
        if (Left.Length > 0) parts.Add($"{Left.Length} left-censored");
        if (Right.Length > 0) parts.Add($"{Right.Length} right-censored");
        if (Interval.GetLength(0) > 0) parts.Add($"{Interval.GetLength(0)} interval-censored");
        return $"CensoredData({Count} values: {string.Join(", ", parts)})";
    }
}
