namespace Gravicode.Science.SciNet;

/// <summary>
/// A <c>[min, max]</c> condition for <see cref="Signal.FindPeaks"/> — scipy's "number, array or
/// (min, max) pair" argument. A plain number is a minimum; either bound may be omitted, and either
/// may be given per sample (an array the length of the signal, read at each peak).
/// </summary>
public sealed record PeakCondition(double? Min = null, double? Max = null, double[]? MinPerSample = null, double[]? MaxPerSample = null)
{
    /// <summary>A minimum.</summary>
    public static implicit operator PeakCondition(double min) => new(min);

    /// <summary>A (min, max) interval.</summary>
    public static implicit operator PeakCondition((double Min, double Max) interval) => new(interval.Min, interval.Max);

    /// <summary>A per-sample minimum.</summary>
    public static implicit operator PeakCondition(double[] min) => new(MinPerSample: min);

    /// <summary>Only a minimum.</summary>
    public static PeakCondition AtLeast(double min) => new(min);

    /// <summary>Only a maximum.</summary>
    public static PeakCondition AtMost(double max) => new(Max: max);

    /// <summary>Both bounds.</summary>
    public static PeakCondition Between(double min, double max) => new(min, max);

    /// <summary>Lower bound at sample <paramref name="peak"/>, or null.</summary>
    internal double? LowerAt(int peak, int n)
    {
        if (MinPerSample is null) return Min;
        if (MinPerSample.Length != n) throw new ArgumentException("array size of lower interval border must match x.");
        return MinPerSample[peak];
    }

    /// <summary>Upper bound at sample <paramref name="peak"/>, or null.</summary>
    internal double? UpperAt(int peak, int n)
    {
        if (MaxPerSample is null) return Max;
        if (MaxPerSample.Length != n) throw new ArgumentException("array size of upper interval border must match x.");
        return MaxPerSample[peak];
    }

    internal bool Accepts(double value, int peak, int n)
    {
        var lo = LowerAt(peak, n);
        var hi = UpperAt(peak, n);
        return (lo is null || lo.Value <= value) && (hi is null || value <= hi.Value);
    }
}

/// <summary>
/// The properties <see cref="Signal.FindPeaks"/> computed while filtering, one entry per returned
/// peak — scipy's <c>properties</c> dictionary. A property is null when the condition that
/// produces it was not requested.
/// </summary>
public sealed record PeakProperties
{
    /// <summary>Height of each peak (<c>height</c> given).</summary>
    public double[]? PeakHeights { get; init; }

    /// <summary>Vertical distance to the left neighbour (<c>threshold</c> given).</summary>
    public double[]? LeftThresholds { get; init; }

    /// <summary>Vertical distance to the right neighbour (<c>threshold</c> given).</summary>
    public double[]? RightThresholds { get; init; }

    /// <summary>Prominence (<c>prominence</c> or <c>width</c> given).</summary>
    public double[]? Prominences { get; init; }

    /// <summary>Left base of each prominence.</summary>
    public int[]? LeftBases { get; init; }

    /// <summary>Right base of each prominence.</summary>
    public int[]? RightBases { get; init; }

    /// <summary>Height at which each width was measured (<c>width</c> given).</summary>
    public double[]? WidthHeights { get; init; }

    /// <summary>Width at <c>relHeight</c> of the prominence.</summary>
    public double[]? Widths { get; init; }

    /// <summary>Interpolated left intersection point.</summary>
    public double[]? LeftIps { get; init; }

    /// <summary>Interpolated right intersection point.</summary>
    public double[]? RightIps { get; init; }

    /// <summary>Plateau size (<c>plateauSize</c> given).</summary>
    public int[]? PlateauSizes { get; init; }

    /// <summary>First sample of each plateau.</summary>
    public int[]? LeftEdges { get; init; }

    /// <summary>Last sample of each plateau.</summary>
    public int[]? RightEdges { get; init; }
}

/// <summary>Peaks found by <see cref="Signal.FindPeaks"/> and their properties.</summary>
public sealed record FindPeaksResult(int[] Peaks, PeakProperties Properties);
