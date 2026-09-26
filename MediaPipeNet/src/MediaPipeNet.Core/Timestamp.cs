using System.Diagnostics;
using System.Globalization;

namespace MediaPipeNet;

/// <summary>
/// A packet timestamp in microseconds, mirroring <c>mediapipe::Timestamp</c>. Timestamps order the
/// packets that flow through a graph; every stream carries strictly increasing timestamps.
/// </summary>
/// <remarks>
/// Besides ordinary values, a few sentinels exist: <see cref="Unset"/>, <see cref="PreStream"/>,
/// <see cref="Min"/>, <see cref="Max"/>, <see cref="PostStream"/> and <see cref="Done"/>.
/// </remarks>
[DebuggerDisplay("{ToString(),nq}")]
public readonly record struct Timestamp : IComparable<Timestamp>
{
    /// <summary>The raw value in microseconds.</summary>
    public long Value { get; }

    /// <summary>Creates a timestamp from a raw microsecond value.</summary>
    public Timestamp(long value) => Value = value;

    /// <summary>No timestamp has been assigned.</summary>
    public static Timestamp Unset { get; } = new(long.MinValue);

    /// <summary>Precedes every regular timestamp; used for header-like packets.</summary>
    public static Timestamp PreStream { get; } = new(long.MinValue + 2);

    /// <summary>The smallest regular timestamp.</summary>
    public static Timestamp Min { get; } = new(long.MinValue + 3);

    /// <summary>The largest regular timestamp.</summary>
    public static Timestamp Max { get; } = new(long.MaxValue - 3);

    /// <summary>Follows every regular timestamp; used for summary packets.</summary>
    public static Timestamp PostStream { get; } = new(long.MaxValue - 2);

    /// <summary>A stream with this bound will never produce another packet.</summary>
    public static Timestamp Done { get; } = new(long.MaxValue);

    /// <summary>True for timestamps that ordinary data packets may carry.</summary>
    public bool IsRangeValue => Value >= Min.Value && Value <= Max.Value;

    /// <summary>True when the timestamp is one of the special sentinels.</summary>
    public bool IsSpecialValue => !IsRangeValue;

    /// <summary>The timestamp converted to milliseconds.</summary>
    public double Milliseconds => Value / 1000.0;

    /// <summary>The timestamp converted to seconds.</summary>
    public double Seconds => Value / 1_000_000.0;

    /// <summary>Creates a timestamp from milliseconds.</summary>
    public static Timestamp FromMilliseconds(long milliseconds) => new(checked(milliseconds * 1000));

    /// <summary>Creates a timestamp from seconds.</summary>
    public static Timestamp FromSeconds(double seconds) => new((long)Math.Round(seconds * 1_000_000.0));

    /// <summary>Creates a timestamp from a <see cref="TimeSpan"/>.</summary>
    public static Timestamp FromTimeSpan(TimeSpan time) => new(time.Ticks / 10);

    /// <summary>The smallest timestamp strictly greater than this one (saturating at <see cref="Done"/>).</summary>
    public Timestamp NextAllowedInStream() => Value >= Max.Value ? Done : new Timestamp(Value + 1);

    /// <inheritdoc />
    public int CompareTo(Timestamp other) => Value.CompareTo(other.Value);

    /// <summary>Compares two timestamps.</summary>
    public static bool operator <(Timestamp a, Timestamp b) => a.Value < b.Value;

    /// <summary>Compares two timestamps.</summary>
    public static bool operator >(Timestamp a, Timestamp b) => a.Value > b.Value;

    /// <summary>Compares two timestamps.</summary>
    public static bool operator <=(Timestamp a, Timestamp b) => a.Value <= b.Value;

    /// <summary>Compares two timestamps.</summary>
    public static bool operator >=(Timestamp a, Timestamp b) => a.Value >= b.Value;

    /// <summary>Returns the larger of two timestamps.</summary>
    public static Timestamp Maximum(Timestamp a, Timestamp b) => a.Value >= b.Value ? a : b;

    /// <summary>Returns the smaller of two timestamps.</summary>
    public static Timestamp Minimum(Timestamp a, Timestamp b) => a.Value <= b.Value ? a : b;

    /// <inheritdoc />
    public override string ToString()
    {
        if (this == Unset) return "Timestamp::Unset";
        if (this == PreStream) return "Timestamp::PreStream";
        if (this == Min) return "Timestamp::Min";
        if (this == Max) return "Timestamp::Max";
        if (this == PostStream) return "Timestamp::PostStream";
        if (this == Done) return "Timestamp::Done";
        return Value.ToString(CultureInfo.InvariantCulture) + "us";
    }
}
