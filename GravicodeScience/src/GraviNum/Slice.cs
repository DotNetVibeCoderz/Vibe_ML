namespace Gravicode.Science.GraviNum;

/// <summary>
/// A one-axis selector, the C# spelling of NumPy's <c>a[start:stop:step]</c>.
/// A slice either picks a single index (which drops the axis) or a strided range (which keeps it).
/// </summary>
public readonly struct Slice
{
    private Slice(int start, int? stop, int step, bool isIndex)
    {
        Start = start;
        Stop = stop;
        Step = step;
        IsIndex = isIndex;
    }

    /// <summary>Inclusive start. Negative values count from the end.</summary>
    public int Start { get; }

    /// <summary>Exclusive stop, or <c>null</c> to run to the end of the axis.</summary>
    public int? Stop { get; }

    /// <summary>Stride; may be negative to walk the axis backwards.</summary>
    public int Step { get; }

    /// <summary>True when this selector picks one element and removes the axis.</summary>
    public bool IsIndex { get; }

    /// <summary>The whole axis, <c>a[:]</c>.</summary>
    public static Slice All => new(0, null, 1, false);

    /// <summary>A single index, <c>a[i]</c>. The axis is dropped from the result.</summary>
    public static Slice At(int index) => new(index, null, 1, true);

    /// <summary>A half-open range, <c>a[start:stop]</c>.</summary>
    public static Slice Range(int start, int? stop = null, int step = 1)
    {
        if (step == 0) throw new ArgumentException("Slice step cannot be zero.", nameof(step));
        return new Slice(start, stop, step, false);
    }

    /// <summary>Everything from <paramref name="start"/> onwards, <c>a[start:]</c>.</summary>
    public static Slice From(int start) => Range(start);

    /// <summary>Everything before <paramref name="stop"/>, <c>a[:stop]</c>.</summary>
    public static Slice To(int stop) => Range(0, stop);

    /// <summary>Reverses the axis, <c>a[::-1]</c>.</summary>
    public static Slice Reversed => new(0, null, -1, false);

    /// <summary>An index selector; lets callers write <c>arr.Slice(1, Slice.All)</c>.</summary>
    public static implicit operator Slice(int index) => At(index);

    /// <summary>Converts a <see cref="System.Range"/>, so <c>arr.Slice(1..3)</c> works.</summary>
    public static implicit operator Slice(Range range)
    {
        var start = range.Start.IsFromEnd ? -range.Start.Value : range.Start.Value;
        int? stop = range.End.IsFromEnd
            ? (range.End.Value == 0 ? null : -range.End.Value)
            : range.End.Value;
        return Range(start, stop);
    }

    /// <summary>
    /// Resolves this selector against an axis of <paramref name="length"/> elements,
    /// returning the first offset, the number of elements produced and the stride in elements.
    /// </summary>
    public (int Offset, int Count, int Step) Resolve(int length)
    {
        if (IsIndex)
        {
            var i = Start < 0 ? Start + length : Start;
            if (i < 0 || i >= length)
                throw new ArgumentOutOfRangeException(nameof(length), $"Index {Start} out of range for axis of length {length}.");
            return (i, 1, 1);
        }

        var step = Step;
        int start, stop;

        if (step > 0)
        {
            start = Start < 0 ? Start + length : Start;
            start = Math.Clamp(start, 0, length);
            stop = Stop is null ? length : (Stop.Value < 0 ? Stop.Value + length : Stop.Value);
            stop = Math.Clamp(stop, 0, length);
            var count = stop > start ? (stop - start + step - 1) / step : 0;
            return (start, count, step);
        }
        else
        {
            start = Start < 0 ? Start + length : Start;
            // A positive-start default of 0 means "the last element" when walking backwards.
            if (Start == 0 && Stop is null) start = length - 1;
            start = Math.Clamp(start, 0, length - 1);
            stop = Stop is null ? -1 : (Stop.Value < 0 ? Stop.Value + length : Stop.Value);
            var count = start > stop ? (start - stop - step - 1) / -step : 0;
            return (start, count, step);
        }
    }
}
