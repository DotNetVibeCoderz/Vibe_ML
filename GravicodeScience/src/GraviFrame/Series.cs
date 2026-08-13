namespace Gravicode.Science.GraviFrame;

/// <summary>The logical element type of a <see cref="Series"/>.</summary>
public enum DataType
{
    /// <summary>Double precision numbers; missing values are <see cref="double.NaN"/>.</summary>
    Numeric,

    /// <summary>Strings; missing values are <c>null</c>.</summary>
    Text,

    /// <summary>Three-valued booleans.</summary>
    Boolean,

    /// <summary>Timestamps.</summary>
    DateTime,
}

/// <summary>
/// One named, strongly typed column of a <see cref="DataFrame"/>.
/// </summary>
/// <remarks>
/// Columns are typed rather than boxed: <see cref="NumericSeries"/> owns a bare
/// <c>double[]</c>, which is what lets column arithmetic hand the buffer straight to GraviNum's
/// SIMD kernels instead of walking an <c>object[]</c>. The abstract surface here is only what
/// the frame itself needs - identity, length, missing-ness and reordering.
/// </remarks>
public abstract class Series(string name, int length)
{
    /// <summary>Column name.</summary>
    public string Name { get; internal set; } = name;

    /// <summary>Number of rows.</summary>
    public int Length { get; } = length;

    /// <summary>Logical element type.</summary>
    public abstract DataType DataType { get; }

    /// <summary>Reads a value as an object; missing values come back as <c>null</c>.</summary>
    public abstract object? GetValue(int index);

    /// <summary>True when the value at <paramref name="index"/> is missing.</summary>
    public abstract bool IsMissing(int index);

    /// <summary>Reorders (and possibly repeats or drops) rows.</summary>
    public abstract Series Take(IReadOnlyList<int> indices);

    /// <summary>A copy of this column under a new name.</summary>
    public abstract Series Rename(string name);

    /// <summary>Number of missing values.</summary>
    public int MissingCount
    {
        get
        {
            var count = 0;
            for (var i = 0; i < Length; i++) if (IsMissing(i)) count++;
            return count;
        }
    }

    /// <summary>Number of present values.</summary>
    public int Count => Length - MissingCount;

    /// <summary>Renders every value for display, using <c>NaN</c> for missing entries.</summary>
    public string[] ToStringArray()
    {
        var result = new string[Length];
        for (var i = 0; i < Length; i++)
            result[i] = IsMissing(i) ? "NaN" : GetValue(i)?.ToString() ?? "NaN";
        return result;
    }

    /// <summary>Distinct values in first-seen order.</summary>
    public IReadOnlyList<object?> Unique()
    {
        var seen = new HashSet<object>();
        var result = new List<object?>();
        var sawNull = false;
        for (var i = 0; i < Length; i++)
        {
            var v = GetValue(i);
            if (v is null)
            {
                if (!sawNull) { sawNull = true; result.Add(null); }
                continue;
            }
            if (seen.Add(v)) result.Add(v);
        }
        return result;
    }

    /// <summary>Counts occurrences of each distinct value, most frequent first.</summary>
    public IReadOnlyList<(object? Value, int Count)> ValueCounts()
    {
        var counts = new Dictionary<object, int>();
        var nulls = 0;
        for (var i = 0; i < Length; i++)
        {
            var v = GetValue(i);
            if (v is null) { nulls++; continue; }
            counts.TryGetValue(v, out var c);
            counts[v] = c + 1;
        }

        var result = counts.Select(kv => ((object?)kv.Key, kv.Value)).ToList();
        if (nulls > 0) result.Add((null, nulls));
        return result.OrderByDescending(t => t.Item2).ToList();
    }

    /// <summary>Row indices where <paramref name="predicate"/> holds.</summary>
    public int[] IndicesWhere(Func<object?, bool> predicate)
    {
        var result = new List<int>();
        for (var i = 0; i < Length; i++) if (predicate(GetValue(i))) result.Add(i);
        return result.ToArray();
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name} ({DataType}, {Length} rows, {MissingCount} missing)";
}

/// <summary>A column of strings.</summary>
public sealed class TextSeries : Series
{
    private readonly string?[] _values;

    /// <summary>Wraps <paramref name="values"/> without copying.</summary>
    public TextSeries(string name, string?[] values) : base(name, values.Length) => _values = values;

    /// <inheritdoc />
    public override DataType DataType => DataType.Text;

    /// <summary>The raw values.</summary>
    public ReadOnlySpan<string?> Values => _values;

    /// <summary>Reads or writes a value.</summary>
    public string? this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    /// <inheritdoc />
    public override object? GetValue(int index) => _values[index];

    /// <inheritdoc />
    public override bool IsMissing(int index) => _values[index] is null;

    /// <inheritdoc />
    public override Series Take(IReadOnlyList<int> indices)
    {
        var result = new string?[indices.Count];
        for (var i = 0; i < indices.Count; i++) result[i] = _values[indices[i]];
        return new TextSeries(Name, result);
    }

    /// <inheritdoc />
    public override Series Rename(string name) => new TextSeries(name, (string?[])_values.Clone());

    /// <summary>Applies a function to every non-missing value.</summary>
    public TextSeries Apply(Func<string, string> f)
    {
        var result = new string?[Length];
        for (var i = 0; i < Length; i++) result[i] = _values[i] is null ? null : f(_values[i]!);
        return new TextSeries(Name, result);
    }

    /// <summary>Encodes the distinct values as integer codes, returning the codes and the categories.</summary>
    public (NumericSeries Codes, string[] Categories) Factorize()
    {
        var categories = new List<string>();
        var lookup = new Dictionary<string, int>();
        var codes = new double[Length];

        for (var i = 0; i < Length; i++)
        {
            var v = _values[i];
            if (v is null) { codes[i] = double.NaN; continue; }
            if (!lookup.TryGetValue(v, out var code))
            {
                code = categories.Count;
                lookup[v] = code;
                categories.Add(v);
            }
            codes[i] = code;
        }
        return (new NumericSeries(Name, codes), categories.ToArray());
    }
}

/// <summary>A column of three-valued booleans.</summary>
public sealed class BooleanSeries : Series
{
    private readonly bool?[] _values;

    /// <summary>Wraps <paramref name="values"/> without copying.</summary>
    public BooleanSeries(string name, bool?[] values) : base(name, values.Length) => _values = values;

    /// <summary>Builds a column from a non-nullable array.</summary>
    public BooleanSeries(string name, bool[] values)
        : base(name, values.Length) => _values = values.Select(v => (bool?)v).ToArray();

    /// <inheritdoc />
    public override DataType DataType => DataType.Boolean;

    /// <summary>Reads or writes a value.</summary>
    public bool? this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    /// <inheritdoc />
    public override object? GetValue(int index) => _values[index];

    /// <inheritdoc />
    public override bool IsMissing(int index) => _values[index] is null;

    /// <inheritdoc />
    public override Series Take(IReadOnlyList<int> indices)
    {
        var result = new bool?[indices.Count];
        for (var i = 0; i < indices.Count; i++) result[i] = _values[indices[i]];
        return new BooleanSeries(Name, result);
    }

    /// <inheritdoc />
    public override Series Rename(string name) => new BooleanSeries(name, (bool?[])_values.Clone());

    /// <summary>Number of true values.</summary>
    public int TrueCount => _values.Count(v => v == true);

    /// <summary>The mask as a plain array, treating missing as false.</summary>
    public bool[] ToMask() => _values.Select(v => v == true).ToArray();

    /// <summary>Converts to 1.0 / 0.0 so the column can take part in arithmetic.</summary>
    public NumericSeries ToNumeric()
        => new(Name, _values.Select(v => v is null ? double.NaN : (v.Value ? 1.0 : 0.0)).ToArray());
}

/// <summary>A column of timestamps.</summary>
public sealed class DateTimeSeries : Series
{
    private readonly DateTime?[] _values;

    /// <summary>Wraps <paramref name="values"/> without copying.</summary>
    public DateTimeSeries(string name, DateTime?[] values) : base(name, values.Length) => _values = values;

    /// <summary>Builds a column from a non-nullable array.</summary>
    public DateTimeSeries(string name, DateTime[] values)
        : base(name, values.Length) => _values = values.Select(v => (DateTime?)v).ToArray();

    /// <inheritdoc />
    public override DataType DataType => DataType.DateTime;

    /// <summary>Reads or writes a value.</summary>
    public DateTime? this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    /// <inheritdoc />
    public override object? GetValue(int index) => _values[index];

    /// <inheritdoc />
    public override bool IsMissing(int index) => _values[index] is null;

    /// <inheritdoc />
    public override Series Take(IReadOnlyList<int> indices)
    {
        var result = new DateTime?[indices.Count];
        for (var i = 0; i < indices.Count; i++) result[i] = _values[indices[i]];
        return new DateTimeSeries(Name, result);
    }

    /// <inheritdoc />
    public override Series Rename(string name) => new DateTimeSeries(name, (DateTime?[])_values.Clone());

    /// <summary>Earliest timestamp present.</summary>
    public DateTime? Min() => _values.Where(v => v.HasValue).Select(v => v!.Value).DefaultIfEmpty().Min();

    /// <summary>Latest timestamp present.</summary>
    public DateTime? Max() => _values.Where(v => v.HasValue).Select(v => v!.Value).DefaultIfEmpty().Max();

    /// <summary>Extracts a numeric component (year, month, day of week, ...) as its own column.</summary>
    public NumericSeries Component(Func<DateTime, double> selector, string name)
        => new(name, _values.Select(v => v is null ? double.NaN : selector(v.Value)).ToArray());

    /// <summary>The year of each timestamp.</summary>
    public NumericSeries Year() => Component(d => d.Year, $"{Name}_year");

    /// <summary>The month of each timestamp.</summary>
    public NumericSeries Month() => Component(d => d.Month, $"{Name}_month");

    /// <summary>The day of each timestamp.</summary>
    public NumericSeries Day() => Component(d => d.Day, $"{Name}_day");

    /// <summary>The day of week of each timestamp, Sunday = 0.</summary>
    public NumericSeries DayOfWeek() => Component(d => (int)d.DayOfWeek, $"{Name}_dayofweek");
}
