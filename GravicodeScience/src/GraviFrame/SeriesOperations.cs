namespace Gravicode.Science.GraviFrame;

/// <summary>Helpers that work across <see cref="Series"/> instances regardless of their type.</summary>
public static class SeriesOperations
{
    /// <summary>
    /// Concatenates columns end to end. All pieces must share a type; the result takes the
    /// first piece's name.
    /// </summary>
    public static Series Concat(IReadOnlyList<Series> pieces)
    {
        if (pieces.Count == 0) throw new ArgumentException("Nothing to concatenate.", nameof(pieces));
        var kind = pieces[0].DataType;
        var name = pieces[0].Name;
        var total = pieces.Sum(p => p.Length);

        foreach (var p in pieces)
            if (p.DataType != kind)
                throw new ArgumentException($"Cannot concatenate a {kind} column with a {p.DataType} one.");

        switch (kind)
        {
            case DataType.Numeric:
            {
                var values = new double[total];
                var offset = 0;
                foreach (var p in pieces)
                {
                    Array.Copy(((NumericSeries)p).Values, 0, values, offset, p.Length);
                    offset += p.Length;
                }
                return new NumericSeries(name, values);
            }
            case DataType.Text:
            {
                var values = new string?[total];
                var offset = 0;
                foreach (var p in pieces)
                {
                    for (var i = 0; i < p.Length; i++) values[offset + i] = (string?)p.GetValue(i);
                    offset += p.Length;
                }
                return new TextSeries(name, values);
            }
            case DataType.Boolean:
            {
                var values = new bool?[total];
                var offset = 0;
                foreach (var p in pieces)
                {
                    for (var i = 0; i < p.Length; i++) values[offset + i] = (bool?)p.GetValue(i);
                    offset += p.Length;
                }
                return new BooleanSeries(name, values);
            }
            default:
            {
                var values = new DateTime?[total];
                var offset = 0;
                foreach (var p in pieces)
                {
                    for (var i = 0; i < p.Length; i++) values[offset + i] = (DateTime?)p.GetValue(i);
                    offset += p.Length;
                }
                return new DateTimeSeries(name, values);
            }
        }
    }

    /// <summary>Builds an all-missing column of the given type and length.</summary>
    public static Series Missing(string name, DataType kind, int length) => kind switch
    {
        DataType.Numeric => new NumericSeries(name, length),
        DataType.Text => new TextSeries(name, new string?[length]),
        DataType.Boolean => new BooleanSeries(name, new bool?[length]),
        _ => new DateTimeSeries(name, new DateTime?[length]),
    };

    /// <summary>
    /// Reorders a column, using -1 in <paramref name="indices"/> to mean "insert a missing value".
    /// Outer joins rely on this.
    /// </summary>
    public static Series TakeAllowingMissing(Series series, IReadOnlyList<int> indices)
    {
        switch (series)
        {
            case NumericSeries n:
            {
                var values = new double[indices.Count];
                for (var i = 0; i < indices.Count; i++)
                    values[i] = indices[i] < 0 ? double.NaN : n[indices[i]];
                return new NumericSeries(series.Name, values);
            }
            case TextSeries t:
            {
                var values = new string?[indices.Count];
                for (var i = 0; i < indices.Count; i++)
                    values[i] = indices[i] < 0 ? null : t[indices[i]];
                return new TextSeries(series.Name, values);
            }
            case BooleanSeries b:
            {
                var values = new bool?[indices.Count];
                for (var i = 0; i < indices.Count; i++)
                    values[i] = indices[i] < 0 ? null : b[indices[i]];
                return new BooleanSeries(series.Name, values);
            }
            default:
            {
                var d = (DateTimeSeries)series;
                var values = new DateTime?[indices.Count];
                for (var i = 0; i < indices.Count; i++)
                    values[i] = indices[i] < 0 ? null : d[indices[i]];
                return new DateTimeSeries(series.Name, values);
            }
        }
    }
}
