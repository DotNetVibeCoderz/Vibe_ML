namespace Gravicode.Science.GraviFrame;

/// <summary>How unmatched rows are treated by <see cref="Joins.Merge"/>.</summary>
public enum JoinKind
{
    /// <summary>Only rows whose key appears on both sides.</summary>
    Inner,

    /// <summary>Every left row; unmatched right columns are filled with missing values.</summary>
    Left,

    /// <summary>Every right row; unmatched left columns are filled with missing values.</summary>
    Right,

    /// <summary>Every row from both sides.</summary>
    Outer,
}

/// <summary>Relational joins between frames.</summary>
public static class Joins
{
    /// <summary>
    /// Joins two frames on a key column.
    /// </summary>
    /// <remarks>
    /// The right side is indexed into a hash table once, then the left side is scanned - so the
    /// cost is linear in the two row counts rather than quadratic. Duplicate keys on the right
    /// fan out into multiple output rows, matching SQL and pandas semantics. Columns that would
    /// collide take <paramref name="suffix"/>.
    /// </remarks>
    public static DataFrame Merge(DataFrame left, DataFrame right, string leftOn, string rightOn,
        JoinKind kind = JoinKind.Inner, string suffix = "_right")
    {
        var leftKey = left[leftOn];
        var rightKey = right[rightOn];

        // Index the right side by key signature.
        var rightIndex = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (var i = 0; i < right.RowCount; i++)
        {
            var key = Signature(rightKey, i);
            if (!rightIndex.TryGetValue(key, out var bucket)) rightIndex[key] = bucket = [];
            bucket.Add(i);
        }

        var leftRows = new List<int>();
        var rightRows = new List<int>();
        var matchedRight = new HashSet<int>();

        for (var i = 0; i < left.RowCount; i++)
        {
            var key = Signature(leftKey, i);
            if (rightIndex.TryGetValue(key, out var bucket))
            {
                foreach (var j in bucket)
                {
                    leftRows.Add(i);
                    rightRows.Add(j);
                    matchedRight.Add(j);
                }
            }
            else if (kind is JoinKind.Left or JoinKind.Outer)
            {
                leftRows.Add(i);
                rightRows.Add(-1);
            }
        }

        if (kind is JoinKind.Right or JoinKind.Outer)
        {
            for (var j = 0; j < right.RowCount; j++)
            {
                if (matchedRight.Contains(j)) continue;
                leftRows.Add(-1);
                rightRows.Add(j);
            }
        }

        var columns = new List<Series>();
        var leftNames = new HashSet<string>(left.ColumnNames, StringComparer.Ordinal);

        foreach (var column in left.Columns)
            columns.Add(SeriesOperations.TakeAllowingMissing(column, leftRows));

        foreach (var column in right.Columns)
        {
            // The right key column is redundant when it carries the same name as the left one.
            if (column.Name == rightOn && rightOn == leftOn) continue;

            var taken = SeriesOperations.TakeAllowingMissing(column, rightRows);
            if (leftNames.Contains(column.Name)) taken = taken.Rename(column.Name + suffix);
            columns.Add(taken);
        }

        // For right and outer joins the key can come from either side; fill in from the right.
        if (kind is JoinKind.Right or JoinKind.Outer && leftOn == rightOn)
            columns[left.ColumnNames.ToList().IndexOf(leftOn)] =
                CoalesceKey(leftKey, rightKey, leftRows, rightRows, leftOn);

        return new DataFrame(columns);
    }

    /// <summary>Joins on several key columns at once.</summary>
    public static DataFrame MergeOn(DataFrame left, DataFrame right, IReadOnlyList<string> keys,
        JoinKind kind = JoinKind.Inner, string suffix = "_right")
    {
        // Build a composite key column on each side and join on that, then drop it again.
        const string composite = "__gravi_join_key__";
        var leftWithKey = left.WithColumn(new TextSeries(composite, CompositeKeys(left, keys)));
        var rightWithKey = right.WithColumn(new TextSeries(composite, CompositeKeys(right, keys)));

        var joined = Merge(leftWithKey, rightWithKey, composite, composite, kind, suffix);
        return joined.Drop(composite);
    }

    private static string?[] CompositeKeys(DataFrame frame, IReadOnlyList<string> keys)
    {
        var columns = keys.Select(k => frame[k]).ToArray();
        var result = new string?[frame.RowCount];
        for (var i = 0; i < frame.RowCount; i++)
            result[i] = string.Join('\u001F', columns.Select(c => c.GetValue(i)?.ToString() ?? "\u0000"));
        return result;
    }

    private static Series CoalesceKey(Series leftKey, Series rightKey,
        List<int> leftRows, List<int> rightRows, string name)
    {
        var values = new string?[leftRows.Count];
        for (var i = 0; i < leftRows.Count; i++)
            values[i] = leftRows[i] >= 0
                ? leftKey.GetValue(leftRows[i])?.ToString()
                : rightKey.GetValue(rightRows[i])?.ToString();

        if (leftKey.DataType != DataType.Numeric) return new TextSeries(name, values);

        var numbers = values.Select(v => v is null ? double.NaN : double.Parse(v)).ToArray();
        return new NumericSeries(name, numbers);
    }

    private static string Signature(Series series, int row) => series.GetValue(row)?.ToString() ?? "\u0000";
}
