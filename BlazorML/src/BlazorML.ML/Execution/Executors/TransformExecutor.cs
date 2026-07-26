using System.Globalization;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;

namespace BlazorML.ML.Execution.Executors;

/// <summary>
/// The twelve data-preparation modules. All of them work directly on <see cref="TabularData"/>
/// rather than on an ML.NET estimator chain, because these transforms have to run and show a
/// preview whether or not a trainer is ever attached.
/// </summary>
public sealed class TransformExecutor : IModuleExecutor
{
    /// <summary>
    /// Joins composite group keys. ASCII unit separator, because it cannot occur in data that
    /// came through a CSV or JSON reader — a comma or pipe could, and would merge distinct groups.
    /// </summary>
    private const char KeySeparator = '\u001F';

    public bool CanExecute(string moduleId) => moduleId.StartsWith("tf.", StringComparison.Ordinal);

    public Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx)
    {
        var result = ctx.Module.Id switch
        {
            "tf.selectColumns" => SelectColumns(ctx),
            "tf.editMetadata" => EditMetadata(ctx),
            "tf.cleanMissing" => CleanMissing(ctx),
            "tf.removeDuplicates" => RemoveDuplicates(ctx),
            "tf.filterRows" => FilterRows(ctx),
            "tf.splitData" => SplitData(ctx),
            "tf.join" => Join(ctx),
            "tf.groupBy" => GroupBy(ctx),
            "tf.normalize" => Normalize(ctx),
            "tf.oneHot" => EncodeCategories(ctx),
            "tf.featurizeText" => FeaturizeText(ctx),
            "tf.featureSelection" => SelectBestFeatures(ctx),
            _ => throw new NotSupportedException($"Unknown transform '{ctx.Module.Id}'.")
        };

        return Task.FromResult(result);
    }

    private static object?[] SelectColumns(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var chosen = ctx.ParamList("columns");
        var keep = ctx.Param("mode") != "drop";

        if (chosen.Count == 0)
        {
            return [input];
        }

        var output = input.Clone();
        var wanted = chosen.ToHashSet(StringComparer.OrdinalIgnoreCase);

        var toRemove = output.Columns
            .Select(c => c.Name)
            .Where(name => keep ? !wanted.Contains(name) : wanted.Contains(name))
            .ToList();

        foreach (var name in toRemove)
        {
            output.RemoveColumn(name);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"{(keep ? "Kept" : "Dropped")} {chosen.Count} columns; {output.ColumnCount} remain.");
        return [output];
    }

    private static object?[] EditMetadata(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var column = ctx.RequireParam("column");
        var output = input.Clone();

        var index = output.IndexOf(column);
        if (index < 0)
        {
            throw new InvalidOperationException($"Column '{column}' is not in the incoming data.");
        }

        var newName = ctx.Param("newName");
        var dataType = ctx.Param("dataType");

        var current = output.Columns[index];
        var name = string.IsNullOrWhiteSpace(newName) ? current.Name : newName;
        var type = dataType is null or "keep" || !Enum.TryParse<ColumnDataType>(dataType, out var parsed)
            ? current.DataType
            : parsed;

        output.Columns[index] = new DataColumn(name, type);

        // Changing the declared type has to change the stored values too, or the trainer will
        // read strings out of a column the schema calls numeric.
        if (type != current.DataType)
        {
            foreach (var row in output.Rows)
            {
                row[index] = Coerce(row[index], type);
            }
        }

        return [output];
    }

    private static object? Coerce(object? value, ColumnDataType type)
    {
        if (TabularData.IsBlank(value))
        {
            return null;
        }

        var text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;

        return type switch
        {
            ColumnDataType.Numeric => TabularData.ToNumber(value),
            ColumnDataType.Boolean => bool.TryParse(text, out var b) ? b : text is "1" or "yes" or "ya",
            ColumnDataType.DateTime => DateTime.TryParse(text, CultureInfo.InvariantCulture,
                DateTimeStyles.None, out var d) ? d : null,
            _ => text
        };
    }

    private static object?[] CleanMissing(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var output = input.Clone();
        var strategy = ctx.Param("strategy") ?? "mean";

        var columns = ctx.ParamList("columns");
        if (columns.Count == 0)
        {
            columns = output.Columns.Select(c => c.Name).ToList();
        }

        if (strategy == "dropRow")
        {
            var indices = columns.Select(output.IndexOf).Where(i => i >= 0).ToList();
            var before = output.RowCount;
            output.Rows.RemoveAll(row => indices.Any(i => i < row.Length && TabularData.IsBlank(row[i])));
            ctx.Log(Core.Domain.LogLevel.Info, $"Dropped {before - output.RowCount:N0} rows with gaps.");
            return [output];
        }

        if (strategy == "dropColumn")
        {
            var dropped = 0;
            foreach (var name in columns)
            {
                var i = output.IndexOf(name);
                if (i >= 0 && output.Rows.Any(r => i < r.Length && TabularData.IsBlank(r[i])))
                {
                    output.RemoveColumn(name);
                    dropped++;
                }
            }

            ctx.Log(Core.Domain.LogLevel.Info, $"Dropped {dropped} columns that had gaps.");
            return [output];
        }

        var filled = 0;
        foreach (var name in columns)
        {
            var index = output.IndexOf(name);
            if (index < 0)
            {
                continue;
            }

            var replacement = ComputeReplacement(output, index, strategy, ctx.Param("constant"));

            foreach (var row in output.Rows)
            {
                if (index < row.Length && TabularData.IsBlank(row[index]))
                {
                    row[index] = replacement;
                    filled++;
                }
            }
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Filled {filled:N0} empty cells using the column {strategy}.");
        return [output];
    }

    private static object? ComputeReplacement(TabularData table, int index, string strategy, string? constant)
    {
        if (strategy == "constant")
        {
            return constant;
        }

        var values = table.Rows
            .Where(r => index < r.Length && !TabularData.IsBlank(r[index]))
            .Select(r => r[index])
            .ToList();

        if (values.Count == 0)
        {
            return null;
        }

        if (strategy == "mode")
        {
            return values.GroupBy(v => Convert.ToString(v, CultureInfo.InvariantCulture))
                .OrderByDescending(g => g.Count())
                .First().First();
        }

        var numbers = values.Select(TabularData.ToNumber).Where(n => n.HasValue).Select(n => n!.Value).ToList();
        if (numbers.Count == 0)
        {
            // A non-numeric column cannot have a mean; the most frequent value is the sane stand-in.
            return values.GroupBy(v => Convert.ToString(v, CultureInfo.InvariantCulture))
                .OrderByDescending(g => g.Count())
                .First().First();
        }

        if (strategy == "median")
        {
            numbers.Sort();
            return numbers[numbers.Count / 2];
        }

        return numbers.Average();
    }

    private static object?[] RemoveDuplicates(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var output = input.CloneSchema();

        var keyColumns = ctx.ParamList("keyColumns");
        var indices = keyColumns.Count == 0
            ? Enumerable.Range(0, input.ColumnCount).ToList()
            : keyColumns.Select(input.IndexOf).Where(i => i >= 0).ToList();

        var keepLast = ctx.Param("keep") == "last";
        var seen = new Dictionary<string, int>();

        foreach (var row in input.Rows)
        {
            var key = string.Join(KeySeparator, indices.Select(i =>
                i < row.Length ? Convert.ToString(row[i], CultureInfo.InvariantCulture) : string.Empty));

            if (seen.TryGetValue(key, out var position))
            {
                if (keepLast)
                {
                    output.Rows[position] = (object?[])row.Clone();
                }

                continue;
            }

            seen[key] = output.Rows.Count;
            output.Rows.Add((object?[])row.Clone());
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Removed {input.RowCount - output.RowCount:N0} duplicate rows.");
        return [output];
    }

    private static object?[] FilterRows(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var column = ctx.RequireParam("column");
        var op = ctx.Param("op") ?? "gt";
        var value = ctx.Param("value");

        var index = input.IndexOf(column);
        if (index < 0)
        {
            throw new InvalidOperationException($"Column '{column}' is not in the incoming data.");
        }

        var output = input.CloneSchema();
        var comparand = TabularData.ToNumber(value);

        foreach (var row in input.Rows)
        {
            var cell = index < row.Length ? row[index] : null;
            if (Matches(cell, op, value, comparand))
            {
                output.Rows.Add((object?[])row.Clone());
            }
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Kept {output.RowCount:N0} of {input.RowCount:N0} rows.");
        return [output];
    }

    private static bool Matches(object? cell, string op, string? value, double? comparand)
    {
        if (op == "notEmpty")
        {
            return !TabularData.IsBlank(cell);
        }

        var text = Convert.ToString(cell, CultureInfo.InvariantCulture) ?? string.Empty;

        if (op == "contains")
        {
            return text.Contains(value ?? string.Empty, StringComparison.OrdinalIgnoreCase);
        }

        var number = TabularData.ToNumber(cell);
        if (number.HasValue && comparand.HasValue)
        {
            return op switch
            {
                "eq" => Math.Abs(number.Value - comparand.Value) < 1e-9,
                "ne" => Math.Abs(number.Value - comparand.Value) >= 1e-9,
                "gt" => number.Value > comparand.Value,
                "gte" => number.Value >= comparand.Value,
                "lt" => number.Value < comparand.Value,
                "lte" => number.Value <= comparand.Value,
                _ => false
            };
        }

        // Fall back to text comparison so a filter on a category column still behaves sensibly.
        return op switch
        {
            "eq" => string.Equals(text, value, StringComparison.OrdinalIgnoreCase),
            "ne" => !string.Equals(text, value, StringComparison.OrdinalIgnoreCase),
            _ => false
        };
    }

    private static object?[] SplitData(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var fraction = Math.Clamp(ctx.ParamDouble("fraction", 0.8), 0.05, 0.95);
        var seed = ctx.ParamInt("randomSeed", 42);
        var stratifyBy = ctx.Param("stratifyColumn");

        var left = input.CloneSchema();
        var right = input.CloneSchema();
        var random = new Random(seed);

        if (!string.IsNullOrWhiteSpace(stratifyBy) && input.Has(stratifyBy))
        {
            // Split within each class so both halves keep the original balance.
            var index = input.IndexOf(stratifyBy);
            var groups = input.Rows.GroupBy(r =>
                Convert.ToString(index < r.Length ? r[index] : null, CultureInfo.InvariantCulture) ?? string.Empty)
                .ToList();

            // Stratifying by a continuous column is a silent disaster: nearly every value is
            // its own group of one, every group rounds entirely into the first output, and the
            // second output comes back almost empty. The run then "succeeds" and reports
            // metrics computed from a handful of rows. Refuse instead.
            var averageGroup = (double)input.RowCount / Math.Max(1, groups.Count);

            if (averageGroup < 4)
            {
                throw new InvalidOperationException(
                    $"'{stratifyBy}' has {groups.Count:N0} distinct values across {input.RowCount:N0} rows, " +
                    "so it cannot be used to stratify — there are not enough rows per value to divide. " +
                    "Stratify by a category column such as the class label, or leave it empty for a plain random split.");
            }

            foreach (var group in groups)
            {
                var shuffled = group.OrderBy(_ => random.Next()).ToList();
                var cut = (int)Math.Round(shuffled.Count * fraction);

                for (var i = 0; i < shuffled.Count; i++)
                {
                    (i < cut ? left : right).Rows.Add((object?[])shuffled[i].Clone());
                }
            }
        }
        else
        {
            var shuffled = input.Rows.OrderBy(_ => random.Next()).ToList();
            var cut = (int)Math.Round(shuffled.Count * fraction);

            for (var i = 0; i < shuffled.Count; i++)
            {
                (i < cut ? left : right).Rows.Add((object?[])shuffled[i].Clone());
            }
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Split into {left.RowCount:N0} and {right.RowCount:N0} rows.");
        return [left, right];
    }

    private static object?[] Join(ModuleExecutionContext ctx)
    {
        var left = ctx.RequireTable(0, "Left");
        var right = ctx.RequireTable(1, "Right");

        var leftKey = ctx.RequireParam("leftKey");
        var rightKey = ctx.RequireParam("rightKey");
        var kind = ctx.Param("kind") ?? "inner";

        var leftIndex = left.IndexOf(leftKey);
        var rightIndex = right.IndexOf(rightKey);

        if (leftIndex < 0 || rightIndex < 0)
        {
            throw new InvalidOperationException(
                $"Join needs '{leftKey}' on the left and '{rightKey}' on the right.");
        }

        var output = left.CloneSchema();
        var rightColumnMap = new Dictionary<int, int>();

        for (var c = 0; c < right.ColumnCount; c++)
        {
            if (c == rightIndex)
            {
                continue;
            }

            rightColumnMap[c] = output.AddColumn(right.Columns[c].Name);
        }

        var lookup = right.Rows.ToLookup(r =>
            Convert.ToString(rightIndex < r.Length ? r[rightIndex] : null, CultureInfo.InvariantCulture) ?? string.Empty);

        var matchedRightKeys = new HashSet<string>();

        foreach (var leftRow in left.Rows)
        {
            var key = Convert.ToString(leftIndex < leftRow.Length ? leftRow[leftIndex] : null,
                CultureInfo.InvariantCulture) ?? string.Empty;

            var matches = lookup[key].ToList();

            if (matches.Count == 0)
            {
                if (kind is "left" or "full")
                {
                    output.Rows.Add(Widen(leftRow, output.ColumnCount));
                }

                continue;
            }

            matchedRightKeys.Add(key);

            foreach (var rightRow in matches)
            {
                var row = Widen(leftRow, output.ColumnCount);
                foreach (var (source, destination) in rightColumnMap)
                {
                    row[destination] = source < rightRow.Length ? rightRow[source] : null;
                }

                output.Rows.Add(row);
            }
        }

        if (kind == "full")
        {
            foreach (var group in lookup.Where(g => !matchedRightKeys.Contains(g.Key)))
            {
                foreach (var rightRow in group)
                {
                    var row = output.NewRow();
                    row[leftIndex] = rightIndex < rightRow.Length ? rightRow[rightIndex] : null;

                    foreach (var (source, destination) in rightColumnMap)
                    {
                        row[destination] = source < rightRow.Length ? rightRow[source] : null;
                    }

                    output.Rows.Add(row);
                }
            }
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Join produced {output.RowCount:N0} rows.");
        return [output];
    }

    private static object?[] Widen(object?[] row, int width)
    {
        var widened = new object?[width];
        Array.Copy(row, widened, Math.Min(row.Length, width));
        return widened;
    }

    private static object?[] GroupBy(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var groupColumns = ctx.ParamList("groupBy");
        var aggregateColumn = ctx.Param("aggregateColumn");
        var aggregate = ctx.Param("aggregate") ?? "mean";

        if (groupColumns.Count == 0)
        {
            throw new InvalidOperationException("Choose at least one column to group by.");
        }

        var groupIndices = groupColumns.Select(input.IndexOf).Where(i => i >= 0).ToList();
        var valueIndex = aggregateColumn is null ? -1 : input.IndexOf(aggregateColumn);

        var output = new TabularData();
        foreach (var i in groupIndices)
        {
            output.AddColumn(input.Columns[i].Name, input.Columns[i].DataType);
        }

        var resultName = aggregate == "count" ? "count" : $"{aggregate}_{aggregateColumn}";
        output.AddColumn(resultName, ColumnDataType.Numeric);

        var groups = input.Rows.GroupBy(r => string.Join(KeySeparator, groupIndices.Select(i =>
            Convert.ToString(i < r.Length ? r[i] : null, CultureInfo.InvariantCulture) ?? string.Empty)));

        foreach (var group in groups)
        {
            var row = output.NewRow();
            var keys = group.Key.Split(KeySeparator);

            for (var i = 0; i < groupIndices.Count && i < keys.Length; i++)
            {
                row[i] = keys[i];
            }

            var numbers = valueIndex < 0
                ? []
                : group.Select(r => TabularData.ToNumber(valueIndex < r.Length ? r[valueIndex] : null))
                    .Where(n => n.HasValue).Select(n => n!.Value).ToList();

            row[^1] = aggregate switch
            {
                "count" => group.Count(),
                "sum" => numbers.Count == 0 ? 0 : numbers.Sum(),
                "min" => numbers.Count == 0 ? null : numbers.Min(),
                "max" => numbers.Count == 0 ? null : numbers.Max(),
                _ => numbers.Count == 0 ? null : numbers.Average()
            };

            output.Rows.Add(row);
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Collapsed {input.RowCount:N0} rows into {output.RowCount:N0} groups.");
        return [output];
    }

    private static object?[] Normalize(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var output = input.Clone();
        output.InferTypes();

        var method = ctx.Param("method") ?? "minmax";
        var columns = ctx.ParamList("columns");

        if (columns.Count == 0)
        {
            columns = output.Columns.Where(c => c.DataType == ColumnDataType.Numeric)
                .Select(c => c.Name).ToList();
        }

        foreach (var name in columns)
        {
            var index = output.IndexOf(name);
            if (index < 0)
            {
                continue;
            }

            var values = output.Rows
                .Select(r => TabularData.ToNumber(index < r.Length ? r[index] : null))
                .ToList();

            var present = values.Where(v => v.HasValue).Select(v => v!.Value).ToList();
            if (present.Count == 0)
            {
                continue;
            }

            var min = present.Min();
            var max = present.Max();
            var mean = present.Average();
            var stdDev = present.Count > 1
                ? Math.Sqrt(present.Sum(v => (v - mean) * (v - mean)) / (present.Count - 1))
                : 1;

            var bins = ctx.ParamInt("bins", 10);

            for (var r = 0; r < output.Rows.Count; r++)
            {
                var value = values[r];
                if (!value.HasValue)
                {
                    continue;
                }

                output.Rows[r][index] = method switch
                {
                    "zscore" => stdDev < 1e-12 ? 0 : (value.Value - mean) / stdDev,
                    "logscale" => Math.Log(Math.Max(value.Value, 1e-9) + 1),
                    "binning" => Math.Abs(max - min) < 1e-12
                        ? 0
                        : Math.Min(bins - 1, (int)((value.Value - min) / ((max - min) / bins))),
                    _ => Math.Abs(max - min) < 1e-12 ? 0 : (value.Value - min) / (max - min)
                };
            }

            output.Columns[index] = output.Columns[index] with { DataType = ColumnDataType.Numeric };
        }

        ctx.Log(Core.Domain.LogLevel.Info, $"Rescaled {columns.Count} columns using {method}.");
        return [output];
    }

    private static object?[] EncodeCategories(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var output = input.Clone();
        var method = ctx.Param("method") ?? "oneHot";
        var columns = ctx.ParamList("columns");

        if (columns.Count == 0)
        {
            output.InferTypes();
            columns = output.Columns.Where(c => c.DataType == ColumnDataType.Categorical)
                .Select(c => c.Name).ToList();
        }

        foreach (var name in columns)
        {
            var index = output.IndexOf(name);
            if (index < 0)
            {
                continue;
            }

            var values = output.Rows
                .Select(r => Convert.ToString(index < r.Length ? r[index] : null, CultureInfo.InvariantCulture) ?? string.Empty)
                .ToList();

            var distinct = values.Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(v => v, StringComparer.OrdinalIgnoreCase).ToList();

            if (method == "ordinal")
            {
                var map = distinct.Select((v, i) => (v, i))
                    .ToDictionary(x => x.v, x => x.i, StringComparer.OrdinalIgnoreCase);

                for (var r = 0; r < output.Rows.Count; r++)
                {
                    output.Rows[r][index] = map.GetValueOrDefault(values[r], -1);
                }

                output.Columns[index] = output.Columns[index] with { DataType = ColumnDataType.Numeric };
                continue;
            }

            if (method == "hash")
            {
                var width = 1 << Math.Clamp(ctx.ParamInt("hashBits", 16), 4, 20);
                for (var r = 0; r < output.Rows.Count; r++)
                {
                    output.Rows[r][index] = Math.Abs(values[r].GetHashCode()) % width;
                }

                output.Columns[index] = output.Columns[index] with { DataType = ColumnDataType.Numeric };
                continue;
            }

            // One-hot: a very high cardinality column would explode the table, so it is refused
            // rather than quietly producing thousands of columns.
            if (distinct.Count > 200)
            {
                throw new InvalidOperationException(
                    $"'{name}' has {distinct.Count} distinct values. One-hot encoding it would add that many " +
                    "columns — use hashing or ordinal encoding instead.");
            }

            var newIndices = distinct.ToDictionary(
                v => v,
                v => output.AddColumn($"{name}_{v}", ColumnDataType.Numeric),
                StringComparer.OrdinalIgnoreCase);

            for (var r = 0; r < output.Rows.Count; r++)
            {
                foreach (var target in newIndices.Values)
                {
                    output.Rows[r][target] = 0;
                }

                if (newIndices.TryGetValue(values[r], out var hit))
                {
                    output.Rows[r][hit] = 1;
                }
            }

            output.RemoveColumn(name);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Encoded {columns.Count} columns using {method}; the table now has {output.ColumnCount} columns.");
        return [output];
    }

    private static object?[] FeaturizeText(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        var column = ctx.RequireParam("column");
        var outputColumn = ctx.Param("outputColumn") ?? "TextFeatures";
        var wordGram = Math.Clamp(ctx.ParamInt("wordGramLength", 2), 1, 5);
        var removeStopWords = ctx.ParamBool("removeStopWords", true);

        var index = input.IndexOf(column);
        if (index < 0)
        {
            throw new InvalidOperationException($"Column '{column}' is not in the incoming data.");
        }

        // Build a vocabulary over the corpus, then emit a bag-of-n-grams count vector per row.
        var documents = input.Rows
            .Select(r => Tokenise(Convert.ToString(index < r.Length ? r[index] : null, CultureInfo.InvariantCulture),
                wordGram, removeStopWords))
            .ToList();

        var vocabulary = documents.SelectMany(d => d)
            .GroupBy(t => t)
            .Where(g => g.Count() > 1)
            .OrderByDescending(g => g.Count())
            .Take(512)
            .Select((g, i) => (Term: g.Key, Index: i))
            .ToDictionary(x => x.Term, x => x.Index);

        var output = input.Clone();
        var target = output.AddColumn(outputColumn, ColumnDataType.Text);

        for (var r = 0; r < output.Rows.Count; r++)
        {
            var counts = new int[vocabulary.Count];
            foreach (var token in documents[r])
            {
                if (vocabulary.TryGetValue(token, out var slot))
                {
                    counts[slot]++;
                }
            }

            output.Rows[r][target] = string.Join(' ', counts);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Built a {vocabulary.Count}-term vocabulary from '{column}'.");
        return [output];
    }

    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "this", "that",
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "pada", "adalah", "atau"
    };

    private static List<string> Tokenise(string? text, int maxGram, bool removeStopWords)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        var words = text.ToLowerInvariant()
            .Split([' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '"', '(', ')', '-', '/'],
                StringSplitOptions.RemoveEmptyEntries)
            .Where(w => !removeStopWords || !StopWords.Contains(w))
            .ToList();

        var tokens = new List<string>(words);

        for (var n = 2; n <= maxGram; n++)
        {
            for (var i = 0; i + n <= words.Count; i++)
            {
                tokens.Add(string.Join('_', words.Skip(i).Take(n)));
            }
        }

        return tokens;
    }

    private static object?[] SelectBestFeatures(ModuleExecutionContext ctx)
    {
        var input = ctx.RequireTable(0, "Dataset");
        input.InferTypes();

        var label = ctx.RequireParam("labelColumn");
        var topN = ctx.ParamInt("topN", 10);
        var method = ctx.Param("method") ?? "correlation";

        var labelIndex = input.IndexOf(label);
        if (labelIndex < 0)
        {
            throw new InvalidOperationException($"Label column '{label}' is not in the incoming data.");
        }

        var labelValues = input.Rows
            .Select(r => TabularData.ToNumber(labelIndex < r.Length ? r[labelIndex] : null) ?? 0)
            .ToList();

        var scores = new List<(string Column, double Score)>();

        for (var c = 0; c < input.ColumnCount; c++)
        {
            if (c == labelIndex || input.Columns[c].DataType != ColumnDataType.Numeric)
            {
                continue;
            }

            var values = input.Rows
                .Select(r => TabularData.ToNumber(c < r.Length ? r[c] : null) ?? 0)
                .ToList();

            var score = method switch
            {
                "variance" => Variance(values),
                "mutualInformation" => MutualInformation(values, labelValues),
                _ => Math.Abs(Correlation(values, labelValues))
            };

            scores.Add((input.Columns[c].Name, double.IsNaN(score) ? 0 : score));
        }

        var ranked = scores.OrderByDescending(s => s.Score).ToList();
        var keep = ranked.Take(topN).Select(s => s.Column).ToHashSet(StringComparer.OrdinalIgnoreCase);

        var output = input.Clone();
        foreach (var name in input.Columns.Select(c => c.Name).ToList())
        {
            if (!keep.Contains(name) && !string.Equals(name, label, StringComparison.OrdinalIgnoreCase))
            {
                output.RemoveColumn(name);
            }
        }

        var scoreTable = TabularData.WithColumns("feature", "score");
        foreach (var (column, score) in ranked)
        {
            scoreTable.AddRow(column, score);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Kept the {keep.Count} strongest of {ranked.Count} numeric features.");

        return [output, new MetricsBundle
        {
            Title = "Feature scores",
            Table = scoreTable,
            Values = ranked.ToDictionary(s => s.Column, s => s.Score)
        }];
    }

    /// <summary>
    /// Mutual information between a numeric feature and the label, both discretised into equal-width
    /// bins. Unlike correlation this picks up non-linear relationships — a feature that matters only
    /// at its extremes scores near zero on correlation but high here.
    /// </summary>
    private static double MutualInformation(List<double> feature, List<double> label, int bins = 10)
    {
        var n = Math.Min(feature.Count, label.Count);
        if (n < 2)
        {
            return 0;
        }

        var featureBins = Discretise(feature, bins, n);
        var labelBins = Discretise(label, bins, n);

        var joint = new Dictionary<(int, int), int>();
        var featureCounts = new Dictionary<int, int>();
        var labelCounts = new Dictionary<int, int>();

        for (var i = 0; i < n; i++)
        {
            var key = (featureBins[i], labelBins[i]);
            joint[key] = joint.GetValueOrDefault(key) + 1;
            featureCounts[featureBins[i]] = featureCounts.GetValueOrDefault(featureBins[i]) + 1;
            labelCounts[labelBins[i]] = labelCounts.GetValueOrDefault(labelBins[i]) + 1;
        }

        var total = 0.0;
        foreach (var ((f, l), count) in joint)
        {
            var pJoint = (double)count / n;
            var pFeature = (double)featureCounts[f] / n;
            var pLabel = (double)labelCounts[l] / n;
            total += pJoint * Math.Log(pJoint / (pFeature * pLabel), 2);
        }

        return Math.Max(0, total);
    }

    private static int[] Discretise(List<double> values, int bins, int n)
    {
        var min = values.Take(n).Min();
        var max = values.Take(n).Max();
        var width = (max - min) / bins;

        var result = new int[n];
        for (var i = 0; i < n; i++)
        {
            result[i] = width < 1e-12 ? 0 : Math.Clamp((int)((values[i] - min) / width), 0, bins - 1);
        }

        return result;
    }

    private static double Variance(List<double> values)
    {
        if (values.Count < 2)
        {
            return 0;
        }

        var mean = values.Average();
        return values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
    }

    private static double Correlation(List<double> a, List<double> b)
    {
        var n = Math.Min(a.Count, b.Count);
        if (n < 2)
        {
            return 0;
        }

        var meanA = a.Take(n).Average();
        var meanB = b.Take(n).Average();

        double covariance = 0, varianceA = 0, varianceB = 0;
        for (var i = 0; i < n; i++)
        {
            var da = a[i] - meanA;
            var db = b[i] - meanB;
            covariance += da * db;
            varianceA += da * da;
            varianceB += db * db;
        }

        return varianceA < 1e-12 || varianceB < 1e-12 ? 0 : covariance / Math.Sqrt(varianceA * varianceB);
    }
}
