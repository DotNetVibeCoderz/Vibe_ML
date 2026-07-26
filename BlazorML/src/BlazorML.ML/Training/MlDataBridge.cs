using System.Globalization;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BlazorML.ML.Training;

/// <summary>
/// Converts a <see cref="TabularData"/> into an ML.NET <see cref="IDataView"/> and back.
/// <para>
/// The route goes through a temporary CSV and a <see cref="TextLoader"/> built at runtime.
/// ML.NET's in-memory loaders want a compile-time row type, which the designer cannot provide —
/// columns are whatever the user's data happens to contain. Declaring the schema on a TextLoader
/// is the supported way to work with a shape only known at run time, and it is also what AutoML's
/// column inference expects.
/// </para>
/// </summary>
public sealed class MlDataBridge : IDisposable
{
    private readonly List<string> _tempFiles = new();

    /// <summary>Column names as ML.NET will see them, in table order.</summary>
    public List<string> ColumnNames { get; } = new();

    /// <summary>
    /// Builds a data view. <paramref name="labelColumn"/> and <paramref name="task"/> decide the
    /// label's declared type, which is what ML.NET keys its trainer validation off.
    /// </summary>
    public IDataView ToDataView(MLContext ml, TabularData table, string? labelColumn,
        MlTask task, out string csvPath)
    {
        table.InferTypes();

        csvPath = Path.Combine(Path.GetTempPath(), $"blazorml-{Guid.NewGuid():n}.csv");
        File.WriteAllText(csvPath, table.ToCsv());
        _tempFiles.Add(csvPath);

        var columns = new List<TextLoader.Column>();
        ColumnNames.Clear();

        for (var i = 0; i < table.Columns.Count; i++)
        {
            var column = table.Columns[i];
            var isLabel = labelColumn is not null &&
                          string.Equals(column.Name, labelColumn, StringComparison.OrdinalIgnoreCase);

            var kind = isLabel ? LabelKind(task, column.DataType) : ValueKind(column.DataType);
            columns.Add(new TextLoader.Column(Sanitise(column.Name), kind, i));
            ColumnNames.Add(Sanitise(column.Name));
        }

        var loader = ml.Data.CreateTextLoader(new TextLoader.Options
        {
            Columns = columns.ToArray(),
            HasHeader = true,
            Separators = [','],
            AllowQuoting = true,
            TrimWhitespace = true,
            MissingRealsAsNaNs = true
        });

        return loader.Load(csvPath);
    }

    /// <summary>
    /// The label type each task needs. Binary classification is the one that bites: ML.NET
    /// requires a Boolean label, and TextLoader parses both 0/1 and true/false into one.
    /// </summary>
    private static DataKind LabelKind(MlTask task, ColumnDataType dataType) => task switch
    {
        MlTask.BinaryClassification => DataKind.Boolean,
        MlTask.Regression or MlTask.Forecasting => DataKind.Single,
        MlTask.Recommendation => DataKind.Single,
        MlTask.MulticlassClassification => DataKind.String,
        _ => ValueKind(dataType)
    };

    private static DataKind ValueKind(ColumnDataType type) => type switch
    {
        ColumnDataType.Numeric => DataKind.Single,
        ColumnDataType.Boolean => DataKind.Boolean,
        _ => DataKind.String
    };

    /// <summary>
    /// ML.NET column names cannot carry characters that clash with its expression syntax, and a
    /// leading digit confuses the estimator chain. Mapped consistently so lookups still match.
    /// </summary>
    public static string Sanitise(string name)
    {
        var cleaned = new string(name.Select(c => char.IsLetterOrDigit(c) || c == '_' ? c : '_').ToArray());
        return char.IsDigit(cleaned.FirstOrDefault()) ? "c_" + cleaned : cleaned;
    }

    /// <summary>
    /// Adds the label column back, filled with a placeholder, when rows arrive without it.
    /// <para>
    /// The fitted pipeline copies the label into a "Label" column, so its input schema demands
    /// that column exists — even at inference time, when the label is precisely the thing being
    /// predicted. Scoring genuinely unlabelled data (a live API call, or a production batch)
    /// would otherwise fail with a schema error. The placeholder is never read: the trainer's
    /// prediction depends only on the feature columns.
    /// </para>
    /// </summary>
    public static TabularData EnsureLabelColumn(TabularData table, string? labelColumn, MlTask task)
    {
        if (string.IsNullOrWhiteSpace(labelColumn) || table.Has(labelColumn))
        {
            return table;
        }

        var withLabel = table.Clone();
        var index = withLabel.AddColumn(labelColumn, task switch
        {
            MlTask.MulticlassClassification => ColumnDataType.Categorical,
            MlTask.BinaryClassification => ColumnDataType.Numeric,
            _ => ColumnDataType.Numeric
        });

        // Values the TextLoader can parse into whatever type the task declares for the label.
        var placeholder = task == MlTask.MulticlassClassification ? "?" : "0";

        foreach (var row in withLabel.Rows)
        {
            row[index] = placeholder;
        }

        return withLabel;
    }

    /// <summary>Reads a data view back into a table so downstream modules can keep working on rows.</summary>
    public static TabularData FromDataView(IDataView view, int rowLimit = 0)
    {
        var table = new TabularData();
        var schema = view.Schema;

        var readable = schema
            .Where(c => !c.IsHidden && IsReadable(c.Type))
            .ToList();

        foreach (var column in readable)
        {
            table.AddColumn(column.Name);
        }

        using var cursor = view.GetRowCursor(readable);
        var getters = readable.Select(c => BuildGetter(cursor, c)).ToList();

        while (cursor.MoveNext())
        {
            var row = table.NewRow();
            for (var i = 0; i < getters.Count; i++)
            {
                row[i] = getters[i]();
            }

            table.Rows.Add(row);

            if (rowLimit > 0 && table.RowCount >= rowLimit)
            {
                break;
            }
        }

        table.InferTypes();
        return table;
    }

    /// <summary>
    /// Decided by the CLR type behind the column rather than by which <see cref="DataViewType"/>
    /// subclass it happens to be. Keying on the subclass silently dropped key-typed columns —
    /// which is what K-Means emits as its cluster index, so clustering results never reached the
    /// scored table at all.
    /// </summary>
    private static bool IsReadable(DataViewType type)
    {
        if (type is VectorDataViewType vector)
        {
            return vector.ItemType.RawType == typeof(float) || vector.ItemType.RawType == typeof(double);
        }

        return type.RawType == typeof(float) || type.RawType == typeof(double)
            || type.RawType == typeof(bool) || type.RawType == typeof(ReadOnlyMemory<char>)
            || type.RawType == typeof(int) || type.RawType == typeof(uint)
            || type.RawType == typeof(long) || type.RawType == typeof(ulong)
            || type.RawType == typeof(short) || type.RawType == typeof(ushort)
            || type.RawType == typeof(byte) || type.RawType == typeof(sbyte);
    }

    private static Func<object?> BuildGetter(DataViewRowCursor cursor, DataViewSchema.Column column)
    {
        var type = column.Type;

        // Vectors are flattened to a compact string: the preview grid and CSV export both need
        // something printable, and keeping the values beats dropping the column. Both element
        // widths are handled — assuming float threw on the spike detector, which emits doubles.
        if (type is VectorDataViewType { ItemType.RawType: var item } && item == typeof(float))
        {
            var getter = cursor.GetGetter<VBuffer<float>>(column);
            return () =>
            {
                VBuffer<float> buffer = default;
                getter(ref buffer);
                return string.Join(' ', buffer.DenseValues().Select(v => v.ToString("G6", CultureInfo.InvariantCulture)));
            };
        }

        if (type is VectorDataViewType { ItemType.RawType: var wide } && wide == typeof(double))
        {
            var getter = cursor.GetGetter<VBuffer<double>>(column);
            return () =>
            {
                VBuffer<double> buffer = default;
                getter(ref buffer);
                return string.Join(' ', buffer.DenseValues().Select(v => v.ToString("G6", CultureInfo.InvariantCulture)));
            };
        }

        if (type.RawType == typeof(float))
        {
            var getter = cursor.GetGetter<float>(column);
            return () => { float v = 0; getter(ref v); return float.IsNaN(v) ? null : v; };
        }

        if (type.RawType == typeof(double))
        {
            var getter = cursor.GetGetter<double>(column);
            return () => { double v = 0; getter(ref v); return double.IsNaN(v) ? null : v; };
        }

        if (type.RawType == typeof(bool))
        {
            var getter = cursor.GetGetter<bool>(column);
            return () => { var v = false; getter(ref v); return v; };
        }

        if (type.RawType == typeof(int))
        {
            var getter = cursor.GetGetter<int>(column);
            return () => { var v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(uint))
        {
            var getter = cursor.GetGetter<uint>(column);
            return () => { uint v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(long))
        {
            var getter = cursor.GetGetter<long>(column);
            return () => { long v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(ulong))
        {
            var getter = cursor.GetGetter<ulong>(column);
            return () => { ulong v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(short))
        {
            var getter = cursor.GetGetter<short>(column);
            return () => { short v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(ushort))
        {
            var getter = cursor.GetGetter<ushort>(column);
            return () => { ushort v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(byte))
        {
            var getter = cursor.GetGetter<byte>(column);
            return () => { byte v = 0; getter(ref v); return v; };
        }

        if (type.RawType == typeof(sbyte))
        {
            var getter = cursor.GetGetter<sbyte>(column);
            return () => { sbyte v = 0; getter(ref v); return v; };
        }

        var text = cursor.GetGetter<ReadOnlyMemory<char>>(column);
        return () => { ReadOnlyMemory<char> v = default; text(ref v); return v.ToString(); };
    }

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try
            {
                if (File.Exists(file))
                {
                    File.Delete(file);
                }
            }
            catch (IOException)
            {
                // A temp file still held open by a lazy cursor is not worth failing a run over;
                // the OS clears the temp directory eventually.
            }
        }

        _tempFiles.Clear();
    }
}
