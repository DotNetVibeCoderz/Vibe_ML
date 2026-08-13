using System.Data;
using System.Data.Common;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>
/// Reads query results into a <see cref="DataFrame"/> through any ADO.NET provider.
/// </summary>
/// <remarks>
/// <para>
/// Written against <see cref="System.Data.Common"/> rather than any particular driver, so it works
/// with SQL Server, PostgreSQL, SQLite, MySQL or anything else that ships a
/// <see cref="DbConnection"/> — and takes no package dependency to do it. The caller brings the
/// connection, which also means connection strings, pooling and credentials stay where they belong.
/// </para>
/// <para>
/// Column types come from the provider rather than being inferred, which is the main reason to
/// prefer this over exporting to CSV and reading that back: the database already knows that a
/// column is a date and not a string that looks like one.
/// </para>
/// </remarks>
public static class SqlReader
{
    /// <summary>
    /// Runs a query and materialises the whole result set.
    /// </summary>
    /// <param name="connection">An open — or openable — connection. It is left in the state it was found.</param>
    /// <param name="sql">The query text.</param>
    /// <param name="parameters">
    /// Values bound as command parameters, keyed by name without the provider's prefix.
    /// </param>
    /// <param name="commandTimeout">Seconds before the command is abandoned; <c>null</c> keeps the provider default.</param>
    /// <remarks>
    /// <b>Pass values through <paramref name="parameters"/>, never by building the SQL string.</b>
    /// Interpolating user input into the query text is how SQL injection happens, and the fact that
    /// it works in testing is exactly what makes it dangerous.
    /// </remarks>
    public static DataFrame Read(DbConnection connection, string sql,
        IReadOnlyDictionary<string, object?>? parameters = null, int? commandTimeout = null)
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentException.ThrowIfNullOrWhiteSpace(sql);

        // Leave the connection as we found it: a caller managing its own lifetime should not have
        // it closed underneath them, and one that handed us a closed connection should not be left
        // with an open one.
        var wasClosed = connection.State == ConnectionState.Closed;
        if (wasClosed) connection.Open();

        try
        {
            using var command = connection.CreateCommand();
            command.CommandText = sql;
            if (commandTimeout is not null) command.CommandTimeout = commandTimeout.Value;

            if (parameters is not null)
                foreach (var (name, value) in parameters)
                {
                    var parameter = command.CreateParameter();
                    parameter.ParameterName = name;
                    parameter.Value = value ?? DBNull.Value;
                    command.Parameters.Add(parameter);
                }

            using var reader = command.ExecuteReader();
            return FromReader(reader);
        }
        finally
        {
            if (wasClosed) connection.Close();
        }
    }

    /// <summary>
    /// Materialises an already-executing reader.
    /// </summary>
    /// <remarks>
    /// Split out so a caller who has built their own command — with a transaction, a cancellation
    /// token, or provider-specific options — can still get a frame out of it.
    /// </remarks>
    public static DataFrame FromReader(IDataReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        var count = reader.FieldCount;
        var names = new string[count];
        var types = new DataType[count];
        var columns = new List<object?>[count];

        for (var i = 0; i < count; i++)
        {
            names[i] = reader.GetName(i) is { Length: > 0 } name ? name : $"column_{i}";
            types[i] = Map(reader.GetFieldType(i));
            columns[i] = [];
        }

        // Duplicate names are legal in SQL — "SELECT a.id, b.id" — but not in a frame, so they are
        // suffixed rather than silently overwriting each other.
        Deduplicate(names);

        while (reader.Read())
            for (var i = 0; i < count; i++)
                columns[i].Add(reader.IsDBNull(i) ? null : reader.GetValue(i));

        var series = new List<Series>(count);
        for (var i = 0; i < count; i++) series.Add(Build(names[i], columns[i], types[i]));

        return new DataFrame(series);
    }

    /// <summary>Maps a CLR type reported by the provider onto a frame column type.</summary>
    /// <remarks>
    /// Anything unrecognised — GUIDs, byte arrays, provider-specific spatial types — becomes text
    /// via <c>ToString</c>. That loses fidelity, and it is better than dropping the column or
    /// throwing on a result set the caller mostly wanted.
    /// </remarks>
    private static DataType Map(Type type)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;

        return Type.GetTypeCode(underlying) switch
        {
            TypeCode.Byte or TypeCode.SByte or TypeCode.Int16 or TypeCode.UInt16
                or TypeCode.Int32 or TypeCode.UInt32 or TypeCode.Int64 or TypeCode.UInt64
                or TypeCode.Single or TypeCode.Double or TypeCode.Decimal => DataType.Numeric,
            TypeCode.Boolean => DataType.Boolean,
            TypeCode.DateTime => DataType.DateTime,
            _ when underlying == typeof(DateTimeOffset) => DataType.DateTime,
            _ => DataType.Text,
        };
    }

    private static Series Build(string name, List<object?> values, DataType type)
    {
        switch (type)
        {
            case DataType.Numeric:
            {
                var result = new double[values.Count];
                for (var i = 0; i < values.Count; i++)
                    result[i] = values[i] is null
                        ? double.NaN
                        : Convert.ToDouble(values[i], System.Globalization.CultureInfo.InvariantCulture);
                return new NumericSeries(name, result);
            }
            case DataType.Boolean:
            {
                var result = new bool?[values.Count];
                for (var i = 0; i < values.Count; i++)
                    result[i] = values[i] is null ? null : Convert.ToBoolean(values[i]);
                return new BooleanSeries(name, result);
            }
            case DataType.DateTime:
            {
                var result = new DateTime?[values.Count];
                for (var i = 0; i < values.Count; i++)
                    result[i] = values[i] switch
                    {
                        null => null,
                        DateTimeOffset offset => offset.UtcDateTime,
                        var v => Convert.ToDateTime(v, System.Globalization.CultureInfo.InvariantCulture),
                    };
                return new DateTimeSeries(name, result);
            }
            default:
            {
                var result = new string?[values.Count];
                for (var i = 0; i < values.Count; i++) result[i] = values[i]?.ToString();
                return new TextSeries(name, result);
            }
        }
    }

    private static void Deduplicate(string[] names)
    {
        var seen = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var i = 0; i < names.Length; i++)
        {
            if (seen.TryAdd(names[i], 1)) continue;

            var suffix = seen[names[i]];
            string candidate;
            do { candidate = $"{names[i]}_{suffix++}"; } while (seen.ContainsKey(candidate));

            seen[names[i]] = suffix;
            seen[candidate] = 1;
            names[i] = candidate;
        }
    }
}

/// <summary>
/// Writes a <see cref="DataFrame"/> into a database table.
/// </summary>
/// <remarks>
/// Row-by-row parameterised inserts wrapped in one transaction. That is not a bulk loader — a
/// provider's own bulk-copy API will beat it by an order of magnitude on millions of rows — but it
/// is portable across every provider and correct, which is the right trade for the sizes a frame
/// usually holds.
/// </remarks>
public static class SqlWriter
{
    /// <summary>
    /// Inserts every row of <paramref name="frame"/> into <paramref name="table"/>.
    /// </summary>
    /// <param name="frame">The rows to write.</param>
    /// <param name="connection">An open — or openable — connection.</param>
    /// <param name="table">Destination table name. It must already exist.</param>
    /// <param name="batchSize">Rows per transaction commit.</param>
    /// <returns>The number of rows written.</returns>
    /// <remarks>
    /// The table name is interpolated into the SQL because it cannot be parameterised, so it must
    /// come from your code rather than from user input. Column names come from the frame and are
    /// checked for anything that is not a plain identifier.
    /// </remarks>
    public static int Write(DataFrame frame, DbConnection connection, string table, int batchSize = 500)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentException.ThrowIfNullOrWhiteSpace(table);

        RequirePlainIdentifier(table, nameof(table));
        foreach (var name in frame.ColumnNames) RequirePlainIdentifier(name, "column name");

        if (frame.RowCount == 0) return 0;

        var wasClosed = connection.State == ConnectionState.Closed;
        if (wasClosed) connection.Open();

        try
        {
            var columns = frame.ColumnNames.ToArray();
            var placeholders = string.Join(", ", columns.Select((_, i) => $"@p{i}"));
            var sql = $"INSERT INTO {table} ({string.Join(", ", columns)}) VALUES ({placeholders})";

            var written = 0;
            var transaction = connection.BeginTransaction();

            try
            {
                using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = sql;

                for (var i = 0; i < columns.Length; i++)
                {
                    var parameter = command.CreateParameter();
                    parameter.ParameterName = $"@p{i}";
                    command.Parameters.Add(parameter);
                }

                for (var row = 0; row < frame.RowCount; row++)
                {
                    for (var i = 0; i < columns.Length; i++)
                        ((DbParameter)command.Parameters[i]!).Value = frame[columns[i]].GetValue(row) ?? DBNull.Value;

                    command.ExecuteNonQuery();
                    written++;

                    if (written % batchSize != 0) continue;

                    transaction.Commit();
                    transaction.Dispose();
                    transaction = connection.BeginTransaction();
                    command.Transaction = transaction;
                }

                transaction.Commit();
            }
            catch
            {
                transaction.Rollback();
                throw;
            }
            finally
            {
                transaction.Dispose();
            }

            return written;
        }
        finally
        {
            if (wasClosed) connection.Close();
        }
    }

    /// <summary>Rejects anything that is not a bare identifier, since these cannot be parameterised.</summary>
    private static void RequirePlainIdentifier(string value, string what)
    {
        if (value.Length > 0 && (char.IsLetter(value[0]) || value[0] == '_')
            && value.All(c => char.IsLetterOrDigit(c) || c == '_')) return;

        throw new ArgumentException(
            $"'{value}' is not a plain identifier, and a {what} cannot be passed as a parameter. " +
            "Rename it, or write the insert yourself.");
    }
}
