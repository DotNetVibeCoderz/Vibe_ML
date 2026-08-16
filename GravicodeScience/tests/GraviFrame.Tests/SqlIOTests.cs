using System.Data;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Microsoft.Data.Sqlite;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for the ADO.NET reader and writer.
/// </summary>
/// <remarks>
/// Run against a real provider — an in-memory SQLite database — rather than a hand-written fake, so
/// that the provider conventions the code has to survive (its own null handling, its type affinity,
/// its parameter prefixes) are the real ones. <see cref="SqlReader.FromReader"/> is additionally
/// checked against a <see cref="DataTable"/>, which is a second, independent implementation of
/// <see cref="IDataReader"/>.
/// </remarks>
public class SqlIOTests : IDisposable
{
    private readonly SqliteConnection _connection;

    public SqlIOTests()
    {
        // A shared in-memory database lives as long as the connection does.
        _connection = new SqliteConnection("Data Source=:memory:");
        _connection.Open();

        Execute("""
            CREATE TABLE people (
                id INTEGER, name TEXT, score REAL, active INTEGER, joined TEXT
            )
            """);

        Execute("""
            INSERT INTO people (id, name, score, active, joined) VALUES
                (1, 'Ada',     92.5, 1, '2020-01-15'),
                (2, 'Linus',   81.0, 0, '2021-06-30'),
                (3, NULL,      NULL, 1, NULL)
            """);
    }

    private void Execute(string sql)
    {
        using var command = _connection.CreateCommand();
        command.CommandText = sql;
        command.ExecuteNonQuery();
    }

    public void Dispose() => _connection.Dispose();

    [Fact]
    public void AQueryBecomesATypedFrame()
    {
        var frame = SqlReader.Read(_connection, "SELECT id, name, score FROM people ORDER BY id");

        Assert.Equal(3, frame.RowCount);
        Assert.Equal(["id", "name", "score"], frame.ColumnNames);

        // The provider reports the types, so no inference is involved.
        Assert.Equal(DataType.Numeric, frame["id"].DataType);
        Assert.Equal(DataType.Text, frame["name"].DataType);
        Assert.Equal(DataType.Numeric, frame["score"].DataType);

        Assert.Equal(1.0, frame.Numeric("id")[0]);
        Assert.Equal("Ada", ((TextSeries)frame["name"])[0]);
        Assert.Equal(92.5, frame.Numeric("score")[0]);
    }

    [Fact]
    public void NullsBecomeTheRightKindOfMissing()
    {
        // A database NULL and a NaN are the same idea; the mapping has to preserve it in every
        // column type rather than turning a null string into "" or a null number into 0.
        var frame = SqlReader.Read(_connection, "SELECT id, name, score FROM people WHERE id = 3");

        Assert.True(frame["name"].IsMissing(0));
        Assert.True(double.IsNaN(frame.Numeric("score")[0]));
    }

    [Fact]
    public void ParametersAreBoundRatherThanInterpolated()
    {
        // The mechanism that makes injection impossible: the value never becomes part of the SQL.
        var frame = SqlReader.Read(_connection,
            "SELECT name FROM people WHERE score > $floor",
            new Dictionary<string, object?> { ["$floor"] = 85.0 });

        Assert.Equal(1, frame.RowCount);
        Assert.Equal("Ada", ((TextSeries)frame["name"])[0]);
    }

    [Fact]
    public void AHostileParameterValueStaysAValue()
    {
        // The classic payload. Bound as a parameter it is just a name that matches nothing, and the
        // table is still there afterwards.
        var frame = SqlReader.Read(_connection,
            "SELECT * FROM people WHERE name = $name",
            new Dictionary<string, object?> { ["$name"] = "x'; DROP TABLE people; --" });

        Assert.Equal(0, frame.RowCount);
        Assert.Equal(3, SqlReader.Read(_connection, "SELECT id FROM people").RowCount);
    }

    [Fact]
    public void TheConnectionIsLeftInTheStateItWasFound()
    {
        // A caller managing its own connection lifetime should not find it closed afterwards, and
        // one that handed over a closed connection should not be left with an open one.
        using var closed = new SqliteConnection("Data Source=:memory:");
        Assert.Equal(ConnectionState.Closed, closed.State);

        using (var command = closed.CreateCommand())
        {
            closed.Open();
            command.CommandText = "CREATE TABLE t (a INTEGER)";
            command.ExecuteNonQuery();
        }

        Assert.Equal(ConnectionState.Open, _connection.State);
        SqlReader.Read(_connection, "SELECT id FROM people");
        Assert.Equal(ConnectionState.Open, _connection.State);
    }

    [Fact]
    public void DuplicateColumnNamesAreSuffixedRatherThanColliding()
    {
        // Legal in SQL, illegal in a frame. Overwriting would silently lose a column.
        var frame = SqlReader.Read(_connection, "SELECT id, id, id FROM people WHERE id = 1");

        Assert.Equal(["id", "id_1", "id_2"], frame.ColumnNames);
        Assert.Equal(1.0, frame.Numeric("id_2")[0]);
    }

    [Fact]
    public void AnEmptyResultSetStillCarriesItsColumns()
    {
        // Returning a frame with no columns would make downstream code fail on a name lookup rather
        // than on the row count, which is a far more confusing place to discover the problem.
        var frame = SqlReader.Read(_connection, "SELECT id, name FROM people WHERE id = 999");

        Assert.Equal(0, frame.RowCount);
        Assert.Equal(["id", "name"], frame.ColumnNames);
    }

    [Fact]
    public void FromReaderWorksAgainstADataTableToo()
    {
        // A second, independent IDataReader implementation from the BCL — so this checks the code
        // against the interface rather than against one provider's habits.
        var table = new DataTable();
        table.Columns.Add("n", typeof(int));
        table.Columns.Add("label", typeof(string));
        table.Columns.Add("flag", typeof(bool));
        table.Columns.Add("when", typeof(DateTime));

        table.Rows.Add(7, "seven", true, new DateTime(2024, 3, 1));
        table.Rows.Add(DBNull.Value, DBNull.Value, DBNull.Value, DBNull.Value);

        using var reader = table.CreateDataReader();
        var frame = SqlReader.FromReader(reader);

        Assert.Equal(2, frame.RowCount);
        Assert.Equal(DataType.Numeric, frame["n"].DataType);
        Assert.Equal(DataType.Boolean, frame["flag"].DataType);
        Assert.Equal(DataType.DateTime, frame["when"].DataType);

        Assert.Equal(7.0, frame.Numeric("n")[0]);
        Assert.Equal(true, ((BooleanSeries)frame["flag"])[0]);
        Assert.Equal(new DateTime(2024, 3, 1), ((DateTimeSeries)frame["when"])[0]);

        for (var i = 0; i < frame.ColumnNames.Count; i++)
            Assert.True(frame[frame.ColumnNames[i]].IsMissing(1));
    }

    [Fact]
    public void WritingAFrameRoundTripsThroughTheDatabase()
    {
        Execute("CREATE TABLE metrics (label TEXT, value REAL, ok INTEGER)");

        var frame = new DataFrame(
        [
            new TextSeries("label", ["a", "b", "c"]),
            new NumericSeries("value", [1.5, 2.5, 3.5]),
            new BooleanSeries("ok", [true, false, true]),
        ]);

        Assert.Equal(3, SqlWriter.Write(frame, _connection, "metrics"));

        var back = SqlReader.Read(_connection, "SELECT label, value, ok FROM metrics ORDER BY label");

        Assert.Equal(3, back.RowCount);
        Assert.Equal([1.5, 2.5, 3.5], back.Numeric("value").Values.ToArray());
        Assert.Equal("b", ((TextSeries)back["label"])[1]);
    }

    [Fact]
    public void MissingValuesAreWrittenAsNull()
    {
        Execute("CREATE TABLE sparse (label TEXT, value REAL)");

        var frame = new DataFrame(
        [
            new TextSeries("label", ["a", null]),
            new NumericSeries("value", [1.0, double.NaN]),
        ]);

        SqlWriter.Write(frame, _connection, "sparse");

        var nulls = SqlReader.Read(_connection,
            "SELECT COUNT(*) AS n FROM sparse WHERE label IS NULL AND value IS NULL");

        Assert.Equal(1.0, nulls.Numeric("n")[0]);
    }

    [Fact]
    public void AFailedInsertRollsBackTheWholeBatch()
    {
        // Half-written data is worse than none, because nothing in the table says which half.
        Execute("CREATE TABLE strict (id INTEGER PRIMARY KEY, label TEXT)");
        Execute("INSERT INTO strict (id, label) VALUES (2, 'existing')");

        var frame = new DataFrame(
        [
            new NumericSeries("id", [1, 2, 3]),      // id 2 collides with the row already there
            new TextSeries("label", ["x", "y", "z"]),
        ]);

        Assert.ThrowsAny<Exception>(() => SqlWriter.Write(frame, _connection, "strict"));

        var count = SqlReader.Read(_connection, "SELECT COUNT(*) AS n FROM strict");
        Assert.Equal(1.0, count.Numeric("n")[0]);
    }

    [Fact]
    public void ATableNameThatIsNotAPlainIdentifierIsRefused()
    {
        // A table name cannot be parameterised, so it is interpolated — which makes rejecting
        // anything but a bare identifier the thing standing between this and an injection.
        var frame = new DataFrame([new NumericSeries("a", [1])]);

        Assert.Throws<ArgumentException>(
            () => SqlWriter.Write(frame, _connection, "people; DROP TABLE people; --"));
    }

    [Fact]
    public void AColumnNameThatIsNotAPlainIdentifierIsRefused()
    {
        var frame = new DataFrame([new NumericSeries("a); DROP TABLE people; --", [1])]);

        Assert.Throws<ArgumentException>(() => SqlWriter.Write(frame, _connection, "people"));
    }

    [Fact]
    public void WritingAnEmptyFrameIsANoOp()
    {
        Execute("CREATE TABLE blank (a REAL)");
        var frame = new DataFrame([new NumericSeries("a", [])]);

        Assert.Equal(0, SqlWriter.Write(frame, _connection, "blank"));
    }
}
