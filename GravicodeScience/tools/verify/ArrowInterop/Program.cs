using System.Globalization;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

// The C# half of tools/verify/arrow_interop.py. It writes the file pyarrow will read and reads the
// files pyarrow writes; the Python side does the asserting. Kept as a separate executable rather
// than folded into the test project because the check needs a Python toolchain the tests do not.

Console.OutputEncoding = System.Text.Encoding.UTF8;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: ArrowInterop <write|read|values> <path> [column]");
    return 1;
}

switch (args[0])
{
    case "write":
        ArrowFile.Write(Reference(), args[1]);
        Console.WriteLine($"wrote {new FileInfo(args[1]).Length} bytes");
        return 0;

    case "read":
    {
        var frame = ArrowFile.Read(args[1]);
        Console.WriteLine($"{frame.RowCount} rows x {frame.ColumnNames.Count} columns");

        foreach (var name in frame.ColumnNames)
        {
            var column = frame[name];
            var preview = string.Join(",", Enumerable.Range(0, frame.RowCount)
                .Select(i => column.IsMissing(i) ? "null" : Render(column.GetValue(i))));
            Console.WriteLine($"{name}={preview}");
        }

        return 0;
    }

    case "values":
    {
        var frame = ArrowFile.Read(args[1]);
        var column = frame[args[2]];

        Console.WriteLine(string.Join(",", Enumerable.Range(0, frame.RowCount)
            .Select(i => column.IsMissing(i) ? "null" : Render(column.GetValue(i)))));

        return 0;
    }

    default:
        Console.Error.WriteLine($"unknown command '{args[0]}'");
        return 1;
}

// Invariant rendering, so the Python side can parse it regardless of the machine's locale.
static string Render(object? value) => value switch
{
    null => "null",
    double d => d.ToString("R", CultureInfo.InvariantCulture),
    DateTime t => t.ToString("O", CultureInfo.InvariantCulture),
    bool b => b ? "true" : "false",
    _ => value.ToString() ?? "",
};

/// <summary>The frame the Python side asserts against. Every column exercises a different edge.</summary>
static DataFrame Reference() => new(
[
    new NumericSeries("value", [1.5, -2.25, double.NaN, 1e300, 0.0]),
    new TextSeries("label", ["alpha", "", null, "ünïcødé 🐱", "last"]),
    new BooleanSeries("flag", [true, false, null, true, false]),
    new DateTimeSeries("when",
    [
        new DateTime(2024, 3, 1, 12, 30, 45),
        new DateTime(1999, 12, 31, 23, 59, 59),
        null,
        DateTime.UnixEpoch,
        new DateTime(2030, 6, 15),
    ]),
]);
