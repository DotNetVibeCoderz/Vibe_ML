using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>The type a message field is read as.</summary>
public enum ColumnType
{
    /// <summary>A double; text that does not parse is missing.</summary>
    Number,
    /// <summary>Text, as given (a nested JSON object or array is kept as its JSON).</summary>
    Text,
    /// <summary>true/false.</summary>
    Boolean,
    /// <summary>An ISO 8601 date and time, normalised to UTC.</summary>
    DateTime,
}

/// <summary>
/// How a message payload becomes a row: a JSON object (one field per column) or one CSV line.
/// </summary>
/// <remarks>
/// Columns without a declared type are typed from the first value seen and then pinned for the
/// life of the query, so every batch has the same schema (a column first seen as null pins as
/// text). Declare the schema when a field's first value might mislead. A value that does not fit
/// its column's type is read as missing rather than failing the batch.
/// </remarks>
public sealed class MessageFormat
{
    private MessageFormat(bool json, (string Column, ColumnType? Type)[] columns, char separator)
    {
        IsJson = json;
        Columns = columns;
        Separator = separator;
    }

    internal bool IsJson { get; }
    internal (string Column, ColumnType? Type)[] Columns { get; }
    internal char Separator { get; }

    /// <summary>
    /// JSON objects. Fields listed in <paramref name="schema"/> come first with the given types;
    /// any other field becomes a column (typed from its first value) the first time it appears.
    /// </summary>
    public static MessageFormat Json(params (string Column, ColumnType Type)[] schema)
        => new(true, [.. schema.Select(s => (s.Column, (ColumnType?)s.Type))], ',');

    /// <summary>
    /// One CSV line per message, without a header: <paramref name="columns"/> names the fields in
    /// order. Without <paramref name="types"/> each column is typed from its first non-empty value.
    /// </summary>
    public static MessageFormat Csv(string[] columns, ColumnType[]? types = null, char separator = ',')
    {
        ArgumentNullException.ThrowIfNull(columns);
        if (columns.Length == 0) throw new ArgumentException("Name the CSV columns.", nameof(columns));
        if (types is not null && types.Length != columns.Length)
            throw new ArgumentException("Give one type per column.", nameof(types));
        return new(false, [.. columns.Select((c, i) => (c, types is null ? null : (ColumnType?)types[i]))], separator);
    }
}

/// <summary>Batching and metadata columns shared by the message-broker sources.</summary>
public class MessageSourceOptions
{
    /// <summary>A batch is closed at this many messages.</summary>
    public int MaxBatchRows { get; set; } = 1000;

    /// <summary>...or this long after its first message, whichever comes first.</summary>
    public TimeSpan MaxBatchDelay { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>The stream ends after this long without a message (a finite run, which flushes
    /// windows). Null waits until the query is cancelled.</summary>
    public TimeSpan? IdleTimeout { get; set; }

    /// <summary>A column holding each message's topic; null for none.</summary>
    public string? TopicColumn { get; set; } = "topic";

    /// <summary>A column holding each message's timestamp (Kafka's message timestamp; for MQTT,
    /// which has none, the arrival time), in UTC; null for none.</summary>
    public string? TimestampColumn { get; set; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchRows, 1);
        if (MaxBatchDelay < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(MaxBatchDelay));
        if (IdleTimeout is { } idle && idle <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(IdleTimeout));
    }
}

/// <summary>One message as a source received it.</summary>
internal sealed record RawMessage(string Topic, string Payload, DateTime? Timestamp, string? Key = null, int Partition = 0, long Offset = 0);

/// <summary>
/// Turns messages into frames with a pinned schema. One decoder lives for one query run.
/// </summary>
internal sealed class MessageDecoder(MessageFormat format, MessageSourceOptions options, params (string Name, Func<RawMessage, object?> Value, ColumnType Type)[] extra)
{
    private readonly List<string> _names = [.. format.Columns.Select(c => c.Column)];
    private readonly Dictionary<string, ColumnType?> _types = format.Columns.ToDictionary(c => c.Column, c => c.Type, StringComparer.Ordinal);

    public long Malformed { get; private set; }

    public DataFrame Decode(IReadOnlyList<RawMessage> messages)
    {
        var rows = new List<Dictionary<string, object?>>(messages.Count);
        foreach (var m in messages)
        {
            var row = format.IsJson ? ParseJson(m.Payload) : ParseCsv(m.Payload);
            if (row is null) { Malformed++; row = []; }
            rows.Add(row);
        }
        var columns = new List<Series>();
        foreach (var name in _names)
        {
            var type = _types[name] ??= Infer(format.IsJson, rows.Select(r => r.GetValueOrDefault(name)).FirstOrDefault(v => v is not null));
            columns.Add(Build(name, type, rows.Select(r => r.GetValueOrDefault(name)).ToArray()));
        }
        if (options.TopicColumn is { } topic) columns.Add(new TextSeries(topic, messages.Select(m => m.Topic).ToArray()));
        if (options.TimestampColumn is { } ts) columns.Add(new DateTimeSeries(ts, messages.Select(m => m.Timestamp).ToArray()));
        foreach (var (name, value, type) in extra) columns.Add(Build(name, type, messages.Select(value).ToArray()));
        return new DataFrame(columns);
    }

    private Dictionary<string, object?>? ParseJson(string payload)
    {
        try
        {
            using var doc = JsonDocument.Parse(payload);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return null;
            var row = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (!_types.ContainsKey(p.Name))
                {
                    _names.Add(p.Name);
                    _types[p.Name] = null;
                }
                row[p.Name] = p.Value.ValueKind switch
                {
                    JsonValueKind.Null or JsonValueKind.Undefined => null,
                    JsonValueKind.Number => p.Value.GetDouble(),
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.String => p.Value.GetString(),
                    _ => p.Value.GetRawText(),
                };
            }
            return row;
        }
        catch (JsonException) { return null; }
    }

    private Dictionary<string, object?>? ParseCsv(string payload)
    {
        var fields = SplitCsv(payload.TrimEnd('\r', '\n'), format.Separator);
        if (fields is null) return null;
        var row = new Dictionary<string, object?>(StringComparer.Ordinal);
        for (var i = 0; i < _names.Count && i < fields.Count; i++)
            row[_names[i]] = fields[i].Length == 0 ? null : fields[i];
        return row;
    }

    /// <summary>One RFC 4180 line: quoted fields may hold separators and doubled quotes.</summary>
    internal static List<string>? SplitCsv(string line, char separator)
    {
        var fields = new List<string>();
        var current = new StringBuilder();
        var quoted = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (quoted)
            {
                if (c != '"') current.Append(c);
                else if (i + 1 < line.Length && line[i + 1] == '"') { current.Append('"'); i++; }
                else quoted = false;
            }
            else if (c == '"' && current.Length == 0) quoted = true;
            else if (c == separator) { fields.Add(current.ToString()); current.Clear(); }
            else current.Append(c);
        }
        if (quoted) return null;
        fields.Add(current.ToString());
        return fields;
    }

    /// <remarks>A JSON string stays text (or a date): "007" is an id, not seven. CSV has only text,
    /// so there numbers and booleans are recognised by their spelling.</remarks>
    private static ColumnType Infer(bool json, object? value) => value switch
    {
        double => ColumnType.Number,
        bool => ColumnType.Boolean,
        string s when !json && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out _) => ColumnType.Number,
        string s when !json && s is "true" or "false" or "True" or "False" => ColumnType.Boolean,
        string s when LooksLikeDate(s) && ParseDate(s) is not null => ColumnType.DateTime,
        _ => ColumnType.Text,
    };

    private static bool LooksLikeDate(string s) => s.Length >= 10 && char.IsAsciiDigit(s[0]) && s[4] == '-' && s[7] == '-';

    internal static DateTime? ParseDate(string s)
        => DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var t) ? t : null;

    private static Series Build(string name, ColumnType type, object?[] values) => type switch
    {
        ColumnType.Number => new NumericSeries(name, values.Select(v => v switch
        {
            double d => d,
            bool b => b ? 1 : 0,
            string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
            _ => double.NaN,
        }).ToArray()),
        ColumnType.Boolean => new BooleanSeries(name, values.Select(v => v switch
        {
            bool b => b,
            string s when bool.TryParse(s, out var b) => b,
            double d => d != 0,
            _ => (bool?)null,
        }).ToArray()),
        ColumnType.DateTime => new DateTimeSeries(name, values.Select(v => v switch
        {
            DateTime t => t,
            string s => ParseDate(s),
            _ => (DateTime?)null,
        }).ToArray()),
        _ => new TextSeries(name, values.Select(v => v switch
        {
            null => null,
            double d => d.ToString("R", CultureInfo.InvariantCulture),
            bool b => b ? "true" : "false",
            _ => v.ToString(),
        }).ToArray()),
    };

    /// <summary>
    /// Groups messages into batches: a batch starts with the first message to arrive and closes
    /// at <see cref="MessageSourceOptions.MaxBatchRows"/> or <see cref="MessageSourceOptions.MaxBatchDelay"/>.
    /// <paramref name="next"/> returns null when its timeout passes with nothing received.
    /// </summary>
    internal static async IAsyncEnumerable<List<RawMessage>> Batches(Func<TimeSpan, CancellationToken, ValueTask<RawMessage?>> next,
        MessageSourceOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        while (true)
        {
            var first = await next(options.IdleTimeout ?? Timeout.InfiniteTimeSpan, cancellationToken).ConfigureAwait(false);
            if (first is null) yield break;                              // idle: the stream ends
            var batch = new List<RawMessage> { first };
            var deadline = DateTime.UtcNow + options.MaxBatchDelay;
            while (batch.Count < options.MaxBatchRows)
            {
                var left = deadline - DateTime.UtcNow;
                if (left <= TimeSpan.Zero) break;
                var m = await next(left, cancellationToken).ConfigureAwait(false);
                if (m is null) break;
                batch.Add(m);
            }
            yield return batch;
        }
    }
}
