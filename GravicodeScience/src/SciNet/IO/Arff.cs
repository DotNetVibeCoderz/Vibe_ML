using System.Globalization;
using System.Text;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Attribute kinds of an ARFF file.</summary>
public enum ArffAttributeType
{
    /// <summary><c>numeric</c>, <c>real</c> or <c>integer</c> — read as double, missing as NaN.</summary>
    Numeric,
    /// <summary>A <c>{a, b, c}</c> nominal set — read as strings, missing as null.</summary>
    Nominal,
    /// <summary>Free text — read as strings, missing as null. (SciPy refuses these.)</summary>
    String,
    /// <summary>A date with a Java <c>SimpleDateFormat</c> pattern — read as <see cref="DateTime"/>, missing as null.</summary>
    Date,
    /// <summary>A relational (multi-instance) attribute: each value is a nested table.</summary>
    Relational,
}

/// <summary>One attribute declaration.</summary>
/// <param name="Name">Attribute name (unquoted).</param>
/// <param name="Type">Its kind.</param>
/// <param name="NominalValues">The allowed values of a nominal attribute, in declaration order; null otherwise.</param>
/// <param name="DateFormat">The Java date pattern of a date attribute; null otherwise.</param>
/// <param name="RelationalAttributes">The inner attributes of a relational attribute; null otherwise.</param>
public sealed record ArffAttribute(string Name, ArffAttributeType Type, IReadOnlyList<string>? NominalValues = null, string? DateFormat = null, IReadOnlyList<ArffAttribute>? RelationalAttributes = null)
{
    /// <summary>The type as SciPy's <c>MetaData.types()</c> spells it.</summary>
    public string TypeName => Type switch
    {
        ArffAttributeType.Numeric => "numeric",
        ArffAttributeType.Nominal => "nominal",
        ArffAttributeType.String => "string",
        ArffAttributeType.Date => "date",
        _ => "relational",
    };
}

/// <summary>Header of an ARFF file (SciPy's <c>MetaData</c>).</summary>
public sealed class ArffMeta
{
    internal ArffMeta(string name, IReadOnlyList<ArffAttribute> attributes)
    {
        Name = name;
        Attributes = attributes;
    }

    /// <summary>The <c>@relation</c> name.</summary>
    public string Name { get; }

    /// <summary>Attributes in declaration order.</summary>
    public IReadOnlyList<ArffAttribute> Attributes { get; }

    /// <summary>Attribute names in order.</summary>
    public string[] Names() => Attributes.Select(a => a.Name).ToArray();

    /// <summary>Attribute type names in order (<c>numeric</c>, <c>nominal</c>, ...).</summary>
    public string[] Types() => Attributes.Select(a => a.TypeName).ToArray();

    /// <summary>An attribute by name.</summary>
    public ArffAttribute this[string name] => Attributes.FirstOrDefault(a => a.Name == name) ?? throw new KeyNotFoundException($"No attribute '{name}'.");
}

/// <summary>The data of an ARFF file, stored by column.</summary>
public sealed class ArffData
{
    private readonly object[] _columns;

    internal ArffData(ArffMeta meta, object[] columns, int count)
    {
        Meta = meta;
        _columns = columns;
        Count = count;
    }

    /// <summary>The header.</summary>
    public ArffMeta Meta { get; }

    /// <summary>Number of instances (rows).</summary>
    public int Count { get; }

    private int Index(string name)
    {
        for (var i = 0; i < Meta.Attributes.Count; i++) if (Meta.Attributes[i].Name == name) return i;
        throw new KeyNotFoundException($"No attribute '{name}'.");
    }

    /// <summary>The raw column: <c>double[]</c>, <c>string?[]</c>, <c>DateTime?[]</c> or <c>ArffData?[]</c> depending on the type.</summary>
    public object Column(string name) => _columns[Index(name)];

    /// <summary>A numeric column (NaN where missing).</summary>
    public double[] Numeric(string name) => _columns[Index(name)] as double[] ?? throw new InvalidOperationException($"Attribute '{name}' is {Meta[name].TypeName}, not numeric.");

    /// <summary>A nominal or string column (null where missing).</summary>
    public string?[] Strings(string name) => _columns[Index(name)] as string?[] ?? throw new InvalidOperationException($"Attribute '{name}' is {Meta[name].TypeName}, not nominal or string.");

    /// <summary>A date column (null where missing).</summary>
    public DateTime?[] Dates(string name) => _columns[Index(name)] as DateTime?[] ?? throw new InvalidOperationException($"Attribute '{name}' is {Meta[name].TypeName}, not date.");

    /// <summary>A relational column: one nested table per instance (null where missing).</summary>
    public ArffData?[] Relational(string name) => _columns[Index(name)] as ArffData?[] ?? throw new InvalidOperationException($"Attribute '{name}' is {Meta[name].TypeName}, not relational.");

    /// <summary>The nominal column as category codes (index into the declared values; -1 where missing).</summary>
    public int[] Codes(string name)
    {
        var a = Meta[name];
        if (a.Type != ArffAttributeType.Nominal) throw new InvalidOperationException($"Attribute '{name}' is not nominal.");
        var s = Strings(name);
        var map = a.NominalValues!.Select((v, i) => (v, i)).GroupBy(t => t.v).ToDictionary(g => g.Key, g => g.First().i);
        return s.Select(v => v is null ? -1 : map[v]).ToArray();
    }
}

/// <summary>
/// Weka ARFF files — <c>scipy.io.arff.loadarff</c>. Beyond SciPy: string attributes are read, and
/// relational attributes become nested tables. Sparse ARFF (<c>{index value, ...}</c> rows) is
/// refused, as SciPy refuses it.
/// </summary>
public static class Arff
{
    /// <summary>Loads an ARFF file.</summary>
    public static ArffData LoadArff(string path)
    {
        using var r = new StreamReader(path, Encoding.UTF8);
        return LoadArff(r);
    }

    /// <summary>Loads an ARFF stream (UTF-8).</summary>
    public static ArffData LoadArff(Stream stream)
    {
        using var r = new StreamReader(stream, Encoding.UTF8, leaveOpen: true);
        return LoadArff(r);
    }

    /// <summary>Loads ARFF text.</summary>
    public static ArffData LoadArff(TextReader reader)
    {
        string? relation = null;
        var attrs = new List<ArffAttribute>();
        var lineNo = 0;
        string? line;
        while (true)
        {
            line = reader.ReadLine();
            lineNo++;
            if (line is null) throw new InvalidDataException("ARFF file has no @data section.");
            var t = line.Trim();
            if (t.Length == 0 || t[0] == '%') continue;
            if (Keyword(t, "@relation"))
            {
                relation = Unquote(t[9..].Trim());
                continue;
            }
            if (Keyword(t, "@attribute"))
            {
                attrs.Add(ParseAttribute(t, reader, ref lineNo));
                continue;
            }
            if (Keyword(t, "@data")) break;
            throw new InvalidDataException($"ARFF line {lineNo}: unexpected header line '{t}'.");
        }
        if (relation is null) throw new InvalidDataException("ARFF file has no @relation line.");
        if (attrs.Count == 0) throw new InvalidDataException("ARFF file declares no attributes.");
        var meta = new ArffMeta(relation, attrs);
        var rows = new List<string>();
        while ((line = reader.ReadLine()) is not null)
        {
            lineNo++;
            var t = line.Trim();
            if (t.Length == 0 || t[0] == '%') continue;
            if (t[0] == '{') throw new NotSupportedException($"ARFF line {lineNo}: sparse ARFF data is not supported (SciPy does not support it either).");
            rows.Add(t);
        }
        return Build(meta, rows, ',');
    }

    private static bool Keyword(string line, string kw)
        => line.Length >= kw.Length && line.AsSpan(0, kw.Length).Equals(kw, StringComparison.OrdinalIgnoreCase)
           && (line.Length == kw.Length || char.IsWhiteSpace(line[kw.Length]));

    private static ArffAttribute ParseAttribute(string line, TextReader reader, ref int lineNo)
    {
        var rest = line[10..].TrimStart();
        string name;
        if (rest.Length > 0 && rest[0] is '\'' or '"')
        {
            var end = ClosingQuote(rest, 0);
            if (end < 0) throw new InvalidDataException($"ARFF line {lineNo}: unterminated quoted attribute name.");
            name = Unescape(rest[1..end]);
            rest = rest[(end + 1)..].Trim();
        }
        else
        {
            var sp = 0;
            while (sp < rest.Length && !char.IsWhiteSpace(rest[sp]) && rest[sp] != '{') sp++;
            name = rest[..sp];
            rest = rest[sp..].Trim();
        }
        if (name.Length == 0) throw new InvalidDataException($"ARFF line {lineNo}: attribute without a name.");
        var lower = rest.ToLowerInvariant();
        if (lower is "numeric" or "real" or "integer") return new ArffAttribute(name, ArffAttributeType.Numeric);
        if (lower == "string") return new ArffAttribute(name, ArffAttributeType.String);
        if (rest.StartsWith('{'))
        {
            var close = rest.LastIndexOf('}');
            if (close < 0) throw new InvalidDataException($"ARFF line {lineNo}: nominal attribute '{name}' has no closing brace.");
            var values = SplitValues(rest[1..close], ',').Select(v => v.Value).ToList();
            return new ArffAttribute(name, ArffAttributeType.Nominal, values);
        }
        if (lower.StartsWith("date"))
        {
            var pattern = rest[4..].Trim();
            pattern = pattern.Length == 0 ? "yyyy-MM-dd'T'HH:mm:ss" : Unquote(pattern); // the ARFF default is ISO-8601
            return new ArffAttribute(name, ArffAttributeType.Date, DateFormat: pattern);
        }
        if (lower == "relational")
        {
            var inner = new List<ArffAttribute>();
            while (true)
            {
                var l = reader.ReadLine() ?? throw new InvalidDataException($"ARFF relational attribute '{name}' has no @end.");
                lineNo++;
                var t = l.Trim();
                if (t.Length == 0 || t[0] == '%') continue;
                if (Keyword(t, "@end")) break;
                if (!Keyword(t, "@attribute")) throw new InvalidDataException($"ARFF line {lineNo}: expected @attribute or @end inside relational attribute '{name}'.");
                inner.Add(ParseAttribute(t, reader, ref lineNo));
            }
            return new ArffAttribute(name, ArffAttributeType.Relational, RelationalAttributes: inner);
        }
        throw new InvalidDataException($"ARFF line {lineNo}: unknown type '{rest}' for attribute '{name}'.");
    }

    private static int ClosingQuote(string s, int open)
    {
        var q = s[open];
        for (var i = open + 1; i < s.Length; i++)
        {
            if (s[i] == '\\') { i++; continue; }
            if (s[i] == q) return i;
        }
        return -1;
    }

    private static string Unquote(string s)
    {
        s = s.Trim();
        if (s.Length >= 2 && s[0] is '\'' or '"' && s[^1] == s[0]) return Unescape(s[1..^1]);
        return s;
    }

    private static string Unescape(string s)
    {
        if (s.IndexOf('\\') < 0) return s;
        var sb = new StringBuilder(s.Length);
        for (var i = 0; i < s.Length; i++)
        {
            if (s[i] != '\\' || i + 1 == s.Length) { sb.Append(s[i]); continue; }
            var c = s[++i];
            sb.Append(c switch { 'n' => '\n', 't' => '\t', 'r' => '\r', _ => c });
        }
        return sb.ToString();
    }

    /// <summary>Splits on <paramref name="delimiter"/> outside quotes; quoted values keep inner whitespace and are flagged so '?' inside quotes is not "missing".</summary>
    private static List<(string Value, bool Quoted)> SplitValues(string s, char delimiter)
    {
        var result = new List<(string, bool)>();
        var i = 0;
        while (true)
        {
            while (i < s.Length && s[i] != delimiter && char.IsWhiteSpace(s[i])) i++;
            if (i < s.Length && s[i] is '\'' or '"')
            {
                var end = ClosingQuote(s, i);
                if (end < 0) throw new InvalidDataException($"Unterminated quote in '{s}'.");
                result.Add((Unescape(s[(i + 1)..end]), true));
                i = end + 1;
                while (i < s.Length && s[i] != delimiter)
                {
                    if (!char.IsWhiteSpace(s[i])) throw new InvalidDataException($"Unexpected text after a quoted value in '{s}'.");
                    i++;
                }
            }
            else
            {
                var start = i;
                while (i < s.Length && s[i] != delimiter) i++;
                result.Add((s[start..i].Trim(), false));
            }
            if (i >= s.Length) break;
            i++; // skip the delimiter
        }
        return result;
    }

    private static ArffData Build(ArffMeta meta, List<string> rows, char delimiter)
    {
        var attrs = meta.Attributes;
        var n = rows.Count;
        var cols = new object[attrs.Count];
        for (var a = 0; a < attrs.Count; a++)
            cols[a] = attrs[a].Type switch
            {
                ArffAttributeType.Numeric => new double[n],
                ArffAttributeType.Date => new DateTime?[n],
                ArffAttributeType.Relational => new ArffData?[n],
                _ => (object)new string?[n],
            };
        var formats = attrs.Select(a => a.Type == ArffAttributeType.Date ? JavaToDotNet(a.DateFormat!) : null).ToArray();
        var nominal = attrs.Select(a => a.NominalValues is null ? null : new HashSet<string>(a.NominalValues, StringComparer.Ordinal)).ToArray();
        for (var r = 0; r < n; r++)
        {
            var vals = delimiter == ',' ? SplitValues(rows[r], ',') : SplitWhitespace(rows[r]);
            if (vals.Count != attrs.Count)
                throw new InvalidDataException($"ARFF data row {r + 1} ('{rows[r]}') has {vals.Count} values for {attrs.Count} attributes.");
            for (var a = 0; a < attrs.Count; a++)
            {
                var (v, quoted) = vals[a];
                var missing = !quoted && v == "?";
                switch (attrs[a].Type)
                {
                    case ArffAttributeType.Numeric:
                        if (missing) ((double[])cols[a])[r] = double.NaN;
                        else if (double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) ((double[])cols[a])[r] = d;
                        else throw new InvalidDataException($"ARFF data row {r + 1}: '{v}' is not a number (attribute '{attrs[a].Name}').");
                        break;
                    case ArffAttributeType.Nominal:
                        if (!missing && !nominal[a]!.Contains(v))
                            throw new InvalidDataException($"ARFF data row {r + 1}: '{v}' is not one of the declared values of '{attrs[a].Name}' ({string.Join(", ", attrs[a].NominalValues!)}).");
                        ((string?[])cols[a])[r] = missing ? null : v;
                        break;
                    case ArffAttributeType.String:
                        ((string?[])cols[a])[r] = missing ? null : v;
                        break;
                    case ArffAttributeType.Date:
                        if (missing) break;
                        if (!DateTime.TryParseExact(v, formats[a], CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt))
                            throw new InvalidDataException($"ARFF data row {r + 1}: '{v}' does not match the date format '{attrs[a].DateFormat}' of '{attrs[a].Name}'.");
                        ((DateTime?[])cols[a])[r] = dt;
                        break;
                    case ArffAttributeType.Relational:
                        if (missing) break;
                        var inner = new ArffMeta(attrs[a].Name, attrs[a].RelationalAttributes!);
                        var lines = v.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                        // Inner instances are comma- or tab-separated; Weka writes either.
                        var delim = lines.Any(l => l.Contains(',')) ? ',' : '\t';
                        ((ArffData?[])cols[a])[r] = Build(inner, lines, delim);
                        break;
                }
            }
        }
        return new ArffData(meta, cols, n);
    }

    private static List<(string, bool)> SplitWhitespace(string s)
        => s.Split(['\t', ' '], StringSplitOptions.RemoveEmptyEntries).Select(x => (Unquote(x), x.Length > 0 && x[0] is '\'' or '"')).ToList();

    /// <summary>
    /// Java <c>SimpleDateFormat</c> → .NET custom format. The common letters coincide
    /// (<c>yyyy MM dd HH mm ss</c>, quoted literals); <c>SSS</c> becomes <c>fff</c>. Time-zone
    /// letters are refused, as in SciPy — Java's <c>Z</c> (-0800) and .NET's <c>zzz</c> (-08:00)
    /// disagree, so accepting them would misread every offset.
    /// </summary>
    private static string JavaToDotNet(string java)
    {
        var sb = new StringBuilder();
        var inQuote = false;
        for (var i = 0; i < java.Length; i++)
        {
            var c = java[i];
            if (c == '\'') { inQuote = !inQuote; sb.Append(c); continue; }
            if (inQuote) { sb.Append(c); continue; }
            switch (c)
            {
                case 'z' or 'Z' or 'X':
                    throw new NotSupportedException($"Date format '{java}' has a time zone; date attributes with time zones are not supported (as in SciPy).");
                case 'S': sb.Append('f'); break;
                case 'a': sb.Append("tt"); break;
                case 'E': sb.Append("ddd"); while (i + 1 < java.Length && java[i + 1] == 'E') i++; break;
                case 'h' or 'H' or 'm' or 's' or 'y' or 'M' or 'd': sb.Append(c); break;
                default:
                    if (char.IsLetter(c)) throw new NotSupportedException($"Date pattern letter '{c}' in '{java}' is not supported.");
                    sb.Append(c is '/' or ':' or '%' or '\\' ? "\\" + c : c.ToString());
                    break;
            }
        }
        return sb.ToString();
    }
}
