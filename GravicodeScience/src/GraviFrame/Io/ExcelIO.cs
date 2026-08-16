using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Xml;
using System.Xml.Linq;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>Settings for reading a worksheet.</summary>
public sealed class ExcelOptions
{
    /// <summary>Sheet name. When null, the first sheet in the workbook is used.</summary>
    public string? SheetName { get; set; }

    /// <summary>Zero-based sheet position, used when <see cref="SheetName"/> is null.</summary>
    public int SheetIndex { get; set; }

    /// <summary>Whether the first row holds column names.</summary>
    public bool HasHeader { get; set; } = true;

    /// <summary>Rows sampled to infer each column's type. Zero means "use every row".</summary>
    public int TypeInferenceRows { get; set; } = 1000;

    /// <summary>Stop after this many data rows; zero means "read everything".</summary>
    public int MaxRows { get; set; }
}

/// <summary>
/// Reads <c>.xlsx</c> workbooks without a spreadsheet library.
/// </summary>
/// <remarks>
/// <para>
/// An <c>.xlsx</c> file is a zip archive of XML parts, all of which the BCL can already open:
/// <c>xl/workbook.xml</c> names the sheets, <c>xl/worksheets/sheetN.xml</c> holds the cells, and
/// <c>xl/sharedStrings.xml</c> holds the text, which the sheet references by index rather than
/// repeating. Reading it directly keeps the dependency count at zero, matching the approach the
/// ONNX reader takes.
/// </para>
/// <para>
/// Three things about the format bite anyone writing a reader for the first time, and each is
/// handled below:
/// </para>
/// <list type="bullet">
/// <item><b>Empty cells are absent, not blank.</b> A row records only the cells that hold
/// something, so a row's third <c>&lt;c&gt;</c> element is not necessarily column C. The cell's
/// own <c>r</c> reference is what says where it belongs, and reading positionally silently shifts
/// every value after a gap.</item>
/// <item><b>Dates are numbers.</b> Excel stores them as days since 1899-12-30 and marks them only
/// through a number format, so a date column arrives looking like five-digit integers.</item>
/// <item><b>The 1900 leap-year bug.</b> Excel believes 1900 was a leap year, for compatibility with
/// Lotus 1-2-3. The epoch used here is 1899-12-30 rather than 1899-12-31, which is what makes every
/// date from 1900-03-01 onwards come out right.</item>
/// </list>
/// <para>
/// This reads the common case — cell values, shared strings, inline strings, dates and booleans. It
/// does not evaluate formulas; a formula cell yields its cached result, which is what the file
/// records and is almost always what the caller wanted.
/// </para>
/// </remarks>
public static class ExcelReader
{
    private static readonly XNamespace Main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
    private static readonly XNamespace Relationships = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

    /// <summary>Excel's day zero. Not 1899-12-31 — see the note on the 1900 leap-year bug.</summary>
    private static readonly DateTime Epoch = new(1899, 12, 30);

    /// <summary>Reads a worksheet into a frame.</summary>
    public static DataFrame Read(string path, ExcelOptions? options = null)
    {
        using var stream = File.OpenRead(path);
        return Read(stream, options);
    }

    /// <summary>Reads a worksheet from an open stream.</summary>
    public static DataFrame Read(Stream stream, ExcelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new ExcelOptions();

        using var archive = new ZipArchive(stream, ZipArchiveMode.Read);

        var sharedStrings = ReadSharedStrings(archive);
        var dateStyles = ReadDateStyles(archive);
        var sheetPath = ResolveSheetPath(archive, options);

        var grid = ReadCells(archive, sheetPath, sharedStrings, dateStyles, options);
        return Build(grid, options);
    }

    /// <summary>The sheet names in workbook order.</summary>
    public static IReadOnlyList<string> SheetNames(string path)
    {
        using var stream = File.OpenRead(path);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Read);

        var workbook = LoadXml(archive, "xl/workbook.xml")
            ?? throw new InvalidDataException("This is not an xlsx workbook: xl/workbook.xml is missing.");

        return [.. workbook.Descendants(Main + "sheet").Select(s => (string?)s.Attribute("name") ?? "")];
    }

    // ---------------------------------------------------------------- parts

    /// <summary>
    /// The shared string table, which is where nearly all text in a workbook actually lives.
    /// </summary>
    /// <remarks>
    /// A string cell holds an index into this table rather than the text, so a column of repeated
    /// labels costs one copy. Rich text splits a single string across several runs, so the runs are
    /// concatenated rather than taking the first.
    /// </remarks>
    private static string[] ReadSharedStrings(ZipArchive archive)
    {
        var document = LoadXml(archive, "xl/sharedStrings.xml");
        if (document is null) return [];

        return [.. document.Root!.Elements(Main + "si").Select(si =>
            string.Concat(si.Descendants(Main + "t").Select(t => t.Value)))];
    }

    /// <summary>
    /// Which cell styles mean "this number is a date".
    /// </summary>
    /// <remarks>
    /// There is no date cell type. A date is a number whose style points at a number format that
    /// looks like a date — either one of the built-in format ids, or a custom format string
    /// containing date tokens. Missing this is what turns a date column into five-digit integers.
    /// </remarks>
    private static HashSet<int> ReadDateStyles(ZipArchive archive)
    {
        var styles = new HashSet<int>();
        var document = LoadXml(archive, "xl/styles.xml");
        if (document is null) return styles;

        // Built-in formats 14-22 and 45-47 are dates and times; the rest are numeric or text.
        var dateFormats = new HashSet<int> { 14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47 };

        foreach (var format in document.Descendants(Main + "numFmt"))
        {
            var id = (int?)format.Attribute("numFmtId");
            var code = (string?)format.Attribute("formatCode");
            if (id is null || code is null) continue;

            // Strip literals and colour tokens before looking for date letters, so a currency
            // format like [Red]"d"#,##0 is not mistaken for one.
            var stripped = StripFormatLiterals(code);
            if (stripped.Contains('y') || stripped.Contains('d')
                || stripped.Contains("mm", StringComparison.Ordinal) || stripped.Contains('h'))
                dateFormats.Add(id.Value);
        }

        var cellFormats = document.Descendants(Main + "cellXfs").FirstOrDefault();
        if (cellFormats is null) return styles;

        var index = 0;
        foreach (var xf in cellFormats.Elements(Main + "xf"))
        {
            var id = (int?)xf.Attribute("numFmtId") ?? 0;
            if (dateFormats.Contains(id)) styles.Add(index);
            index++;
        }

        return styles;
    }

    private static string StripFormatLiterals(string code)
    {
        var builder = new StringBuilder(code.Length);
        var inQuotes = false;
        var inBracket = false;

        for (var i = 0; i < code.Length; i++)
        {
            var c = code[i];

            if (c == '"') { inQuotes = !inQuotes; continue; }
            if (c == '[') { inBracket = true; continue; }
            if (c == ']') { inBracket = false; continue; }
            if (inQuotes || inBracket) continue;
            if (c == '\\') { i++; continue; }

            builder.Append(c);
        }

        return builder.ToString();
    }

    /// <summary>
    /// Locates the requested sheet's XML part.
    /// </summary>
    /// <remarks>
    /// The workbook lists sheets by name and relationship id, and the relationship file maps that
    /// id to a path. Assuming <c>sheet1.xml</c> is the first sheet is wrong often enough to matter:
    /// deleting and re-adding sheets leaves the numbering out of step with the order.
    /// </remarks>
    private static string ResolveSheetPath(ZipArchive archive, ExcelOptions options)
    {
        var workbook = LoadXml(archive, "xl/workbook.xml")
            ?? throw new InvalidDataException("This is not an xlsx workbook: xl/workbook.xml is missing.");

        var sheets = workbook.Descendants(Main + "sheet").ToList();
        if (sheets.Count == 0) throw new InvalidDataException("The workbook holds no sheets.");

        XElement sheet;
        if (options.SheetName is not null)
        {
            sheet = sheets.FirstOrDefault(s => (string?)s.Attribute("name") == options.SheetName)
                ?? throw new ArgumentException(
                    $"No sheet named '{options.SheetName}'. The workbook holds: " +
                    string.Join(", ", sheets.Select(s => (string?)s.Attribute("name"))));
        }
        else
        {
            if (options.SheetIndex < 0 || options.SheetIndex >= sheets.Count)
                throw new ArgumentOutOfRangeException(nameof(options),
                    $"Sheet index {options.SheetIndex} is outside the {sheets.Count} sheets in this workbook.");
            sheet = sheets[options.SheetIndex];
        }

        var id = (string?)sheet.Attribute(Relationships + "id");
        var relationships = LoadXml(archive, "xl/_rels/workbook.xml.rels");

        if (id is not null && relationships is not null)
        {
            var target = relationships.Root!.Elements()
                .FirstOrDefault(r => (string?)r.Attribute("Id") == id)
                ?.Attribute("Target")?.Value;

            if (target is not null)
                return target.StartsWith("/", StringComparison.Ordinal)
                    ? target.TrimStart('/')
                    : "xl/" + target.Replace("../", "");
        }

        // No relationship part: fall back to position, which is right for anything a normal tool wrote.
        return $"xl/worksheets/sheet{sheets.IndexOf(sheet) + 1}.xml";
    }

    // ---------------------------------------------------------------- cells

    /// <summary>
    /// Reads the sheet into a dense grid of strings, filling the gaps absent cells leave.
    /// </summary>
    /// <remarks>
    /// Streamed with <see cref="XmlReader"/> rather than loaded as a document: a worksheet is the
    /// one part of a workbook that can be genuinely large, and holding a million rows of XML nodes
    /// in memory to produce a frame is a waste of both.
    /// </remarks>
    private static List<string?[]> ReadCells(ZipArchive archive, string sheetPath,
        string[] sharedStrings, HashSet<int> dateStyles, ExcelOptions options)
    {
        var entry = archive.GetEntry(sheetPath)
            ?? throw new InvalidDataException($"The workbook has no part at '{sheetPath}'.");

        var rows = new List<List<(int Column, string? Value)>>();
        var width = 0;

        using var stream = entry.Open();
        using var reader = XmlReader.Create(stream, new XmlReaderSettings { IgnoreWhitespace = true });

        List<(int, string?)>? current = null;
        var limit = options.MaxRows > 0 ? options.MaxRows + (options.HasHeader ? 1 : 0) : int.MaxValue;

        while (reader.Read())
        {
            if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "row")
            {
                if (reader.IsEmptyElement) { rows.Add([]); continue; }
                current = [];
                continue;
            }

            if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "row")
            {
                if (current is not null) rows.Add(current);
                current = null;
                if (rows.Count >= limit) break;
                continue;
            }

            if (reader.NodeType != XmlNodeType.Element || reader.LocalName != "c" || current is null) continue;

            var reference = reader.GetAttribute("r");
            var type = reader.GetAttribute("t");
            var styleText = reader.GetAttribute("s");

            // The cell's own reference says which column it is. Counting elements instead shifts
            // every value after a gap, because empty cells simply are not written.
            var column = reference is not null ? ColumnOf(reference) : current.Count;

            string? raw = null;
            if (!reader.IsEmptyElement)
            {
                using var cell = reader.ReadSubtree();
                cell.Read();
                raw = ReadCellValue(cell, type);
            }

            var value = Interpret(raw, type, styleText, sharedStrings, dateStyles);
            current.Add((column, value));
            width = Math.Max(width, column + 1);
        }

        if (current is not null) rows.Add(current);

        var grid = new List<string?[]>(rows.Count);
        foreach (var row in rows)
        {
            var dense = new string?[width];
            foreach (var (column, value) in row)
                if (column < width) dense[column] = value;
            grid.Add(dense);
        }

        return grid;
    }

    private static string? ReadCellValue(XmlReader cell, string? type)
    {
        string? value = null;
        var inline = new StringBuilder();

        while (cell.Read())
        {
            if (cell.NodeType != XmlNodeType.Element) continue;

            switch (cell.LocalName)
            {
                case "v":
                    value = cell.ReadElementContentAsString();
                    break;
                case "t" when type == "inlineStr":
                    inline.Append(cell.ReadElementContentAsString());
                    break;
            }
        }

        return type == "inlineStr" ? inline.ToString() : value;
    }

    /// <summary>Turns a raw cell value into text, resolving shared strings, booleans and dates.</summary>
    private static string? Interpret(string? raw, string? type, string? styleText,
        string[] sharedStrings, HashSet<int> dateStyles)
    {
        if (raw is null) return null;

        switch (type)
        {
            case "s":
                return int.TryParse(raw, out var index) && index >= 0 && index < sharedStrings.Length
                    ? sharedStrings[index]
                    : null;
            case "b":
                return raw == "1" ? "true" : "false";
            case "e":
                return null;        // #DIV/0!, #N/A and friends are missing data, not text
            case "str" or "inlineStr":
                return raw;
        }

        // Untyped means numeric — and possibly a date wearing a number's clothes.
        if (styleText is not null && int.TryParse(styleText, out var style) && dateStyles.Contains(style)
            && double.TryParse(raw, NumberStyles.Any, CultureInfo.InvariantCulture, out var serial))
            return Epoch.AddDays(serial).ToString("O", CultureInfo.InvariantCulture);

        return raw;
    }

    /// <summary>Turns a cell reference like <c>BC12</c> into a zero-based column index.</summary>
    /// <remarks>
    /// Base 26 with no zero digit: A is 1, Z is 26, AA is 27. Treating it as ordinary base 26 puts
    /// every column past Z one place out.
    /// </remarks>
    private static int ColumnOf(string reference)
    {
        var column = 0;
        foreach (var c in reference)
        {
            if (!char.IsAsciiLetter(c)) break;
            column = column * 26 + (char.ToUpperInvariant(c) - 'A' + 1);
        }
        return Math.Max(0, column - 1);
    }

    // ---------------------------------------------------------------- assembly

    private static DataFrame Build(List<string?[]> grid, ExcelOptions options)
    {
        if (grid.Count == 0) return new DataFrame([]);

        var width = grid[0].Length;
        var names = new string[width];

        var start = 0;
        if (options.HasHeader)
        {
            for (var i = 0; i < width; i++)
            {
                var header = grid[0][i];
                names[i] = string.IsNullOrWhiteSpace(header) ? $"column_{i}" : header.Trim();
            }
            start = 1;
        }
        else
        {
            for (var i = 0; i < width; i++) names[i] = $"column_{i}";
        }

        Deduplicate(names);

        var rowCount = grid.Count - start;
        if (options.MaxRows > 0) rowCount = Math.Min(rowCount, options.MaxRows);
        rowCount = Math.Max(0, rowCount);

        var columns = new List<Series>(width);

        for (var i = 0; i < width; i++)
        {
            var raw = new string?[rowCount];
            for (var row = 0; row < rowCount; row++)
            {
                var source = grid[start + row];
                var value = i < source.Length ? source[i] : null;
                raw[row] = string.IsNullOrWhiteSpace(value) ? null : value;
            }

            columns.Add(Convert(names[i], raw, Infer(raw, options)));
        }

        return new DataFrame(columns);
    }

    private static DataType Infer(string?[] values, ExcelOptions options)
    {
        var limit = options.TypeInferenceRows <= 0
            ? values.Length
            : Math.Min(options.TypeInferenceRows, values.Length);

        var numeric = true;
        var boolean = true;
        var date = true;
        var seenAny = false;

        for (var i = 0; i < limit; i++)
        {
            var v = values[i];
            if (v is null) continue;
            seenAny = true;

            if (numeric && !double.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out _)) numeric = false;
            if (boolean && v is not ("true" or "false")) boolean = false;
            if (date && !DateTime.TryParse(v, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out _)) date = false;
            if (!numeric && !boolean && !date) break;
        }

        if (!seenAny) return DataType.Text;
        if (boolean) return DataType.Boolean;
        if (numeric) return DataType.Numeric;
        if (date) return DataType.DateTime;
        return DataType.Text;
    }

    private static Series Convert(string name, string?[] raw, DataType type)
    {
        switch (type)
        {
            case DataType.Numeric:
            {
                var values = new double[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null
                        && double.TryParse(raw[i], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)
                        ? v : double.NaN;
                return new NumericSeries(name, values);
            }
            case DataType.Boolean:
            {
                var values = new bool?[raw.Length];
                for (var i = 0; i < raw.Length; i++) values[i] = raw[i] is null ? null : raw[i] == "true";
                return new BooleanSeries(name, values);
            }
            case DataType.DateTime:
            {
                var values = new DateTime?[raw.Length];
                for (var i = 0; i < raw.Length; i++)
                    values[i] = raw[i] is not null
                        && DateTime.TryParse(raw[i], CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var d)
                        ? d : null;
                return new DateTimeSeries(name, values);
            }
            default:
                return new TextSeries(name, raw);
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

    private static XDocument? LoadXml(ZipArchive archive, string path)
    {
        var entry = archive.GetEntry(path);
        if (entry is null) return null;

        using var stream = entry.Open();
        return XDocument.Load(stream);
    }
}

/// <summary>
/// Writes a <see cref="DataFrame"/> out as a minimal <c>.xlsx</c> workbook.
/// </summary>
/// <remarks>
/// Enough of the format for Excel, LibreOffice and every reader to open the file: the content
/// types, a workbook part, one worksheet and the relationships tying them together. Text is written
/// inline rather than through a shared string table — that costs size on repetitive data and
/// removes a whole part and its index bookkeeping, which is the right trade for an exporter.
/// </remarks>
public static class ExcelWriter
{
    /// <summary>Writes the frame to <paramref name="path"/> as a single-sheet workbook.</summary>
    public static void Write(DataFrame frame, string path, string sheetName = "Sheet1")
    {
        using var stream = File.Create(path);
        Write(frame, stream, sheetName);
    }

    /// <summary>Writes the frame to a stream as a single-sheet workbook.</summary>
    public static void Write(DataFrame frame, Stream stream, string sheetName = "Sheet1")
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(stream);

        using var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true);

        WriteEntry(archive, "[Content_Types].xml", ContentTypes());
        WriteEntry(archive, "_rels/.rels", RootRelationships());
        WriteEntry(archive, "xl/workbook.xml", Workbook(sheetName));
        WriteEntry(archive, "xl/_rels/workbook.xml.rels", WorkbookRelationships());
        WriteEntry(archive, "xl/styles.xml", Styles());
        WriteEntry(archive, "xl/worksheets/sheet1.xml", Worksheet(frame));
    }

    private static string ContentTypes() =>
        """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
          <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
          <Default Extension="xml" ContentType="application/xml"/>
          <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
          <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
          <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
        </Types>
        """;

    private static string RootRelationships() =>
        """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
        </Relationships>
        """;

    private static string Workbook(string sheetName) =>
        $"""
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
          <sheets><sheet name="{Escape(sheetName)}" sheetId="1" r:id="rId1"/></sheets>
        </workbook>
        """;

    private static string WorkbookRelationships() =>
        """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
          <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
        </Relationships>
        """;

    /// <summary>
    /// Two cell formats: index 0 is general, index 1 is an ISO date.
    /// </summary>
    /// <remarks>
    /// A date written without a style would open as a five-digit serial number, which is the same
    /// trap the reader has to work around from the other side.
    /// </remarks>
    private static string Styles() =>
        """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <numFmts count="1"><numFmt numFmtId="164" formatCode="yyyy\-mm\-dd\ hh:mm:ss"/></numFmts>
          <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
          <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
          <borders count="1"><border/></borders>
          <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
          <cellXfs count="2">
            <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
            <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
          </cellXfs>
        </styleSheet>
        """;

    private static readonly DateTime Epoch = new(1899, 12, 30);

    private static string Worksheet(DataFrame frame)
    {
        var builder = new StringBuilder();
        builder.Append("""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>""");
        builder.Append("""<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>""");

        var columns = frame.ColumnNames.ToArray();

        builder.Append("<row r=\"1\">");
        for (var i = 0; i < columns.Length; i++)
            builder.Append($"""<c r="{Reference(i, 1)}" t="inlineStr"><is><t>{Escape(columns[i])}</t></is></c>""");
        builder.Append("</row>");

        for (var row = 0; row < frame.RowCount; row++)
        {
            builder.Append($"<row r=\"{row + 2}\">");

            for (var i = 0; i < columns.Length; i++)
            {
                var series = frame[columns[i]];
                if (series.IsMissing(row)) continue;      // an absent cell is how xlsx spells "empty"

                var reference = Reference(i, row + 2);
                builder.Append(series switch
                {
                    NumericSeries numeric =>
                        $"""<c r="{reference}"><v>{numeric[row].ToString("R", CultureInfo.InvariantCulture)}</v></c>""",
                    BooleanSeries boolean =>
                        $"""<c r="{reference}" t="b"><v>{(boolean[row]!.Value ? 1 : 0)}</v></c>""",
                    DateTimeSeries date =>
                        $"""<c r="{reference}" s="1"><v>{(date[row]!.Value - Epoch).TotalDays.ToString("R", CultureInfo.InvariantCulture)}</v></c>""",
                    _ =>
                        $"""<c r="{reference}" t="inlineStr"><is><t>{Escape(series.GetValue(row)?.ToString() ?? "")}</t></is></c>""",
                });
            }

            builder.Append("</row>");
        }

        builder.Append("</sheetData></worksheet>");
        return builder.ToString();
    }

    /// <summary>Turns a zero-based column and a one-based row into a cell reference like <c>AA7</c>.</summary>
    private static string Reference(int column, int row)
    {
        var name = new StringBuilder();
        var value = column + 1;

        while (value > 0)
        {
            var remainder = (value - 1) % 26;     // no zero digit: A is 1, so shift before dividing
            name.Insert(0, (char)('A' + remainder));
            value = (value - 1) / 26;
        }

        return name.Append(row).ToString();
    }

    private static string Escape(string value) => new XText(value).ToString();

    private static void WriteEntry(ZipArchive archive, string path, string content)
    {
        var entry = archive.CreateEntry(path, CompressionLevel.Optimal);
        using var stream = entry.Open();
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(content);
    }
}
