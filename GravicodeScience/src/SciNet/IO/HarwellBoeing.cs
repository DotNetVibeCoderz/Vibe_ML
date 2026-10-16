using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// A Fortran edit descriptor for one repeated field, as used in Harwell-Boeing headers:
/// <c>(26I3)</c>, <c>(3E25.16)</c>, <c>(1P,4D20.12)</c>, ...
/// </summary>
/// <param name="Repeat">Fields per line.</param>
/// <param name="Kind"><c>I</c>, <c>E</c>, <c>D</c>, <c>F</c> or <c>G</c>.</param>
/// <param name="Width">Field width in characters.</param>
/// <param name="Digits">Digits after the decimal point (0 for integers).</param>
public sealed record FortranFormat(int Repeat, char Kind, int Width, int Digits)
{
    private static readonly Regex Pattern = new(@"^\(\s*(?:\d*P\s*,?\s*)?(\d*)\s*([IEDFG])\s*(\d+)(?:\s*\.\s*(\d+))?(?:\s*E\s*\d+)?\s*\)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    /// <summary>Parses a parenthesised Fortran format such as <c>(10I8)</c>.</summary>
    public static FortranFormat Parse(string text)
    {
        var m = Pattern.Match(text.Trim());
        if (!m.Success) throw new FormatException($"Unsupported Fortran format '{text}'.");
        var repeat = m.Groups[1].Value.Length == 0 ? 1 : int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        var width = int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);
        var digits = m.Groups[4].Success ? int.Parse(m.Groups[4].Value, CultureInfo.InvariantCulture) : 0;
        if (repeat <= 0 || width <= 0) throw new FormatException($"Fortran format '{text}' has a zero repeat or width.");
        return new FortranFormat(repeat, char.ToUpperInvariant(m.Groups[2].Value[0]), width, digits);
    }

    /// <summary>True for an integer descriptor.</summary>
    public bool IsInteger => Kind == 'I';

    /// <summary>The descriptor in Fortran syntax, e.g. <c>(3E25.16)</c>.</summary>
    public override string ToString() => IsInteger ? $"({Repeat}I{Width})" : $"({Repeat}{Kind}{Width}.{Digits})";

    /// <summary>SciPy's <c>IntFormat.from_number</c>: one digit more than the largest value needs, as many per 80-column line as fit.</summary>
    public static FortranFormat ForIntegers(long max)
    {
        var width = Digits10(Math.Abs(max)) + 1 + (max < 0 ? 1 : 0);
        return new FortranFormat(80 / width, 'I', width, 0);
    }

    /// <summary>
    /// SciPy's <c>ExpFormat.from_number</c> for doubles: 16 fractional digits (17 significant, so
    /// every double round-trips), a 4-character exponent allowance, one sign column → E25.16, three per line.
    /// </summary>
    public static FortranFormat ForReals() => new(3, 'E', 25, 16);

    private static int Digits10(long n) => n == 0 ? 1 : (int)Math.Floor(Math.Log10(n)) + 1;
}

/// <summary>The header of a Harwell-Boeing file.</summary>
/// <param name="Title">Up to 72 characters.</param>
/// <param name="Key">Up to 8 characters.</param>
/// <param name="TotalLines">Data lines (excluding the header).</param>
/// <param name="PointerLines">Lines holding the column pointers.</param>
/// <param name="IndicesLines">Lines holding the row indices.</param>
/// <param name="ValuesLines">Lines holding the values (0 for a pattern matrix).</param>
/// <param name="RhsLines">Lines holding right-hand sides (ignored on read).</param>
/// <param name="MatrixType">Three letters: value type (R real, P pattern, I integer, C complex), structure (U unsymmetric, S symmetric, Z skew, H hermitian, R rectangular), storage (A assembled, E elemental).</param>
/// <param name="Rows">Rows.</param>
/// <param name="Columns">Columns.</param>
/// <param name="Nnz">Stored entries.</param>
/// <param name="PointerFormat">Format of the pointer lines.</param>
/// <param name="IndicesFormat">Format of the index lines.</param>
/// <param name="ValuesFormat">Format of the value lines (null for a pattern matrix).</param>
public sealed record HarwellBoeingInfo(
    string Title, string Key, int TotalLines, int PointerLines, int IndicesLines, int ValuesLines, int RhsLines,
    string MatrixType, int Rows, int Columns, int Nnz,
    FortranFormat PointerFormat, FortranFormat IndicesFormat, FortranFormat? ValuesFormat);

/// <summary>
/// Harwell-Boeing sparse matrix files — <c>scipy.io.hb_read</c> / <c>hb_write</c>, extended: SciPy
/// reads only real unsymmetric assembled (RUA) matrices; this reader also takes symmetric and
/// skew-symmetric storage (expanded to the full matrix), pattern and integer values, rectangular
/// structure, and files with right-hand sides (skipped).
/// </summary>
/// <remarks>
/// Fields are cut at their declared Fortran widths, so values packed with no separating space
/// (<c>-0.2D+01-0.6D+01</c>, common in the original collection) read correctly; a line whose
/// fields are all whitespace-separated is split on whitespace instead, which also accepts files
/// whose writer did not honour its own declared width (SciPy writes 24-column values under an
/// <c>E25.16</c> descriptor). Fortran <c>D</c> exponents and exponents without a letter
/// (<c>1.5-300</c>) are understood. Complex and elemental matrices are refused.
/// </remarks>
public static class HarwellBoeing
{
    /// <summary>Reads a Harwell-Boeing file into a CSC matrix.</summary>
    public static CscMatrix HbRead(string path)
    {
        using var r = new StreamReader(path, Encoding.Latin1);
        return HbRead(r, out _);
    }

    /// <summary>Reads a Harwell-Boeing stream into a CSC matrix.</summary>
    public static CscMatrix HbRead(Stream stream)
    {
        using var r = new StreamReader(stream, Encoding.Latin1, leaveOpen: true);
        return HbRead(r, out _);
    }

    /// <summary>Reads a Harwell-Boeing file and returns its header as well.</summary>
    public static CscMatrix HbRead(string path, out HarwellBoeingInfo info)
    {
        using var r = new StreamReader(path, Encoding.Latin1);
        return HbRead(r, out info);
    }

    private static string Field(string line, int start, int width) =>
        start >= line.Length ? "" : line.Substring(start, Math.Min(width, line.Length - start));

    private static int Int(string s, string what)
    {
        s = s.Trim();
        if (s.Length == 0) return 0;
        if (!int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
            throw new InvalidDataException($"Harwell-Boeing header: '{s}' is not an integer ({what}).");
        return v;
    }

    private static HarwellBoeingInfo ReadHeader(TextReader r)
    {
        string Line(string what) => r.ReadLine() ?? throw new InvalidDataException($"Harwell-Boeing file ends before its {what} header line.");
        var l1 = Line("title");
        var l2 = Line("line-count");
        var l3 = Line("type");
        var l4 = Line("format");
        var title = Field(l1, 0, 72).TrimEnd();
        var key = Field(l1, 72, 8).Trim();
        var total = Int(Field(l2, 0, 14), "total lines");
        var ptrLines = Int(Field(l2, 14, 14), "pointer lines");
        var indLines = Int(Field(l2, 28, 14), "index lines");
        var valLines = Int(Field(l2, 42, 14), "value lines");
        var rhsLines = Int(Field(l2, 56, 14), "rhs lines");
        var type = Field(l3, 0, 3).ToUpperInvariant();
        if (type.Length != 3) throw new InvalidDataException($"Harwell-Boeing matrix type '{type}' is not three letters.");
        var rows = Int(Field(l3, 14, 14), "rows");
        var cols = Int(Field(l3, 28, 14), "columns");
        var nnz = Int(Field(l3, 42, 14), "entries");
        var ptrText = Field(l4, 0, 16);
        var indText = Field(l4, 16, 16);
        var valText = Field(l4, 32, 20);
        FortranFormat ptrFmt, indFmt;
        FortranFormat? valFmt = null;
        try
        {
            ptrFmt = FortranFormat.Parse(ptrText);
            indFmt = FortranFormat.Parse(indText);
            if (type[0] != 'P' && valText.Trim().Length > 0) valFmt = FortranFormat.Parse(valText);
        }
        catch (FormatException)
        {
            // Not column-aligned: fall back to whitespace-separated descriptors.
            var parts = l4.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) throw new InvalidDataException($"Harwell-Boeing format line '{l4}' does not hold pointer and index formats.");
            ptrFmt = FortranFormat.Parse(parts[0]);
            indFmt = FortranFormat.Parse(parts[1]);
            if (type[0] != 'P' && parts.Length > 2) valFmt = FortranFormat.Parse(parts[2]);
        }
        if (rhsLines > 0) Line("right-hand-side");
        return new HarwellBoeingInfo(title, key, total, ptrLines, indLines, valLines, rhsLines, type, rows, cols, nnz, ptrFmt, indFmt, valFmt);
    }

    private static CscMatrix HbRead(TextReader r, out HarwellBoeingInfo info)
    {
        info = ReadHeader(r);
        var type = info.MatrixType;
        if (type[0] == 'C') throw new NotSupportedException("Complex Harwell-Boeing matrices are not supported (CscMatrix is real).");
        if (type[0] is not ('R' or 'P' or 'I')) throw new InvalidDataException($"Unknown Harwell-Boeing value type '{type[0]}'.");
        if (type[1] == 'H') throw new NotSupportedException("Hermitian Harwell-Boeing matrices are complex and not supported.");
        if (type[1] is not ('U' or 'S' or 'Z' or 'R')) throw new InvalidDataException($"Unknown Harwell-Boeing structure '{type[1]}'.");
        if (type[2] == 'E') throw new NotSupportedException("Elemental (unassembled) Harwell-Boeing matrices are not supported.");
        if (type[2] != 'A') throw new InvalidDataException($"Unknown Harwell-Boeing storage '{type[2]}'.");
        if (type[0] != 'P' && info.ValuesFormat is null) throw new InvalidDataException("Harwell-Boeing header has no value format for a non-pattern matrix.");
        if (type[1] is 'S' or 'Z' && info.Rows != info.Columns) throw new InvalidDataException("A symmetric Harwell-Boeing matrix must be square.");

        var ptr = ReadFields(r, info.PointerLines, info.Columns + 1, info.PointerFormat, "pointer");
        var ind = ReadFields(r, info.IndicesLines, info.Nnz, info.IndicesFormat, "index");
        var val = type[0] == 'P'
            ? Enumerable.Repeat(1.0, info.Nnz).ToArray()
            : ReadFields(r, info.ValuesLines, info.Nnz, info.ValuesFormat!, "value");

        var indptr = new int[info.Columns + 1];
        for (var j = 0; j <= info.Columns; j++) indptr[j] = checked((int)ptr[j] - 1);
        if (indptr[0] != 0 || indptr[^1] != info.Nnz) throw new InvalidDataException($"Harwell-Boeing pointers run {indptr[0] + 1}..{indptr[^1] + 1}, expected 1..{info.Nnz + 1}.");
        var rowsIdx = new int[info.Nnz];
        for (var k = 0; k < info.Nnz; k++)
        {
            rowsIdx[k] = checked((int)ind[k] - 1);
            if ((uint)rowsIdx[k] >= (uint)info.Rows) throw new InvalidDataException($"Harwell-Boeing row index {ind[k]} outside 1..{info.Rows}.");
        }
        var m = new CscMatrix(val, rowsIdx, indptr, info.Rows, info.Columns, copy: false);
        if (type[1] is 'U' or 'R') return m;

        // Symmetric storage holds one triangle; mirror the off-diagonal entries.
        var sign = type[1] == 'Z' ? -1.0 : 1.0;
        var ri = new List<int>(2 * info.Nnz);
        var ci = new List<int>(2 * info.Nnz);
        var vv = new List<double>(2 * info.Nnz);
        for (var j = 0; j < info.Columns; j++)
            for (var k = indptr[j]; k < indptr[j + 1]; k++)
            {
                ri.Add(rowsIdx[k]); ci.Add(j); vv.Add(val[k]);
                if (rowsIdx[k] != j) { ri.Add(j); ci.Add(rowsIdx[k]); vv.Add(sign * val[k]); }
            }
        return CscMatrix.FromTriplets(info.Rows, info.Columns, ri.ToArray(), ci.ToArray(), vv.ToArray());
    }

    /// <summary>Reads <paramref name="count"/> numbers from exactly <paramref name="lines"/> lines.</summary>
    private static double[] ReadFields(TextReader r, int lines, int count, FortranFormat fmt, string what)
    {
        var result = new double[count];
        var k = 0;
        for (var l = 0; l < lines; l++)
        {
            var line = r.ReadLine() ?? throw new InvalidDataException($"Harwell-Boeing file ends inside its {what} block.");
            var want = Math.Min(fmt.Repeat, count - k);
            if (want <= 0) continue;
            var tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == want)
            {
                foreach (var t in tokens) result[k++] = ParseNumber(t, what);
                continue;
            }
            for (var f = 0; f < want; f++)
            {
                var start = f * fmt.Width;
                var field = start >= line.Length ? "" : line.Substring(start, Math.Min(fmt.Width, line.Length - start));
                result[k++] = ParseNumber(field, what);
            }
        }
        if (k != count) throw new InvalidDataException($"Harwell-Boeing {what} block holds {k} numbers, {count} expected.");
        return result;
    }

    private static double ParseNumber(string field, string what)
    {
        var s = field.Trim();
        if (s.Length == 0) return 0; // a blank Fortran field reads as zero
        s = s.Replace('D', 'E').Replace('d', 'E');
        if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) return v;
        // Fortran drops the exponent letter when the exponent needs three digits: 0.1234-102.
        for (var i = s.Length - 1; i > 0; i--)
            if ((s[i] == '+' || s[i] == '-') && char.IsDigit(s[i - 1]) && s.IndexOf('E') < 0)
                if (double.TryParse(s[..i] + "E" + s[i..], NumberStyles.Float, CultureInfo.InvariantCulture, out v)) return v;
        throw new InvalidDataException($"Harwell-Boeing {what} field '{field}' is not a number.");
    }

    /// <summary>
    /// Writes a sparse matrix in Harwell-Boeing format (<c>scipy.io.hb_write</c>), choosing the
    /// pointer/index/value formats the way SciPy does, but padding every field to its declared width.
    /// </summary>
    /// <param name="path">Destination path.</param>
    /// <param name="matrix">The matrix.</param>
    /// <param name="title">Title (at most 72 characters).</param>
    /// <param name="key">Key (at most 8 characters).</param>
    /// <param name="matrixType">
    /// <c>RUA</c> (default), <c>RRA</c>, <c>PUA</c> (pattern), <c>IUA</c> (integer), or the
    /// symmetric/skew forms <c>RSA</c>, <c>RZA</c>, <c>PSA</c>, <c>ISA</c>, which store the lower
    /// triangle only — the matrix is checked to be symmetric (skew-symmetric) first.
    /// </param>
    public static void HbWrite(string path, SparseMatrixBase matrix, string title = "Default title", string key = "0", string matrixType = "RUA")
    {
        using var f = File.Create(path);
        HbWrite(f, matrix, title, key, matrixType);
    }

    /// <summary>Writes a sparse matrix in Harwell-Boeing format to a stream.</summary>
    /// <param name="stream">Destination.</param>
    /// <param name="matrix">The matrix.</param>
    /// <param name="title">Title (at most 72 characters).</param>
    /// <param name="key">Key (at most 8 characters).</param>
    /// <param name="matrixType">See the path overload.</param>
    public static void HbWrite(Stream stream, SparseMatrixBase matrix, string title = "Default title", string key = "0", string matrixType = "RUA")
    {
        if (title.Length > 72) throw new ArgumentException("Title cannot be longer than 72 characters.", nameof(title));
        if (key.Length > 8) throw new ArgumentException("Key cannot be longer than 8 characters.", nameof(key));
        var type = matrixType.ToUpperInvariant();
        if (type.Length != 3 || type[0] is not ('R' or 'P' or 'I') || type[1] is not ('U' or 'S' or 'Z' or 'R') || type[2] != 'A')
            throw new ArgumentException($"Matrix type '{matrixType}' is not writable; use R/P/I + U/S/Z/R + A.", nameof(matrixType));
        var csc = matrix.ToCsc().SumDuplicates().SortIndices();
        int rows = csc.Rows, cols = csc.Columns;
        var ptrSrc = csc.Indptr;
        var idxSrc = csc.Indices;
        var valSrc = csc.Data;

        var ptr = new List<int>(cols + 1) { 0 };
        var idx = new List<int>(csc.Nnz);
        var val = new List<double>(csc.Nnz);
        if (type[1] is 'S' or 'Z')
        {
            if (rows != cols) throw new ArgumentException("A symmetric Harwell-Boeing matrix must be square.");
            var expect = type[1] == 'S' ? "symmetric" : "skew-symmetric";
            var t = csc.ToCsr();
            for (var j = 0; j < cols; j++)
            {
                for (var k = ptrSrc[j]; k < ptrSrc[j + 1]; k++)
                {
                    var i = idxSrc[k];
                    var mirror = t[j, i];
                    var ok = type[1] == 'S' ? mirror == valSrc[k] : mirror == -valSrc[k];
                    if (!ok) throw new ArgumentException($"Matrix is not {expect}: entry ({i}, {j}) = {valSrc[k]} but ({j}, {i}) = {mirror}.");
                    if (i < j || (i == j && type[1] == 'Z')) continue;
                    idx.Add(i);
                    val.Add(valSrc[k]);
                }
                ptr.Add(idx.Count);
            }
        }
        else
        {
            for (var j = 0; j <= cols; j++) if (j > 0) ptr.Add(ptrSrc[j]);
            for (var k = 0; k < csc.Nnz; k++) { idx.Add(idxSrc[k]); val.Add(valSrc[k]); }
        }
        if (type[0] == 'I')
            foreach (var v in val)
                if (v != Math.Floor(v) || Math.Abs(v) > long.MaxValue) throw new ArgumentException($"Value {v} is not an integer; use matrix type R.. instead of I...");

        var nnz = idx.Count;
        var ptrFmt = FortranFormat.ForIntegers(ptr.Count == 0 ? 1 : ptr.Max() + 1L);
        var idxFmt = FortranFormat.ForIntegers(nnz == 0 ? 1 : idx.Max() + 1L);
        FortranFormat? valFmt = type[0] switch
        {
            'P' => null,
            'I' => FortranFormat.ForIntegers(-(long)(val.Count == 0 ? 1 : val.Max(Math.Abs))),
            _ => FortranFormat.ForReals(),
        };
        static int Lines(int n, FortranFormat f) => (n + f.Repeat - 1) / f.Repeat;
        var pl = Lines(ptr.Count, ptrFmt);
        var il = Lines(nnz, idxFmt);
        var vl = valFmt is null ? 0 : Lines(nnz, valFmt);

        var sb = new StringBuilder();
        sb.Append(title.PadRight(72)).Append(key.PadRight(8)).Append('\n');
        sb.Append(I14(pl + il + vl)).Append(I14(pl)).Append(I14(il)).Append(I14(vl)).Append('\n');
        sb.Append(type.PadRight(14)).Append(I14(rows)).Append(I14(cols)).Append(I14(nnz)).Append(I14(0)).Append('\n');
        sb.Append(ptrFmt.ToString().PadRight(16)).Append(idxFmt.ToString().PadRight(16)).Append((valFmt?.ToString() ?? "").PadRight(20)).Append('\n');
        Block(sb, ptr.Select(p => (double)(p + 1)).ToList(), ptrFmt);
        Block(sb, idx.Select(i => (double)(i + 1)).ToList(), idxFmt);
        if (valFmt is not null) Block(sb, val, valFmt);
        var bytes = Encoding.ASCII.GetBytes(sb.ToString());
        stream.Write(bytes);

        static string I14(long v) => v.ToString(CultureInfo.InvariantCulture).PadLeft(14);
    }

    private static void Block(StringBuilder sb, List<double> values, FortranFormat fmt)
    {
        for (var k = 0; k < values.Count; k++)
        {
            var v = values[k];
            var s = fmt.IsInteger
                ? ((long)v).ToString(CultureInfo.InvariantCulture)
                // Standard "E16": 17 significant digits, exact for every double. (A custom
                // "0.000…E+00" pattern looks equivalent but .NET caps custom formats at 15
                // significant digits, which silently loses the last two.)
                : v.ToString("E" + fmt.Digits.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            if (s.Length >= fmt.Width) throw new InvalidOperationException($"Value {v} does not fit format {fmt}.");
            sb.Append(s.PadLeft(fmt.Width));
            if ((k + 1) % fmt.Repeat == 0 || k == values.Count - 1) sb.Append('\n');
        }
    }
}
