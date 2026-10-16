using System.Buffers;
using System.Buffers.Text;
using System.Globalization;
using System.Numerics;
using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;

namespace Gravicode.Science.SciNet.IO;

/// <summary>Header of a Matrix Market file (<c>scipy.io.mminfo</c>).</summary>
/// <param name="Rows">Number of rows.</param>
/// <param name="Columns">Number of columns.</param>
/// <param name="Entries">Stored entries of a coordinate file; rows x columns for an array file (as SciPy reports it).</param>
/// <param name="Format"><c>coordinate</c> or <c>array</c>.</param>
/// <param name="Field"><c>real</c>, <c>integer</c>, <c>unsigned-integer</c>, <c>complex</c> or <c>pattern</c>.</param>
/// <param name="Symmetry"><c>general</c>, <c>symmetric</c>, <c>skew-symmetric</c> or <c>hermitian</c>.</param>
public sealed record MatrixMarketInfo(int Rows, int Columns, long Entries, string Format, string Field, string Symmetry);

/// <summary>
/// Result of <see cref="MatrixMarket.MmRead(string)"/>: a dense array for <c>array</c> files, a
/// COO matrix for <c>coordinate</c> files, with symmetric storage already expanded to the full
/// matrix (as SciPy's <c>mmread</c> does).
/// </summary>
public sealed class MatrixMarketData
{
    internal MatrixMarketData(MatrixMarketInfo info, NdArray? dense, ComplexNdArray? denseComplex, CooMatrix? sparse, double[]? sparseImaginary)
    {
        Info = info;
        Dense = dense;
        DenseComplex = denseComplex;
        Sparse = sparse;
        _imag = sparseImaginary;
    }

    private readonly double[]? _imag;

    /// <summary>The file header.</summary>
    public MatrixMarketInfo Info { get; }

    /// <summary>True for a coordinate (sparse) file.</summary>
    public bool IsSparse => Sparse is not null;

    /// <summary>True for a complex field.</summary>
    public bool IsComplex => Info.Field == "complex";

    /// <summary>The matrix of a real/integer array file; null otherwise.</summary>
    public NdArray? Dense { get; }

    /// <summary>The matrix of a complex array file; null otherwise.</summary>
    public ComplexNdArray? DenseComplex { get; }

    /// <summary>The matrix of a coordinate file (the real part when complex; ones for <c>pattern</c>); null otherwise.</summary>
    public CooMatrix? Sparse { get; }

    /// <summary>Imaginary parts of a complex coordinate file, aligned with <c>Sparse.Data</c>.</summary>
    public ReadOnlySpan<double> SparseImaginary => _imag;

    /// <summary>The matrix as a dense real array whatever the format; throws for complex data.</summary>
    public NdArray ToDense()
    {
        if (IsComplex) throw new InvalidOperationException("The matrix is complex; use DenseComplex or Sparse + SparseImaginary.");
        return Dense ?? Sparse!.ToDense();
    }
}

/// <summary>
/// Matrix Market exchange format — <c>scipy.io.mmread</c>, <c>mmwrite</c>, <c>mminfo</c>.
/// </summary>
/// <remarks>
/// The body is parsed straight from UTF-8 bytes with no per-line strings, split into chunks at
/// line boundaries and parsed in parallel; writing formats chunks in parallel the same way.
/// Numbers are written in .NET's shortest round-trip form unless a precision is given, so a
/// write/read cycle is exact.
/// </remarks>
public static class MatrixMarket
{
    private const int ParallelThresholdBytes = 1 << 20;

    // ------------------------------------------------------------------ reading

    /// <summary>Reads only the header (<c>scipy.io.mminfo</c>).</summary>
    public static MatrixMarketInfo MmInfo(string path)
    {
        using var f = File.OpenRead(path);
        return MmInfo(f);
    }

    /// <summary>Reads only the header of a Matrix Market stream.</summary>
    public static MatrixMarketInfo MmInfo(Stream stream)
    {
        // The header and size line are at the top; 64 KB is ample unless comments are enormous.
        var buf = new byte[1 << 16];
        var n = stream.ReadAtLeast(buf, buf.Length, throwOnEndOfStream: false);
        while (true)
        {
            try { return ParseHeader(buf.AsSpan(0, n), out _); }
            catch (InvalidDataException) when (n == buf.Length)
            {
                Array.Resize(ref buf, buf.Length * 2);
                n += stream.ReadAtLeast(buf.AsSpan(n), buf.Length - n, throwOnEndOfStream: false);
            }
        }
    }

    /// <summary>Reads a Matrix Market file (<c>scipy.io.mmread</c>).</summary>
    public static MatrixMarketData MmRead(string path) => MmRead(File.ReadAllBytes(path));

    /// <summary>Reads a Matrix Market stream (to its end).</summary>
    public static MatrixMarketData MmRead(Stream stream) => MmRead(MatFile.ReadAll(stream));

    /// <summary>Reads a Matrix Market file and returns it as a sparse COO matrix (array files are converted).</summary>
    public static CooMatrix MmReadSparse(string path)
    {
        var d = MmRead(path);
        if (d.IsComplex) throw new InvalidOperationException("The matrix is complex; use MmRead.");
        return d.Sparse ?? CooMatrix.FromDense(d.Dense!);
    }

    private static MatrixMarketData MmRead(byte[] file)
    {
        var info = ParseHeader(file, out var bodyStart);
        var body = new ReadOnlyMemory<byte>(file, bodyStart, file.Length - bodyStart);
        return info.Format == "coordinate" ? ReadCoordinate(info, body) : ReadArray(info, body);
    }

    private static MatrixMarketInfo ParseHeader(ReadOnlySpan<byte> s, out int bodyStart)
    {
        var pos = 0;
        var first = NextLine(s, ref pos);
        var banner = Encoding.ASCII.GetString(first).Trim();
        var parts = banner.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 5 || !parts[0].Equals("%%MatrixMarket", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Not a Matrix Market file: the first line must be '%%MatrixMarket matrix <format> <field> <symmetry>'.");
        if (!parts[1].Equals("matrix", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"Matrix Market object '{parts[1]}' is not supported; only 'matrix'.");
        var format = parts[2].ToLowerInvariant();
        var field = parts[3].ToLowerInvariant();
        var symmetry = parts[4].ToLowerInvariant();
        if (format is not ("coordinate" or "array")) throw new InvalidDataException($"Unknown Matrix Market format '{format}'.");
        if (field is not ("real" or "integer" or "unsigned-integer" or "complex" or "pattern" or "double")) throw new InvalidDataException($"Unknown Matrix Market field '{field}'.");
        if (field == "double") field = "real";
        if (symmetry is not ("general" or "symmetric" or "skew-symmetric" or "hermitian")) throw new InvalidDataException($"Unknown Matrix Market symmetry '{symmetry}'.");
        if (format == "array" && field == "pattern") throw new InvalidDataException("Pattern field is only valid for coordinate files.");
        if (symmetry == "hermitian" && field != "complex") throw new InvalidDataException("Hermitian symmetry requires the complex field.");

        while (true)
        {
            if (pos >= s.Length) throw new InvalidDataException("Matrix Market file ends before its size line.");
            var line = NextLine(s, ref pos);
            var t = line.Trim((byte)' ').Trim((byte)'\t').Trim((byte)'\r');
            if (t.Length == 0 || t[0] == (byte)'%') continue;
            var p = 0;
            var rows = (int)ParseLong(t, ref p);
            var cols = (int)ParseLong(t, ref p);
            long entries;
            if (format == "coordinate") entries = ParseLong(t, ref p);
            else entries = (long)rows * cols;
            if (rows < 0 || cols < 0 || entries < 0) throw new InvalidDataException("Matrix Market size line has negative values.");
            if (symmetry != "general" && rows != cols) throw new InvalidDataException($"A {symmetry} matrix must be square, header says {rows} x {cols}.");
            bodyStart = pos;
            return new MatrixMarketInfo(rows, cols, entries, format, field, symmetry);
        }
    }

    private static ReadOnlySpan<byte> NextLine(ReadOnlySpan<byte> s, ref int pos)
    {
        var rest = s[pos..];
        var nl = rest.IndexOf((byte)'\n');
        if (nl < 0) { pos = s.Length; return rest; }
        pos += nl + 1;
        return rest[..nl];
    }

    private static bool IsSpace(byte c) => c is (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n' or (byte)'\f' or (byte)'\v';

    private static long ParseLong(ReadOnlySpan<byte> s, ref int p)
    {
        while (p < s.Length && IsSpace(s[p])) p++;
        if (p >= s.Length) throw new InvalidDataException("Matrix Market: expected an integer, found end of line.");
        var neg = false;
        if (s[p] == (byte)'-') { neg = true; p++; }
        else if (s[p] == (byte)'+') p++;
        var start = p;
        long v = 0;
        while (p < s.Length && (uint)(s[p] - '0') <= 9)
        {
            v = checked(v * 10 + (s[p] - '0'));
            p++;
        }
        if (p == start || (p < s.Length && !IsSpace(s[p])))
            throw new InvalidDataException($"Matrix Market: '{Encoding.ASCII.GetString(Token(s, start))}' is not an integer.");
        return neg ? -v : v;
    }

    private static ReadOnlySpan<byte> Token(ReadOnlySpan<byte> s, int start)
    {
        var e = start;
        while (e < s.Length && !IsSpace(s[e])) e++;
        return s[start..e];
    }

    private static double ParseDouble(ReadOnlySpan<byte> s, ref int p)
    {
        while (p < s.Length && IsSpace(s[p])) p++;
        if (p >= s.Length) throw new InvalidDataException("Matrix Market: expected a number, found end of line.");
        var start = p;
        while (p < s.Length && !IsSpace(s[p])) p++;
        var tok = s[start..p];
        if (IOFastFloat.TryParse(tok, out var v)) return v;
        // Rarer spellings: inf/nan, Fortran 'D' exponents.
        Span<char> chars = stackalloc char[Math.Min(tok.Length, 64)];
        if (tok.Length <= 64)
        {
            for (var i = 0; i < tok.Length; i++) chars[i] = (char)(tok[i] is (byte)'D' or (byte)'d' ? (byte)'e' : tok[i]);
            if (double.TryParse(chars, NumberStyles.Float, CultureInfo.InvariantCulture, out v)) return v;
            var lower = new string(chars).ToLowerInvariant();
            if (lower is "inf" or "+inf" or "infinity") return double.PositiveInfinity;
            if (lower is "-inf" or "-infinity") return double.NegativeInfinity;
            if (lower is "nan" or "+nan" or "-nan") return double.NaN;
        }
        throw new InvalidDataException($"Matrix Market: '{Encoding.ASCII.GetString(tok)}' is not a number.");
    }

    /// <summary>Splits the body into roughly equal chunks that start at line beginnings.</summary>
    private static List<(int Start, int End)> Chunks(ReadOnlySpan<byte> body)
    {
        var chunks = new List<(int, int)>();
        var n = body.Length < ParallelThresholdBytes ? 1 : Math.Min(Environment.ProcessorCount * 4, body.Length / (256 * 1024) + 1);
        var start = 0;
        for (var c = 1; c <= n && start < body.Length; c++)
        {
            var end = c == n ? body.Length : Math.Max(start, (int)((long)body.Length * c / n));
            if (end < body.Length)
            {
                var nl = body[end..].IndexOf((byte)'\n');
                end = nl < 0 ? body.Length : end + nl + 1;
            }
            if (end > start) chunks.Add((start, end));
            start = end;
        }
        return chunks;
    }

    private static MatrixMarketData ReadCoordinate(MatrixMarketInfo info, ReadOnlyMemory<byte> body)
    {
        var complex = info.Field == "complex";
        var pattern = info.Field == "pattern";
        var chunks = Chunks(body.Span);
        // Every entry takes one line, so a chunk's line count bounds its entries. Parsing goes
        // straight into the final arrays at each chunk's offset: per-chunk buffers that are then
        // merged cost more than the parse itself (large-object-heap churn and an extra copy).
        var caps = new long[chunks.Count];
        var offs = new long[chunks.Count + 1];
        for (var c = 0; c < chunks.Count; c++)
        {
            var span = body.Span[chunks[c].Start..chunks[c].End];
            caps[c] = span.Count((byte)'\n') + (span.Length > 0 && span[^1] != (byte)'\n' ? 1 : 0);
            offs[c + 1] = offs[c] + caps[c];
        }
        var capTotal = offs[^1];
        if (capTotal < info.Entries) throw new InvalidDataException($"Matrix Market: header promises {info.Entries} entries, the body has only {capTotal} lines.");
        var row = new int[checked((int)capTotal)];
        var col = new int[row.Length];
        var re = new double[row.Length];
        var im = complex ? new double[row.Length] : null;
        var counts = new int[chunks.Count];
        void Parse(int c)
        {
            // Locals rather than captured variables: closure fields are re-read on every access.
            var span = body.Span[chunks[c].Start..chunks[c].End];
            var rows = info.Rows;
            var columns = info.Columns;
            var o = (int)offs[c];
            var limit = (int)caps[c];
            var rowOut = row.AsSpan(o, limit);
            var colOut = col.AsSpan(o, limit);
            var reOut = re.AsSpan(o, limit);
            var imOut = complex ? im.AsSpan(o, limit) : default;
            var pos = 0;
            var k = 0;
            while (pos < span.Length)
            {
                var line = NextLine(span, ref pos);
                var p = 0;
                while (p < line.Length && IsSpace(line[p])) p++;
                if (p == line.Length || line[p] == (byte)'%') continue;
                var i = ParseLong(line, ref p);
                var j = ParseLong(line, ref p);
                if (i < 1 || i > rows || j < 1 || j > columns)
                    throw new InvalidDataException($"Matrix Market: entry ({i}, {j}) outside the {rows} x {columns} matrix.");
                rowOut[k] = (int)i - 1;
                colOut[k] = (int)j - 1;
                if (pattern) reOut[k] = 1.0;
                else
                {
                    reOut[k] = ParseDouble(line, ref p);
                    if (complex) imOut[k] = ParseDouble(line, ref p);
                }
                k++;
            }
            counts[c] = k;
        }
        if (chunks.Count == 1) Parse(0);
        else Parallel.For(0, chunks.Count, Parse);

        long total = 0;
        foreach (var k in counts) total += k;
        if (total != info.Entries) throw new InvalidDataException($"Matrix Market: header promises {info.Entries} entries, the body holds {total}.");
        if (total != capTotal)
        {
            // Blank or comment lines in the body left gaps: close them up, then trim.
            var w = 0;
            for (var c = 0; c < chunks.Count; c++)
            {
                if (w != offs[c])
                {
                    Array.Copy(row, offs[c], row, w, counts[c]);
                    Array.Copy(col, offs[c], col, w, counts[c]);
                    Array.Copy(re, offs[c], re, w, counts[c]);
                    if (complex) Array.Copy(im!, offs[c], im!, w, counts[c]);
                }
                w += counts[c];
            }
            Array.Resize(ref row, w);
            Array.Resize(ref col, w);
            Array.Resize(ref re, w);
            if (complex) Array.Resize(ref im, w);
        }

        // Symmetric storage keeps one triangle; mirror every off-diagonal entry.
        var sym = info.Symmetry;
        if (sym != "general")
        {
            long extra = 0;
            for (var k = 0; k < total; k++) if (row[k] != col[k]) extra++;
            var n = checked((int)(total + extra));
            Array.Resize(ref row, n);
            Array.Resize(ref col, n);
            Array.Resize(ref re, n);
            if (complex) Array.Resize(ref im, n);
        }
        if (sym != "general")
        {
            var w = (int)total;
            for (var k = 0; k < total; k++)
            {
                if (row[k] == col[k]) continue;
                row[w] = col[k];
                col[w] = row[k];
                switch (sym)
                {
                    case "symmetric": re[w] = re[k]; if (complex) im![w] = im![k]; break;
                    case "skew-symmetric": re[w] = -re[k]; if (complex) im![w] = -im![k]; break;
                    default: re[w] = re[k]; im![w] = -im![k]; break; // hermitian: conjugate
                }
                w++;
            }
        }
        var coo = new CooMatrix(info.Rows, info.Columns, row, col, re, canonical: false);
        return new MatrixMarketData(info, null, null, coo, im);
    }

    private static MatrixMarketData ReadArray(MatrixMarketInfo info, ReadOnlyMemory<byte> body)
    {
        var complex = info.Field == "complex";
        int rows = info.Rows, cols = info.Columns;
        var sym = info.Symmetry;
        // Values come column by column; a symmetric file lists only the lower triangle (the
        // skew-symmetric one without the zero diagonal).
        long expected = sym switch
        {
            "general" => (long)rows * cols,
            "skew-symmetric" => (long)rows * (rows - 1) / 2,
            _ => (long)rows * (rows + 1) / 2,
        };
        var stride = complex ? 2 : 1;
        var vals = new double[checked(expected * stride)];
        var chunks = Chunks(body.Span);
        var counts = new int[chunks.Count];
        var local = new double[chunks.Count][];
        void Parse(int c)
        {
            var span = body.Span[chunks[c].Start..chunks[c].End];
            var cap = (span.Count((byte)'\n') + 1) * stride;
            var buf = new double[cap];
            var k = 0;
            var pos = 0;
            while (pos < span.Length)
            {
                var line = NextLine(span, ref pos);
                var p = 0;
                while (p < line.Length && IsSpace(line[p])) p++;
                if (p == line.Length || line[p] == (byte)'%') continue;
                buf[k++] = ParseDouble(line, ref p);
                if (complex) buf[k++] = ParseDouble(line, ref p);
            }
            counts[c] = k;
            local[c] = buf;
        }
        if (chunks.Count == 1) Parse(0);
        else Parallel.For(0, chunks.Count, Parse);
        var got = counts.Sum(x => (long)x);
        if (got != vals.Length) throw new InvalidDataException($"Matrix Market: array body holds {got / stride} values, {expected} expected.");
        var o = 0;
        for (var c = 0; c < chunks.Count; c++) { Array.Copy(local[c], 0, vals, o, counts[c]); o += counts[c]; }

        var re = new double[(long)rows * cols];
        var im = complex ? new double[re.Length] : null;
        var q = 0;
        for (var j = 0; j < cols; j++)
        {
            var i0 = sym switch { "general" => 0, "skew-symmetric" => j + 1, _ => j };
            for (var i = i0; i < rows; i++, q++)
            {
                var a = vals[q * stride];
                var b = complex ? vals[q * stride + 1] : 0;
                re[(long)i * cols + j] = a;
                if (complex) im![(long)i * cols + j] = b;
                if (sym == "general" || i == j) continue;
                var t = (long)j * cols + i;
                switch (sym)
                {
                    case "symmetric": re[t] = a; if (complex) im![t] = b; break;
                    case "skew-symmetric": re[t] = -a; if (complex) im![t] = -b; break;
                    default: re[t] = a; im![t] = -b; break;
                }
            }
        }
        if (!complex) return new MatrixMarketData(info, new NdArray(re, rows, cols), null, null, null);
        var z = new Complex[re.Length];
        for (var k = 0; k < z.Length; k++) z[k] = new Complex(re[k], im![k]);
        return new MatrixMarketData(info, null, new ComplexNdArray(z, rows, cols), null, null);
    }

    // ------------------------------------------------------------------ writing

    /// <summary>Writes a dense real matrix in <c>array</c> format (<c>scipy.io.mmwrite</c>).</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="a">Rank-2 array (rank 1 is written as a column).</param>
    /// <param name="comment">Comment lines placed after the banner (each line gets a <c>%</c>).</param>
    /// <param name="field"><c>real</c> (default) or <c>integer</c>; <c>pattern</c> is invalid for arrays.</param>
    /// <param name="precision">Significant digits; null writes the shortest exact (round-trip) form.</param>
    /// <param name="symmetry">
    /// <c>"AUTO"</c> (default, SciPy's): detect symmetry when the larger dimension is below 100,
    /// otherwise write <c>general</c>; null always detects; or name it explicitly
    /// (<c>general</c>, <c>symmetric</c>, <c>skew-symmetric</c>).
    /// </param>
    public static void MmWrite(string path, NdArray a, string? comment = null, string? field = null, int? precision = null, string? symmetry = "AUTO")
    {
        using var f = File.Create(path);
        MmWrite(f, a, comment, field, precision, symmetry);
    }

    /// <summary>Writes a dense real matrix to a stream in <c>array</c> format.</summary>
    /// <param name="stream">Destination.</param>
    /// <param name="a">Rank-2 array.</param>
    /// <param name="comment">Comment text.</param>
    /// <param name="field"><c>real</c> or <c>integer</c>.</param>
    /// <param name="precision">Significant digits, or null for round-trip.</param>
    /// <param name="symmetry">See <see cref="MmWrite(string, NdArray, string?, string?, int?, string?)"/>.</param>
    public static void MmWrite(Stream stream, NdArray a, string? comment = null, string? field = null, int? precision = null, string? symmetry = "AUTO")
    {
        if (a.Rank == 1) a = a.Reshape(a.Size, 1);
        if (a.Rank != 2) throw new ArgumentException("Matrix Market holds 2-D matrices only.", nameof(a));
        field ??= "real";
        if (field is not ("real" or "integer" or "unsigned-integer")) throw new ArgumentException($"Field '{field}' is not valid for a real dense matrix.", nameof(field));
        var v = a.AsContiguous().AsSpan().ToArray();
        int rows = a.Rows, cols = a.Columns;
        CheckIntegers(v, field);
        symmetry = ResolveSymmetry(symmetry, rows, cols, () => DetectDense(v, null, rows, cols));
        if (symmetry == "hermitian") symmetry = "symmetric";
        WriteArray(stream, rows, cols, v, null, comment, field, precision, symmetry);
    }

    /// <summary>Writes a dense complex matrix in <c>array</c> format.</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="a">Rank-2 complex array.</param>
    /// <param name="comment">Comment text.</param>
    /// <param name="precision">Significant digits, or null for round-trip.</param>
    /// <param name="symmetry">As for the real overload; <c>hermitian</c> is also possible.</param>
    public static void MmWrite(string path, ComplexNdArray a, string? comment = null, int? precision = null, string? symmetry = "AUTO")
    {
        if (a.Rank != 2) throw new ArgumentException("Matrix Market holds 2-D matrices only.", nameof(a));
        int rows = a.Shape[0], cols = a.Shape[1];
        var re = a.Real().ToArray();
        var im = a.Imaginary().ToArray();
        symmetry = ResolveSymmetry(symmetry, rows, cols, () => DetectDense(re, im, rows, cols));
        using var f = File.Create(path);
        WriteArray(f, rows, cols, re, im, comment, "complex", precision, symmetry);
    }

    /// <summary>Writes a sparse real matrix in <c>coordinate</c> format.</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="a">Any sparse matrix; entries are written in its COO order (duplicates kept).</param>
    /// <param name="comment">Comment text.</param>
    /// <param name="field"><c>real</c> (default), <c>integer</c> or <c>pattern</c>.</param>
    /// <param name="precision">Significant digits, or null for round-trip.</param>
    /// <param name="symmetry">As for the dense overload; symmetric output keeps the lower triangle only.</param>
    public static void MmWrite(string path, SparseMatrixBase a, string? comment = null, string? field = null, int? precision = null, string? symmetry = "AUTO")
    {
        using var f = File.Create(path);
        MmWrite(f, a, comment, field, precision, symmetry);
    }

    /// <summary>Writes a sparse real matrix to a stream in <c>coordinate</c> format.</summary>
    /// <param name="stream">Destination.</param>
    /// <param name="a">Sparse matrix.</param>
    /// <param name="comment">Comment text.</param>
    /// <param name="field"><c>real</c>, <c>integer</c> or <c>pattern</c>.</param>
    /// <param name="precision">Significant digits, or null for round-trip.</param>
    /// <param name="symmetry">See the path overload.</param>
    public static void MmWrite(Stream stream, SparseMatrixBase a, string? comment = null, string? field = null, int? precision = null, string? symmetry = "AUTO")
    {
        field ??= "real";
        if (field is not ("real" or "integer" or "unsigned-integer" or "pattern")) throw new ArgumentException($"Field '{field}' is not valid for a real sparse matrix.", nameof(field));
        var coo = a.ToCoo();
        var data = coo.Data.ToArray();
        if (field != "pattern") CheckIntegers(data, field);
        symmetry = ResolveSymmetry(symmetry, a.Rows, a.Columns, () => DetectSparse(a.ToCsr(), null));
        if (symmetry == "hermitian") symmetry = "symmetric";
        WriteCoordinate(stream, a.Rows, a.Columns, coo.Row.ToArray(), coo.Col.ToArray(), data, null, comment, field, precision, symmetry);
    }

    /// <summary>Writes a complex sparse matrix: real part <paramref name="real"/> and imaginary parts aligned with its COO entries.</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="real">Real part (its COO order defines the entry order).</param>
    /// <param name="imaginary">Imaginary part of each stored entry of <c>real.ToCoo()</c>.</param>
    /// <param name="comment">Comment text.</param>
    /// <param name="precision">Significant digits, or null for round-trip.</param>
    /// <param name="symmetry"><c>general</c> (default), <c>symmetric</c>, <c>skew-symmetric</c> or <c>hermitian</c>; not detected.</param>
    public static void MmWrite(string path, SparseMatrixBase real, double[] imaginary, string? comment = null, int? precision = null, string symmetry = "general")
    {
        var coo = real.ToCoo();
        if (imaginary.Length != coo.Nnz) throw new ArgumentException("One imaginary value per stored entry is required.", nameof(imaginary));
        symmetry = ValidateSymmetry(symmetry, real.Rows, real.Columns);
        using var f = File.Create(path);
        WriteCoordinate(f, real.Rows, real.Columns, coo.Row.ToArray(), coo.Col.ToArray(), coo.Data.ToArray(), imaginary, comment, "complex", precision, symmetry);
    }

    private static void CheckIntegers(double[] v, string field)
    {
        if (field is not ("integer" or "unsigned-integer")) return;
        foreach (var x in v)
            if (x != Math.Floor(x) || double.IsInfinity(x) || (field == "unsigned-integer" && x < 0))
                throw new ArgumentException($"Value {x} cannot be written with the '{field}' field.");
    }

    private static string ValidateSymmetry(string symmetry, int rows, int cols)
    {
        var s = symmetry.ToLowerInvariant();
        if (s is not ("general" or "symmetric" or "skew-symmetric" or "hermitian")) throw new ArgumentException($"Unknown symmetry '{symmetry}'.");
        if (s != "general" && rows != cols) throw new ArgumentException($"A {s} matrix must be square.");
        return s;
    }

    private static string ResolveSymmetry(string? symmetry, int rows, int cols, Func<string> detect)
    {
        if (symmetry is not null && symmetry.Equals("AUTO", StringComparison.OrdinalIgnoreCase))
            // SciPy's AUTO: searching for symmetry is O(nnz) with random access, so it is only done for small matrices.
            symmetry = Math.Max(rows, cols) < 100 ? null : "general";
        if (symmetry is null) return rows != cols ? "general" : detect();
        return ValidateSymmetry(symmetry, rows, cols);
    }

    /// <summary>SciPy's <c>_get_symmetry</c> for dense data: symmetric beats skew beats hermitian.</summary>
    private static string DetectDense(double[] re, double[]? im, int n, int _)
    {
        bool sym = true, skew = true, herm = im is not null;
        for (var j = 0; j < n && (sym || skew || herm); j++)
            for (var i = j; i < n; i++)
            {
                double ar = re[i * n + j], br = re[j * n + i];
                double ai = im?[i * n + j] ?? 0, bi = im?[j * n + i] ?? 0;
                if (i == j)
                {
                    if (ar != 0 || ai != 0) skew = false;
                    if (ai != 0) herm = false;
                    continue;
                }
                if (ar != br || ai != bi) sym = false;
                if (ar != -br || ai != -bi) skew = false;
                if (ar != br || ai != -bi) herm = false;
            }
        return sym ? "symmetric" : skew ? "skew-symmetric" : herm ? "hermitian" : "general";
    }

    private static string DetectSparse(CsrMatrix a, double[]? _)
    {
        var c = a.SumDuplicates().SortIndices();
        var ptr = c.Indptr;
        var idx = c.Indices;
        var val = c.Data;
        long lower = 0, upper = 0;
        for (var i = 0; i < c.Rows; i++)
            for (var k = ptr[i]; k < ptr[i + 1]; k++)
            {
                if (val[k] == 0) continue;
                if (idx[k] < i) lower++;
                else if (idx[k] > i) upper++;
            }
        if (lower != upper) return "general";
        bool sym = true, skew = true;
        for (var i = 0; i < c.Rows && (sym || skew); i++)
            for (var k = ptr[i]; k < ptr[i + 1]; k++)
            {
                var j = idx[k];
                var aij = val[k];
                if (j == i) { if (aij != 0) skew = false; continue; }
                if (j > i) continue;
                var aji = Lookup(ptr, idx, val, j, i);
                if (aij != aji) sym = false;
                if (aij != -aji) skew = false;
            }
        return sym ? "symmetric" : skew ? "skew-symmetric" : "general";

        static double Lookup(ReadOnlySpan<int> ptr, ReadOnlySpan<int> idx, ReadOnlySpan<double> val, int r, int col)
        {
            var k = idx[ptr[r]..ptr[r + 1]].BinarySearch(col);
            return k >= 0 ? val[ptr[r] + k] : 0.0;
        }
    }

    private static byte[] HeaderBytes(string format, string field, string symmetry, string? comment, string sizeLine, int? precision)
    {
        if (precision is < 1 or > 17) throw new ArgumentOutOfRangeException(nameof(precision), "Precision must be between 1 and 17 significant digits (17 already round-trips any double).");
        var sb = new StringBuilder();
        sb.Append("%%MatrixMarket matrix ").Append(format).Append(' ').Append(field).Append(' ').Append(symmetry).Append('\n');
        if (!string.IsNullOrEmpty(comment))
            foreach (var line in comment.Replace("\r\n", "\n").Split('\n'))
                sb.Append('%').Append(line).Append('\n');
        sb.Append(sizeLine).Append('\n');
        return Encoding.UTF8.GetBytes(sb.ToString());
    }

    private static int FormatNumber(double v, Span<byte> dst, string field, int? precision)
    {
        if (field is "integer" or "unsigned-integer")
        {
            if (!((long)v).TryFormat(dst, out var w, default, CultureInfo.InvariantCulture)) throw new InvalidOperationException();
            return w;
        }
        int written;
        var ok = precision is { } p
            ? v.TryFormat(dst, out written, "G" + p.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)
            : v.TryFormat(dst, out written, "R", CultureInfo.InvariantCulture);
        if (!ok) throw new InvalidOperationException("Number formatting overflowed its buffer.");
        return written;
    }

    /// <summary>Formats <paramref name="count"/> lines in parallel chunks and writes them in order.</summary>
    private static void WriteLines(Stream s, int count, int maxLineBytes, Func<int, Span<byte>, int> formatLine)
    {
        const int linesPerChunk = 1 << 15;
        var nChunks = (count + linesPerChunk - 1) / linesPerChunk;
        if (nChunks <= 1)
        {
            var buf = new byte[Math.Max(1, count) * maxLineBytes];
            var w = 0;
            for (var k = 0; k < count; k++) w += formatLine(k, buf.AsSpan(w));
            s.Write(buf, 0, w);
            return;
        }
        // Format a window of chunks in parallel, flush in order, keep memory bounded.
        var window = Environment.ProcessorCount * 2;
        for (var c0 = 0; c0 < nChunks; c0 += window)
        {
            var c1 = Math.Min(nChunks, c0 + window);
            var bufs = new byte[c1 - c0][];
            var lens = new int[c1 - c0];
            Parallel.For(c0, c1, c =>
            {
                var a = c * linesPerChunk;
                var b = Math.Min(count, a + linesPerChunk);
                var buf = ArrayPool<byte>.Shared.Rent((b - a) * maxLineBytes);
                var w = 0;
                for (var k = a; k < b; k++) w += formatLine(k, buf.AsSpan(w));
                bufs[c - c0] = buf;
                lens[c - c0] = w;
            });
            for (var c = 0; c < bufs.Length; c++)
            {
                s.Write(bufs[c], 0, lens[c]);
                ArrayPool<byte>.Shared.Return(bufs[c]);
            }
        }
    }

    private static void WriteCoordinate(Stream s, int rows, int cols, int[] row, int[] col, double[] re, double[]? im, string? comment, string field, int? precision, string symmetry)
    {
        // Symmetric storage keeps the lower triangle (strictly lower for skew-symmetric).
        int[] keep;
        if (symmetry == "general") keep = [];
        else
        {
            var list = new List<int>(row.Length);
            for (var k = 0; k < row.Length; k++)
                if (symmetry == "skew-symmetric" ? row[k] > col[k] : row[k] >= col[k]) list.Add(k);
            keep = list.ToArray();
        }
        var count = symmetry == "general" ? row.Length : keep.Length;
        s.Write(HeaderBytes("coordinate", field, symmetry, comment, $"{rows} {cols} {count}", precision));
        var general = symmetry == "general";
        WriteLines(s, count, 96, (q, dst) =>
        {
            var k = general ? q : keep[q];
            var w = 0;
            (row[k] + 1).TryFormat(dst, out var n, default, CultureInfo.InvariantCulture); w += n;
            dst[w++] = (byte)' ';
            (col[k] + 1).TryFormat(dst[w..], out n, default, CultureInfo.InvariantCulture); w += n;
            if (field != "pattern")
            {
                dst[w++] = (byte)' ';
                w += FormatNumber(re[k], dst[w..], field, precision);
                if (im is not null)
                {
                    dst[w++] = (byte)' ';
                    w += FormatNumber(im[k], dst[w..], field, precision);
                }
            }
            dst[w++] = (byte)'\n';
            return w;
        });
    }

    private static void WriteArray(Stream s, int rows, int cols, double[] re, double[]? im, string? comment, string field, int? precision, string symmetry)
    {
        s.Write(HeaderBytes("array", field, symmetry, comment, $"{rows} {cols}", precision));
        // Column-major order; symmetric storage lists each column from the diagonal down.
        var order = new List<int>((int)Math.Min((long)rows * cols, int.MaxValue));
        for (var j = 0; j < cols; j++)
        {
            var i0 = symmetry switch { "general" => 0, "skew-symmetric" => j + 1, _ => j };
            for (var i = i0; i < rows; i++) order.Add(i * cols + j);
        }
        var ord = order.ToArray();
        WriteLines(s, ord.Length, 72, (q, dst) =>
        {
            var k = ord[q];
            var w = FormatNumber(re[k], dst, field, precision);
            if (im is not null)
            {
                dst[w++] = (byte)' ';
                w += FormatNumber(im[k], dst[w..], field, precision);
            }
            dst[w++] = (byte)'\n';
            return w;
        });
    }
}
