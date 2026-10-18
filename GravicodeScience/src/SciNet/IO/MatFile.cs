using System.Buffers.Binary;

namespace Gravicode.Science.SciNet.IO;

/// <summary>One entry of <see cref="MatFile.Whosmat(string)"/>: name, MATLAB dimensions and class (SciPy's spelling: <c>double</c>, <c>sparse</c>, <c>logical</c>, <c>char</c>, <c>struct</c>, ...).</summary>
/// <param name="Name">Variable name.</param>
/// <param name="Shape">MATLAB dimensions.</param>
/// <param name="ClassName">Class as SciPy's <c>whosmat</c> reports it.</param>
public sealed record MatVariableInfo(string Name, int[] Shape, string ClassName);

/// <summary>
/// MATLAB MAT files — <c>scipy.io.loadmat</c>, <c>savemat</c>, <c>whosmat</c>. Reads level 4,
/// level 5 (MATLAB v4, v6, v7: compressed or not, either byte order) and v7.3 (HDF5) with every
/// array class; writes level 5 (optionally compressed), level 4 and v7.3.
/// </summary>
/// <remarks>
/// <para>MATLAB v7.3 files are HDF5 containers behind a MAT header; they are read with SciNet's
/// own managed <see cref="Hdf5File"/> reader into the same <see cref="MatValue"/> model, so a
/// variable loads identically whichever format it was saved in (SciPy itself cannot read v7.3).
/// <c>format: "7.3"</c> writes them with SciNet's <see cref="Hdf5Writer"/> in MATLAB's layout (see
/// <see cref="Save(string, IReadOnlyDictionary{string, object?}, bool, string, string)"/>); SciPy
/// cannot write v7.3 either.</para>
/// <para>Arrays come back row-major with MATLAB's shape (<c>A(2,3)</c> is <c>array[1, 2]</c>).
/// The conversion from MATLAB's column-major storage is a cache-blocked transpose — for a large
/// 2-D double it is fused with the read, so no intermediate copy is made.</para>
/// </remarks>
public static class MatFile
{
    /// <summary>Loads a MAT file (<c>scipy.io.loadmat</c>).</summary>
    /// <param name="path">File path.</param>
    /// <param name="variableNames">Only these variables (others are skipped without being inflated); null for all.</param>
    /// <param name="squeezeMe">Remove length-1 dimensions (SciPy's <c>squeeze_me</c>); a 1 x 1 value gets no dimensions.</param>
    /// <param name="simplifyCells">
    /// SciPy's <c>simplify_cells</c>: implies <paramref name="squeezeMe"/>, and every structure
    /// array with more than one element becomes a cell array of scalar structures, so that every
    /// <see cref="MatStruct"/> in the result can be used as a dictionary.
    /// </param>
    /// <returns>Variables by name, in file order. A function handle or classdef object becomes a <see cref="MatUnsupported"/> rather than failing the load.</returns>
    public static IReadOnlyDictionary<string, MatValue> Load(string path, IEnumerable<string>? variableNames = null, bool squeezeMe = false, bool simplifyCells = false)
        => Load(File.ReadAllBytes(path), variableNames, squeezeMe, simplifyCells);

    /// <summary>Loads a MAT file from a stream (read to its end).</summary>
    /// <param name="stream">Source stream.</param>
    /// <param name="variableNames">Only these variables; null for all.</param>
    /// <param name="squeezeMe">Remove length-1 dimensions.</param>
    /// <param name="simplifyCells">See <see cref="Load(string, IEnumerable{string}?, bool, bool)"/>.</param>
    public static IReadOnlyDictionary<string, MatValue> Load(Stream stream, IEnumerable<string>? variableNames = null, bool squeezeMe = false, bool simplifyCells = false)
        => Load(ReadAll(stream), variableNames, squeezeMe, simplifyCells);

    private static IReadOnlyDictionary<string, MatValue> Load(byte[] file, IEnumerable<string>? variableNames, bool squeezeMe, bool simplifyCells)
    {
        var wanted = variableNames is null ? null : new HashSet<string>(variableNames, StringComparer.Ordinal);
        var (major, _) = Version(file);
        var vars = major switch
        {
            0 => MatV4.Read(file, wanted),
            2 => MatV73Reader.ReadAll(file, wanted),
            _ => MatV5Reader.Open(file).ReadAll(file, wanted),
        };
        if (!squeezeMe && !simplifyCells) return vars;
        var result = new OrderedDictionary<string, MatValue>(StringComparer.Ordinal);
        foreach (var (k, v) in vars)
        {
            var s = v.Squeeze();
            result[k] = simplifyCells ? Simplify(s) : s;
        }
        return result;
    }

    private static MatValue Simplify(MatValue v)
    {
        switch (v)
        {
            case MatStruct st:
            {
                var elements = new MatValue[st.Count];
                for (var e = 0; e < st.Count; e++)
                {
                    var raw = st.RawElement(e);
                    var fields = new List<KeyValuePair<string, object?>>(raw.Length);
                    for (var f = 0; f < raw.Length; f++) fields.Add(new(st.FieldNames[f], Simplify(raw[f])));
                    elements[e] = new MatStruct(fields, st.ObjectClassName);
                }
                return st.Count == 1 ? elements[0] : new MatCell(st.Dimensions.ToArray(), elements);
            }
            case MatCell c:
                return new MatCell(c.Dimensions.ToArray(), c.Select(Simplify).ToArray());
            default:
                return v;
        }
    }

    /// <summary>Lists the variables without loading their data (<c>scipy.io.whosmat</c>).</summary>
    public static IReadOnlyList<MatVariableInfo> Whosmat(string path) => Whosmat(File.ReadAllBytes(path));

    /// <summary>Lists the variables of a MAT stream without loading their data.</summary>
    public static IReadOnlyList<MatVariableInfo> Whosmat(Stream stream) => Whosmat(ReadAll(stream));

    private static IReadOnlyList<MatVariableInfo> Whosmat(byte[] file)
    {
        var (major, _) = Version(file);
        return major switch
        {
            0 => MatV4.List(file),
            2 => MatV73Reader.List(file),
            _ => MatV5Reader.Open(file).List(file),
        };
    }

    /// <summary>
    /// The file's format version as SciPy's <c>matfile_version</c> reports it: (0, 0) for level 4,
    /// (1, 0) for level 5, (2, 0) for v7.3/HDF5.
    /// </summary>
    public static (int Major, int Minor) GetVersion(string path)
    {
        using var f = File.OpenRead(path);
        var head = new byte[128];
        var n = f.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        return Version(head.AsSpan(0, n), checkV73: false);
    }

    private static (int Major, int Minor) Version(byte[] file)
    {
        var v = Version(file, checkV73: false);
        if (v.Major == 2 && !Hdf5File.IsHdf5(new MemoryStream(file, writable: false)))
            throw new InvalidDataException("Malformed MAT file: the header says v7.3 (HDF5) but there is no HDF5 superblock.");
        return v;
    }

    private static (int Major, int Minor) Version(ReadOnlySpan<byte> file, bool checkV73)
    {
        if (file.Length < 4) throw new InvalidDataException("Not a MAT file: fewer than 4 bytes.");
        // A level 4 file starts with the 4-byte MOPT integer (< 5000, so it always contains a
        // zero byte); a level 5 header starts with descriptive text, which never does.
        if (file[..4].IndexOf((byte)0) >= 0) return (0, 0);
        if (file.Length < 128) throw new InvalidDataException("Not a MAT file: shorter than the 128-byte level 5 header.");
        var bigEndian = file[126] == (byte)'M' && file[127] == (byte)'I';
        int version = bigEndian ? BinaryPrimitives.ReadUInt16BigEndian(file[124..]) : BinaryPrimitives.ReadUInt16LittleEndian(file[124..]);
        var major = version >> 8;
        if (major == 2)
        {
            return (2, 0);
        }
        if (major != 1) throw new InvalidDataException($"Not a MAT file: unknown version 0x{version:X4}.");
        return (1, version & 0xFF);
    }

    /// <summary>Saves variables to a MAT file (<c>scipy.io.savemat</c>).</summary>
    /// <param name="path">Destination path (overwritten).</param>
    /// <param name="variables">Name → value; values are converted with <see cref="MatValue.FromObject"/>.</param>
    /// <param name="doCompression">Compress each variable with zlib, as MATLAB v7 does by default (level 5; for v7.3, deflate on datasets of 4 KiB or more).</param>
    /// <param name="format"><c>"5"</c> (MATLAB 5 and later; the default), <c>"4"</c>, or <c>"7.3"</c> (HDF5-based, MATLAB's <c>-v7.3</c>; needed for variables over 2 GB).</param>
    /// <param name="onedAs"><c>"row"</c> (default) writes 1-D data as 1 x N, <c>"column"</c> as N x 1.</param>
    /// <remarks>
    /// A v7.3 file is written in the layout MATLAB uses: a 512-byte user block with the MAT header,
    /// a <c>MATLAB_class</c> attribute on every variable, column-major data, cells and struct-array
    /// elements as object references into <c>/#refs#</c>. Every class the loader returns can be
    /// written, except function handles and classdef objects (<see cref="MatUnsupported"/>).
    /// Variables load back in the order they were saved.
    /// </remarks>
    public static void Save(string path, IReadOnlyDictionary<string, object?> variables, bool doCompression = false, string format = "5", string onedAs = "row")
    {
        using var f = File.Create(path);
        Save(f, variables, doCompression, format, onedAs);
    }

    /// <summary>Saves variables to a stream.</summary>
    /// <param name="stream">Destination stream.</param>
    /// <param name="variables">Name → value.</param>
    /// <param name="doCompression">Compress each variable with zlib (level 5) or deflate large datasets (v7.3).</param>
    /// <param name="format"><c>"5"</c>, <c>"4"</c> or <c>"7.3"</c>.</param>
    /// <param name="onedAs"><c>"row"</c> or <c>"column"</c>.</param>
    public static void Save(Stream stream, IReadOnlyDictionary<string, object?> variables, bool doCompression = false, string format = "5", string onedAs = "row")
    {
        ArgumentNullException.ThrowIfNull(variables);
        var column = onedAs switch
        {
            "row" => false,
            "column" => true,
            _ => throw new ArgumentException("onedAs must be \"row\" or \"column\".", nameof(onedAs)),
        };
        var converted = variables.Select(kv => new KeyValuePair<string, MatValue>(kv.Key, MatValue.FromObject(kv.Value, column))).ToList();
        WriteConverted(stream, converted, doCompression, format);
    }

    /// <summary>Saves already-built MAT values (round-trips what <see cref="Load(string, IEnumerable{string}?, bool, bool)"/> returned).</summary>
    /// <param name="path">Destination path.</param>
    /// <param name="variables">Name → value.</param>
    /// <param name="doCompression">Compress each variable (level 5) or deflate large datasets (v7.3).</param>
    /// <param name="format"><c>"5"</c>, <c>"4"</c> or <c>"7.3"</c>.</param>
    public static void Save(string path, IReadOnlyDictionary<string, MatValue> variables, bool doCompression = false, string format = "5")
    {
        using var f = File.Create(path);
        WriteConverted(f, variables, doCompression, format);
    }

    private static void WriteConverted(Stream stream, IEnumerable<KeyValuePair<string, MatValue>> vars, bool doCompression, string format)
    {
        switch (format)
        {
            case "5":
                MatV5Writer.Write(stream, vars, doCompression);
                break;
            case "4":
                if (doCompression) throw new ArgumentException("The v4 format has no compression.", nameof(doCompression));
                MatV4.Write(stream, vars);
                break;
            case "7.3":
                MatV73Writer.Write(stream, vars, doCompression);
                break;
            default:
                throw new ArgumentException("format must be \"5\", \"4\" or \"7.3\".", nameof(format));
        }
    }

    internal static void ValidateName(string name)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Variable names must not be empty.");
        if (name.Length > 63) throw new ArgumentException($"Variable name '{name}' exceeds MATLAB's 63-character limit.");
    }

    internal static byte[] ReadAll(Stream stream)
    {
        if (stream is MemoryStream ms && ms.TryGetBuffer(out var seg) && seg.Offset == 0 && ms.Position == 0 && seg.Count == seg.Array!.Length)
            return seg.Array;
        using var copy = new MemoryStream();
        stream.CopyTo(copy);
        return copy.ToArray();
    }
}
