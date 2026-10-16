using System.Buffers.Binary;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.IO;

/// <summary>An IDL array of strings, pointers or other non-numeric values, with its shape.</summary>
/// <param name="Shape">Shape in SciPy's order (IDL's dimensions reversed).</param>
/// <param name="Items">Elements in row-major order of <paramref name="Shape"/>.</param>
public sealed record IdlObjectArray(int[] Shape, object?[] Items);

/// <summary>
/// An IDL structure (or array of structures) — SciPy's <c>recarray</c>. Field names are kept as
/// IDL stores them (upper case); lookups ignore case.
/// </summary>
public sealed class IdlStructure
{
    private readonly object?[][] _rows;

    internal IdlStructure(string name, string[] fields, object?[][] rows, int[] shape)
    {
        Name = name;
        FieldNames = fields;
        _rows = rows;
        Shape = shape;
    }

    /// <summary>The IDL structure name (empty for anonymous structures).</summary>
    public string Name { get; }

    /// <summary>Field names in order.</summary>
    public IReadOnlyList<string> FieldNames { get; }

    /// <summary>Number of structure elements.</summary>
    public int Count => _rows.Length;

    /// <summary>Shape in SciPy's order (IDL's dimensions reversed); [n] for a 1-D array of structures.</summary>
    public int[] Shape { get; }

    private int Field(string name)
    {
        for (var i = 0; i < FieldNames.Count; i++)
            if (string.Equals(FieldNames[i], name, StringComparison.OrdinalIgnoreCase)) return i;
        throw new KeyNotFoundException($"No field '{name}'. Fields: {string.Join(", ", FieldNames)}.");
    }

    /// <summary>A field of element <paramref name="index"/>.</summary>
    public object? this[int index, string field] => _rows[index][Field(field)];

    /// <summary>A field of a single-element structure.</summary>
    public object? this[string field] => Count == 1
        ? _rows[0][Field(field)]
        : throw new InvalidOperationException($"Structure array has {Count} elements; use [index, field] or Column(field).");

    /// <summary>One field across all elements.</summary>
    public object?[] Column(string field)
    {
        var f = Field(field);
        return _rows.Select(r => r[f]).ToArray();
    }

    internal object?[][] Rows => _rows;
}

/// <summary>
/// IDL SAVE files — <c>scipy.io.readsav</c>: scalars, arrays of up to 8 dimensions, strings,
/// structures (nested, replicated, inherited), heap pointers and compressed files.
/// </summary>
/// <remarks>
/// Values: integer and float scalars keep their .NET type (<see cref="byte"/>, <see cref="short"/>,
/// <see cref="int"/>, <see cref="long"/>, <see cref="ushort"/>, <see cref="uint"/>,
/// <see cref="ulong"/>, <see cref="float"/>, <see cref="double"/>); complex scalars are
/// <see cref="Complex"/>; strings are <see cref="string"/> (Latin-1). Numeric arrays are
/// <see cref="NdArray"/> (or <see cref="ComplexNdArray"/>), other arrays <see cref="IdlObjectArray"/>,
/// structures <see cref="IdlStructure"/>. Pointers are replaced by what they point to (null for
/// a null pointer). As in SciPy, array shapes are IDL's dimensions reversed over the same memory,
/// so <c>arr[k, j, i]</c> is IDL's <c>arr[i, j, k]</c>. Object references (classes) are not resolved.
/// </remarks>
public static class IdlSave
{
    private const int StartMarker = 0, CommonVariable = 1, Variable = 2, SystemVariable = 3, EndMarker = 6, Timestamp = 10,
        Compiled = 12, Identification = 13, Version = 14, HeapHeader = 15, HeapData = 16, Promote64 = 17, Notice = 19, Description = 20;

    private sealed class Pointer(int index)
    {
        public int Index { get; } = index;
    }

    private sealed class ArrayDesc
    {
        public long NBytes;
        public long NElements;
        public int NDims;
        public int[] Dims = [];
    }

    private sealed class TagDesc
    {
        public int TypeCode;
        public bool IsArray;
        public bool IsStruct;
        public string Name = "";
    }

    private sealed class StructDesc
    {
        public string Name = "";
        public TagDesc[] Tags = [];
        public Dictionary<string, ArrayDesc> ArrTable = new();
        public Dictionary<string, StructDesc> StructTable = new();
    }

    private sealed class Cursor(byte[] b)
    {
        public readonly byte[] B = b;
        public int Pos;

        public void Need(int n)
        {
            if (Pos + n > B.Length) throw new InvalidDataException($"IDL SAVE record ends early (needed {n} bytes at offset {Pos} of {B.Length}).");
        }

        public int Long() { Need(4); var v = BinaryPrimitives.ReadInt32BigEndian(B.AsSpan(Pos)); Pos += 4; return v; }
        public uint ULong() { Need(4); var v = BinaryPrimitives.ReadUInt32BigEndian(B.AsSpan(Pos)); Pos += 4; return v; }
        public long Int64() { Need(8); var v = BinaryPrimitives.ReadInt64BigEndian(B.AsSpan(Pos)); Pos += 8; return v; }
        public ulong UInt64() { Need(8); var v = BinaryPrimitives.ReadUInt64BigEndian(B.AsSpan(Pos)); Pos += 8; return v; }
        public float Float() { Need(4); var v = BinaryPrimitives.ReadSingleBigEndian(B.AsSpan(Pos)); Pos += 4; return v; }
        public double Double() { Need(8); var v = BinaryPrimitives.ReadDoubleBigEndian(B.AsSpan(Pos)); Pos += 8; return v; }
        public void Skip(int n) { Need(n); Pos += n; }
        public void Align() { if (Pos % 4 != 0) Pos += 4 - Pos % 4; }

        public ReadOnlySpan<byte> Bytes(long n)
        {
            if (n < 0 || n > int.MaxValue) throw new InvalidDataException($"IDL SAVE length {n} is invalid.");
            Need((int)n);
            var s = B.AsSpan(Pos, (int)n);
            Pos += (int)n;
            return s;
        }

        /// <summary>A string with one length word.</summary>
        public string String()
        {
            var n = Long();
            if (n <= 0) return "";
            var s = Encoding.Latin1.GetString(Bytes(n));
            Align();
            return s;
        }

        /// <summary>A data string: its length appears twice.</summary>
        public string StringData()
        {
            var n = Long();
            if (n <= 0) return "";
            n = Long();
            var s = Encoding.Latin1.GetString(Bytes(n));
            Align();
            return s;
        }
    }

    /// <summary>Reads an IDL SAVE file; variable names are lower-cased, as SciPy does.</summary>
    public static IReadOnlyDictionary<string, object?> ReadSav(string path) => ReadSav(File.ReadAllBytes(path));

    /// <summary>Reads an IDL SAVE stream.</summary>
    public static IReadOnlyDictionary<string, object?> ReadSav(Stream stream) => ReadSav(MatFile.ReadAll(stream));

    private static IReadOnlyDictionary<string, object?> ReadSav(byte[] file)
    {
        if (file.Length < 4 || file[0] != (byte)'S' || file[1] != (byte)'R') throw new InvalidDataException("Not an IDL SAVE file: missing the 'SR' signature.");
        var compressed = (file[2], file[3]) switch
        {
            (0, 4) => false,
            (0, 6) => true,
            _ => throw new InvalidDataException($"Invalid IDL SAVE record format 0x{file[2]:X2}{file[3]:X2}."),
        };
        var structs = new Dictionary<string, StructDesc>(StringComparer.Ordinal);
        var heap = new Dictionary<int, object?>();
        var variables = new List<(string Name, object? Data)>();
        var pos = 4L;
        while (true)
        {
            if (pos + 16 > file.Length) throw new InvalidDataException("IDL SAVE file ends without an END_MARKER record.");
            var rectype = BinaryPrimitives.ReadInt32BigEndian(file.AsSpan((int)pos));
            var next = (long)BinaryPrimitives.ReadUInt32BigEndian(file.AsSpan((int)pos + 4)) + ((long)BinaryPrimitives.ReadUInt32BigEndian(file.AsSpan((int)pos + 8)) << 32);
            var start = pos + 16;
            if (rectype == EndMarker) break;
            if (next <= start || next > file.Length) throw new InvalidDataException($"IDL SAVE record at byte {pos} points to {next}, outside the file.");
            var payload = file.AsSpan((int)start, (int)(next - start)).ToArray();
            if (compressed) payload = Inflate(payload);
            var c = new Cursor(payload);
            switch (rectype)
            {
                case Variable:
                {
                    var name = c.String();
                    variables.Add((name.ToLowerInvariant(), ReadVariableData(c, structs)));
                    break;
                }
                case HeapData:
                {
                    var index = c.Long();
                    c.Skip(4);
                    heap[index] = ReadVariableData(c, structs);
                    break;
                }
                case StartMarker or CommonVariable or SystemVariable or Timestamp or Compiled or Identification or Version
                    or HeapHeader or Promote64 or Notice or Description:
                    break; // metadata: nothing to return
                default:
                    throw new InvalidDataException($"Unknown IDL SAVE record type {rectype}.");
            }
            pos = next;
        }

        var result = new OrderedDictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (name, data) in variables) result[name] = Resolve(data, heap, 0);
        return result;
    }

    private static byte[] Inflate(byte[] payload)
    {
        try
        {
            using var z = new ZLibStream(new MemoryStream(payload), CompressionMode.Decompress);
            using var o = new MemoryStream();
            z.CopyTo(o);
            return o.ToArray();
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException("IDL SAVE compressed record is corrupt.", e);
        }
    }

    private static object? ReadVariableData(Cursor c, Dictionary<string, StructDesc> structs)
    {
        var typecode = c.Long();
        var flags = c.Long();
        if ((flags & 2) != 0) throw new NotSupportedException("IDL system variables are not supported.");
        var isArray = (flags & 4) != 0;
        var isStruct = (flags & 32) != 0;
        ArrayDesc? arr = null;
        StructDesc? sd = null;
        if (isStruct) { arr = ReadArrayDesc(c); sd = ReadStructDesc(c, structs); }
        else if (isArray) arr = ReadArrayDesc(c);
        if (typecode == 0)
        {
            if (c.Pos == c.B.Length) return null; // a NULL value
            throw new InvalidDataException("Unexpected IDL type code 0.");
        }
        if (c.Long() != 7) throw new InvalidDataException("IDL SAVE: VARSTART is not 7.");
        if (isStruct) return ReadStructure(c, arr!, sd!);
        if (isArray) return ReadArray(c, typecode, arr!);
        return ReadScalar(c, typecode);
    }

    private static ArrayDesc ReadArrayDesc(Cursor c)
    {
        var start = c.Long();
        var a = new ArrayDesc();
        if (start == 8)
        {
            c.Skip(4);
            a.NBytes = c.Long();
            a.NElements = c.Long();
            a.NDims = c.Long();
            c.Skip(8);
            var nmax = c.Long();
            if (nmax is < 0 or > 8) throw new InvalidDataException($"IDL SAVE array descriptor has {nmax} dimensions.");
            a.Dims = new int[nmax];
            for (var i = 0; i < nmax; i++) a.Dims[i] = c.Long();
        }
        else if (start == 18)
        {
            // 64-bit array descriptor (IDL 8 large arrays)
            c.Skip(8);
            a.NBytes = (long)c.UInt64();
            a.NElements = (long)c.UInt64();
            a.NDims = c.Long();
            c.Skip(8);
            a.Dims = new int[8];
            for (var i = 0; i < 8; i++)
            {
                if (c.Long() != 0) throw new InvalidDataException("IDL SAVE: expected a zero in a 64-bit array descriptor.");
                a.Dims[i] = c.Long();
            }
        }
        else throw new InvalidDataException($"Unknown IDL SAVE ARRSTART {start}.");
        if (a.NDims is < 0 or > 8) throw new InvalidDataException($"IDL SAVE array has {a.NDims} dimensions.");
        return a;
    }

    private static StructDesc ReadStructDesc(Cursor c, Dictionary<string, StructDesc> structs)
    {
        if (c.Long() != 9) throw new InvalidDataException("IDL SAVE: STRUCTSTART should be 9.");
        var name = c.String();
        var predef = c.Long();
        var ntags = c.Long();
        c.Long(); // nbytes
        if ((predef & 1) != 0)
            return structs.TryGetValue(name, out var known) ? known : throw new InvalidDataException($"IDL SAVE refers to structure '{name}' before defining it.");
        var sd = new StructDesc { Name = name, Tags = new TagDesc[ntags] };
        for (var i = 0; i < ntags; i++)
        {
            var offset = c.Long();
            if (offset == -1) c.UInt64();
            var t = new TagDesc { TypeCode = c.Long() };
            var tf = c.Long();
            t.IsArray = (tf & 4) != 0;
            t.IsStruct = (tf & 32) != 0;
            sd.Tags[i] = t;
        }
        foreach (var t in sd.Tags) t.Name = c.String();
        foreach (var t in sd.Tags) if (t.IsArray) sd.ArrTable[t.Name] = ReadArrayDesc(c);
        foreach (var t in sd.Tags) if (t.IsStruct) sd.StructTable[t.Name] = ReadStructDesc(c, structs);
        if ((predef & 2) != 0 || (predef & 4) != 0)
        {
            c.String(); // class name
            var nsup = c.Long();
            for (var i = 0; i < nsup; i++) c.String();
            for (var i = 0; i < nsup; i++) ReadStructDesc(c, structs);
        }
        structs[name] = sd;
        return sd;
    }

    private static object? ReadScalar(Cursor c, int typecode)
    {
        switch (typecode)
        {
            case 1:
                if (c.Long() != 1) throw new InvalidDataException("IDL SAVE: malformed byte scalar.");
                return c.Bytes(4)[0];
            case 2: return (short)c.Long();
            case 3: return c.Long();
            case 4: return c.Float();
            case 5: return c.Double();
            case 6: { var re = c.Float(); var im = c.Float(); return new Complex(re, im); }
            case 7: return c.StringData();
            case 9: { var re = c.Double(); var im = c.Double(); return new Complex(re, im); }
            case 10: return new Pointer(c.Long());
            case 11: return new Pointer(c.Long()); // object reference: resolved like a pointer
            case 12: return (ushort)c.ULong();
            case 13: return c.ULong();
            case 14: return c.Int64();
            case 15: return c.UInt64();
            default: throw new InvalidDataException($"Unknown IDL type code {typecode}.");
        }
    }

    /// <summary>SciPy's shape convention: IDL's leading dimensions, reversed.</summary>
    private static int[] Shape(ArrayDesc a)
    {
        if (a.NDims <= 1) return [checked((int)a.NElements)];
        var dims = a.Dims.Take(a.NDims).ToArray();
        Array.Reverse(dims);
        return dims;
    }

    private static object ReadArray(Cursor c, int typecode, ArrayDesc a)
    {
        var n = checked((int)a.NElements);
        var shape = Shape(a);
        object result;
        switch (typecode)
        {
            case 1 or 3 or 4 or 5 or 13 or 14 or 15:
            {
                if (typecode == 1) c.Long(); // byte arrays repeat their length
                var size = typecode switch { 1 => 1, 3 or 4 or 13 => 4, _ => 8 };
                var s = c.Bytes(a.NBytes);
                if (s.Length < (long)n * size) throw new InvalidDataException("IDL SAVE array is shorter than its descriptor.");
                var v = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var x = s.Slice(i * size, size);
                    v[i] = typecode switch
                    {
                        1 => x[0],
                        3 => BinaryPrimitives.ReadInt32BigEndian(x),
                        4 => BinaryPrimitives.ReadSingleBigEndian(x),
                        5 => BinaryPrimitives.ReadDoubleBigEndian(x),
                        13 => BinaryPrimitives.ReadUInt32BigEndian(x),
                        14 => BinaryPrimitives.ReadInt64BigEndian(x),
                        _ => BinaryPrimitives.ReadUInt64BigEndian(x),
                    };
                }
                result = new NdArray(v, shape);
                break;
            }
            case 2 or 12:
            {
                // 16-bit elements are stored in 32-bit slots (value in the low half).
                var s = c.Bytes(a.NBytes * 2);
                var v = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var x = s.Slice(4 * i + 2, 2);
                    v[i] = typecode == 2 ? BinaryPrimitives.ReadInt16BigEndian(x) : BinaryPrimitives.ReadUInt16BigEndian(x);
                }
                result = new NdArray(v, shape);
                break;
            }
            case 6 or 9:
            {
                var size = typecode == 6 ? 4 : 8;
                var s = c.Bytes(a.NBytes);
                var z = new Complex[n];
                for (var i = 0; i < n; i++)
                {
                    var re = size == 4 ? BinaryPrimitives.ReadSingleBigEndian(s[(8 * i)..]) : BinaryPrimitives.ReadDoubleBigEndian(s[(16 * i)..]);
                    var im = size == 4 ? BinaryPrimitives.ReadSingleBigEndian(s[(8 * i + 4)..]) : BinaryPrimitives.ReadDoubleBigEndian(s[(16 * i + 8)..]);
                    z[i] = new Complex(re, im);
                }
                result = new ComplexNdArray(z, shape);
                break;
            }
            default:
            {
                var items = new object?[n];
                for (var i = 0; i < n; i++) items[i] = ReadScalar(c, typecode);
                result = new IdlObjectArray(shape, items);
                break;
            }
        }
        c.Align();
        return result;
    }

    private static IdlStructure ReadStructure(Cursor c, ArrayDesc a, StructDesc sd)
    {
        var n = checked((int)a.NElements);
        var rows = new object?[n][];
        for (var i = 0; i < n; i++)
        {
            var row = new object?[sd.Tags.Length];
            for (var k = 0; k < sd.Tags.Length; k++)
            {
                var t = sd.Tags[k];
                row[k] = t.IsStruct ? ReadStructure(c, sd.ArrTable[t.Name], sd.StructTable[t.Name])
                    : t.IsArray ? ReadArray(c, t.TypeCode, sd.ArrTable[t.Name])
                    : ReadScalar(c, t.TypeCode);
            }
            rows[i] = row;
        }
        return new IdlStructure(sd.Name, sd.Tags.Select(t => t.Name).ToArray(), rows, Shape(a));
    }

    /// <summary>Replaces pointers by their heap targets, recursively (null for index 0 or a dangling pointer).</summary>
    private static object? Resolve(object? v, Dictionary<int, object?> heap, int depth)
    {
        if (depth > 64) throw new InvalidDataException("IDL SAVE pointers nest more than 64 deep (cyclic heap?).");
        switch (v)
        {
            case Pointer p:
                return p.Index != 0 && heap.TryGetValue(p.Index, out var target) ? Resolve(target, heap, depth + 1) : null;
            case IdlObjectArray oa:
                for (var i = 0; i < oa.Items.Length; i++) oa.Items[i] = Resolve(oa.Items[i], heap, depth + 1);
                return oa;
            case IdlStructure s:
                foreach (var row in s.Rows)
                    for (var k = 0; k < row.Length; k++) row[k] = Resolve(row[k], heap, depth + 1);
                return s;
            default:
                return v;
        }
    }
}
