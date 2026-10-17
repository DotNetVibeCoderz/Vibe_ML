using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Gravicode.Science.SciNet.IO;

/// <summary>HDF5 datatype classes (the class field of the datatype message).</summary>
public enum Hdf5TypeClass
{
    /// <summary>Fixed-point (integer) numbers.</summary>
    Integer = 0,
    /// <summary>IEEE floating-point numbers (half, single, double).</summary>
    Float = 1,
    /// <summary>Time values (obsolete; read as raw bytes only).</summary>
    Time = 2,
    /// <summary>Fixed-length strings.</summary>
    String = 3,
    /// <summary>Bit fields (read as unsigned integers).</summary>
    Bitfield = 4,
    /// <summary>Opaque byte blobs.</summary>
    Opaque = 5,
    /// <summary>Compound (record) types.</summary>
    Compound = 6,
    /// <summary>Object or region references.</summary>
    Reference = 7,
    /// <summary>Enumerations over an integer base type (h5py stores <c>bool</c> this way).</summary>
    Enum = 8,
    /// <summary>Variable-length sequences and strings, stored in the global heap.</summary>
    VariableLength = 9,
    /// <summary>Fixed-size arrays of a base type.</summary>
    Array = 10,
}

/// <summary>A member of a compound datatype.</summary>
/// <param name="Name">Member name.</param>
/// <param name="Offset">Byte offset within the record.</param>
/// <param name="Type">Member type.</param>
public sealed record Hdf5CompoundMember(string Name, int Offset, Hdf5DataType Type);

/// <summary>
/// An HDF5 datatype as read from a file. Only what is needed to decode the data is modelled:
/// byte order, size, signedness, string padding/charset, compound members, enum members, and the
/// base type of arrays, enums and variable-length types.
/// </summary>
public sealed class Hdf5DataType
{
    private Hdf5DataType() { }

    /// <summary>The datatype class.</summary>
    public Hdf5TypeClass Class { get; private init; }

    /// <summary>Size of one element in bytes (for variable-length types, of the in-file heap descriptor).</summary>
    public int Size { get; private init; }

    /// <summary>True for signed integers (and enums over them).</summary>
    public bool IsSigned { get; private init; }

    /// <summary>True when numbers are stored big-endian.</summary>
    public bool IsBigEndian { get; private init; }

    /// <summary>Precision in bits for integers and bit fields (normally <c>8 * Size</c>).</summary>
    public int Precision { get; private init; }

    /// <summary>Bit offset of the value within an integer or bit field (normally 0).</summary>
    public int BitOffset { get; private init; }

    /// <summary>True for UTF-8 strings, false for ASCII.</summary>
    public bool IsUtf8 { get; private init; }

    /// <summary>String padding: 0 = NUL-terminated, 1 = NUL-padded, 2 = space-padded.</summary>
    public int StringPadding { get; private init; }

    /// <summary>True for a variable-length string (a <see cref="Hdf5TypeClass.VariableLength"/> of characters).</summary>
    public bool IsVariableLengthString { get; private init; }

    /// <summary>Compound members (empty otherwise).</summary>
    public IReadOnlyList<Hdf5CompoundMember> Members { get; private init; } = [];

    /// <summary>Base type of an enum, array or variable-length sequence.</summary>
    public Hdf5DataType? BaseType { get; private init; }

    /// <summary>Enum members, name → value, in declaration order (empty otherwise).</summary>
    public IReadOnlyList<KeyValuePair<string, long>> EnumMembers { get; private init; } = [];

    /// <summary>Dimensions of an array type (empty otherwise).</summary>
    public IReadOnlyList<int> ArrayDimensions { get; private init; } = [];

    /// <summary>For references: true for dataset-region references, false for object references.</summary>
    public bool IsRegionReference { get; private init; }

    /// <summary>Opaque tag (empty otherwise).</summary>
    public string Tag { get; private init; } = "";

    internal int FloatExponentBits { get; private init; }
    internal int FloatMantissaBits { get; private init; }
    internal int ReferenceVersion { get; private init; }

    /// <summary>True for types whose values convert to <c>double</c>: integers, floats, bit fields and enums.</summary>
    public bool IsNumeric => Class is Hdf5TypeClass.Integer or Hdf5TypeClass.Float or Hdf5TypeClass.Bitfield
        || (Class == Hdf5TypeClass.Enum && BaseType!.IsNumeric);

    /// <summary>True for an enum with exactly the members FALSE = 0 and TRUE = 1, which is how h5py stores <c>bool</c>.</summary>
    public bool IsBoolean => Class == Hdf5TypeClass.Enum && EnumMembers.Count == 2
        && EnumMembers.Any(m => m.Key == "FALSE" && m.Value == 0) && EnumMembers.Any(m => m.Key == "TRUE" && m.Value == 1);

    /// <summary>A NumPy-style description such as <c>&lt;f8</c>, <c>&gt;i4</c>, <c>|S10</c>, <c>vlen str</c>.</summary>
    public override string ToString()
    {
        var o = IsBigEndian ? ">" : "<";
        return Class switch
        {
            Hdf5TypeClass.Integer => $"{(Size == 1 ? "|" : o)}{(IsSigned ? 'i' : 'u')}{Size}",
            Hdf5TypeClass.Float => $"{o}f{Size}",
            Hdf5TypeClass.Bitfield => $"{o}b{Size}",
            Hdf5TypeClass.String => $"|S{Size}",
            Hdf5TypeClass.Opaque => $"|V{Size}",
            Hdf5TypeClass.Compound => "{" + string.Join(", ", Members.Select(m => $"{m.Name}: {m.Type}")) + "}",
            Hdf5TypeClass.Reference => IsRegionReference ? "region ref" : "object ref",
            Hdf5TypeClass.Enum => IsBoolean ? "bool" : $"enum({BaseType})",
            Hdf5TypeClass.VariableLength => IsVariableLengthString ? "vlen str" : $"vlen({BaseType})",
            Hdf5TypeClass.Array => $"({string.Join(", ", ArrayDimensions)}){BaseType}",
            _ => "time",
        };
    }

    internal static Hdf5DataType Parse(H5Buf b)
    {
        var b0 = b.U8();
        var cls = b0 & 0x0F;
        var ver = b0 >> 4;
        var bits = b.U8() | (b.U8() << 8) | (b.U8() << 16);
        var size = (int)b.U32();
        switch (cls)
        {
            case 0:
            case 4:
            {
                var off = b.U16();
                var prec = b.U16();
                return new Hdf5DataType
                {
                    Class = cls == 0 ? Hdf5TypeClass.Integer : Hdf5TypeClass.Bitfield,
                    Size = size, IsBigEndian = (bits & 1) != 0, IsSigned = cls == 0 && (bits & 8) != 0,
                    BitOffset = off, Precision = prec,
                };
            }
            case 1:
            {
                if ((bits & 0x40) != 0) throw new NotSupportedException("VAX-order HDF5 floating-point data is not supported.");
                b.U16();
                var prec = b.U16();
                b.U8();
                var expSize = b.U8();
                b.U8();
                var mantSize = b.U8();
                b.U32();
                return new Hdf5DataType
                {
                    Class = Hdf5TypeClass.Float, Size = size, IsBigEndian = (bits & 1) != 0, Precision = prec,
                    FloatExponentBits = expSize, FloatMantissaBits = mantSize, IsSigned = true,
                };
            }
            case 2:
                b.U16();
                return new Hdf5DataType { Class = Hdf5TypeClass.Time, Size = size, IsBigEndian = (bits & 1) != 0 };
            case 3:
                return new Hdf5DataType { Class = Hdf5TypeClass.String, Size = size, StringPadding = bits & 0xF, IsUtf8 = ((bits >> 4) & 0xF) == 1 };
            case 5:
            {
                var tagLen = bits & 0xFF;
                var raw = b.Span(tagLen);
                var z = raw.IndexOf((byte)0);
                return new Hdf5DataType { Class = Hdf5TypeClass.Opaque, Size = size, Tag = Encoding.ASCII.GetString(z >= 0 ? raw[..z] : raw) };
            }
            case 6:
            {
                var n = bits & 0xFFFF;
                var members = new Hdf5CompoundMember[n];
                for (var i = 0; i < n; i++)
                {
                    var start = b.Pos;
                    var name = b.CString();
                    int offset;
                    Hdf5DataType mt;
                    if (ver == 1)
                    {
                        b.Align8(start);
                        offset = (int)b.U32();
                        var dimensionality = b.U8();
                        b.Skip(3 + 4 + 4);
                        var dims = new int[4];
                        for (var k = 0; k < 4; k++) dims[k] = (int)b.U32();
                        mt = Parse(b);
                        if (dimensionality > 0)
                            mt = new Hdf5DataType { Class = Hdf5TypeClass.Array, Size = mt.Size * dims.Take(dimensionality).Aggregate(1, (p, d) => p * d), BaseType = mt, ArrayDimensions = dims[..dimensionality] };
                    }
                    else if (ver == 2)
                    {
                        b.Align8(start);
                        offset = (int)b.U32();
                        mt = Parse(b);
                    }
                    else
                    {
                        var w = size < 1 << 8 ? 1 : size < 1 << 16 ? 2 : size < 1 << 24 ? 3 : 4;
                        offset = (int)b.UN(w);
                        mt = Parse(b);
                    }
                    if (offset < 0 || offset + mt.Size > size) throw new InvalidDataException($"Malformed HDF5 compound type: member '{name}' lies outside the record.");
                    members[i] = new Hdf5CompoundMember(name, offset, mt);
                }
                return new Hdf5DataType { Class = Hdf5TypeClass.Compound, Size = size, Members = members };
            }
            case 7:
            {
                var kind = bits & 0xF;
                if (ver >= 4) return new Hdf5DataType { Class = Hdf5TypeClass.Reference, Size = size, ReferenceVersion = ver, IsRegionReference = kind != 0 && kind != 2 };
                return new Hdf5DataType { Class = Hdf5TypeClass.Reference, Size = size, IsRegionReference = kind == 1, ReferenceVersion = ver };
            }
            case 8:
            {
                var n = bits & 0xFFFF;
                var bt = Parse(b);
                var names = new string[n];
                for (var i = 0; i < n; i++)
                {
                    var start = b.Pos;
                    names[i] = b.CString();
                    if (ver < 3) b.Align8(start);
                }
                var members = new KeyValuePair<string, long>[n];
                for (var i = 0; i < n; i++)
                {
                    var raw = b.Span(bt.Size);
                    members[i] = new(names[i], (long)Hdf5Convert.ReadInteger(bt, raw));
                }
                return new Hdf5DataType
                {
                    Class = Hdf5TypeClass.Enum, Size = size, BaseType = bt, EnumMembers = members,
                    IsSigned = bt.IsSigned, IsBigEndian = bt.IsBigEndian,
                };
            }
            case 9:
            {
                var isString = (bits & 0xF) == 1;
                var bt = Parse(b);
                return new Hdf5DataType
                {
                    Class = Hdf5TypeClass.VariableLength, Size = size, BaseType = bt, IsVariableLengthString = isString,
                    StringPadding = (bits >> 4) & 0xF, IsUtf8 = ((bits >> 8) & 0xF) == 1,
                };
            }
            case 10:
            {
                var rank = b.U8();
                if (ver < 3) b.Skip(3);
                var dims = new int[rank];
                for (var i = 0; i < rank; i++) dims[i] = (int)b.U32();
                if (ver < 3) b.Skip(4 * rank);
                var bt = Parse(b);
                return new Hdf5DataType { Class = Hdf5TypeClass.Array, Size = size, BaseType = bt, ArrayDimensions = dims };
            }
            default:
                throw new NotSupportedException($"HDF5 datatype class {cls} is not supported.");
        }
    }
}

/// <summary>Decoding of raw HDF5 element bytes into .NET values.</summary>
internal static class Hdf5Convert
{
    /// <summary>One integer (any size, either byte order, honouring precision/offset) as a raw 64-bit pattern, sign-extended for signed types.</summary>
    public static ulong ReadInteger(Hdf5DataType t, ReadOnlySpan<byte> s)
    {
        ulong v = 0;
        var n = t.Size;
        if (n > 8) throw new NotSupportedException($"{8 * n}-bit integers are not supported.");
        if (t.IsBigEndian) for (var i = 0; i < n; i++) v = (v << 8) | s[i];
        else for (var i = n - 1; i >= 0; i--) v = (v << 8) | s[i];
        var prec = t.Precision == 0 ? 8 * n : t.Precision;
        if (t.BitOffset != 0) v >>= t.BitOffset;
        if (prec < 64)
        {
            v &= (1UL << prec) - 1;
            if (t.IsSigned && (v & (1UL << (prec - 1))) != 0) v |= ~((1UL << prec) - 1);
        }
        return v;
    }

    private static bool IsStandardInteger(Hdf5DataType t)
        => t.Size is 1 or 2 or 4 or 8 && t.BitOffset == 0 && (t.Precision == 0 || t.Precision == 8 * t.Size);

    private static void CheckFloat(Hdf5DataType t)
    {
        var ok = (t.Size, t.FloatExponentBits, t.FloatMantissaBits) is (2, 5, 10) or (4, 8, 23) or (8, 11, 52);
        if (!ok) throw new NotSupportedException($"HDF5 {8 * t.Size}-bit floating point with {t.FloatExponentBits} exponent / {t.FloatMantissaBits} mantissa bits is not an IEEE half/single/double and is not supported.");
    }

    /// <summary>Numeric elements (integer, float, bitfield, enum) widened to double.</summary>
    public static void ToDouble(Hdf5DataType t, ReadOnlySpan<byte> src, Span<double> dst)
    {
        if (t.Class == Hdf5TypeClass.Enum) t = t.BaseType!;
        var n = dst.Length;
        var size = t.Size;
        if (src.Length < (long)n * size) throw new InvalidDataException("Malformed HDF5 data: fewer bytes than elements.");
        src = src[..(n * size)];
        if (t.Class == Hdf5TypeClass.Float)
        {
            CheckFloat(t);
            var swap = t.IsBigEndian != !BitConverter.IsLittleEndian;
            switch (size)
            {
                case 8:
                    if (!swap) { MemoryMarshal.Cast<byte, double>(src).CopyTo(dst); return; }
                    for (var i = 0; i < n; i++) dst[i] = BinaryPrimitives.ReadDoubleBigEndian(src.Slice(8 * i, 8));
                    return;
                case 4:
                    if (!swap) { var f = MemoryMarshal.Cast<byte, float>(src); for (var i = 0; i < n; i++) dst[i] = f[i]; return; }
                    for (var i = 0; i < n; i++) dst[i] = BinaryPrimitives.ReadSingleBigEndian(src.Slice(4 * i, 4));
                    return;
                default:
                    for (var i = 0; i < n; i++)
                        dst[i] = (double)(t.IsBigEndian ? BinaryPrimitives.ReadHalfBigEndian(src.Slice(2 * i, 2)) : BinaryPrimitives.ReadHalfLittleEndian(src.Slice(2 * i, 2)));
                    return;
            }
        }
        if (t.Class is not (Hdf5TypeClass.Integer or Hdf5TypeClass.Bitfield))
            throw new InvalidOperationException($"HDF5 type {t} is not numeric.");
        if (IsStandardInteger(t) && t.IsBigEndian == !BitConverter.IsLittleEndian)
        {
            switch (size, t.IsSigned)
            {
                case (1, true): for (var i = 0; i < n; i++) dst[i] = (sbyte)src[i]; return;
                case (1, false): for (var i = 0; i < n; i++) dst[i] = src[i]; return;
                case (2, true): { var v = MemoryMarshal.Cast<byte, short>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
                case (2, false): { var v = MemoryMarshal.Cast<byte, ushort>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
                case (4, true): { var v = MemoryMarshal.Cast<byte, int>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
                case (4, false): { var v = MemoryMarshal.Cast<byte, uint>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
                case (8, true): { var v = MemoryMarshal.Cast<byte, long>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
                case (8, false): { var v = MemoryMarshal.Cast<byte, ulong>(src); for (var i = 0; i < n; i++) dst[i] = v[i]; return; }
            }
        }
        for (var i = 0; i < n; i++)
        {
            var raw = ReadInteger(t, src.Slice(i * size, size));
            dst[i] = t.IsSigned ? (long)raw : raw;
        }
    }

    /// <summary>Numeric elements converted to <typeparamref name="T"/>; out-of-range values throw rather than wrap.</summary>
    public static T[] ToNumber<T>(Hdf5DataType t, ReadOnlySpan<byte> src, int n) where T : unmanaged, INumberBase<T>
    {
        var bt = t.Class == Hdf5TypeClass.Enum ? t.BaseType! : t;
        var r = new T[n];
        if (bt.Class == Hdf5TypeClass.Float)
        {
            CheckFloat(bt);
            if (typeof(T) == typeof(double)) { ToDouble(bt, src, MemoryMarshal.Cast<T, double>(r.AsSpan())); return r; }
            if (typeof(T) == typeof(float) && bt.Size == 4 && !bt.IsBigEndian && BitConverter.IsLittleEndian)
            {
                src[..(4 * n)].CopyTo(MemoryMarshal.AsBytes(r.AsSpan()));
                return r;
            }
            var d = new double[n];
            ToDouble(bt, src, d);
            for (var i = 0; i < n; i++) r[i] = T.CreateChecked(d[i]);
            return r;
        }
        if (bt.Class is not (Hdf5TypeClass.Integer or Hdf5TypeClass.Bitfield))
            throw new InvalidOperationException($"HDF5 type {t} is not numeric.");
        var size = bt.Size;
        if (src.Length < (long)n * size) throw new InvalidDataException("Malformed HDF5 data: fewer bytes than elements.");
        if (Unsafe.SizeOf<T>() == size && IsStandardInteger(bt) && !bt.IsBigEndian && BitConverter.IsLittleEndian && SameIntegerKind<T>(bt.IsSigned))
        {
            src[..(size * n)].CopyTo(MemoryMarshal.AsBytes(r.AsSpan()));
            return r;
        }
        for (var i = 0; i < n; i++)
        {
            var raw = ReadInteger(bt, src.Slice(i * size, size));
            r[i] = bt.IsSigned ? T.CreateChecked((long)raw) : T.CreateChecked(raw);
        }
        return r;
    }

    private static bool SameIntegerKind<T>(bool signed)
    {
        if (typeof(T) == typeof(sbyte) || typeof(T) == typeof(short) || typeof(T) == typeof(int) || typeof(T) == typeof(long)) return signed;
        if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(uint) || typeof(T) == typeof(ulong)) return !signed;
        return false;
    }

    /// <summary>Decodes a fixed-length string element according to its padding.</summary>
    public static string FixedString(Hdf5DataType t, ReadOnlySpan<byte> s)
    {
        if (t.StringPadding is 0 or 1)
        {
            var z = s.IndexOf((byte)0);
            if (z >= 0) s = s[..z];
        }
        else
        {
            var e = s.Length;
            while (e > 0 && (s[e - 1] == (byte)' ' || s[e - 1] == 0)) e--;
            s = s[..e];
        }
        return t.IsUtf8 ? Encoding.UTF8.GetString(s) : Encoding.Latin1.GetString(s);
    }

    /// <summary>
    /// Elements in their natural .NET form: <c>sbyte[]</c> ... <c>ulong[]</c>, <c>Half[]</c>,
    /// <c>float[]</c>, <c>double[]</c>, <c>bool[]</c> (h5py booleans), <c>string[]</c> (fixed or
    /// variable-length strings), <see cref="Hdf5Reference"/>[], <c>Array[]</c> (variable-length
    /// sequences and array-typed elements), <c>Dictionary&lt;string, object?&gt;[]</c> (compound
    /// records), <c>byte[][]</c> (opaque).
    /// </summary>
    public static Array ToNatural(Hdf5Core core, Hdf5DataType t, ReadOnlySpan<byte> src, int n)
    {
        if (src.Length < (long)n * t.Size) throw new InvalidDataException("Malformed HDF5 data: fewer bytes than elements.");
        switch (t.Class)
        {
            case Hdf5TypeClass.Integer:
            case Hdf5TypeClass.Bitfield:
                return (t.Size, t.IsSigned) switch
                {
                    (1, true) => ToNumber<sbyte>(t, src, n),
                    (1, false) => ToNumber<byte>(t, src, n),
                    (2, true) => ToNumber<short>(t, src, n),
                    (2, false) => ToNumber<ushort>(t, src, n),
                    (4, true) => ToNumber<int>(t, src, n),
                    (4, false) => ToNumber<uint>(t, src, n),
                    (_, true) => ToNumber<long>(t, src, n),
                    _ => ToNumber<ulong>(t, src, n),
                };
            case Hdf5TypeClass.Float:
                return t.Size switch
                {
                    2 => ToNumber<Half>(t, src, n),
                    4 => ToNumber<float>(t, src, n),
                    _ => ToNumber<double>(t, src, n),
                };
            case Hdf5TypeClass.Enum:
            {
                if (t.IsBoolean)
                {
                    var r = new bool[n];
                    for (var i = 0; i < n; i++) r[i] = ReadInteger(t.BaseType!, src.Slice(i * t.Size, t.Size)) != 0;
                    return r;
                }
                return ToNatural(core, t.BaseType!, src, n);
            }
            case Hdf5TypeClass.String:
            {
                var r = new string[n];
                for (var i = 0; i < n; i++) r[i] = FixedString(t, src.Slice(i * t.Size, t.Size));
                return r;
            }
            case Hdf5TypeClass.VariableLength:
            {
                if (t.IsVariableLengthString)
                {
                    var r = new string?[n];
                    for (var i = 0; i < n; i++)
                    {
                        var bytes = core.ReadVlen(src.Slice(i * t.Size, t.Size), 1);
                        if (bytes is null) continue;
                        var z = bytes.AsSpan().IndexOf((byte)0);
                        var s = z >= 0 ? bytes.AsSpan(0, z) : bytes;
                        r[i] = t.IsUtf8 ? Encoding.UTF8.GetString(s) : Encoding.Latin1.GetString(s);
                    }
                    return r;
                }
                var seq = new Array[n];
                var bt = t.BaseType!;
                for (var i = 0; i < n; i++)
                {
                    var el = src.Slice(i * t.Size, t.Size);
                    var count = (int)BinaryPrimitives.ReadUInt32LittleEndian(el);
                    var bytes = core.ReadVlen(el, bt.Size) ?? [];
                    seq[i] = ToNatural(core, bt, bytes, Math.Min(count, bt.Size == 0 ? 0 : bytes.Length / bt.Size));
                }
                return seq;
            }
            case Hdf5TypeClass.Reference:
            {
                if (t.IsRegionReference || t.ReferenceVersion >= 4)
                    throw new NotSupportedException("HDF5 region references and HDF5 1.12-style references are not supported; object references are.");
                var r = new Hdf5Reference[n];
                for (var i = 0; i < n; i++)
                {
                    var b = new H5Buf(src.Slice(i * t.Size, t.Size).ToArray(), core.SizeOfOffsets, core.SizeOfLengths);
                    var a = b.Offset();
                    r[i] = new Hdf5Reference(a <= 0 ? -1 : a);
                }
                return r;
            }
            case Hdf5TypeClass.Array:
            {
                var r = new Array[n];
                var bt = t.BaseType!;
                var count = t.Size / Math.Max(1, bt.Size);
                for (var i = 0; i < n; i++) r[i] = ToNatural(core, bt, src.Slice(i * t.Size, t.Size), count);
                return r;
            }
            case Hdf5TypeClass.Compound:
            {
                var cols = Columns(core, t, src, n);
                var r = new Dictionary<string, object?>[n];
                for (var i = 0; i < n; i++)
                {
                    var d = new Dictionary<string, object?>(StringComparer.Ordinal);
                    foreach (var (k, col) in cols) d[k] = col.GetValue(i);
                    r[i] = d;
                }
                return r;
            }
            case Hdf5TypeClass.Opaque:
            {
                var r = new byte[n][];
                for (var i = 0; i < n; i++) r[i] = src.Slice(i * t.Size, t.Size).ToArray();
                return r;
            }
            default:
                throw new NotSupportedException($"Decoding HDF5 {t.Class} data is not supported.");
        }
    }

    /// <summary>A compound array split into one natural array per member (columnar).</summary>
    public static OrderedDictionary<string, Array> Columns(Hdf5Core core, Hdf5DataType t, ReadOnlySpan<byte> src, int n)
    {
        if (t.Class != Hdf5TypeClass.Compound) throw new InvalidOperationException($"HDF5 type {t} is not compound.");
        var result = new OrderedDictionary<string, Array>(StringComparer.Ordinal);
        foreach (var m in t.Members)
        {
            var ms = m.Type.Size;
            var buf = new byte[(long)ms * n];
            for (var i = 0; i < n; i++) src.Slice(i * t.Size + m.Offset, ms).CopyTo(buf.AsSpan(i * ms, ms));
            result[m.Name] = ToNatural(core, m.Type, buf, n);
        }
        return result;
    }
}

file static class Unsafe
{
    public static int SizeOf<T>() where T : unmanaged => System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
}
