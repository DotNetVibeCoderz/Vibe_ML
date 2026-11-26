using System.Collections.Concurrent;
using System.Reflection;
using System.Text;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>Encodes and decodes one kind of value for <see cref="WireCodec"/>.</summary>
public interface IWireSerializer
{
    /// <summary>The type this serializer handles.</summary>
    Type Type { get; }

    /// <summary>Writes a value.</summary>
    void Write(BinaryWriter writer, object value);

    /// <summary>Reads a value.</summary>
    object Read(BinaryReader reader);
}

/// <summary>
/// The binary format task arguments and results travel in between a client and its workers.
/// </summary>
/// <remarks>
/// <para>
/// A closed set of tagged types - numbers, strings, arrays of them, <see cref="NdArray"/>,
/// <see cref="DataFrame"/>, nested object arrays, task references and remote calls - plus whatever
/// an application registers with <see cref="Register"/>. Nothing is constructed from a type name
/// on the wire except the methods of a <see cref="RemoteCall"/>, and those are checked against
/// <see cref="RemoteTaskAttribute"/>. That is the difference from BinaryFormatter-style
/// serialisation, where the payload chooses what to instantiate.
/// </para>
/// <para>
/// DataFrames are written column by column in their own typed layout rather than as Arrow: NaN
/// and missing stay distinct per type, timestamps keep their ticks and categorical columns their
/// codes, so a partition comes back exactly as it left.
/// </para>
/// </remarks>
public static class WireCodec
{
    private static readonly ConcurrentDictionary<string, IWireSerializer> Custom = new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, MethodInfo> Methods = new(StringComparer.Ordinal);

    /// <summary>Teaches the codec a new type. Both the client and every worker must register it.</summary>
    public static void Register(IWireSerializer serializer)
        => Custom[serializer.Type.FullName ?? serializer.Type.Name] = serializer;

    /// <summary>A value as bytes.</summary>
    public static byte[] Serialize(object? value)
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true)) Write(w, value);
        return ms.ToArray();
    }

    /// <summary>A value from bytes.</summary>
    public static object? Deserialize(byte[] bytes)
    {
        using var r = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        return Read(r);
    }

    private enum Tag : byte
    {
        Null, Bool, Int, Long, Double, String, DoubleArray, IntArray, LongArray, StringArray, BoolArray, Bytes,
        ObjectArray, NdArray, Frame, FrameArray, NdArrayArray, DoubleJagged, Key, Date, Call, Ref, RefList, Custom,
        NullableBoolArray, Tuple2,
    }

    /// <summary>Writes one value.</summary>
    public static void Write(BinaryWriter w, object? value)
    {
        switch (value)
        {
            case null: w.Write((byte)Tag.Null); break;
            case bool b: w.Write((byte)Tag.Bool); w.Write(b); break;
            case int i: w.Write((byte)Tag.Int); w.Write(i); break;
            case long l: w.Write((byte)Tag.Long); w.Write(l); break;
            case double d: w.Write((byte)Tag.Double); w.Write(d); break;
            case string s: w.Write((byte)Tag.String); w.Write(s); break;
            case DateTime t: w.Write((byte)Tag.Date); w.Write(t.Ticks); w.Write((byte)t.Kind); break;
            case double[] a: w.Write((byte)Tag.DoubleArray); WriteDoubles(w, a); break;
            case int[] a: w.Write((byte)Tag.IntArray); w.Write(a.Length); foreach (var x in a) w.Write(x); break;
            case long[] a: w.Write((byte)Tag.LongArray); w.Write(a.Length); foreach (var x in a) w.Write(x); break;
            case bool[] a: w.Write((byte)Tag.BoolArray); w.Write(a.Length); foreach (var x in a) w.Write(x); break;
            case bool?[] a: w.Write((byte)Tag.NullableBoolArray); w.Write(a.Length); foreach (var x in a) w.Write(x is null ? (byte)2 : x.Value ? (byte)1 : (byte)0); break;
            case byte[] a: w.Write((byte)Tag.Bytes); w.Write(a.Length); w.Write(a); break;
            case string?[] a: w.Write((byte)Tag.StringArray); WriteStrings(w, a); break;
            case NdArray n: w.Write((byte)Tag.NdArray); WriteNdArray(w, n); break;
            case DataFrame f: w.Write((byte)Tag.Frame); WriteFrame(w, f); break;
            case DataFrame[] fs: w.Write((byte)Tag.FrameArray); w.Write(fs.Length); foreach (var f in fs) WriteFrame(w, f); break;
            case NdArray[] ns: w.Write((byte)Tag.NdArrayArray); w.Write(ns.Length); foreach (var n in ns) WriteNdArray(w, n); break;
            case double[][] j: w.Write((byte)Tag.DoubleJagged); w.Write(j.Length); foreach (var a in j) WriteDoubles(w, a); break;
            case TaskKey k: w.Write((byte)Tag.Key); WriteKey(w, k); break;
            case TaskRef r: w.Write((byte)Tag.Ref); WriteKey(w, r.Key); break;
            case TaskRefList l: w.Write((byte)Tag.RefList); w.Write(l.Keys.Count); foreach (var k in l.Keys) WriteKey(w, k); break;
            case RemoteCall c: w.Write((byte)Tag.Call); WriteCall(w, c); break;
            case object?[] o: w.Write((byte)Tag.ObjectArray); w.Write(o.Length); foreach (var x in o) Write(w, x); break;
            default:
                if (Custom.TryGetValue(value.GetType().FullName ?? "", out var custom))
                {
                    w.Write((byte)Tag.Custom);
                    w.Write(value.GetType().FullName!);
                    custom.Write(w, value);
                    break;
                }
                throw new NotSupportedException(
                    $"Cannot send a {value.GetType().Name} to a worker. Use numbers, strings, arrays, NdArray or DataFrame, or register an IWireSerializer.");
        }
    }

    /// <summary>Reads one value.</summary>
    public static object? Read(BinaryReader r)
    {
        var tag = (Tag)r.ReadByte();
        switch (tag)
        {
            case Tag.Null: return null;
            case Tag.Bool: return r.ReadBoolean();
            case Tag.Int: return r.ReadInt32();
            case Tag.Long: return r.ReadInt64();
            case Tag.Double: return r.ReadDouble();
            case Tag.String: return r.ReadString();
            case Tag.Date: { var ticks = r.ReadInt64(); return new DateTime(ticks, (DateTimeKind)r.ReadByte()); }
            case Tag.DoubleArray: return ReadDoubles(r);
            case Tag.IntArray: { var a = new int[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = r.ReadInt32(); return a; }
            case Tag.LongArray: { var a = new long[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = r.ReadInt64(); return a; }
            case Tag.BoolArray: { var a = new bool[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = r.ReadBoolean(); return a; }
            case Tag.NullableBoolArray:
            {
                var a = new bool?[r.ReadInt32()];
                for (var i = 0; i < a.Length; i++) { var b = r.ReadByte(); a[i] = b == 2 ? null : b == 1; }
                return a;
            }
            case Tag.Bytes: return r.ReadBytes(r.ReadInt32());
            case Tag.StringArray: return ReadStrings(r);
            case Tag.NdArray: return ReadNdArray(r);
            case Tag.Frame: return ReadFrame(r);
            case Tag.FrameArray: { var a = new DataFrame[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = ReadFrame(r); return a; }
            case Tag.NdArrayArray: { var a = new NdArray[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = ReadNdArray(r); return a; }
            case Tag.DoubleJagged: { var a = new double[r.ReadInt32()][]; for (var i = 0; i < a.Length; i++) a[i] = ReadDoubles(r); return a; }
            case Tag.Key: return ReadKey(r);
            case Tag.Ref: return new TaskRef(ReadKey(r));
            case Tag.RefList: { var a = new TaskKey[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = ReadKey(r); return new TaskRefList(a); }
            case Tag.Call: return ReadCall(r);
            case Tag.ObjectArray: { var a = new object?[r.ReadInt32()]; for (var i = 0; i < a.Length; i++) a[i] = Read(r); return a; }
            case Tag.Custom:
            {
                var type = r.ReadString();
                return Custom.TryGetValue(type, out var s) ? s.Read(r)
                    : throw new NotSupportedException($"No serializer registered for {type} on this side.");
            }
            default: throw new InvalidDataException($"Unknown wire tag {(byte)tag}.");
        }
    }

    internal static void WriteKey(BinaryWriter w, TaskKey k) { w.Write(k.Name); w.Write(k.Index); }

    internal static TaskKey ReadKey(BinaryReader r) => new(r.ReadString(), r.ReadInt32());

    private static void WriteDoubles(BinaryWriter w, double[] a)
    {
        w.Write(a.Length);
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));
    }

    private static double[] ReadDoubles(BinaryReader r)
    {
        var a = new double[r.ReadInt32()];
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan());
        var read = 0;
        while (read < bytes.Length)
        {
            var n = r.Read(bytes[read..]);
            if (n == 0) throw new EndOfStreamException();
            read += n;
        }
        return a;
    }

    private static void WriteStrings(BinaryWriter w, IReadOnlyList<string?> a)
    {
        w.Write(a.Count);
        foreach (var s in a)
        {
            w.Write(s is not null);
            if (s is not null) w.Write(s);
        }
    }

    private static string?[] ReadStrings(BinaryReader r)
    {
        var a = new string?[r.ReadInt32()];
        for (var i = 0; i < a.Length; i++) a[i] = r.ReadBoolean() ? r.ReadString() : null;
        return a;
    }

    private static void WriteNdArray(BinaryWriter w, NdArray n)
    {
        w.Write(n.Rank);
        foreach (var d in n.Shape) w.Write(d);
        WriteDoubles(w, n.ToArray());
    }

    private static NdArray ReadNdArray(BinaryReader r)
    {
        var shape = new int[r.ReadInt32()];
        for (var i = 0; i < shape.Length; i++) shape[i] = r.ReadInt32();
        return new NdArray(ReadDoubles(r), shape);
    }

    private static void WriteFrame(BinaryWriter w, DataFrame f)
    {
        w.Write(f.ColumnCount);
        w.Write(f.RowCount);
        foreach (var c in f.Columns)
        {
            w.Write(c.Name);
            switch (c)
            {
                case NumericSeries n: w.Write((byte)0); WriteDoubles(w, n.Values); break;
                case TextSeries t: w.Write((byte)1); WriteStrings(w, t.Values.ToArray()); break;
                case BooleanSeries b:
                    w.Write((byte)2);
                    for (var i = 0; i < b.Length; i++) w.Write(b[i] is null ? (byte)2 : b[i]!.Value ? (byte)1 : (byte)0);
                    break;
                case DateTimeSeries d:
                    w.Write((byte)3);
                    for (var i = 0; i < d.Length; i++) w.Write(d[i]?.Ticks ?? long.MinValue);
                    break;
                case CategoricalSeries k:
                    w.Write((byte)4);
                    w.Write(k.IsOrdered);
                    WriteStrings(w, k.Categories);
                    for (var i = 0; i < k.Length; i++) w.Write(k.Codes[i]);
                    break;
                default:
                    w.Write((byte)1);
                    WriteStrings(w, Enumerable.Range(0, c.Length).Select(i => c.GetValue(i)?.ToString()).ToArray());
                    break;
            }
        }
    }

    private static DataFrame ReadFrame(BinaryReader r)
    {
        var columns = r.ReadInt32();
        var rows = r.ReadInt32();
        var list = new List<Series>(columns);
        for (var c = 0; c < columns; c++)
        {
            var name = r.ReadString();
            switch (r.ReadByte())
            {
                case 0: list.Add(new NumericSeries(name, ReadDoubles(r))); break;
                case 1: list.Add(new TextSeries(name, ReadStrings(r))); break;
                case 2:
                {
                    var a = new bool?[rows];
                    for (var i = 0; i < rows; i++) { var b = r.ReadByte(); a[i] = b == 2 ? null : b == 1; }
                    list.Add(new BooleanSeries(name, a));
                    break;
                }
                case 3:
                {
                    var a = new DateTime?[rows];
                    for (var i = 0; i < rows; i++) { var t = r.ReadInt64(); a[i] = t == long.MinValue ? null : new DateTime(t); }
                    list.Add(new DateTimeSeries(name, a));
                    break;
                }
                case 4:
                {
                    var ordered = r.ReadBoolean();
                    var categories = ReadStrings(r).Select(s => s!).ToArray();
                    var codes = new int[rows];
                    for (var i = 0; i < rows; i++) codes[i] = r.ReadInt32();
                    list.Add(new CategoricalSeries(name, codes, categories, ordered));
                    break;
                }
                default: throw new InvalidDataException("Unknown column kind.");
            }
        }
        return new DataFrame(list);
    }

    private static void WriteCall(BinaryWriter w, RemoteCall call)
    {
        w.Write(MethodId(call.Method));
        w.Write(call.Arguments.Count);
        foreach (var a in call.Arguments) Write(w, a);
    }

    private static RemoteCall ReadCall(BinaryReader r)
    {
        var method = ResolveMethod(r.ReadString());
        var args = new object?[r.ReadInt32()];
        for (var i = 0; i < args.Length; i++) args[i] = Read(r);
        return new RemoteCall(method, args); // re-checks [RemoteTask]
    }

    /// <summary>"Namespace.Type, Assembly::Method(ParamType1,ParamType2)".</summary>
    internal static string MethodId(MethodInfo m)
        => $"{m.DeclaringType!.FullName}, {m.DeclaringType.Assembly.GetName().Name}::{m.Name}({string.Join(",", m.GetParameters().Select(p => p.ParameterType.FullName))})";

    internal static MethodInfo ResolveMethod(string id) => Methods.GetOrAdd(id, static id =>
    {
        var split = id.IndexOf("::", StringComparison.Ordinal);
        var typeName = id[..split];
        var signature = id[(split + 2)..];
        var name = signature[..signature.IndexOf('(')];
        var parameters = signature[(name.Length + 1)..^1];

        var type = Type.GetType(typeName, throwOnError: false)
            ?? AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType(typeName[..typeName.IndexOf(',')], throwOnError: false))
                .FirstOrDefault(t => t is not null)
            ?? throw new TypeLoadException($"Type {typeName} is not loaded on this worker; preload its assembly.");

        var method = type.GetMethods(BindingFlags.Public | BindingFlags.Static).FirstOrDefault(m =>
            m.Name == name && string.Join(",", m.GetParameters().Select(p => p.ParameterType.FullName)) == parameters)
            ?? throw new MissingMethodException(typeName, signature);
        RemoteCall.EnsureCallable(method);
        return method;
    });
}
