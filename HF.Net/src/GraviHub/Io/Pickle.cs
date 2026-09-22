using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Gravicode.HFNet.GraviHub.Io;

/// <summary>
/// A reader for Python's pickle format, limited to what a PyTorch checkpoint contains.
/// </summary>
/// <remarks>
/// <para>
/// <b>This is not a general unpickler, and deliberately cannot become one.</b> Pickle is a stack
/// machine whose <c>REDUCE</c> opcode calls an arbitrary named callable, so a faithful
/// implementation of the format is a remote code execution primitive - which is the entire reason
/// safetensors exists. Here <c>GLOBAL</c> resolves against a fixed allow-list of the handful of
/// constructors <c>torch.save</c> emits, and anything else throws.
/// </para>
/// <para>
/// The result is a tree of plain .NET objects: <see cref="Dictionary{TKey,TValue}"/> for dicts,
/// <see cref="List{T}"/> for lists and tuples, <see cref="PickleGlobal"/> for a resolved name, and
/// <see cref="PickleReduce"/> for a constructor call that the caller interprets.
/// </para>
/// </remarks>
public sealed class PickleReader
{
    private readonly Stream _stream;
    private readonly List<object?> _memo = [];
    private readonly List<object?> _stack = [];
    private readonly List<int> _marks = [];
    private readonly Func<IReadOnlyList<object?>, object?>? _persistentLoad;

    /// <summary>Creates a reader over a pickle stream.</summary>
    /// <param name="stream">The stream to read. Read sequentially and not disposed.</param>
    /// <param name="persistentLoad">
    /// Resolves a <c>persistent_id</c>. PyTorch uses these to point at tensor storage held outside
    /// the pickle, so a checkpoint reader must supply one.
    /// </param>
    public PickleReader(Stream stream, Func<IReadOnlyList<object?>, object?>? persistentLoad = null)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _persistentLoad = persistentLoad;
    }

    /// <summary>Runs the stream to its <c>STOP</c> opcode and returns the single result.</summary>
    /// <exception cref="NotSupportedException">An opcode or a global outside the allow-list appeared.</exception>
    public object? Load()
    {
        while (true)
        {
            var read = _stream.ReadByte();
            if (read < 0) throw new InvalidDataException("Pickle stream ended before STOP.");

            var op = (byte)read;
            switch (op)
            {
                // ---------------------------------------------------------- protocol frame
                case (byte)'\x80': _stream.ReadByte(); break;                 // PROTO
                case (byte)'\x95': ReadUInt64(); break;                       // FRAME
                case (byte)'.': return Pop();                                 // STOP

                // ---------------------------------------------------------- constants
                case (byte)'N': Push(null); break;                            // NONE
                case (byte)'\x88': Push(true); break;                         // NEWTRUE
                case (byte)'\x89': Push(false); break;                        // NEWFALSE

                // ---------------------------------------------------------- integers
                case (byte)'J': Push(ReadInt32()); break;                     // BININT
                case (byte)'K': Push((long)(byte)_stream.ReadByte()); break;  // BININT1
                case (byte)'M': Push((long)ReadUInt16()); break;              // BININT2
                case (byte)'\x8a': Push(ReadLong((byte)_stream.ReadByte())); break;  // LONG1
                case (byte)'\x8b': Push(ReadLong(ReadInt32())); break;        // LONG4
                case (byte)'I': Push(ParseLong(ReadLine())); break;           // INT
                case (byte)'L': Push(ParseLong(ReadLine().TrimEnd('L'))); break; // LONG

                // ---------------------------------------------------------- floats
                case (byte)'G': Push(ReadDoubleBigEndian()); break;           // BINFLOAT
                case (byte)'F': Push(double.Parse(ReadLine(), CultureInfo.InvariantCulture)); break; // FLOAT

                // ---------------------------------------------------------- strings and bytes
                case (byte)'X': Push(ReadUtf8(ReadInt32())); break;           // BINUNICODE
                case (byte)'\x8c': Push(ReadUtf8((byte)_stream.ReadByte())); break; // SHORT_BINUNICODE
                case (byte)'\x8d': Push(ReadUtf8((int)ReadUInt64())); break;  // BINUNICODE8
                case (byte)'C': Push(ReadBytes((byte)_stream.ReadByte())); break;   // SHORT_BINBYTES
                case (byte)'B': Push(ReadBytes(ReadInt32())); break;          // BINBYTES
                case (byte)'\x8e': Push(ReadBytes((int)ReadUInt64())); break; // BINBYTES8
                case (byte)'V': Push(ReadLine()); break;                      // UNICODE
                case (byte)'S': Push(Unquote(ReadLine())); break;             // STRING

                // ---------------------------------------------------------- memo
                case (byte)'q': Memoize((byte)_stream.ReadByte()); break;     // BINPUT
                case (byte)'r': Memoize(ReadInt32()); break;                  // LONG_BINPUT
                case (byte)'\x94': Memoize(_memo.Count); break;               // MEMOIZE
                case (byte)'h': Push(_memo[(byte)_stream.ReadByte()]); break; // BINGET
                case (byte)'j': Push(_memo[ReadInt32()]); break;              // LONG_BINGET

                // ---------------------------------------------------------- containers
                case (byte)'}': Push(new Dictionary<object, object?>()); break;   // EMPTY_DICT
                case (byte)']': Push(new List<object?>()); break;                 // EMPTY_LIST
                case (byte)')': Push(new List<object?>()); break;                 // EMPTY_TUPLE
                case (byte)'(': _marks.Add(_stack.Count); break;                  // MARK

                case (byte)'\x85': PushTuple(1); break;                       // TUPLE1
                case (byte)'\x86': PushTuple(2); break;                       // TUPLE2
                case (byte)'\x87': PushTuple(3); break;                       // TUPLE3
                case (byte)'t': PushTuple(PopMark()); break;                  // TUPLE
                case (byte)'l': PushTuple(PopMark()); break;                  // LIST
                case (byte)'d': PushDict(PopMark()); break;                   // DICT

                case (byte)'a': Append(); break;                              // APPEND
                case (byte)'e': AppendMany(PopMark()); break;                 // APPENDS
                case (byte)'s': SetItem(); break;                             // SETITEM
                case (byte)'u': SetItems(PopMark()); break;                   // SETITEMS

                case (byte)'\x8f': Push(new List<object?>()); break;          // EMPTY_SET
                case (byte)'\x90': AppendMany(PopMark()); break;              // ADDITEMS
                case (byte)'\x91': PushTuple(PopMark()); break;               // FROZENSET

                // ---------------------------------------------------------- objects
                case (byte)'c': PushGlobal(ReadLine(), ReadLine()); break;    // GLOBAL
                case (byte)'\x93':                                            // STACK_GLOBAL
                {
                    var name = (string)Pop()!;
                    var module = (string)Pop()!;
                    PushGlobal(module, name);
                    break;
                }

                case (byte)'R': Reduce(); break;                              // REDUCE
                case (byte)'\x81': Reduce(); break;                           // NEWOBJ
                case (byte)'b': Build(); break;                               // BUILD

                case (byte)'Q': PersistentId(Pop()); break;                   // BINPERSID
                case (byte)'P': PersistentId(ReadLine()); break;              // PERSID

                default:
                    throw new NotSupportedException(
                        $"Pickle opcode 0x{op:x2} ('{(char)op}') at offset {_stream.Position - 1} is not supported. "
                        + "This reader covers what torch.save emits, not pickle in general.");
            }
        }
    }

    // ------------------------------------------------------------------ stack helpers

    private void Push(object? value) => _stack.Add(value);

    private object? Pop()
    {
        if (_stack.Count == 0) throw new InvalidDataException("Pickle stack underflow.");
        var value = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        return value;
    }

    private object? Peek() => _stack.Count == 0
        ? throw new InvalidDataException("Pickle stack underflow.")
        : _stack[^1];

    private void Memoize(int index)
    {
        while (_memo.Count <= index) _memo.Add(null);
        _memo[index] = Peek();
    }

    /// <summary>Number of items pushed since the most recent MARK.</summary>
    private int PopMark()
    {
        if (_marks.Count == 0) throw new InvalidDataException("Pickle MARK underflow.");
        var mark = _marks[^1];
        _marks.RemoveAt(_marks.Count - 1);
        return _stack.Count - mark;
    }

    private void PushTuple(int count)
    {
        var items = new List<object?>(count);
        for (var i = _stack.Count - count; i < _stack.Count; i++) items.Add(_stack[i]);
        _stack.RemoveRange(_stack.Count - count, count);
        Push(items);
    }

    private void PushDict(int count)
    {
        var dictionary = new Dictionary<object, object?>();
        for (var i = _stack.Count - count; i < _stack.Count; i += 2)
        {
            dictionary[_stack[i]!] = _stack[i + 1];
        }
        _stack.RemoveRange(_stack.Count - count, count);
        Push(dictionary);
    }

    private void Append()
    {
        var value = Pop();
        ((List<object?>)Peek()!).Add(value);
    }

    private void AppendMany(int count)
    {
        var target = _stack[_stack.Count - count - 1];
        var list = (List<object?>)target!;
        for (var i = _stack.Count - count; i < _stack.Count; i++) list.Add(_stack[i]);
        _stack.RemoveRange(_stack.Count - count, count);
    }

    private void SetItem()
    {
        var value = Pop();
        var key = Pop();
        ((Dictionary<object, object?>)Peek()!)[key!] = value;
    }

    private void SetItems(int count)
    {
        var target = (Dictionary<object, object?>)_stack[_stack.Count - count - 1]!;
        for (var i = _stack.Count - count; i < _stack.Count; i += 2)
        {
            target[_stack[i]!] = _stack[i + 1];
        }
        _stack.RemoveRange(_stack.Count - count, count);
    }

    private void PushGlobal(string module, string name)
    {
        var qualified = $"{module}.{name}";
        if (!AllowedGlobals.Contains(qualified))
        {
            throw new NotSupportedException(
                $"The checkpoint refers to '{qualified}', which this reader will not resolve. "
                + "Only the torch constructors needed to rebuild tensors are allowed; a file needing "
                + "anything else is running code, not storing weights. Prefer the .safetensors form.");
        }

        Push(new PickleGlobal(module, name));
    }

    private void Reduce()
    {
        var args = Pop();
        var callable = Pop();

        var arguments = args as List<object?> ?? (args is null ? [] : [args]);
        var global = callable as PickleGlobal;

        // A container constructor has to be honoured here rather than deferred. torch.save writes a
        // state dict as OrderedDict() followed by SETITEMS against the instance, so leaving it as an
        // uninterpreted reduce makes the very next opcode fail on a type that is not a dictionary.
        if (global?.Qualified == "collections.OrderedDict")
        {
            var dictionary = new Dictionary<object, object?>();

            // OrderedDict(items) is also legal, where items is a sequence of key/value pairs.
            if (arguments.Count > 0 && arguments[0] is List<object?> items)
            {
                foreach (var item in items)
                {
                    if (item is List<object?> { Count: 2 } pair) dictionary[pair[0]!] = pair[1];
                }
            }

            Push(dictionary);
            return;
        }

        Push(new PickleReduce(global, arguments));
    }

    /// <summary>Applies <c>BUILD</c>, which sets state on the object beneath it.</summary>
    /// <remarks>
    /// For a torch checkpoint the state is an <c>OrderedDict</c> being populated, so the state is
    /// merged into the target when both are dictionaries and otherwise attached to the reduce for
    /// the caller to inspect.
    /// </remarks>
    private void Build()
    {
        var state = Pop();
        var target = Peek();

        if (target is Dictionary<object, object?> targetDictionary && state is Dictionary<object, object?> source)
        {
            foreach (var pair in source) targetDictionary[pair.Key] = pair.Value;
            return;
        }

        if (target is PickleReduce reduce) reduce.State = state;
    }

    private void PersistentId(object? id)
    {
        if (_persistentLoad is null)
        {
            throw new NotSupportedException(
                "The pickle contains a persistent id but no resolver was supplied.");
        }

        var arguments = id as IReadOnlyList<object?> ?? [id];
        Push(_persistentLoad(arguments));
    }

    // ------------------------------------------------------------------ primitives

    private int ReadInt32()
    {
        Span<byte> buffer = stackalloc byte[4];
        _stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadInt32LittleEndian(buffer);
    }

    private ushort ReadUInt16()
    {
        Span<byte> buffer = stackalloc byte[2];
        _stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt16LittleEndian(buffer);
    }

    private ulong ReadUInt64()
    {
        Span<byte> buffer = stackalloc byte[8];
        _stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt64LittleEndian(buffer);
    }

    /// <summary>Reads a BINFLOAT, which pickle stores big-endian unlike everything else.</summary>
    private double ReadDoubleBigEndian()
    {
        Span<byte> buffer = stackalloc byte[8];
        _stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadDoubleBigEndian(buffer);
    }

    /// <summary>Reads a little-endian signed integer of arbitrary width (LONG1 / LONG4).</summary>
    private long ReadLong(int byteCount)
    {
        if (byteCount == 0) return 0;

        var bytes = new byte[byteCount];
        _stream.ReadExactly(bytes);

        long value = 0;
        for (var i = byteCount - 1; i >= 0; i--) value = (value << 8) | bytes[i];

        // Sign-extend from the stored width. Shapes and strides are positive, but a storage offset
        // in a rebuilt view can legitimately be negative.
        if ((bytes[^1] & 0x80) != 0 && byteCount < 8)
        {
            value -= 1L << (byteCount * 8);
        }

        return value;
    }

    private byte[] ReadBytes(int count)
    {
        var buffer = new byte[count];
        _stream.ReadExactly(buffer);
        return buffer;
    }

    private string ReadUtf8(int count) => Encoding.UTF8.GetString(ReadBytes(count));

    private string ReadLine()
    {
        var builder = new StringBuilder();
        int read;
        while ((read = _stream.ReadByte()) >= 0 && read != '\n')
        {
            builder.Append((char)read);
        }
        return builder.ToString().TrimEnd('\r');
    }

    private static long ParseLong(string text)
        => text.Length == 0 ? 0 : long.Parse(text, CultureInfo.InvariantCulture);

    private static string Unquote(string text)
    {
        var trimmed = text.Trim();
        if (trimmed.Length >= 2 && (trimmed[0] == '\'' || trimmed[0] == '"') && trimmed[^1] == trimmed[0])
        {
            return trimmed[1..^1];
        }
        return trimmed;
    }

    /// <summary>
    /// The only names <c>GLOBAL</c> will resolve. Everything a torch checkpoint legitimately needs
    /// to rebuild a tensor is here; nothing here executes anything.
    /// </summary>
    private static readonly HashSet<string> AllowedGlobals = new(StringComparer.Ordinal)
    {
        "collections.OrderedDict",
        "torch._utils._rebuild_tensor",
        "torch._utils._rebuild_tensor_v2",
        "torch._utils._rebuild_parameter",
        "torch._utils._rebuild_sparse_tensor",
        "torch.storage._load_from_bytes",
        "torch.FloatStorage",
        "torch.DoubleStorage",
        "torch.HalfStorage",
        "torch.BFloat16Storage",
        "torch.LongStorage",
        "torch.IntStorage",
        "torch.ShortStorage",
        "torch.CharStorage",
        "torch.ByteStorage",
        "torch.BoolStorage",
        "torch.Size",
        "torch.device",
        "numpy.core.multiarray._reconstruct",
        "numpy.ndarray",
        "numpy.dtype",
        "_codecs.encode",
    };
}

/// <summary>A resolved <c>GLOBAL</c>: a module-qualified Python name, not a callable.</summary>
/// <param name="Module">The Python module, for example <c>torch</c>.</param>
/// <param name="Name">The name within it, for example <c>FloatStorage</c>.</param>
public sealed record PickleGlobal(string Module, string Name)
{
    /// <summary>The dotted name, as it appeared in the pickle.</summary>
    public string Qualified => $"{Module}.{Name}";

    /// <inheritdoc />
    public override string ToString() => Qualified;
}

/// <summary>A <c>REDUCE</c>: a constructor the pickle asked to call, captured rather than invoked.</summary>
/// <param name="Callable">The named constructor, when it resolved to one.</param>
/// <param name="Arguments">The positional arguments it was given.</param>
public sealed record PickleReduce(PickleGlobal? Callable, IReadOnlyList<object?> Arguments)
{
    /// <summary>State applied by a following <c>BUILD</c>, when there was one.</summary>
    public object? State { get; set; }

    /// <inheritdoc />
    public override string ToString() => $"{Callable?.Qualified ?? "?"}({Arguments.Count} args)";
}
