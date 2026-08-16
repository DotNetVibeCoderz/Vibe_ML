using System.Buffers.Binary;
using System.Text;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>
/// A minimal FlatBuffers encoder, enough to write Arrow's IPC metadata.
/// </summary>
/// <remarks>
/// <para>
/// Arrow's metadata is FlatBuffers, not Protobuf, so the ONNX writer's machinery does not transfer.
/// The formats differ in a way that shapes everything here: Protobuf is written front to back with
/// length prefixes, while a FlatBuffer is built <b>back to front</b> so that every reference can be
/// a backward offset to something already written. Nothing is length-prefixed and nothing needs a
/// second pass.
/// </para>
/// <para>
/// The consequence is that children must be written before their parents. A table that references
/// a string has to have that string serialised first, which inverts the order code normally reads
/// in — hence the <c>Create*</c> helpers below take offsets rather than values.
/// </para>
/// <para>
/// The other thing to know is the <b>vtable</b>. A table does not store its fields at fixed
/// positions; it stores a signed offset to a vtable, which lists where each field lives (or zero
/// when absent). That is what makes the format both forward-compatible and compact — a default
/// value is simply not written. Vtables are deduplicated, because two tables of the same shape can
/// share one.
/// </para>
/// <para>
/// This is a writer for the subset Arrow needs, not a general implementation: no unions beyond a
/// type byte plus offset, no structs beyond inline fixed layouts, no nested buffers.
/// </para>
/// </remarks>
internal sealed class FlatBufferBuilder
{
    private byte[] _buffer;
    private int _space;

    private int _minAlign = 1;
    private int _vtableStart = -1;
    private int[] _vtable = [];
    private int _vtableSize;
    private int _objectStart;

    private readonly List<int> _vtables = [];

    /// <summary>Creates a builder with an initial capacity.</summary>
    public FlatBufferBuilder(int initialSize = 1024)
    {
        _buffer = new byte[Math.Max(initialSize, 32)];
        _space = _buffer.Length;
    }

    /// <summary>Bytes written so far.</summary>
    public int Offset => _buffer.Length - _space;

    /// <summary>The finished buffer, from the root offset onwards.</summary>
    public byte[] ToArray()
    {
        var result = new byte[Offset];
        Array.Copy(_buffer, _space, result, 0, Offset);
        return result;
    }

    // ---------------------------------------------------------------- primitives

    private void Grow(int needed)
    {
        if (_space >= needed) return;

        // Doubling from the front: the payload lives at the END of the array, so growth copies it
        // to the end of the new one rather than the start.
        var oldLength = _buffer.Length;
        var newLength = Math.Max(oldLength * 2, oldLength + needed);

        var grown = new byte[newLength];
        Array.Copy(_buffer, _space, grown, newLength - Offset, Offset);

        _space += newLength - oldLength;
        _buffer = grown;
    }

    /// <summary>Pads so that a field of <paramref name="size"/> bytes will land aligned.</summary>
    /// <remarks>
    /// The running maximum is what <see cref="Finish"/> aligns the whole buffer to. Without it a
    /// buffer containing 8-byte structs ends up 4-byte aligned, which this reader tolerates and
    /// Arrow's own verifier rejects outright.
    /// </remarks>
    private void Align(int size, int additional = 0)
    {
        if (size > _minAlign) _minAlign = size;

        var misalignment = (~(Offset + additional) + 1) & (size - 1);
        Grow(misalignment);
        _space -= misalignment;
        _buffer.AsSpan(_space, misalignment).Clear();
    }

    private void PutByte(byte value)
    {
        Grow(1);
        _buffer[--_space] = value;
    }

    private void PutShort(short value)
    {
        Grow(2);
        _space -= 2;
        BinaryPrimitives.WriteInt16LittleEndian(_buffer.AsSpan(_space), value);
    }

    private void PutInt(int value)
    {
        Grow(4);
        _space -= 4;
        BinaryPrimitives.WriteInt32LittleEndian(_buffer.AsSpan(_space), value);
    }

    private void PutLong(long value)
    {
        Grow(8);
        _space -= 8;
        BinaryPrimitives.WriteInt64LittleEndian(_buffer.AsSpan(_space), value);
    }

    // ---------------------------------------------------------------- tables

    /// <summary>Begins a table with room for <paramref name="fieldCount"/> fields.</summary>
    public void StartTable(int fieldCount)
    {
        Align(4);
        if (_vtable.Length < fieldCount) _vtable = new int[fieldCount];
        Array.Clear(_vtable, 0, fieldCount);

        _vtableSize = fieldCount;
        _objectStart = Offset;
        _vtableStart = 0;
    }

    private void Slot(int field) => _vtable[field] = Offset;

    /// <summary>Adds a byte field, skipping it when it equals the default.</summary>
    public void AddByte(int field, byte value, byte defaultValue)
    {
        if (value == defaultValue) return;
        Align(1);
        PutByte(value);
        Slot(field);
    }

    /// <summary>Adds a bool field.</summary>
    public void AddBool(int field, bool value, bool defaultValue)
        => AddByte(field, value ? (byte)1 : (byte)0, defaultValue ? (byte)1 : (byte)0);

    /// <summary>Adds a short field.</summary>
    public void AddShort(int field, short value, short defaultValue)
    {
        if (value == defaultValue) return;
        Align(2);
        PutShort(value);
        Slot(field);
    }

    /// <summary>Adds an int field.</summary>
    public void AddInt(int field, int value, int defaultValue)
    {
        if (value == defaultValue) return;
        Align(4);
        PutInt(value);
        Slot(field);
    }

    /// <summary>Adds a long field.</summary>
    public void AddLong(int field, long value, long defaultValue)
    {
        if (value == defaultValue) return;
        Align(8);
        PutLong(value);
        Slot(field);
    }

    /// <summary>Adds a reference to something already written.</summary>
    public void AddOffset(int field, int offset)
    {
        if (offset == 0) return;
        Align(4);
        // An offset is stored relative to its own position, which is what keeps a FlatBuffer
        // position-independent once written.
        PutInt(Offset - offset + 4);
        Slot(field);
    }

    /// <summary>
    /// Finishes the table, writing its vtable and reusing an identical one when possible.
    /// </summary>
    public int EndTable()
    {
        if (_vtableStart < 0) throw new InvalidOperationException("No table is being built.");

        // Align BEFORE the soffset placeholder, not just before the fields. A table begins with
        // that int32, so its start must be 4-aligned; a table whose last field was a short would
        // otherwise finish two bytes out. Every offset still resolves and every structural decode
        // still succeeds — Arrow's verifier is what rejects it, with nothing to say about why.
        Align(4);
        PutInt(0);                       // placeholder for the soffset to the vtable
        var tableEnd = Offset;

        // The vtable lists, per field, the offset from the table start - or zero when absent.
        for (var i = _vtableSize - 1; i >= 0; i--)
            PutShort(_vtable[i] != 0 ? (short)(tableEnd - _vtable[i]) : (short)0);

        PutShort((short)(tableEnd - _objectStart));          // table size
        PutShort((short)((_vtableSize + 2) * 2));            // vtable size

        // Deduplicate: two tables of the same shape can share one vtable.
        var start = Offset;
        var existing = 0;

        foreach (var candidate in _vtables)
        {
            var a = _buffer.Length - candidate;
            var b = _space;
            var length = BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(b));

            if (length != BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(a))) continue;
            if (!_buffer.AsSpan(a + 2, length - 2).SequenceEqual(_buffer.AsSpan(b + 2, length - 2))) continue;

            existing = candidate;
            break;
        }

        if (existing != 0)
        {
            _space = _buffer.Length - tableEnd;
            BinaryPrimitives.WriteInt32LittleEndian(
                _buffer.AsSpan(_space), existing - tableEnd);
        }
        else
        {
            _vtables.Add(start);
            BinaryPrimitives.WriteInt32LittleEndian(
                _buffer.AsSpan(_buffer.Length - tableEnd), start - tableEnd);
        }

        _vtableStart = -1;
        return tableEnd;
    }

    // ---------------------------------------------------------------- strings and vectors

    /// <summary>Writes a UTF-8 string and returns its offset.</summary>
    public int CreateString(string? value)
    {
        if (value is null) return 0;

        var bytes = Encoding.UTF8.GetBytes(value);
        PutByte(0);                                  // strings are null-terminated in FlatBuffers
        Align(4, bytes.Length + 4);

        Grow(bytes.Length);
        _space -= bytes.Length;
        bytes.CopyTo(_buffer.AsSpan(_space));

        PutInt(bytes.Length);
        return Offset;
    }

    /// <summary>Begins a vector of <paramref name="count"/> elements of the given width.</summary>
    public void StartVector(int elementSize, int count, int alignment)
    {
        Align(4, count * elementSize);
        Align(alignment, count * elementSize);
    }

    /// <summary>Finishes a vector, writing its length.</summary>
    public int EndVector(int count)
    {
        PutInt(count);
        return Offset;
    }

    /// <summary>Writes a vector of offsets, which must already have been written.</summary>
    public int CreateOffsetVector(IReadOnlyList<int> offsets)
    {
        StartVector(4, offsets.Count, 4);
        for (var i = offsets.Count - 1; i >= 0; i--)
        {
            Align(4);
            PutInt(Offset - offsets[i] + 4);
        }
        return EndVector(offsets.Count);
    }

    /// <summary>Writes a vector of longs.</summary>
    public int CreateLongVector(IReadOnlyList<long> values)
    {
        StartVector(8, values.Count, 8);
        for (var i = values.Count - 1; i >= 0; i--) PutLong(values[i]);
        return EndVector(values.Count);
    }

    /// <summary>
    /// Writes a vector of inline structs, each built by <paramref name="write"/> in reverse order.
    /// </summary>
    /// <remarks>
    /// Structs are laid out inline rather than by reference, so a vector of them is one contiguous
    /// block. They are written last-to-first like everything else in a FlatBuffer.
    /// </remarks>
    public int CreateStructVector(int count, int structSize, int alignment, Action<FlatBufferBuilder, int> write)
    {
        StartVector(structSize, count, alignment);
        for (var i = count - 1; i >= 0; i--) write(this, i);
        return EndVector(count);
    }

    /// <summary>Pads to <paramref name="alignment"/> before a struct of <paramref name="size"/> bytes.</summary>
    public void PrepareStruct(int size, int alignment)
    {
        Align(alignment, size);
    }

    /// <summary>Writes a struct field inline. Fields go in reverse declaration order.</summary>
    public void PutStructLong(long value) => PutLong(value);

    /// <summary>Writes a struct field inline.</summary>
    public void PutStructInt(int value) => PutInt(value);

    /// <summary>Writes padding inside a struct.</summary>
    public void PutStructPadding(int bytes)
    {
        Grow(bytes);
        _space -= bytes;
        _buffer.AsSpan(_space, bytes).Clear();
    }

    /// <summary>Finishes the buffer with <paramref name="root"/> as its root table.</summary>
    public void Finish(int root)
    {
        Align(_minAlign, 4);
        PutInt(Offset - root + 4);
    }
}

/// <summary>
/// A minimal FlatBuffers decoder, enough to read Arrow's IPC metadata.
/// </summary>
/// <remarks>
/// Reading is the mirror of writing and considerably simpler: a table stores a signed offset back
/// to its vtable, and the vtable says where each field sits relative to the table. A zero entry
/// means the field was never written and the reader should use the schema's default — which is why
/// every accessor here takes one.
/// </remarks>
internal readonly struct FlatBufferTable(byte[] buffer, int position)
{
    private readonly byte[] _buffer = buffer;

    /// <summary>Position of the table within the buffer.</summary>
    public int Position { get; } = position;

    /// <summary>True when this refers to an actual table rather than an absent one.</summary>
    public bool Exists => Position != 0;

    /// <summary>The root table of a finished buffer.</summary>
    public static FlatBufferTable Root(byte[] buffer, int start = 0)
        => new(buffer, start + BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(start)));

    /// <summary>Position of a field, or zero when it was not written.</summary>
    private int Field(int voffset)
    {
        var vtable = Position - BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(Position));
        var vtableSize = BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(vtable));

        if (voffset >= vtableSize) return 0;

        var offset = BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(vtable + voffset));
        return offset == 0 ? 0 : Position + offset;
    }

    /// <summary>Reads a byte field.</summary>
    public byte GetByte(int field, byte defaultValue = 0)
    {
        var at = Field(4 + field * 2);
        return at == 0 ? defaultValue : _buffer[at];
    }

    /// <summary>Reads a bool field.</summary>
    public bool GetBool(int field, bool defaultValue = false)
        => GetByte(field, defaultValue ? (byte)1 : (byte)0) != 0;

    /// <summary>Reads a short field.</summary>
    public short GetShort(int field, short defaultValue = 0)
    {
        var at = Field(4 + field * 2);
        return at == 0 ? defaultValue : BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(at));
    }

    /// <summary>Reads an int field.</summary>
    public int GetInt(int field, int defaultValue = 0)
    {
        var at = Field(4 + field * 2);
        return at == 0 ? defaultValue : BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at));
    }

    /// <summary>Reads a long field.</summary>
    public long GetLong(int field, long defaultValue = 0)
    {
        var at = Field(4 + field * 2);
        return at == 0 ? defaultValue : BinaryPrimitives.ReadInt64LittleEndian(_buffer.AsSpan(at));
    }

    /// <summary>Reads a nested table, which may not exist.</summary>
    public FlatBufferTable GetTable(int field)
    {
        var at = Field(4 + field * 2);
        return at == 0
            ? new FlatBufferTable(_buffer, 0)
            : new FlatBufferTable(_buffer, at + BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at)));
    }

    /// <summary>Reads a UTF-8 string field.</summary>
    public string? GetString(int field)
    {
        var at = Field(4 + field * 2);
        if (at == 0) return null;

        var start = at + BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at));
        var length = BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(start));
        return Encoding.UTF8.GetString(_buffer, start + 4, length);
    }

    /// <summary>Number of elements in a vector field.</summary>
    public int VectorLength(int field)
    {
        var at = Field(4 + field * 2);
        if (at == 0) return 0;

        var start = at + BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at));
        return BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(start));
    }

    /// <summary>Position of a vector's first element.</summary>
    private int VectorStart(int field)
    {
        var at = Field(4 + field * 2);
        if (at == 0) return 0;

        var start = at + BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at));
        return start + 4;
    }

    /// <summary>Reads the table at <paramref name="index"/> of a vector-of-tables field.</summary>
    public FlatBufferTable GetTableAt(int field, int index)
    {
        var start = VectorStart(field);
        if (start == 0) return new FlatBufferTable(_buffer, 0);

        var at = start + index * 4;
        return new FlatBufferTable(_buffer, at + BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(at)));
    }

    /// <summary>Position of the struct at <paramref name="index"/> of a vector-of-structs field.</summary>
    public int GetStructAt(int field, int index, int structSize)
    {
        var start = VectorStart(field);
        return start == 0 ? 0 : start + index * structSize;
    }

    /// <summary>Reads a long at an absolute position, for inline struct fields.</summary>
    public long ReadLongAt(int position) => BinaryPrimitives.ReadInt64LittleEndian(_buffer.AsSpan(position));

    /// <summary>Reads an int at an absolute position, for inline struct fields.</summary>
    public int ReadIntAt(int position) => BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(position));
}
