using System.Buffers.Binary;
using System.Text;

namespace Gravicode.Science.GraviFrame.Io;

/// <summary>
/// Reads and writes the Arrow IPC file format, for exchange with pandas, Polars and Spark.
/// </summary>
/// <remarks>
/// <para>
/// Arrow is a memory layout before it is a file format, and that is the point: a column is a
/// validity bitmap plus a values buffer, contiguous and untagged, so a reader can point at it
/// rather than parse it. <see cref="DataFrame"/> is already columnar, so what this adds is the
/// buffer conventions and the metadata that describes them.
/// </para>
/// <para>
/// Three things about the layout are worth knowing before reading the code:
/// </para>
/// <list type="bullet">
/// <item><b>Validity is a bitmap, not a sentinel.</b> One bit per value, least-significant bit
/// first, where 1 means present. That is why <c>NaN</c> cannot survive a round trip as a *value*
/// here — in a `NumericSeries` NaN means missing, so it becomes a cleared validity bit and the
/// slot's numeric content is irrelevant.</item>
/// <item><b>Strings are offsets plus one blob.</b> A UTF-8 column is an <c>int32</c> offsets buffer
/// of <c>n + 1</c> entries and a single data buffer. The offsets are cumulative, so the length of
/// row <c>i</c> is <c>offsets[i + 1] - offsets[i]</c> and no per-value length is stored.</item>
/// <item><b>Every buffer is padded to 8 bytes</b> (Arrow recommends 64). Getting this wrong
/// produces a file that this reader accepts and pyarrow rejects, because the offsets it computes
/// no longer land where the metadata says.</item>
/// </list>
/// <para>
/// The metadata is FlatBuffers — see <see cref="FlatBufferBuilder"/>. The field numbers below are
/// from Arrow's <c>Schema.fbs</c> and <c>Message.fbs</c>; they are positional and cannot be
/// inferred, so each block names the table it is writing.
/// </para>
/// <para>
/// <b>Scope.</b> One record batch per file, no dictionary encoding on the wire, no compression, no
/// nested types. A `CategoricalSeries` is written as its expanded strings rather than as an Arrow
/// dictionary array: dictionary batches are a separate message type with their own IPC bookkeeping,
/// and writing one badly produces a file that loads with silently wrong values.
/// </para>
/// </remarks>
public static class ArrowFile
{
    private static readonly byte[] Magic = "ARROW1"u8.ToArray();

    // Message header union tags, from Message.fbs.
    private const byte HeaderSchema = 1;
    private const byte HeaderRecordBatch = 3;

    // Type union tags, from Schema.fbs.
    private const byte TypeInt = 2;
    private const byte TypeFloatingPoint = 3;
    private const byte TypeUtf8 = 5;
    private const byte TypeBool = 6;
    private const byte TypeTimestamp = 10;

    /// <summary>Writes a frame as an Arrow IPC file.</summary>
    public static void Write(DataFrame frame, string path)
    {
        using var stream = File.Create(path);
        Write(frame, stream);
    }

    /// <summary>
    /// Writes a frame as an Arrow IPC file to a stream.
    /// </summary>
    /// <remarks>
    /// The file is magic, a schema message, one record batch, a footer, the footer's length and the
    /// magic again. The trailing length is what makes the format seekable: a reader jumps to the
    /// end, reads backwards to find the footer, and from there knows where every batch begins
    /// without scanning.
    /// </remarks>
    public static void Write(DataFrame frame, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(stream);

        var columns = frame.ColumnNames.Select(n => frame[n]).ToArray();

        stream.Write(Magic);
        Pad(stream, 8);                       // the body must start aligned

        var schemaOffset = stream.Position;
        WriteMessage(stream, BuildSchema(columns), body: null);
        var schemaLength = (int)(stream.Position - schemaOffset);

        var batchOffset = stream.Position;
        var (metadata, body) = BuildRecordBatch(columns, frame.RowCount);
        WriteMessage(stream, metadata, body);
        var metadataLength = (int)(stream.Position - batchOffset) - body.Length;

        var footer = BuildFooter(columns, batchOffset, metadataLength, body.Length);
        stream.Write(footer);

        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(length, footer.Length);
        stream.Write(length);
        stream.Write(Magic);
    }

    /// <summary>Reads an Arrow IPC file.</summary>
    public static DataFrame Read(string path)
    {
        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    /// <summary>
    /// Reads an Arrow IPC file from a seekable stream.
    /// </summary>
    /// <remarks>
    /// Reads the footer first, which is what the format is designed for: it names every batch's
    /// position, so nothing has to be scanned. Only the first batch is materialised here, since the
    /// writer produces one.
    /// </remarks>
    public static DataFrame Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanSeek) throw new ArgumentException("Reading Arrow needs a seekable stream.", nameof(stream));

        var all = ReadAll(stream);

        if (all.Length < 12 || !all.AsSpan(0, 6).SequenceEqual(Magic)
                            || !all.AsSpan(all.Length - 6).SequenceEqual(Magic))
            throw new InvalidDataException("This is not an Arrow IPC file: the magic marker is missing.");

        var footerLength = BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(all.Length - 10));
        var footerStart = all.Length - 10 - footerLength;

        var footer = FlatBufferTable.Root(all, footerStart);
        var schema = footer.GetTable(1);                       // Footer.schema
        if (!schema.Exists) throw new InvalidDataException("The Arrow footer carries no schema.");

        var fields = ReadSchema(schema);

        if (footer.VectorLength(3) == 0)                       // Footer.recordBatches
            return new DataFrame([.. fields.Select(f => Empty(f.Name, f.Kind))]);

        // Block is an inline struct: offset:long, metaDataLength:int, padding:int, bodyLength:long.
        var block = footer.GetStructAt(3, 0, 24);
        var batchOffset = footer.ReadLongAt(block);
        var metadataLength = footer.ReadIntAt(block + 8);

        // The message flatbuffer starts after the 8-byte continuation-and-length prefix.
        var message = FlatBufferTable.Root(all, (int)batchOffset + 8);

        // Message.header is field 2, not 1: a FlatBuffers union generates BOTH a tag field and an
        // offset field, so header_type takes slot 1 and everything after it shifts by one.
        var batch = message.GetTable(2);
        var bodyStart = (int)batchOffset + metadataLength;

        return ReadRecordBatch(all, fields, batch, bodyStart);
    }

    // ---------------------------------------------------------------- schema

    private enum ArrowKind { Double, Integer, Utf8, Bool, Timestamp }

    /// <summary>
    /// A column's name, its logical kind, and whatever else its Arrow type carries.
    /// </summary>
    /// <param name="Name">The column's name, as it appears in the schema.</param>
    /// <param name="Kind">The logical type the column was mapped to.</param>
    /// <param name="Unit">Timestamp unit: 0 second, 1 milli, 2 micro, 3 nano.</param>
    /// <param name="BitWidth">Integer width in bits, when the source column was an integer.</param>
    /// <param name="Signed">Whether that integer was signed.</param>
    /// <remarks>
    /// These travel with the field rather than in a lookup on the side. Timestamp unit and integer
    /// width are both per-field, and both are silent failures when guessed: nanoseconds read as
    /// microseconds put every date a thousandfold too close to the epoch, and an int64 read as a
    /// double comes back as a denormal indistinguishable from zero.
    /// </remarks>
    private readonly record struct FieldInfo(
        string Name, ArrowKind Kind, short Unit = 2, int BitWidth = 64, bool Signed = true);

    private static ArrowKind KindOf(Series column) => column switch
    {
        NumericSeries => ArrowKind.Double,
        BooleanSeries => ArrowKind.Bool,
        DateTimeSeries => ArrowKind.Timestamp,
        _ => ArrowKind.Utf8,                     // text and categorical both go out as strings
    };

    /// <summary>Builds the Schema message body.</summary>
    private static byte[] BuildSchema(Series[] columns)
    {
        var builder = new FlatBufferBuilder(1024);

        var fieldOffsets = new int[columns.Length];
        for (var i = 0; i < columns.Length; i++)
            fieldOffsets[i] = BuildField(builder, columns[i].Name, KindOf(columns[i]));

        var fieldsVector = builder.CreateOffsetVector(fieldOffsets);

        builder.StartTable(4);                   // Schema
        builder.AddShort(0, 0, 0);               // endianness: Little
        builder.AddOffset(1, fieldsVector);
        var schema = builder.EndTable();

        builder.StartTable(5);                   // Message
        builder.AddShort(0, 4, 0);               // version: V5
        builder.AddByte(1, HeaderSchema, 0);     // header_type
        builder.AddOffset(2, schema);            // header
        builder.AddLong(3, 0, 0);                // bodyLength
        var message = builder.EndTable();

        builder.Finish(message);
        return builder.ToArray();
    }

    /// <summary>Builds one Field table, with its type as a union.</summary>
    private static int BuildField(FlatBufferBuilder builder, string name, ArrowKind kind)
    {
        // Children must be written first: the name string and the type table both precede the
        // Field table that references them.
        var nameOffset = builder.CreateString(name);

        int typeOffset;
        byte typeTag;

        switch (kind)
        {
            case ArrowKind.Double:
                builder.StartTable(1);                 // FloatingPoint
                builder.AddShort(0, 2, 0);             // precision: DOUBLE
                typeOffset = builder.EndTable();
                typeTag = TypeFloatingPoint;
                break;

            case ArrowKind.Bool:
                builder.StartTable(0);                 // Bool
                typeOffset = builder.EndTable();
                typeTag = TypeBool;
                break;

            case ArrowKind.Timestamp:
                builder.StartTable(2);                 // Timestamp
                builder.AddShort(0, 2, 0);             // unit: MICROSECOND
                typeOffset = builder.EndTable();       // timezone left null: naive local time
                typeTag = TypeTimestamp;
                break;

            default:
                builder.StartTable(0);                 // Utf8
                typeOffset = builder.EndTable();
                typeTag = TypeUtf8;
                break;
        }

        // An EMPTY children vector, not an absent one. FlatBuffers treats the two as equivalent
        // and Arrow's C++ reader does not: it walks `field->children()` without a null check, so a
        // Field that omits the vector fails verification with nothing to say about why.
        var children = builder.CreateOffsetVector([]);

        builder.StartTable(7);                         // Field
        builder.AddOffset(0, nameOffset);
        builder.AddBool(1, true, false);               // nullable
        builder.AddByte(2, typeTag, 0);                // type_type
        builder.AddOffset(3, typeOffset);              // type
        builder.AddOffset(5, children);
        return builder.EndTable();
    }

    /// <summary>Reads the field list out of a Schema table.</summary>
    private static FieldInfo[] ReadSchema(FlatBufferTable schema)
    {
        var count = schema.VectorLength(1);
        var fields = new FieldInfo[count];

        for (var i = 0; i < count; i++)
        {
            var field = schema.GetTableAt(1, i);
            var name = field.GetString(0) ?? $"column_{i}";
            var tag = field.GetByte(2);
            var type = field.GetTable(3);

            var kind = tag switch
            {
                TypeFloatingPoint => ArrowKind.Double,
                TypeInt => ArrowKind.Integer,          // integers widen into the numeric column
                TypeBool => ArrowKind.Bool,
                TypeTimestamp => ArrowKind.Timestamp,
                _ => ArrowKind.Utf8,
            };

            fields[i] = kind switch
            {
                // Timestamp.unit is field 0; Int carries bitWidth at 0 and is_signed at 1.
                ArrowKind.Timestamp => new FieldInfo(name, kind, Unit: type.GetShort(0, 0)),
                ArrowKind.Integer => new FieldInfo(name, kind,
                    BitWidth: type.GetInt(0, 32), Signed: type.GetBool(1)),
                _ => new FieldInfo(name, kind),
            };
        }

        return fields;
    }

    // ---------------------------------------------------------------- record batch

    /// <summary>Builds the RecordBatch metadata and its body.</summary>
    private static (byte[] Metadata, byte[] Body) BuildRecordBatch(Series[] columns, int rowCount)
    {
        var body = new MemoryStream();
        var buffers = new List<(long Offset, long Length)>();
        var nullCounts = new long[columns.Length];

        for (var i = 0; i < columns.Length; i++)
            nullCounts[i] = AppendColumn(body, buffers, columns[i], rowCount);

        var builder = new FlatBufferBuilder(1024);

        // FieldNode is an inline struct of (length, null_count), written in reverse.
        var nodes = builder.CreateStructVector(columns.Length, 16, 8, (b, i) =>
        {
            b.PrepareStruct(16, 8);
            b.PutStructLong(nullCounts[i]);
            b.PutStructLong(rowCount);
        });

        // Buffer is an inline struct of (offset, length), also reversed.
        var bufferVector = builder.CreateStructVector(buffers.Count, 16, 8, (b, i) =>
        {
            b.PrepareStruct(16, 8);
            b.PutStructLong(buffers[i].Length);
            b.PutStructLong(buffers[i].Offset);
        });

        builder.StartTable(5);                         // RecordBatch
        builder.AddLong(0, rowCount, 0);
        builder.AddOffset(1, nodes);
        builder.AddOffset(2, bufferVector);
        var batch = builder.EndTable();

        builder.StartTable(5);                         // Message
        builder.AddShort(0, 4, 0);                     // version: V5
        builder.AddByte(1, HeaderRecordBatch, 0);
        builder.AddOffset(2, batch);
        builder.AddLong(3, body.Length, 0);
        var message = builder.EndTable();

        builder.Finish(message);
        return (builder.ToArray(), body.ToArray());
    }

    /// <summary>Appends one column's buffers to the body, returning its null count.</summary>
    private static long AppendColumn(MemoryStream body, List<(long, long)> buffers, Series column, int rowCount)
    {
        // Validity first, for every type: one bit per row, LSB first, 1 = present.
        var validity = new byte[(rowCount + 7) / 8];
        long nulls = 0;

        for (var i = 0; i < rowCount; i++)
        {
            if (column.IsMissing(i)) { nulls++; continue; }
            validity[i / 8] |= (byte)(1 << (i % 8));
        }

        AppendBuffer(body, buffers, validity);

        switch (column)
        {
            case NumericSeries numeric:
            {
                var values = new byte[rowCount * 8];
                for (var i = 0; i < rowCount; i++)
                {
                    // A missing slot is already flagged in the bitmap; writing zero rather than NaN
                    // keeps the file readable by consumers that ignore the payload of null slots.
                    var value = numeric.IsMissing(i) ? 0.0 : numeric[i];
                    BinaryPrimitives.WriteDoubleLittleEndian(values.AsSpan(i * 8), value);
                }
                AppendBuffer(body, buffers, values);
                break;
            }

            case BooleanSeries boolean:
            {
                var bits = new byte[(rowCount + 7) / 8];
                for (var i = 0; i < rowCount; i++)
                    if (boolean[i] == true) bits[i / 8] |= (byte)(1 << (i % 8));
                AppendBuffer(body, buffers, bits);
                break;
            }

            case DateTimeSeries dates:
            {
                var values = new byte[rowCount * 8];
                for (var i = 0; i < rowCount; i++)
                {
                    var ticks = dates[i]?.Ticks ?? 0;
                    // Microseconds since the Unix epoch, which is the unit declared in the schema.
                    var micros = ticks == 0 ? 0 : (ticks - DateTime.UnixEpoch.Ticks) / 10;
                    BinaryPrimitives.WriteInt64LittleEndian(values.AsSpan(i * 8), micros);
                }
                AppendBuffer(body, buffers, values);
                break;
            }

            default:
            {
                // Utf8: cumulative int32 offsets of n + 1 entries, then one blob.
                var offsets = new byte[(rowCount + 1) * 4];
                var data = new MemoryStream();
                var running = 0;

                for (var i = 0; i < rowCount; i++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(offsets.AsSpan(i * 4), running);

                    var text = column.GetValue(i)?.ToString();
                    if (text is null) continue;

                    var bytes = Encoding.UTF8.GetBytes(text);
                    data.Write(bytes);
                    running += bytes.Length;
                }

                BinaryPrimitives.WriteInt32LittleEndian(offsets.AsSpan(rowCount * 4), running);

                AppendBuffer(body, buffers, offsets);
                AppendBuffer(body, buffers, data.ToArray());
                break;
            }
        }

        return nulls;
    }

    /// <summary>Writes one buffer, records its extent, and pads to the 8-byte boundary.</summary>
    private static void AppendBuffer(MemoryStream body, List<(long, long)> buffers, byte[] bytes)
    {
        var offset = body.Position;
        body.Write(bytes);
        buffers.Add((offset, bytes.Length));

        // Padding is not optional: the next buffer's recorded offset assumes it.
        var padding = (8 - (int)(body.Position % 8)) % 8;
        for (var i = 0; i < padding; i++) body.WriteByte(0);
    }

    /// <summary>Rebuilds the columns from a record batch's buffers.</summary>
    private static DataFrame ReadRecordBatch(byte[] all, FieldInfo[] fields, FlatBufferTable batch, int bodyStart)
    {
        var rowCount = (int)batch.GetLong(0);
        var columns = new List<Series>(fields.Length);
        var buffer = 0;

        for (var f = 0; f < fields.Length; f++)
        {
            var (validityOffset, validityLength) = BufferAt(batch, buffer++);
            var validity = validityLength > 0 ? bodyStart + (int)validityOffset : -1;

            bool Present(int row) =>
                validity < 0 || (all[validity + row / 8] & (1 << (row % 8))) != 0;

            switch (fields[f].Kind)
            {
                case ArrowKind.Double:
                {
                    var (offset, _) = BufferAt(batch, buffer++);
                    var start = bodyStart + (int)offset;
                    var values = new double[rowCount];

                    for (var i = 0; i < rowCount; i++)
                        values[i] = Present(i)
                            ? BinaryPrimitives.ReadDoubleLittleEndian(all.AsSpan(start + i * 8))
                            : double.NaN;

                    columns.Add(new NumericSeries(fields[f].Name, values));
                    break;
                }

                case ArrowKind.Integer:
                {
                    // Widen into the numeric column, honouring the declared width and sign.
                    // Reinterpreting the bytes as a double instead gives denormals that all print
                    // as zero, which looks like a column of missing data rather than a bug.
                    var (offset, _) = BufferAt(batch, buffer++);
                    var start = bodyStart + (int)offset;
                    var width = fields[f].BitWidth / 8;
                    var signed = fields[f].Signed;
                    var values = new double[rowCount];

                    for (var i = 0; i < rowCount; i++)
                    {
                        if (!Present(i)) { values[i] = double.NaN; continue; }

                        var at = all.AsSpan(start + i * width);
                        values[i] = width switch
                        {
                            1 => signed ? (sbyte)at[0] : at[0],
                            2 => signed ? BinaryPrimitives.ReadInt16LittleEndian(at)
                                        : BinaryPrimitives.ReadUInt16LittleEndian(at),
                            4 => signed ? BinaryPrimitives.ReadInt32LittleEndian(at)
                                        : BinaryPrimitives.ReadUInt32LittleEndian(at),
                            _ => signed ? BinaryPrimitives.ReadInt64LittleEndian(at)
                                        : BinaryPrimitives.ReadUInt64LittleEndian(at),
                        };
                    }

                    columns.Add(new NumericSeries(fields[f].Name, values));
                    break;
                }

                case ArrowKind.Bool:
                {
                    var (offset, _) = BufferAt(batch, buffer++);
                    var start = bodyStart + (int)offset;
                    var values = new bool?[rowCount];

                    for (var i = 0; i < rowCount; i++)
                        values[i] = Present(i) ? (all[start + i / 8] & (1 << (i % 8))) != 0 : null;

                    columns.Add(new BooleanSeries(fields[f].Name, values));
                    break;
                }

                case ArrowKind.Timestamp:
                {
                    var (offset, _) = BufferAt(batch, buffer++);
                    var start = bodyStart + (int)offset;
                    var values = new DateTime?[rowCount];

                    // The unit is per-field. Reading nanoseconds as microseconds would put every
                    // date a thousand times too close to the epoch, and look plausible.
                    var unit = fields[f].Unit;
                    var perUnit = unit switch
                    {
                        0 => TimeSpan.TicksPerSecond,
                        1 => TimeSpan.TicksPerMillisecond,
                        2 => 10L,                              // microseconds
                        _ => 1L,                               // nanoseconds round to 100ns ticks
                    };

                    for (var i = 0; i < rowCount; i++)
                    {
                        if (!Present(i)) { values[i] = null; continue; }

                        var raw = BinaryPrimitives.ReadInt64LittleEndian(all.AsSpan(start + i * 8));
                        var ticks = unit == 3 ? raw / 100 : raw * perUnit;
                        values[i] = DateTime.UnixEpoch.AddTicks(ticks);
                    }

                    columns.Add(new DateTimeSeries(fields[f].Name, values));
                    break;
                }

                default:
                {
                    var (offsetsOffset, _) = BufferAt(batch, buffer++);
                    var (dataOffset, _) = BufferAt(batch, buffer++);

                    var offsets = bodyStart + (int)offsetsOffset;
                    var data = bodyStart + (int)dataOffset;
                    var values = new string?[rowCount];

                    for (var i = 0; i < rowCount; i++)
                    {
                        if (!Present(i)) { values[i] = null; continue; }

                        var from = BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(offsets + i * 4));
                        var to = BinaryPrimitives.ReadInt32LittleEndian(all.AsSpan(offsets + (i + 1) * 4));
                        values[i] = Encoding.UTF8.GetString(all, data + from, to - from);
                    }

                    columns.Add(new TextSeries(fields[f].Name, values));
                    break;
                }
            }
        }

        return new DataFrame(columns);
    }

    /// <summary>Reads the (offset, length) of one buffer from the batch's buffer vector.</summary>
    private static (long Offset, long Length) BufferAt(FlatBufferTable batch, int index)
    {
        var at = batch.GetStructAt(2, index, 16);
        return (batch.ReadLongAt(at), batch.ReadLongAt(at + 8));
    }

    // ---------------------------------------------------------------- footer

    private static byte[] BuildFooter(Series[] columns, long batchOffset, int metadataLength, long bodyLength)
    {
        var builder = new FlatBufferBuilder(1024);

        var fieldOffsets = new int[columns.Length];
        for (var i = 0; i < columns.Length; i++)
            fieldOffsets[i] = BuildField(builder, columns[i].Name, KindOf(columns[i]));

        var fieldsVector = builder.CreateOffsetVector(fieldOffsets);

        builder.StartTable(4);                   // Schema
        builder.AddShort(0, 0, 0);               // endianness: Little
        builder.AddOffset(1, fieldsVector);
        var schema = builder.EndTable();

        // Block: offset:long, metaDataLength:int, padding:int, bodyLength:long.
        var blocks = builder.CreateStructVector(1, 24, 8, (b, _) =>
        {
            b.PrepareStruct(24, 8);
            b.PutStructLong(bodyLength);
            b.PutStructPadding(4);
            b.PutStructInt(metadataLength);
            b.PutStructLong(batchOffset);
        });

        // Same again: an empty dictionaries vector rather than an absent one.
        var dictionaries = builder.CreateOffsetVector([]);

        builder.StartTable(5);                   // Footer
        builder.AddShort(0, 4, 0);               // version: V5
        builder.AddOffset(1, schema);
        builder.AddOffset(2, dictionaries);
        builder.AddOffset(3, blocks);            // recordBatches
        var footer = builder.EndTable();

        builder.Finish(footer);
        return builder.ToArray();
    }

    // ---------------------------------------------------------------- plumbing

    /// <summary>
    /// Writes one IPC message: a continuation marker, the metadata length, the metadata and the body.
    /// </summary>
    /// <remarks>
    /// The leading <c>0xFFFFFFFF</c> distinguishes the current encapsulated format from the
    /// pre-0.15 one, where the length came first. Omitting it produces a file that older readers
    /// accept and current ones reject.
    /// </remarks>
    private static void WriteMessage(Stream stream, byte[] metadata, byte[]? body)
    {
        var padded = (metadata.Length + 7) / 8 * 8;

        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(prefix, 0xFFFFFFFF);
        BinaryPrimitives.WriteInt32LittleEndian(prefix[4..], padded);
        stream.Write(prefix);

        stream.Write(metadata);
        for (var i = metadata.Length; i < padded; i++) stream.WriteByte(0);

        if (body is not null) stream.Write(body);
    }

    private static void Pad(Stream stream, int alignment)
    {
        var padding = (alignment - (int)(stream.Position % alignment)) % alignment;
        for (var i = 0; i < padding; i++) stream.WriteByte(0);
    }

    private static byte[] ReadAll(Stream stream)
    {
        if (stream is MemoryStream memory) return memory.ToArray();

        stream.Seek(0, SeekOrigin.Begin);
        using var copy = new MemoryStream();
        stream.CopyTo(copy);
        return copy.ToArray();
    }

    private static Series Empty(string name, ArrowKind kind) => kind switch
    {
        ArrowKind.Double or ArrowKind.Integer => new NumericSeries(name, []),
        ArrowKind.Bool => new BooleanSeries(name, new bool?[0]),
        ArrowKind.Timestamp => new DateTimeSeries(name, new DateTime?[0]),
        _ => new TextSeries(name, []),
    };
}
