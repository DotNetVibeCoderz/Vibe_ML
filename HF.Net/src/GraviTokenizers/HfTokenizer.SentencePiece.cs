using System.Text.Json;

namespace Gravicode.HFNet.GraviTokenizers;

public sealed partial class HfTokenizer
{
    /// <summary>
    /// Builds an XLM-RoBERTa style tokenizer from a SentencePiece <c>sentencepiece.bpe.model</c>, the
    /// only tokenizer file some repositories ship (<c>microsoft/trocr-small-printed</c> among them).
    /// </summary>
    /// <param name="modelPath">The SentencePiece model, a serialized <c>ModelProto</c>.</param>
    /// <remarks>
    /// <para>
    /// The result is what transformers' <c>XLMRobertaTokenizer</c> converts the same file to: a
    /// Unigram model over the pieces and their scores, SentencePiece's own normalization table, a
    /// whitespace split with the <c>▁</c> marker, and <c>&lt;s&gt; $A &lt;/s&gt;</c> around every text.
    /// It is built as that <c>tokenizer.json</c> and read by the same loader, so the two cannot drift.
    /// </para>
    /// <para>
    /// The ids are fairseq's, not SentencePiece's. SentencePiece numbers <c>&lt;unk&gt;</c>,
    /// <c>&lt;s&gt;</c>, <c>&lt;/s&gt;</c> 0, 1, 2; fairseq puts <c>&lt;s&gt; &lt;pad&gt; &lt;/s&gt;
    /// &lt;unk&gt;</c> at 0-3, so every other piece sits one place higher than in the file, and
    /// <c>&lt;mask&gt;</c> goes after the last. A model trained with fairseq reads every token as its
    /// neighbour if the file's own numbering is used.
    /// </para>
    /// </remarks>
    public static HfTokenizer FromSentencePieceModel(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        var model = SentencePieceModel.Read(File.ReadAllBytes(modelPath));
        if (model.Pieces.Count < 3)
        {
            throw new InvalidDataException(
                $"'{modelPath}' holds {model.Pieces.Count} pieces; a SentencePiece model has at least <unk>, <s> and </s>.");
        }

        // fairseq's four specials, the file's pieces from its fourth on, then <mask>.
        var vocabulary = new List<(string Piece, double Score)> { ("<s>", 0), ("<pad>", 0), ("</s>", 0), ("<unk>", 0) };
        vocabulary.AddRange(model.Pieces.Skip(3).Select(p => (p.Piece, (double)p.Score)));
        vocabulary.Add(("<mask>", 0));
        var mask = vocabulary.Count - 1;

        using var stream = new MemoryStream();
        using (var json = new Utf8JsonWriter(stream))
        {
            json.WriteStartObject();

            json.WriteStartArray("added_tokens");
            foreach (var (id, content) in new[] { (0, "<s>"), (1, "<pad>"), (2, "</s>"), (3, "<unk>") })
            {
                WriteAddedToken(json, id, content, special: true, normalized: false, lstrip: false);
            }

            // Control and user-defined pieces past the first three are matched verbatim, as the
            // converter adds them; only control pieces count as special.
            for (var i = 3; i < model.Pieces.Count; i++)
            {
                var piece = model.Pieces[i];
                if (piece.Type is SentencePieceModel.Control or SentencePieceModel.UserDefined)
                {
                    WriteAddedToken(json, i + 1, piece.Piece, piece.Type == SentencePieceModel.Control, normalized: false, lstrip: false);
                }
            }

            WriteAddedToken(json, mask, "<mask>", special: true, normalized: true, lstrip: true);
            json.WriteEndArray();

            if (model.PrecompiledCharsmap is { Length: > 0 } charsmap)
            {
                json.WriteStartObject("normalizer");
                json.WriteString("type", "Precompiled");
                json.WriteString("precompiled_charsmap", Convert.ToBase64String(charsmap));
                json.WriteEndObject();
            }

            json.WriteStartObject("pre_tokenizer");
            json.WriteString("type", "Sequence");
            json.WriteStartArray("pretokenizers");
            json.WriteStartObject(); json.WriteString("type", "WhitespaceSplit"); json.WriteEndObject();
            WriteMetaspace(json);
            json.WriteEndArray();
            json.WriteEndObject();

            json.WriteStartObject("post_processor");
            json.WriteString("type", "TemplateProcessing");
            json.WriteStartArray("single");
            WriteSpecial(json, "<s>"); WriteSequence(json, "A"); WriteSpecial(json, "</s>");
            json.WriteEndArray();
            json.WriteStartArray("pair");
            WriteSpecial(json, "<s>"); WriteSequence(json, "A"); WriteSpecial(json, "</s>");
            WriteSpecial(json, "</s>"); WriteSequence(json, "B"); WriteSpecial(json, "</s>");
            json.WriteEndArray();
            json.WriteStartObject("special_tokens");
            foreach (var (token, id) in new[] { ("<s>", 0), ("</s>", 2) })
            {
                json.WriteStartObject(token);
                json.WriteString("id", token);
                json.WriteStartArray("ids"); json.WriteNumberValue(id); json.WriteEndArray();
                json.WriteStartArray("tokens"); json.WriteStringValue(token); json.WriteEndArray();
                json.WriteEndObject();
            }

            json.WriteEndObject();
            json.WriteEndObject();

            json.WritePropertyName("decoder");
            WriteMetaspace(json);

            json.WriteStartObject("model");
            json.WriteString("type", "Unigram");
            json.WriteNumber("unk_id", 3);
            json.WriteStartArray("vocab");
            foreach (var (piece, score) in vocabulary)
            {
                json.WriteStartArray();
                json.WriteStringValue(piece);
                json.WriteNumberValue(score);
                json.WriteEndArray();
            }

            json.WriteEndArray();
            json.WriteBoolean("byte_fallback", false);
            json.WriteEndObject();

            json.WriteEndObject();
        }

        return FromJson(System.Text.Encoding.UTF8.GetString(stream.ToArray()));

        static void WriteAddedToken(Utf8JsonWriter json, int id, string content, bool special, bool normalized, bool lstrip)
        {
            json.WriteStartObject();
            json.WriteNumber("id", id);
            json.WriteString("content", content);
            json.WriteBoolean("single_word", false);
            json.WriteBoolean("lstrip", lstrip);
            json.WriteBoolean("rstrip", false);
            json.WriteBoolean("normalized", normalized);
            json.WriteBoolean("special", special);
            json.WriteEndObject();
        }

        static void WriteMetaspace(Utf8JsonWriter json)
        {
            json.WriteStartObject();
            json.WriteString("type", "Metaspace");
            json.WriteString("replacement", "▁");
            json.WriteString("prepend_scheme", "always");
            json.WriteBoolean("split", true);
            json.WriteEndObject();
        }

        static void WriteSpecial(Utf8JsonWriter json, string id)
        {
            json.WriteStartObject();
            json.WriteStartObject("SpecialToken");
            json.WriteString("id", id);
            json.WriteNumber("type_id", 0);
            json.WriteEndObject();
            json.WriteEndObject();
        }

        static void WriteSequence(Utf8JsonWriter json, string id)
        {
            json.WriteStartObject();
            json.WriteStartObject("Sequence");
            json.WriteString("id", id);
            json.WriteNumber("type_id", 0);
            json.WriteEndObject();
            json.WriteEndObject();
        }
    }
}

/// <summary>The parts of a SentencePiece <c>ModelProto</c> a tokenizer needs: the pieces and the normalization table.</summary>
/// <remarks>
/// Read with a few lines of protobuf wire format rather than a protobuf library: the schema is
/// fixed, and only two of its fields matter here - <c>pieces</c> (1) and <c>normalizer_spec</c> (3),
/// whose <c>precompiled_charsmap</c> (2) is the table <c>Precompiled</c> reads.
/// </remarks>
internal sealed class SentencePieceModel
{
    internal const int Normal = 1;
    internal const int Unknown = 2;
    internal const int Control = 3;
    internal const int UserDefined = 4;

    internal List<(string Piece, float Score, int Type)> Pieces { get; } = [];

    internal byte[]? PrecompiledCharsmap { get; private set; }

    internal static SentencePieceModel Read(byte[] data)
    {
        var model = new SentencePieceModel();
        var reader = new ProtoReader(data, 0, data.Length);

        while (reader.Next(out var field, out var wire))
        {
            if (field == 1 && wire == 2)
            {
                model.Pieces.Add(ReadPiece(reader.Message()));
            }
            else if (field == 3 && wire == 2)
            {
                var spec = reader.Message();
                while (spec.Next(out var specField, out var specWire))
                {
                    if (specField == 2 && specWire == 2) model.PrecompiledCharsmap = spec.Bytes();
                    else spec.Skip(specWire);
                }
            }
            else
            {
                reader.Skip(wire);
            }
        }

        return model;
    }

    private static (string, float, int) ReadPiece(ProtoReader reader)
    {
        string piece = "";
        float score = 0;
        var type = Normal;   // the schema's default; NORMAL pieces usually leave the field out

        while (reader.Next(out var field, out var wire))
        {
            switch (field, wire)
            {
                case (1, 2): piece = System.Text.Encoding.UTF8.GetString(reader.Bytes()); break;
                case (2, 5): score = reader.Float(); break;
                case (3, 0): type = (int)reader.Varint(); break;
                default: reader.Skip(wire); break;
            }
        }

        return (piece, score, type);
    }

    /// <summary>A cursor over one protobuf message.</summary>
    private sealed class ProtoReader(byte[] data, int start, int end)
    {
        private int _position = start;

        internal bool Next(out int field, out int wire)
        {
            if (_position >= end)
            {
                field = wire = 0;
                return false;
            }

            var key = Varint();
            field = (int)(key >> 3);
            wire = (int)(key & 7);
            return true;
        }

        internal ulong Varint()
        {
            ulong value = 0;
            for (var shift = 0; shift < 64; shift += 7)
            {
                if (_position >= end) throw new InvalidDataException("A SentencePiece model ends inside a number.");

                var b = data[_position++];
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;
            }

            throw new InvalidDataException("A SentencePiece model holds a number longer than 64 bits.");
        }

        internal float Float()
        {
            Require(4);
            var value = BitConverter.ToSingle(data, _position);
            _position += 4;
            return value;
        }

        internal byte[] Bytes()
        {
            var length = Length();
            var bytes = data.AsSpan(_position, length).ToArray();
            _position += length;
            return bytes;
        }

        internal ProtoReader Message()
        {
            var length = Length();
            var message = new ProtoReader(data, _position, _position + length);
            _position += length;
            return message;
        }

        internal void Skip(int wire)
        {
            switch (wire)
            {
                case 0: Varint(); break;
                case 1: Require(8); _position += 8; break;
                case 2:
                {
                    var length = Length();
                    _position += length;
                    break;
                }

                case 5: Require(4); _position += 4; break;
                default: throw new InvalidDataException($"A SentencePiece model uses protobuf wire type {wire}, which it never writes.");
            }
        }

        // Callers must store the result before adding it: `_position += Length()` reads _position
        // first, before the length's own bytes are consumed, and desynchronises the stream.
        private int Length()
        {
            var length = (int)Varint();
            Require(length);
            return length;
        }

        private void Require(int count)
        {
            if (count < 0 || _position + count > end)
            {
                throw new InvalidDataException("A SentencePiece model is truncated: a field runs past the end of its message.");
            }
        }
    }
}
