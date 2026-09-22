using Gravicode.HFNet.GraviTokenizers;

// GraviTokenizers sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
Console.WriteLine("=== GraviTokenizers ===\n");

// --- 1. WordPiece, checked against the reference tokenization -----------------
var bert = HfTokenizer.FromPretrained("bert-base-uncased");
var text = "Hello, world! Tokenizers are unbelievable.";
var encoded = bert.Encode(text);

Console.WriteLine($"bert-base-uncased  vocab={bert.VocabularySize}  pad={bert.PadId} cls={bert.ClassifierId} sep={bert.SeparatorId}");
Console.WriteLine($"  tokens : {string.Join(' ', encoded.Tokens)}");
Console.WriteLine($"  ids    : {string.Join(' ', encoded.Ids)}");
Console.WriteLine($"  decoded: {bert.Decode(encoded.Ids)}");
Console.WriteLine();

// Offsets have to point back into the untouched input.
Console.WriteLine("  offsets -> source spans:");
for (var i = 0; i < encoded.Length && i < 8; i++)
{
    var span = encoded.Span(text, i);
    Console.WriteLine($"    {encoded.Tokens[i],-16} {encoded.Offsets[i],-10} '{span}'");
}
Console.WriteLine();

// --- 2. Byte-level BPE --------------------------------------------------------
var gpt2 = HfTokenizer.FromPretrained("gpt2");
var g = gpt2.Encode("Hello world");
Console.WriteLine($"gpt2  vocab={gpt2.VocabularySize}");
Console.WriteLine($"  tokens : {string.Join(' ', g.Tokens)}");
Console.WriteLine($"  ids    : {string.Join(' ', g.Ids)}");
Console.WriteLine($"  decoded: '{gpt2.Decode(g.Ids)}'");
Console.WriteLine();

// --- 3. Pairs and batches -----------------------------------------------------
var pair = bert.Encode("What is HF.Net?", "A Hugging Face style library for .NET.");
Console.WriteLine($"pair  tokens={pair.Length}  typeIds={string.Join("", pair.TypeIds)}");

var batch = bert.EncodeBatch(["short one", "a rather longer sentence to force padding"], BatchOptions.Default);
Console.WriteLine($"batch {batch}  mask row1={string.Join("", batch.Encodings[0].AttentionMask)}");
Console.WriteLine();

// --- 4. Training from a corpus ------------------------------------------------
string[] corpus =
[
    "mesin belajar dari data", "data melatih mesin", "belajar mesin itu menarik",
    "model belajar dari contoh", "contoh data melatih model",
];
var trained = Tokenizer.TrainWordPiece(corpus, vocabularySize: 120);
Console.WriteLine($"trained WordPiece vocab={trained.VocabularySize}");
Console.WriteLine($"  'mesin belajar' -> {string.Join(' ', trained.Encode("mesin belajar").Tokens)}");
