using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviTokenizers;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>
/// GraviTokenizers: throughput over a multilingual corpus, one thread against all of them.
/// </summary>
/// <remarks>
/// The blueprint asks for CPU against GPU. Tokenization has no GPU path worth measuring here: it is
/// branchy string work on short inputs, and the transfer to a device would cost more than the work.
/// What does scale is running documents in parallel, which is what the second benchmark measures.
/// </remarks>
[BenchmarkCategory("GraviTokenizers")]
public class TokenizerBenchmarks
{
    /// <summary>Nine languages across three scripts, repeated to a thousand documents.</summary>
    private static readonly string[] Sentences =
    [
        "The quick brown fox jumps over the lazy dog while the markets open in London.",
        "Model bahasa ini dilatih dengan data dari berbagai sumber di Indonesia.",
        "Le modèle de langue a été entraîné sur des textes provenant de nombreuses sources.",
        "Das Sprachmodell wurde mit Texten aus vielen verschiedenen Quellen trainiert.",
        "El modelo de lenguaje fue entrenado con textos de muchas fuentes diferentes.",
        "Il modello linguistico è stato addestrato su testi provenienti da molte fonti.",
        "Языковая модель была обучена на текстах из множества источников.",
        "言語モデルは多くの情報源からのテキストで学習されました。",
        "O modelo de linguagem foi treinado com textos de muitas fontes diferentes.",
    ];

    private string[] _corpus = [];
    private HfTokenizer _tokenizer = null!;

    [Params("bert-base-multilingual-cased", "xlm-roberta-base", "gpt2")]
    public string Tokenizer { get; set; } = "";

    [GlobalSetup]
    public void Setup()
    {
        _tokenizer = HfTokenizer.FromPretrained(Tokenizer);
        _corpus = [.. Enumerable.Range(0, 1000).Select(i => Sentences[i % Sentences.Length] + $" #{i}")];
    }

    [Benchmark(Baseline = true, Description = "1,000 documents, one thread")]
    public int Sequential()
    {
        var tokens = 0;
        foreach (var text in _corpus) tokens += _tokenizer.Encode(text).Length;
        return tokens;
    }

    [Benchmark(Description = "1,000 documents, EncodeBatch (parallel)")]
    public int Parallel() => _tokenizer.EncodeBatch(_corpus, BatchOptions.Default).Encodings.Sum(e => e.Length);
}
