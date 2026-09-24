using Gravicode.HFNet.GraviDiffusers;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.HFNet.GraviTransformers.Vision;
using Gravicode.Science.GraviNum;
using HFGallery.Controls;
using Slice = HFGallery.Controls.Slice;

namespace HFGallery.Cases;

/// <summary>
/// Keeps loaded models alive for the life of the gallery.
/// </summary>
/// <remarks>
/// A model is hundreds of megabytes and takes seconds to load. Reloading it each time a case runs
/// would make the gallery feel broken even though every individual step is working, so models are
/// held and shared - which also means moving between two cases that use the same checkpoint is
/// instant.
/// </remarks>
internal static class ModelCache
{
    private static readonly SemaphoreSlim Gate = new(1, 1);
    private static readonly Dictionary<string, TransformerModel> Models = new(StringComparer.Ordinal);
    private static readonly Dictionary<string, HfTokenizer> Tokenizers = new(StringComparer.Ordinal);
    private static readonly Dictionary<string, VisionTransformer> Vision = new(StringComparer.Ordinal);

    internal static async Task<TransformerModel> ModelAsync(
        string id, IProgress<string> progress, CancellationToken token)
    {
        await Gate.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (Models.TryGetValue(id, out var cached)) return cached;

            progress.Report($"Loading {id} — the first run downloads it.");
            var model = await Task.Run(() => TransformerModel.Load(id), token).ConfigureAwait(false);

            Models[id] = model;
            progress.Report($"{model.Config.Layers} layers, {model.Config.HiddenSize} hidden, ready.");
            return model;
        }
        finally { Gate.Release(); }
    }

    internal static async Task<VisionTransformer> VisionAsync(
        string id, IProgress<string> progress, CancellationToken token)
    {
        await Gate.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (Vision.TryGetValue(id, out var cached)) return cached;

            progress.Report($"Loading {id} — the first run downloads it.");
            var model = await Task.Run(() => VisionTransformer.Load(id), token).ConfigureAwait(false);

            Vision[id] = model;
            progress.Report($"{model.Config.Layers} layers, {model.Config.Patches} patches, ready.");
            return model;
        }
        finally { Gate.Release(); }
    }

    internal static async Task<HfTokenizer> TokenizerAsync(
        string id, IProgress<string> progress, CancellationToken token)
    {
        await Gate.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (Tokenizers.TryGetValue(id, out var cached)) return cached;

            progress.Report($"Fetching the tokenizer for {id}.");
            var tokenizer = await Task.Run(() => HfTokenizer.FromPretrained(id), token).ConfigureAwait(false);

            Tokenizers[id] = tokenizer;
            return tokenizer;
        }
        finally { Gate.Release(); }
    }
}

/// <summary>Every case the gallery offers, in the order the rail shows them.</summary>
public static class Catalog
{
    /// <summary>The cases.</summary>
    public static IReadOnlyList<GalleryCase> All { get; } =
    [
        new ImageCase(),
        new SentimentCase(),
        new FillMaskCase(),
        new EntitiesCase(),
        new AnswerCase(),
        new SearchCase(),
        new EmbeddingMapCase(),
        new TokenizerCase(),
        new CheckpointCase(),
        new SchedulerCase(),
    ];
}

// ====================================================================== image classification

internal sealed class ImageCase : GalleryCase
{
    /// <summary>Where the sample pictures come from, so the case needs nothing checked in.</summary>
    private const string Gallery = "huggingface/documentation-images";

    public override string Title => "What is in this picture";
    public override string Blurb => "Run a Vision Transformer over an image and read the label off it.";
    public override string Library => "GraviTransformers";
    public override string Model => "google/vit-base-patch16-224";
    public override string InputLabel => "A local image path, or a file from huggingface/documentation-images";

    public override string? DefaultInput => "bee.jpg";

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers.Vision;

        // ViT is the same encoder block as BERT over a different embedding: the image is cut into
        // 16px squares, each square is projected to one vector, and a learned [CLS] vector goes in
        // front. From there it is a 197-token sequence like any other.
        using var model = VisionTransformer.Load("google/vit-base-patch16-224");

        foreach (var prediction in model.Classify(path, topK: 5))
            Console.WriteLine($"{prediction.Label,-30} {prediction.Score:P2}");

        // bee                             94.46 %
        // pot, flowerpot                   1.32 %
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.VisionAsync(Model, progress, token).ConfigureAwait(false);

        string path;
        if (File.Exists(input))
        {
            path = input;
        }
        else
        {
            progress.Report($"Fetching {input} from {Gallery}.");
            path = await Task.Run(() => Hub.DownloadDatasetFile(Gallery, input), token).ConfigureAwait(false);
        }

        progress.Report($"Running the encoder over {model.Config.Patches} patches.");
        var predictions = await Task.Run(() => model.Classify(path, topK: 5), token).ConfigureAwait(false);

        var best = predictions[0];

        return new CaseResult
        {
            Summary = $"{best.Label} at {best.Score:P1}.",

            // Sequential by rank: these are five candidates for one answer, so what the colour
            // should say is which is more likely, not which class this is.
            Bars = [.. predictions.Select(p => new Datum(p.Label, p.Score))],

            Facts =
            [
                ("image", Path.GetFileName(path)),
                ("resolution", $"{model.Processor.Size}x{model.Processor.Size}"),
                ("patches", $"{model.Config.Grid}x{model.Config.Grid} of {model.Config.PatchSize}px"),
                ("classes", $"{model.Config.LabelCount:N0}"),
            ],
        };
    }
}

// ====================================================================== sentiment

internal sealed class SentimentCase : GalleryCase
{
    public override string Title => "Sentiment";
    public override string Blurb => "Classify text with a fine-tuned model, labels and all.";
    public override string Library => "GraviTransformers";
    public override string Model => "distilbert-base-uncased-finetuned-sst-2-english";
    public override string InputLabel => "Text to classify";

    public override string? DefaultInput =>
        "The picture is beautiful and the pacing is a mess, but I would watch it again.";

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        // A fine-tuned checkpoint carries its own classification head and its own label
        // names, so nothing here has to be told what the classes are.
        using var model = TransformerModel.Load(
            "distilbert-base-uncased-finetuned-sst-2-english");

        foreach (var prediction in model.Predict(text))
            Console.WriteLine($"{prediction.Label,-9} {prediction.Score:P2}");
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);
        var predictions = await Task.Run(() => model.Predict(input), token).ConfigureAwait(false);

        var best = predictions[0];

        return new CaseResult
        {
            Summary = $"{best.Label} at {best.Score:P1}.",

            // Categorical, not sequential: these are two named classes, not two magnitudes of
            // one thing, and the colour should say which class rather than which is bigger.
            Bars = [.. predictions.Select(p => new Datum(p.Label, p.Score, p.Index))],

            Facts =
            [
                ("model", model.Config.ModelType),
                ("labels", string.Join(", ", model.Labels)),
                ("layers", model.Config.Layers.ToString()),
            ],
        };
    }
}

// ====================================================================== fill mask

internal sealed class FillMaskCase : GalleryCase
{
    public override string Title => "Fill in the blank";
    public override string Blurb => "Ask a pretrained encoder what belongs in a gap.";
    public override string Library => "GraviTransformers";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "A sentence containing [MASK]";

    public override string? DefaultInput => "The capital of France is [MASK].";

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");

        // The sharpest check that a checkpoint loaded correctly: a model with a transposed
        // weight still produces plausible vectors, but it does not answer this with "paris".
        foreach (var fill in model.FillMask(text, topK: 6))
            Console.WriteLine($"{fill.Token,-14} {fill.Score:P2}");
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);

        if (!input.Contains("[MASK]", StringComparison.Ordinal))
        {
            return new CaseResult { Summary = "Put [MASK] somewhere in the sentence." };
        }

        var fills = await Task.Run(() => model.FillMask(input, topK: 6), token).ConfigureAwait(false);

        return new CaseResult
        {
            Summary = $"Most likely: {fills[0].Token} ({fills[0].Score:P1}).",

            // Sequential by rank here, because the six candidates are one quantity - probability -
            // and their identity carries no meaning beyond their order.
            Bars = [.. fills.Select(f => new Datum(f.Token, f.Score))],
        };
    }
}

// ====================================================================== entities

internal sealed class EntitiesCase : GalleryCase
{
    public override string Title => "Named entities";
    public override string Blurb => "Find people, places and organisations, as spans of the original text.";
    public override string Library => "GraviTransformers";
    public override string Model => "dslim/bert-base-NER";
    public override string InputLabel => "Text to tag";

    public override string? DefaultInput =>
        "Kang Fadhil founded Gravicode Studios in Bandung, and later spoke about HF.Net at a "
        + "Microsoft event in Jakarta.";

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("dslim/bert-base-NER");

        // Spans come back as substrings of the input, taken from the tokenizer's offsets - so
        // the original casing and any punctuation inside an entity survive. Rejoining subword
        // pieces instead loses both and leaves "##" to clean up by guesswork.
        foreach (var entity in model.FindEntities(text))
            Console.WriteLine($"{entity.Label,-6} {entity.Text}  [{entity.Start}..{entity.End})");
        """;

    private static int CategoryOf(string label) => label switch
    {
        "PER" => 0,
        "ORG" => 1,
        "LOC" => 2,
        _ => 3,
    };

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);
        var entities = await Task.Run(() => model.FindEntities(input), token).ConfigureAwait(false);

        var byType = entities
            .GroupBy(e => e.Label)
            .Select(g => new Datum(g.Key, g.Count(), CategoryOf(g.Key)))
            .OrderByDescending(d => d.Value)
            .ToList();

        return new CaseResult
        {
            Summary = entities.Count == 0
                ? "No entities found."
                : $"{entities.Count} entities across {byType.Count} types.",

            Spans = new SpanText(input, [.. entities.Select(
                e => new SpanMark(e.Start, e.End, e.Label, e.Score, CategoryOf(e.Label)))]),

            Bars = byType,
            Unit = "",

            // All four, always, and never filtered to the ones that turned up: the hue for a
            // type is fixed by CategoryOf, so dropping an absent type would slide every later
            // swatch onto the wrong colour.
            Legend = ["PER", "ORG", "LOC", "MISC"],
        };
    }
}

// ====================================================================== question answering

internal sealed class AnswerCase : GalleryCase
{
    public override string Title => "Question answering";
    public override string Blurb => "Extract the answer as a span of the passage, never as invented text.";
    public override string Library => "GraviTransformers";
    public override string Model => "distilbert-base-cased-distilled-squad";
    public override string InputLabel => "Question";
    public override string SecondInputLabel => "Passage to answer from";

    public override string? DefaultInput => "What does HF.Net read?";

    public override string? DefaultSecondInput =>
        "HF.Net is a Hugging Face style machine learning stack for .NET, built by Gravicode "
        + "Studios. It reads both safetensors and PyTorch checkpoints, and its tokenizers produce "
        + "the same ids as the reference implementation. For throughput it exports to ONNX.";

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("distilbert-base-cased-distilled-squad");

        // The question and the passage go in as a pair, so this is also what exercises the
        // segment embeddings. Only positions inside the passage are eligible, and the end can
        // never precede the start - an unconstrained argmax over each produces neither.
        foreach (var answer in model.Answer(question, passage, topK: 3))
            Console.WriteLine($"{answer.Text}  ({answer.Score:P1})");
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);
        var answers = await Task.Run(() => model.Answer(input, second, topK: 4), token)
            .ConfigureAwait(false);

        var best = answers[0];

        return new CaseResult
        {
            Summary = best.IsEmpty ? "No answer in the passage." : $"“{best.Text}”",

            Spans = new SpanText(second, [new SpanMark(best.Start, best.End, "answer", best.Score, 0)]),

            Bars = [.. answers.Select(a => new Datum(
                a.Text.Length > 34 ? a.Text[..32] + "…" : a.Text, a.Score))],
        };
    }
}

// ====================================================================== semantic search

internal sealed class SearchCase : GalleryCase
{
    public override string Title => "Semantic search";
    public override string Blurb => "Embed a corpus once, then rank it against a query by meaning.";
    public override string Library => "GraviTransformers";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "Query";

    public override string? DefaultInput => "a dog playing outside";

    public override string SecondInputLabel => "Corpus, one document per line";

    public override string? DefaultSecondInput => string.Join('\n',
    [
        "The cat sat quietly on the mat.",
        "A golden retriever played fetch in the park.",
        "Quarterly revenue exceeded analyst expectations.",
        "The board approved the merger this morning.",
        "Puppies need a great deal of exercise.",
        "Interest rates were left unchanged.",
    ]);

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");

        // Embedded once. Every query after that is a handful of dot products, which is the
        // whole reason to do it in this order.
        var corpus = model.EmbedBatch(documents);
        var query = model.Embed(text);

        var ranked = documents
            .Select((d, i) => (Document: d, Score: Cosine(corpus.Row(i), query)))
            .OrderByDescending(r => r.Score);
        """;

    /// <summary>Cosine similarity between two vectors.</summary>
    internal static double Cosine(NdArray a, NdArray b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (var i = 0; i < a.Size; i++)
        {
            dot += a.At(i) * b.At(i);
            normA += a.At(i) * a.At(i);
            normB += b.At(i) * b.At(i);
        }

        var denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denominator == 0 ? 0 : dot / denominator;
    }

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);

        var documents = second.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (documents.Length == 0) return new CaseResult { Summary = "The corpus is empty." };

        progress.Report($"Embedding {documents.Length} documents and the query.");

        var (corpus, query) = await Task.Run(
            () => (model.EmbedBatch(documents), model.Embed(input)), token).ConfigureAwait(false);

        var ranked = Enumerable.Range(0, documents.Length)
            .Select(i => (Document: documents[i], Score: Cosine(corpus.Row(i), query)))
            .OrderByDescending(r => r.Score)
            .ToList();

        return new CaseResult
        {
            Summary = $"Closest: “{ranked[0].Document}”",

            Bars = [.. ranked.Select(r => new Datum(
                r.Document.Length > 40 ? r.Document[..38] + "…" : r.Document, r.Score))],
        };
    }
}

// ====================================================================== embedding map

internal sealed class EmbeddingMapCase : GalleryCase
{
    public override string Title => "Embedding map";
    public override string Blurb => "Project sentences to two dimensions and watch the topics separate.";
    public override string Library => "GraviTransformers";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "Sentences, one per line, prefixed by a group name and a colon";

    public override string? DefaultInput => string.Join('\n',
    [
        "animals: The cat sat quietly on the mat.",
        "animals: A golden retriever played fetch in the park.",
        "animals: Puppies need a great deal of exercise.",
        "finance: Quarterly revenue exceeded analyst expectations.",
        "finance: The board approved the merger this morning.",
        "finance: Interest rates were left unchanged.",
        "cooking: Fold the egg whites in gently.",
        "cooking: Simmer the stock for about two hours.",
    ]);

    public override string Code => """
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");
        var vectors = model.EmbedBatch(sentences);

        // 768 dimensions cannot be looked at, so the two directions of greatest variance stand
        // in for the rest. Centre first: without it the first component is just the mean, and
        // every point lands on one line.
        var centred = Centre(vectors);
        var (x, y) = FirstTwoComponents(centred);
        """;

    /// <summary>
    /// Projects onto the first two principal components, found by power iteration.
    /// </summary>
    /// <remarks>
    /// Power iteration rather than a full SVD because two components out of 768 is exactly the
    /// case it suits, and the second is found by deflating the first out of the data - taking the
    /// top two eigenvectors of a matrix built once would cost far more for the same answer.
    /// </remarks>
    private static (double[] X, double[] Y) Project(NdArray vectors)
    {
        var rows = vectors.Shape[0];
        var width = vectors.Shape[1];

        var data = new double[rows][];
        var mean = new double[width];

        for (var i = 0; i < rows; i++)
        {
            data[i] = vectors.Row(i).ToArray();
            for (var d = 0; d < width; d++) mean[d] += data[i][d] / rows;
        }

        foreach (var row in data)
        {
            for (var d = 0; d < width; d++) row[d] -= mean[d];
        }

        var first = Component(data, width, null);
        var second = Component(data, width, first);

        var x = new double[rows];
        var y = new double[rows];

        for (var i = 0; i < rows; i++)
        {
            for (var d = 0; d < width; d++)
            {
                x[i] += data[i][d] * first[d];
                y[i] += data[i][d] * second[d];
            }
        }

        return (x, y);
    }

    private static double[] Component(double[][] data, int width, double[]? deflate)
    {
        var random = new GraviRandom(7);
        var vector = new double[width];
        for (var d = 0; d < width; d++) vector[d] = random.Normal();

        for (var step = 0; step < 48; step++)
        {
            var next = new double[width];

            // Multiply by the covariance without ever forming it: X^T (X v) is the same product
            // and costs rows x width instead of width squared.
            foreach (var row in data)
            {
                var dot = 0.0;
                for (var d = 0; d < width; d++) dot += row[d] * vector[d];
                for (var d = 0; d < width; d++) next[d] += row[d] * dot;
            }

            if (deflate is not null)
            {
                var overlap = 0.0;
                for (var d = 0; d < width; d++) overlap += next[d] * deflate[d];
                for (var d = 0; d < width; d++) next[d] -= overlap * deflate[d];
            }

            var norm = Math.Sqrt(next.Sum(v => v * v));
            if (norm < 1e-12) break;

            for (var d = 0; d < width; d++) vector[d] = next[d] / norm;
        }

        return vector;
    }

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token).ConfigureAwait(false);

        var lines = input.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (lines.Length < 3) return new CaseResult { Summary = "Give it at least three sentences." };

        var groups = new List<string>();
        var sentences = new List<string>();

        foreach (var line in lines)
        {
            var colon = line.IndexOf(':');
            groups.Add(colon > 0 ? line[..colon].Trim() : "all");
            sentences.Add(colon > 0 ? line[(colon + 1)..].Trim() : line);
        }

        progress.Report($"Embedding {sentences.Count} sentences.");
        var vectors = await Task.Run(() => model.EmbedBatch(sentences), token).ConfigureAwait(false);

        var (x, y) = Project(vectors);
        var order = groups.Distinct().ToList();

        return new CaseResult
        {
            Summary = $"{sentences.Count} sentences, {order.Count} groups, projected onto two components.",

            Scatter = [.. Enumerable.Range(0, sentences.Count).Select(
                i => new Point2(x[i], y[i], sentences[i], order.IndexOf(groups[i])))],

            Legend = order,

            Facts = [.. order.Select(g => (g, $"{groups.Count(h => h == g)} sentences"))],
        };
    }
}

// ====================================================================== tokenizer

internal sealed class TokenizerCase : GalleryCase
{
    public override string Title => "Tokenizer";
    public override string Blurb => "See the pieces a model actually reads, and where each came from.";
    public override string Library => "GraviTokenizers";
    public override string Model => "bert-base-uncased";
    public override CaseCost Cost => CaseCost.Light;
    public override string InputLabel => "Text to tokenize";

    public override string? DefaultInput => "Tokenizers are unbelievably useful, aren't they?";

    public override string Code => """
        using Gravicode.HFNet.GraviTokenizers;

        var tokenizer = HfTokenizer.FromPretrained("bert-base-uncased");
        var encoding = tokenizer.Encode(text);

        // Offsets index the ORIGINAL string, so a piece can always be traced back to the
        // characters it came from - capitals and punctuation included.
        for (var i = 0; i < encoding.Length; i++)
            Console.WriteLine($"{encoding.Tokens[i],-14} '{encoding.Span(text, i)}'");
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var tokenizer = await ModelCache.TokenizerAsync(Model, progress, token).ConfigureAwait(false);
        var encoding = tokenizer.Encode(input);

        var marks = new List<SpanMark>();
        var category = 0;

        for (var i = 0; i < encoding.Length; i++)
        {
            if (encoding.SpecialTokensMask[i] == 1) continue;

            var (start, end) = encoding.Offsets[i];
            if (end <= start) continue;

            // Alternating hues, so two adjacent pieces of one word stay visibly two pieces.
            marks.Add(new SpanMark(start, end, encoding.Tokens[i], 1, category % 6));
            category++;
        }

        return new CaseResult
        {
            Summary = $"{encoding.Length} tokens from {input.Length} characters.",
            Spans = new SpanText(input, marks),

            Facts =
            [
                ("vocabulary", $"{tokenizer.VocabularySize:N0}"),
                ("tokens", encoding.Length.ToString()),
                ("characters", input.Length.ToString()),
                ("ids", string.Join(' ', encoding.Ids.Take(12))),
            ],
        };
    }
}

// ====================================================================== checkpoint

internal sealed class CheckpointCase : GalleryCase
{
    public override string Title => "Inside a checkpoint";
    public override string Blurb => "Where the weight actually sits in a 420 MB model file.";
    public override string Library => "GraviHub";
    public override CaseCost Cost => CaseCost.Heavy;
    public override string InputLabel => "Model id";

    public override string? DefaultInput => "bert-base-uncased";

    public override string Code => """
        using Gravicode.HFNet.GraviHub;
        using Gravicode.HFNet.GraviHub.Io;

        var path = Hub.DownloadFile(modelId, "model.safetensors");

        // Memory-mapped: listing four hundred tensors reads the header, not the gigabytes
        // behind it. Reading the declared header length first is what keeps that honest -
        // a fixed-size prefix read made this step 600x slower.
        using var reader = SafeTensors.Open(path);

        foreach (var tensor in reader.Tensors.OrderByDescending(t => t.ByteCount).Take(12))
            Console.WriteLine($"{tensor.Name,-52} {tensor.ByteCount / 1024 / 1024} MB");
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        progress.Report($"Reading {input}'s file listing.");

        var info = await Task.Run(() => Hub.ModelInfo(input), token).ConfigureAwait(false);
        var weights = info.FilesWithExtension(".safetensors").FirstOrDefault();

        if (weights.Path is null)
        {
            return new CaseResult { Summary = $"{input} publishes no safetensors file." };
        }

        progress.Report($"Downloading {weights.Path} ({(weights.Size ?? 0) / (1024 * 1024.0):0.#} MB on a first run).");
        var path = await Task.Run(() => Hub.DownloadFile(input, weights.Path), token).ConfigureAwait(false);

        return await Task.Run(() =>
        {
            using var reader = SafeTensors.Open(path);

            // Grouped by block: four hundred rows is a list, not a picture, and the question the
            // treemap answers is which *part* of the model the bytes are in.
            var groups = reader.Tensors
                .GroupBy(t => Group(t.Name))
                .Select(g => new Slice(g.Key, g.Sum(t => (double)t.ByteCount)))
                .OrderByDescending(s => s.Value)
                .ToList();

            var total = groups.Sum(g => g.Value);

            return new CaseResult
            {
                Summary = $"{reader.Tensors.Count} tensors, {total / (1024 * 1024):N0} MB, "
                    + $"largest group {groups[0].Label}.",

                Treemap = groups,

                Facts =
                [
                    ("file", $"{new FileInfo(path).Length / (1024 * 1024.0):N0} MB"),
                    ("tensors", reader.Tensors.Count.ToString()),
                    ("largest", reader.Tensors.OrderByDescending(t => t.ByteCount).First().Name),
                    ("dtype", reader.Tensors[0].DType.ToString()),
                ],
            };
        }, token).ConfigureAwait(false);
    }

    /// <summary>Collapses a parameter path to the part of the model it belongs to.</summary>
    /// <remarks>
    /// Across the layers rather than per layer. Twelve identical layers produce twenty-five slices,
    /// most of them too thin to label, and the shape they make says nothing a reader did not
    /// already know - the question a treemap of a checkpoint answers is which *kind* of parameter
    /// the bytes went into.
    /// </remarks>
    private static string Group(string name)
    {
        if (name.Contains("embeddings.LayerNorm", StringComparison.Ordinal)) return "layer norms";
        if (name.Contains("embeddings", StringComparison.Ordinal)) return "embeddings";
        if (name.Contains("pooler", StringComparison.Ordinal)) return "pooler";
        if (name.Contains("cls.", StringComparison.Ordinal)) return "output head";
        if (name.Contains("LayerNorm", StringComparison.Ordinal)) return "layer norms";
        if (name.Contains("attention", StringComparison.Ordinal)) return "attention";
        if (name.Contains("intermediate", StringComparison.Ordinal)
            || name.Contains("output", StringComparison.Ordinal)) return "feed-forward";

        return "other";
    }
}

// ====================================================================== schedulers

internal sealed class SchedulerCase : GalleryCase
{
    public override string Title => "Diffusion schedules";
    public override string Blurb => "How much signal is left at each step, and why the schedule must match the weights.";
    public override string Library => "GraviDiffusers";
    public override CaseCost Cost => CaseCost.Instant;
    public override string? DefaultInput => null;

    public override string Code => """
        using Gravicode.HFNet.GraviDiffusers;

        // Stable Diffusion trains on ScaledLinear. Using plain Linear betas with the same
        // endpoints produces images that are recognisably structured and consistently washed
        // out - which reads as a bad prompt rather than as the wrong schedule.
        var sd = NoiseSchedule.StableDiffusion;
        var ddpm = NoiseSchedule.Ddpm;
        var cosine = new NoiseSchedule(1000, schedule: BetaSchedule.SquaredCosine);

        // AlphaBar is the signal left after t steps: 1 at the start, near zero at the end.
        foreach (var t in new[] { 0, 250, 500, 750, 999 })
            Console.WriteLine($"{t,4}  {sd.AlphaBar(t):F4}");
        """;

    public override Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        const int Points = 100;

        static IReadOnlyList<double> Sample(NoiseSchedule schedule)
            => [.. Enumerable.Range(0, Points).Select(
                i => schedule.AlphaBar(i * (schedule.TrainTimesteps - 1) / (Points - 1)))];

        var stableDiffusion = NoiseSchedule.StableDiffusion;
        var ddpm = NoiseSchedule.Ddpm;
        var cosine = new NoiseSchedule(1000, schedule: BetaSchedule.SquaredCosine);

        return Task.FromResult(new CaseResult
        {
            Summary = "Signal remaining at each training timestep. All three end near zero and "
                + "differ entirely in how they get there.",

            Lines =
            [
                new Series("ScaledLinear (Stable Diffusion)", Sample(stableDiffusion), 0),
                new Series("Linear (DDPM)", Sample(ddpm), 1),
                new Series("SquaredCosine", Sample(cosine), 2),
            ],

            XLabel = "timestep 0 → 999",

            Facts =
            [
                ("SD at t=500", $"{stableDiffusion.AlphaBar(500):F4}"),
                ("DDPM at t=500", $"{ddpm.AlphaBar(500):F4}"),
                ("cosine at t=500", $"{cosine.AlphaBar(500):F4}"),
                ("all at t=-1", "1.0000"),
            ],
        });
    }
}
