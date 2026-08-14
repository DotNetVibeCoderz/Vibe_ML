using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Io;
using Gravicode.Science.GraviText.Transformers;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for loading a full transformer checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// Loading weights is easy to get <em>nearly</em> right, and nearly right is invisible: a
/// transposed projection or a skipped layer produces a model that runs and returns plausible
/// vectors. So these check the things that distinguish a correct load from a plausible one — that
/// every parameter actually moved, that a wrong transpose is refused rather than absorbed, and that
/// a partial load is not reported as success.
/// </para>
/// <para>
/// The end-to-end check — the encoder's output against an independent NumPy implementation of the
/// same architecture, agreeing to 2.6e-07 — lives in <c>tools/verify/checkpoint_interop.py</c>,
/// because it needs a Python toolchain the tests do not.
/// </para>
/// </remarks>
public class CheckpointTests : IDisposable
{
    private readonly List<string> _temporary = [];

    private static readonly TransformerConfig Tiny = new(
        VocabularySize: 20, HiddenSize: 8, Layers: 2, Heads: 2,
        IntermediateSize: 16, MaxPositions: 12);

    public void Dispose()
    {
        foreach (var path in _temporary)
            if (File.Exists(path)) File.Delete(path);
    }

    private string TempFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.onnx");
        _temporary.Add(path);
        return path;
    }

    /// <summary>
    /// Writes a checkpoint for <see cref="Tiny"/>, in PyTorch's (outputs, inputs) orientation.
    /// </summary>
    /// <param name="skipLayer">Omit every tensor of this layer, to simulate a partial export.</param>
    /// <param name="fill">Value written into every element, so a load is easy to verify.</param>
    private string WriteCheckpoint(int skipLayer = -1, double fill = 0.25)
    {
        var builder = new OnnxGraphBuilder("input", Tiny.HiddenSize);

        NdArray Filled(int rows, int columns)
        {
            var array = NdArray.Zeros(rows, columns);
            for (var i = 0; i < array.Size; i++) array.SetAt(i, fill);
            return array;
        }

        NdArray Vector(int size)
        {
            var array = NdArray.Zeros(size);
            for (var i = 0; i < size; i++) array.SetAt(i, fill);
            return array;
        }

        builder.AddInitializer("bert.embeddings.word_embeddings.weight",
            Filled(Tiny.VocabularySize, Tiny.HiddenSize));
        builder.AddInitializer("bert.embeddings.position_embeddings.weight",
            Filled(Tiny.MaxPositions, Tiny.HiddenSize));
        builder.AddInitializer("bert.embeddings.LayerNorm.weight", Vector(Tiny.HiddenSize));
        builder.AddInitializer("bert.embeddings.LayerNorm.bias", Vector(Tiny.HiddenSize));

        for (var layer = 0; layer < Tiny.Layers; layer++)
        {
            if (layer == skipLayer) continue;

            var prefix = $"bert.encoder.layer.{layer}";

            foreach (var part in new[]
            {
                "attention.self.query", "attention.self.key",
                "attention.self.value", "attention.output.dense",
            })
            {
                builder.AddInitializer($"{prefix}.{part}.weight", Filled(Tiny.HiddenSize, Tiny.HiddenSize));
                builder.AddInitializer($"{prefix}.{part}.bias", Vector(Tiny.HiddenSize));
            }

            builder.AddInitializer($"{prefix}.attention.output.LayerNorm.weight", Vector(Tiny.HiddenSize));
            builder.AddInitializer($"{prefix}.attention.output.LayerNorm.bias", Vector(Tiny.HiddenSize));

            // PyTorch stores (out, in), so the expansion is (intermediate, hidden).
            builder.AddInitializer($"{prefix}.intermediate.dense.weight",
                Filled(Tiny.IntermediateSize, Tiny.HiddenSize));
            builder.AddInitializer($"{prefix}.intermediate.dense.bias", Vector(Tiny.IntermediateSize));

            builder.AddInitializer($"{prefix}.output.dense.weight",
                Filled(Tiny.HiddenSize, Tiny.IntermediateSize));
            builder.AddInitializer($"{prefix}.output.dense.bias", Vector(Tiny.HiddenSize));

            builder.AddInitializer($"{prefix}.output.LayerNorm.weight", Vector(Tiny.HiddenSize));
            builder.AddInitializer($"{prefix}.output.LayerNorm.bias", Vector(Tiny.HiddenSize));
        }

        var path = TempFile();
        builder.AddNode("Identity", ["input"], "output");
        builder.Save(path, "output", Tiny.HiddenSize);
        return path;
    }

    [Fact]
    public void ACompleteCheckpointFillsEveryParameter()
    {
        var model = new TransformerModel(Tiny);
        var report = TransformerCheckpoint.Load(model, WriteCheckpoint());

        Assert.True(report.IsComplete, $"missing: {string.Join(", ", report.Missing)}");
        Assert.Empty(report.Unused);
        Assert.True(model.HasPretrainedWeights);

        // 4 embedding tensors + per layer: 8 attention + 4 norm + 4 feed-forward.
        Assert.Equal(4 + Tiny.Layers * 16, report.Loaded.Count);
    }

    [Fact]
    public void TheWeightsActuallyReachTheEncoderBlocks()
    {
        // The failure this exists for: LoadOnnxWeights filled only the embedding tables, so a
        // "pretrained" model still had twelve randomly initialised layers and nothing said so.
        const double fill = 0.125;

        var model = new TransformerModel(Tiny);
        TransformerCheckpoint.Load(model, WriteCheckpoint(fill: fill));

        foreach (var layer in model.Layers)
        {
            Assert.Equal(fill, layer.Attention.Query.Weights[0, 0], 6);
            Assert.Equal(fill, layer.Attention.Key.Weights[0, 0], 6);
            Assert.Equal(fill, layer.Attention.Value.Weights[0, 0], 6);
            Assert.Equal(fill, layer.Attention.Output.Weights[0, 0], 6);
            Assert.Equal(fill, layer.Intermediate.Weights[0, 0], 6);
            Assert.Equal(fill, layer.OutputProjection.Weights[0, 0], 6);

            Assert.Equal(fill, layer.AttentionNorm.Gamma.At(0), 6);
            Assert.Equal(fill, layer.OutputNorm.Beta.At(0), 6);
        }
    }

    [Fact]
    public void LoadingChangesWhatTheModelComputes()
    {
        // A load that quietly did nothing would leave the output identical.
        int[] tokens = [3, 7, 2, 5];

        var before = new TransformerModel(Tiny).Forward(tokens);

        var after = new TransformerModel(Tiny);
        TransformerCheckpoint.Load(after, WriteCheckpoint());

        Assert.False(UFunc.AllClose(before, after.Forward(tokens), 1e-9),
            "the output was unchanged, so the weights never took effect");
    }

    [Fact]
    public void AWrongTransposeIsRefusedRatherThanAbsorbed()
    {
        // The attention projections are square and accept either reading, so a wrong transpose
        // would load silently. The feed-forward weight is hidden x intermediate and is what
        // catches it.
        var model = new TransformerModel(Tiny);

        var error = Assert.Throws<InvalidDataException>(() => TransformerCheckpoint.Load(
            model, WriteCheckpoint(), CheckpointNames.HuggingFaceBert with { Transposed = false }));

        Assert.Contains("Transposed", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTransposeIsAppliedInTheRightDirection()
    {
        // A rectangular weight with distinct values, so the orientation is checkable rather than
        // merely consistent.
        var builder = new OnnxGraphBuilder("input", Tiny.HiddenSize);

        var expansion = NdArray.Zeros(Tiny.IntermediateSize, Tiny.HiddenSize);
        for (var i = 0; i < Tiny.IntermediateSize; i++)
            for (var j = 0; j < Tiny.HiddenSize; j++)
                expansion[i, j] = i * 100 + j;

        WriteMinimalCheckpoint(builder, expansion);

        var path = TempFile();
        builder.AddNode("Identity", ["input"], "output");
        builder.Save(path, "output", Tiny.HiddenSize);

        var model = new TransformerModel(Tiny);
        TransformerCheckpoint.Load(model, path, strict: false);

        // Stored (out, in) becomes (in, out): element [i, j] of the file lands at [j, i].
        var loaded = model.Layers[0].Intermediate.Weights;
        Assert.Equal(Tiny.HiddenSize, loaded.Shape[0]);
        Assert.Equal(Tiny.IntermediateSize, loaded.Shape[1]);

        Assert.Equal(expansion[3, 5], loaded[5, 3], 6);
        Assert.Equal(expansion[7, 1], loaded[1, 7], 6);
    }

    /// <summary>Writes just enough for one layer's expansion weight to be loadable.</summary>
    private static void WriteMinimalCheckpoint(OnnxGraphBuilder builder, NdArray expansion)
    {
        builder.AddInitializer("bert.encoder.layer.0.intermediate.dense.weight", expansion);
    }

    [Fact]
    public void APartialCheckpointIsRefusedInStrictMode()
    {
        var model = new TransformerModel(Tiny);

        var error = Assert.Throws<InvalidDataException>(
            () => TransformerCheckpoint.Load(model, WriteCheckpoint(skipLayer: 1)));

        Assert.Contains("missing", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void APartialLoadIsNotReportedAsPretrained()
    {
        // The point of the flag. A model with one of two layers loaded produces output that is
        // neither the checkpoint's nor a random model's, and nothing downstream could tell.
        var model = new TransformerModel(Tiny);
        var report = TransformerCheckpoint.Load(model, WriteCheckpoint(skipLayer: 1), strict: false);

        Assert.False(report.IsComplete);
        Assert.False(model.HasPretrainedWeights);
        Assert.Equal(16, report.Missing.Count);          // one layer's worth
    }

    [Fact]
    public void TheReportNamesWhatWasMissing()
    {
        var model = new TransformerModel(Tiny);
        var report = TransformerCheckpoint.Load(model, WriteCheckpoint(skipLayer: 0), strict: false);

        Assert.All(report.Missing, name =>
            Assert.Contains("layer.0", name, StringComparison.Ordinal));
    }

    [Fact]
    public void AMismatchedArchitectureIsRejected()
    {
        // A checkpoint for a wider model would otherwise load its first columns and look fine.
        var wider = new TransformerConfig(
            VocabularySize: 20, HiddenSize: 16, Layers: 2, Heads: 2,
            IntermediateSize: 32, MaxPositions: 12);

        var model = new TransformerModel(wider);
        Assert.ThrowsAny<InvalidDataException>(() => TransformerCheckpoint.Load(model, WriteCheckpoint()));
    }

    [Fact]
    public void InspectListsWhatTheFileHolds()
    {
        // The first thing to run on an unfamiliar checkpoint, since names are a convention and the
        // file is the only authority on which one it follows.
        var contents = TransformerCheckpoint.Inspect(WriteCheckpoint());

        Assert.Equal(4 + Tiny.Layers * 16, contents.Count);
        Assert.Contains(contents, t => t.Name == "bert.embeddings.word_embeddings.weight");

        var tokens = contents.First(t => t.Name == "bert.embeddings.word_embeddings.weight");
        Assert.Equal([Tiny.VocabularySize, Tiny.HiddenSize], tokens.Shape);
    }

    [Fact]
    public void UnusedTensorsAreReportedRatherThanIgnored()
    {
        // A checkpoint carrying a pooler or a classification head has tensors this model does not
        // consume. Reporting them is how a wrong naming convention becomes visible.
        var builder = new OnnxGraphBuilder("input", Tiny.HiddenSize);
        builder.AddInitializer("bert.pooler.dense.weight", NdArray.Zeros(Tiny.HiddenSize, Tiny.HiddenSize));

        var path = TempFile();
        builder.AddNode("Identity", ["input"], "output");
        builder.Save(path, "output", Tiny.HiddenSize);

        var model = new TransformerModel(Tiny);
        var report = TransformerCheckpoint.Load(model, path, strict: false);

        Assert.Contains("bert.pooler.dense.weight", report.Unused);
    }

    [Fact]
    public void AReprefixedConventionRewritesEveryName()
    {
        // A RoBERTa export differs from BERT's mostly in the prefix, so swapping it is often the
        // whole adaptation.
        var names = CheckpointNames.Reprefixed("roberta.");

        Assert.Equal("roberta.embeddings.word_embeddings.weight", names.TokenEmbeddings);
        Assert.StartsWith("roberta.encoder.layer.{0}", names.QueryWeight, StringComparison.Ordinal);
        Assert.DoesNotContain("bert.", names.OutputNormShift[..8], StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnprefixedConventionDropsTheModelName()
    {
        var names = CheckpointNames.Unprefixed;

        Assert.Equal("embeddings.word_embeddings.weight", names.TokenEmbeddings);
        Assert.Equal("encoder.layer.{0}.attention.self.query.weight", names.QueryWeight);
    }

    [Fact]
    public void MalformedRequestsAreRejected()
    {
        var model = new TransformerModel(Tiny);

        Assert.Throws<ArgumentException>(() => TransformerCheckpoint.Load(model, ""));
        Assert.Throws<ArgumentNullException>(() => TransformerCheckpoint.Load(null!, "x.onnx"));

        Assert.Throws<ArgumentException>(
            () => model.ReplaceTokenEmbeddings(NdArray.Zeros(10, 999)));
    }
}
