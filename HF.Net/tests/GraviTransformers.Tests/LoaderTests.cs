using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>Tests for reading <c>config.json</c>.</summary>
public sealed class PretrainedConfigTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-config", Guid.NewGuid().ToString("N"));

    public PretrainedConfigTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private PretrainedConfig Load(string json)
    {
        var path = Path.Combine(_directory, $"{Guid.NewGuid():N}.json");
        File.WriteAllText(path, json);
        return PretrainedConfig.Load(path);
    }

    [Fact]
    public void ReadsBertFieldNames()
    {
        var config = Load("""
        {
          "model_type": "bert",
          "hidden_size": 768,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "intermediate_size": 3072,
          "vocab_size": 30522,
          "max_position_embeddings": 512
        }
        """);

        Assert.Equal("bert", config.ModelType);
        Assert.Equal(768, config.HiddenSize);
        Assert.Equal(12, config.Layers);
        Assert.Equal(12, config.Heads);
        Assert.Equal(3072, config.IntermediateSize);
        Assert.Equal(64, config.HeadSize);
    }

    [Fact]
    public void ReadsDistilBertsDifferentFieldNames()
    {
        // DistilBERT writes dim, n_layers, n_heads and hidden_dim for the same four quantities.
        var config = Load("""
        {
          "model_type": "distilbert",
          "dim": 768,
          "n_layers": 6,
          "n_heads": 12,
          "hidden_dim": 3072,
          "vocab_size": 30522
        }
        """);

        Assert.Equal(768, config.HiddenSize);
        Assert.Equal(6, config.Layers);
        Assert.Equal(12, config.Heads);
        Assert.Equal(3072, config.IntermediateSize);
    }

    [Fact]
    public void DefaultsIntermediateSizeToFourTimesHidden()
    {
        var config = Load("""
        {"model_type": "bert", "hidden_size": 128, "num_hidden_layers": 2,
         "num_attention_heads": 2, "vocab_size": 100}
        """);

        Assert.Equal(512, config.IntermediateSize);
    }

    [Fact]
    public void GivesRobertaItsPositionOffset()
    {
        // RoBERTa reserves two position slots, which is why its max_position_embeddings is 514.
        var roberta = Load("""
        {"model_type": "roberta", "hidden_size": 768, "num_hidden_layers": 12,
         "num_attention_heads": 12, "vocab_size": 50265, "max_position_embeddings": 514}
        """);

        var bert = Load("""
        {"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12,
         "num_attention_heads": 12, "vocab_size": 30522}
        """);

        Assert.Equal(2, roberta.PositionOffset);
        Assert.Equal(0, bert.PositionOffset);
    }

    [Fact]
    public void ReadsLabelNames()
    {
        var config = Load("""
        {"model_type": "bert", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
         "vocab_size": 10, "id2label": {"0": "NEGATIVE", "1": "POSITIVE"}}
        """);

        Assert.Equal(2, config.LabelCount);
        Assert.Equal("NEGATIVE", config.IdToLabel[0]);
        Assert.Equal("POSITIVE", config.IdToLabel[1]);
    }

    [Fact]
    public void RefusesAConfigWithNoLayerCountRatherThanGuessing()
    {
        var exception = Assert.Throws<InvalidDataException>(() => Load("""
        {"model_type": "bert", "hidden_size": 768, "num_attention_heads": 12, "vocab_size": 100}
        """));

        Assert.Contains("layer count", exception.Message);
    }

    [Fact]
    public void RefusesAConfigWithNoHiddenSize()
        => Assert.Throws<InvalidDataException>(() => Load("""{"model_type": "bert"}"""));
}

/// <summary>
/// Tests that build a complete, tiny BERT checkpoint and load it.
/// </summary>
/// <remarks>
/// A synthetic checkpoint is what makes the loader testable without the network. It covers the
/// parts most likely to be silently wrong - the name prefix, the transpose convention, and the
/// legacy gamma/beta spelling - each of which produces a model that runs and returns nonsense.
/// </remarks>
public sealed class CheckpointLoaderTests : IDisposable
{
    private const int Hidden = 8;
    private const int Heads = 2;
    private const int Layers = 2;
    private const int Intermediate = 16;
    private const int Vocabulary = 24;
    private const int Positions = 12;

    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-ckpt", Guid.NewGuid().ToString("N"));

    public CheckpointLoaderTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private static PretrainedConfig Config => new()
    {
        ModelType = "bert",
        HiddenSize = Hidden,
        Layers = Layers,
        Heads = Heads,
        IntermediateSize = Intermediate,
        VocabularySize = Vocabulary,
        MaxPositions = Positions,
    };

    /// <summary>Builds a checkpoint with every tensor a BERT encoder needs.</summary>
    /// <param name="prefix">The parameter prefix, for example <c>bert.</c>.</param>
    /// <param name="legacyNormNames">Whether to write gamma/beta instead of weight/bias.</param>
    private string BuildCheckpoint(string prefix = "bert.", bool legacyNormNames = false)
    {
        var random = new GraviRandom(11);
        var tensors = new Dictionary<string, NdArray>(StringComparer.Ordinal);

        NdArray Matrix(int rows, int columns)
        {
            var array = NdArray.Zeros(rows, columns);
            for (var i = 0; i < array.Size; i++) array.SetAt(i, random.Normal(0, 0.05));
            return array;
        }

        NdArray Vector(int size)
        {
            var array = NdArray.Zeros(size);
            for (var i = 0; i < size; i++) array.SetAt(i, random.Normal(0, 0.05));
            return array;
        }

        var scale = legacyNormNames ? "gamma" : "weight";
        var shift = legacyNormNames ? "beta" : "bias";

        tensors[$"{prefix}embeddings.word_embeddings.weight"] = Matrix(Vocabulary, Hidden);
        tensors[$"{prefix}embeddings.position_embeddings.weight"] = Matrix(Positions, Hidden);
        tensors[$"{prefix}embeddings.token_type_embeddings.weight"] = Matrix(2, Hidden);
        tensors[$"{prefix}embeddings.LayerNorm.{scale}"] = NdArray.Ones(Hidden);
        tensors[$"{prefix}embeddings.LayerNorm.{shift}"] = NdArray.Zeros(Hidden);

        for (var layer = 0; layer < Layers; layer++)
        {
            var block = $"{prefix}encoder.layer.{layer}";

            foreach (var projection in (string[])["query", "key", "value"])
            {
                // Stored transposed, as PyTorch's nn.Linear does: (outputs, inputs).
                tensors[$"{block}.attention.self.{projection}.weight"] = Matrix(Hidden, Hidden);
                tensors[$"{block}.attention.self.{projection}.bias"] = Vector(Hidden);
            }

            tensors[$"{block}.attention.output.dense.weight"] = Matrix(Hidden, Hidden);
            tensors[$"{block}.attention.output.dense.bias"] = Vector(Hidden);
            tensors[$"{block}.attention.output.LayerNorm.{scale}"] = NdArray.Ones(Hidden);
            tensors[$"{block}.attention.output.LayerNorm.{shift}"] = NdArray.Zeros(Hidden);

            // The non-square one, which is what settles the transpose convention.
            tensors[$"{block}.intermediate.dense.weight"] = Matrix(Intermediate, Hidden);
            tensors[$"{block}.intermediate.dense.bias"] = Vector(Intermediate);

            tensors[$"{block}.output.dense.weight"] = Matrix(Hidden, Intermediate);
            tensors[$"{block}.output.dense.bias"] = Vector(Hidden);
            tensors[$"{block}.output.LayerNorm.{scale}"] = NdArray.Ones(Hidden);
            tensors[$"{block}.output.LayerNorm.{shift}"] = NdArray.Zeros(Hidden);
        }

        var path = Path.Combine(_directory, $"{Guid.NewGuid():N}.safetensors");
        SafeTensors.Write(path, tensors, SafeTensorDType.F64);
        return path;
    }

    [Fact]
    public void LoadsEveryParameterTheEncoderNeeds()
    {
        using var weights = WeightStoreFor(BuildCheckpoint());
        var (model, report) = CheckpointLoader.Load(Config, weights);

        Assert.True(report.IsComplete, report.ToString());
        Assert.Equal("bert.", report.Prefix);
        Assert.True(model.HasPretrainedWeights);
    }

    [Fact]
    public void DetectsAnUnprefixedCheckpoint()
    {
        using var weights = WeightStoreFor(BuildCheckpoint(prefix: ""));
        var (_, report) = CheckpointLoader.Load(Config, weights);

        Assert.True(report.IsComplete, report.ToString());
        Assert.Equal("", report.Prefix);
    }

    [Fact]
    public void AcceptsTheLegacyGammaAndBetaSpelling()
    {
        // bert-base-uncased itself is still published this way, from its TensorFlow origins.
        using var weights = WeightStoreFor(BuildCheckpoint(legacyNormNames: true));
        var (_, report) = CheckpointLoader.Load(Config, weights);

        Assert.True(report.IsComplete, report.ToString());
    }

    [Fact]
    public void TheLoadedModelProducesFiniteHiddenStates()
    {
        using var weights = WeightStoreFor(BuildCheckpoint());
        var (model, _) = CheckpointLoader.Load(Config, weights);

        var hidden = model.Forward([1, 2, 3, 4], [1, 1, 1, 1]);

        Assert.Equal([4, Hidden], hidden.Shape.ToArray());
        Assert.All(hidden.ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }

    [Fact]
    public void ChangingALaterTokenLeavesNothingUndefined()
    {
        using var weights = WeightStoreFor(BuildCheckpoint());
        var (model, _) = CheckpointLoader.Load(Config, weights);

        var first = model.Forward([1, 2, 3, 4], [1, 1, 1, 1]);
        var second = model.Forward([1, 2, 3, 5], [1, 1, 1, 1]);

        // An encoder is bidirectional, so every position may move. What must not happen is the
        // output being identical, which would mean the token embeddings were never consulted.
        Assert.NotEqual(first.ToArray(), second.ToArray());
    }

    [Fact]
    public void AMissingParameterIsReportedWithTheNameItLookedFor()
    {
        var path = BuildCheckpoint();

        // Remove one tensor by rewriting the file without it.
        var tensors = SafeTensors.ReadAll(path);
        tensors.Remove("bert.encoder.layer.1.output.dense.weight");

        var truncated = Path.Combine(_directory, "truncated.safetensors");
        SafeTensors.Write(truncated, tensors, SafeTensorDType.F64);

        using var weights = WeightStoreFor(truncated);

        var exception = Assert.Throws<InvalidDataException>(() => CheckpointLoader.Load(Config, weights));
        Assert.Contains("bert.encoder.layer.1.output.dense.weight", exception.Message);
    }

    [Fact]
    public void LenientModeReportsRatherThanThrows()
    {
        var path = BuildCheckpoint();
        var tensors = SafeTensors.ReadAll(path);
        tensors.Remove("bert.encoder.layer.0.intermediate.dense.bias");

        var truncated = Path.Combine(_directory, "lenient.safetensors");
        SafeTensors.Write(truncated, tensors, SafeTensorDType.F64);

        using var weights = WeightStoreFor(truncated);
        var (model, report) = CheckpointLoader.Load(Config, weights, strict: false);

        Assert.False(report.IsComplete);
        Assert.Contains("bert.encoder.layer.0.intermediate.dense.bias", report.Missing);

        // A partial load must not be marked as pretrained - the flag is what downstream code uses
        // to decide whether the model's output means anything.
        Assert.False(model.HasPretrainedWeights);
    }

    [Fact]
    public void TokenTypeEmbeddingsAreFoldedIntoTheWordEmbeddings()
    {
        var path = BuildCheckpoint();
        var original = SafeTensors.ReadAll(path);

        using var weights = WeightStoreFor(path);
        var (model, _) = CheckpointLoader.Load(Config, weights);

        var words = original["bert.embeddings.word_embeddings.weight"];
        var types = original["bert.embeddings.token_type_embeddings.weight"];

        for (var d = 0; d < Hidden; d++)
        {
            var expected = words[0, d] + types[0, d];
            Assert.True(Math.Abs(model.TokenEmbeddings[0, d] - expected) < 1e-12,
                $"dimension {d}: {model.TokenEmbeddings[0, d]} vs {expected}");
        }
    }

    private static WeightStore WeightStoreFor(string path)
    {
        var directory = Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileNameWithoutExtension(path));
        Directory.CreateDirectory(directory);

        var target = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(target)) File.Copy(path, target);

        return WeightStore.Open(directory);
    }
}

/// <summary>Tests for the numerics shared by the task heads.</summary>
public sealed class SoftmaxTests
{
    [Fact]
    public void SumsToOne()
    {
        var probabilities = TransformerModel.Softmax([1.0, 2.0, 3.0]);

        Assert.True(Math.Abs(probabilities.Sum() - 1.0) < 1e-12);
    }

    [Fact]
    public void PreservesOrder()
    {
        var probabilities = TransformerModel.Softmax([0.5, 3.0, -2.0]);

        Assert.True(probabilities[1] > probabilities[0]);
        Assert.True(probabilities[0] > probabilities[2]);
    }

    [Fact]
    public void SurvivesLargeLogits()
    {
        // exp(900) overflows a double. A real model's logits reach the high tens, and an
        // implementation without the max subtraction returns NaN for a model that is working.
        var probabilities = TransformerModel.Softmax([900.0, 899.0, 1.0]);

        Assert.All(probabilities, p => Assert.False(double.IsNaN(p)));
        Assert.True(Math.Abs(probabilities.Sum() - 1.0) < 1e-12);
        Assert.True(probabilities[0] > probabilities[1]);
    }

    [Fact]
    public void IsShiftInvariant()
    {
        var first = TransformerModel.Softmax([1.0, 2.0, 3.0]);
        var second = TransformerModel.Softmax([101.0, 102.0, 103.0]);

        for (var i = 0; i < first.Length; i++)
        {
            Assert.True(Math.Abs(first[i] - second[i]) < 1e-12);
        }
    }
}
