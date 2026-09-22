using Gravicode.HFNet.GraviPEFT;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>Tests for the LoRA adapter's arithmetic.</summary>
public sealed class LoraAdapterTests
{
    [Fact]
    public void StartsAsExactlyNoChange()
    {
        // The defining property: B is zero, so an adapted model is bit-identical to the base model
        // before any training. Random initialisation of both matrices trains to a worse place.
        var adapter = new LoraAdapter(inputs: 16, outputs: 8, new LoraConfig(Rank: 4));

        Assert.All(adapter.B.ToArray(), v => Assert.Equal(0.0, v));
        Assert.All(adapter.Delta().ToArray(), v => Assert.Equal(0.0, v));
    }

    [Fact]
    public void AIsNotZeroOrTheGradientWouldNeverFlow()
    {
        var adapter = new LoraAdapter(inputs: 16, outputs: 8, new LoraConfig(Rank: 4));

        Assert.Contains(adapter.A.ToArray(), v => v != 0.0);
    }

    [Fact]
    public void DeltaIsTheScaledLowRankProduct()
    {
        var config = new LoraConfig(Rank: 2, Alpha: 4);

        var a = new NdArray([1, 2, 3, 4, 5, 6], 2, 3);   // [rank, inputs]
        var b = new NdArray([1, 0, 0, 1], 2, 2);         // [outputs, rank]

        var delta = LoraAdapter.FromMatrices(a, b, config).Delta();

        // scaling = alpha / rank = 2. B is the identity, so delta is 2 * A.
        Assert.Equal([2, 3], delta.Shape.ToArray());
        Assert.Equal([2, 4, 6, 8, 10, 12], delta.ToArray());
    }

    [Fact]
    public void ScalingIsAlphaOverRank()
    {
        Assert.Equal(2.0, new LoraConfig(Rank: 8, Alpha: 16).Scaling);
        Assert.Equal(1.0, new LoraConfig(Rank: 16, Alpha: 16).Scaling);
    }

    [Fact]
    public void MergeAddsTheDeltaToTheWeight()
    {
        // Non-square on purpose. A square weight matches both orientations on shape alone, so it
        // cannot distinguish a correct merge from a transposed one - which is the same reason the
        // checkpoint loader settles its transpose convention on the feed-forward weight.
        var config = new LoraConfig(Rank: 1, Alpha: 1);

        var a = new NdArray([1, 1, 1], 1, 3);   // [rank, inputs] with inputs = 3
        var b = new NdArray([2, 3], 2, 1);      // [outputs, rank] with outputs = 2

        var weight = NdArray.Zeros(2, 3);       // [outputs, inputs]
        LoraAdapter.FromMatrices(a, b, config).MergeInto(weight);

        // delta = B A = [[2,2,2],[3,3,3]], scaled by alpha/rank = 1.
        Assert.Equal([2, 2, 2, 3, 3, 3], weight.ToArray());
    }

    [Fact]
    public void MergeHandlesTheTransposedWeightLayout()
    {
        // The encoder stores [inputs, outputs] while the checkpoint format is [outputs, inputs];
        // the adapter has to recognise which one it was handed.
        var config = new LoraConfig(Rank: 1, Alpha: 1);

        var a = new NdArray([1, 0, 0], 1, 3);   // inputs = 3
        var b = new NdArray([5, 7], 2, 1);      // outputs = 2

        var transposed = NdArray.Zeros(3, 2);   // [inputs, outputs]
        LoraAdapter.FromMatrices(a, b, config).MergeInto(transposed);

        Assert.Equal(5.0, transposed[0, 0]);
        Assert.Equal(7.0, transposed[0, 1]);
        Assert.Equal(0.0, transposed[1, 0]);
    }

    [Fact]
    public void MergeRejectsAWeightOfTheWrongShape()
    {
        var adapter = new LoraAdapter(inputs: 4, outputs: 4, new LoraConfig(Rank: 2));

        var exception = Assert.Throws<ArgumentException>(() => adapter.MergeInto(NdArray.Zeros(5, 3)));
        Assert.Contains("4x4", exception.Message);
    }

    [Fact]
    public void ParameterCountIsFarBelowTheFullWeight()
    {
        // 768x768 is 589,824 values; rank 8 is 2 x 8 x 768 = 12,288, about 2%.
        var adapter = new LoraAdapter(inputs: 768, outputs: 768, new LoraConfig(Rank: 8));

        Assert.Equal(12_288, adapter.ParameterCount);
        Assert.True(adapter.ParameterCount < 768L * 768 / 40);
    }

    [Fact]
    public void DefaultTargetsAreQueryAndValue()
        => Assert.Equal(["query", "value"], new LoraConfig().Targets);
}

/// <summary>Tests for reading and writing the Hugging Face PEFT adapter layout.</summary>
public sealed class AdapterSetTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-peft", Guid.NewGuid().ToString("N"));

    public AdapterSetTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private static LoraAdapterSet BuildSet(LoraConfig? config = null)
    {
        config ??= new LoraConfig(Rank: 2, Alpha: 4);

        var adapters = new Dictionary<string, LoraAdapter>
        {
            ["bert.encoder.layer.0.attention.self.query"] = new(8, 8, config, seed: 1),
            ["bert.encoder.layer.0.attention.self.value"] = new(8, 8, config, seed: 2),
        };

        return new LoraAdapterSet(adapters, config);
    }

    [Fact]
    public void RoundTripsThroughThePeftLayout()
    {
        var original = BuildSet();
        original.Save(_directory, "bert-base-uncased");

        Assert.True(File.Exists(Path.Combine(_directory, "adapter_model.safetensors")));
        Assert.True(File.Exists(Path.Combine(_directory, "adapter_config.json")));

        var reloaded = LoraAdapterSet.Load(
            Path.Combine(_directory, "adapter_model.safetensors"), original.Config);

        Assert.Equal(original.Adapters.Count, reloaded.Adapters.Count);

        // Compared with a tolerance, not for equality: the PEFT layout stores fp32, so a round
        // trip through it necessarily loses the tail of a double.
        foreach (var (name, adapter) in original.Adapters)
        {
            Assert.True(reloaded.Adapters.ContainsKey(name), $"missing {name}");
            AssertClose(adapter.A.ToArray(), reloaded.Adapters[name].A.ToArray());
            AssertClose(adapter.B.ToArray(), reloaded.Adapters[name].B.ToArray());
        }

        static void AssertClose(double[] expected, double[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-6,
                    $"element {i}: {actual[i]} vs {expected[i]}");
            }
        }
    }

    [Fact]
    public void WritesTheNamesPeftExpects()
    {
        BuildSet().Save(_directory);

        var names = Gravicode.HFNet.GraviHub.Io.SafeTensors
            .Inspect(Path.Combine(_directory, "adapter_model.safetensors"))
            .Select(t => t.Name)
            .ToList();

        Assert.Contains(
            "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight", names);
        Assert.Contains(
            "base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight", names);
    }

    [Fact]
    public void StripsThePeftPrefixWhenReading()
    {
        BuildSet().Save(_directory);

        var reloaded = LoraAdapterSet.Load(Path.Combine(_directory, "adapter_model.safetensors"));

        Assert.All(reloaded.Adapters.Keys, name =>
        {
            Assert.DoesNotContain("base_model", name);
            Assert.DoesNotContain("lora_", name);
        });
    }

    [Fact]
    public void RefusesAnAdapterMissingHalfItsPair()
    {
        // Applying a half pair would add a zero update that looks like a working adapter doing
        // nothing at all.
        var tensors = new Dictionary<string, NdArray>
        {
            ["base_model.model.layer.0.query.lora_A.weight"] = NdArray.Ones(2, 8),
        };

        var path = Path.Combine(_directory, "broken.safetensors");
        Gravicode.HFNet.GraviHub.Io.SafeTensors.Write(path, tensors);

        var exception = Assert.Throws<InvalidDataException>(() => LoraAdapterSet.Load(path));
        Assert.Contains("incomplete", exception.Message);
    }

    [Fact]
    public void ParameterCountIsTheSumOfItsAdapters()
    {
        var set = BuildSet();

        Assert.Equal(set.Adapters.Values.Sum(a => a.ParameterCount), set.ParameterCount);
    }

    [Fact]
    public void ConfigRoundTripsThroughAdapterConfigJson()
    {
        var config = new LoraConfig(Rank: 16, Alpha: 32, TargetModules: ["query", "key"]);
        BuildSet(config).Save(_directory, "some-model");

        var json = File.ReadAllText(Path.Combine(_directory, "adapter_config.json"));

        Assert.Contains("\"r\": 16", json);
        Assert.Contains("\"lora_alpha\": 32", json);
        Assert.Contains("some-model", json);
    }
}
