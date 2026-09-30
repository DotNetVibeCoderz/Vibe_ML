using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>
/// GraviPEFT: the resources one LoRA fine-tuning epoch takes - time, and with the memory diagnoser,
/// what it allocates.
/// </summary>
/// <remarks>
/// Thirty-two short sentences, packed eight to a step. The trainable share of the encoder is what
/// LoRA is for, so it is printed once in setup: about 0.3% at rank 8 on query, key and value.
/// </remarks>
[BenchmarkCategory("GraviPEFT")]
public class PeftBenchmarks
{
    private static readonly string[] Texts =
    [
        "The food was good.", "The food was not good.", "The staff were helpful.", "The staff were not helpful.",
        "I would recommend it.", "I would not recommend it.", "Delivery was quick.", "Delivery was anything but quick.",
        "The room was clean.", "The room was not clean.", "Setup was easy.", "Setup was not easy.",
        "The price was fair.", "The price was not fair.", "The music was beautiful.", "The music was far from beautiful.",
        "It arrived on time.", "It did not arrive on time.", "The book held my attention.", "The book never held my attention.",
        "We were happy with it.", "We were not happy with it.", "The coffee tasted fresh.", "The coffee did not taste fresh.",
        "The film was fun.", "The film was not fun.", "Everything worked.", "Nothing worked.",
        "The view was lovely.", "The view was not lovely.", "Support replied fast.", "Support never replied.",
    ];

    private static readonly string[] Labels = [.. Texts.Select((_, i) => i % 2 == 0 ? "positive" : "negative")];

    private TransformerModel _model = null!;

    [Params("prajjwal1/bert-tiny", "bert-base-uncased")]
    public string Model { get; set; } = "";

    [GlobalSetup]
    public void Setup()
    {
        _model = TransformerModel.Load(Model);
        var (adapter, encoder, fraction) = PEFT.ApplyLoRA(_model, Config).ParameterEfficiency();
        Console.WriteLine($"// {Model}: {adapter:N0} trainable of {encoder:N0} ({fraction:P2})");
    }

    [GlobalCleanup]
    public void Cleanup() => _model.Dispose();

    private static LoraConfig Config => new(Rank: 8, Alpha: 16, TargetModules: ["query", "key", "value"]);

    [Benchmark(Description = "One LoRA epoch, 32 sentences, batch 8")]
    public double Epoch()
    {
        var peft = PEFT.ApplyLoRA(_model, Config);
        return peft.Train(Texts, Labels, new TrainingOptions { Epochs = 1, BatchSize = 8, LearningRate = 1e-3 }).EpochLosses[0];
    }
}
