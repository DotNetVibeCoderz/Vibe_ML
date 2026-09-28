using Gravicode.HFNet.GraviPEFT;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>
/// A training step packs its examples end to end; that has to be exactly each example's own pass.
/// </summary>
public sealed class PackingTests
{
    private static readonly int[][] Sequences = [[2, 17, 5, 29, 11, 3], [4, 9], [1, 3, 4, 2, 21, 22, 23, 2, 6]];
    private static readonly int[][] Segments = [[0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1]];

    private static LoraEncoder Encoder(Dictionary<(int, Projection), LoraAdapter> adapters)
    {
        var (source, config) = GradientTests.TinyModel(51);
        var random = new GraviRandom(52);
        var delta = new NdArray([.. Enumerable.Range(0, 8).Select(_ => 0.5 * random.Normal(0, 1))], 8);
        return LoraEncoder.Build(source, config, (l, p) => adapters[(l, p)], delta);
    }

    [Fact]
    public void Packed_sequences_come_out_as_each_one_alone()
    {
        // Positions restart, attention stays inside each sequence, and a pair keeps its segments.
        var encoder = Encoder(GradientTests.EveryProjection());

        var packed = encoder.Forward(
            [.. Sequences.SelectMany(s => s)], typeIds: [.. Segments.SelectMany(s => s)], lengths: [.. Sequences.Select(s => s.Length)]);

        var alone = Sequences.Zip(Segments).SelectMany(p => encoder.Forward(p.First, typeIds: p.Second)).ToArray();

        Assert.Equal(alone.Length, packed.Length);
        for (var i = 0; i < alone.Length; i++)
        {
            Assert.True(Math.Abs(alone[i] - packed[i]) < 1e-12, $"[{i}] packed {packed[i]} against alone {alone[i]}");
        }
    }

    [Fact]
    public void A_packed_backward_pass_gives_the_sum_of_the_separate_gradients()
    {
        var adapters = GradientTests.EveryProjection();
        var encoder = Encoder(adapters);
        var random = new GraviRandom(53);
        double[][] upstream = [.. Sequences.Select(s => Enumerable.Range(0, s.Length * 8).Select(_ => random.Normal(0, 1)).ToArray())];

        var separate = new LoraGradients();
        for (var n = 0; n < Sequences.Length; n++)
        {
            var tape = new LoraEncoder.Tape();
            encoder.Forward(Sequences[n], tape, typeIds: Segments[n]);
            encoder.Backward(upstream[n], tape, separate);
        }

        var packed = new LoraGradients();
        var packedTape = new LoraEncoder.Tape();
        encoder.Forward([.. Sequences.SelectMany(s => s)], packedTape,
            typeIds: [.. Segments.SelectMany(s => s)], lengths: [.. Sequences.Select(s => s.Length)]);
        encoder.Backward([.. upstream.SelectMany(u => u)], packedTape, packed);

        foreach (var ((layer, projection), adapter) in adapters)
        {
            var (a1, b1) = separate.For(adapter);
            var (a2, b2) = packed.For(adapter);

            for (var i = 0; i < a1.Length; i++)
            {
                Assert.True(Math.Abs(a1[i] - a2[i]) < 1e-12 * Math.Max(1, Math.Abs(a1[i])), $"layer {layer} {projection} dA[{i}]");
            }

            for (var i = 0; i < b1.Length; i++)
            {
                Assert.True(Math.Abs(b1[i] - b2[i]) < 1e-12 * Math.Max(1, Math.Abs(b1[i])), $"layer {layer} {projection} dB[{i}]");
            }
        }
    }

    [Fact]
    public void Lengths_that_do_not_cover_the_ids_are_refused()
    {
        var encoder = Encoder(GradientTests.EveryProjection());
        Assert.Throws<ArgumentException>(() => encoder.Forward([1, 2, 3, 4], lengths: [2, 1]));
    }
}
