using Gravicode.HFNet.GraviDiffusers;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>The generator behind diffusers' ONNX latents, against NumPy itself.</summary>
public sealed class NumpyRandomTests
{
    [Fact]
    public void Random_sample_matches_numpy_bit_for_bit()
    {
        // np.random.RandomState(42).random_sample(3)
        var random = new NumpyRandom(42);
        Assert.Equal([0.3745401188473625, 0.9507143064099162, 0.7319939418114051], [random.NextDouble(), random.NextDouble(), random.NextDouble()]);
    }

    [Theory]
    [InlineData(42u, new[] { 0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254, -0.23415337472333597 })]
    [InlineData(0u, new[] { 1.764052345967664, 0.4001572083672233, 0.9787379841057392 })]
    public void Randn_matches_numpy_bit_for_bit(uint seed, double[] expected)
    {
        // np.random.RandomState(seed).randn(n): the cached second value of each polar pair comes
        // out on the next call, so an odd count checks the cache too.
        Assert.Equal(expected, new NumpyRandom(seed).Normal(expected.Length));
    }
}
