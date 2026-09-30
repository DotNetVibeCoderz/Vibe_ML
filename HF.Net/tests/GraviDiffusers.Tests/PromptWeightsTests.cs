using Gravicode.HFNet.GraviDiffusers;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>The emphasis syntax, against AUTOMATIC1111's <c>parse_prompt_attention</c>.</summary>
/// <remarks>
/// Expected weights are that function's own output, run in Python, down to its floating-point
/// accumulation (1.5730000000000004 is 1.1 x 1.1 x 1.3 multiplied in that order).
/// </remarks>
public sealed class PromptWeightsTests
{
    public static TheoryData<string, (string Text, double Weight)[]> Cases => new()
    {
        { "normal text", [("normal text", 1.0)] },
        { "an (important) word", [("an ", 1.0), ("important", 1.1), (" word", 1.0)] },
        {
            "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky)))",
            [("a ", 1.0), ("house", 1.5730000000000004), (" ", 1.1), ("on", 1.0), (" a ", 1.1), ("hill", 0.55), (", sun, ", 1.1), ("sky", 1.4641000000000006)]
        },
        { "(unbalanced", [("unbalanced", 1.1)] },
        { @"\(literal\) paren", [("(literal) paren", 1.0)] },
        {
            "[[small]] and (big:1.5) (nested (deep))",
            [("small", 0.8264462809917354), (" and ", 1.0), ("big", 1.5), (" ", 1.0), ("nested ", 1.1), ("deep", 1.2100000000000002)]
        },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Emphasis_parses_as_the_reference_does(string prompt, (string Text, double Weight)[] expected)
    {
        var pieces = PromptWeights.Parse(prompt);

        Assert.Equal(expected.Select(e => e.Text), pieces.Select(p => p.Text));
        Assert.Equal(expected.Select(e => e.Weight), pieces.Select(p => p.Weight));
    }

    [Fact]
    public void A_plain_prompt_has_no_emphasis()
    {
        Assert.False(PromptWeights.HasEmphasis("a lighthouse at dawn, oil painting"));
        Assert.True(PromptWeights.HasEmphasis("a (lighthouse) at dawn"));
        Assert.True(PromptWeights.HasEmphasis(@"a \(literal\) bracket"));
    }
}
