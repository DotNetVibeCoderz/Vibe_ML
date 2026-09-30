using System.Globalization;
using System.Text.RegularExpressions;

namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>
/// Emphasis in a prompt - <c>(word)</c>, <c>[word]</c>, <c>(word:1.4)</c> - parsed into weighted
/// pieces of text.
/// </summary>
/// <remarks>
/// <para>
/// The syntax and the arithmetic are AUTOMATIC1111's <c>parse_prompt_attention</c>, which most
/// Stable Diffusion prompts found in the wild are written for: round brackets multiply by 1.1,
/// square brackets divide by it, <c>(text:w)</c> multiplies by <c>w</c>, brackets nest, a backslash
/// escapes a bracket, and an unclosed bracket applies to the rest of the prompt. Neighbouring pieces
/// that end up with the same weight are joined.
/// </para>
/// <para>
/// The weights are applied to the text encoder's output one token at a time and the result is
/// rescaled so its mean is unchanged, again as that implementation does - see
/// <see cref="DiffusionPipeline.PromptWeighting"/>.
/// </para>
/// </remarks>
public static partial class PromptWeights
{
    private const double RoundBracket = 1.1;
    private const double SquareBracket = 1 / 1.1;

    [GeneratedRegex(@"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:")]
    private static partial Regex Attention();

    /// <summary>Splits a prompt into pieces of text and the weight each carries.</summary>
    /// <param name="prompt">A prompt using the emphasis syntax.</param>
    public static IReadOnlyList<(string Text, double Weight)> Parse(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);

        var pieces = new List<(string Text, double Weight)>();
        var round = new Stack<int>();
        var square = new Stack<int>();

        void Multiply(int start, double multiplier)
        {
            for (var p = start; p < pieces.Count; p++) pieces[p] = (pieces[p].Text, pieces[p].Weight * multiplier);
        }

        foreach (Match match in Attention().Matches(prompt))
        {
            var text = match.Value;
            var weight = match.Groups[1];

            if (text.StartsWith('\\')) pieces.Add((text[1..], 1.0));
            else if (text == "(") round.Push(pieces.Count);
            else if (text == "[") square.Push(pieces.Count);
            else if (weight.Success && round.Count > 0) Multiply(round.Pop(), double.Parse(weight.Value, CultureInfo.InvariantCulture));
            else if (text == ")" && round.Count > 0) Multiply(round.Pop(), RoundBracket);
            else if (text == "]" && square.Count > 0) Multiply(square.Pop(), SquareBracket);
            else pieces.Add((text, 1.0));
        }

        foreach (var start in round) Multiply(start, RoundBracket);
        foreach (var start in square) Multiply(start, SquareBracket);

        if (pieces.Count == 0) pieces.Add(("", 1.0));

        for (var i = 0; i + 1 < pieces.Count;)
        {
            if (pieces[i].Weight == pieces[i + 1].Weight)
            {
                pieces[i] = (pieces[i].Text + pieces[i + 1].Text, pieces[i].Weight);
                pieces.RemoveAt(i + 1);
            }
            else
            {
                i++;
            }
        }

        return pieces;
    }

    /// <summary>Whether a prompt uses any emphasis at all.</summary>
    public static bool HasEmphasis(string prompt)
    {
        var pieces = Parse(prompt);
        return pieces.Count > 1 || pieces[0].Weight != 1.0 || pieces[0].Text != prompt;
    }
}
