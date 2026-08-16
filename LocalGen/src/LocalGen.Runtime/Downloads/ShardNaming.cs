using System.Text.RegularExpressions;

namespace LocalGen.Runtime.Downloads;

/// <summary>
/// Understands the naming convention for split GGUF files.
/// </summary>
/// <remarks>
/// A model too large for one file is published as
/// <c>model-00001-of-00003.gguf</c>, <c>model-00002-of-00003.gguf</c>, and so on. llama.cpp opens
/// the first shard and looks for its siblings in the same directory, so downloading only the file
/// a user named would fail at load time with a missing tensor — which is a confusing way to learn
/// that the model has more parts.
/// </remarks>
public static partial class ShardNaming
{
    [GeneratedRegex(@"-(\d{5})-of-(\d{5})\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex ShardSuffix();

    /// <summary>Whether a file name is one shard of a split model.</summary>
    public static bool IsShard(string fileName) => ShardSuffix().IsMatch(fileName);

    /// <summary>
    /// Reads the shard position, or null when the name is not a shard.
    /// </summary>
    public static (int Index, int Total)? Parse(string fileName)
    {
        var match = ShardSuffix().Match(fileName);

        return match.Success
            ? (int.Parse(match.Groups[1].Value), int.Parse(match.Groups[2].Value))
            : null;
    }

    /// <summary>
    /// Returns every file belonging to the same model as <paramref name="fileName"/>, in order.
    /// A name that is not a shard comes back on its own, so callers need no special case.
    /// </summary>
    public static IReadOnlyList<string> Expand(string fileName)
    {
        var match = ShardSuffix().Match(fileName);

        if (!match.Success)
        {
            return [fileName];
        }

        var total = int.Parse(match.Groups[2].Value);

        // A malformed count would otherwise produce an unbounded download list.
        if (total is < 1 or > 999)
        {
            return [fileName];
        }

        var prefix = fileName[..match.Index];
        var results = new List<string>(total);

        for (var i = 1; i <= total; i++)
        {
            results.Add($"{prefix}-{i:D5}-of-{total:D5}.gguf");
        }

        return results;
    }

    /// <summary>
    /// Reduces a listing to one entry per model: the first shard stands for its whole set, and
    /// unsharded files pass through. Used so a catalogue shows one row per model rather than one
    /// per file.
    /// </summary>
    public static IReadOnlyList<string> CollapseShards(IEnumerable<string> fileNames)
    {
        var results = new List<string>();

        foreach (var name in fileNames)
        {
            var shard = Parse(name);

            if (shard is null || shard.Value.Index == 1)
            {
                results.Add(name);
            }
        }

        return results;
    }
}
