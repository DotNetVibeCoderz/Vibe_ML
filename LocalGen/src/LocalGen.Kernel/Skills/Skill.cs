using System.Text;

namespace LocalGen.Kernel.Skills;

/// <summary>
/// A packaged capability the model can load on demand.
/// </summary>
/// <remarks>
/// A skill is a directory containing <c>SKILL.md</c> — YAML-ish frontmatter plus instructions —
/// and optionally <c>assets/</c> and <c>scripts/</c>. Instructions alone would only be a prompt
/// snippet; bundling templates and runnable scripts is what lets a skill carry real behaviour,
/// such as a report template plus the script that fills it in.
/// </remarks>
public sealed record Skill
{
    /// <summary>Directory name, used as the identifier.</summary>
    public required string Name { get; init; }

    /// <summary>One-line summary shown in the gallery and used to decide relevance.</summary>
    public required string Description { get; init; }

    public string Version { get; init; } = "1.0.0";

    public string Author { get; init; } = string.Empty;

    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>The instruction body of <c>SKILL.md</c>, everything after the frontmatter.</summary>
    public required string Instructions { get; init; }

    /// <summary>Absolute path to the skill directory.</summary>
    public required string Path { get; init; }

    /// <summary>Relative paths of the bundled assets, such as templates.</summary>
    public IReadOnlyList<string> Assets { get; init; } = [];

    /// <summary>Relative paths of the runnable scripts.</summary>
    public IReadOnlyList<string> Scripts { get; init; } = [];

    public bool Enabled { get; init; } = true;

    /// <summary>Renders the inventory the model sees when it activates the skill.</summary>
    public string ToPrompt()
    {
        var builder = new StringBuilder();
        builder.Append("# Skill: ").AppendLine(Name);
        builder.AppendLine(Description).AppendLine();
        builder.AppendLine(Instructions);

        if (Assets.Count > 0)
        {
            builder.AppendLine().AppendLine("## Bundled assets");
            builder.AppendLine("Read these with read_skill_asset(skill, path):");
            foreach (var asset in Assets)
            {
                builder.Append("- ").AppendLine(asset);
            }
        }

        if (Scripts.Count > 0)
        {
            builder.AppendLine().AppendLine("## Bundled scripts");
            builder.AppendLine("Run these with run_skill_script(skill, script, arguments):");
            foreach (var script in Scripts)
            {
                builder.Append("- ").AppendLine(script);
            }
        }

        return builder.ToString();
    }
}

/// <summary>
/// Parses <c>SKILL.md</c>: a <c>---</c>-delimited frontmatter block followed by markdown.
/// </summary>
/// <remarks>
/// A hand-rolled reader rather than a YAML dependency — the frontmatter is a flat set of scalars
/// and inline lists, and skills are user-authored files where a clear error beats full YAML.
/// </remarks>
public static class SkillManifestParser
{
    public static (IReadOnlyDictionary<string, string> Frontmatter, string Body) Parse(string content)
    {
        var normalized = content.Replace("\r\n", "\n");
        var frontmatter = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!normalized.StartsWith("---\n", StringComparison.Ordinal))
        {
            return (frontmatter, normalized.Trim());
        }

        var end = normalized.IndexOf("\n---", 3, StringComparison.Ordinal);
        if (end < 0)
        {
            // Unterminated frontmatter: treat the whole file as instructions rather than failing.
            return (frontmatter, normalized.Trim());
        }

        var block = normalized[4..end];
        var body = normalized[(end + 4)..].TrimStart('\n', '-').Trim();

        string? currentKey = null;
        var listValues = new List<string>();

        foreach (var rawLine in block.Split('\n'))
        {
            var line = rawLine.TrimEnd();

            if (line.Length == 0 || line.TrimStart().StartsWith('#'))
            {
                continue;
            }

            // A "- item" line continues the list started by the previous key.
            if (line.TrimStart().StartsWith("- ", StringComparison.Ordinal) && currentKey is not null)
            {
                listValues.Add(line.TrimStart()[2..].Trim().Trim('"', '\''));
                frontmatter[currentKey] = string.Join(", ", listValues);
                continue;
            }

            var separator = line.IndexOf(':');
            if (separator <= 0)
            {
                continue;
            }

            currentKey = line[..separator].Trim();
            listValues.Clear();

            var value = line[(separator + 1)..].Trim();

            // Inline list form: tags: [a, b, c]
            if (value.StartsWith('[') && value.EndsWith(']'))
            {
                value = value[1..^1];
            }

            frontmatter[currentKey] = value.Trim().Trim('"', '\'');
        }

        return (frontmatter, body);
    }
}
