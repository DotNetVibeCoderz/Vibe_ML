using System.ComponentModel;
using System.Text;
using LocalGen.Kernel.Plugins;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Skills;

/// <summary>
/// Lets the model discover and use installed skills.
/// </summary>
/// <remarks>
/// Skills are surfaced through kernel functions rather than being pasted into the system prompt,
/// so a user can install dozens without spending context on skills the conversation never
/// touches: the model sees one line per skill and pulls in the full instructions only on use.
/// </remarks>
public sealed class SkillsPlugin
{
    private readonly SkillStore _store;
    private readonly CodeExecutionPlugin _codeExecution;

    /// <summary>Guards against a template or asset flooding the context window.</summary>
    private const int MaxAssetLength = 32_000;

    public SkillsPlugin(SkillStore store, CodeExecutionPlugin codeExecution)
    {
        _store = store;
        _codeExecution = codeExecution;
    }

    [KernelFunction("list_skills")]
    [Description("Lists the installed skills with a one-line description of each.")]
    public string ListSkills()
    {
        var skills = _store.List().Where(static s => s.Enabled).ToList();

        if (skills.Count == 0)
        {
            return "No skills are installed. Browse and install them from the Skills gallery.";
        }

        var builder = new StringBuilder("Available skills:\n");

        foreach (var skill in skills)
        {
            builder.Append("- ").Append(skill.Name).Append(" (v").Append(skill.Version).Append("): ")
                   .AppendLine(skill.Description);
        }

        builder.AppendLine().AppendLine("Call use_skill(name) to load one's full instructions.");
        return builder.ToString();
    }

    [KernelFunction("use_skill")]
    [Description(
        "Loads a skill's full instructions along with the assets and scripts it bundles. " +
        "Call this before following a skill's procedure.")]
    public string UseSkill([Description("Name of the skill")] string name)
    {
        var skill = _store.Get(name);

        if (skill is null)
        {
            var available = string.Join(", ", _store.List().Select(static s => s.Name));
            return $"Error: no skill named '{name}'. Installed skills: {available}";
        }

        return skill.Enabled
            ? skill.ToPrompt()
            : $"Error: the skill '{name}' is disabled.";
    }

    [KernelFunction("read_skill_asset")]
    [Description("Reads a template or asset bundled with a skill.")]
    public async Task<string> ReadAssetAsync(
        [Description("Name of the skill")] string skill,
        [Description("Asset path relative to the skill, e.g. \"assets/report-template.md\"")] string path,
        CancellationToken cancellationToken = default)
    {
        var loaded = _store.Get(skill);

        if (loaded is null)
        {
            return $"Error: no skill named '{skill}'.";
        }

        // The path comes from the model, so it is resolved and confined to the skill directory.
        var root = Path.GetFullPath(loaded.Path) + Path.DirectorySeparatorChar;
        var resolved = Path.GetFullPath(Path.Combine(loaded.Path, path));

        if (!resolved.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            return $"Error: '{path}' is outside the '{skill}' skill directory.";
        }

        if (!File.Exists(resolved))
        {
            return $"Error: '{path}' does not exist in the '{skill}' skill. " +
                   $"Available: {string.Join(", ", loaded.Assets.Concat(loaded.Scripts))}";
        }

        var content = await File.ReadAllTextAsync(resolved, cancellationToken).ConfigureAwait(false);

        return content.Length <= MaxAssetLength
            ? content
            : content[..MaxAssetLength] + $"\n… [truncated, {content.Length:N0} characters total]";
    }

    [KernelFunction("run_skill_script")]
    [Description(
        "Runs a script bundled with a skill and returns its output. " +
        "The language is inferred from the script's extension.")]
    public async Task<string> RunScriptAsync(
        [Description("Name of the skill")] string skill,
        [Description("Script path relative to the skill, e.g. \"scripts/build-report.py\"")] string script,
        [Description("Arguments passed to the script")] string arguments = "",
        CancellationToken cancellationToken = default)
    {
        var loaded = _store.Get(skill);

        if (loaded is null)
        {
            return $"Error: no skill named '{skill}'.";
        }

        var root = Path.GetFullPath(loaded.Path) + Path.DirectorySeparatorChar;
        var resolved = Path.GetFullPath(Path.Combine(loaded.Path, script));

        if (!resolved.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            return $"Error: '{script}' is outside the '{skill}' skill directory.";
        }

        if (!File.Exists(resolved))
        {
            return $"Error: '{script}' does not exist in the '{skill}' skill. " +
                   $"Available scripts: {string.Join(", ", loaded.Scripts)}";
        }

        var interpreter = Path.GetExtension(resolved).ToLowerInvariant() switch
        {
            ".py" => "python",
            ".js" => "node",
            ".ps1" => OperatingSystem.IsWindows() ? "powershell -NoProfile -File" : "pwsh -NoProfile -File",
            ".sh" => "bash",
            _ => null
        };

        if (interpreter is null)
        {
            return $"Error: '{Path.GetExtension(resolved)}' scripts are not supported. " +
                   "Use .py, .js, .ps1 or .sh.";
        }

        // Delegated to the code execution tool so the same timeout, output limits and process
        // teardown apply to skill scripts as to model-authored code.
        return await _codeExecution
            .RunCommandAsync($"{interpreter} \"{resolved}\" {arguments}".Trim(), cancellationToken)
            .ConfigureAwait(false);
    }

    [KernelFunction("search_skill_gallery")]
    [Description("Searches the online skill gallery for skills that can be installed.")]
    public async Task<string> SearchGalleryAsync(
        [Description("Search terms; leave empty to list everything")] string query = "",
        CancellationToken cancellationToken = default)
    {
        var entries = await _store.SearchGalleryAsync(query, null, cancellationToken).ConfigureAwait(false);

        if (entries.Count == 0)
        {
            return string.IsNullOrWhiteSpace(query)
                ? "The skill gallery is empty or unreachable."
                : $"No gallery skills matched '{query}'.";
        }

        var builder = new StringBuilder("Skills in the gallery:\n");

        foreach (var entry in entries)
        {
            builder.Append("- ").Append(entry.Name).Append(" (v").Append(entry.Version).Append(')')
                   .Append(entry.Installed ? " [installed]" : string.Empty)
                   .Append(": ").AppendLine(entry.Description);
        }

        return builder.ToString();
    }
}
