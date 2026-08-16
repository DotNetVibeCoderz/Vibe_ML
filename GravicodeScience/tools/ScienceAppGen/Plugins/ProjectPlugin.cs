using System.ComponentModel;
using System.Text;
using Microsoft.SemanticKernel;
using ScienceAppGen.Services;

namespace ScienceAppGen.Plugins;

/// <summary>
/// The tools Jack uses to actually build an application: project creation, file access and the
/// dotnet CLI.
/// </summary>
/// <remarks>
/// Descriptions here are prompt text, not documentation - the model chooses a function from them,
/// so each one says what the tool does *and* when to reach for it. Every path goes through
/// <see cref="ProjectService.ResolvePath"/>, which refuses anything outside the open project.
/// </remarks>
public sealed class ProjectPlugin(ProjectService projects, LogService logs)
{
    /// <summary>Raised after any change to the file tree, so the explorer can refresh.</summary>
    public event Action? FilesChanged;

    [KernelFunction("CreateProject")]
    [Description("Creates a new .NET project from a template and opens it. Call this first when " +
                 "the user asks for a new application. Returns the files that were created.")]
    public string CreateProject(
        [Description("Project name. Becomes the folder name, assembly name and root namespace.")]
        string name,
        [Description("Template id: blank, numerics, dataframe, ml-pipeline, clustering, nlp, " +
                     "graph, bayesian, timeseries, notebook. Use 'blank' if unsure.")]
        string template = "blank",
        [Description("Parent directory. Leave empty to use the configured projects root.")]
        string directory = "")
    {
        var chosen = TemplateService.Find(template);
        if (chosen is null)
        {
            var available = string.Join(", ", TemplateService.All.Select(t => t.Id));
            return $"No template named '{template}'. Available: {available}.";
        }

        var safeName = SanitiseName(name);
        var parent = string.IsNullOrWhiteSpace(directory)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "ScienceAppGen")
            : directory;
        var target = Path.Combine(parent, safeName);

        if (Directory.Exists(target) && Directory.EnumerateFileSystemEntries(target).Any())
            return $"'{target}' already exists and is not empty. Pick another name.";

        try
        {
            TemplateService.Create(chosen, target, safeName);
            projects.OpenProject(target);
            logs.Assistant($"Created '{safeName}' from the {chosen.Name} template.");
            FilesChanged?.Invoke();

            var files = string.Join("\n", projects.ListFiles().Select(f => "  " + f));
            return $"Created and opened '{safeName}' at {target} using the '{chosen.Id}' template.\n" +
                   $"Files:\n{files}\n\nBuild it with BuildProject before telling the user it works.";
        }
        catch (Exception ex)
        {
            logs.Error("assistant", $"CreateProject failed: {ex.Message}");
            return $"Failed to create the project: {ex.Message}";
        }
    }

    [KernelFunction("ListTemplates")]
    [Description("Lists the available project templates with what each one produces. Use this " +
                 "when choosing a template or when the user asks what is available.")]
    public string ListTemplates()
    {
        var builder = new StringBuilder();
        foreach (var group in TemplateService.All.GroupBy(t => t.Category))
        {
            builder.AppendLine($"{group.Key}:");
            foreach (var template in group)
                builder.AppendLine($"  {template.Id,-14}{template.Description}");
        }
        return builder.ToString();
    }

    [KernelFunction("WriteFile")]
    [Description("Creates or overwrites a file in the open project. This is how you deliver code - " +
                 "write the file rather than printing it in chat. Directories are created as needed.")]
    public string WriteFile(
        [Description("Path relative to the project root, e.g. Models/Customer.cs")] string path,
        [Description("The complete file content. Not a patch or a fragment.")] string content)
    {
        if (!projects.HasProject) return "No project is open. Call CreateProject or ask the user to open one.";

        try
        {
            projects.WriteFile(path, content);
            FilesChanged?.Invoke();
            return $"Wrote {path} ({content.Split('\n').Length} lines).";
        }
        catch (Exception ex)
        {
            return $"Failed to write {path}: {ex.Message}";
        }
    }

    [KernelFunction("ReadFile")]
    [Description("Reads a file from the open project. Read before editing so you change the real " +
                 "content rather than what you assume is there.")]
    public string ReadFile([Description("Path relative to the project root.")] string path)
    {
        if (!projects.HasProject) return "No project is open.";

        try
        {
            var content = projects.ReadFile(path);
            return content.Length > 60_000
                ? content[..60_000] + "\n... (truncated)"
                : content;
        }
        catch (FileNotFoundException)
        {
            return $"{path} does not exist. Call ListFiles to see what does.";
        }
        catch (Exception ex)
        {
            return $"Failed to read {path}: {ex.Message}";
        }
    }

    [KernelFunction("ListFiles")]
    [Description("Lists the files in the open project, skipping bin, obj and .git. Use it to " +
                 "orient yourself before making changes.")]
    public string ListFiles(
        [Description("Optional subdirectory to list. Leave empty for the whole project.")]
        string subdirectory = "",
        [Description("Optional glob, e.g. *.cs")] string pattern = "*")
    {
        if (!projects.HasProject) return "No project is open.";

        var files = projects.ListFiles(string.IsNullOrWhiteSpace(subdirectory) ? null : subdirectory, pattern);
        return files.Count == 0 ? "No matching files." : string.Join("\n", files);
    }

    [KernelFunction("DeleteFile")]
    [Description("Deletes a file from the open project. Use sparingly and say why.")]
    public string DeleteFile([Description("Path relative to the project root.")] string path)
    {
        if (!projects.HasProject) return "No project is open.";

        try
        {
            projects.DeleteFile(path);
            FilesChanged?.Invoke();
            return $"Deleted {path}.";
        }
        catch (Exception ex)
        {
            return $"Failed to delete {path}: {ex.Message}";
        }
    }

    [KernelFunction("BuildProject")]
    [Description("Builds the open project and returns the compiler output. Always call this after " +
                 "writing code, and fix what it reports before claiming the work is done.")]
    public async Task<string> BuildProjectAsync()
    {
        if (!projects.HasProject) return "No project is open.";

        var result = await projects.BuildAsync();
        return result.Succeeded
            ? "Build succeeded.\n" + Tail(result.Output, 20)
            : $"Build FAILED (exit code {result.ExitCode}). Fix these errors:\n{Tail(result.Output, 60)}";
    }

    [KernelFunction("RunProject")]
    [Description("Runs the open project and returns its output. Use it to show the user that the " +
                 "application actually works.")]
    public async Task<string> RunProjectAsync()
    {
        if (!projects.HasProject) return "No project is open.";

        var build = await projects.BuildAsync();
        if (!build.Succeeded)
            return $"Cannot run: the build failed.\n{Tail(build.Output, 40)}";

        var result = await projects.RunAsync();
        return result.Succeeded
            ? $"Ran successfully:\n{Tail(result.Output, 60)}"
            : $"Run failed (exit code {result.ExitCode}):\n{Tail(result.Output, 60)}";
    }

    [KernelFunction("GetProjectInfo")]
    [Description("Reports which project is open and what it contains. Call this when you need to " +
                 "know the current state before acting.")]
    public string GetProjectInfo()
    {
        if (!projects.HasProject)
            return "No project is open. Use CreateProject to make one.";

        var files = projects.ListFiles();
        return $"""
            Project : {projects.ProjectName}
            Path    : {projects.ProjectRoot}
            Files   : {files.Count}

            {string.Join("\n", files.Take(60))}
            """;
    }

    /// <summary>Keeps only the last <paramref name="lines"/> lines, so tool output stays small.</summary>
    private static string Tail(string text, int lines)
    {
        var split = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        return split.Length <= lines ? text : string.Join('\n', split[^lines..]);
    }

    private static string SanitiseName(string name)
    {
        var cleaned = new string(name
            .Select(c => char.IsLetterOrDigit(c) || c is '.' or '_' or '-' ? c : '_')
            .ToArray()).Trim('_', '.', '-');
        return cleaned.Length == 0 ? "NewProject" : cleaned;
    }
}
