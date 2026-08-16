using System.IO.Compression;
using System.Text.Json;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Kernel.Skills;

/// <summary>An installable skill listed by the gallery.</summary>
public sealed record SkillGalleryEntry
{
    public required string Name { get; init; }

    public required string Description { get; init; }

    public string Author { get; init; } = string.Empty;

    public string Version { get; init; } = "1.0.0";

    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>URL of the skill's zip archive.</summary>
    public required string DownloadUrl { get; init; }

    public bool Installed { get; init; }
}

/// <summary>
/// Loads, installs and removes skills.
/// </summary>
/// <remarks>
/// Skills live as plain directories under the data folder, so a user can author one by hand or
/// drop in a folder from a colleague. Installation from the gallery is just a zip extraction, and
/// entries are re-read from disk on demand so hand edits show up without a restart.
/// </remarks>
public sealed class SkillStore
{
    /// <summary>File names that mark a directory as a skill.</summary>
    private static readonly string[] ManifestNames = ["SKILL.md", "skill.md"];

    private readonly LocalGenOptions _options;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<SkillStore> _logger;

    public SkillStore(
        IOptions<LocalGenOptions> options,
        IHttpClientFactory httpClientFactory,
        ILogger<SkillStore> logger)
    {
        _options = options.Value;
        _httpClientFactory = httpClientFactory;
        _logger = logger;

        Directory.CreateDirectory(_options.SkillsDirectory);
    }

    public string SkillsDirectory => _options.SkillsDirectory;

    /// <summary>Every skill currently installed.</summary>
    public IReadOnlyList<Skill> List()
    {
        if (!Directory.Exists(_options.SkillsDirectory))
        {
            return [];
        }

        var skills = new List<Skill>();

        foreach (var directory in Directory.EnumerateDirectories(_options.SkillsDirectory))
        {
            var skill = TryLoad(directory);
            if (skill is not null)
            {
                skills.Add(skill);
            }
        }

        return [.. skills.OrderBy(static s => s.Name, StringComparer.OrdinalIgnoreCase)];
    }

    public Skill? Get(string name)
    {
        var directory = Path.Combine(_options.SkillsDirectory, name);
        return Directory.Exists(directory) ? TryLoad(directory) : null;
    }

    /// <summary>Reads a skill directory, returning null when it has no manifest.</summary>
    private Skill? TryLoad(string directory)
    {
        var manifestPath = ManifestNames
            .Select(name => Path.Combine(directory, name))
            .FirstOrDefault(File.Exists);

        if (manifestPath is null)
        {
            return null;
        }

        try
        {
            var (frontmatter, body) = SkillManifestParser.Parse(File.ReadAllText(manifestPath));
            var name = frontmatter.GetValueOrDefault("name", Path.GetFileName(directory));

            return new Skill
            {
                Name = name,
                Description = frontmatter.GetValueOrDefault("description", string.Empty),
                Version = frontmatter.GetValueOrDefault("version", "1.0.0"),
                Author = frontmatter.GetValueOrDefault("author", string.Empty),
                Tags = SplitList(frontmatter.GetValueOrDefault("tags")),
                Instructions = body,
                Path = directory,
                Assets = ListRelative(directory, "assets"),
                Scripts = ListRelative(directory, "scripts"),
                Enabled = !frontmatter.TryGetValue("enabled", out var enabled) ||
                          !string.Equals(enabled, "false", StringComparison.OrdinalIgnoreCase)
            };
        }
        catch (IOException ex)
        {
            _logger.LogWarning(ex, "Could not read the skill at {Directory}", directory);
            return null;
        }
    }

    /// <summary>Installs a skill from a zip archive URL or a local zip path.</summary>
    public async Task<Skill> InstallAsync(
        string source,
        string? name = null,
        CancellationToken cancellationToken = default)
    {
        var archivePath = source;
        var isTemporary = false;

        if (Uri.TryCreate(source, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https")
        {
            if (_options.Runtime.OfflineMode)
            {
                throw new OfflineModeException($"install skill from {source}");
            }

            archivePath = Path.Combine(Path.GetTempPath(), $"localgen-skill-{Guid.NewGuid():N}.zip");
            isTemporary = true;

            using var http = _httpClientFactory.CreateClient(nameof(SkillStore));
            await using var stream = await http.GetStreamAsync(uri, cancellationToken).ConfigureAwait(false);
            await using var file = File.Create(archivePath);
            await stream.CopyToAsync(file, cancellationToken).ConfigureAwait(false);
        }

        if (!File.Exists(archivePath))
        {
            throw new LocalGenException($"No skill archive found at '{source}'.");
        }

        try
        {
            var skillName = name ?? Path.GetFileNameWithoutExtension(archivePath);
            var destination = Path.Combine(_options.SkillsDirectory, skillName);

            if (Directory.Exists(destination))
            {
                Directory.Delete(destination, recursive: true);
            }

            Directory.CreateDirectory(destination);
            ExtractSafely(archivePath, destination);

            // Archives commonly wrap everything in a single top-level folder; unwrap it so the
            // manifest ends up where the loader expects it.
            NormalizeLayout(destination);

            var skill = TryLoad(destination)
                        ?? throw new LocalGenException(
                            $"'{skillName}' was extracted but contains no SKILL.md manifest.");

            _logger.LogInformation("Installed skill '{Skill}' v{Version}", skill.Name, skill.Version);
            return skill;
        }
        finally
        {
            if (isTemporary && File.Exists(archivePath))
            {
                File.Delete(archivePath);
            }
        }
    }

    public bool Remove(string name)
    {
        var directory = Path.Combine(_options.SkillsDirectory, name);

        if (!Directory.Exists(directory))
        {
            return false;
        }

        Directory.Delete(directory, recursive: true);
        _logger.LogInformation("Removed skill '{Skill}'", name);
        return true;
    }

    /// <summary>
    /// Lists skills available from a remote index — a JSON array of gallery entries. Defaults to
    /// the LocalGen community index but accepts any URL, so teams can host their own.
    /// </summary>
    public async Task<IReadOnlyList<SkillGalleryEntry>> SearchGalleryAsync(
        string query = "",
        string? indexUrl = null,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return [];
        }

        indexUrl ??= "https://raw.githubusercontent.com/gravicode/LocalGen/main/skills/index.json";

        try
        {
            using var http = _httpClientFactory.CreateClient(nameof(SkillStore));
            var json = await http.GetStringAsync(indexUrl, cancellationToken).ConfigureAwait(false);

            var entries = JsonSerializer.Deserialize<List<SkillGalleryEntry>>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            }) ?? [];

            var installed = List().Select(static s => s.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);

            return
            [
                .. entries
                    .Where(entry => string.IsNullOrWhiteSpace(query) || Matches(entry, query))
                    .Select(entry => entry with { Installed = installed.Contains(entry.Name) })
            ];
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException)
        {
            _logger.LogWarning(ex, "Could not read the skill gallery at {Url}", indexUrl);
            return [];
        }
    }

    private static bool Matches(SkillGalleryEntry entry, string query) =>
        entry.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
        entry.Description.Contains(query, StringComparison.OrdinalIgnoreCase) ||
        entry.Tags.Any(tag => tag.Contains(query, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Extracts an archive, rejecting entries whose paths escape the destination. Zip files can
    /// carry <c>../</c> paths, and skills are downloaded from the internet.
    /// </summary>
    private static void ExtractSafely(string archivePath, string destination)
    {
        var root = Path.GetFullPath(destination) + Path.DirectorySeparatorChar;

        using var archive = ZipFile.OpenRead(archivePath);

        foreach (var entry in archive.Entries)
        {
            var target = Path.GetFullPath(Path.Combine(destination, entry.FullName));

            if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase))
            {
                throw new LocalGenException(
                    $"The archive contains an entry that escapes the skill directory: '{entry.FullName}'.");
            }

            // A trailing separator marks a directory entry, which has no content to write.
            if (string.IsNullOrEmpty(entry.Name))
            {
                Directory.CreateDirectory(target);
                continue;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            entry.ExtractToFile(target, overwrite: true);
        }
    }

    /// <summary>Flattens a single wrapping folder so the manifest sits at the skill root.</summary>
    private static void NormalizeLayout(string destination)
    {
        if (ManifestNames.Any(name => File.Exists(Path.Combine(destination, name))))
        {
            return;
        }

        var directories = Directory.GetDirectories(destination);
        var files = Directory.GetFiles(destination);

        if (directories.Length != 1 || files.Length != 0)
        {
            return;
        }

        var inner = directories[0];

        foreach (var path in Directory.GetFileSystemEntries(inner))
        {
            var target = Path.Combine(destination, Path.GetFileName(path));

            if (Directory.Exists(path))
            {
                Directory.Move(path, target);
            }
            else
            {
                File.Move(path, target);
            }
        }

        Directory.Delete(inner, recursive: true);
    }

    private static IReadOnlyList<string> ListRelative(string skillDirectory, string subdirectory)
    {
        var path = Path.Combine(skillDirectory, subdirectory);

        if (!Directory.Exists(path))
        {
            return [];
        }

        return
        [
            .. Directory
                .EnumerateFiles(path, "*", SearchOption.AllDirectories)
                .Select(file => Path.GetRelativePath(skillDirectory, file).Replace('\\', '/'))
                .Order()
        ];
    }

    private static IReadOnlyList<string> SplitList(string? value) =>
        string.IsNullOrWhiteSpace(value)
            ? []
            : [.. value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
}
