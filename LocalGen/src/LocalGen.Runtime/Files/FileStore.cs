using System.Security.Cryptography;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Runtime.Files;

/// <summary>A file held by the store.</summary>
public sealed record StoredFile
{
    /// <summary>Opaque identifier, and the name the file is stored under.</summary>
    public required string Id { get; init; }

    /// <summary>The name the user gave it, kept for display.</summary>
    public required string FileName { get; init; }

    public required string MediaType { get; init; }

    public long SizeBytes { get; init; }

    public DateTimeOffset UploadedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Absolute path on disk.</summary>
    public required string Path { get; init; }

    /// <summary>Relative URL the server serves this file from.</summary>
    public string RelativeUrl => $"/api/files/{Id}";

    public bool IsImage => MediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase);

    public bool IsVideo => MediaType.StartsWith("video/", StringComparison.OrdinalIgnoreCase);

    public bool IsAudio => MediaType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase);
}

/// <summary>
/// Stores files attached to a conversation and hands back a URL for each.
/// </summary>
/// <remarks>
/// Attachments need a stable address for two reasons: a transcript renders an image by URL, and
/// the OpenAI wire format carries images as <c>image_url</c> parts. Files are content-addressed by
/// hash, so attaching the same image to five messages stores it once.
/// </remarks>
public sealed class FileStore
{
    /// <summary>Ceiling on one upload. Attachments are context, not a file transfer service.</summary>
    public const long MaxFileSizeBytes = 64L * 1024 * 1024;

    private readonly LocalGenOptions _options;
    private readonly ILogger<FileStore> _logger;

    public FileStore(IOptions<LocalGenOptions> options, ILogger<FileStore> logger)
    {
        _options = options.Value;
        _logger = logger;

        Directory.CreateDirectory(Root);
    }

    /// <summary>Directory uploads are kept in.</summary>
    public string Root => Path.Combine(_options.DataDirectory, "uploads");

    /// <summary>Absolute base address files are served from, for building full URLs.</summary>
    public string BaseUrl => _options.Server.BaseUrl;

    /// <summary>Stores a stream and returns its handle.</summary>
    public async Task<StoredFile> SaveAsync(
        Stream content,
        string fileName,
        string? mediaType = null,
        CancellationToken cancellationToken = default)
    {
        var safeName = Path.GetFileName(fileName);

        if (string.IsNullOrWhiteSpace(safeName))
        {
            throw new LocalGenException("A file name is required.");
        }

        // Buffered so the content can be hashed and then written; uploads are capped well below
        // the size where holding one in memory would matter.
        using var buffer = new MemoryStream();
        await CopyWithLimitAsync(content, buffer, cancellationToken).ConfigureAwait(false);

        buffer.Position = 0;
        var hash = Convert.ToHexString(await SHA256.HashDataAsync(buffer, cancellationToken)
            .ConfigureAwait(false))[..24].ToLowerInvariant();

        var extension = Path.GetExtension(safeName).ToLowerInvariant();
        var id = $"{hash}{extension}";
        var path = Path.Combine(Root, id);

        // Content-addressed: the same bytes are already on disk under the same name.
        if (!File.Exists(path))
        {
            buffer.Position = 0;

            await using var target = File.Create(path);
            await buffer.CopyToAsync(target, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Stored {FileName} as {Id} ({Bytes:N0} bytes)", safeName, id, buffer.Length);
        }

        return new StoredFile
        {
            Id = id,
            FileName = safeName,
            // A client that does not know the type sends the generic fallback, so that is
            // treated as "unknown" rather than as an answer — otherwise an uploaded PNG would
            // be recorded as a binary blob and never render as an image.
            MediaType = IsSpecific(mediaType) ? mediaType! : GuessMediaType(extension),
            SizeBytes = buffer.Length,
            Path = path
        };
    }

    public async Task<StoredFile> SaveAsync(
        string sourcePath,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(sourcePath))
        {
            throw new LocalGenException($"No file at '{sourcePath}'.");
        }

        await using var stream = File.OpenRead(sourcePath);
        return await SaveAsync(stream, Path.GetFileName(sourcePath), null, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>Resolves a stored file by id, or null when it is not there.</summary>
    public StoredFile? Get(string id)
    {
        // The id becomes a path segment, so anything that could escape the directory is rejected.
        if (string.IsNullOrWhiteSpace(id) ||
            id.Contains('/') || id.Contains('\\') || id.Contains(".."))
        {
            return null;
        }

        var path = Path.Combine(Root, id);

        if (!File.Exists(path))
        {
            return null;
        }

        var info = new FileInfo(path);

        return new StoredFile
        {
            Id = id,
            FileName = id,
            MediaType = GuessMediaType(info.Extension.ToLowerInvariant()),
            SizeBytes = info.Length,
            UploadedAt = info.CreationTimeUtc,
            Path = path
        };
    }

    public async Task<byte[]> ReadAsync(string id, CancellationToken cancellationToken = default)
    {
        var file = Get(id) ?? throw new LocalGenException($"No stored file '{id}'.");
        return await File.ReadAllBytesAsync(file.Path, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Copies with a hard ceiling, so a lying Content-Length cannot fill the disk.</summary>
    private static async Task CopyWithLimitAsync(
        Stream source,
        Stream destination,
        CancellationToken cancellationToken)
    {
        var buffer = new byte[81920];
        long total = 0;
        int read;

        while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            total += read;

            if (total > MaxFileSizeBytes)
            {
                throw new LocalGenException(
                    $"The file is larger than the {MaxFileSizeBytes / 1024 / 1024} MB attachment limit.");
            }

            await destination.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Whether a client-supplied media type actually says anything.</summary>
    private static bool IsSpecific(string? mediaType) =>
        !string.IsNullOrWhiteSpace(mediaType) &&
        !mediaType.Equals("application/octet-stream", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Maps an extension to a media type. Deliberately a small table rather than a platform
    /// lookup: the answer has to be the same on every OS, because it travels in the wire format.
    /// </summary>
    public static string GuessMediaType(string extension) => extension.ToLowerInvariant() switch
    {
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".svg" => "image/svg+xml",
        ".tiff" or ".tif" => "image/tiff",

        ".mp4" => "video/mp4",
        ".webm" => "video/webm",
        ".mov" => "video/quicktime",
        ".mkv" => "video/x-matroska",

        ".mp3" => "audio/mpeg",
        ".wav" => "audio/wav",
        ".ogg" => "audio/ogg",
        ".m4a" => "audio/mp4",
        ".flac" => "audio/flac",

        ".pdf" => "application/pdf",
        ".json" => "application/json",
        ".xml" => "application/xml",
        ".zip" => "application/zip",
        ".md" or ".markdown" => "text/markdown",
        ".csv" => "text/csv",
        ".html" or ".htm" => "text/html",
        ".txt" or ".log" => "text/plain",

        _ => "application/octet-stream"
    };
}
