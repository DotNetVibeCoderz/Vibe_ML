namespace Gravicode.HFNet.GraviHub;

/// <summary>Which kind of repository a Hub identifier refers to.</summary>
/// <remarks>
/// The Hub serves models, datasets and Spaces from three separate URL namespaces. A dataset and a
/// model may share a name, so the kind is part of the address rather than something to infer.
/// </remarks>
public enum RepoKind
{
    /// <summary>A model repository: <c>huggingface.co/{id}</c>.</summary>
    Model,

    /// <summary>A dataset repository: <c>huggingface.co/datasets/{id}</c>.</summary>
    Dataset,

    /// <summary>A Space: <c>huggingface.co/spaces/{id}</c>.</summary>
    Space,
}

/// <summary>An error returned by the Hub, or by the local cache standing in for it.</summary>
public sealed class HubException : Exception
{
    /// <summary>Creates an exception with a message.</summary>
    public HubException(string message) : base(message) { }

    /// <summary>Creates an exception with a message and an inner cause.</summary>
    public HubException(string message, Exception inner) : base(message, inner) { }

    /// <summary>The HTTP status the Hub replied with, when the failure came from a request.</summary>
    public System.Net.HttpStatusCode? Status { get; init; }

    /// <summary>The repository the request was for, when there was one.</summary>
    public string? RepoId { get; init; }
}

/// <summary>One file inside a Hub repository, as reported by the repository listing.</summary>
/// <param name="Path">Path relative to the repository root, using forward slashes.</param>
/// <param name="Size">Size in bytes, or <c>null</c> when the listing did not report one.</param>
/// <param name="Sha">Git blob SHA, or the LFS OID for a file stored in LFS.</param>
/// <param name="IsLfs">Whether the file is stored through Git LFS rather than in the git tree.</param>
public readonly record struct RepoFile(string Path, long? Size, string? Sha, bool IsLfs)
{
    /// <summary>The file extension, lowercased, including the leading dot.</summary>
    public string Extension => System.IO.Path.GetExtension(Path).ToLowerInvariant();

    /// <inheritdoc />
    public override string ToString()
        => Size is null ? Path : $"{Path} ({Formatting.Bytes(Size.Value)})";
}

/// <summary>Metadata describing a Hub repository.</summary>
/// <param name="Id">The canonical repository id, for example <c>bert-base-uncased</c>.</param>
/// <param name="Kind">Whether this is a model, a dataset or a Space.</param>
/// <param name="Sha">The commit the metadata was read at.</param>
/// <param name="LastModified">When the repository last changed, when reported.</param>
/// <param name="Downloads">Downloads in the last 30 days, when reported.</param>
/// <param name="Likes">Number of likes, when reported.</param>
/// <param name="Tags">Hub tags, including task and library tags.</param>
/// <param name="PipelineTag">The declared pipeline, for example <c>text-classification</c>.</param>
/// <param name="Files">The files in the repository.</param>
/// <param name="Private">Whether the repository is private.</param>
/// <param name="Gated">Whether the repository is gated behind an access request.</param>
public sealed record RepoInfo(
    string Id,
    RepoKind Kind,
    string? Sha,
    DateTimeOffset? LastModified,
    long Downloads,
    long Likes,
    IReadOnlyList<string> Tags,
    string? PipelineTag,
    IReadOnlyList<RepoFile> Files,
    bool Private,
    bool Gated)
{
    /// <summary>Finds a file by exact path, or returns <c>null</c>.</summary>
    public RepoFile? File(string path)
    {
        foreach (var f in Files)
        {
            if (string.Equals(f.Path, path, StringComparison.Ordinal)) return f;
        }
        return null;
    }

    /// <summary>All files whose extension matches, for example <c>".safetensors"</c>.</summary>
    public IReadOnlyList<RepoFile> FilesWithExtension(string extension)
        => [.. Files.Where(f => string.Equals(f.Extension, extension, StringComparison.OrdinalIgnoreCase))];

    /// <summary>Whether the repository carries weights in safetensors form.</summary>
    public bool HasSafeTensors => FilesWithExtension(".safetensors").Count > 0;

    /// <inheritdoc />
    public override string ToString()
        => $"{Kind} {Id} ({Files.Count} files, {Downloads:N0} downloads)";
}

/// <summary>Progress for a single file transfer.</summary>
/// <param name="Path">The repository-relative path being transferred.</param>
/// <param name="BytesTransferred">Bytes moved so far.</param>
/// <param name="TotalBytes">Total size when the server reported one.</param>
public readonly record struct TransferProgress(string Path, long BytesTransferred, long? TotalBytes)
{
    /// <summary>Completed fraction in <c>[0,1]</c>, or <c>null</c> when the total is unknown.</summary>
    public double? Fraction => TotalBytes is > 0 ? (double)BytesTransferred / TotalBytes.Value : null;

    /// <inheritdoc />
    public override string ToString()
        => Fraction is { } f
            ? $"{Path} {f:P0} ({Formatting.Bytes(BytesTransferred)} / {Formatting.Bytes(TotalBytes!.Value)})"
            : $"{Path} {Formatting.Bytes(BytesTransferred)}";
}

/// <summary>Settings for <see cref="HubClient"/>.</summary>
public sealed record HubOptions
{
    /// <summary>Base address of the Hub. Point this at a mirror or a private deployment.</summary>
    public string Endpoint { get; init; } = "https://huggingface.co";

    /// <summary>
    /// Access token. When left null the client reads <c>HF_TOKEN</c>, then
    /// <c>HUGGING_FACE_HUB_TOKEN</c>, from the environment.
    /// </summary>
    /// <remarks>
    /// A token is required for private and gated repositories and raises the anonymous rate limit
    /// everywhere else.
    /// </remarks>
    public string? Token { get; init; }

    /// <summary>Where downloaded files are kept. Defaults to <c>%LOCALAPPDATA%/huggingface/hub</c>.</summary>
    public string? CacheRoot { get; init; }

    /// <summary>How long a single request may take. Large weights need a generous value.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromMinutes(30);

    /// <summary>How many times a failed transfer is retried before it throws.</summary>
    public int MaxRetries { get; init; } = 3;

    /// <summary>
    /// When true a cached file is used without asking the Hub whether it is current. Off by
    /// default: the check is one cheap HEAD against a download that may be gigabytes.
    /// </summary>
    public bool OfflineMode { get; init; }

    /// <summary>Resolves the token to use, consulting the environment when none was set.</summary>
    public string? ResolveToken()
        => Token
           ?? Environment.GetEnvironmentVariable("HF_TOKEN")
           ?? Environment.GetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN");
}

/// <summary>Byte and duration formatting shared by the Hub types' <c>ToString</c>s.</summary>
internal static class Formatting
{
    private static readonly string[] Units = ["B", "KB", "MB", "GB", "TB"];

    /// <summary>Formats a byte count with a binary unit, for example <c>413.7 MB</c>.</summary>
    internal static string Bytes(long bytes)
    {
        double value = bytes;
        var unit = 0;
        while (value >= 1024 && unit < Units.Length - 1)
        {
            value /= 1024;
            unit++;
        }
        return unit == 0 ? $"{bytes} B" : $"{value:0.#} {Units[unit]}";
    }
}
