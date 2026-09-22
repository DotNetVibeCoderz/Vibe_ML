namespace Gravicode.HFNet.GraviHub;

/// <summary>
/// The one-line entry point to the Hugging Face Hub: <c>Hub.DownloadModel("bert-base-uncased")</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is a facade over a process-wide <see cref="HubClient"/>. It exists because the common case
/// is a script or a notebook cell that wants one model and has no interest in owning a client, a
/// cache or a cancellation token. Anything beyond that - progress reporting, a private endpoint, a
/// second token - should construct <see cref="HubClient"/> directly.
/// </para>
/// <para>
/// The synchronous methods block on the shared client. That is acceptable in a console app or a
/// notebook and is the wrong thing inside a UI or a server: use the <c>Async</c> overloads there,
/// or a client of your own.
/// </para>
/// </remarks>
public static class Hub
{
    private static readonly Lock Gate = new();
    private static HubClient? _shared;
    private static HubOptions _options = new();

    /// <summary>The client the static methods run through, created on first use.</summary>
    public static HubClient Shared
    {
        get
        {
            lock (Gate)
            {
                return _shared ??= new HubClient(_options);
            }
        }
    }

    /// <summary>
    /// Replaces the settings the shared client uses, disposing the existing one.
    /// </summary>
    /// <param name="options">Endpoint, token, cache location and retry policy.</param>
    /// <remarks>
    /// Call this before the first download. Anything already handed out by <see cref="Shared"/>
    /// keeps running against the old client until it is finished with.
    /// </remarks>
    public static void Configure(HubOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        lock (Gate)
        {
            _options = options;
            _shared?.Dispose();
            _shared = null;
        }
    }

    /// <summary>Sets the access token used for subsequent requests.</summary>
    /// <param name="token">A Hub token, or null to fall back to the environment.</param>
    public static void Login(string? token) => Configure(_options with { Token = token });

    /// <summary>The account the current token belongs to, or <c>null</c> when anonymous.</summary>
    public static string? WhoAmI() => Shared.WhoAmIAsync().GetAwaiter().GetResult();

    // ------------------------------------------------------------------ models

    /// <summary>
    /// Downloads a model repository and returns the local directory holding it.
    /// </summary>
    /// <param name="repoId">A model id such as <c>bert-base-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit. Defaults to <c>main</c>.</param>
    /// <param name="weightsOnly">
    /// When true - the default - only the files an inference run needs are fetched: safetensors
    /// weights, the config, and the tokenizer.
    /// </param>
    /// <returns>The directory the files landed in.</returns>
    /// <remarks>
    /// A typical model repository carries the same weights two or three times over: safetensors
    /// beside a PyTorch pickle beside an ONNX export, plus TensorFlow and Flax variants on the
    /// older ones. <paramref name="weightsOnly"/> exists because the honest default is to fetch one
    /// of them rather than all of them - it is routinely the difference between 400 MB and 1.6 GB.
    /// </remarks>
    public static string DownloadModel(string repoId, string revision = "main", bool weightsOnly = true)
        => DownloadModelAsync(repoId, revision, weightsOnly).GetAwaiter().GetResult();

    /// <inheritdoc cref="DownloadModel(string, string, bool)"/>
    /// <param name="repoId">A model id such as <c>bert-base-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="weightsOnly">Whether to fetch only what inference needs.</param>
    /// <param name="progress">Receives per-file transfer progress.</param>
    /// <param name="cancellationToken">Cancels the transfer.</param>
    public static Task<string> DownloadModelAsync(
        string repoId,
        string revision = "main",
        bool weightsOnly = true,
        IProgress<TransferProgress>? progress = null,
        CancellationToken cancellationToken = default)
        => Shared.SnapshotAsync(
            repoId,
            RepoKind.Model,
            revision,
            allowPatterns: weightsOnly ? InferencePatterns : null,
            ignorePatterns: weightsOnly ? null : HeavyDuplicatePatterns,
            progress,
            cancellationToken);

    /// <summary>Downloads a single file from a model repository.</summary>
    /// <param name="repoId">The model id.</param>
    /// <param name="fileName">The repository-relative path.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <returns>The local path of the downloaded file.</returns>
    public static string DownloadFile(string repoId, string fileName, string revision = "main")
        => Shared.DownloadFileAsync(repoId, fileName, RepoKind.Model, revision).GetAwaiter().GetResult();

    /// <summary>Reads a model repository's metadata and file listing.</summary>
    public static RepoInfo ModelInfo(string repoId, string revision = "main")
        => Shared.GetRepoInfoAsync(repoId, RepoKind.Model, revision).GetAwaiter().GetResult();

    /// <summary>Searches the Hub for models.</summary>
    /// <param name="query">Free text matched against model names.</param>
    /// <param name="limit">How many results to return.</param>
    /// <param name="task">An optional pipeline tag such as <c>text-classification</c>.</param>
    public static IReadOnlyList<RepoInfo> SearchModels(string query, int limit = 20, string? task = null)
        => Shared.SearchAsync(query, RepoKind.Model, limit, task).GetAwaiter().GetResult();

    // ------------------------------------------------------------------ datasets

    /// <summary>Downloads a dataset repository and returns the local directory holding it.</summary>
    /// <param name="repoId">A dataset id such as <c>stanfordnlp/imdb</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static string DownloadDataset(string repoId, string revision = "main")
        => Shared.SnapshotAsync(repoId, RepoKind.Dataset, revision).GetAwaiter().GetResult();

    /// <summary>Downloads a single file from a dataset repository.</summary>
    public static string DownloadDatasetFile(string repoId, string fileName, string revision = "main")
        => Shared.DownloadFileAsync(repoId, fileName, RepoKind.Dataset, revision).GetAwaiter().GetResult();

    /// <summary>Reads a dataset repository's metadata and file listing.</summary>
    public static RepoInfo DatasetInfo(string repoId, string revision = "main")
        => Shared.GetRepoInfoAsync(repoId, RepoKind.Dataset, revision).GetAwaiter().GetResult();

    /// <summary>Searches the Hub for datasets.</summary>
    public static IReadOnlyList<RepoInfo> SearchDatasets(string query, int limit = 20)
        => Shared.SearchAsync(query, RepoKind.Dataset, limit).GetAwaiter().GetResult();

    // ------------------------------------------------------------------ upload

    /// <summary>Uploads a small file to a repository.</summary>
    /// <param name="repoId">The destination repository id.</param>
    /// <param name="localPath">The file to upload.</param>
    /// <param name="pathInRepo">Where it should land in the repository.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="commitMessage">The commit summary.</param>
    /// <remarks>
    /// Capped at <see cref="HubClient.UploadLimitBytes"/>; see <see cref="HubClient.UploadFileAsync"/>
    /// for why weights need the Hub's own CLI.
    /// </remarks>
    public static void Upload(
        string repoId,
        string localPath,
        string pathInRepo,
        RepoKind kind = RepoKind.Model,
        string? commitMessage = null)
        => Shared.UploadFileAsync(repoId, localPath, pathInRepo, kind, "main", commitMessage)
            .GetAwaiter().GetResult();

    // ------------------------------------------------------------------ cache

    /// <summary>The cache the shared client downloads into.</summary>
    public static HubCache Cache => Shared.Cache;

    /// <summary>
    /// Files worth fetching for an inference run: safetensors weights and the JSON that describes
    /// the model and its tokenizer.
    /// </summary>
    private static readonly string[] InferencePatterns =
    [
        "*.safetensors",
        "*.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "*/*.safetensors",
        "*/config.json",
    ];

    /// <summary>
    /// Formats that duplicate weights already present in another form. Excluded from a full
    /// snapshot so that "download everything" does not mean "download it three times".
    /// </summary>
    private static readonly string[] HeavyDuplicatePatterns =
    [
        "*.msgpack",
        "*.h5",
        "*flax*",
        "*.tflite",
    ];
}
