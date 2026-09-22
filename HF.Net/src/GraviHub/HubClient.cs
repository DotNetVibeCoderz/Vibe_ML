using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Gravicode.HFNet.GraviHub;

/// <summary>
/// A client for the Hugging Face Hub: repository metadata, file download and upload, and a local
/// cache that makes a second run of the same program cost nothing.
/// </summary>
/// <remarks>
/// <para>
/// One instance holds one <see cref="HttpClient"/> and is safe to share across threads. Create it
/// once for the lifetime of the process; creating one per download exhausts sockets under load,
/// which shows up as an intermittent <c>SocketException</c> long after the code that caused it.
/// </para>
/// <para>
/// Every download goes through <see cref="HubCache"/>. A file already in the cache is revalidated
/// with a HEAD request against its ETag rather than re-fetched, so the steady-state cost of
/// <c>DownloadFile</c> on a 400 MB checkpoint is one round trip.
/// </para>
/// </remarks>
public sealed class HubClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private bool _disposed;

    /// <summary>Creates a client with the given settings.</summary>
    /// <param name="options">Endpoint, token, cache location and retry policy. Defaults are fine.</param>
    public HubClient(HubOptions? options = null)
    {
        Options = options ?? new HubOptions();
        Cache = new HubCache(Options.CacheRoot);

        _http = new HttpClient
        {
            BaseAddress = new Uri(Options.Endpoint.TrimEnd('/') + "/"),
            Timeout = Options.Timeout,
        };
        _ownsHttp = true;

        _http.DefaultRequestHeaders.UserAgent.ParseAdd("HF.Net/0.1 (GraviHub; Gravicode Studios)");

        var token = Options.ResolveToken();
        if (!string.IsNullOrWhiteSpace(token))
        {
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
    }

    /// <summary>Creates a client over a caller-supplied <see cref="HttpClient"/>.</summary>
    /// <param name="http">The transport. The caller keeps ownership and disposes it.</param>
    /// <param name="options">Endpoint, token, cache location and retry policy.</param>
    /// <remarks>
    /// This is the constructor to use from a host with its own handler pipeline - retry policies,
    /// proxies, instrumentation. The token from <paramref name="options"/> is still applied, unless
    /// the supplied client already carries an <c>Authorization</c> header.
    /// </remarks>
    public HubClient(HttpClient http, HubOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(http);

        Options = options ?? new HubOptions();
        Cache = new HubCache(Options.CacheRoot);
        _http = http;
        _ownsHttp = false;

        http.BaseAddress ??= new Uri(Options.Endpoint.TrimEnd('/') + "/");

        var token = Options.ResolveToken();
        if (http.DefaultRequestHeaders.Authorization is null && !string.IsNullOrWhiteSpace(token))
        {
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
    }

    /// <summary>The settings this client was created with.</summary>
    public HubOptions Options { get; }

    /// <summary>The on-disk cache backing every download.</summary>
    public HubCache Cache { get; }

    /// <summary>Whether a token was found, in the options or in the environment.</summary>
    public bool IsAuthenticated => !string.IsNullOrWhiteSpace(Options.ResolveToken());

    // ------------------------------------------------------------------ metadata

    /// <summary>Reads a repository's metadata, including its full file listing.</summary>
    /// <param name="repoId">A repository id such as <c>bert-base-uncased</c> or <c>google/vit-base-patch16-224</c>.</param>
    /// <param name="kind">Whether the id names a model, a dataset or a Space.</param>
    /// <param name="revision">A branch, tag or commit. Defaults to <c>main</c>.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public async Task<RepoInfo> GetRepoInfoAsync(
        string repoId,
        RepoKind kind = RepoKind.Model,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var url = $"api/{ApiSegment(kind)}/{repoId}/revision/{Uri.EscapeDataString(revision)}";
        using var response = await SendWithRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, url), repoId, cancellationToken).ConfigureAwait(false);

        var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        var files = await ListFilesAsync(repoId, kind, revision, cancellationToken).ConfigureAwait(false);

        return new RepoInfo(
            Id: root.TryGetProperty("id", out var id) ? id.GetString() ?? repoId : repoId,
            Kind: kind,
            Sha: root.TryGetProperty("sha", out var sha) ? sha.GetString() : null,
            LastModified: root.TryGetProperty("lastModified", out var modified)
                && modified.ValueKind == JsonValueKind.String
                && DateTimeOffset.TryParse(modified.GetString(), out var parsed) ? parsed : null,
            Downloads: root.TryGetProperty("downloads", out var downloads) && downloads.TryGetInt64(out var d) ? d : 0,
            Likes: root.TryGetProperty("likes", out var likes) && likes.TryGetInt64(out var l) ? l : 0,
            Tags: root.TryGetProperty("tags", out var tags) && tags.ValueKind == JsonValueKind.Array
                ? [.. tags.EnumerateArray().Select(t => t.GetString() ?? "").Where(t => t.Length > 0)]
                : [],
            PipelineTag: root.TryGetProperty("pipeline_tag", out var pipeline) ? pipeline.GetString() : null,
            Files: files,
            Private: root.TryGetProperty("private", out var isPrivate) && isPrivate.ValueKind == JsonValueKind.True,
            Gated: root.TryGetProperty("gated", out var gated) && gated.ValueKind != JsonValueKind.False
                && gated.ValueKind != JsonValueKind.Null);
    }

    /// <summary>Lists every file in a repository, recursively.</summary>
    /// <param name="repoId">The repository id.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <remarks>
    /// This uses the tree API rather than the <c>siblings</c> array on the repository document.
    /// <c>siblings</c> reports names only, so a caller trying to pick the smaller of two weight
    /// files, or to skip a 10 GB shard, has nothing to decide on.
    /// </remarks>
    public async Task<IReadOnlyList<RepoFile>> ListFilesAsync(
        string repoId,
        RepoKind kind = RepoKind.Model,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var files = new List<RepoFile>();
        var cursor = default(string);

        // The tree endpoint pages with a Link header once a repository has more than a thousand
        // entries. A sharded 70B checkpoint clears that easily.
        do
        {
            var url = $"api/{ApiSegment(kind)}/{repoId}/tree/{Uri.EscapeDataString(revision)}?recursive=true"
                + (cursor is null ? "" : $"&cursor={Uri.EscapeDataString(cursor)}");

            using var response = await SendWithRetryAsync(
                () => new HttpRequestMessage(HttpMethod.Get, url), repoId, cancellationToken).ConfigureAwait(false);

            var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            using var document = JsonDocument.Parse(json);

            foreach (var entry in document.RootElement.EnumerateArray())
            {
                if (entry.TryGetProperty("type", out var type) && type.GetString() != "file") continue;

                var path = entry.GetProperty("path").GetString()!;
                var isLfs = entry.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object;

                long? size = null;
                if (isLfs && lfs.TryGetProperty("size", out var lfsSize) && lfsSize.TryGetInt64(out var ls)) size = ls;
                else if (entry.TryGetProperty("size", out var plain) && plain.TryGetInt64(out var ps)) size = ps;

                var oid = isLfs && lfs.TryGetProperty("oid", out var lfsOid)
                    ? lfsOid.GetString()
                    : entry.TryGetProperty("oid", out var treeOid) ? treeOid.GetString() : null;

                files.Add(new RepoFile(path, size, oid, isLfs));
            }

            cursor = NextCursor(response);
        }
        while (cursor is not null);

        files.Sort(static (a, b) => string.CompareOrdinal(a.Path, b.Path));
        return files;
    }

    /// <summary>Searches the Hub.</summary>
    /// <param name="query">Free text matched against repository names.</param>
    /// <param name="kind">Whether to search models or datasets.</param>
    /// <param name="limit">How many results to return.</param>
    /// <param name="filter">An optional Hub filter tag, for example <c>text-classification</c>.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <remarks>
    /// The listing endpoint does not return a file tree, so the <see cref="RepoInfo.Files"/> of
    /// each result is empty. Call <see cref="GetRepoInfoAsync"/> on the one you pick.
    /// </remarks>
    public async Task<IReadOnlyList<RepoInfo>> SearchAsync(
        string query,
        RepoKind kind = RepoKind.Model,
        int limit = 20,
        string? filter = null,
        CancellationToken cancellationToken = default)
    {
        var url = $"api/{ApiSegment(kind)}?search={Uri.EscapeDataString(query ?? "")}&limit={limit}&sort=downloads&direction=-1"
            + (string.IsNullOrWhiteSpace(filter) ? "" : $"&filter={Uri.EscapeDataString(filter)}");

        using var response = await SendWithRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, url), query ?? "search", cancellationToken).ConfigureAwait(false);

        var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        using var document = JsonDocument.Parse(json);

        var results = new List<RepoInfo>();
        foreach (var entry in document.RootElement.EnumerateArray())
        {
            results.Add(new RepoInfo(
                Id: entry.GetProperty("id").GetString()!,
                Kind: kind,
                Sha: entry.TryGetProperty("sha", out var sha) ? sha.GetString() : null,
                LastModified: entry.TryGetProperty("lastModified", out var modified)
                    && modified.ValueKind == JsonValueKind.String
                    && DateTimeOffset.TryParse(modified.GetString(), out var parsed) ? parsed : null,
                Downloads: entry.TryGetProperty("downloads", out var downloads) && downloads.TryGetInt64(out var d) ? d : 0,
                Likes: entry.TryGetProperty("likes", out var likes) && likes.TryGetInt64(out var l) ? l : 0,
                Tags: entry.TryGetProperty("tags", out var tags) && tags.ValueKind == JsonValueKind.Array
                    ? [.. tags.EnumerateArray().Select(t => t.GetString() ?? "").Where(t => t.Length > 0)]
                    : [],
                PipelineTag: entry.TryGetProperty("pipeline_tag", out var pipeline) ? pipeline.GetString() : null,
                Files: [],
                Private: entry.TryGetProperty("private", out var isPrivate) && isPrivate.ValueKind == JsonValueKind.True,
                Gated: entry.TryGetProperty("gated", out var gated) && gated.ValueKind != JsonValueKind.False
                    && gated.ValueKind != JsonValueKind.Null));
        }

        return results;
    }

    /// <summary>Reads the account the current token belongs to.</summary>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The account name, or <c>null</c> when no token is configured.</returns>
    public async Task<string?> WhoAmIAsync(CancellationToken cancellationToken = default)
    {
        if (!IsAuthenticated) return null;

        using var response = await _http.GetAsync("api/whoami-v2", cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode) return null;

        var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        using var document = JsonDocument.Parse(json);
        return document.RootElement.TryGetProperty("name", out var name) ? name.GetString() : null;
    }

    // ------------------------------------------------------------------ download

    /// <summary>Downloads one file, returning the path it landed at in the cache.</summary>
    /// <param name="repoId">The repository id.</param>
    /// <param name="fileName">The repository-relative path, for example <c>model.safetensors</c>.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives transfer progress. Optional.</param>
    /// <param name="cancellationToken">Cancels the transfer.</param>
    /// <remarks>
    /// A partial download is written to a <c>.part</c> file and moved into place only once it is
    /// complete, so an interrupted transfer can never be mistaken for a finished one. A
    /// <c>.part</c> left from a previous run is resumed with a range request when the server
    /// allows it.
    /// </remarks>
    public async Task<string> DownloadFileAsync(
        string repoId,
        string fileName,
        RepoKind kind = RepoKind.Model,
        string revision = "main",
        IProgress<TransferProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);

        var destination = Cache.PathFor(repoId, kind, revision, fileName);

        if (Options.OfflineMode)
        {
            if (File.Exists(destination)) return destination;
            throw new HubException($"'{fileName}' is not cached and OfflineMode is on.") { RepoId = repoId };
        }

        var url = ResolveUrl(repoId, fileName, kind, revision);

        // A cached file is revalidated rather than re-fetched. The ETag of an LFS object is its
        // content hash, so this is an exact check, not a heuristic on timestamps.
        if (File.Exists(destination))
        {
            var cachedTag = Cache.ReadETag(destination);
            if (cachedTag is not null)
            {
                var remoteTag = await TryReadETagAsync(url, repoId, cancellationToken).ConfigureAwait(false);
                if (remoteTag is not null && remoteTag == cachedTag)
                {
                    progress?.Report(new TransferProgress(fileName, new FileInfo(destination).Length, new FileInfo(destination).Length));
                    return destination;
                }
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var partial = destination + ".part";
        var resumeFrom = File.Exists(partial) ? new FileInfo(partial).Length : 0;

        using var response = await SendWithRetryAsync(
            () =>
            {
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (resumeFrom > 0) request.Headers.Range = new RangeHeaderValue(resumeFrom, null);
                return request;
            },
            repoId,
            cancellationToken,
            HttpCompletionOption.ResponseHeadersRead).ConfigureAwait(false);

        // The server may ignore the range and send the whole file; appending in that case would
        // produce a file that is too long and silently corrupt.
        var appending = resumeFrom > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        var alreadyHave = appending ? resumeFrom : 0;
        var total = (response.Content.Headers.ContentLength ?? 0) + alreadyHave;

        await using (var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var output = new FileStream(
            partial,
            appending ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            1 << 20))
        {
            var buffer = new byte[1 << 20];
            var written = alreadyHave;
            int read;

            while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                written += read;
                progress?.Report(new TransferProgress(fileName, written, total > 0 ? total : null));
            }
        }

        File.Move(partial, destination, overwrite: true);

        var etag = NormalizeETag(response.Headers.ETag?.Tag);
        if (etag is not null) Cache.WriteETag(destination, etag);

        return destination;
    }

    /// <summary>
    /// Downloads a whole repository revision, or the part of it that matches a pattern.
    /// </summary>
    /// <param name="repoId">The repository id.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="allowPatterns">Glob patterns to include; everything when null.</param>
    /// <param name="ignorePatterns">Glob patterns to exclude, applied after the allow list.</param>
    /// <param name="progress">Receives per-file transfer progress. Optional.</param>
    /// <param name="cancellationToken">Cancels the transfer.</param>
    /// <returns>The local directory holding the snapshot.</returns>
    /// <remarks>
    /// Most repositories carry the same weights several times over - safetensors beside a PyTorch
    /// pickle beside an ONNX export. Downloading a snapshot without patterns therefore routinely
    /// transfers three times what is needed; <c>allowPatterns: ["*.safetensors", "*.json"]</c> is
    /// the usual intent.
    /// </remarks>
    public async Task<string> SnapshotAsync(
        string repoId,
        RepoKind kind = RepoKind.Model,
        string revision = "main",
        IReadOnlyList<string>? allowPatterns = null,
        IReadOnlyList<string>? ignorePatterns = null,
        IProgress<TransferProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var files = await ListFilesAsync(repoId, kind, revision, cancellationToken).ConfigureAwait(false);

        foreach (var file in files)
        {
            if (allowPatterns is { Count: > 0 } && !allowPatterns.Any(p => GlobMatch(file.Path, p))) continue;
            if (ignorePatterns is { Count: > 0 } && ignorePatterns.Any(p => GlobMatch(file.Path, p))) continue;

            await DownloadFileAsync(repoId, file.Path, kind, revision, progress, cancellationToken).ConfigureAwait(false);
        }

        return Cache.DirectoryFor(repoId, kind, revision);
    }

    // ------------------------------------------------------------------ upload

    /// <summary>Uploads one file to a repository, as a single commit.</summary>
    /// <param name="repoId">The destination repository id.</param>
    /// <param name="localPath">The file to upload.</param>
    /// <param name="pathInRepo">Where it should land, relative to the repository root.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="revision">The branch to commit on.</param>
    /// <param name="commitMessage">The commit summary.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <remarks>
    /// <b>This path is for small files.</b> The commit API takes file content inline as base64, so
    /// the whole file is held in memory twice and travels in the request body. Anything the Hub
    /// would store in LFS - which is to say any real weight file - needs the LFS batch protocol,
    /// which this client does not implement; <see cref="UploadLimitBytes"/> is enforced rather than
    /// letting a 5 GB upload fail slowly and unhelpfully.
    /// </remarks>
    public async Task UploadFileAsync(
        string repoId,
        string localPath,
        string pathInRepo,
        RepoKind kind = RepoKind.Model,
        string revision = "main",
        string? commitMessage = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);
        ArgumentException.ThrowIfNullOrWhiteSpace(localPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(pathInRepo);

        if (!IsAuthenticated)
        {
            throw new HubException("Uploading needs a token. Set HF_TOKEN or pass one in HubOptions.")
            {
                RepoId = repoId,
            };
        }

        var info = new FileInfo(localPath);
        if (!info.Exists) throw new FileNotFoundException("File to upload was not found.", localPath);

        if (info.Length > UploadLimitBytes)
        {
            throw new HubException(
                $"'{localPath}' is {info.Length / (1024.0 * 1024):0.#} MB. The inline commit API is capped at "
                + $"{UploadLimitBytes / (1024 * 1024)} MB here because larger files must go through Git LFS, "
                + "which GraviHub does not implement. Use the huggingface-cli for large weights.")
            {
                RepoId = repoId,
            };
        }

        var content = Convert.ToBase64String(await File.ReadAllBytesAsync(localPath, cancellationToken).ConfigureAwait(false));

        // The commit endpoint takes NDJSON: a header line, then one line per operation.
        var body = new StringBuilder();
        body.Append(JsonSerializer.Serialize(new
        {
            key = "header",
            value = new { summary = commitMessage ?? $"Upload {pathInRepo} via HF.Net", description = "" },
        })).Append('\n');

        body.Append(JsonSerializer.Serialize(new
        {
            key = "file",
            value = new { content, path = pathInRepo.Replace('\\', '/'), encoding = "base64" },
        })).Append('\n');

        var url = $"api/{ApiSegment(kind)}/{repoId}/commit/{Uri.EscapeDataString(revision)}";

        using var response = await SendWithRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new StringContent(body.ToString(), Encoding.UTF8, "application/x-ndjson"),
            },
            repoId,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>The largest file <see cref="UploadFileAsync"/> will send inline.</summary>
    public const long UploadLimitBytes = 10L * 1024 * 1024;

    // ------------------------------------------------------------------ plumbing

    /// <summary>Builds the download URL for a file, which is also what a browser would use.</summary>
    public string ResolveUrl(string repoId, string fileName, RepoKind kind = RepoKind.Model, string revision = "main")
    {
        var prefix = kind switch
        {
            RepoKind.Model => "",
            RepoKind.Dataset => "datasets/",
            RepoKind.Space => "spaces/",
            _ => "",
        };

        var encoded = string.Join('/', fileName.Replace('\\', '/').Split('/').Select(Uri.EscapeDataString));
        return $"{Options.Endpoint.TrimEnd('/')}/{prefix}{repoId}/resolve/{Uri.EscapeDataString(revision)}/{encoded}";
    }

    private static string ApiSegment(RepoKind kind) => kind switch
    {
        RepoKind.Model => "models",
        RepoKind.Dataset => "datasets",
        RepoKind.Space => "spaces",
        _ => "models",
    };

    private async Task<string?> TryReadETagAsync(string url, string repoId, CancellationToken cancellationToken)
    {
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Head, url);
            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            return response.IsSuccessStatusCode ? NormalizeETag(response.Headers.ETag?.Tag) : null;
        }
        catch (HttpRequestException)
        {
            // Revalidation is an optimisation. If it fails the caller still gets the cached file,
            // which is better than failing a download that may not have been necessary.
            return null;
        }
    }

    /// <summary>Strips quotes and the weak marker so two spellings of one ETag compare equal.</summary>
    private static string? NormalizeETag(string? tag)
        => string.IsNullOrEmpty(tag) ? null : tag.Trim().TrimStart('W', '/').Trim('"');

    private static string? NextCursor(HttpResponseMessage response)
    {
        if (!response.Headers.TryGetValues("Link", out var links)) return null;

        foreach (var link in links)
        {
            foreach (var part in link.Split(','))
            {
                if (!part.Contains("rel=\"next\"", StringComparison.Ordinal)) continue;

                var start = part.IndexOf('<');
                var end = part.IndexOf('>');
                if (start < 0 || end <= start) continue;

                var uri = part[(start + 1)..end];
                var query = new Uri(uri, UriKind.RelativeOrAbsolute);
                var text = query.IsAbsoluteUri ? query.Query : uri;

                const string Marker = "cursor=";
                var at = text.IndexOf(Marker, StringComparison.Ordinal);
                if (at < 0) continue;

                var value = text[(at + Marker.Length)..];
                var amp = value.IndexOf('&');
                return Uri.UnescapeDataString(amp < 0 ? value : value[..amp]);
            }
        }

        return null;
    }

    private async Task<HttpResponseMessage> SendWithRetryAsync(
        Func<HttpRequestMessage> factory,
        string context,
        CancellationToken cancellationToken,
        HttpCompletionOption completion = HttpCompletionOption.ResponseContentRead)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        HttpResponseMessage? response = null;
        Exception? failure = null;

        for (var attempt = 0; attempt <= Options.MaxRetries; attempt++)
        {
            if (attempt > 0)
            {
                // Exponential backoff with a cap. The Hub answers a burst with 429 and expects the
                // client to slow down rather than to keep hammering at a fixed interval.
                var delay = TimeSpan.FromMilliseconds(Math.Min(500 * Math.Pow(2, attempt - 1), 8000));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }

            response?.Dispose();

            try
            {
                using var request = factory();
                response = await _http.SendAsync(request, completion, cancellationToken).ConfigureAwait(false);
                failure = null;
            }
            catch (HttpRequestException ex)
            {
                failure = ex;
                continue;
            }
            catch (TaskCanceledException ex) when (!cancellationToken.IsCancellationRequested)
            {
                failure = ex;
                continue;
            }

            if (response.IsSuccessStatusCode) return response;

            // 4xx other than 429 will not become true by being asked again.
            if (response.StatusCode is not (HttpStatusCode.TooManyRequests or HttpStatusCode.RequestTimeout)
                && (int)response.StatusCode < 500)
            {
                break;
            }
        }

        if (failure is not null)
        {
            throw new HubException($"Request for '{context}' failed: {failure.Message}", failure) { RepoId = context };
        }

        var status = response!.StatusCode;
        var detail = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        response.Dispose();

        var hint = status switch
        {
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden =>
                " The repository may be private or gated; set HF_TOKEN to a token with access, and accept the licence on the Hub if it is gated.",
            HttpStatusCode.NotFound =>
                " Check the id, the revision, and whether it is a dataset rather than a model.",
            HttpStatusCode.TooManyRequests =>
                " The Hub is rate limiting. An authenticated client gets a much higher limit.",
            _ => "",
        };

        throw new HubException(
            $"The Hub answered {(int)status} {status} for '{context}'.{hint}"
            + (string.IsNullOrWhiteSpace(detail) ? "" : $" Response: {Truncate(detail, 400)}"))
        {
            Status = status,
            RepoId = context,
        };
    }

    private static string Truncate(string text, int limit)
        => text.Length <= limit ? text : text[..limit] + "...";

    /// <summary>Matches a repository path against a glob with <c>*</c> and <c>?</c>.</summary>
    /// <remarks>
    /// <c>*</c> crosses directory separators here, unlike a shell glob. Hub patterns are used as
    /// <c>"*.safetensors"</c> against paths that may sit in a subdirectory, and a non-crossing
    /// <c>*</c> would silently miss every sharded checkpoint.
    /// </remarks>
    internal static bool GlobMatch(string path, string pattern)
    {
        var regex = "^" + System.Text.RegularExpressions.Regex.Escape(pattern)
            .Replace("\\*", ".*")
            .Replace("\\?", ".") + "$";

        return System.Text.RegularExpressions.Regex.IsMatch(
            path, regex, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsHttp) _http.Dispose();
    }
}
