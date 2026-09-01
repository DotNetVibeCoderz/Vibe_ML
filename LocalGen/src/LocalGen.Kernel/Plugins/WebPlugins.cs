using System.ComponentModel;
using System.Text;
using System.Net.Http.Json;
using System.Text.Json;
using AngleSharp;
using AngleSharp.Dom;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Fetches a page and reduces it to readable text.
/// </summary>
/// <remarks>
/// Raw HTML is mostly markup and script, which would waste the context window and bury the
/// content. AngleSharp parses the document properly so that the extracted text reflects the DOM
/// rather than a regex over tags.
/// </remarks>
public sealed class WebScrapePlugin
{
    private const string ToolName = "WebScrape";
    private const int MaxContentLength = 20_000;

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly LocalGenOptions _options;
    private readonly ILogger<WebScrapePlugin> _logger;

    public WebScrapePlugin(
        IHttpClientFactory httpClientFactory,
        LocalGenOptions options,
        ILogger<WebScrapePlugin> logger)
    {
        _httpClientFactory = httpClientFactory;
        _options = options;
        _logger = logger;
    }

    [KernelFunction("scrape_url")]
    [Description("Fetches a web page and returns its readable text content, with scripts and navigation removed.")]
    public async Task<string> ScrapeAsync(
        [Description("Absolute URL of the page")] string url,
        [Description("Return raw HTML instead of extracted text")] bool raw = false,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return "Error: LocalGen is in offline mode, so web pages cannot be fetched.";
        }

        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            uri.Scheme is not ("http" or "https"))
        {
            return $"Error: '{url}' is not an http or https URL.";
        }

        try
        {
            using var http = _httpClientFactory.CreateClient(ToolName);
            using var response = await http.GetAsync(uri, cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return $"Error: {uri} returned HTTP {(int)response.StatusCode} {response.ReasonPhrase}.";
            }

            var html = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

            if (raw)
            {
                return Truncate(html);
            }

            var context = BrowsingContext.New(Configuration.Default);
            using var document = await context
                .OpenAsync(request => request.Content(html).Address(uri.ToString()), cancellationToken)
                .ConfigureAwait(false);

            return Truncate(ExtractText(document, uri));
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "Scrape failed for {Url}", url);
            return $"Error: could not fetch {uri} — {ex.Message}";
        }
        catch (TaskCanceledException)
        {
            return $"Error: the request to {uri} timed out.";
        }
    }

    [KernelFunction("extract_links")]
    [Description("Returns the links found on a web page, as text and absolute URL pairs.")]
    public async Task<string> ExtractLinksAsync(
        [Description("Absolute URL of the page")] string url,
        [Description("Maximum number of links to return")] int limit = 50,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return "Error: LocalGen is in offline mode, so web pages cannot be fetched.";
        }

        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            return $"Error: '{url}' is not a valid URL.";
        }

        try
        {
            using var http = _httpClientFactory.CreateClient(ToolName);
            var html = await http.GetStringAsync(uri, cancellationToken).ConfigureAwait(false);

            var context = BrowsingContext.New(Configuration.Default);
            using var document = await context
                .OpenAsync(request => request.Content(html).Address(uri.ToString()), cancellationToken)
                .ConfigureAwait(false);

            var builder = new StringBuilder();
            var count = 0;

            foreach (var anchor in document.QuerySelectorAll("a[href]"))
            {
                var href = anchor.GetAttribute("href");
                if (string.IsNullOrWhiteSpace(href) || href.StartsWith('#'))
                {
                    continue;
                }

                // Relative hrefs are resolved so the model gets URLs it can actually fetch.
                var absolute = Uri.TryCreate(uri, href, out var resolved) ? resolved.ToString() : href;
                var text = anchor.TextContent.Trim();

                builder.Append(string.IsNullOrEmpty(text) ? "(no text)" : text)
                       .Append(" → ")
                       .AppendLine(absolute);

                if (++count >= limit)
                {
                    break;
                }
            }

            return count == 0 ? "No links were found on that page." : builder.ToString();
        }
        catch (HttpRequestException ex)
        {
            return $"Error: could not fetch {uri} — {ex.Message}";
        }
    }

    /// <summary>
    /// Pulls the body text out of a parsed document, dropping the elements that never carry
    /// content and preferring the main article region when the page marks one.
    /// </summary>
    private static string ExtractText(IDocument document, Uri uri)
    {
        foreach (var selector in (string[])["script", "style", "noscript", "svg", "nav", "footer", "header", "form"])
        {
            foreach (var element in document.QuerySelectorAll(selector).ToList())
            {
                element.Remove();
            }
        }

        var root = document.QuerySelector("main")
                   ?? document.QuerySelector("article")
                   ?? document.Body;

        var builder = new StringBuilder();
        builder.Append("# ").AppendLine(document.Title ?? uri.Host);
        builder.Append("Source: ").AppendLine(uri.ToString()).AppendLine();

        var text = root?.TextContent ?? string.Empty;

        // Collapse the runs of blank lines that stripping elements leaves behind.
        foreach (var line in text.Split('\n'))
        {
            var trimmed = line.Trim();
            if (trimmed.Length > 0)
            {
                builder.AppendLine(trimmed);
            }
        }

        return builder.ToString();
    }

    private static string Truncate(string value) =>
        value.Length <= MaxContentLength
            ? value
            : value[..MaxContentLength] + $"\n… [truncated, {value.Length:N0} characters total]";
}

/// <summary>
/// Internet search through Tavily, whose API is built for LLM consumption: it returns extracted
/// answers and content snippets rather than a page of blue links that would need scraping.
/// </summary>
public sealed class SearchPlugin
{
    private const string TavilyEndpoint = "https://api.tavily.com/search";

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly LocalGenOptions _options;
    private readonly ILogger<SearchPlugin> _logger;

    public SearchPlugin(
        IHttpClientFactory httpClientFactory,
        LocalGenOptions options,
        ILogger<SearchPlugin> logger)
    {
        _httpClientFactory = httpClientFactory;
        _options = options;
        _logger = logger;
    }

    [KernelFunction("web_search")]
    [Description("Searches the internet and returns relevant results with short content extracts.")]
    public async Task<string> SearchAsync(
        [Description("The search query")] string query,
        [Description("Number of results to return")] int maxResults = 5,
        [Description("Use a deeper, slower search")] bool deep = false,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return "Error: LocalGen is in offline mode, so web search is unavailable.";
        }

        var apiKey = _options.Tools.TavilyApiKey;
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return "Error: web search needs a Tavily API key. Set LocalGen:Tools:TavilyApiKey " +
                   "(get one free at https://tavily.com).";
        }

        try
        {
            using var http = _httpClientFactory.CreateClient(nameof(SearchPlugin));

            var payload = new
            {
                api_key = apiKey,
                query,
                max_results = Math.Clamp(maxResults, 1, 20),
                search_depth = deep ? "advanced" : "basic",
                include_answer = true
            };

            using var response = await http
                .PostAsJsonAsync(TavilyEndpoint, payload, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                return $"Error: Tavily returned HTTP {(int)response.StatusCode}. {body}";
            }

            await using var stream = await response.Content
                .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var document = await JsonDocument
                .ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);

            return FormatResults(document.RootElement, query);
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "Tavily search failed for '{Query}'", query);
            return $"Error: the search request failed — {ex.Message}";
        }
    }

    private static string FormatResults(JsonElement root, string query)
    {
        var builder = new StringBuilder();
        builder.Append("Search results for: ").AppendLine(query).AppendLine();

        // Tavily's synthesised answer is usually the most directly useful part.
        if (root.TryGetProperty("answer", out var answer) &&
            answer.ValueKind == JsonValueKind.String &&
            !string.IsNullOrWhiteSpace(answer.GetString()))
        {
            builder.Append("Summary: ").AppendLine(answer.GetString()).AppendLine();
        }

        if (!root.TryGetProperty("results", out var results) ||
            results.ValueKind != JsonValueKind.Array)
        {
            return builder.Append("No results were returned.").ToString();
        }

        var index = 1;
        foreach (var result in results.EnumerateArray())
        {
            builder.Append(index++).Append(". ")
                   .AppendLine(result.TryGetProperty("title", out var title) ? title.GetString() : "(untitled)");

            if (result.TryGetProperty("url", out var url))
            {
                builder.Append("   ").AppendLine(url.GetString());
            }

            if (result.TryGetProperty("content", out var content))
            {
                var text = content.GetString() ?? string.Empty;
                builder.Append("   ")
                       .AppendLine(text.Length > 500 ? text[..500] + "…" : text);
            }

            builder.AppendLine();
        }

        return builder.ToString();
    }
}

/// <summary>Downloads a file from a URL into the workspace.</summary>
public sealed class DownloadPlugin
{
    private const string ToolName = "Download";

    /// <summary>Caps a single download so a tool call cannot fill the disk.</summary>
    private const long MaxDownloadBytes = 200L * 1024 * 1024;

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly PathGuard _guard;
    private readonly LocalGenOptions _options;

    public DownloadPlugin(
        IHttpClientFactory httpClientFactory,
        PathGuard guard,
        LocalGenOptions options)
    {
        _httpClientFactory = httpClientFactory;
        _guard = guard;
        _options = options;
    }

    [KernelFunction("download_file")]
    [Description("Downloads a file from a URL into the workspace and returns the saved path.")]
    public async Task<string> DownloadAsync(
        [Description("Absolute URL of the file")] string url,
        [Description("Optional file name; defaults to the name in the URL")] string? fileName = null,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return "Error: LocalGen is in offline mode, so downloads are unavailable.";
        }

        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            uri.Scheme is not ("http" or "https"))
        {
            return $"Error: '{url}' is not an http or https URL.";
        }

        try
        {
            var name = string.IsNullOrWhiteSpace(fileName)
                ? Path.GetFileName(uri.LocalPath)
                : fileName;

            if (string.IsNullOrWhiteSpace(name))
            {
                name = $"download-{DateTime.UtcNow:yyyyMMddHHmmss}";
            }

            // Strip any directory components the URL or the model supplied.
            var destination = _guard.Resolve(Path.GetFileName(name), ToolName);

            using var http = _httpClientFactory.CreateClient(ToolName);
            using var response = await http
                .GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return $"Error: {uri} returned HTTP {(int)response.StatusCode}.";
            }

            var declaredLength = response.Content.Headers.ContentLength;
            if (declaredLength > MaxDownloadBytes)
            {
                return $"Error: the file is {declaredLength / 1024 / 1024} MB, over the " +
                       $"{MaxDownloadBytes / 1024 / 1024} MB limit for this tool.";
            }

            await using var source = await response.Content
                .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using var target = File.Create(destination);

            var buffer = new byte[81920];
            long written = 0;
            int read;

            while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                written += read;

                // Servers may omit Content-Length, so the cap is also enforced while streaming.
                if (written > MaxDownloadBytes)
                {
                    target.Close();
                    File.Delete(destination);
                    return $"Error: the download exceeded the {MaxDownloadBytes / 1024 / 1024} MB limit.";
                }

                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            }

            return $"Downloaded {written:N0} bytes to {destination}.";
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (HttpRequestException ex)
        {
            return $"Error: the download failed — {ex.Message}";
        }
    }
}
