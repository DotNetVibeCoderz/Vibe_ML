using System.ComponentModel;
using System.Data;
using System.Globalization;
using System.Text;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using HtmlAgilityPack;
using Microsoft.SemanticKernel;

namespace BlazorML.Agents.Plugins;

/// <summary>Dates, times and arithmetic — the things a chat model is worst at doing in its head.</summary>
public sealed class TimeAndMathPlugin
{
    [KernelFunction, Description("Gets the current date and time.")]
    public string CurrentDateTime(
        [Description("IANA time zone id, for example Asia/Jakarta. Defaults to the server's zone.")]
        string? timeZone = null)
    {
        var now = DateTimeOffset.Now;

        if (!string.IsNullOrWhiteSpace(timeZone))
        {
            try
            {
                var zone = TimeZoneInfo.FindSystemTimeZoneById(timeZone);
                now = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, zone);
            }
            catch (TimeZoneNotFoundException)
            {
                return $"Time zone '{timeZone}' was not recognised.";
            }
        }

        return now.ToString("dddd, dd MMMM yyyy HH:mm:ss zzz", CultureInfo.InvariantCulture);
    }

    [KernelFunction, Description("Counts the days between two dates.")]
    public string DaysBetween(
        [Description("The earlier date, as yyyy-MM-dd.")] string from,
        [Description("The later date, as yyyy-MM-dd.")] string to)
    {
        if (!DateTime.TryParse(from, CultureInfo.InvariantCulture, DateTimeStyles.None, out var start) ||
            !DateTime.TryParse(to, CultureInfo.InvariantCulture, DateTimeStyles.None, out var end))
        {
            return "Both dates must be written as yyyy-MM-dd.";
        }

        return $"{(end - start).TotalDays:N0} days";
    }

    [KernelFunction, Description("Evaluates an arithmetic expression, for example (1200*0.85)/12.")]
    public string Calculate([Description("The expression to evaluate.")] string expression)
    {
        try
        {
            // DataTable.Compute handles the arithmetic without pulling in an expression library
            // and without ever executing arbitrary code.
            var result = new DataTable().Compute(expression, null);
            return Convert.ToString(result, CultureInfo.InvariantCulture) ?? "no result";
        }
        catch (Exception e) when (e is EvaluateException or SyntaxErrorException or InvalidCastException)
        {
            return $"'{expression}' is not an expression I can evaluate: {e.Message}";
        }
    }

    [KernelFunction, Description("Computes mean, median, min, max and standard deviation for a list of numbers.")]
    public string Statistics(
        [Description("Numbers separated by commas or spaces.")] string numbers)
    {
        var values = numbers
            .Split([',', ' ', '\n', '\t', ';'], StringSplitOptions.RemoveEmptyEntries)
            .Select(s => double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : (double?)null)
            .Where(v => v.HasValue)
            .Select(v => v!.Value)
            .OrderBy(v => v)
            .ToList();

        if (values.Count == 0)
        {
            return "No numbers were found in that input.";
        }

        var mean = values.Average();
        var stdDev = values.Count > 1
            ? Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1))
            : 0;

        return JsonSerializer.Serialize(new
        {
            count = values.Count,
            mean,
            median = values[values.Count / 2],
            min = values[0],
            max = values[^1],
            stdDev
        });
    }
}

/// <summary>Reaching outside the app: web search, page scraping, and reading a file from a URL.</summary>
public sealed class WebPlugin(ISettingsService settings, IHttpClientFactory httpFactory)
{
    [KernelFunction, Description("Searches the web and returns the top results with a short summary of each.")]
    public async Task<string> SearchWeb(
        [Description("What to search for.")] string query,
        CancellationToken ct = default)
    {
        var options = await settings.GetAsync<ToolOptions>(SettingsSections.Tools, ct);

        if (!options.EnableWebSearch)
        {
            return "Web search is switched off for this workspace.";
        }

        if (string.IsNullOrWhiteSpace(options.TavilyApiKey))
        {
            return "Web search needs a Tavily API key. Add one under Settings → Tools.";
        }

        using var client = httpFactory.CreateClient("agent");
        client.Timeout = TimeSpan.FromSeconds(options.WebRequestTimeoutSeconds);

        var payload = JsonSerializer.Serialize(new
        {
            api_key = options.TavilyApiKey,
            query,
            max_results = options.MaxSearchResults,
            search_depth = "basic",
            include_answer = true
        });

        using var content = new StringContent(payload, Encoding.UTF8, "application/json");
        using var response = await client.PostAsync("https://api.tavily.com/search", content, ct);

        if (!response.IsSuccessStatusCode)
        {
            return $"The search service returned {(int)response.StatusCode}. Check the Tavily API key.";
        }

        var body = await response.Content.ReadAsStringAsync(ct);

        using var document = JsonDocument.Parse(body);
        var builder = new StringBuilder();

        if (document.RootElement.TryGetProperty("answer", out var answer) &&
            answer.ValueKind == JsonValueKind.String)
        {
            builder.AppendLine($"Summary: {answer.GetString()}");
            builder.AppendLine();
        }

        if (document.RootElement.TryGetProperty("results", out var results))
        {
            foreach (var result in results.EnumerateArray())
            {
                builder.AppendLine($"- {result.GetProperty("title").GetString()}");
                builder.AppendLine($"  {result.GetProperty("url").GetString()}");

                if (result.TryGetProperty("content", out var snippet))
                {
                    builder.AppendLine($"  {Trim(snippet.GetString(), 400)}");
                }
            }
        }

        return builder.Length == 0 ? "No results were found." : builder.ToString();
    }

    [KernelFunction, Description("Reads a web page and returns its visible text.")]
    public async Task<string> ScrapeUrl(
        [Description("The page URL.")] string url,
        CancellationToken ct = default)
    {
        var options = await settings.GetAsync<ToolOptions>(SettingsSections.Tools, ct);

        if (!options.EnableUrlScraping)
        {
            return "Page scraping is switched off for this workspace.";
        }

        using var client = httpFactory.CreateClient("agent");
        client.Timeout = TimeSpan.FromSeconds(options.WebRequestTimeoutSeconds);

        try
        {
            var html = await client.GetStringAsync(url, ct);

            var document = new HtmlDocument();
            document.LoadHtml(html);

            // Scripts and styles are text too, and including them buries the actual content.
            foreach (var node in document.DocumentNode
                         .SelectNodes("//script|//style|//nav|//footer")?.ToList() ?? [])
            {
                node.Remove();
            }

            var text = HtmlEntity.DeEntitize(document.DocumentNode.InnerText) ?? string.Empty;
            var collapsed = string.Join('\n', text
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line => line.Length > 1));

            return Trim(collapsed, 12000);
        }
        catch (HttpRequestException e)
        {
            return $"Could not fetch {url}: {e.Message}";
        }
        catch (TaskCanceledException)
        {
            return $"{url} did not respond within {options.WebRequestTimeoutSeconds} seconds.";
        }
    }

    [KernelFunction, Description("Downloads a text, CSV or JSON file from a URL and returns its contents.")]
    public async Task<string> ReadFileFromUrl(
        [Description("The file URL.")] string url,
        [Description("Maximum characters to return.")] int maxCharacters = 8000,
        CancellationToken ct = default)
    {
        var options = await settings.GetAsync<ToolOptions>(SettingsSections.Tools, ct);

        using var client = httpFactory.CreateClient("agent");
        client.Timeout = TimeSpan.FromSeconds(options.WebRequestTimeoutSeconds);

        try
        {
            var content = await client.GetStringAsync(url, ct);
            return Trim(content, Math.Clamp(maxCharacters, 100, 50000));
        }
        catch (HttpRequestException e)
        {
            return $"Could not read {url}: {e.Message}";
        }
    }

    private static string Trim(string? text, int max)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        return text.Length <= max ? text : text[..max] + $"\n… truncated at {max:N0} characters.";
    }
}
