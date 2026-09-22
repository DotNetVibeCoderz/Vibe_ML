using System.ComponentModel;
using System.Data;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;
using HFAppGen.Models;
using HFAppGen.Services;

namespace HFAppGen.Plugins;

/// <summary>
/// General-purpose tools: web search, page scraping, arithmetic and the clock.
/// </summary>
/// <remarks>
/// These exist because a language model is unreliable at exactly these things. It cannot know
/// today's date, it is a poor calculator, and its knowledge has a cutoff. Giving it tools is
/// cheaper and far more accurate than hoping it guesses correctly.
/// </remarks>
public sealed partial class CommonToolsPlugin(AppSettings settings, LogService logs)
{
    private static readonly HttpClient Http = new();

    // ---------------------------------------------------------------- search

    [KernelFunction("SearchInternet")]
    [Description("Searches the web with Tavily and returns titles, URLs and snippets. Use it for " +
                 "anything after your knowledge cutoff, for current library versions, or to find " +
                 "documentation before writing code against an unfamiliar API.")]
    public async Task<string> SearchInternetAsync(
        [Description("The search query. Be specific.")] string query,
        [Description("How many results to return, 1 to 10.")] int maxResults = 5)
    {
        if (string.IsNullOrWhiteSpace(settings.TavilyApiKey))
            return "Web search is unavailable: no Tavily API key is configured. " +
                   "Set tools.tavilyApiKey in Settings, or answer from what you already know.";

        logs.Info("search", $"Tavily: {query}");

        try
        {
            var request = new JsonObject
            {
                ["api_key"] = settings.TavilyApiKey,
                ["query"] = query,
                ["max_results"] = Math.Clamp(maxResults, 1, 10),
                ["search_depth"] = "basic",
                ["include_answer"] = true,
            };

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(settings.WebTimeoutSeconds));
            using var response = await Http.PostAsync("https://api.tavily.com/search",
                new StringContent(request.ToJsonString(), Encoding.UTF8, "application/json"), cts.Token);

            var body = await response.Content.ReadAsStringAsync(cts.Token);
            if (!response.IsSuccessStatusCode)
                return $"Search failed with HTTP {(int)response.StatusCode}: {Truncate(body, 400)}";

            var payload = JsonNode.Parse(body);
            var builder = new StringBuilder();

            var answer = (string?)payload?["answer"];
            if (!string.IsNullOrWhiteSpace(answer)) builder.AppendLine($"Summary: {answer}\n");

            var index = 1;
            foreach (var result in payload?["results"]?.AsArray() ?? [])
            {
                builder.AppendLine($"{index++}. {(string?)result?["title"]}");
                builder.AppendLine($"   {(string?)result?["url"]}");
                var content = (string?)result?["content"];
                if (!string.IsNullOrWhiteSpace(content)) builder.AppendLine($"   {Truncate(content, 320)}");
                builder.AppendLine();
            }

            return builder.Length == 0 ? "No results." : builder.ToString();
        }
        catch (TaskCanceledException)
        {
            return $"Search timed out after {settings.WebTimeoutSeconds}s.";
        }
        catch (Exception ex)
        {
            logs.Error("search", ex.Message);
            return $"Search failed: {ex.Message}";
        }
    }

    // ---------------------------------------------------------------- scraping

    [KernelFunction("ScrapeWebPage")]
    [Description("Fetches a web page and returns its readable text with the markup stripped. Use " +
                 "it after SearchInternet to read a promising result in full.")]
    public async Task<string> ScrapeWebPageAsync(
        [Description("Absolute URL, including https://")] string url,
        [Description("Maximum characters to return.")] int maxCharacters = 8000)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri)
            || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            return $"'{url}' is not a valid http(s) URL.";

        logs.Info("scrape", url);

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(settings.WebTimeoutSeconds));
            using var request = new HttpRequestMessage(HttpMethod.Get, uri);
            // Some sites serve a stub to unknown agents.
            request.Headers.UserAgent.ParseAdd("Mozilla/5.0 (compatible; HFAppGen/1.0)");

            using var response = await Http.SendAsync(request, cts.Token);
            if (!response.IsSuccessStatusCode)
                return $"Fetch failed with HTTP {(int)response.StatusCode}.";

            var html = await response.Content.ReadAsStringAsync(cts.Token);
            return Truncate(ExtractText(html), maxCharacters);
        }
        catch (TaskCanceledException)
        {
            return $"Fetch timed out after {settings.WebTimeoutSeconds}s.";
        }
        catch (Exception ex)
        {
            return $"Fetch failed: {ex.Message}";
        }
    }

    /// <summary>Strips scripts, styles and tags, then collapses whitespace.</summary>
    private static string ExtractText(string html)
    {
        var text = ScriptAndStyle().Replace(html, " ");
        text = Tags().Replace(text, " ");
        text = System.Net.WebUtility.HtmlDecode(text);
        return Whitespace().Replace(text, " ").Trim();
    }

    [GeneratedRegex(@"<(script|style|noscript)[^>]*>.*?</\1>",
        RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ScriptAndStyle();

    [GeneratedRegex("<[^>]+>")]
    private static partial Regex Tags();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    // ---------------------------------------------------------------- maths

    [KernelFunction("MathCalculation")]
    [Description("Evaluates an arithmetic expression exactly. Use it for any calculation whose " +
                 "answer matters - do not compute it in your head. Supports + - * / % and " +
                 "parentheses, and the functions sqrt, abs, pow, log, log10, exp, sin, cos, tan, " +
                 "floor, ceil, round, min, max.")]
    public string MathCalculation(
        [Description("The expression, e.g. sqrt(2) * 100 / (3 + 4)")] string expression)
    {
        if (string.IsNullOrWhiteSpace(expression)) return "Empty expression.";

        try
        {
            var reduced = ApplyFunctions(expression);

            // DataTable.Compute handles the arithmetic and precedence; the functions above are
            // reduced to literals first because it does not know them.
            var value = new DataTable().Compute(reduced, null);
            if (value is null or DBNull) return $"Could not evaluate '{expression}'.";

            var result = Convert.ToDouble(value, CultureInfo.InvariantCulture);
            return double.IsInteger(result) && Math.Abs(result) < 1e15
                ? $"{expression} = {result:0}"
                : $"{expression} = {result:G15}";
        }
        catch (Exception ex)
        {
            return $"Could not evaluate '{expression}': {ex.Message}";
        }
    }

    /// <summary>Replaces supported function calls with their computed value, innermost first.</summary>
    private static string ApplyFunctions(string expression)
    {
        var pattern = FunctionCall();

        for (var pass = 0; pass < 24; pass++)
        {
            var match = pattern.Match(expression);
            if (!match.Success) break;

            var name = match.Groups["fn"].Value.ToLowerInvariant();
            var arguments = match.Groups["args"].Value
                .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Select(a => Convert.ToDouble(new DataTable().Compute(a, null), CultureInfo.InvariantCulture))
                .ToArray();

            double value = name switch
            {
                "sqrt" => Math.Sqrt(arguments[0]),
                "abs" => Math.Abs(arguments[0]),
                "pow" => Math.Pow(arguments[0], arguments[1]),
                "log" => arguments.Length > 1 ? Math.Log(arguments[0], arguments[1]) : Math.Log(arguments[0]),
                "log10" => Math.Log10(arguments[0]),
                "exp" => Math.Exp(arguments[0]),
                "sin" => Math.Sin(arguments[0]),
                "cos" => Math.Cos(arguments[0]),
                "tan" => Math.Tan(arguments[0]),
                "floor" => Math.Floor(arguments[0]),
                "ceil" => Math.Ceiling(arguments[0]),
                "round" => arguments.Length > 1
                    ? Math.Round(arguments[0], (int)arguments[1])
                    : Math.Round(arguments[0]),
                "min" => arguments.Min(),
                "max" => arguments.Max(),
                _ => throw new NotSupportedException($"Unknown function '{name}'."),
            };

            expression = expression.Remove(match.Index, match.Length)
                .Insert(match.Index, value.ToString("R", CultureInfo.InvariantCulture));
        }

        return expression;
    }

    // Only matches calls whose arguments contain no nested parentheses, so repeated passes
    // resolve the innermost call first and work outwards.
    [GeneratedRegex(@"(?<fn>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?<args>[^()]*)\)")]
    private static partial Regex FunctionCall();

    // ---------------------------------------------------------------- time

    [KernelFunction("GetCurrentDateTime")]
    [Description("Returns the current date and time. Call this whenever the answer depends on " +
                 "today - you have no other way to know it.")]
    public string GetCurrentDateTime(
        [Description("Optional IANA or Windows time zone id, e.g. Asia/Jakarta. Empty for local.")]
        string timeZone = "")
    {
        var now = DateTimeOffset.Now;
        var zoneName = TimeZoneInfo.Local.DisplayName;

        if (!string.IsNullOrWhiteSpace(timeZone))
        {
            try
            {
                var zone = TimeZoneInfo.FindSystemTimeZoneById(timeZone);
                now = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, zone);
                zoneName = zone.DisplayName;
            }
            catch (TimeZoneNotFoundException)
            {
                return $"Unknown time zone '{timeZone}'. Try Asia/Jakarta, UTC, or leave it empty.";
            }
        }

        return $"""
            Local   : {now:dddd, d MMMM yyyy HH:mm:ss} ({zoneName})
            ISO     : {now:O}
            UTC     : {DateTimeOffset.UtcNow:yyyy-MM-dd HH:mm:ss}Z
            Week    : {ISOWeek.GetWeekOfYear(now.DateTime)} of {now.Year}
            """;
    }

    [KernelFunction("CalculateDateDifference")]
    [Description("Calculates the span between two dates, or shifts a date by a number of days. " +
                 "Use it instead of counting days yourself.")]
    public string CalculateDateDifference(
        [Description("Start date, ISO format preferred, e.g. 2026-08-13.")] string startDate,
        [Description("End date. Leave empty to use today.")] string endDate = "",
        [Description("Optional number of days to add to the start date instead of comparing.")]
        int addDays = 0)
    {
        if (!DateTime.TryParse(startDate, CultureInfo.InvariantCulture, DateTimeStyles.None, out var start))
            return $"Could not read '{startDate}' as a date.";

        if (addDays != 0)
        {
            var shifted = start.AddDays(addDays);
            return $"{start:yyyy-MM-dd} {(addDays > 0 ? "+" : "-")} {Math.Abs(addDays)} days = " +
                   $"{shifted:yyyy-MM-dd} ({shifted:dddd})";
        }

        var end = DateTime.Today;
        if (!string.IsNullOrWhiteSpace(endDate)
            && !DateTime.TryParse(endDate, CultureInfo.InvariantCulture, DateTimeStyles.None, out end))
            return $"Could not read '{endDate}' as a date.";

        var span = end - start;
        return $"""
            From    : {start:yyyy-MM-dd} ({start:dddd})
            To      : {end:yyyy-MM-dd} ({end:dddd})
            Days    : {span.TotalDays:N0}
            Weeks   : {span.TotalDays / 7:N1}
            Years   : {span.TotalDays / 365.25:N2}
            """;
    }

    private static string Truncate(string text, int limit)
        => text.Length <= limit ? text : text[..limit] + $"\n... (truncated at {limit:N0} characters)";
}
