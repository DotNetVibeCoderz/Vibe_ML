using System.Net;
using System.Text.Json;
using BlazorML.Agents.Plugins;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Agents.Tests;

/// <summary>
/// The data functions are what stop the assistant guessing at column contents, and the utility
/// functions cover the arithmetic a language model is worst at. Neither needs a provider.
/// </summary>
public class DataPluginTests(WorkspaceFixture workspace) : IClassFixture<WorkspaceFixture>
{
    private DataPlugin Plugin => new(workspace.Scopes);

    [Fact]
    public async Task ListDatasets_returns_the_workspace_datasets_with_their_shape()
    {
        var body = JsonDocument.Parse(await Plugin.ListDatasets()).RootElement;

        Assert.True(body.GetArrayLength() >= 1);

        var dataset = body.EnumerateArray().First(d => d.GetProperty("Id").GetString() == workspace.DatasetId);

        Assert.Equal("Pelanggan Uji", dataset.GetProperty("Name").GetString());
        Assert.Equal(60, dataset.GetProperty("RowCount").GetInt32());
        Assert.Equal(3, dataset.GetProperty("ColumnCount").GetInt32());
    }

    [Fact]
    public async Task DescribeDataset_reports_each_column_with_its_type_and_range()
    {
        var body = JsonDocument.Parse(await Plugin.DescribeDataset(workspace.DatasetId)).RootElement;
        var columns = body.GetProperty("columns").EnumerateArray().ToList();

        Assert.Equal(3, columns.Count);

        var age = columns.First(c => c.GetProperty("Name").GetString() == "umur");

        Assert.Equal("Numeric", age.GetProperty("type").GetString());
        Assert.True(age.GetProperty("Min").GetDouble() >= 18);
        Assert.True(age.GetProperty("Max").GetDouble() < 70);
    }

    [Fact]
    public async Task QueryDataset_returns_real_rows_and_respects_the_limit()
    {
        var body = JsonDocument.Parse(await Plugin.QueryDataset(workspace.DatasetId, limit: 5)).RootElement;

        Assert.Equal(60, body.GetProperty("matched").GetInt32());
        Assert.Equal(5, body.GetProperty("returned").GetInt32());
        Assert.Equal(5, body.GetProperty("rows").GetArrayLength());
    }

    [Fact]
    public async Task QueryDataset_filters_numerically()
    {
        var body = JsonDocument.Parse(
            await Plugin.QueryDataset(workspace.DatasetId, filter: "umur > 40")).RootElement;

        var matched = body.GetProperty("matched").GetInt32();

        Assert.True(matched is > 0 and < 60);

        // Values come back as they are stored — text read from the file — so the assertion
        // parses rather than assuming a JSON number.
        foreach (var row in body.GetProperty("rows").EnumerateArray())
        {
            var age = double.Parse(row.GetProperty("umur").GetString()!,
                System.Globalization.CultureInfo.InvariantCulture);

            Assert.True(age > 40);
        }
    }

    [Fact]
    public async Task QueryDataset_can_narrow_to_named_columns()
    {
        var body = JsonDocument.Parse(
            await Plugin.QueryDataset(workspace.DatasetId, columns: "umur", limit: 2)).RootElement;

        foreach (var row in body.GetProperty("rows").EnumerateArray())
        {
            Assert.Single(row.EnumerateObject());
            Assert.True(row.TryGetProperty("umur", out _));
        }
    }

    /// <summary>
    /// This is the function that answers "is my data balanced?", which is the question that most
    /// often explains a disappointing model.
    /// </summary>
    [Fact]
    public async Task ValueCounts_reports_the_class_balance()
    {
        var body = JsonDocument.Parse(
            await Plugin.ValueCounts(workspace.DatasetId, "churn")).RootElement;

        var counts = body.GetProperty("counts");

        Assert.Equal(2, body.GetProperty("distinct").GetInt32());
        Assert.Equal(45, counts.GetProperty("0").GetInt32());
        Assert.Equal(15, counts.GetProperty("1").GetInt32());
    }

    [Fact]
    public async Task ValueCounts_on_an_unknown_column_lists_the_ones_that_exist()
    {
        var reply = await Plugin.ValueCounts(workspace.DatasetId, "tidak_ada");

        Assert.Contains("not a column", reply);
        Assert.Contains("umur", reply);
        Assert.Contains("kota", reply);
    }

    [Fact]
    public async Task An_unknown_dataset_id_comes_back_as_guidance_not_an_exception()
    {
        Assert.Contains("ListDatasets", await Plugin.DescribeDataset("tidak-ada"));
        Assert.Contains("no longer exists", await Plugin.QueryDataset("tidak-ada"));
    }
}

public class TimeAndMathPluginTests
{
    private readonly TimeAndMathPlugin _plugin = new();

    [Theory]
    [InlineData("2 + 2", 4)]
    [InlineData("(1200 * 0.85) / 12", 85)]
    [InlineData("100 / 8", 12.5)]
    [InlineData("2 * (3 + 4) - 1", 13)]
    public void Calculate_evaluates_arithmetic(string expression, double expected)
    {
        // Compared numerically: the decimal formatting of the result is not the contract.
        Assert.Equal(expected, double.Parse(_plugin.Calculate(expression),
            System.Globalization.CultureInfo.InvariantCulture), 6);
    }

    [Fact]
    public void Calculate_explains_an_expression_it_cannot_evaluate()
    {
        var reply = _plugin.Calculate("bukan ekspresi (((");

        Assert.Contains("not an expression", reply);
    }

    [Fact]
    public void DaysBetween_counts_the_days()
    {
        Assert.Equal("30 days", _plugin.DaysBetween("2026-01-01", "2026-01-31"));
    }

    [Fact]
    public void DaysBetween_says_what_format_it_wants_when_given_something_else()
    {
        Assert.Contains("yyyy-MM-dd", _plugin.DaysBetween("kemarin", "besok"));
    }

    [Fact]
    public void Statistics_computes_the_summary_figures()
    {
        var body = JsonDocument.Parse(_plugin.Statistics("10, 20, 30, 40")).RootElement;

        Assert.Equal(4, body.GetProperty("count").GetInt32());
        Assert.Equal(25, body.GetProperty("mean").GetDouble());
        Assert.Equal(10, body.GetProperty("min").GetDouble());
        Assert.Equal(40, body.GetProperty("max").GetDouble());
    }

    [Fact]
    public void Statistics_ignores_values_that_are_not_numbers()
    {
        var body = JsonDocument.Parse(_plugin.Statistics("10 abc 20")).RootElement;

        Assert.Equal(2, body.GetProperty("count").GetInt32());
    }

    [Fact]
    public void Statistics_says_so_when_there_is_nothing_to_summarise()
    {
        Assert.Contains("No numbers", _plugin.Statistics("tidak ada angka"));
    }

    [Fact]
    public void CurrentDateTime_honours_a_time_zone_and_rejects_a_bad_one()
    {
        Assert.NotEmpty(_plugin.CurrentDateTime("Asia/Jakarta"));
        Assert.Contains("not recognised", _plugin.CurrentDateTime("Bukan/Zona"));
    }
}

/// <summary>
/// The web functions are tested against a stub transport. What matters is the behaviour around
/// the call — the switches being honoured, the missing-key message, the shape of the summary —
/// not that Tavily is reachable from a test machine.
/// </summary>
public class WebPluginTests
{
    private static (WebPlugin Plugin, ISettingsService Settings) Build(
        ToolOptions options, HttpStatusCode status = HttpStatusCode.OK, string body = "{}")
    {
        var settings = new StubSettings(options);
        var factory = new StubHttpClientFactory(status, body);

        return (new WebPlugin(settings, factory), settings);
    }

    [Fact]
    public async Task SearchWeb_says_it_is_switched_off_when_it_is()
    {
        var (plugin, _) = Build(new ToolOptions { EnableWebSearch = false });

        Assert.Contains("switched off", await plugin.SearchWeb("apa saja"));
    }

    [Fact]
    public async Task SearchWeb_asks_for_a_key_rather_than_calling_without_one()
    {
        var (plugin, _) = Build(new ToolOptions { EnableWebSearch = true, TavilyApiKey = "" });

        var reply = await plugin.SearchWeb("apa saja");

        Assert.Contains("Tavily API key", reply);
        Assert.Contains("Settings", reply);
    }

    [Fact]
    public async Task SearchWeb_formats_the_answer_and_the_results()
    {
        const string payload = """
            {
              "answer": "Ringkasan singkat.",
              "results": [
                { "title": "Judul Satu", "url": "https://contoh.id/1", "content": "Cuplikan satu." }
              ]
            }
            """;

        var (plugin, _) = Build(
            new ToolOptions { EnableWebSearch = true, TavilyApiKey = "kunci" }, body: payload);

        var reply = await plugin.SearchWeb("apa saja");

        Assert.Contains("Ringkasan singkat.", reply);
        Assert.Contains("Judul Satu", reply);
        Assert.Contains("https://contoh.id/1", reply);
    }

    [Fact]
    public async Task SearchWeb_reports_a_rejected_key_instead_of_pretending_it_worked()
    {
        var (plugin, _) = Build(
            new ToolOptions { EnableWebSearch = true, TavilyApiKey = "salah" },
            HttpStatusCode.Unauthorized);

        var reply = await plugin.SearchWeb("apa saja");

        Assert.Contains("401", reply);
        Assert.Contains("Tavily API key", reply);
    }

    [Fact]
    public async Task ScrapeUrl_strips_scripts_and_styles_from_the_text()
    {
        const string html = """
            <html><head><style>.a{color:red}</style></head>
            <body><script>var x = 1;</script><p>Isi yang sebenarnya.</p></body></html>
            """;

        var (plugin, _) = Build(new ToolOptions { EnableUrlScraping = true }, body: html);

        var reply = await plugin.ScrapeUrl("https://contoh.id");

        Assert.Contains("Isi yang sebenarnya.", reply);
        Assert.DoesNotContain("var x = 1", reply);
        Assert.DoesNotContain("color:red", reply);
    }

    [Fact]
    public async Task ScrapeUrl_respects_the_switch()
    {
        var (plugin, _) = Build(new ToolOptions { EnableUrlScraping = false });

        Assert.Contains("switched off", await plugin.ScrapeUrl("https://contoh.id"));
    }

    [Fact]
    public async Task ReadFileFromUrl_truncates_rather_than_returning_an_unbounded_document()
    {
        var (plugin, _) = Build(new ToolOptions(), body: new string('x', 5000));

        var reply = await plugin.ReadFileFromUrl("https://contoh.id/besar.csv", maxCharacters: 500);

        Assert.Contains("truncated", reply);
        Assert.True(reply.Length < 700);
    }

    private sealed class StubSettings(ToolOptions options) : ISettingsService
    {
        public event Action<string>? SectionChanged;

        public Task<T> GetAsync<T>(string section, CancellationToken ct = default) where T : class, new() =>
            Task.FromResult(options as T ?? new T());

        public Task SaveAsync<T>(string section, T value, CancellationToken ct = default) where T : class
        {
            SectionChanged?.Invoke(section);
            return Task.CompletedTask;
        }

        public void Invalidate(string? section = null) { }
    }

    private sealed class StubHttpClientFactory(HttpStatusCode status, string body) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) => new(new StubHandler(status, body));

        private sealed class StubHandler(HttpStatusCode status, string body) : HttpMessageHandler
        {
            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request, CancellationToken cancellationToken) =>
                Task.FromResult(new HttpResponseMessage(status) { Content = new StringContent(body) });
        }
    }
}
