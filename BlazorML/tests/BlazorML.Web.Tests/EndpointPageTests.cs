using System.Text.Json;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using BlazorML.Web.Services;
using Bunit;
using Bunit.TestDoubles;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Page = BlazorML.Web.Components.Pages.Endpoint;

namespace BlazorML.Web.Tests;

/// <summary>
/// The Endpoint page itself.
/// <para>
/// Everything the page adds is generated per endpoint and switched by clicking — the language
/// tabs, the console, the copy buttons. The services underneath are covered elsewhere; what is
/// only true once the component renders is that the right snippet reaches the screen and that the
/// controls are wired to the endpoint they sit on.
/// </para>
/// </summary>
public sealed class EndpointPageTests : TestContext
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"blazorml-endpoint-{Guid.NewGuid():n}");

    public EndpointPageTests()
    {
        Directory.CreateDirectory(_root);

        Services.AddDbContext<AppDbContext>(b =>
            b.UseSqlite($"Data Source={Path.Combine(_root, "page.db")}"));

        Services.AddScoped<EndpointService>();
        Services.AddScoped<EndpointTester>();
        Services.AddSingleton<IHttpClientFactory>(new UnusedClients());

        this.AddTestAuthorization().SetAuthorized("admin@gravicode.com");
        JSInterop.Mode = JSRuntimeMode.Loose;
        JSInterop.Setup<bool>("blazorml.copyText", _ => true).SetResult(true);

        using var scope = Services.CreateScope();
        scope.ServiceProvider.GetRequiredService<AppDbContext>().Database.EnsureCreated();
    }

    /// <summary>Publishes a model with the enriched schema the page reads.</summary>
    private ServiceEndpoint Publish(string name = "Prediksi churn", bool live = true,
        string? schemaJson = null)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var model = new TrainedModel
        {
            Name = "churn",
            Version = 1,
            Task = MlTask.BinaryClassification,
            Algorithm = "Boosted Decision Tree",
            LabelColumn = "beli",
            StorageKey = "models/churn.zip",
            InputSchemaJson = schemaJson ?? ModelInputSchema.Serialise([
                new ModelInputField
                {
                    Name = "pendapatan",
                    Type = ColumnDataType.Numeric,
                    Example = JsonSerializer.SerializeToElement(72.5)
                },
                new ModelInputField
                {
                    Name = "kota",
                    Type = ColumnDataType.Categorical,
                    Example = JsonSerializer.SerializeToElement("Bandung")
                }
            ])
        };

        db.TrainedModels.Add(model);
        db.SaveChanges();

        var endpoints = scope.ServiceProvider.GetRequiredService<EndpointService>();
        var (endpoint, _) = endpoints.PublishAsync(model.Id, name, "Endpoint uji", null)
            .GetAwaiter().GetResult();

        if (!live)
        {
            endpoints.SetStatusAsync(endpoint.Id, EndpointStatus.Stopped).GetAwaiter().GetResult();
        }

        return endpoint;
    }

    // ------------------------------------------------------------------ empty

    [Fact]
    public void With_nothing_published_the_page_says_what_to_do_next()
    {
        var page = RenderComponent<Page>();

        Assert.Contains("Belum ada endpoint", page.Markup);

        // An empty screen is an invitation to act, so it carries the way out of it.
        Assert.Contains("/model", page.Find(".empty a").GetAttribute("href"));
        Assert.Empty(page.FindAll(".codeblock"));
    }

    // ------------------------------------------------------------------- tabs

    [Fact]
    public void All_four_languages_are_offered_and_curl_is_shown_first()
    {
        Publish();

        var page = RenderComponent<Page>();
        var tabs = page.FindAll(".tabs .tab").Select(t => t.TextContent.Trim()).ToList();

        Assert.Equal(["cURL", "C#", "Python", "Node.js"], tabs);
        Assert.Contains("curl -X POST", page.Find(".codeblock pre").TextContent);
    }

    [Theory]
    [InlineData("C#", "HttpClient")]
    [InlineData("Python", "import requests")]
    [InlineData("Node.js", "await fetch")]
    public void Choosing_a_language_swaps_the_snippet(string tab, string expected)
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll(".tabs .tab").First(t => t.TextContent.Trim() == tab).Click();

        var code = page.Find(".codeblock pre").TextContent;

        Assert.Contains(expected, code);
        Assert.DoesNotContain("curl -X POST", code);
    }

    [Fact]
    public void The_snippet_carries_this_endpoint_columns_not_a_placeholder_schema()
    {
        var endpoint = Publish();

        var page = RenderComponent<Page>();
        var code = page.Find(".codeblock pre").TextContent;

        Assert.Contains($"/api/v1/score/{endpoint.Slug}", code);
        Assert.Contains("pendapatan", code);
        Assert.Contains("Bandung", code);
        Assert.DoesNotContain("kolom1", code);
    }

    [Fact]
    public void A_model_with_no_recorded_schema_says_so_rather_than_inventing_columns()
    {
        Publish(schemaJson: "[]");

        var page = RenderComponent<Page>();

        Assert.Contains("tidak mencatat kolomnya", page.Markup);
    }

    // ---------------------------------------------------------------- console

    [Fact]
    public void The_console_is_closed_until_asked_for()
    {
        Publish();

        var page = RenderComponent<Page>();

        // Several endpoints on one page, each with a console open, is a wall rather than a list.
        Assert.Empty(page.FindAll("textarea"));

        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        Assert.Single(page.FindAll("textarea"));
    }

    [Fact]
    public void The_console_opens_on_the_generated_payload()
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        var body = page.Find("textarea").GetAttribute("value")!;

        using var parsed = JsonDocument.Parse(body);
        var row = parsed.RootElement.GetProperty("rows")[0];

        Assert.Equal(72.5, row.GetProperty("pendapatan").GetDouble());
        Assert.Equal("Bandung", row.GetProperty("kota").GetString());
    }

    [Fact]
    public void Sending_without_a_key_says_which_button_makes_one()
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();
        page.FindAll("button").First(b => b.TextContent.Contains("Kirim permintaan")).Click();

        var notice = page.Find(".notice--danger").TextContent;

        Assert.Contains("kunci API", notice, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Kunci baru", notice);
    }

    /// <summary>
    /// The payload is editable, so it has to be the edited text that gets sent — a console that
    /// quietly posts the original sample would be worse than one that could not be edited at all.
    /// </summary>
    [Fact]
    public void What_is_typed_into_the_body_is_what_gets_sent()
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        page.Find("input[type=password]").Input("bml_kunci");
        page.Find("textarea").Input("{ bukan json }");
        page.FindAll("button").First(b => b.TextContent.Contains("Kirim permintaan")).Click();

        // The complaint is about the typed text, and it was reached without any HTTP call — the
        // factory in this fixture throws if one is attempted.
        Assert.Contains("belum valid", page.Find(".notice--danger").TextContent);
    }

    [Fact]
    public void Restoring_the_sample_puts_the_generated_payload_back()
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        page.Find("textarea").Input("{}");
        page.FindAll("button").First(b => b.TextContent.Contains("Kembalikan contoh")).Click();

        Assert.Contains("pendapatan", page.Find("textarea").GetAttribute("value"));
    }

    [Fact]
    public void A_stopped_endpoint_warns_before_the_call_rather_than_after_the_401()
    {
        Publish(live: false);

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        Assert.Contains("Aktifkan dulu", page.Find(".notice--warn").TextContent);
    }

    [Fact]
    public void Making_a_key_shows_it_once_and_opens_the_console_it_belongs_to()
    {
        Publish();

        var page = RenderComponent<Page>();
        page.FindAll("button").First(b => b.TextContent.Contains("Kunci baru")).Click();

        Assert.Contains("bml_", page.Find(".notice--ok").TextContent);

        // The one moment the plain key exists, so it is put where it is about to be needed.
        Assert.Single(page.FindAll("textarea"));
        Assert.StartsWith("bml_", page.Find("input[type=password]").GetAttribute("value"));
    }

    // ---------------------------------------------------------------- copying

    [Fact]
    public void The_copy_button_confirms_on_itself()
    {
        Publish();

        var page = RenderComponent<Page>();
        var copy = page.Find(".codeblock-copy");

        Assert.Contains("Salin", copy.TextContent);

        copy.Click();

        // Answered where the question was asked, rather than by a toast somewhere else.
        Assert.Contains("Tersalin", page.Find(".codeblock-copy").TextContent);
    }

    [Fact]
    public void Each_endpoint_gets_its_own_tabs_console_and_copy_button()
    {
        Publish("Prediksi churn");
        Publish("Prediksi harga");

        var page = RenderComponent<Page>();

        Assert.Equal(2, page.FindAll("article.card").Count);
        Assert.Equal(2, page.FindAll(".codeblock").Count);
        Assert.Equal(8, page.FindAll(".tabs .tab").Count);

        // Opening one console must not open the other's.
        page.FindAll("button").First(b => b.TextContent.Contains("Buka panel")).Click();

        Assert.Single(page.FindAll("textarea"));
    }

    /// <summary>The page never sends a trial call in these tests, so its client is never built.</summary>
    private sealed class UnusedClients : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) =>
            throw new InvalidOperationException("No trial call should be made in a component test.");
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A file SQLite still holds is not worth failing a run over.
        }
    }
}
