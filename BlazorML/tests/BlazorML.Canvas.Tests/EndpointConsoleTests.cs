using System.Text.Json;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using BlazorML.Web.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The Endpoint page's code samples and trial console, in a real browser.
/// <para>
/// The component tests cover what the page renders. What they cannot cover is the browser's own
/// behaviour: once a <c>&lt;textarea&gt;</c> has been typed in, it stops reflecting changes to its
/// child text node, so a payload rendered as element content would silently refuse to reset. That
/// is invisible to bUnit, which compares markup rather than what a browser would display.
/// </para>
/// </summary>
[Collection("canvas")]
public class EndpointConsoleTests(StudioFixture studio)
{
    /// <summary>
    /// Publishes an endpoint straight into the running app's database. Deliberately stopped: a
    /// live one would need its model file on disk too, and the console's request and response
    /// path is fully exercised by the 401 that a stopped endpoint returns. Scoring a real model
    /// over HTTP is covered by the API tests, which have the whole app to hand.
    /// </summary>
    private async Task<string> PublishAsync(string name)
    {
        var options = new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite($"Data Source={studio.DatabasePath}")
            .Options;

        await using var db = new AppDbContext(options);

        var model = new TrainedModel
        {
            Name = "churn-uji",
            Version = 1,
            Task = MlTask.BinaryClassification,
            Algorithm = "Boosted Decision Tree",
            LabelColumn = "beli",
            StorageKey = "models/tidak-ada.zip",
            InputSchemaJson = ModelInputSchema.Serialise([
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
        await db.SaveChangesAsync();

        var endpoints = new EndpointService(db);
        var (endpoint, key) = await endpoints.PublishAsync(model.Id, name, "Dibuat oleh tes", null);
        await endpoints.SetStatusAsync(endpoint.Id, EndpointStatus.Stopped);

        return key;
    }

    private async Task<StudioFixture.PageSession> OpenAsync(string name)
    {
        var key = await PublishAsync(name);
        var session = await studio.ShellPageAsync("/endpoint");

        await session.Page.Locator("article.card")
            .Filter(new LocatorFilterOptions { HasTextString = name })
            .First.WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        // Stored on the session so each test can reach its own key without republishing.
        Keys[name] = key;
        return session;
    }

    private static readonly Dictionary<string, string> Keys = [];

    private static ILocator Card(IPage page, string name) =>
        page.Locator("article.card").Filter(new LocatorFilterOptions { HasTextString = name }).First;

    /// <summary>
    /// Opens a card's console, pressing again if the first click landed before the circuit was
    /// listening. A click Blazor never received is a no-op, and it surfaces later as a confusing
    /// "no textarea" timeout rather than as the race it is.
    /// </summary>
    private static async Task<ILocator> OpenConsoleAsync(IPage page, ILocator card)
    {
        var button = card.Locator("button", new LocatorLocatorOptions { HasTextString = "Buka panel" });
        var body = card.Locator("textarea");

        await button.WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        for (var attempt = 1; attempt <= 3; attempt++)
        {
            await button.ClickAsync();

            try
            {
                await body.WaitForAsync(new LocatorWaitForOptions { Timeout = 8000 });
                return body;
            }
            catch (TimeoutException) when (attempt < 3)
            {
                await page.WaitForTimeoutAsync(1000);
            }
        }

        await body.WaitForAsync(new LocatorWaitForOptions { Timeout = 8000 });
        return body;
    }

    // ------------------------------------------------------------------- tabs

    [CanvasFact]
    public async Task Switching_language_swaps_the_snippet_in_the_browser()
    {
        var name = $"Tab {Guid.NewGuid():n}"[..12];
        await using var session = await OpenAsync(name);

        var card = Card(session.Page, name);
        var code = card.Locator(".codeblock pre");

        Assert.Contains("curl -X POST", await code.InnerTextAsync());

        await card.Locator(".tab", new LocatorLocatorOptions { HasTextString = "Python" }).ClickAsync();
        await session.Page.WaitForTimeoutAsync(250);

        var python = await code.InnerTextAsync();

        Assert.Contains("import requests", python);
        Assert.Contains("pendapatan", python);
        Assert.DoesNotContain("curl -X POST", python);
    }

    [CanvasFact]
    public async Task The_copy_button_confirms_and_puts_the_snippet_on_the_clipboard()
    {
        var name = $"Salin {Guid.NewGuid():n}"[..14];
        await using var session = await OpenAsync(name);
        var page = session.Page;

        await page.Context.GrantPermissionsAsync(["clipboard-read", "clipboard-write"]);

        var card = Card(page, name);
        await card.Locator(".codeblock-copy").ClickAsync();

        // The confirmation lands on the button that was pressed, not somewhere else on the page.
        await card.Locator(".codeblock-copy", new LocatorLocatorOptions { HasTextString = "Tersalin" })
            .WaitForAsync(new LocatorWaitForOptions { Timeout = 5000 });

        var clipboard = await page.EvaluateAsync<string>("() => navigator.clipboard.readText()");

        Assert.Contains("curl -X POST", clipboard);
        Assert.Contains("X-Api-Key", clipboard);
    }

    // ---------------------------------------------------------------- console

    /// <summary>
    /// The reason this suite exists for this feature. Type in the box, press the button that puts
    /// the sample back, and the box has to actually change on screen.
    /// </summary>
    [CanvasFact]
    public async Task Restoring_the_sample_replaces_what_the_user_typed()
    {
        var name = $"Isi {Guid.NewGuid():n}"[..12];
        await using var session = await OpenAsync(name);
        var page = session.Page;

        var card = Card(page, name);
        var body = await OpenConsoleAsync(page, card);

        Assert.Contains("pendapatan", await body.InputValueAsync());

        await body.FillAsync("{ \"rows\": [] }");
        Assert.Equal("{ \"rows\": [] }", await body.InputValueAsync());

        await card.Locator("button", new LocatorLocatorOptions { HasTextString = "Kembalikan contoh" })
            .ClickAsync();
        await page.WaitForTimeoutAsync(300);

        // InputValueAsync reads the live value, which is exactly what a child text node would fail
        // to update after the fill above.
        var restored = await body.InputValueAsync();

        Assert.Contains("pendapatan", restored);
        Assert.Contains("Bandung", restored);
    }

    [CanvasFact]
    public async Task Sending_a_request_shows_the_status_the_api_answered_with()
    {
        var name = $"Kirim {Guid.NewGuid():n}"[..14];
        await using var session = await OpenAsync(name);
        var page = session.Page;

        var card = Card(page, name);
        await OpenConsoleAsync(page, card);

        await card.Locator("input[type=password]").FillAsync(Keys[name]);
        await card.Locator("button", new LocatorLocatorOptions { HasTextString = "Kirim permintaan" })
            .ClickAsync();

        var status = card.Locator(".tag--danger");
        await status.WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        // This endpoint is stopped, so 401 is the correct answer and the console reports it as a
        // response rather than as a failure to reach anything.
        Assert.Contains("401", await status.InnerTextAsync());
        Assert.Contains("Unauthorised", await card.Locator(".codeblock pre").Last.InnerTextAsync());
    }

    [CanvasFact]
    public async Task Sending_without_a_key_is_refused_without_a_round_trip()
    {
        var name = $"Kosong {Guid.NewGuid():n}"[..15];
        await using var session = await OpenAsync(name);
        var page = session.Page;

        var card = Card(page, name);
        await OpenConsoleAsync(page, card);

        await card.Locator("button", new LocatorLocatorOptions { HasTextString = "Kirim permintaan" })
            .ClickAsync();

        var notice = card.Locator(".notice--danger");
        await notice.WaitForAsync(new LocatorWaitForOptions { Timeout = 10000 });

        Assert.Contains("kunci API", await notice.InnerTextAsync());
    }

    [CanvasFact]
    public async Task The_page_does_not_scroll_sideways_with_a_wide_snippet_on_a_phone()
    {
        var name = $"Sempit {Guid.NewGuid():n}"[..15];
        await using var session = await OpenAsync(name);
        var page = session.Page;

        await page.SetViewportSizeAsync(390, 840);
        await page.WaitForTimeoutAsync(400);

        // A curl command is one long line. It has to scroll inside its own block, never take the
        // page with it.
        var overflows = await page.EvaluateAsync<bool>(
            "() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1");

        Assert.False(overflows, "The endpoint page scrolls horizontally on a phone-sized viewport.");
    }
}
