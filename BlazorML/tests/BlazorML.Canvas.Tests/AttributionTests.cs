using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The credit line in the app shell.
/// <para>
/// Browser tests rather than component tests, for two reasons. It lives in <c>MainLayout</c>, and
/// a layout does not inherit a page's render mode — the same trap that left the theme toggle
/// inert. And adding a bar to the shell changes the height every full-viewport page has to work
/// within, which is only true or false once a real browser has laid the page out.
/// </para>
/// </summary>
[Collection("canvas")]
public class AttributionTests(StudioFixture studio)
{
    [CanvasFact]
    public async Task The_credit_appears_on_the_shell_for_a_signed_in_user()
    {
        await using var session = await studio.ShellPageAsync();
        var page = session.Page;

        var credit = page.Locator("footer.credit");

        await credit.WaitForAsync(new LocatorWaitForOptions { Timeout = 15000 });

        var text = await credit.InnerTextAsync();

        Assert.Contains("Gravicode Studios", text);
        Assert.Contains("Kang Fadhil", text);
    }

    [CanvasTheory]
    [InlineData("/dataset")]
    [InlineData("/eksperimen")]
    [InlineData("/model")]
    [InlineData("/endpoint")]
    [InlineData("/galeri")]
    [InlineData("/pengaturan")]
    public async Task The_credit_is_on_every_page_of_the_shell(string route)
    {
        await using var session = await studio.ShellPageAsync(route);

        // "In the app shell" has to mean every page, not the one page it was added to.
        Assert.Equal(1, await session.Page.Locator("footer.credit").CountAsync());
    }

    [CanvasFact]
    public async Task The_sign_in_page_credits_the_same_names()
    {
        var context = await studio.Browser!.NewContextAsync();
        await using var session = new StudioFixture.PageSession(context, await context.NewPageAsync());

        await session.Page.GotoAsync($"{studio.BaseUrl}/masuk");

        var text = await session.Page.Locator("footer.credit").InnerTextAsync();

        // Both surfaces read the same configured values, so the two spellings cannot drift.
        Assert.Contains("Gravicode Studios", text);
        Assert.Contains("Kang Fadhil", text);
    }

    /// <summary>
    /// The designer sizes its canvas against the viewport. The credit bar sits outside that grid,
    /// so its height has to be subtracted — miss it and a surface meant to pan grows a scrollbar.
    /// </summary>
    [CanvasFact]
    public async Task Adding_the_credit_did_not_give_the_designer_a_scrollbar()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.WaitForTimeoutAsync(500);

        var overflow = await page.EvaluateAsync<int>(
            "() => document.documentElement.scrollHeight - document.documentElement.clientHeight");

        Assert.True(overflow <= 1, $"The designer overflows its viewport by {overflow}px.");
    }

    /// <summary>
    /// Found while measuring the page for the test above: <c>SectionOutlet</c> and
    /// <c>SectionContent</c> were being emitted as unknown HTML elements, so every page's title
    /// and action buttons rendered at the top of the body instead of in the top bar. It read as
    /// deliberate because the block landed immediately under the bar, like a second row.
    /// </summary>
    [CanvasFact]
    public async Task The_page_title_and_its_actions_are_in_the_top_bar()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var topbar = await page.Locator(".topbar").InnerTextAsync();

        Assert.Contains("Prediksi churn pelanggan", topbar);
        Assert.Contains("Jalankan", topbar);

        // Razor does not warn about an unresolved component; it renders the tag verbatim. The only
        // way to notice is to look for the tag in the DOM.
        Assert.Equal(0, await page.Locator("sectioncontent, sectionoutlet").CountAsync());
    }

    [CanvasFact]
    public async Task The_credit_stays_below_the_content_rather_than_covering_it()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;

        var credit = await page.Locator("footer.credit").BoundingBoxAsync();
        var content = await page.Locator("main").BoundingBoxAsync();

        Assert.NotNull(credit);
        Assert.NotNull(content);

        // On a page shorter than the viewport the credit should be pushed to the bottom, not left
        // floating directly under the last card.
        Assert.True(credit!.Y >= content!.Y + content.Height - 1,
            $"The credit starts at {credit.Y:F0}px, inside content ending at " +
            $"{content.Y + content.Height:F0}px.");
    }
}
