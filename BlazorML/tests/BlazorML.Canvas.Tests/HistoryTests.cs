using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The run history and version pages. Both existed only as services until now — the data was
/// being recorded and nothing ever showed it — so these check the whole path from the canvas
/// through to a restored version.
/// </summary>
[Collection("canvas")]
public class HistoryTests(StudioFixture studio)
{
    private async Task<StudioFixture.PageSession> HistoryPageAsync(string tab = "Run")
    {
        var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.WaitForSelectorAsync(".tabs");

        if (tab != "Run")
        {
            await page.ClickAsync($".tab:has-text('{tab}')");
        }

        return session;
    }

    /// <summary>
    /// Runs the experiment and waits for it to actually finish. Waiting on the Jalankan button
    /// alone races the start: it is still on screen for the moment before the run begins, so the
    /// wait returns immediately and the test reads the history before anything was recorded.
    /// </summary>
    private static async Task RunAndWaitAsync(IPage page)
    {
        await page.ClickAsync("button:has-text('Jalankan')");

        // First the running badge appears…
        await page.WaitForSelectorAsync(".tag--warn:has-text('Berjalan')",
            new PageWaitForSelectorOptions { Timeout = 30000 });

        // …then it goes away again.
        await page.WaitForSelectorAsync(".tag--warn:has-text('Berjalan')", new PageWaitForSelectorOptions
        {
            State = WaitForSelectorState.Detached,
            Timeout = 180000
        });
    }

    [CanvasFact]
    public async Task The_designer_offers_a_way_into_the_history()
    {
        await using var session = await studio.DesignerPageAsync();

        Assert.Equal(1, await session.Page.Locator("a[aria-label='Riwayat run dan versi']").CountAsync());
    }

    [CanvasFact]
    public async Task A_never_run_experiment_says_so_and_points_back_to_the_canvas()
    {
        // The seeded samples have never been run in this fresh workspace, so this is the state a
        // new user meets first.
        await using var session = await HistoryPageAsync();
        var page = session.Page;

        await page.WaitForSelectorAsync(".page");

        var text = await page.Locator(".page").InnerTextAsync();

        Assert.True(text.Contains("belum pernah dijalankan"),
            $"A freshly copied experiment did not show the empty state.\n---\n{text}\n---");

        Assert.Equal(1, await page.Locator(".empty a[href*='/eksperimen/']").CountAsync());
    }

    [CanvasFact]
    public async Task Running_the_experiment_puts_it_in_the_history_with_its_metrics()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await RunAndWaitAsync(page);

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.WaitForSelectorAsync("table");

        var rows = await page.Locator("tbody tr").CountAsync();
        Assert.True(rows >= 1, "The run did not appear in the history.");

        // A metric column proves the numbers survived the round trip into the run record.
        var headers = await page.Locator("thead th").AllInnerTextsAsync();
        Assert.True(headers.Count > 3, $"Only these columns appeared: {string.Join(", ", headers)}");
    }

    [CanvasFact]
    public async Task The_comparison_chart_draws_once_there_is_a_run_to_compare()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await RunAndWaitAsync(page);

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.WaitForSelectorAsync(".chart-plot svg", new PageWaitForSelectorOptions { Timeout = 20000 });

        Assert.True(await page.Locator(".chart-plot svg rect").CountAsync() >= 1);

        // Changing the metric redraws rather than leaving the previous one on screen.
        var options = await page.Locator("#metrik option").CountAsync();

        if (options > 1)
        {
            var second = await page.Locator("#metrik option").Nth(1).GetAttributeAsync("value");
            await page.SelectOptionAsync("#metrik", second!);
            await page.WaitForTimeoutAsync(600);

            Assert.Contains(second!, await page.Locator(".chart figcaption").InnerTextAsync());
        }
    }

    // ------------------------------------------------------------------ versions

    [CanvasFact]
    public async Task The_version_tab_lists_the_history_and_marks_the_current_one()
    {
        await using var session = await HistoryPageAsync("Versi");
        var page = session.Page;

        await page.WaitForSelectorAsync("tbody tr");

        Assert.True(await page.Locator("tbody tr").CountAsync() >= 1);
        Assert.Equal(1, await page.Locator(".tag--ok:has-text('sekarang')").CountAsync());
    }

    [CanvasFact]
    public async Task Editing_the_canvas_adds_a_version()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.ClickAsync(".tab:has-text('Versi')");
        await page.WaitForSelectorAsync("tbody tr");

        var before = await page.Locator("tbody tr").CountAsync();

        await page.GotoAsync(page.Url.Replace("/riwayat", string.Empty));
        await page.WaitForSelectorAsync(".node");

        await page.FillAsync(".palette-search input", "Normalize");
        await page.Locator(".palette-item").First.ClickAsync();
        await page.ClickAsync("button:has-text('Simpan')");
        await page.WaitForSelectorAsync("button:has-text('Tersimpan')");

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.ClickAsync(".tab:has-text('Versi')");

        await page.WaitForFunctionAsync(
            $"document.querySelectorAll('tbody tr').length === {before + 1}",
            null, new PageWaitForFunctionOptions { Timeout = 15000 });
    }

    /// <summary>
    /// Restoring is what makes the version history worth keeping. It has to be append-only: the
    /// version you restored away from must still be there afterwards.
    /// </summary>
    [CanvasFact]
    public async Task Restoring_a_version_brings_the_canvas_back_and_keeps_the_history()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var originalNodes = await page.Locator(".node").CountAsync();

        // Add a module and save, so there is something to undo.
        await page.FillAsync(".palette-search input", "Normalize");
        await page.Locator(".palette-item").First.ClickAsync();
        await page.ClickAsync("button:has-text('Simpan')");
        await page.WaitForSelectorAsync("button:has-text('Tersimpan')");

        Assert.Equal(originalNodes + 1, await page.Locator(".node").CountAsync());

        await page.ClickAsync("a[aria-label='Riwayat run dan versi']");
        await page.ClickAsync(".tab:has-text('Versi')");
        await page.WaitForSelectorAsync("tbody tr");

        var versionsBefore = await page.Locator("tbody tr").CountAsync();

        // Restore the version before the edit — the second row, since the newest is on top.
        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();
        await page.Locator("tbody tr").Nth(1).Locator("button:has-text('Kembalikan')").ClickAsync();

        await page.WaitForSelectorAsync(".notice--ok", new PageWaitForSelectorOptions { Timeout = 15000 });

        // Append-only: restoring added a version rather than removing any.
        Assert.Equal(versionsBefore + 1, await page.Locator("tbody tr").CountAsync());

        await page.GotoAsync(page.Url.Replace("/riwayat", string.Empty));
        await page.WaitForSelectorAsync(".node");

        Assert.Equal(originalNodes, await page.Locator(".node").CountAsync());
    }
}
