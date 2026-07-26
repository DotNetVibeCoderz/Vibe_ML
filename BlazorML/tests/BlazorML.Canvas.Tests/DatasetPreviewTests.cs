using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The dataset row preview against the seeded workspace, in a browser.
/// <para>
/// The component tests use a stub service and a hand-built table. This one reads a real seeded
/// CSV off real storage and puts it on a real screen, which is the only way to see the parts that
/// only exist there: that a wide table scrolls inside its own frame instead of taking the page
/// with it, and that the header stays put while the rows move.
/// </para>
/// </summary>
[Collection("canvas")]
public class DatasetPreviewTests(StudioFixture studio)
{
    private static ILocator Dialog(IPage page) => page.Locator(".dialog");

    private static async Task<ILocator> OpenPreviewAsync(IPage page)
    {
        var card = page.Locator("article.card").First;
        await card.WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        var button = card.Locator("button", new LocatorLocatorOptions { HasTextString = "Lihat data" });
        var dialog = Dialog(page);

        // Clicking before the circuit is listening is a no-op that surfaces later as a confusing
        // "no dialog" timeout rather than as the race it is.
        for (var attempt = 1; attempt <= 3; attempt++)
        {
            await button.ClickAsync();

            try
            {
                await dialog.Locator("tbody tr").First
                    .WaitForAsync(new LocatorWaitForOptions { Timeout = 8000 });

                return dialog;
            }
            catch (TimeoutException) when (attempt < 3)
            {
                await page.WaitForTimeoutAsync(1000);
            }
        }

        await dialog.Locator("tbody tr").First.WaitForAsync(new LocatorWaitForOptions { Timeout = 8000 });
        return dialog;
    }

    [CanvasFact]
    public async Task A_seeded_dataset_shows_its_real_rows()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var dialog = await OpenPreviewAsync(session.Page);

        var rows = await dialog.Locator("tbody tr").CountAsync();
        var headers = await dialog.Locator("thead th").CountAsync();

        Assert.InRange(rows, 1, 25);

        // The row-number column plus at least one real column.
        Assert.True(headers >= 2, $"Only {headers} headers were drawn.");
        Assert.False(string.IsNullOrWhiteSpace(await dialog.Locator("tbody tr").First.InnerTextAsync()));
    }

    [CanvasFact]
    public async Task Asking_for_more_rows_reads_more_of_the_file()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;
        var dialog = await OpenPreviewAsync(page);

        var before = await dialog.Locator("tbody tr").CountAsync();

        await dialog.Locator("select").SelectOptionAsync("500");
        await page.WaitForTimeoutAsync(700);

        var after = await dialog.Locator("tbody tr").CountAsync();

        // The seeded samples are small, so this either grows or was already the whole file — what
        // must not happen is coming back with fewer rows than before.
        Assert.True(after >= before, $"Row count fell from {before} to {after}.");
    }

    [CanvasFact]
    public async Task The_two_tabs_swap_between_the_rows_and_the_column_profile()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;
        var dialog = await OpenPreviewAsync(page);

        await dialog.Locator(".tab", new LocatorLocatorOptions { HasTextString = "Kolom" }).ClickAsync();
        await page.WaitForTimeoutAsync(300);

        // The design system uppercases table headers, and innerText reflects text-transform,
        // so the comparison has to ignore case rather than assume the source spelling.
        Assert.Contains("rata-rata", (await dialog.InnerTextAsync()).ToLowerInvariant());

        await dialog.Locator(".tab", new LocatorLocatorOptions { HasTextString = "Data" }).ClickAsync();
        await page.WaitForTimeoutAsync(300);

        Assert.DoesNotContain("rata-rata", (await dialog.InnerTextAsync()).ToLowerInvariant());
    }

    /// <summary>
    /// A dataset can be far wider than the dialog. The table has to scroll inside its own frame;
    /// a page that scrolls sideways instead loses the navigation rail off the left edge.
    /// </summary>
    [CanvasFact]
    public async Task A_wide_table_scrolls_inside_its_frame_not_the_page()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;

        await OpenPreviewAsync(page);

        var scrollable = await page.EvaluateAsync<bool>(
            "() => { const w = document.querySelector('.dialog .table-wrap'); " +
            "return getComputedStyle(w).overflowX === 'auto'; }");

        var pageOverflows = await page.EvaluateAsync<bool>(
            "() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1");

        Assert.True(scrollable, "The preview table cannot scroll on its own.");
        Assert.False(pageOverflows, "The dataset page scrolls horizontally with the preview open.");
    }

    [CanvasFact]
    public async Task The_column_names_stay_visible_while_the_rows_scroll()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;
        var dialog = await OpenPreviewAsync(page);

        await dialog.Locator("select").SelectOptionAsync("500");
        await page.WaitForTimeoutAsync(700);

        var header = dialog.Locator("thead th").Nth(1);
        var before = await header.BoundingBoxAsync();

        var scrolled = await page.EvaluateAsync<double>(
            "() => { const w = document.querySelector('.dialog .table-wrap'); " +
            "w.scrollTop = w.scrollHeight; return w.scrollTop; }");
        await page.WaitForTimeoutAsync(300);

        var after = await header.BoundingBoxAsync();

        Assert.NotNull(before);
        Assert.NotNull(after);

        // Checked first, because without it this test is vacuous: with no height cap the frame
        // never overflows, nothing scrolls, and "the header did not move" is trivially true.
        Assert.True(scrolled > 0,
            "The preview frame did not scroll, so nothing about the header was proven.");

        // Sticky only works if the frame is the scroll container. Otherwise the dialog scrolls
        // instead and the column names are the first thing to leave.
        Assert.True(Math.Abs(after!.Y - before!.Y) < 2,
            $"The header moved from {before.Y:F0}px to {after.Y:F0}px while the rows scrolled.");
    }

    [CanvasFact]
    public async Task Closing_and_reopening_shows_the_rows_again()
    {
        await using var session = await studio.ShellPageAsync("/dataset");
        var page = session.Page;
        var dialog = await OpenPreviewAsync(page);

        await dialog.Locator("button[aria-label=Tutup]").ClickAsync();
        await page.WaitForTimeoutAsync(300);

        Assert.Equal(0, await Dialog(page).CountAsync());

        var reopened = await OpenPreviewAsync(page);

        Assert.True(await reopened.Locator("tbody tr").CountAsync() > 0);
    }
}
