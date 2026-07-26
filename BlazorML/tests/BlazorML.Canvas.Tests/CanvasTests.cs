using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The designer canvas, driven through a real browser. Everything here is behaviour that only
/// exists once JS has run: D3 laying out nodes, edges routed between ports, drag, zoom, and the
/// round trip back into Blazor over the circuit.
/// </summary>
[Collection("canvas")]
public class CanvasTests(StudioFixture studio)
{
    // --------------------------------------------------------------- rendering

    [CanvasFact]
    public async Task The_seeded_experiment_paints_its_nodes_and_edges()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var nodes = await page.Locator(".node").CountAsync();
        var edges = await page.Locator("path.dz-edge").CountAsync();

        // The churn sample wires import → clean → split → train → score → evaluate plus an
        // algorithm, so both counts have to be real rather than "at least one".
        Assert.True(nodes >= 6, $"Only {nodes} nodes were painted.");
        Assert.True(edges >= 5, $"Only {edges} edges were drawn.");
    }

    [CanvasFact]
    public async Task Each_node_shows_its_label_and_its_ports()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var first = page.Locator(".node").First;

        Assert.False(string.IsNullOrWhiteSpace(await first.Locator(".node-title").InnerTextAsync()));
        Assert.True(await page.Locator(".port--out").CountAsync() > 0);
        Assert.True(await page.Locator(".port--in").CountAsync() > 0);
    }

    [CanvasFact]
    public async Task A_node_carries_the_pen_colour_of_its_category()
    {
        // Colour encodes category throughout the product; if the variable does not reach the
        // node the whole scheme silently stops meaning anything.
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var pen = await page.Locator(".node").First.EvaluateAsync<string>(
            "el => getComputedStyle(el).getPropertyValue('--pen')");

        Assert.False(string.IsNullOrWhiteSpace(pen));
    }

    [CanvasFact]
    public async Task The_palette_lists_every_module_grouped_by_category()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        Assert.Equal(63, await page.Locator(".palette-item").CountAsync());
        Assert.True(await page.Locator(".palette-group").CountAsync() >= 8);
    }

    // ------------------------------------------------------------ interaction

    [CanvasFact]
    public async Task Clicking_a_node_selects_it_and_opens_its_settings()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var node = page.Locator(".node").First;
        var label = await node.Locator(".node-title").InnerTextAsync();

        await node.ClickAsync();

        await page.WaitForSelectorAsync(".node.is-selected");
        await page.WaitForSelectorAsync(".inspector-body .field");

        Assert.Contains(label, await page.Locator(".inspector-head").InnerTextAsync());
    }

    [CanvasFact]
    public async Task Dragging_a_node_moves_it_on_the_canvas()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var node = page.Locator(".node").First;
        var before = await node.BoundingBoxAsync();

        await page.Mouse.MoveAsync(before!.X + 60, before.Y + 12);
        await page.Mouse.DownAsync();
        await page.Mouse.MoveAsync(before.X + 200, before.Y + 130, new MouseMoveOptions { Steps = 12 });
        await page.Mouse.UpAsync();

        var after = await node.BoundingBoxAsync();

        Assert.True(Math.Abs(after!.X - before.X) > 40, "The node did not move horizontally.");
        Assert.True(Math.Abs(after.Y - before.Y) > 40, "The node did not move vertically.");
    }

    [CanvasFact]
    public async Task Moving_a_node_marks_the_experiment_as_having_unsaved_changes()
    {
        // Proves the drag reached Blazor, not just the DOM: the save button is driven by the
        // component's state, which only changes if the interop callback arrived.
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var save = page.Locator("button:has-text('Tersimpan')");
        Assert.Equal(1, await save.CountAsync());

        var node = page.Locator(".node").First;
        var box = await node.BoundingBoxAsync();

        await page.Mouse.MoveAsync(box!.X + 60, box.Y + 12);
        await page.Mouse.DownAsync();
        await page.Mouse.MoveAsync(box.X + 160, box.Y + 90, new MouseMoveOptions { Steps = 10 });
        await page.Mouse.UpAsync();

        await page.WaitForSelectorAsync("button:has-text('Simpan')",
            new PageWaitForSelectorOptions { Timeout = 10000 });
    }

    [CanvasFact]
    public async Task Clicking_a_palette_module_adds_a_node()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var before = await page.Locator(".node").CountAsync();

        await page.FillAsync(".palette-search input", "Normalize");
        await page.Locator(".palette-item").First.ClickAsync();

        await page.WaitForFunctionAsync(
            $"document.querySelectorAll('.node').length === {before + 1}",
            null, new PageWaitForFunctionOptions { Timeout = 10000 });
    }

    [CanvasFact]
    public async Task Searching_the_palette_narrows_it_to_matching_modules()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.FillAsync(".palette-search input", "regresi");
        await page.WaitForFunctionAsync("document.querySelectorAll('.palette-item').length < 63");

        var count = await page.Locator(".palette-item").CountAsync();

        Assert.True(count is > 0 and < 20, $"The search matched {count} modules.");
    }

    [CanvasFact]
    public async Task Clicking_an_edge_removes_it()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var before = await page.Locator("path.dz-edge").CountAsync();

        // The clickable target is the wide transparent path laid over the visible line.
        await page.Locator("path.dz-hit").First.ClickAsync(new LocatorClickOptions { Force = true });

        await page.WaitForFunctionAsync(
            $"document.querySelectorAll('path.dz-edge').length === {before - 1}",
            null, new PageWaitForFunctionOptions { Timeout = 10000 });
    }

    [CanvasFact]
    public async Task Dragging_between_ports_creates_a_connection()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        // Start from a clean canvas so the two modules are unambiguous.
        await page.FillAsync(".palette-search input", "Enter Data Manually");
        await page.Locator(".palette-item").First.ClickAsync();

        await page.FillAsync(".palette-search input", "Normalize");
        await page.Locator(".palette-item").First.ClickAsync();

        await page.WaitForTimeoutAsync(500);

        var before = await page.Locator("path.dz-edge").CountAsync();

        var nodes = page.Locator(".node");
        var count = await nodes.CountAsync();

        var source = nodes.Nth(count - 2).Locator(".port--out").First;
        var target = nodes.Nth(count - 1).Locator(".port--in").First;

        var from = await source.BoundingBoxAsync();
        var to = await target.BoundingBoxAsync();

        await page.Mouse.MoveAsync(from!.X + from.Width / 2, from.Y + from.Height / 2);
        await page.Mouse.DownAsync();
        await page.Mouse.MoveAsync(to!.X + to.Width / 2, to.Y + to.Height / 2,
            new MouseMoveOptions { Steps = 15 });
        await page.Mouse.UpAsync();

        await page.WaitForFunctionAsync(
            $"document.querySelectorAll('path.dz-edge').length === {before + 1}",
            null, new PageWaitForFunctionOptions { Timeout = 10000 });
    }

    [CanvasFact]
    public async Task Zoom_controls_change_the_canvas_transform()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var world = page.Locator(".dz-world");
        var before = await world.EvaluateAsync<string>("el => el.style.transform");

        await page.ClickAsync("button[aria-label='Perbesar']");
        await page.WaitForTimeoutAsync(400);

        var after = await world.EvaluateAsync<string>("el => el.style.transform");

        Assert.NotEqual(before, after);
    }

    [CanvasFact]
    public async Task Fit_brings_every_node_into_view()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("button[aria-label='Paskan tampilan']");
        await page.WaitForTimeoutAsync(600);

        var viewport = await page.Locator(".dz-viewport").BoundingBoxAsync();
        var nodes = page.Locator(".node");
        var count = await nodes.CountAsync();

        for (var i = 0; i < count; i++)
        {
            var box = await nodes.Nth(i).BoundingBoxAsync();

            Assert.True(box!.X + box.Width > viewport!.X && box.X < viewport.X + viewport.Width,
                $"Node {i} sits outside the viewport horizontally after Fit.");
        }
    }

    // ------------------------------------------------------------------ theme

    [CanvasFact]
    public async Task The_theme_toggle_switches_the_document_and_the_canvas_follows()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var beforeSurface = await page.EvaluateAsync<string>(
            "() => getComputedStyle(document.body).backgroundColor");

        await page.ClickAsync(".topbar button[aria-label*='tema']");
        await page.WaitForTimeoutAsync(400);

        var theme = await page.EvaluateAsync<string>(
            "() => document.documentElement.getAttribute('data-theme')");

        var afterSurface = await page.EvaluateAsync<string>(
            "() => getComputedStyle(document.body).backgroundColor");

        Assert.Contains(theme, (string[])["light", "dark"]);
        Assert.NotEqual(beforeSurface, afterSurface);
    }

    [CanvasFact]
    public async Task The_chosen_theme_survives_a_reload()
    {
        // The inline script in App.razor applies the stored theme before first paint; without it
        // the page flashes the wrong theme, which reads as a bug.
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync(".topbar button[aria-label*='tema']");
        await page.WaitForTimeoutAsync(400);

        var chosen = await page.EvaluateAsync<string>(
            "() => document.documentElement.getAttribute('data-theme')");

        await page.ReloadAsync();
        await page.WaitForSelectorAsync(".node");

        var afterReload = await page.EvaluateAsync<string>(
            "() => document.documentElement.getAttribute('data-theme')");

        Assert.Equal(chosen, afterReload);
    }

    // ------------------------------------------------------------- chat panel

    [CanvasFact]
    public async Task The_assistant_panel_opens_and_closes()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        Assert.Equal(0, await page.Locator(".wicak").CountAsync());

        await page.ClickAsync("button[aria-label='Profesor Wicak']");
        await page.WaitForSelectorAsync(".wicak");

        await page.ClickAsync("button[aria-label='Profesor Wicak']");
        await page.WaitForSelectorAsync(".wicak", new PageWaitForSelectorOptions
        {
            State = WaitForSelectorState.Detached
        });
    }

    [CanvasFact]
    public async Task Without_credentials_the_assistant_says_what_to_configure()
    {
        // No provider key is seeded, so this is what a first-run user actually sees.
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("button[aria-label='Profesor Wicak']");

        // The panel's shell renders immediately but its state arrives from an async initialiser,
        // so waiting on .wicak alone reads the frame before the message is there.
        await page.WaitForSelectorAsync(".wicak .wicak-empty");

        var text = await page.Locator(".wicak").InnerTextAsync();

        // Regression test. Ollama's default endpoint used to count as a configured provider, so
        // a fresh install showed a chat panel that looked ready and failed on the first message.
        Assert.True(text.Contains("belum tersambung"),
            $"The panel presented itself as connected on a workspace with no credentials.\n---\n{text}\n---");

        Assert.Equal(1, await page.Locator(".wicak a[href='/pengaturan']").CountAsync());
    }

    // ------------------------------------------------------------- responsive

    [CanvasFact]
    public async Task On_a_narrow_viewport_the_page_does_not_scroll_sideways()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.SetViewportSizeAsync(390, 840);
        await page.WaitForTimeoutAsync(400);

        var overflows = await page.EvaluateAsync<bool>(
            "() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1");

        Assert.False(overflows, "The page scrolls horizontally on a phone-sized viewport.");
    }
}
