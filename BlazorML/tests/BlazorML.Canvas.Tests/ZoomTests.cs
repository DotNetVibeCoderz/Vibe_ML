using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// Pan and zoom keeping the two canvas layers together.
/// <para>
/// The canvas is deliberately split: nodes are HTML, edges are SVG. That buys node chrome from
/// the app's own CSS and D3 path interpolation for the run trace, but it means every view change
/// has to be applied twice, in two different coordinate systems, and nothing in the DOM enforces
/// that the two agree. Counting nodes and edges cannot see them disagree — only measuring can.
/// </para>
/// </summary>
[Collection("canvas")]
public class ZoomTests(StudioFixture studio)
{
    /// <summary>
    /// Every edge endpoint, in world units, to the nearest port it could belong to. Dividing by
    /// the current scale keeps the number comparable across zoom levels: a fixed pixel budget
    /// would pass at 0.25× for the wrong reason and fail at 2.5× for no reason.
    /// </summary>
    private const string WorstGap = @"() => {
        const layer = document.querySelector('.dz-edge-layer');
        const k = layer.getScreenCTM().a;

        const ports = [...document.querySelectorAll('.dz-nodes .port')].map(p => {
            const r = p.getBoundingClientRect();
            return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        });

        if (!ports.length) return -1;

        let worst = 0;

        for (const path of document.querySelectorAll('path.dz-edge')) {
            const m = path.getScreenCTM();
            const at = l => {
                const p = path.getPointAtLength(l);
                return { x: p.x * m.a + p.y * m.c + m.e, y: p.x * m.b + p.y * m.d + m.f };
            };

            for (const end of [at(0), at(path.getTotalLength())]) {
                const gap = Math.min(...ports.map(p => Math.hypot(p.x - end.x, p.y - end.y)));
                worst = Math.max(worst, gap / k);
            }
        }

        return worst;
    }";

    /// <summary>
    /// The arrow head is inset from the port by its marker, and the route leaves the port with a
    /// short straight run, so the endpoints are near the ports rather than exactly on them. This
    /// budget covers that and nothing more — the defect this guards against detached the edges by
    /// well over a hundred pixels after a single zoom step.
    /// </summary>
    private const double Budget = 30;

    private static async Task AssertAttachedAsync(IPage page, string after)
    {
        var gap = await page.EvaluateAsync<double>(WorstGap);

        Assert.True(gap >= 0, "No ports were found, so the measurement proves nothing.");
        Assert.True(gap <= Budget,
            $"After {after}, an edge ended {gap:F0} world px from the nearest port.");
    }

    [CanvasFact]
    public async Task Edges_stay_attached_to_their_ports_through_zooming_in_and_out()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.WaitForTimeoutAsync(400);
        await AssertAttachedAsync(page, "loading");

        // In twice, out four times, back in twice: covers both sides of 1× and returns to it.
        var steps = new[]
        {
            ("Perbesar", 2), ("Perkecil", 4), ("Perbesar", 2)
        };

        foreach (var (label, times) in steps)
        {
            for (var i = 0; i < times; i++)
            {
                await page.ClickAsync($"button[aria-label='{label}']");
                await page.WaitForTimeoutAsync(300);

                await AssertAttachedAsync(page, $"{label.ToLowerInvariant()} ×{i + 1}");
            }
        }
    }

    [CanvasFact]
    public async Task Edges_stay_attached_after_fitting_the_view()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("button[aria-label='Perbesar']");
        await page.WaitForTimeoutAsync(300);

        await page.ClickAsync("button[aria-label='Paskan tampilan']");
        await page.WaitForTimeoutAsync(500);

        await AssertAttachedAsync(page, "fitting the view");
    }

    [CanvasFact]
    public async Task Edges_stay_attached_while_the_canvas_is_panned()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        var canvas = await page.Locator(".dz-viewport").BoundingBoxAsync();
        Assert.NotNull(canvas);

        // Drag empty canvas, away from any node, so the gesture pans rather than moving a node.
        var x = canvas!.X + canvas.Width - 60;
        var y = canvas.Y + canvas.Height - 60;

        await page.Mouse.MoveAsync(x, y);
        await page.Mouse.DownAsync();
        await page.Mouse.MoveAsync(x - 180, y - 120, new MouseMoveOptions { Steps = 10 });
        await page.Mouse.UpAsync();
        await page.WaitForTimeoutAsync(300);

        await AssertAttachedAsync(page, "panning");
    }

    /// <summary>
    /// The two layers have to move together, which is only meaningful if they moved at all — a
    /// canvas that ignored the zoom entirely would satisfy the tests above.
    /// </summary>
    [CanvasFact]
    public async Task Both_layers_take_the_same_transform()
    {
        await using var session = await studio.DesignerPageAsync();
        var page = session.Page;

        await page.ClickAsync("button[aria-label='Perbesar']");
        await page.WaitForTimeoutAsync(400);

        var scales = await page.EvaluateAsync<double[]>(@"() => {
            const world = new DOMMatrix(getComputedStyle(document.querySelector('.dz-world')).transform);
            const layer = document.querySelector('.dz-edge-layer').transform.baseVal.consolidate();
            return [world.a, layer ? layer.matrix.a : 0];
        }");

        Assert.True(scales[0] > 1.01, $"The node layer did not scale (got {scales[0]:F3}).");
        Assert.Equal(scales[0], scales[1], 3);
    }
}
