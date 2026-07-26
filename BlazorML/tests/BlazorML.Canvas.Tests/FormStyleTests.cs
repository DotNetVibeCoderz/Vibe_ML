using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// The design system actually reaching the form controls.
/// <para>
/// Only a browser can answer this. A stylesheet that does not match an element produces no error
/// anywhere — the element simply renders as the browser would have drawn it, and bUnit compares
/// markup, which is identical either way. The rule here used to enumerate input types, so
/// <c>&lt;input @bind="x" /&gt;</c> — which Razor emits with no type attribute — matched nothing
/// and fell through to the default box on twenty-five fields.
/// </para>
/// </summary>
[Collection("canvas")]
public class FormStyleTests(StudioFixture studio)
{
    /// <summary>Reads back the properties that say whether the design system applied.</summary>
    private const string Probe = @"(selector) => {
        const el = document.querySelector(selector);
        if (!el) return null;

        const s = getComputedStyle(el);
        return {
            radius: parseFloat(s.borderTopLeftRadius),
            borderWidth: parseFloat(s.borderTopWidth),
            fontSize: parseFloat(s.fontSize),
            paddingTop: parseFloat(s.paddingTop),
            fontFamily: s.fontFamily
        };
    }";

    /// <summary>Playwright materialises this itself, so it needs settable members and a default
    /// constructor — a positional record cannot be built from the wire.</summary>
    private sealed class Box
    {
        public double Radius { get; set; }
        public double BorderWidth { get; set; }
        public double FontSize { get; set; }
        public double PaddingTop { get; set; }
        public string FontFamily { get; set; } = string.Empty;
    }

    private static async Task<Box> StyleOfAsync(IPage page, string selector)
    {
        var box = await page.EvaluateAsync<Box?>(Probe, selector);

        Assert.True(box is not null, $"Nothing on the page matched '{selector}'.");
        return box!;
    }

    /// <summary>Settings carries a bare text input, a select and a checkbox side by side.</summary>
    private async Task<StudioFixture.PageSession> SettingsAsync()
    {
        var session = await studio.ShellPageAsync("/pengaturan");

        await session.Page.Locator(".field input:not([type])").First
            .WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        return session;
    }

    [CanvasFact]
    public async Task A_text_input_with_no_type_attribute_still_gets_the_design_system_box()
    {
        await using var session = await SettingsAsync();

        var input = await StyleOfAsync(session.Page, ".field input:not([type])");

        // The three things the user could see were wrong: square corners, no border, and the
        // browser's own larger default text.
        Assert.True(input.Radius > 0, "A bare <input> has square corners.");
        Assert.True(input.BorderWidth >= 2, $"A bare <input> has a {input.BorderWidth}px border.");
        Assert.Contains("Plex Sans", input.FontFamily);
    }

    /// <summary>
    /// The comparison is the point: a bare input has to be indistinguishable from a typed one, or
    /// two fields sitting in the same form look like two different controls.
    /// </summary>
    [CanvasFact]
    public async Task A_bare_input_is_boxed_exactly_like_a_select_beside_it()
    {
        await using var session = await SettingsAsync();

        var input = await StyleOfAsync(session.Page, ".field input:not([type])");
        var select = await StyleOfAsync(session.Page, ".field select");

        Assert.Equal(select.Radius, input.Radius, 1);
        Assert.Equal(select.BorderWidth, input.BorderWidth, 1);
        Assert.Equal(select.FontSize, input.FontSize, 1);
        Assert.Equal(select.PaddingTop, input.PaddingTop, 1);
    }

    [CanvasTheory]
    [InlineData("password")]
    [InlineData("number")]
    [InlineData("search")]
    public async Task Typed_inputs_are_boxed_the_same_way(string type)
    {
        await using var session = await studio.ShellPageAsync("/pengaturan");
        var page = session.Page;

        await page.Locator(".field input:not([type])").First
            .WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        var bare = await StyleOfAsync(page, ".field input:not([type])");

        // Injected rather than hunted for: not every type is on this page, and the rule has to
        // hold for the ones a future form will use.
        await page.EvaluateAsync(@"(type) => {
            const field = document.querySelector('.field');
            const probe = document.createElement('input');
            probe.type = type;
            probe.id = 'probe-jenis';
            field.appendChild(probe);
        }", type);

        var typed = await StyleOfAsync(page, "#probe-jenis");

        Assert.Equal(bare.Radius, typed.Radius, 1);
        Assert.Equal(bare.BorderWidth, typed.BorderWidth, 1);
        Assert.Equal(bare.PaddingTop, typed.PaddingTop, 1);
    }

    /// <summary>
    /// Checkboxes, radios, sliders and file pickers must keep their own drawing. Styling every
    /// input the same way would replace them with 100%-wide bordered boxes.
    /// </summary>
    [CanvasTheory]
    [InlineData("checkbox")]
    [InlineData("radio")]
    [InlineData("range")]
    [InlineData("file")]
    public async Task Controls_that_are_not_text_boxes_are_left_alone(string type)
    {
        await using var session = await studio.ShellPageAsync("/pengaturan");
        var page = session.Page;

        await page.Locator(".field").First.WaitForAsync(new LocatorWaitForOptions { Timeout = 20000 });

        await page.EvaluateAsync(@"(type) => {
            const probe = document.createElement('input');
            probe.type = type;
            probe.id = 'probe-bukan-teks';
            document.querySelector('.field').appendChild(probe);
        }", type);

        var probe = await StyleOfAsync(page, "#probe-bukan-teks");

        Assert.True(probe.PaddingTop < 5,
            $"An <input type={type}> picked up the text box's {probe.PaddingTop}px padding.");
    }

    [CanvasFact]
    public async Task The_seeded_checkbox_is_still_its_own_size()
    {
        await using var session = await SettingsAsync();

        var size = await session.Page.EvaluateAsync<double>(
            "() => document.querySelector('.field--inline input[type=checkbox]').getBoundingClientRect().width");

        // 17px in the stylesheet. A full-width bordered box here would be unmistakable.
        Assert.InRange(size, 12, 24);
    }

    /// <summary>
    /// Focus is where the exclusion list earns itself twice: the design system replaces the
    /// outline with a hard offset shadow, which reads as focus on a bordered box and not at all
    /// on a 17px checkbox. Removing the outline there would leave keyboard users with nothing.
    /// </summary>
    [CanvasFact]
    public async Task A_focused_text_input_shows_the_shadow_and_a_focused_checkbox_keeps_its_outline()
    {
        await using var session = await SettingsAsync();
        var page = session.Page;

        await page.Locator(".field input:not([type])").First.FocusAsync();

        var shadow = await page.EvaluateAsync<string>(
            "() => getComputedStyle(document.activeElement).boxShadow");

        Assert.NotEqual("none", shadow);

        var outline = await page.EvaluateAsync<string>(@"() => {
            const box = document.querySelector('.field--inline input[type=checkbox]');
            box.focus();
            return getComputedStyle(box).outlineStyle;
        }");

        Assert.NotEqual("none", outline);
    }
}
