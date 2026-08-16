using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Media;
using Avalonia.VisualTree;
using FluentAssertions;
using LocalGen.Desktop.Controls;

namespace LocalGen.Desktop.Tests;

/// <summary>
/// Tests for the markdown renderer.
/// </summary>
/// <remarks>
/// These exist because the renderer's output is a control tree, and the only other way to check
/// it is to look at a screenshot — which cannot prove a table became a real grid rather than a
/// paragraph of pipe characters. Written after a model that always fenced its tables made the
/// table path impossible to confirm by eye.
/// </remarks>
public class MarkdownViewTests
{
    [AvaloniaFact]
    public void A_pipe_table_becomes_a_grid_with_a_cell_per_value()
    {
        var view = Render(
            """
            | Quant  | Bits | Size  |
            |--------|------|-------|
            | Q4_K_M | 4    | 4.5GB |
            | Q5_K_M | 5    | 5.4GB |
            """);

        var grid = Descendants<Grid>(view)
            .FirstOrDefault(g => g.ColumnDefinitions.Count == 3 && g.RowDefinitions.Count == 3);

        grid.Should().NotBeNull("a three-column, three-row table should produce a matching grid");

        var text = AllText(view);
        text.Should().Contain("Quant").And.Contain("Q4_K_M").And.Contain("5.4GB");
    }

    [AvaloniaFact]
    public void A_table_header_row_is_distinguished_from_the_body()
    {
        var view = Render(
            """
            | Name | Value |
            |------|-------|
            | a    | 1     |
            """);

        // The header is what makes a table readable as data; it has to differ visually.
        var headerCells = Descendants<SelectableTextBlock>(view)
            .Where(t => t.FontWeight == FontWeight.SemiBold)
            .ToList();

        headerCells.Should().NotBeEmpty("header cells are rendered in semibold");
    }

    [AvaloniaFact]
    public void A_fenced_code_block_keeps_its_language_and_offers_a_copy_button()
    {
        var view = Render(
            """
            ```csharp
            Console.WriteLine("hello");
            ```
            """);

        AllText(view).Should().Contain("csharp");

        Descendants<Button>(view)
            .Should().Contain(b => (b.Content as string) == "Copy");

        // Code must not be reflowed — a wrapped line changes what the code says.
        Descendants<SelectableTextBlock>(view)
            .Should().Contain(t => t.TextWrapping == TextWrapping.NoWrap && t.Text!.Contains("Console.WriteLine"));
    }

    [AvaloniaFact]
    public void Code_is_not_treated_as_markdown()
    {
        // A fenced block containing a table is code, not a table. This is the behaviour that
        // made the table path hard to see by eye, and it is correct.
        var view = Render(
            """
            ```markdown
            | a | b |
            |---|---|
            | 1 | 2 |
            ```
            """);

        Descendants<Grid>(view)
            .Should().NotContain(g => g.ColumnDefinitions.Count == 2 && g.RowDefinitions.Count == 3);

        AllText(view).Should().Contain("| a | b |");
    }

    [AvaloniaFact]
    public void Nested_lists_keep_their_markers_and_order()
    {
        var view = Render(
            """
            1. First
               - nested one
               - nested two
            2. Second
            """);

        var text = AllText(view);

        text.Should().Contain("1.").And.Contain("2.");
        text.Should().Contain("•");
        text.Should().Contain("nested one").And.Contain("nested two");
    }

    [AvaloniaFact]
    public void Emphasis_maps_to_weight_and_style()
    {
        var view = Render("Plain **bold** and *italic* and ~~struck~~.");

        var runs = Descendants<SelectableTextBlock>(view)
            .SelectMany(t => t.Inlines ?? [])
            .OfType<Run>()
            .ToList();

        runs.Should().Contain(r => r.Text == "bold" && r.FontWeight == FontWeight.SemiBold);
        runs.Should().Contain(r => r.Text == "italic" && r.FontStyle == FontStyle.Italic);
        runs.Should().Contain(r => r.Text == "struck" && r.TextDecorations != null);
    }

    [AvaloniaFact]
    public void An_image_becomes_an_image_control_rather_than_a_link()
    {
        var view = Render("![a diagram](http://127.0.0.1:11434/api/files/x.png)");

        Descendants<Image>(view).Should().NotBeEmpty("an image reference renders as a bitmap host");
        AllText(view).Should().Contain("a diagram");
    }

    [AvaloniaFact]
    public void Video_and_audio_become_cards_rather_than_broken_images()
    {
        // Avalonia has no media element, so these are offered to the platform instead. The card
        // has to say what it is before the user clicks it.
        var video = AllText(Render("![clip](http://localhost/clip.mp4)"));
        video.Should().Contain("clip.mp4").And.Contain("video");

        var audio = AllText(Render("![track](http://localhost/track.mp3)"));
        audio.Should().Contain("track.mp3").And.Contain("audio");
    }

    [AvaloniaFact]
    public void A_link_is_clickable_and_shows_its_target()
    {
        var view = Render("See [the docs](https://example.com/guide).");

        var link = Descendants<TextBlock>(view)
            .FirstOrDefault(t => t.Text == "the docs");

        link.Should().NotBeNull();
        link!.TextDecorations.Should().NotBeNull("a link is underlined");
        ToolTip.GetTip(link).Should().Be("https://example.com/guide");
    }

    [AvaloniaFact]
    public void A_task_list_renders_checkboxes_in_their_recorded_state()
    {
        var view = Render(
            """
            - [x] done
            - [ ] pending
            """);

        var boxes = Descendants<CheckBox>(view).ToList();

        boxes.Should().HaveCount(2);
        boxes.Select(b => b.IsChecked).Should().BeEquivalentTo([true, false]);

        // The transcript is a record, not an editable checklist.
        boxes.Should().OnlyContain(b => !b.IsHitTestVisible);
    }

    [AvaloniaFact]
    public void Empty_and_whitespace_markdown_render_no_content()
    {
        // The control's own host panel is always present; what must be absent is any text.
        Descendants<TextBlock>(Render(null)).Should().BeEmpty();
        Descendants<TextBlock>(Render("   ")).Should().BeEmpty();
    }

    [AvaloniaFact]
    public void Changing_the_markdown_replaces_the_previous_content()
    {
        // The transcript rebuilds on every streamed token, so a stale tree would accumulate.
        var view = Render("first");
        AllText(view).Should().Contain("first");

        view.Markdown = "second";
        Layout(view);

        var text = AllText(view);
        text.Should().Contain("second");
        text.Should().NotContain("first");
    }

    // ─────────────────────────  Helpers  ─────────────────────────

    private static MarkdownView Render(string? markdown)
    {
        var view = new MarkdownView { Markdown = markdown };

        // Hosted in a window so the control is attached to a visual tree and templates apply.
        var window = new Window { Width = 800, Height = 600, Content = view };
        window.Show();

        Layout(view);
        return view;
    }

    private static void Layout(MarkdownView view)
    {
        var window = view.GetVisualRoot() as Window;
        window?.Measure(new Avalonia.Size(800, 600));
        window?.Arrange(new Avalonia.Rect(0, 0, 800, 600));
    }

    private static IEnumerable<T> Descendants<T>(Control root) where T : Control =>
        root.GetLogicalDescendants().OfType<T>();

    /// <summary>Flattens every piece of text in the rendered tree, for content assertions.</summary>
    private static string AllText(Control root)
    {
        var parts = new List<string>();

        foreach (var block in root.GetLogicalDescendants().OfType<TextBlock>())
        {
            if (!string.IsNullOrEmpty(block.Text))
            {
                parts.Add(block.Text);
            }

            foreach (var run in (block.Inlines ?? []).OfType<Run>())
            {
                if (!string.IsNullOrEmpty(run.Text))
                {
                    parts.Add(run.Text);
                }
            }
        }

        return string.Join("\n", parts);
    }
}
