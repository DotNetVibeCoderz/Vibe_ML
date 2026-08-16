using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Extensions.TaskLists;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace LocalGen.Desktop.Controls;

/// <summary>
/// Renders markdown as Avalonia controls.
/// </summary>
/// <remarks>
/// Parsing is left to Markdig — a solved problem — and this walks the resulting syntax tree,
/// building real controls rather than a formatted string. That is what makes tables selectable,
/// code blocks copyable, images actual bitmaps and links clickable, which a single
/// <see cref="TextBlock"/> could not be.
///
/// Model output arrives a token at a time, so the whole document is rebuilt on each change. That
/// is affordable because a chat turn is small; the image cache is what keeps it from stuttering.
/// </remarks>
public sealed class MarkdownView : UserControl
{
    public static readonly StyledProperty<string?> MarkdownProperty =
        AvaloniaProperty.Register<MarkdownView, string?>(nameof(Markdown));

    private static readonly MarkdownPipeline Pipeline = new MarkdownPipelineBuilder()
        // Tables, task lists, strikethrough, autolinks and footnotes: what people actually
        // write, and what models actually emit.
        .UseAdvancedExtensions()
        .UseEmphasisExtras()
        .UsePipeTables()
        .UseGridTables()
        .UseTaskLists()
        .UseAutoLinks()
        .Build();

    private readonly StackPanel _root = new() { Spacing = 10 };

    public MarkdownView()
    {
        Content = _root;
        // Selection and copying happen inside the individual text blocks.
        Focusable = false;
    }

    public string? Markdown
    {
        get => GetValue(MarkdownProperty);
        set => SetValue(MarkdownProperty, value);
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property == MarkdownProperty)
        {
            Rebuild();
        }
    }

    private void Rebuild()
    {
        _root.Children.Clear();

        var text = Markdown;

        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        var document = Markdig.Markdown.Parse(text, Pipeline);

        foreach (var block in document)
        {
            var control = RenderBlock(block);

            if (control is not null)
            {
                _root.Children.Add(control);
            }
        }
    }

    // ─────────────────────────────  Blocks  ─────────────────────────────

    private Control? RenderBlock(Block block) => block switch
    {
        HeadingBlock heading => RenderHeading(heading),
        ParagraphBlock paragraph => RenderParagraph(paragraph),
        FencedCodeBlock code => RenderCode(code.Lines.ToString(), code.Info),
        CodeBlock code => RenderCode(code.Lines.ToString(), null),
        ListBlock list => RenderList(list),
        QuoteBlock quote => RenderQuote(quote),
        Table table => RenderTable(table),
        ThematicBreakBlock => RenderRule(),
        HtmlBlock html => RenderCode(html.Lines.ToString(), "html"),
        _ => null
    };

    private Control RenderHeading(HeadingBlock heading)
    {
        var block = new SelectableTextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontWeight = FontWeight.SemiBold,
            // A fourth-ish ramp, tightened as it grows so large headings do not look inflated.
            FontSize = heading.Level switch
            {
                1 => 21,
                2 => 18,
                3 => 16,
                4 => 14.5,
                _ => 13.5
            },
            LetterSpacing = heading.Level <= 2 ? -0.4 : 0,
            Margin = new Thickness(0, heading.Level <= 2 ? 6 : 2, 0, 0)
        };

        AppendInlines(block.Inlines!, heading.Inline);

        // A rule under the top two levels gives the document structure without an extra device.
        if (heading.Level <= 2)
        {
            return new StackPanel
            {
                Spacing = 6,
                Children =
                {
                    block,
                    new Border
                    {
                        Height = 1,
                        Background = Brush("Edge")
                    }
                }
            };
        }

        return block;
    }

    /// <summary>
    /// Renders a paragraph, splitting media out into their own blocks.
    /// </summary>
    /// <remarks>
    /// An image or video reference is a block-level thing even when markdown puts it inline, so
    /// text accumulates into a run of prose and each media reference interrupts it. Writing
    /// <c>![diagram](x.png)</c> on its own line — which is how they almost always appear — then
    /// produces just the image, with no empty text block above it.
    /// </remarks>
    private Control RenderParagraph(ParagraphBlock paragraph)
    {
        var blocks = new List<Control>();
        var current = NewTextBlock();

        void Flush()
        {
            if (current.Inlines is { Count: > 0 })
            {
                blocks.Add(current);
            }

            current = NewTextBlock();
        }

        foreach (var inline in paragraph.Inline ?? [])
        {
            if (TryRenderMedia(inline, out var media))
            {
                Flush();
                blocks.Add(media!);
                continue;
            }

            AppendInline(current.Inlines!, inline);
        }

        Flush();

        if (blocks.Count == 1)
        {
            return blocks[0];
        }

        var stack = new StackPanel { Spacing = 8 };

        foreach (var block in blocks)
        {
            stack.Children.Add(block);
        }

        return stack;
    }

    private Control RenderCode(string code, string? language)
    {
        code = code.TrimEnd('\n', '\r');

        var header = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("*,Auto"),
            Margin = new Thickness(12, 7, 7, 0)
        };

        var label = new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(language) ? "code" : language.Trim(),
            FontFamily = MonoFont,
            FontSize = 10.5,
            Foreground = Brush("TextMuted"),
            VerticalAlignment = VerticalAlignment.Center
        };

        // The one action people actually want from a code block in a chat transcript.
        var copy = new Button
        {
            Content = "Copy",
            Padding = new Thickness(8, 2),
            FontSize = 11,
            Background = Brushes.Transparent,
            BorderThickness = new Thickness(0),
            Foreground = Brush("TextMuted"),
            Cursor = new Cursor(StandardCursorType.Hand)
        };

        copy.Click += async (_, _) =>
        {
            var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;

            if (clipboard is not null)
            {
                await clipboard.SetTextAsync(code);
                copy.Content = "Copied";

                await Task.Delay(1400);
                copy.Content = "Copy";
            }
        };

        Grid.SetColumn(label, 0);
        Grid.SetColumn(copy, 1);
        header.Children.Add(label);
        header.Children.Add(copy);

        var body = new SelectableTextBlock
        {
            Text = code,
            FontFamily = MonoFont,
            FontSize = 12,
            // Code must not be reflowed; long lines scroll instead.
            TextWrapping = TextWrapping.NoWrap,
            Margin = new Thickness(12, 4, 12, 12)
        };

        return new Border
        {
            Background = Brush("SurfaceSunken"),
            BorderBrush = Brush("Edge"),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(4),
            Child = new StackPanel
            {
                Children =
                {
                    header,
                    new ScrollViewer
                    {
                        HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                        VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
                        Content = body
                    }
                }
            }
        };
    }

    private Control RenderList(ListBlock list)
    {
        var stack = new StackPanel { Spacing = 3 };
        var index = list.IsOrdered && int.TryParse(list.OrderedStart, out var start) ? start : 1;

        foreach (var item in list.OfType<ListItemBlock>())
        {
            var row = new Grid
            {
                ColumnDefinitions = new ColumnDefinitions("Auto,*")
            };

            var marker = new TextBlock
            {
                Text = list.IsOrdered ? $"{index++}." : "•",
                FontFamily = list.IsOrdered ? FontFamily.Default : FontFamily.Default,
                Foreground = Brush("TextMuted"),
                Margin = new Thickness(2, 0, 8, 0),
                MinWidth = list.IsOrdered ? 20 : 12
            };

            var content = new StackPanel { Spacing = 4 };

            foreach (var child in item)
            {
                // A task list item carries its checkbox on the first inline of its paragraph.
                if (child is ParagraphBlock paragraph && TryRenderTaskItem(paragraph, out var task))
                {
                    marker.Text = string.Empty;
                    marker.MinWidth = 0;
                    content.Children.Add(task!);
                    continue;
                }

                var rendered = RenderBlock(child);

                if (rendered is not null)
                {
                    content.Children.Add(rendered);
                }
            }

            Grid.SetColumn(marker, 0);
            Grid.SetColumn(content, 1);
            row.Children.Add(marker);
            row.Children.Add(content);

            stack.Children.Add(row);
        }

        return stack;
    }

    private bool TryRenderTaskItem(ParagraphBlock paragraph, out Control? control)
    {
        control = null;

        if (paragraph.Inline?.FirstChild is not TaskList task)
        {
            return false;
        }

        var text = NewTextBlock();

        foreach (var inline in paragraph.Inline.Skip(1))
        {
            AppendInline(text.Inlines!, inline);
        }

        control = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*"),
            Children =
            {
                new CheckBox
                {
                    IsChecked = task.Checked,
                    // The transcript is a record of what was said, not an editable checklist.
                    IsHitTestVisible = false,
                    Margin = new Thickness(0, 0, 4, 0),
                    MinWidth = 0,
                    Padding = new Thickness(0),
                    [Grid.ColumnProperty] = 0
                },
                new ContentControl { Content = text, [Grid.ColumnProperty] = 1 }
            }
        };

        return true;
    }

    private Control RenderQuote(QuoteBlock quote)
    {
        var content = new StackPanel { Spacing = 6 };

        foreach (var child in quote)
        {
            var rendered = RenderBlock(child);

            if (rendered is not null)
            {
                content.Children.Add(rendered);
            }
        }

        return new Border
        {
            BorderBrush = Brush("SignalDim"),
            BorderThickness = new Thickness(2, 0, 0, 0),
            Padding = new Thickness(12, 2, 0, 2),
            Child = content
        };
    }

    /// <summary>
    /// Renders a pipe or grid table as a real <see cref="Grid"/>, so columns align across rows
    /// and every cell stays selectable.
    /// </summary>
    private Control RenderTable(Table table)
    {
        var rows = table.OfType<TableRow>().ToList();

        if (rows.Count == 0)
        {
            return new StackPanel();
        }

        var columnCount = rows.Max(row => row.Count);

        var grid = new Grid();

        for (var i = 0; i < columnCount; i++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
        }

        for (var i = 0; i < rows.Count; i++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        }

        for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];

            for (var columnIndex = 0; columnIndex < row.Count; columnIndex++)
            {
                if (row[columnIndex] is not TableCell cell)
                {
                    continue;
                }

                var content = new StackPanel { Spacing = 3 };

                foreach (var block in cell)
                {
                    var rendered = RenderBlock(block);

                    if (rendered is Control control)
                    {
                        if (control is SelectableTextBlock text)
                        {
                            text.FontWeight = row.IsHeader ? FontWeight.SemiBold : FontWeight.Normal;

                            if (row.IsHeader)
                            {
                                text.Foreground = Brush("TextSecondary");
                            }
                        }

                        content.Children.Add(control);
                    }
                }

                var alignment = table.ColumnDefinitions.Count > columnIndex
                    ? table.ColumnDefinitions[columnIndex].Alignment
                    : null;

                var border = new Border
                {
                    Padding = new Thickness(10, 6),
                    // Hairlines between cells only, so the table reads as data rather than a box.
                    BorderBrush = Brush("Edge"),
                    BorderThickness = new Thickness(0, 0, 0, rowIndex == rows.Count - 1 ? 0 : 1),
                    Background = row.IsHeader ? Brush("SurfaceOverlay") : Brushes.Transparent,
                    HorizontalAlignment = alignment switch
                    {
                        TableColumnAlign.Right => HorizontalAlignment.Right,
                        TableColumnAlign.Center => HorizontalAlignment.Center,
                        _ => HorizontalAlignment.Stretch
                    },
                    Child = content
                };

                Grid.SetRow(border, rowIndex);
                Grid.SetColumn(border, columnIndex);
                grid.Children.Add(border);
            }
        }

        return new Border
        {
            BorderBrush = Brush("Edge"),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(4),
            ClipToBounds = true,
            Child = new ScrollViewer
            {
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
                Content = grid
            }
        };
    }

    private Control RenderRule() => new Border
    {
        Height = 1,
        Background = Brush("Edge"),
        Margin = new Thickness(0, 6)
    };

    // ─────────────────────────────  Inlines  ─────────────────────────────

    private void AppendInlines(InlineCollection target, ContainerInline? container)
    {
        foreach (var inline in container ?? [])
        {
            AppendInline(target, inline);
        }
    }

    private void AppendInline(
        InlineCollection target,
        Markdig.Syntax.Inlines.Inline inline,
        FontWeight weight = FontWeight.Normal,
        FontStyle style = FontStyle.Normal,
        bool strikethrough = false)
    {
        switch (inline)
        {
            case LiteralInline literal:
                target.Add(new Run(literal.Content.ToString())
                {
                    FontWeight = weight,
                    FontStyle = style,
                    TextDecorations = strikethrough ? TextDecorations.Strikethrough : null
                });
                break;

            case EmphasisInline emphasis:
            {
                // Markdig reports the delimiter and its count: '*' or '_' twice is bold, once is
                // italic, and '~' twice is strikethrough via the extras extension.
                var nextWeight = emphasis is { DelimiterChar: '*' or '_', DelimiterCount: >= 2 }
                    ? FontWeight.SemiBold
                    : weight;

                var nextStyle = emphasis is { DelimiterChar: '*' or '_', DelimiterCount: 1 }
                    ? FontStyle.Italic
                    : style;

                var nextStrike = strikethrough || emphasis.DelimiterChar == '~';

                foreach (var child in emphasis)
                {
                    AppendInline(target, child, nextWeight, nextStyle, nextStrike);
                }

                break;
            }

            case CodeInline code:
                target.Add(new Run(code.Content)
                {
                    FontFamily = MonoFont,
                    Foreground = Brush("Data")
                });
                break;

            case LinkInline link:
                AppendLink(target, link, weight, style, strikethrough);
                break;

            case AutolinkInline autolink:
                target.Add(BuildLinkRun(autolink.Url, autolink.Url));
                break;

            case LineBreakInline lineBreak:
                if (lineBreak.IsHard)
                {
                    target.Add(new LineBreak());
                }
                else
                {
                    target.Add(new Run(" "));
                }

                break;

            case HtmlInline:
                // Raw HTML is not rendered; showing nothing is better than showing a tag.
                break;

            case ContainerInline container:
                foreach (var child in container)
                {
                    AppendInline(target, child, weight, style, strikethrough);
                }

                break;
        }
    }

    private void AppendLink(
        InlineCollection target,
        LinkInline link,
        FontWeight weight,
        FontStyle style,
        bool strikethrough)
    {
        var url = link.Url ?? string.Empty;
        var label = string.Concat(link.OfType<LiteralInline>().Select(l => l.Content.ToString()));

        if (string.IsNullOrEmpty(label))
        {
            label = url;
        }

        if (link.IsImage)
        {
            // Reached only for an image nested somewhere a media block cannot be emitted;
            // the label keeps the reference visible.
            target.Add(BuildLinkRun($"🖼 {label}", url));
            return;
        }

        _ = weight;
        _ = style;
        _ = strikethrough;

        target.Add(BuildLinkRun(label, url));
    }

    /// <summary>
    /// Builds a clickable link. Wrapped in an <see cref="InlineUIContainer"/> because a
    /// <see cref="Run"/> cannot receive pointer input on its own.
    /// </summary>
    private InlineUIContainer BuildLinkRun(string label, string url)
    {
        var text = new TextBlock
        {
            Text = label,
            Foreground = Brush("Data"),
            TextDecorations = TextDecorations.Underline,
            Cursor = new Cursor(StandardCursorType.Hand),
            VerticalAlignment = VerticalAlignment.Center
        };

        ToolTip.SetTip(text, url);
        text.PointerPressed += (_, e) =>
        {
            e.Handled = true;
            MediaResolver.OpenExternally(url);
        };

        return new InlineUIContainer(text) { BaselineAlignment = BaselineAlignment.TextBottom };
    }

    // ─────────────────────────────  Media  ─────────────────────────────

    private bool TryRenderMedia(Markdig.Syntax.Inlines.Inline inline, out Control? control)
    {
        control = null;

        if (inline is not LinkInline link || string.IsNullOrEmpty(link.Url))
        {
            return false;
        }

        var url = link.Url;
        var kind = MediaResolver.Classify(url);

        // An explicit image reference is media even when the extension says nothing.
        if (link.IsImage && kind == MediaKind.Link)
        {
            kind = MediaKind.Image;
        }

        var alt = string.Concat(link.OfType<LiteralInline>().Select(l => l.Content.ToString()));

        switch (kind)
        {
            case MediaKind.Image:
                control = BuildImage(url, alt);
                return true;

            case MediaKind.Video:
            case MediaKind.Audio:
            case MediaKind.Document:
                control = BuildMediaCard(url, string.IsNullOrEmpty(alt) ? Path.GetFileName(url.Split('?')[0]) : alt, kind);
                return true;

            default:
                return false;
        }
    }

    private Control BuildImage(string url, string alt)
    {
        var image = new Image
        {
            Stretch = Stretch.Uniform,
            // Bounded so a large attachment does not push the rest of the turn off screen.
            MaxHeight = 340,
            MaxWidth = 560,
            HorizontalAlignment = HorizontalAlignment.Left
        };

        var caption = new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(alt) ? "loading image…" : alt,
            FontSize = 11,
            Foreground = Brush("TextMuted"),
            TextWrapping = TextWrapping.Wrap
        };

        var container = new Border
        {
            Background = Brush("SurfaceSunken"),
            BorderBrush = Brush("Edge"),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(4),
            Padding = new Thickness(8),
            HorizontalAlignment = HorizontalAlignment.Left,
            Cursor = new Cursor(StandardCursorType.Hand),
            Child = new StackPanel
            {
                Spacing = 6,
                Children = { image, caption }
            }
        };

        ToolTip.SetTip(container, $"{url}\nClick to open");
        container.PointerPressed += (_, _) => MediaResolver.OpenExternally(url);

        // Loading is fire-and-forget: the layout is already correct, and the bitmap fills in.
        _ = LoadImageAsync(url, alt, image, caption);

        return container;
    }

    private static async Task LoadImageAsync(string url, string alt, Image image, TextBlock caption)
    {
        var bitmap = await MediaResolver.LoadImageAsync(url);

        if (bitmap is not null)
        {
            image.Source = bitmap;
            caption.Text = string.IsNullOrWhiteSpace(alt) ? string.Empty : alt;
            caption.IsVisible = !string.IsNullOrWhiteSpace(alt);
        }
        else
        {
            image.IsVisible = false;
            caption.Text = string.IsNullOrWhiteSpace(alt)
                ? $"Image could not be loaded — {url}"
                : $"{alt} (could not be loaded)";
        }
    }

    /// <summary>
    /// A card for video, audio and documents.
    /// </summary>
    /// <remarks>
    /// Avalonia has no media element, so embedding a player would mean shipping one. Handing the
    /// file to the user's own application is the honest alternative, and the card makes clear
    /// what the attachment is before they click.
    /// </remarks>
    private Control BuildMediaCard(string url, string label, MediaKind kind)
    {
        var (glyph, description) = kind switch
        {
            MediaKind.Video => ("▶", "video"),
            MediaKind.Audio => ("♪", "audio"),
            _ => ("▤", "document")
        };

        // The alt text is usually a caption ("clip") while the file name says what will actually
        // open ("recording-2026-08-16.mp4"). Both matter, so the caption leads and the file name
        // sits on the second line rather than being lost.
        var fileName = Path.GetFileName(url.Split('?')[0].Split('#')[0]);

        var subtitle = !string.IsNullOrEmpty(fileName) &&
                       !string.Equals(fileName, label, StringComparison.OrdinalIgnoreCase)
            ? $"{description} · {fileName}"
            : $"{description} · click to open";

        var card = new Border
        {
            Background = Brush("SurfaceSunken"),
            BorderBrush = Brush("Edge"),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(4),
            Padding = new Thickness(12, 10),
            HorizontalAlignment = HorizontalAlignment.Left,
            Cursor = new Cursor(StandardCursorType.Hand),
            Child = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Spacing = 12,
                Children =
                {
                    new TextBlock
                    {
                        Text = glyph,
                        FontSize = 18,
                        Foreground = Brush("Signal"),
                        VerticalAlignment = VerticalAlignment.Center
                    },
                    new StackPanel
                    {
                        Spacing = 1,
                        VerticalAlignment = VerticalAlignment.Center,
                        Children =
                        {
                            new TextBlock
                            {
                                Text = label,
                                FontWeight = FontWeight.SemiBold,
                                MaxWidth = 380,
                                TextTrimming = TextTrimming.CharacterEllipsis
                            },
                            new TextBlock
                            {
                                Text = subtitle,
                                FontSize = 11,
                                Foreground = Brush("TextMuted")
                            }
                        }
                    }
                }
            }
        };

        ToolTip.SetTip(card, url);
        card.PointerPressed += (_, _) => MediaResolver.OpenExternally(url);

        return card;
    }

    // ─────────────────────────────  Helpers  ─────────────────────────────

    private SelectableTextBlock NewTextBlock() => new()
    {
        TextWrapping = TextWrapping.Wrap,
        Inlines = []
    };

    private static FontFamily MonoFont { get; } =
        new("Cascadia Mono,Consolas,JetBrains Mono,SF Mono,Menlo,DejaVu Sans Mono,monospace");

    /// <summary>
    /// Resolves a theme brush by key, so rendered markdown follows the active theme instead of
    /// freezing whatever colour was current when the control was built.
    /// </summary>
    private IBrush? Brush(string key) =>
        Application.Current?.TryFindResource(key, Application.Current.ActualThemeVariant, out var brush) == true
            ? brush as IBrush
            : null;
}
