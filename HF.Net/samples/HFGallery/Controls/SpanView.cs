using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Controls.Primitives;
using Avalonia.Media;
using HFGallery.Cases;

namespace HFGallery.Controls;

/// <summary>
/// The original text with the model's spans marked in place.
/// </summary>
/// <remarks>
/// Three tasks in the gallery - entities, question answering, tokenization - all answer with a
/// range of the input rather than with new text, and showing that range where it actually sits is
/// the whole point: a list of extracted strings loses the context that made them findable, and
/// quietly hides an off-by-one in the offsets. Built from inline runs rather than drawn, so
/// wrapping, selection and text scaling come from the text stack instead of being reimplemented.
/// </remarks>
public sealed class SpanView : SelectableTextBlock
{
    /// <summary>The text and its marks.</summary>
    public static readonly StyledProperty<SpanText?> SourceProperty =
        AvaloniaProperty.Register<SpanView, SpanText?>(nameof(Source));

    /// <summary>Whether each mark is followed by its label.</summary>
    public static readonly StyledProperty<bool> ShowLabelsProperty =
        AvaloniaProperty.Register<SpanView, bool>(nameof(ShowLabels), true);

    static SpanView()
    {
        AffectsRender<SpanView>(SourceProperty, ShowLabelsProperty);
    }

    /// <inheritdoc cref="SourceProperty" />
    public SpanText? Source
    {
        get => GetValue(SourceProperty);
        set => SetValue(SourceProperty, value);
    }

    /// <inheritdoc cref="ShowLabelsProperty" />
    public bool ShowLabels
    {
        get => GetValue(ShowLabelsProperty);
        set => SetValue(ShowLabelsProperty, value);
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property == SourceProperty
            || change.Property == ShowLabelsProperty
            || change.Property == ThemeVariantScope.ActualThemeVariantProperty)
        {
            Rebuild();
        }
    }

    private void Rebuild()
    {
        var inlines = new InlineCollection();

        if (Source is not { } source || source.Text.Length == 0)
        {
            Inlines = inlines;
            return;
        }

        var plain = Palette.Resource(this, "TextBaseBrush", "#C2CBD6");
        var faint = Palette.Resource(this, "TextFaintBrush", "#5D6875");

        // Sorted and clipped, because nothing guarantees a model returns its spans in order and a
        // mark that starts before the previous one ended would silently swallow text.
        var marks = source.Marks
            .Where(m => m.End > m.Start && m.Start >= 0 && m.End <= source.Text.Length)
            .OrderBy(m => m.Start)
            .ToList();

        var cursor = 0;

        foreach (var mark in marks)
        {
            if (mark.Start < cursor) continue;

            if (mark.Start > cursor)
            {
                inlines.Add(new Run(source.Text[cursor..mark.Start]) { Foreground = plain });
            }

            var hue = Palette.Categorical(this, mark.Category);

            inlines.Add(new Run(source.Text[mark.Start..mark.End])
            {
                Foreground = Palette.Resource(this, "TextHighBrush", "#E7EDF4"),
                Background = Tint(hue),
                FontWeight = FontWeight.Medium,
            });

            if (ShowLabels)
            {
                // The bar carries the identity in colour; the label stays in muted ink so the
                // passage does not turn into a field of coloured words.
                inlines.Add(new Run("▌") { Foreground = hue });
                inlines.Add(new Run(mark.Label + " ")
                {
                    Foreground = faint,
                    FontSize = FontSize - 3,
                    FontFamily = Mono(),
                });
            }

            cursor = mark.End;
        }

        if (cursor < source.Text.Length)
        {
            inlines.Add(new Run(source.Text[cursor..]) { Foreground = plain });
        }

        Inlines = inlines;
    }

    /// <summary>The face labels are set in, so a token reads as a token.</summary>
    private FontFamily Mono()
        => this.TryFindResource("MonoFont", ActualThemeVariant, out var value) && value is FontFamily family
            ? family
            : FontFamily.Default;

    /// <summary>The highlight wash: the category hue, faint enough to keep the text legible.</summary>
    private static IBrush Tint(IBrush hue)
        => hue is ISolidColorBrush solid
            ? new SolidColorBrush(Color.FromArgb(0x40, solid.Color.R, solid.Color.G, solid.Color.B))
            : Brushes.Transparent;
}
