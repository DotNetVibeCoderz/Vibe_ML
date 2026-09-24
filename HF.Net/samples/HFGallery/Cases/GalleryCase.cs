using HFGallery.Controls;

namespace HFGallery.Cases;

/// <summary>A run of a use case: whichever of these the case filled in gets drawn.</summary>
/// <remarks>
/// One result shape rather than a view per case. The cases differ in what they compute, not in how
/// a ranked magnitude or a projected point should look - and giving each its own view is how a
/// gallery ends up with eight subtly different bar charts.
/// </remarks>
public sealed record CaseResult
{
    /// <summary>A sentence describing what came back, shown above the visualisation.</summary>
    public string Summary { get; init; } = "";

    /// <summary>Ranked magnitudes, drawn as horizontal bars.</summary>
    public IReadOnlyList<Datum>? Bars { get; init; }

    /// <summary>What the bar values are: <c>%</c>, <c>ms</c>, or empty for a plain count.</summary>
    public string Unit { get; init; } = "%";

    /// <summary>Points in a two-dimensional projection.</summary>
    public IReadOnlyList<Point2>? Scatter { get; init; }

    /// <summary>Curves over an ordered variable.</summary>
    public IReadOnlyList<Series>? Lines { get; init; }

    /// <summary>What the line chart's horizontal axis counts.</summary>
    public string XLabel { get; init; } = "";

    /// <summary>Part-to-whole across wildly different magnitudes.</summary>
    public IReadOnlyList<Slice>? Treemap { get; init; }

    /// <summary>Text with marked spans, for the tasks whose output is a span.</summary>
    public SpanText? Spans { get; init; }

    /// <summary>Category names in palette order, for the key beside the visualisation.</summary>
    public IReadOnlyList<string>? Legend { get; init; }

    /// <summary>A small table of facts, shown beside the visualisation.</summary>
    public IReadOnlyList<Fact>? Facts { get; init; }
}

/// <summary>One row of the fact table.</summary>
/// <param name="Key">What the row is.</param>
/// <param name="Value">Its value, already formatted.</param>
/// <remarks>
/// A named type rather than a tuple because these are bound to in XAML, and a tuple exposes
/// <c>Item1</c> as a field, which the binding engine will not resolve.
/// </remarks>
public readonly record struct Fact(string Key, string Value)
{
    /// <summary>Lets a case write its facts as plain pairs.</summary>
    public static implicit operator Fact((string Key, string Value) pair) => new(pair.Key, pair.Value);
}

/// <summary>A passage with labelled spans marked in it.</summary>
/// <param name="Text">The original text, untouched.</param>
/// <param name="Marks">The spans to highlight.</param>
public readonly record struct SpanText(string Text, IReadOnlyList<SpanMark> Marks);

/// <summary>One highlighted range.</summary>
/// <param name="Start">Index of the first character.</param>
/// <param name="End">Index one past the last.</param>
/// <param name="Label">What the span is.</param>
/// <param name="Score">Confidence, for the tooltip.</param>
/// <param name="Category">Index into the categorical palette.</param>
public readonly record struct SpanMark(int Start, int End, string Label, double Score, int Category);

/// <summary>How long a case takes and what it needs, so the UI can warn before it starts.</summary>
public enum CaseCost
{
    /// <summary>Runs immediately, no network, no model.</summary>
    Instant,

    /// <summary>Downloads a tokenizer or metadata - seconds.</summary>
    Light,

    /// <summary>Downloads and runs a model - tens of seconds on a first run.</summary>
    Heavy,
}

/// <summary>One entry in the gallery.</summary>
public abstract class GalleryCase
{
    /// <summary>Short name, shown in the rail.</summary>
    public abstract string Title { get; }

    /// <summary>One line saying what the case demonstrates.</summary>
    public abstract string Blurb { get; }

    /// <summary>Which library the case is mainly about.</summary>
    public abstract string Library { get; }

    /// <summary>What running it costs.</summary>
    public virtual CaseCost Cost => CaseCost.Heavy;

    /// <summary>The model it loads, or empty when it needs none.</summary>
    public virtual string Model => "";

    /// <summary>The input the user can edit, or null when the case takes none.</summary>
    public virtual string? DefaultInput => null;

    /// <summary>A label for the input box.</summary>
    public virtual string InputLabel => "Input";

    /// <summary>A second input, for the cases that take a pair.</summary>
    public virtual string? DefaultSecondInput => null;

    /// <summary>A label for the second input box.</summary>
    public virtual string SecondInputLabel => "Context";

    /// <summary>The C# that does what this case does, shown in the code panel.</summary>
    public abstract string Code { get; }

    /// <summary>Runs the case.</summary>
    /// <param name="input">The first input, as edited by the user.</param>
    /// <param name="second">The second input, where the case takes one.</param>
    /// <param name="progress">Receives a line of status per step.</param>
    /// <param name="token">Cancels the run.</param>
    public abstract Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token);
}
