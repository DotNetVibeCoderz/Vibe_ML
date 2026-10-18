using Matplotlib.Net;

namespace ScienceGallery.Models;

/// <summary>
/// One real-world use case: what it is for, which libraries it uses, and the code that runs it.
/// </summary>
/// <remarks>
/// Each showcase lives in its own file under <c>Showcases/</c>, and that file is embedded into the
/// app so the code panel can show it verbatim. Keep everything a reader needs inside the file - the
/// gallery shows the file, not a cleaned-up excerpt.
/// </remarks>
public abstract class Showcase
{
    /// <summary>A short, specific title: what gets done, not which API does it.</summary>
    public abstract string Title { get; }

    /// <summary>Where this is used.</summary>
    public abstract Domain Domain { get; }

    /// <summary>The libraries it uses, most important first.</summary>
    public abstract GraviLibrary[] Libraries { get; }

    /// <summary>One sentence for the card.</summary>
    public abstract string Summary { get; }

    /// <summary>Two to four sentences: who does this, why, and what the result tells them.</summary>
    public abstract string Scenario { get; }

    /// <summary>Does the work and reports into <paramref name="output"/>. Runs off the UI thread.</summary>
    public abstract void Run(ShowcaseOutput output);

    /// <summary>The embedded source file that holds this showcase.</summary>
    public string SourceName => GetType().Name + ".cs";
}

/// <summary>What a showcase hands back to the gallery.</summary>
public sealed class ShowcaseOutput
{
    private readonly List<(string Label, string Value, string? Note)> _metrics = [];
    private readonly List<string> _lines = [];

    /// <summary>The chart, if the showcase draws one.</summary>
    public Figure? Figure { get; private set; }

    /// <summary>Key numbers shown beside the chart.</summary>
    public IReadOnlyList<(string Label, string Value, string? Note)> Metrics => _metrics;

    /// <summary>Plain-text findings, shown under the metrics.</summary>
    public IReadOnlyList<string> Lines => _lines;

    /// <summary>Shows <paramref name="figure"/> as the result chart.</summary>
    public void Chart(Figure figure) => Figure = figure;

    /// <summary>Adds a key number, with an optional note on what it means.</summary>
    public void Metric(string label, string value, string? note = null) => _metrics.Add((label, value, note));

    /// <summary>Adds a line of findings.</summary>
    public void Line(string text) => _lines.Add(text);
}
