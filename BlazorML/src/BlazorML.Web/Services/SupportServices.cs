using System.Collections.Concurrent;
using BlazorML.Core.Abstractions;
using Markdig;
using Microsoft.AspNetCore.Components;

namespace BlazorML.Web.Services;

/// <summary>
/// Renders assistant messages to HTML. Advanced extensions are on so tables, task lists and
/// fenced code all render, and the pipeline is built once because it is not cheap to construct.
/// </summary>
public sealed class MarkdownRenderer
{
    private static readonly MarkdownPipeline Pipeline = new MarkdownPipelineBuilder()
        .UseAdvancedExtensions()
        .UseSoftlineBreakAsHardlineBreak()
        .UseEmojiAndSmiley()
        .UseAutoLinks()
        .Build();

    public MarkupString ToHtml(string? markdown) =>
        new(string.IsNullOrWhiteSpace(markdown) ? string.Empty : Markdown.ToHtml(markdown, Pipeline));

    public string ToPlainText(string? markdown) =>
        string.IsNullOrWhiteSpace(markdown) ? string.Empty : Markdown.ToPlainText(markdown, Pipeline);
}

/// <summary>
/// Tracks the active theme for the circuit. The initial value comes from localStorage via a
/// small script in App.razor, so the page does not flash the wrong theme on load.
/// </summary>
public sealed class ThemeState
{
    private string _theme = "system";

    public string Theme
    {
        get => _theme;
        set
        {
            if (_theme == value)
            {
                return;
            }

            _theme = value;
            Changed?.Invoke(value);
        }
    }

    public event Action<string>? Changed;
}

/// <summary>
/// Carries run progress from the background task doing the training to whichever component is
/// showing the canvas. In-process and per-experiment, which is all a single-server Blazor app needs.
/// </summary>
public sealed class RunBroadcaster
{
    private static readonly ConcurrentDictionary<string, List<Action<RunProgress>>> Listeners = new();

    public IDisposable Subscribe(string experimentId, Action<RunProgress> handler)
    {
        var list = Listeners.GetOrAdd(experimentId, _ => []);

        lock (list)
        {
            list.Add(handler);
        }

        return new Subscription(experimentId, handler);
    }

    public void Publish(string experimentId, RunProgress progress)
    {
        if (!Listeners.TryGetValue(experimentId, out var list))
        {
            return;
        }

        Action<RunProgress>[] snapshot;
        lock (list)
        {
            snapshot = list.ToArray();
        }

        foreach (var handler in snapshot)
        {
            try
            {
                handler(progress);
            }
            catch (Exception)
            {
                // A component that has already been disposed must not stop the run from
                // reporting progress to the others.
            }
        }
    }

    private sealed class Subscription(string experimentId, Action<RunProgress> handler) : IDisposable
    {
        public void Dispose()
        {
            if (!Listeners.TryGetValue(experimentId, out var list))
            {
                return;
            }

            lock (list)
            {
                list.Remove(handler);
            }
        }
    }
}
