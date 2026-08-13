using System.Collections.ObjectModel;
using Avalonia.Threading;

namespace ScienceAppGen.Services;

/// <summary>How prominent a log line is. Maps to the spectrum bands in the theme.</summary>
public enum LogLevel
{
    /// <summary>Routine progress.</summary>
    Info,

    /// <summary>Build and tooling output.</summary>
    Build,

    /// <summary>Something completed.</summary>
    Success,

    /// <summary>Something is off but the operation continued.</summary>
    Warning,

    /// <summary>The operation failed.</summary>
    Error,

    /// <summary>The assistant said or did something.</summary>
    Assistant,
}

/// <summary>One line in the logs panel.</summary>
/// <param name="Timestamp">When it happened.</param>
/// <param name="Level">How prominent it is.</param>
/// <param name="Source">Which subsystem produced it.</param>
/// <param name="Message">The text.</param>
public sealed record LogEntry(DateTime Timestamp, LogLevel Level, string Source, string Message)
{
    /// <summary>Clock time, for the gutter.</summary>
    public string Time => Timestamp.ToString("HH:mm:ss");

    /// <summary>The theme brush key for this level's spectrum band.</summary>
    public string LevelBrush => Level switch
    {
        LogLevel.Error => "Spec1",
        LogLevel.Warning => "Spec2",
        LogLevel.Build => "Spec3",
        LogLevel.Success => "Spec4",
        LogLevel.Assistant => "Spec6",
        _ => "Spec5",
    };
}

/// <summary>
/// The application's log sink, bound directly to the logs panel.
/// </summary>
/// <remarks>
/// Writes marshal onto the UI thread, so background work - a build, a tool call, a streaming
/// response - can log without the caller thinking about dispatchers. The collection is capped so
/// a chatty build cannot grow it without bound.
/// </remarks>
public sealed class LogService
{
    private const int MaxEntries = 5_000;

    /// <summary>Every retained log line, oldest first.</summary>
    public ObservableCollection<LogEntry> Entries { get; } = [];

    /// <summary>Raised after a line is appended, so the panel can scroll to it.</summary>
    public event Action<LogEntry>? EntryAdded;

    /// <summary>Appends a line.</summary>
    public void Write(LogLevel level, string source, string message)
    {
        var entry = new LogEntry(DateTime.Now, level, source, message);

        if (Dispatcher.UIThread.CheckAccess()) Append(entry);
        else Dispatcher.UIThread.Post(() => Append(entry));
    }

    /// <summary>Appends an informational line.</summary>
    public void Info(string source, string message) => Write(LogLevel.Info, source, message);

    /// <summary>Appends a build line.</summary>
    public void Build(string source, string message) => Write(LogLevel.Build, source, message);

    /// <summary>Appends a success line.</summary>
    public void Success(string source, string message) => Write(LogLevel.Success, source, message);

    /// <summary>Appends a warning.</summary>
    public void Warning(string source, string message) => Write(LogLevel.Warning, source, message);

    /// <summary>Appends an error.</summary>
    public void Error(string source, string message) => Write(LogLevel.Error, source, message);

    /// <summary>Appends a line attributed to the assistant.</summary>
    public void Assistant(string message) => Write(LogLevel.Assistant, "Jack", message);

    /// <summary>Empties the panel.</summary>
    public void Clear()
    {
        if (Dispatcher.UIThread.CheckAccess()) Entries.Clear();
        else Dispatcher.UIThread.Post(Entries.Clear);
    }

    private void Append(LogEntry entry)
    {
        Entries.Add(entry);
        while (Entries.Count > MaxEntries) Entries.RemoveAt(0);
        EntryAdded?.Invoke(entry);
    }
}
