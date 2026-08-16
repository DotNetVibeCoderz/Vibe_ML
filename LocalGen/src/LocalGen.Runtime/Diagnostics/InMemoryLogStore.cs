using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace LocalGen.Runtime.Diagnostics;

/// <summary>One captured log line.</summary>
public sealed record LogEntry
{
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    public required LogLevel Level { get; init; }

    public required string Category { get; init; }

    public required string Message { get; init; }

    public string? Exception { get; init; }
}

/// <summary>
/// A bounded, in-memory log buffer.
/// </summary>
/// <remarks>
/// The Admin Control "Logs" panel needs recent log lines without tailing a file, and the desktop
/// app may host the server in-process where there is no console to read. This keeps the last
/// <see cref="Capacity"/> entries and raises an event per line so a UI can append live.
/// </remarks>
public sealed class InMemoryLogStore
{
    public const int Capacity = 2000;

    private readonly ConcurrentQueue<LogEntry> _entries = new();

    public event EventHandler<LogEntry>? EntryAdded;

    public void Add(LogEntry entry)
    {
        _entries.Enqueue(entry);

        while (_entries.Count > Capacity && _entries.TryDequeue(out _))
        {
        }

        EntryAdded?.Invoke(this, entry);
    }

    /// <summary>Most recent entries last, optionally filtered by minimum level.</summary>
    public IReadOnlyList<LogEntry> Recent(int limit = 200, LogLevel minimumLevel = LogLevel.Trace)
    {
        var filtered = _entries.Where(e => e.Level >= minimumLevel).ToArray();
        return filtered.Length <= limit ? filtered : filtered[^limit..];
    }

    public void Clear()
    {
        while (_entries.TryDequeue(out _))
        {
        }
    }
}

/// <summary>Bridges the logging pipeline into <see cref="InMemoryLogStore"/>.</summary>
[ProviderAlias("InMemory")]
public sealed class InMemoryLoggerProvider(InMemoryLogStore store) : ILoggerProvider
{
    public ILogger CreateLogger(string categoryName) => new InMemoryLogger(store, categoryName);

    public void Dispose()
    {
    }

    private sealed class InMemoryLogger(InMemoryLogStore store, string category) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Debug;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(logLevel))
            {
                return;
            }

            store.Add(new LogEntry
            {
                Level = logLevel,
                // Only the type name is useful in a UI; the namespace is noise.
                Category = category[(category.LastIndexOf('.') + 1)..],
                Message = formatter(state, exception),
                Exception = exception?.ToString()
            });
        }
    }
}
