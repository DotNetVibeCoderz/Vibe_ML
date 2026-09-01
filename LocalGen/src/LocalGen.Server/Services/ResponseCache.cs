using System.Security.Cryptography;
using System.Text;
using LocalGen.Core.Configuration;
using LocalGen.Core.Inference;
using Microsoft.Extensions.Options;

namespace LocalGen.Server.Services;

/// <summary>A generation held for replay, with everything needed to reconstruct the response.</summary>
public sealed record CachedCompletion
{
    public required string Text { get; init; }

    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];

    public FinishReason FinishReason { get; init; } = FinishReason.Stop;

    public TokenUsage Usage { get; init; } = TokenUsage.Empty;

    public DateTimeOffset StoredAt { get; init; } = DateTimeOffset.UtcNow;
}

/// <summary>
/// Caches completed generations, keyed on the prompt and every setting that shapes the output.
/// </summary>
/// <remarks>
/// The expensive part of a local deployment is the GPU, and the same prompt is asked repeatedly by
/// evaluation harnesses, retried agent steps and refreshed dashboards. Caching is off by default
/// and, when on, applies only to reproducible requests unless the operator opts in further: a
/// caller who set a temperature asked for variety, and replaying one answer forever would quietly
/// take that away.
/// </remarks>
public sealed class ResponseCache
{
    private readonly CacheOptions _options;
    private readonly Dictionary<string, LinkedListNode<CacheEntry>> _index = new(StringComparer.Ordinal);

    // Most-recently-used at the head. The list gives O(1) eviction of the coldest entry; the
    // dictionary gives O(1) lookup. Neither alone does both.
    private readonly LinkedList<CacheEntry> _order = new();
    private readonly Lock _gate = new();

    private long _hits;
    private long _misses;

    public ResponseCache(IOptions<LocalGenOptions> options) => _options = options.Value.Cache;

    public bool IsEnabled => _options.Enabled && _options.MaxEntries > 0;

    public long Hits => Interlocked.Read(ref _hits);

    public long Misses => Interlocked.Read(ref _misses);

    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _index.Count;
            }
        }
    }

    /// <summary>
    /// Whether this request may be served from cache at all.
    /// </summary>
    /// <remarks>
    /// A seeded request is reproducible even at a high temperature, so it caches safely. An
    /// unseeded one is only reproducible when sampling is greedy — temperature zero. Anything
    /// else needs the operator to have said explicitly that stale variety is acceptable.
    /// </remarks>
    public bool IsCacheable(ChatRequest request)
    {
        if (!IsEnabled)
        {
            return false;
        }

        if (_options.CacheNonDeterministic)
        {
            return true;
        }

        return request.Options.Seed is not null || request.Options.Temperature is 0f;
    }

    /// <summary>
    /// Fingerprints everything that can change the generated text.
    /// </summary>
    /// <remarks>
    /// Built from the domain request rather than the raw HTTP body so that two callers phrasing
    /// the same conversation differently on the wire — a bare string versus a single text part —
    /// still share an entry. Image bytes are folded in whole: two different pictures with the same
    /// file name must not collide.
    /// </remarks>
    public static string ComputeKey(ChatRequest request)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        void Write(string value) => hash.AppendData(Encoding.UTF8.GetBytes(value));

        Write("v1\u001fmodel\u001f");
        Write(request.Model);

        foreach (var message in request.Messages)
        {
            Write($"\u001e{message.Role}\u001f{message.Name}\u001f{message.ToolCallId}\u001f");

            foreach (var part in message.Content)
            {
                switch (part)
                {
                    case ContentPart.Text text:
                        Write($"t:{text.Value}\u001f");
                        break;
                    case ContentPart.Image image:
                        Write($"i:{image.MediaType}:{image.Data.Length}\u001f");
                        hash.AppendData(image.Data.Span);
                        break;
                    case ContentPart.Document document:
                        Write($"d:{document.FileName}\u001f{document.ExtractedText}\u001f");
                        break;
                }
            }

            foreach (var call in message.ToolCalls)
            {
                Write($"c:{call.Name}\u001f{call.ArgumentsJson}\u001f");
            }
        }

        foreach (var tool in request.Tools)
        {
            Write($"\u001dtool:{tool.Name}\u001f{tool.Description}\u001f{tool.ParametersJsonSchema}");
        }

        var options = request.Options;

        Write(string.Join(
            '\u001f',
            [
                "\u001dopts",
                options.Temperature?.ToString("R") ?? "-",
                options.TopP?.ToString("R") ?? "-",
                options.TopK?.ToString() ?? "-",
                options.MinP?.ToString("R") ?? "-",
                options.RepeatPenalty?.ToString("R") ?? "-",
                options.RepeatLastN?.ToString() ?? "-",
                options.PresencePenalty?.ToString("R") ?? "-",
                options.FrequencyPenalty?.ToString("R") ?? "-",
                options.MaxTokens?.ToString() ?? "-",
                options.Seed?.ToString() ?? "-",
                options.ContextSize?.ToString() ?? "-",
                options.JsonMode ? "json" : "-",
                options.Grammar ?? "-",
                string.Join('\u0000', options.StopSequences)
            ]));

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>Returns a live entry, or null on a miss or an expired one.</summary>
    public CachedCompletion? TryGet(string key)
    {
        if (!IsEnabled)
        {
            return null;
        }

        lock (_gate)
        {
            if (!_index.TryGetValue(key, out var node))
            {
                Interlocked.Increment(ref _misses);
                return null;
            }

            if (DateTimeOffset.UtcNow - node.Value.Completion.StoredAt > _options.Ttl)
            {
                _order.Remove(node);
                _index.Remove(key);
                Interlocked.Increment(ref _misses);
                return null;
            }

            _order.Remove(node);
            _order.AddFirst(node);

            Interlocked.Increment(ref _hits);
            return node.Value.Completion;
        }
    }

    public void Set(string key, CachedCompletion completion)
    {
        if (!IsEnabled)
        {
            return;
        }

        lock (_gate)
        {
            if (_index.TryGetValue(key, out var existing))
            {
                _order.Remove(existing);
                _index.Remove(key);
            }

            var node = _order.AddFirst(new CacheEntry(key, completion));
            _index[key] = node;

            while (_index.Count > _options.MaxEntries && _order.Last is { } coldest)
            {
                _order.RemoveLast();
                _index.Remove(coldest.Value.Key);
            }
        }
    }

    /// <summary>Drops every entry — used when a model is reloaded or replaced under the same name.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _index.Clear();
            _order.Clear();
        }
    }

    private readonly record struct CacheEntry(string Key, CachedCompletion Completion);
}
