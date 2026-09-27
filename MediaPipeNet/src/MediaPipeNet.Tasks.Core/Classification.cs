namespace MediaPipeNet.Tasks;

/// <summary>Post-processing options of classification tasks (MediaPipe's <c>ClassifierOptions</c>).</summary>
public sealed record ClassifierOptions
{
    /// <summary>Default options: every category above 0, sorted by score.</summary>
    public static ClassifierOptions Default { get; } = new();

    /// <summary>Maximum categories returned (≤ 0 = all).</summary>
    public int MaxResults { get; init; } = -1;

    /// <summary>Minimum score for a category to be returned.</summary>
    public float ScoreThreshold { get; init; }

    /// <summary>If set, only these category names are returned.</summary>
    public IReadOnlySet<string>? CategoryAllowlist { get; init; }

    /// <summary>If set, these category names are never returned.</summary>
    public IReadOnlySet<string>? CategoryDenylist { get; init; }

    /// <summary>
    /// Turns a score vector into categories: filtered by threshold and allow/deny lists, sorted by
    /// descending score and truncated to <see cref="MaxResults"/>.
    /// </summary>
    public List<Category> Select(ReadOnlySpan<float> scores, IReadOnlyList<string>? labels)
    {
        var categories = new List<Category>();
        for (int i = 0; i < scores.Length; i++)
        {
            float score = scores[i];
            if (score < ScoreThreshold) continue;
            var label = labels is not null && i < labels.Count ? labels[i] : null;
            if (CategoryAllowlist is { } allow && (label is null || !allow.Contains(label))) continue;
            if (CategoryDenylist is { } deny && label is not null && deny.Contains(label)) continue;
            categories.Add(new Category(i, score, label));
        }
        categories.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (MaxResults > 0 && categories.Count > MaxResults) categories.RemoveRange(MaxResults, categories.Count - MaxResults);
        return categories;
    }
}
