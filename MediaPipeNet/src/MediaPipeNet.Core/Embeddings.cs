namespace MediaPipeNet;

/// <summary>
/// A feature vector produced by an embedder (image, text). Compare embeddings of the same model with
/// <see cref="CosineSimilarity(Embedding, Embedding)"/>.
/// </summary>
/// <param name="Values">The float feature vector (L2-normalized when the task was asked to).</param>
/// <param name="QuantizedValues">
/// The scalar-quantized vector when requested: each value v (assumed in [-1, 1]) stored as
/// <c>clamp(round(v · 128), −128, 127)</c>, as MediaPipe does.
/// </param>
/// <param name="HeadIndex">Index of the model output head.</param>
/// <param name="HeadName">Name of the model output head, when known.</param>
public sealed record Embedding(IReadOnlyList<float> Values, IReadOnlyList<sbyte>? QuantizedValues = null, int HeadIndex = 0, string? HeadName = null)
{
    /// <summary>Number of dimensions.</summary>
    public int Dimension => Values.Count;

    /// <summary>
    /// Cosine similarity in [−1, 1] of two embeddings of the same model (uses the quantized values when
    /// both embeddings only make sense that way, i.e. when both carry them and neither has float values).
    /// </summary>
    /// <exception cref="ArgumentException">The embeddings have different dimensions or a zero norm.</exception>
    public static double CosineSimilarity(Embedding a, Embedding b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Values.Count == 0 && b.Values.Count == 0 && a.QuantizedValues is { } qa && b.QuantizedValues is { } qb)
            return Cosine(qa.Count, qb.Count, i => qa[i], i => qb[i]);
        return Cosine(a.Values.Count, b.Values.Count, i => a.Values[i], i => b.Values[i]);
    }

    private static double Cosine(int na, int nb, Func<int, double> a, Func<int, double> b)
    {
        if (na != nb) throw new ArgumentException($"Cannot compare embeddings of different dimensions ({na} vs {nb}).");
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < na; i++)
        {
            double x = a(i), y = b(i);
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        if (aa <= 0 || bb <= 0) throw new ArgumentException("Cannot compute the cosine similarity of a zero vector.");
        return dot / Math.Sqrt(aa * bb);
    }

    /// <summary>
    /// Builds an embedding from raw model output, optionally L2-normalizing and scalar-quantizing it
    /// (MediaPipe's <c>TensorsToEmbeddingsCalculator</c>).
    /// </summary>
    public static Embedding FromTensor(ReadOnlySpan<float> tensor, bool l2Normalize, bool quantize, int headIndex = 0, string? headName = null)
    {
        var values = tensor.ToArray();
        if (l2Normalize)
        {
            double sum = 0;
            foreach (var v in values) sum += v * v;
            if (sum > 0)
            {
                float inv = (float)(1.0 / Math.Sqrt(sum));
                for (int i = 0; i < values.Length; i++) values[i] *= inv;
            }
        }
        sbyte[]? quantized = null;
        if (quantize)
        {
            quantized = new sbyte[values.Length];
            for (int i = 0; i < values.Length; i++) quantized[i] = (sbyte)Math.Clamp(MathF.Round(values[i] * 128f), -128f, 127f);
        }
        return new Embedding(quantize ? [] : values, quantized, headIndex, headName);
    }
}
