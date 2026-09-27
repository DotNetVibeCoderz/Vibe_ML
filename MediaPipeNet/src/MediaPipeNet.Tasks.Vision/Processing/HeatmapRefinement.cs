namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// Refines regressed landmark positions with a per-landmark heatmap, as MediaPipe's
/// <c>RefineLandmarksFromHeatmapCalculator</c> does for BlazePose: each landmark moves to the
/// sigmoid-weighted centroid of a <c>kernelSize</c>² window of its heatmap channel, provided the
/// window's peak confidence reaches <c>minConfidence</c>.
/// </summary>
public static class HeatmapRefinement
{
    /// <summary>
    /// Refines <paramref name="xy"/> in place: interleaved (x, y) pairs in normalized tensor coordinates,
    /// one per heatmap channel. <paramref name="heatmap"/> is HWC (<paramref name="height"/> ×
    /// <paramref name="width"/> × channels) holding logits.
    /// </summary>
    public static void Refine(Span<float> xy, ReadOnlySpan<float> heatmap, int height, int width, int kernelSize = 7, float minConfidence = 0.5f)
    {
        int channels = xy.Length / 2;
        if (heatmap.Length < height * width * channels)
            throw new ArgumentException($"Heatmap needs {height * width * channels} values for {channels} landmarks.", nameof(heatmap));
        int rowSize = width * channels;
        int offset = (kernelSize - 1) / 2;
        for (int lm = 0; lm < channels; lm++)
        {
            int centerCol = (int)(xy[2 * lm] * width);
            int centerRow = (int)(xy[2 * lm + 1] * height);
            if (centerCol < 0 || centerCol >= width || centerRow < 0 || centerRow >= height) continue;

            int beginCol = Math.Max(0, centerCol - offset), endCol = Math.Min(width, centerCol + offset + 1);
            int beginRow = Math.Max(0, centerRow - offset), endRow = Math.Min(height, centerRow + offset + 1);
            float sum = 0, weightedCol = 0, weightedRow = 0, max = 0;
            for (int row = beginRow; row < endRow; row++)
            {
                for (int col = beginCol; col < endCol; col++)
                {
                    float confidence = DetectionDecoder.Sigmoid(heatmap[rowSize * row + channels * col + lm]);
                    sum += confidence;
                    max = MathF.Max(max, confidence);
                    weightedCol += col * confidence;
                    weightedRow += row * confidence;
                }
            }
            if (max >= minConfidence && sum > 0)
            {
                xy[2 * lm] = weightedCol / width / sum;
                xy[2 * lm + 1] = weightedRow / height / sum;
            }
        }
    }
}
