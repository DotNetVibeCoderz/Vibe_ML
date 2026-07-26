using BlazorML.Core.Analysis;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;

namespace BlazorML.ML.Evaluation;

/// <summary>
/// Computes metrics from a scored table rather than from an ML.NET metrics object.
/// <para>
/// ML.NET reports summary numbers but not the ROC curve points or a residual sample, and the
/// designer needs both to draw anything useful. Working from the scored rows gives one code path
/// that produces the summary numbers <em>and</em> the shapes the charts need, and it also works
/// for tables scored outside ML.NET — an LLM classification, say.
/// </para>
/// </summary>
public static class Evaluator
{
    public const string PredictedLabel = "PredictedLabel";
    public const string Probability = "Probability";
    public const string Score = "Score";
    public const string ClusterId = "PredictedClusterId";

    public static EvaluationResult Evaluate(TabularData scored, MlTask task, string labelColumn,
        string? positiveClass = null)
    {
        return task switch
        {
            MlTask.BinaryClassification => Binary(scored, labelColumn, positiveClass),
            MlTask.MulticlassClassification => Multiclass(scored, labelColumn),
            MlTask.Regression or MlTask.Forecasting => Regression(scored, labelColumn),
            MlTask.Clustering => Clustering(scored),
            MlTask.Recommendation => Regression(scored, labelColumn),
            MlTask.AnomalyDetection => Anomaly(scored),
            _ => new EvaluationResult { Task = task }
        };
    }

    // ------------------------------------------------------------------ Binary

    private static EvaluationResult Binary(TabularData scored, string labelColumn, string? positiveClass)
    {
        var result = new EvaluationResult { Task = MlTask.BinaryClassification };

        var actualIndex = scored.IndexOf(labelColumn);
        var predictedIndex = scored.IndexOf(PredictedLabel);
        var probabilityIndex = scored.IndexOf(Probability);

        if (actualIndex < 0 || predictedIndex < 0)
        {
            return result;
        }

        var positive = positiveClass ?? InferPositiveClass(scored, actualIndex);
        int tp = 0, fp = 0, tn = 0, fn = 0;
        var scoredPairs = new List<(double Probability, bool Actual)>();

        foreach (var row in scored.Rows)
        {
            var actual = IsPositive(row[actualIndex], positive);
            var predicted = IsPositive(row[predictedIndex], positive);

            if (actual && predicted) tp++;
            else if (!actual && predicted) fp++;
            else if (!actual && !predicted) tn++;
            else fn++;

            if (probabilityIndex >= 0)
            {
                var probability = TabularData.ToNumber(row[probabilityIndex]);
                if (probability.HasValue)
                {
                    scoredPairs.Add((probability.Value, actual));
                }
            }
        }

        var total = tp + fp + tn + fn;
        var precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
        var recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);

        result.Metrics["Accuracy"] = total == 0 ? 0 : (double)(tp + tn) / total;
        result.Metrics["Precision"] = precision;
        result.Metrics["Recall"] = recall;
        result.Metrics["F1"] = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        result.Metrics["Specificity"] = tn + fp == 0 ? 0 : (double)tn / (tn + fp);

        result.ClassLabels = [$"not {positive}", positive];
        result.ConfusionMatrix = [[tn, fp], [fn, tp]];

        if (scoredPairs.Count > 0)
        {
            result.Metrics["AUC"] = AreaUnderRoc(scoredPairs);
            result.RocCurve = RocCurve(scoredPairs);
        }

        return result;
    }

    /// <summary>
    /// Rank-based AUC (the Mann-Whitney U identity), which handles ties correctly and needs one
    /// sort rather than a sweep per threshold.
    /// </summary>
    private static double AreaUnderRoc(List<(double Probability, bool Actual)> pairs)
    {
        var positives = pairs.Count(p => p.Actual);
        var negatives = pairs.Count - positives;

        if (positives == 0 || negatives == 0)
        {
            return double.NaN;
        }

        var ordered = pairs.OrderBy(p => p.Probability).ToList();
        var ranks = new double[ordered.Count];

        var i = 0;
        while (i < ordered.Count)
        {
            var j = i;
            while (j + 1 < ordered.Count && Math.Abs(ordered[j + 1].Probability - ordered[i].Probability) < 1e-12)
            {
                j++;
            }

            // Tied scores share the average of the ranks they span.
            var averageRank = (i + j) / 2.0 + 1;
            for (var k = i; k <= j; k++)
            {
                ranks[k] = averageRank;
            }

            i = j + 1;
        }

        var positiveRankSum = 0.0;
        for (var k = 0; k < ordered.Count; k++)
        {
            if (ordered[k].Actual)
            {
                positiveRankSum += ranks[k];
            }
        }

        return (positiveRankSum - positives * (positives + 1) / 2.0) / ((double)positives * negatives);
    }

    private static List<RocPoint> RocCurve(List<(double Probability, bool Actual)> pairs, int maxPoints = 100)
    {
        var positives = pairs.Count(p => p.Actual);
        var negatives = pairs.Count - positives;

        if (positives == 0 || negatives == 0)
        {
            return [];
        }

        var ordered = pairs.OrderByDescending(p => p.Probability).ToList();
        var curve = new List<RocPoint> { new(0, 0, 1) };

        // Sample the sorted list so the chart stays light on large test sets.
        var stride = Math.Max(1, ordered.Count / maxPoints);
        int tp = 0, fp = 0;

        for (var i = 0; i < ordered.Count; i++)
        {
            if (ordered[i].Actual) tp++; else fp++;

            if (i % stride == 0 || i == ordered.Count - 1)
            {
                curve.Add(new RocPoint((double)fp / negatives, (double)tp / positives, ordered[i].Probability));
            }
        }

        curve.Add(new RocPoint(1, 1, 0));
        return curve;
    }

    private static string InferPositiveClass(TabularData scored, int actualIndex)
    {
        foreach (var row in scored.Rows)
        {
            var text = Convert.ToString(row[actualIndex])?.Trim();
            if (!string.IsNullOrEmpty(text) && text is not ("0" or "false" or "False" or "no"))
            {
                return text;
            }
        }

        return "1";
    }

    private static bool IsPositive(object? value, string positive)
    {
        var text = Convert.ToString(value)?.Trim() ?? string.Empty;

        if (string.Equals(text, positive, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return text.ToLowerInvariant() switch
        {
            "1" or "true" or "yes" or "ya" => positive.ToLowerInvariant() is "1" or "true" or "yes" or "ya",
            _ => false
        };
    }

    // -------------------------------------------------------------- Multiclass

    private static EvaluationResult Multiclass(TabularData scored, string labelColumn)
    {
        var result = new EvaluationResult { Task = MlTask.MulticlassClassification };

        var actualIndex = scored.IndexOf(labelColumn);
        var predictedIndex = scored.IndexOf(PredictedLabel);

        if (actualIndex < 0 || predictedIndex < 0)
        {
            return result;
        }

        var labels = scored.Rows
            .Select(r => Convert.ToString(r[actualIndex]) ?? string.Empty)
            .Concat(scored.Rows.Select(r => Convert.ToString(r[predictedIndex]) ?? string.Empty))
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var index = labels
            .Select((label, i) => (label, i))
            .ToDictionary(x => x.label, x => x.i, StringComparer.OrdinalIgnoreCase);

        var matrix = labels.Select(_ => new int[labels.Count].ToList()).ToList();
        var correct = 0;

        foreach (var row in scored.Rows)
        {
            var actual = Convert.ToString(row[actualIndex]) ?? string.Empty;
            var predicted = Convert.ToString(row[predictedIndex]) ?? string.Empty;

            if (!index.TryGetValue(actual, out var a) || !index.TryGetValue(predicted, out var p))
            {
                continue;
            }

            matrix[a][p]++;
            if (a == p)
            {
                correct++;
            }
        }

        result.ClassLabels = labels;
        result.ConfusionMatrix = matrix;
        result.Metrics["MicroAccuracy"] = scored.RowCount == 0 ? 0 : (double)correct / scored.RowCount;

        // Macro accuracy averages per-class recall, so a rare class counts as much as a common one.
        var recalls = new List<double>();
        for (var c = 0; c < labels.Count; c++)
        {
            var rowTotal = matrix[c].Sum();
            if (rowTotal > 0)
            {
                recalls.Add((double)matrix[c][c] / rowTotal);
            }
        }

        result.Metrics["MacroAccuracy"] = recalls.Count == 0 ? 0 : recalls.Average();
        result.Metrics["Classes"] = labels.Count;

        return result;
    }

    // -------------------------------------------------------------- Regression

    private static EvaluationResult Regression(TabularData scored, string labelColumn)
    {
        var result = new EvaluationResult { Task = MlTask.Regression };

        var actualIndex = scored.IndexOf(labelColumn);
        var predictedIndex = scored.IndexOf(Score) >= 0 ? scored.IndexOf(Score) : scored.IndexOf(PredictedLabel);

        if (actualIndex < 0 || predictedIndex < 0)
        {
            return result;
        }

        var pairs = new List<(double Actual, double Predicted)>();
        foreach (var row in scored.Rows)
        {
            var actual = TabularData.ToNumber(row[actualIndex]);
            var predicted = TabularData.ToNumber(row[predictedIndex]);

            if (actual.HasValue && predicted.HasValue)
            {
                pairs.Add((actual.Value, predicted.Value));
            }
        }

        if (pairs.Count == 0)
        {
            return result;
        }

        var mae = pairs.Average(p => Math.Abs(p.Actual - p.Predicted));
        var mse = pairs.Average(p => Math.Pow(p.Actual - p.Predicted, 2));
        var mean = pairs.Average(p => p.Actual);
        var totalSumSquares = pairs.Sum(p => Math.Pow(p.Actual - mean, 2));
        var residualSumSquares = pairs.Sum(p => Math.Pow(p.Actual - p.Predicted, 2));

        result.Metrics["MeanAbsoluteError"] = mae;
        result.Metrics["MeanSquaredError"] = mse;
        result.Metrics["RootMeanSquaredError"] = Math.Sqrt(mse);
        result.Metrics["RSquared"] = totalSumSquares < 1e-12 ? 0 : 1 - residualSumSquares / totalSumSquares;

        // The scatter is sampled: a chart with 200k points is slower to draw and no clearer.
        var stride = Math.Max(1, pairs.Count / 500);
        result.Residuals = pairs.Where((_, i) => i % stride == 0)
            .Select(p => new ScatterPoint(p.Actual, p.Predicted)).ToList();

        return result;
    }

    // -------------------------------------------------------------- Clustering

    private static EvaluationResult Clustering(TabularData scored)
    {
        var result = new EvaluationResult { Task = MlTask.Clustering };

        var clusterIndex = scored.IndexOf(ClusterId);
        if (clusterIndex < 0)
        {
            return result;
        }

        var sizes = new Dictionary<string, int>();
        foreach (var row in scored.Rows)
        {
            var cluster = Convert.ToString(row[clusterIndex]) ?? "?";
            sizes[cluster] = sizes.GetValueOrDefault(cluster) + 1;
        }

        result.Metrics["Clusters"] = sizes.Count;
        result.Metrics["LargestCluster"] = sizes.Count == 0 ? 0 : sizes.Values.Max();
        result.Metrics["SmallestCluster"] = sizes.Count == 0 ? 0 : sizes.Values.Min();

        // A balance close to 1 means the clusters came out evenly sized.
        result.Metrics["Balance"] = sizes.Count == 0 || sizes.Values.Max() == 0
            ? 0
            : (double)sizes.Values.Min() / sizes.Values.Max();

        result.ClassLabels = sizes.Keys.OrderBy(k => k).ToList();
        result.ConfusionMatrix = [result.ClassLabels.Select(k => sizes[k]).ToList()];

        return result;
    }

    private static EvaluationResult Anomaly(TabularData scored)
    {
        var result = new EvaluationResult { Task = MlTask.AnomalyDetection };

        var scoreIndex = scored.IndexOf(Score);
        if (scoreIndex < 0)
        {
            return result;
        }

        var scores = scored.Rows
            .Select(r => TabularData.ToNumber(r[scoreIndex]))
            .Where(s => s.HasValue)
            .Select(s => s!.Value)
            .OrderBy(s => s)
            .ToList();

        if (scores.Count == 0)
        {
            return result;
        }

        var p95 = scores[(int)(scores.Count * 0.95)];
        result.Metrics["MeanScore"] = scores.Average();
        result.Metrics["MaxScore"] = scores[^1];
        result.Metrics["P95Score"] = p95;
        result.Metrics["FlaggedAtP95"] = scores.Count(s => s >= p95);

        return result;
    }
}
