using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn;

/// <summary>
/// Evaluation metrics for classification and regression, plus the confusion matrix and
/// classification report that summarise them.
/// </summary>
public static class Metrics
{
    // ---------------------------------------------------------------- classification

    /// <summary>Fraction of predictions that match the truth.</summary>
    public static double Accuracy(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var correct = 0;
        for (var i = 0; i < yTrue.Size; i++) if (yTrue.At(i) == yPredicted.At(i)) correct++;
        return (double)correct / yTrue.Size;
    }

    /// <summary>
    /// Counts of true and predicted label pairs; rows are truth, columns are predictions.
    /// </summary>
    public static NdArray ConfusionMatrix(NdArray yTrue, NdArray yPredicted, IReadOnlyList<double>? labels = null)
    {
        RequireSameLength(yTrue, yPredicted);
        var classes = labels ?? yTrue.ToArray().Concat(yPredicted.ToArray()).Distinct().OrderBy(v => v).ToArray();
        var index = classes.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);

        var matrix = NdArray.Zeros(classes.Count, classes.Count);
        for (var i = 0; i < yTrue.Size; i++)
        {
            if (!index.TryGetValue(yTrue.At(i), out var row)) continue;
            if (!index.TryGetValue(yPredicted.At(i), out var column)) continue;
            matrix[row, column] += 1.0;
        }
        return matrix;
    }

    /// <summary>
    /// Precision for one class: of everything predicted as this class, how much really was.
    /// </summary>
    public static double Precision(NdArray yTrue, NdArray yPredicted, double positiveLabel = 1.0)
    {
        var (tp, fp, _, _) = Counts(yTrue, yPredicted, positiveLabel);
        return tp + fp == 0 ? 0.0 : tp / (tp + fp);
    }

    /// <summary>Recall for one class: of everything that really was this class, how much was found.</summary>
    public static double Recall(NdArray yTrue, NdArray yPredicted, double positiveLabel = 1.0)
    {
        var (tp, _, fn, _) = Counts(yTrue, yPredicted, positiveLabel);
        return tp + fn == 0 ? 0.0 : tp / (tp + fn);
    }

    /// <summary>Harmonic mean of precision and recall.</summary>
    public static double F1Score(NdArray yTrue, NdArray yPredicted, double positiveLabel = 1.0)
    {
        var precision = Precision(yTrue, yPredicted, positiveLabel);
        var recall = Recall(yTrue, yPredicted, positiveLabel);
        return precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
    }

    /// <summary>Specificity: of everything that was not this class, how much was correctly excluded.</summary>
    public static double Specificity(NdArray yTrue, NdArray yPredicted, double positiveLabel = 1.0)
    {
        var (_, fp, _, tn) = Counts(yTrue, yPredicted, positiveLabel);
        return tn + fp == 0 ? 0.0 : tn / (tn + fp);
    }

    /// <summary>Matthews correlation coefficient, a balanced score even on skewed classes.</summary>
    public static double MatthewsCorrelation(NdArray yTrue, NdArray yPredicted, double positiveLabel = 1.0)
    {
        var (tp, fp, fn, tn) = Counts(yTrue, yPredicted, positiveLabel);
        var denominator = Math.Sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return denominator == 0 ? 0.0 : (tp * tn - fp * fn) / denominator;
    }

    /// <summary>
    /// Precision averaged over classes. <paramref name="average"/> is <c>macro</c> (unweighted),
    /// <c>weighted</c> (by class support) or <c>micro</c> (pooled counts, equal to accuracy).
    /// </summary>
    public static double PrecisionAverage(NdArray yTrue, NdArray yPredicted, string average = "macro")
        => Average(yTrue, yPredicted, average, Precision);

    /// <summary>Recall averaged over classes.</summary>
    public static double RecallAverage(NdArray yTrue, NdArray yPredicted, string average = "macro")
        => Average(yTrue, yPredicted, average, Recall);

    /// <summary>F1 averaged over classes.</summary>
    public static double F1Average(NdArray yTrue, NdArray yPredicted, string average = "macro")
        => Average(yTrue, yPredicted, average, F1Score);

    private static double Average(NdArray yTrue, NdArray yPredicted, string average,
        Func<NdArray, NdArray, double, double> metric)
    {
        var classes = yTrue.ToArray().Distinct().OrderBy(v => v).ToArray();
        if (average == "micro") return Accuracy(yTrue, yPredicted);

        var total = 0.0;
        var weightTotal = 0.0;
        foreach (var c in classes)
        {
            var support = yTrue.ToArray().Count(v => v == c);
            var weight = average == "weighted" ? support : 1.0;
            total += metric(yTrue, yPredicted, c) * weight;
            weightTotal += weight;
        }
        return weightTotal == 0 ? 0.0 : total / weightTotal;
    }

    /// <summary>
    /// Area under the ROC curve for a binary problem, computed from the rank of the scores.
    /// </summary>
    /// <remarks>
    /// The rank-based (Mann-Whitney) formulation is used rather than trapezoidal integration of a
    /// sampled curve: it is exact, handles ties correctly, and needs one sort rather than a
    /// threshold sweep.
    /// </remarks>
    public static double RocAucScore(NdArray yTrue, NdArray scores, double positiveLabel = 1.0)
    {
        RequireSameLength(yTrue, scores);
        var n = yTrue.Size;

        var order = Enumerable.Range(0, n).OrderBy(i => scores.At(i)).ToArray();
        var ranks = new double[n];
        var i2 = 0;
        while (i2 < n)
        {
            var j = i2;
            while (j + 1 < n && scores.At(order[j + 1]) == scores.At(order[i2])) j++;
            var averageRank = (i2 + j) / 2.0 + 1;
            for (var k = i2; k <= j; k++) ranks[order[k]] = averageRank;
            i2 = j + 1;
        }

        double positives = 0, negatives = 0, rankSum = 0;
        for (var i = 0; i < n; i++)
        {
            if (yTrue.At(i) == positiveLabel) { positives++; rankSum += ranks[i]; }
            else negatives++;
        }

        if (positives == 0 || negatives == 0) return double.NaN;
        return (rankSum - positives * (positives + 1) / 2.0) / (positives * negatives);
    }

    /// <summary>The ROC curve as false-positive rate, true-positive rate and threshold triples.</summary>
    public static (double[] FalsePositiveRate, double[] TruePositiveRate, double[] Thresholds) RocCurve(
        NdArray yTrue, NdArray scores, double positiveLabel = 1.0)
    {
        RequireSameLength(yTrue, scores);
        var order = Enumerable.Range(0, yTrue.Size).OrderByDescending(i => scores.At(i)).ToArray();

        var positives = yTrue.ToArray().Count(v => v == positiveLabel);
        var negatives = yTrue.Size - positives;

        var fpr = new List<double> { 0.0 };
        var tpr = new List<double> { 0.0 };
        var thresholds = new List<double> { double.PositiveInfinity };

        double tp = 0, fp = 0;
        foreach (var i in order)
        {
            if (yTrue.At(i) == positiveLabel) tp++; else fp++;
            fpr.Add(negatives == 0 ? 0 : fp / negatives);
            tpr.Add(positives == 0 ? 0 : tp / positives);
            thresholds.Add(scores.At(i));
        }
        return (fpr.ToArray(), tpr.ToArray(), thresholds.ToArray());
    }

    /// <summary>Mean cross-entropy between true labels and predicted probabilities.</summary>
    public static double LogLoss(NdArray yTrue, NdArray probabilities, IReadOnlyList<double>? classes = null)
    {
        var labels = classes ?? yTrue.ToArray().Distinct().OrderBy(v => v).ToArray();
        var index = labels.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);

        var total = 0.0;
        for (var i = 0; i < yTrue.Size; i++)
        {
            var column = index[yTrue.At(i)];
            var p = Math.Clamp(probabilities[i, column], 1e-15, 1 - 1e-15);
            total -= Math.Log(p);
        }
        return total / yTrue.Size;
    }

    /// <summary>
    /// The per-class precision, recall, F1 and support table, plus the accuracy and averages.
    /// </summary>
    public static string ClassificationReport(NdArray yTrue, NdArray yPredicted,
        IReadOnlyDictionary<double, string>? labelNames = null)
    {
        RequireSameLength(yTrue, yPredicted);
        var classes = yTrue.ToArray().Concat(yPredicted.ToArray()).Distinct().OrderBy(v => v).ToArray();

        var nameWidth = Math.Max(12, classes
            .Select(c => (labelNames is not null && labelNames.TryGetValue(c, out var n) ? n : c.ToString("0.##")).Length)
            .DefaultIfEmpty(12)
            .Max() + 2);

        var sb = new StringBuilder();
        sb.AppendLine($"{"class".PadRight(nameWidth)}{"precision",10}{"recall",10}{"f1-score",10}{"support",10}");
        sb.AppendLine(new string('-', nameWidth + 40));

        foreach (var c in classes)
        {
            var name = labelNames is not null && labelNames.TryGetValue(c, out var n) ? n : c.ToString("0.##");
            var support = yTrue.ToArray().Count(v => v == c);
            sb.AppendLine(
                $"{name.PadRight(nameWidth)}" +
                $"{Precision(yTrue, yPredicted, c),10:F4}" +
                $"{Recall(yTrue, yPredicted, c),10:F4}" +
                $"{F1Score(yTrue, yPredicted, c),10:F4}" +
                $"{support,10}");
        }

        sb.AppendLine(new string('-', nameWidth + 40));
        sb.AppendLine($"{"accuracy".PadRight(nameWidth)}{"",10}{"",10}{Accuracy(yTrue, yPredicted),10:F4}{yTrue.Size,10}");
        sb.AppendLine(
            $"{"macro avg".PadRight(nameWidth)}" +
            $"{PrecisionAverage(yTrue, yPredicted),10:F4}" +
            $"{RecallAverage(yTrue, yPredicted),10:F4}" +
            $"{F1Average(yTrue, yPredicted),10:F4}" +
            $"{yTrue.Size,10}");
        sb.Append(
            $"{"weighted avg".PadRight(nameWidth)}" +
            $"{PrecisionAverage(yTrue, yPredicted, "weighted"),10:F4}" +
            $"{RecallAverage(yTrue, yPredicted, "weighted"),10:F4}" +
            $"{F1Average(yTrue, yPredicted, "weighted"),10:F4}" +
            $"{yTrue.Size,10}");

        return sb.ToString();
    }

    /// <summary>Renders a confusion matrix as a labelled grid.</summary>
    public static string FormatConfusionMatrix(NdArray matrix, IReadOnlyList<double>? classes = null)
    {
        var n = matrix.Shape[0];
        var labels = classes?.Select(c => c.ToString("0.##")).ToArray()
            ?? Enumerable.Range(0, n).Select(i => i.ToString()).ToArray();

        var width = Math.Max(8, labels.Max(l => l.Length) + 2);
        var sb = new StringBuilder();
        sb.Append("true \\ pred".PadRight(width));
        foreach (var label in labels) sb.Append(label.PadLeft(width));
        sb.AppendLine();

        for (var i = 0; i < n; i++)
        {
            sb.Append(labels[i].PadRight(width));
            for (var j = 0; j < n; j++) sb.Append(matrix[i, j].ToString("0").PadLeft(width));
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static (double TruePositive, double FalsePositive, double FalseNegative, double TrueNegative) Counts(
        NdArray yTrue, NdArray yPredicted, double positiveLabel)
    {
        RequireSameLength(yTrue, yPredicted);
        double tp = 0, fp = 0, fn = 0, tn = 0;
        for (var i = 0; i < yTrue.Size; i++)
        {
            var actual = yTrue.At(i) == positiveLabel;
            var predicted = yPredicted.At(i) == positiveLabel;
            if (actual && predicted) tp++;
            else if (!actual && predicted) fp++;
            else if (actual) fn++;
            else tn++;
        }
        return (tp, fp, fn, tn);
    }

    // ---------------------------------------------------------------- regression

    /// <summary>Mean squared error.</summary>
    public static double MeanSquaredError(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var acc = 0.0;
        for (var i = 0; i < yTrue.Size; i++)
        {
            var e = yTrue.At(i) - yPredicted.At(i);
            acc += e * e;
        }
        return acc / yTrue.Size;
    }

    /// <summary>Root mean squared error, in the units of the target.</summary>
    public static double RootMeanSquaredError(NdArray yTrue, NdArray yPredicted)
        => Math.Sqrt(MeanSquaredError(yTrue, yPredicted));

    /// <summary>Mean absolute error, which weights outliers less than the squared error does.</summary>
    public static double MeanAbsoluteError(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var acc = 0.0;
        for (var i = 0; i < yTrue.Size; i++) acc += Math.Abs(yTrue.At(i) - yPredicted.At(i));
        return acc / yTrue.Size;
    }

    /// <summary>Mean absolute percentage error, as a fraction.</summary>
    public static double MeanAbsolutePercentageError(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var acc = 0.0;
        var counted = 0;
        for (var i = 0; i < yTrue.Size; i++)
        {
            if (yTrue.At(i) == 0) continue;
            acc += Math.Abs((yTrue.At(i) - yPredicted.At(i)) / yTrue.At(i));
            counted++;
        }
        return counted == 0 ? double.NaN : acc / counted;
    }

    /// <summary>Coefficient of determination: the fraction of variance the model explains.</summary>
    public static double R2Score(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var mean = Statistics.Mean(yTrue);
        double residual = 0, total = 0;
        for (var i = 0; i < yTrue.Size; i++)
        {
            var e = yTrue.At(i) - yPredicted.At(i);
            var d = yTrue.At(i) - mean;
            residual += e * e;
            total += d * d;
        }
        return total == 0 ? 0.0 : 1.0 - residual / total;
    }

    /// <summary>R-squared penalised for the number of predictors.</summary>
    public static double AdjustedR2Score(NdArray yTrue, NdArray yPredicted, int featureCount)
    {
        var r2 = R2Score(yTrue, yPredicted);
        var n = yTrue.Size;
        if (n - featureCount - 1 <= 0) return double.NaN;
        return 1.0 - (1.0 - r2) * (n - 1) / (n - featureCount - 1);
    }

    /// <summary>Fraction of target variance explained, ignoring any constant bias.</summary>
    public static double ExplainedVarianceScore(NdArray yTrue, NdArray yPredicted)
    {
        RequireSameLength(yTrue, yPredicted);
        var residuals = NdArray.Zeros(yTrue.Size);
        for (var i = 0; i < yTrue.Size; i++) residuals.SetAt(i, yTrue.At(i) - yPredicted.At(i));

        var variance = Statistics.Var(yTrue);
        return variance == 0 ? 0.0 : 1.0 - Statistics.Var(residuals) / variance;
    }

    /// <summary>A one-shot regression summary.</summary>
    public static string RegressionReport(NdArray yTrue, NdArray yPredicted, int featureCount = 0)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  MSE   : {MeanSquaredError(yTrue, yPredicted):F6}");
        sb.AppendLine($"  RMSE  : {RootMeanSquaredError(yTrue, yPredicted):F6}");
        sb.AppendLine($"  MAE   : {MeanAbsoluteError(yTrue, yPredicted):F6}");
        sb.AppendLine($"  R2    : {R2Score(yTrue, yPredicted):F6}");
        if (featureCount > 0)
            sb.AppendLine($"  Adj R2: {AdjustedR2Score(yTrue, yPredicted, featureCount):F6}");
        return sb.ToString();
    }

    // ---------------------------------------------------------------- clustering

    /// <summary>
    /// Silhouette score: how much closer each point sits to its own cluster than to the nearest
    /// other one. Ranges from -1 (misassigned) through 0 (on a boundary) to 1 (well separated).
    /// </summary>
    public static double SilhouetteScore(NdArray x, NdArray labels)
    {
        var n = x.Shape[0];
        var distinct = labels.ToArray().Distinct().Where(l => l >= 0).ToArray();
        if (distinct.Length < 2) return 0.0;

        var members = distinct.ToDictionary(
            l => l,
            l => Enumerable.Range(0, n).Where(i => labels.At(i) == l).ToArray());

        var total = 0.0;
        var counted = 0;

        for (var i = 0; i < n; i++)
        {
            var own = labels.At(i);
            if (own < 0) continue;
            if (members[own].Length <= 1) { counted++; continue; }

            var a = members[own].Where(j => j != i).Average(j => Distance(x, i, j));
            var b = distinct.Where(l => l != own).Min(l => members[l].Average(j => Distance(x, i, j)));

            total += (b - a) / Math.Max(a, b);
            counted++;
        }
        return counted == 0 ? 0.0 : total / counted;
    }

    /// <summary>Total squared distance from each point to its cluster centre.</summary>
    public static double Inertia(NdArray x, NdArray labels, NdArray centroids)
    {
        var acc = 0.0;
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var c = (int)labels.At(i);
            if (c < 0) continue;
            for (var j = 0; j < x.Shape[1]; j++)
            {
                var d = x[i, j] - centroids[c, j];
                acc += d * d;
            }
        }
        return acc;
    }

    private static double Distance(NdArray x, int i, int j)
    {
        var acc = 0.0;
        for (var d = 0; d < x.Shape[1]; d++)
        {
            var delta = x[i, d] - x[j, d];
            acc += delta * delta;
        }
        return Math.Sqrt(acc);
    }

    private static void RequireSameLength(NdArray a, NdArray b)
    {
        if (a.Size != b.Size)
            throw new ArgumentException($"Expected equal lengths, got {a.Size} and {b.Size}.");
    }
}
