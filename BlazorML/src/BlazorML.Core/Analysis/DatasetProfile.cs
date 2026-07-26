using BlazorML.Core.Domain;

namespace BlazorML.Core.Analysis;

/// <summary>Column statistics computed once at import and reused by the designer, the inspector and the agent.</summary>
public class DatasetProfile
{
    public int RowCount { get; set; }
    public List<ColumnProfile> Columns { get; set; } = new();

    /// <summary>First rows of the file, kept small enough to embed in the dataset record.</summary>
    public List<List<string>> SampleRows { get; set; } = new();

    public ColumnProfile? Column(string name) =>
        Columns.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));
}

public class ColumnProfile
{
    public string Name { get; set; } = string.Empty;
    public ColumnDataType DataType { get; set; }
    public int MissingCount { get; set; }
    public int DistinctCount { get; set; }

    public double? Min { get; set; }
    public double? Max { get; set; }
    public double? Mean { get; set; }
    public double? StdDev { get; set; }
    public double? Median { get; set; }

    /// <summary>Most frequent values, used for the categorical bar in the column inspector.</summary>
    public List<CategoryCount> TopValues { get; set; } = new();

    /// <summary>Equal-width bins over the numeric range, used for the histogram sparkline.</summary>
    public List<HistogramBin> Histogram { get; set; } = new();

    public bool IsNumeric => DataType == ColumnDataType.Numeric;
    public double MissingRatio(int rowCount) => rowCount == 0 ? 0 : (double)MissingCount / rowCount;
}

public record CategoryCount(string Value, int Count);

public record HistogramBin(double Start, double End, int Count);

/// <summary>Metrics from an evaluation module, shaped so one component can chart any task.</summary>
public class EvaluationResult
{
    public MlTask Task { get; set; }
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>Row-major counts, labels in <see cref="ClassLabels"/>.</summary>
    public List<List<int>>? ConfusionMatrix { get; set; }
    public List<string>? ClassLabels { get; set; }

    /// <summary>Points on the ROC curve for binary tasks.</summary>
    public List<RocPoint>? RocCurve { get; set; }

    /// <summary>Actual/predicted pairs sampled for the regression scatter.</summary>
    public List<ScatterPoint>? Residuals { get; set; }

    public List<FeatureWeight>? FeatureImportance { get; set; }
}

public record RocPoint(double FalsePositiveRate, double TruePositiveRate, double Threshold);

public record ScatterPoint(double Actual, double Predicted);

public record FeatureWeight(string Feature, double Weight);
