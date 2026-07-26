using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Training;
using Microsoft.ML;

namespace BlazorML.ML.Execution.Executors;

/// <summary>
/// Algorithms that do not fit the "concatenate everything into a Features vector" shape.
/// Matrix factorization needs two key columns rather than a feature vector, and the time-series
/// estimators work on one ordered column, so each gets its own wiring here.
/// </summary>
internal static class SpecialTrainers
{
    public static TrainedModelHandle Fit(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string? label) => spec.ModuleId switch
    {
        "algo.rec.matrixFactorization" => MatrixFactorization(ctx, spec, table, label),
        "algo.anomaly.spike" => SpikeDetection(ctx, spec, table),
        "algo.forecast.ssa" => Forecast(ctx, spec, table),
        "algo.vision.imageClassification" or "algo.nlp.textClassification" =>
            VisionNlpTrainers.Fit(ctx, spec, table, label),
        _ => throw new NotSupportedException($"'{spec.DisplayName}' has no special training path.")
    };

    private static TrainedModelHandle MatrixFactorization(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string? label)
    {
        var userColumn = spec.Text("userColumn")
            ?? throw new InvalidOperationException("Matrix Factorization needs a user column.");
        var itemColumn = spec.Text("itemColumn")
            ?? throw new InvalidOperationException("Matrix Factorization needs an item column.");

        if (string.IsNullOrWhiteSpace(label))
        {
            throw new InvalidOperationException(
                "Matrix Factorization needs a rating column as the label — the number it learns to predict.");
        }

        var user = MlDataBridge.Sanitise(userColumn);
        var item = MlDataBridge.Sanitise(itemColumn);

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, label, MlTask.Recommendation, out _);

        // The trainer indexes a sparse matrix, so both sides have to become key types first.
        var pipeline = ctx.Ml.Transforms.Conversion.MapValueToKey("userKey", user)
            .Append(ctx.Ml.Transforms.Conversion.MapValueToKey("itemKey", item))
            .Append(ctx.Ml.Recommendation().Trainers.MatrixFactorization(
                labelColumnName: MlDataBridge.Sanitise(label),
                matrixColumnIndexColumnName: "userKey",
                matrixRowIndexColumnName: "itemKey",
                approximationRank: spec.Int("approximationRank", 32),
                learningRate: spec.Double("learningRate", 0.1),
                numberOfIterations: spec.Int("numberOfIterations", 20)));

        var model = pipeline.Fit(view);

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Factorised {table.RowCount:N0} ratings into {spec.Int("approximationRank", 32)} latent factors.");

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = MlTask.Recommendation,
            Algorithm = spec.DisplayName,
            LabelColumn = label,
            FeatureColumns = [userColumn, itemColumn],
            TrainingSample = table
        };
    }

    private static TrainedModelHandle SpikeDetection(ModuleExecutionContext ctx, TrainerSpec spec, TabularData table)
    {
        var column = spec.Text("column")
            ?? throw new InvalidOperationException("Spike Detection needs the column holding the series values.");

        var name = MlDataBridge.Sanitise(column);

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, null, MlTask.AnomalyDetection, out _);

        var estimator = ctx.Ml.Transforms.DetectIidSpike(
            outputColumnName: "SpikeVector",
            inputColumnName: name,
            confidence: spec.Double("confidence", 95),
            pvalueHistoryLength: spec.Int("pvalueHistoryLength", 30));

        var model = estimator.Fit(view);

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Fitted spike detection over {table.RowCount:N0} points at {spec.Double("confidence", 95)}% confidence.");

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = MlTask.AnomalyDetection,
            Algorithm = spec.DisplayName,
            LabelColumn = null,
            FeatureColumns = [column],
            TrainingSample = table
        };
    }

    private static TrainedModelHandle Forecast(ModuleExecutionContext ctx, TrainerSpec spec, TabularData table)
    {
        var column = spec.Text("column")
            ?? throw new InvalidOperationException("SSA Forecasting needs the column holding the series values.");

        var name = MlDataBridge.Sanitise(column);
        var horizon = spec.Int("horizon", 7);
        var windowSize = spec.Int("windowSize", 14);
        var seriesLength = Math.Min(spec.Int("seriesLength", 60), table.RowCount);

        if (table.RowCount < windowSize * 2)
        {
            throw new InvalidOperationException(
                $"SSA needs at least {windowSize * 2} points for a window of {windowSize}; the series has {table.RowCount}.");
        }

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, null, MlTask.Forecasting, out _);

        var estimator = ctx.Ml.Forecasting.ForecastBySsa(
            outputColumnName: "Forecast",
            inputColumnName: name,
            windowSize: windowSize,
            seriesLength: seriesLength,
            trainSize: table.RowCount,
            horizon: horizon);

        var model = estimator.Fit(view);

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Fitted SSA on {table.RowCount:N0} points; horizon {horizon} periods.");

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = MlTask.Forecasting,
            Algorithm = spec.DisplayName,
            LabelColumn = column,
            FeatureColumns = [column],
            TrainingSample = table
        };
    }
}
