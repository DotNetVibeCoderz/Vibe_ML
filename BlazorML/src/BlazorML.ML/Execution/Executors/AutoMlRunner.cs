using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Training;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace BlazorML.ML.Execution.Executors;

/// <summary>
/// The AutoML module. Hands the data to ML.NET's experiment runner, then reports the leaderboard
/// so the user can see what was tried and not just what won.
/// </summary>
internal static class AutoMlRunner
{
    public static object?[] Run(ModuleExecutionContext ctx)
    {
        var table = ctx.RequireTable(0, "Dataset");
        var label = ctx.RequireParam("labelColumn");
        var seconds = (uint)Math.Clamp(ctx.ParamInt("maxSeconds", 60), 10, 7200);

        var task = Enum.TryParse<MlTask>(ctx.Param("task"), out var parsed)
            ? parsed
            : MlTask.BinaryClassification;

        if (!table.Has(label))
        {
            throw new InvalidOperationException($"Label column '{label}' is not in the incoming data.");
        }

        if (task is MlTask.ImageClassification or MlTask.TextClassification)
        {
            return FineTune(ctx, table, label, task);
        }

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, label, task, out _);
        var labelName = MlDataBridge.Sanitise(label);

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Searching for up to {seconds}s over {table.RowCount:N0} rows, predicting '{label}'.");

        var leaderboard = TabularData.WithColumns("rank", "algorithm", "score", "seconds");

        return task switch
        {
            MlTask.Regression => Regression(ctx, view, labelName, label, seconds, leaderboard, table),
            MlTask.MulticlassClassification => Multiclass(ctx, view, labelName, label, seconds, leaderboard, table),
            MlTask.Recommendation => throw new NotSupportedException(
                "AutoML does not cover recommendation. Use Matrix Factorization with Tune Hyperparameters instead."),
            _ => Binary(ctx, view, labelName, label, seconds, leaderboard, table)
        };
    }

    /// <summary>
    /// Vision and NLP take the transfer-learning path rather than a search. Said plainly in the
    /// log, because "AutoML" implies trying many pipelines and that is not what happens here:
    /// there is one pretrained backbone and it is fine-tuned on the user's labels.
    /// </summary>
    private static object?[] FineTune(ModuleExecutionContext ctx, TabularData table, string label, MlTask task)
    {
        var isImage = task == MlTask.ImageClassification;
        var moduleId = isImage ? "algo.vision.imageClassification" : "algo.nlp.textClassification";

        var inputColumn = ctx.Param("inputColumn")
            ?? throw new InvalidOperationException(
                isImage
                    ? "Tell AutoML which column holds the image paths."
                    : "Tell AutoML which column holds the text.");

        var spec = new TrainerSpec
        {
            ModuleId = moduleId,
            DisplayName = isImage ? "Image Classification" : "Text Classification",
            Task = task,
            Parameters = new Dictionary<string, string?>
            {
                [isImage ? "imagePathColumn" : "textColumn"] = inputColumn,
                ["imageFolder"] = ctx.Param("imageFolder"),
                ["epochs"] = isImage ? "50" : "3"
            }
        };

        ctx.Log(Core.Domain.LogLevel.Info,
            $"{spec.DisplayName}: fine-tuning a pretrained model rather than searching pipelines — " +
            "there is one architecture here, and it is a good one.");

        var handle = VisionNlpTrainers.Fit(ctx, spec, table, label);

        var leaderboard = TabularData.WithColumns("rank", "algorithm", "score", "seconds");
        leaderboard.AddRow(1, spec.DisplayName, double.NaN, 0);

        return
        [
            handle,
            new MetricsBundle
            {
                Title = "Fine-tuned model",
                Table = leaderboard,
                Values = new Dictionary<string, double>()
            }
        ];
    }

    private static object?[] Binary(ModuleExecutionContext ctx, IDataView view, string labelName, string label,
        uint seconds, TabularData leaderboard, TabularData table)
    {
        var experiment = ctx.Ml.Auto().CreateBinaryClassificationExperiment(seconds);
        var result = experiment.Execute(view, labelColumnName: labelName);

        var ranked = result.RunDetails
            .Where(r => r.ValidationMetrics is not null)
            .OrderByDescending(r => r.ValidationMetrics!.AreaUnderRocCurve)
            .ToList();

        for (var i = 0; i < ranked.Count; i++)
        {
            leaderboard.AddRow(i + 1, ranked[i].TrainerName,
                ranked[i].ValidationMetrics!.AreaUnderRocCurve, ranked[i].RuntimeInSeconds);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Tried {result.RunDetails.Count()} pipelines. Best: {result.BestRun.TrainerName} " +
            $"at AUC {result.BestRun.ValidationMetrics.AreaUnderRocCurve:F4}.");

        return Package(result.BestRun.Model, view, MlTask.BinaryClassification,
            result.BestRun.TrainerName, label, table, leaderboard,
            new Dictionary<string, double>
            {
                ["AUC"] = result.BestRun.ValidationMetrics.AreaUnderRocCurve,
                ["Accuracy"] = result.BestRun.ValidationMetrics.Accuracy,
                ["F1"] = result.BestRun.ValidationMetrics.F1Score
            });
    }

    private static object?[] Multiclass(ModuleExecutionContext ctx, IDataView view, string labelName, string label,
        uint seconds, TabularData leaderboard, TabularData table)
    {
        var experiment = ctx.Ml.Auto().CreateMulticlassClassificationExperiment(seconds);
        var result = experiment.Execute(view, labelColumnName: labelName);

        var ranked = result.RunDetails
            .Where(r => r.ValidationMetrics is not null)
            .OrderByDescending(r => r.ValidationMetrics!.MacroAccuracy)
            .ToList();

        for (var i = 0; i < ranked.Count; i++)
        {
            leaderboard.AddRow(i + 1, ranked[i].TrainerName,
                ranked[i].ValidationMetrics!.MacroAccuracy, ranked[i].RuntimeInSeconds);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Tried {result.RunDetails.Count()} pipelines. Best: {result.BestRun.TrainerName} " +
            $"at macro accuracy {result.BestRun.ValidationMetrics.MacroAccuracy:F4}.");

        return Package(result.BestRun.Model, view, MlTask.MulticlassClassification,
            result.BestRun.TrainerName, label, table, leaderboard,
            new Dictionary<string, double>
            {
                ["MacroAccuracy"] = result.BestRun.ValidationMetrics.MacroAccuracy,
                ["MicroAccuracy"] = result.BestRun.ValidationMetrics.MicroAccuracy,
                ["LogLoss"] = result.BestRun.ValidationMetrics.LogLoss
            });
    }

    private static object?[] Regression(ModuleExecutionContext ctx, IDataView view, string labelName, string label,
        uint seconds, TabularData leaderboard, TabularData table)
    {
        var experiment = ctx.Ml.Auto().CreateRegressionExperiment(seconds);
        var result = experiment.Execute(view, labelColumnName: labelName);

        var ranked = result.RunDetails
            .Where(r => r.ValidationMetrics is not null)
            .OrderByDescending(r => r.ValidationMetrics!.RSquared)
            .ToList();

        for (var i = 0; i < ranked.Count; i++)
        {
            leaderboard.AddRow(i + 1, ranked[i].TrainerName,
                ranked[i].ValidationMetrics!.RSquared, ranked[i].RuntimeInSeconds);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Tried {result.RunDetails.Count()} pipelines. Best: {result.BestRun.TrainerName} " +
            $"at R² {result.BestRun.ValidationMetrics.RSquared:F4}.");

        return Package(result.BestRun.Model, view, MlTask.Regression,
            result.BestRun.TrainerName, label, table, leaderboard,
            new Dictionary<string, double>
            {
                ["RSquared"] = result.BestRun.ValidationMetrics.RSquared,
                ["RootMeanSquaredError"] = result.BestRun.ValidationMetrics.RootMeanSquaredError,
                ["MeanAbsoluteError"] = result.BestRun.ValidationMetrics.MeanAbsoluteError
            });
    }

    private static object?[] Package(ITransformer model, IDataView view, MlTask task, string algorithm,
        string label, TabularData table, TabularData leaderboard, Dictionary<string, double> metrics)
    {
        var handle = new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = task,
            Algorithm = $"AutoML · {algorithm}",
            LabelColumn = label,
            FeatureColumns = table.Columns
                .Select(c => c.Name)
                .Where(n => !string.Equals(n, label, StringComparison.OrdinalIgnoreCase))
                .ToList(),
            TrainingSample = table
        };

        return
        [
            handle,
            new MetricsBundle
            {
                Title = "AutoML leaderboard",
                Table = leaderboard,
                Values = metrics
            }
        ];
    }
}
