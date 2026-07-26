using BlazorML.Core.Analysis;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.ML.Evaluation;
using BlazorML.ML.Training;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BlazorML.ML.Execution.Executors;

/// <summary>Algorithm modules do no work themselves: they hand a <see cref="TrainerSpec"/> downstream.</summary>
public sealed class AlgorithmExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("algo.", StringComparison.Ordinal);

    public Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx)
    {
        var spec = new TrainerSpec
        {
            ModuleId = ctx.Module.Id,
            DisplayName = ctx.Module.Name,
            Task = ctx.Module.Task,
            Parameters = ctx.Module.Parameters.ToDictionary(p => p.Name, p => ctx.Param(p.Name))
        };

        ctx.Log(Core.Domain.LogLevel.Info, $"Configured {spec}.");
        return Task.FromResult<object?[]>([spec]);
    }
}

/// <summary>Train Model, Train Clustering, AutoML, Tune Hyperparameters and Cross Validate.</summary>
public sealed class TrainingExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("train.", StringComparison.Ordinal);

    public Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx) => ctx.Module.Id switch
    {
        "train.model" => Task.FromResult(TrainSupervised(ctx)),
        "train.clustering" => Task.FromResult(TrainSupervised(ctx)),
        "train.crossValidate" => Task.FromResult(CrossValidate(ctx)),
        "train.sweep" => Task.FromResult(Sweep(ctx)),
        "train.autoML" => Task.FromResult(AutoMlRunner.Run(ctx)),
        _ => throw new NotSupportedException($"Unknown training module '{ctx.Module.Id}'.")
    };

    internal static object?[] TrainSupervised(ModuleExecutionContext ctx)
    {
        var spec = ctx.Require<TrainerSpec>(0, "Algorithm");
        var table = ctx.RequireTable(1, "Dataset");

        var label = ctx.Param("labelColumn") ?? spec.Text("labelColumn");
        var features = ctx.ParamList("featureColumns");

        var handle = Fit(ctx, spec, table, label, features);
        ctx.Log(Core.Domain.LogLevel.Info,
            $"Trained {spec.DisplayName} on {table.RowCount:N0} rows using {handle.FeatureColumns.Count} features.");

        return [handle];
    }

    /// <summary>The one place a trainer is actually fitted, shared by training, sweeps and folds.</summary>
    internal static TrainedModelHandle Fit(ModuleExecutionContext ctx, TrainerSpec spec, TabularData table,
        string? label, IReadOnlyCollection<string>? features)
    {
        if (spec.Task != MlTask.Clustering && spec.Task != MlTask.AnomalyDetection &&
            string.IsNullOrWhiteSpace(label))
        {
            throw new InvalidOperationException(
                $"'{ctx.Node.Label}' needs a label column: {spec.DisplayName} learns to predict one.");
        }

        if (PipelineBuilder.NeedsSpecialPath(spec.ModuleId))
        {
            return SpecialTrainers.Fit(ctx, spec, table, label);
        }

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, label, spec.Task, out _);

        var featurization = PipelineBuilder.BuildFeaturization(
            ctx.Ml, table, spec.Task, label, features, out var used);

        // PCA cannot project onto more dimensions than the data has. The module's default rank is
        // 5, so a two-column dataset would otherwise fail with ML.NET's own "rank cannot be larger
        // than the original dimension", which says nothing about what the user should change.
        // One-hot encoding only widens the vector, so the column count is a safe lower bound.
        if (spec.ModuleId == "algo.anomaly.pca" && spec.Int("rank", 5) > used.Count)
        {
            ctx.Log(Core.Domain.LogLevel.Warning,
                $"Reduced the component count from {spec.Int("rank", 5)} to {used.Count}: " +
                $"there are only {used.Count} feature columns to project onto.");

            spec = spec.With("rank", used.Count.ToString());
        }

        var trainer = PipelineBuilder.BuildTrainer(ctx.Ml, spec);
        IEstimator<ITransformer> pipeline = featurization.Append(trainer);

        // Multiclass predictions come back as keys; map them to the original labels so the scored
        // table shows "setosa" rather than "2".
        if (spec.Task == MlTask.MulticlassClassification)
        {
            pipeline = pipeline.Append(ctx.Ml.Transforms.Conversion.MapKeyToValue(
                Evaluator.PredictedLabel, Evaluator.PredictedLabel));
        }

        var model = pipeline.Fit(view);

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = spec.Task,
            Algorithm = spec.DisplayName,
            LabelColumn = label,
            FeatureColumns = used,
            TrainingSample = table
        };
    }

    private static object?[] CrossValidate(ModuleExecutionContext ctx)
    {
        var spec = ctx.Require<TrainerSpec>(0, "Algorithm");
        var table = ctx.RequireTable(1, "Dataset");
        var label = ctx.RequireParam("labelColumn");
        var folds = Math.Clamp(ctx.ParamInt("folds", 5), 2, 20);

        var shuffled = table.Rows.OrderBy(_ => Random.Shared.Next()).ToList();
        var foldSize = Math.Max(1, shuffled.Count / folds);

        var perFold = new List<Dictionary<string, double>>();
        TrainedModelHandle? best = null;
        var bestScore = double.NegativeInfinity;

        for (var f = 0; f < folds; f++)
        {
            ctx.Ct.ThrowIfCancellationRequested();

            var testStart = f * foldSize;
            var testEnd = f == folds - 1 ? shuffled.Count : testStart + foldSize;

            var train = table.CloneSchema();
            var test = table.CloneSchema();

            for (var i = 0; i < shuffled.Count; i++)
            {
                (i >= testStart && i < testEnd ? test : train).Rows.Add((object?[])shuffled[i].Clone());
            }

            var handle = Fit(ctx, spec, train, label, null);
            var scored = ScoringExecutor.Score(ctx, handle, test, true);
            var evaluation = Evaluator.Evaluate(scored, spec.Task, label);

            perFold.Add(evaluation.Metrics);

            var headline = Headline(evaluation, spec.Task);
            ctx.Log(Core.Domain.LogLevel.Info, $"Fold {f + 1}/{folds}: {headline:F4}");

            if (headline > bestScore)
            {
                bestScore = headline;
                best = handle;
            }
        }

        // Report the mean and the spread: a single fold's score hides how stable the model is.
        var summary = new Dictionary<string, double>();
        var table2 = TabularData.WithColumns("metric", "mean", "stdDev", "min", "max");

        foreach (var metric in perFold.SelectMany(m => m.Keys).Distinct())
        {
            var values = perFold.Where(m => m.ContainsKey(metric)).Select(m => m[metric]).ToList();
            if (values.Count == 0)
            {
                continue;
            }

            var mean = values.Average();
            var stdDev = values.Count > 1
                ? Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1))
                : 0;

            summary[metric] = mean;
            summary[metric + "_stdDev"] = stdDev;
            table2.AddRow(metric, mean, stdDev, values.Min(), values.Max());
        }

        var bundle = new MetricsBundle
        {
            Title = $"{folds}-fold cross validation",
            Table = table2,
            Values = summary,
            Evaluation = new EvaluationResult { Task = spec.Task, Metrics = summary }
        };

        return [bundle, best];
    }

    private static object?[] Sweep(ModuleExecutionContext ctx)
    {
        var spec = ctx.Require<TrainerSpec>(0, "Algorithm");
        var table = ctx.RequireTable(1, "Dataset");
        var label = ctx.RequireParam("labelColumn");
        var iterations = Math.Clamp(ctx.ParamInt("iterations", 12), 2, 200);
        var grid = ParseGrid(ctx.Param("sweep"));

        if (grid.Count == 0)
        {
            throw new InvalidOperationException(
                "The parameter grid is empty. Write one parameter per line, for example: numberOfLeaves: 10, 20, 40");
        }

        var combinations = Combinations(grid);
        if (ctx.Param("mode") != "grid")
        {
            combinations = combinations.OrderBy(_ => Random.Shared.Next()).Take(iterations).ToList();
        }

        // Hold out a fixed slice so every candidate is judged on the same rows.
        var split = (int)(table.RowCount * 0.8);
        var train = table.CloneSchema();
        var test = table.CloneSchema();

        for (var i = 0; i < table.RowCount; i++)
        {
            (i < split ? train : test).Rows.Add((object?[])table.Rows[i].Clone());
        }

        var results = TabularData.WithColumns("run", "settings", "score");
        TrainedModelHandle? best = null;
        var bestScore = double.NegativeInfinity;

        for (var i = 0; i < combinations.Count; i++)
        {
            ctx.Ct.ThrowIfCancellationRequested();

            var candidate = spec;
            foreach (var (name, value) in combinations[i])
            {
                candidate = candidate.With(name, value);
            }

            try
            {
                var handle = Fit(ctx, candidate, train, label, null);
                var scored = ScoringExecutor.Score(ctx, handle, test, true);
                var score = Headline(Evaluator.Evaluate(scored, spec.Task, label), spec.Task);

                var settings = string.Join(", ", combinations[i].Select(kv => $"{kv.Key}={kv.Value}"));
                results.AddRow(i + 1, settings, score);

                if (score > bestScore)
                {
                    bestScore = score;
                    best = handle;
                }
            }
            catch (Exception e)
            {
                // One bad combination should not end the sweep; record it and carry on.
                ctx.Log(Core.Domain.LogLevel.Warning, $"Run {i + 1} failed: {e.Message}");
                results.AddRow(i + 1, string.Join(", ", combinations[i].Select(kv => $"{kv.Key}={kv.Value}")), double.NaN);
            }
        }

        if (best is null)
        {
            throw new InvalidOperationException("Every combination in the sweep failed. Check the log for the reason.");
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Best of {combinations.Count} runs scored {bestScore:F4}.");

        return [best, new MetricsBundle
        {
            Title = "Sweep results",
            Table = results,
            Values = new Dictionary<string, double> { ["BestScore"] = bestScore }
        }];
    }

    private static Dictionary<string, List<string>> ParseGrid(string? text)
    {
        var grid = new Dictionary<string, List<string>>();

        foreach (var line in (text ?? string.Empty)
                     .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var separator = line.IndexOf(':');
            if (separator <= 0)
            {
                continue;
            }

            var name = line[..separator].Trim();
            var values = line[(separator + 1)..]
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .ToList();

            if (values.Count > 0)
            {
                grid[name] = values;
            }
        }

        return grid;
    }

    private static List<Dictionary<string, string>> Combinations(Dictionary<string, List<string>> grid)
    {
        var result = new List<Dictionary<string, string>> { new() };

        foreach (var (name, values) in grid)
        {
            result = result
                .SelectMany(partial => values.Select(v => new Dictionary<string, string>(partial) { [name] = v }))
                .ToList();
        }

        return result;
    }

    /// <summary>The single number a sweep or fold comparison is ranked on, higher being better.</summary>
    internal static double Headline(EvaluationResult evaluation, MlTask task)
    {
        var m = evaluation.Metrics;

        return task switch
        {
            MlTask.BinaryClassification => m.GetValueOrDefault("AUC") is var auc && !double.IsNaN(auc) && auc > 0
                ? auc : m.GetValueOrDefault("Accuracy"),
            MlTask.MulticlassClassification => m.GetValueOrDefault("MacroAccuracy"),
            MlTask.Regression or MlTask.Forecasting or MlTask.Recommendation => m.GetValueOrDefault("RSquared"),
            MlTask.Clustering => m.GetValueOrDefault("Balance"),
            _ => m.Values.FirstOrDefault()
        };
    }
}

/// <summary>Score Model, Evaluate Model and Feature Importance.</summary>
public sealed class ScoringExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("score.", StringComparison.Ordinal);

    public Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx) => ctx.Module.Id switch
    {
        "score.model" => Task.FromResult(ScoreModel(ctx)),
        "score.evaluate" => Task.FromResult(EvaluateModel(ctx)),
        "score.featureImportance" => Task.FromResult(FeatureImportance(ctx)),
        _ => throw new NotSupportedException($"Unknown scoring module '{ctx.Module.Id}'.")
    };

    private static object?[] ScoreModel(ModuleExecutionContext ctx)
    {
        var handle = ctx.Require<TrainedModelHandle>(0, "Trained model");
        var table = ctx.RequireTable(1, "Dataset");
        var keepInput = ctx.ParamBool("appendOriginal", true);

        var scored = Score(ctx, handle, table, keepInput);
        ctx.Log(Core.Domain.LogLevel.Info, $"Scored {scored.RowCount:N0} rows.");
        return [scored];
    }

    /// <summary>Runs rows through a fitted model and returns them with the prediction columns attached.</summary>
    internal static TabularData Score(ModuleExecutionContext ctx, TrainedModelHandle handle,
        TabularData table, bool keepInput)
    {
        if (handle.Task == MlTask.Forecasting)
        {
            throw new InvalidOperationException(
                "Forecasting models produce their horizon at training time. Read the forecast from the " +
                "training node's output instead of scoring rows through it.");
        }

        // Scoring unlabelled rows is the normal case in production, so put a placeholder label
        // back if the caller did not supply one.
        var input = MlDataBridge.EnsureLabelColumn(table, handle.LabelColumn, handle.Task);

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, input, handle.LabelColumn, handle.Task, out _);
        var transformed = handle.Transformer.Transform(view);

        var predictions = MlDataBridge.FromDataView(transformed);

        // ML.NET names the clustering output PredictedLabel, the same as a classifier's. Renaming
        // it here keeps the scored table self-describing — a cluster index is not a class label —
        // and lets the evaluator tell the two tasks apart from the columns alone.
        if (handle.Task == MlTask.Clustering && predictions.Has(Evaluator.PredictedLabel))
        {
            var index = predictions.IndexOf(Evaluator.PredictedLabel);
            predictions.Columns[index] = predictions.Columns[index] with { Name = Evaluator.ClusterId };
        }

        if (!keepInput)
        {
            return predictions;
        }

        // Carry the original columns through: evaluation needs the true label next to the
        // prediction, and the user wants to see the row the prediction belongs to.
        var output = table.Clone();
        foreach (var column in predictions.Columns)
        {
            if (!IsPredictionColumn(column.Name) || output.Has(column.Name))
            {
                continue;
            }

            var source = predictions.IndexOf(column.Name);
            var target = output.AddColumn(column.Name, column.DataType);

            for (var r = 0; r < output.RowCount && r < predictions.RowCount; r++)
            {
                output.Rows[r][target] = predictions.Rows[r][source];
            }
        }

        return output;
    }

    private static bool IsPredictionColumn(string name) =>
        name is Evaluator.PredictedLabel or Evaluator.Probability or Evaluator.Score or Evaluator.ClusterId
            or "PredictedRating" or "Alert" or "P-Value" or "Martingale score";

    private static object?[] EvaluateModel(ModuleExecutionContext ctx)
    {
        var scored = ctx.RequireTable(0, "Scored dataset");

        var task = ctx.Param("task") is { } text && text != "auto" && Enum.TryParse<MlTask>(text, out var parsed)
            ? parsed
            : InferTask(scored);

        var label = FindLabelColumn(scored, task);
        var evaluation = Evaluator.Evaluate(scored, task, label, ctx.Param("positiveClass"));

        var summary = string.Join(", ", evaluation.Metrics.Take(4).Select(m => $"{m.Key} {m.Value:F4}"));
        ctx.Log(Core.Domain.LogLevel.Info, $"{task}: {summary}");

        var metricTable = TabularData.WithColumns("metric", "value");
        foreach (var (name, value) in evaluation.Metrics)
        {
            metricTable.AddRow(name, value);
        }

        return
        [
            new MetricsBundle
            {
                Title = $"{task} metrics",
                Evaluation = evaluation,
                Table = metricTable,
                Values = evaluation.Metrics
            }
        ];
    }

    /// <summary>Reads the task back off the prediction columns the scorer produced.</summary>
    private static MlTask InferTask(TabularData scored)
    {
        if (scored.Has(Evaluator.ClusterId))
        {
            return MlTask.Clustering;
        }

        if (scored.Has(Evaluator.PredictedLabel))
        {
            var distinct = scored.Rows
                .Select(r => Convert.ToString(r[scored.IndexOf(Evaluator.PredictedLabel)]))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Count();

            return distinct <= 2 ? MlTask.BinaryClassification : MlTask.MulticlassClassification;
        }

        return scored.Has(Evaluator.Score) ? MlTask.Regression : MlTask.None;
    }

    /// <summary>
    /// The truth column to compare against. Scored tables keep the original columns, so the label
    /// is whichever column the prediction is trying to reproduce.
    /// </summary>
    private static string FindLabelColumn(TabularData scored, MlTask task)
    {
        foreach (var candidate in (string[])["Label", "label", "target", "Target", "y"])
        {
            if (scored.Has(candidate))
            {
                return candidate;
            }
        }

        var predictionColumns = new[]
        {
            Evaluator.PredictedLabel, Evaluator.Probability, Evaluator.Score, Evaluator.ClusterId
        };

        // Fall back to the last non-prediction column, which is where a label usually sits.
        var candidates = scored.Columns
            .Select(c => c.Name)
            .Where(n => !predictionColumns.Contains(n, StringComparer.OrdinalIgnoreCase))
            .ToList();

        return candidates.LastOrDefault() ?? "Label";
    }

    private static object?[] FeatureImportance(ModuleExecutionContext ctx)
    {
        var handle = ctx.Require<TrainedModelHandle>(0, "Trained model");
        var table = ctx.RequireTable(1, "Dataset");
        var label = ctx.Param("labelColumn") ?? handle.LabelColumn
            ?? throw new InvalidOperationException("Feature importance needs to know which column is the label.");

        var permutations = Math.Clamp(ctx.ParamInt("permutations", 5), 1, 50);

        var baseline = TrainingExecutor.Headline(
            Evaluator.Evaluate(Score(ctx, handle, table, true), handle.Task, label), handle.Task);

        var weights = new List<FeatureWeight>();

        foreach (var feature in handle.FeatureColumns)
        {
            ctx.Ct.ThrowIfCancellationRequested();

            var index = table.IndexOf(feature);
            if (index < 0)
            {
                continue;
            }

            var drops = new List<double>();

            for (var p = 0; p < permutations; p++)
            {
                // Shuffling one column breaks its link to the label. How far the score falls is
                // how much the model was relying on that column.
                var shuffled = table.Clone();
                var values = shuffled.Rows.Select(r => r[index]).OrderBy(_ => Random.Shared.Next()).ToList();

                for (var r = 0; r < shuffled.RowCount; r++)
                {
                    shuffled.Rows[r][index] = values[r];
                }

                var score = TrainingExecutor.Headline(
                    Evaluator.Evaluate(Score(ctx, handle, shuffled, true), handle.Task, label), handle.Task);

                drops.Add(baseline - score);
            }

            weights.Add(new FeatureWeight(feature, drops.Average()));
        }

        var ranked = weights.OrderByDescending(w => w.Weight).ToList();
        var importanceTable = TabularData.WithColumns("feature", "importance");
        foreach (var weight in ranked)
        {
            importanceTable.AddRow(weight.Feature, weight.Weight);
        }

        ctx.Log(Core.Domain.LogLevel.Info,
            ranked.Count == 0
                ? "No features could be measured."
                : $"'{ranked[0].Feature}' matters most, costing {ranked[0].Weight:F4} when shuffled.");

        return
        [
            new MetricsBundle
            {
                Title = "Feature importance",
                Table = importanceTable,
                Evaluation = new EvaluationResult { Task = handle.Task, FeatureImportance = ranked },
                Values = ranked.ToDictionary(w => w.Feature, w => w.Weight)
            }
        ];
    }
}
