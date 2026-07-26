using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Evaluation;
using BlazorML.ML.Training;
using Microsoft.ML;

#if VISION_NLP
// The text trainer is an extension method on the multiclass catalog, so its namespace has to be
// in scope for the call below to resolve.
using Microsoft.ML.TorchSharp;
#endif

namespace BlazorML.ML.Execution.Executors;

/// <summary>
/// Image and text classification by transfer learning: a pretrained backbone with a new head
/// fitted to the user's labels. This is what "AutoML for vision and NLP without deep coding"
/// means in practice — sensible defaults over a proven architecture, not an architecture search.
/// <para>
/// Compiled only when the solution is built with <c>-p:EnableVisionNlp=true</c>. The two backbones
/// need roughly 1.2 GB of native binaries between them — TensorFlow for images, libtorch for
/// text — which is around twenty times the rest of the solution's restore, so making everyone
/// pay it for two optional modules would be the wrong default. Both modules stay visible in the
/// catalogue either way, and say plainly how to enable them when they are not built in.
/// </para>
/// </summary>
internal static class VisionNlpTrainers
{
    public static bool IsBuiltIn =>
#if VISION_NLP
        true;
#else
        false;
#endif

    private const string NotBuiltIn =
        "{0} needs the vision and NLP pack, which is not included in this build. Rebuild with " +
        "-p:EnableVisionNlp=true — it downloads about 1.2 GB of machine-learning runtimes, which " +
        "is why it is off by default.";

    public static TrainedModelHandle Fit(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string? label)
    {
        if (string.IsNullOrWhiteSpace(label))
        {
            throw new InvalidOperationException(
                $"'{spec.DisplayName}' needs a label column: the category each row belongs to.");
        }

        return spec.ModuleId switch
        {
            "algo.vision.imageClassification" => FitImages(ctx, spec, table, label),
            "algo.nlp.textClassification" => FitText(ctx, spec, table, label),
            _ => throw new NotSupportedException($"'{spec.DisplayName}' has no vision or NLP path.")
        };
    }

#if VISION_NLP

    private static TrainedModelHandle FitImages(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string label)
    {
        var pathColumn = spec.Text("imagePathColumn")
            ?? throw new InvalidOperationException(
                "Image Classification needs the column holding each image's file path.");

        var folder = spec.Text("imageFolder") ?? string.Empty;

        if (!table.Has(pathColumn))
        {
            throw new InvalidOperationException($"Column '{pathColumn}' is not in the incoming data.");
        }

        var missing = table.Rows.Take(20)
            .Select(r => Convert.ToString(r[table.IndexOf(pathColumn)]))
            .Where(p => !string.IsNullOrWhiteSpace(p) && !File.Exists(Path.Combine(folder, p!)))
            .Take(3)
            .ToList();

        if (missing.Count > 0)
        {
            // Checked before training rather than after: transfer learning takes minutes, and
            // finding out at the end that the paths were wrong wastes all of it.
            throw new InvalidOperationException(
                $"These image files were not found under '{folder}': {string.Join(", ", missing)}. " +
                "Set the image folder, or use paths relative to it.");
        }

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, label, MlTask.MulticlassClassification, out _);

        var labelName = MlDataBridge.Sanitise(label);
        var pathName = MlDataBridge.Sanitise(pathColumn);

        var pipeline = ctx.Ml.Transforms.Conversion.MapValueToKey(PipelineBuilder.Label, labelName)
            .Append(ctx.Ml.Transforms.LoadRawImageBytes("ImageBytes", folder, pathName))
            .Append(ctx.Ml.MulticlassClassification.Trainers.ImageClassification(
                labelColumnName: PipelineBuilder.Label,
                featureColumnName: "ImageBytes"))
            .Append(ctx.Ml.Transforms.Conversion.MapKeyToValue(
                Evaluator.PredictedLabel, Evaluator.PredictedLabel));

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Fine-tuning an image classifier over {table.RowCount:N0} images. This takes minutes, not seconds.");

        var model = pipeline.Fit(view);

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = MlTask.ImageClassification,
            Algorithm = spec.DisplayName,
            LabelColumn = label,
            FeatureColumns = [pathColumn],
            TrainingSample = table
        };
    }

    private static TrainedModelHandle FitText(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string label)
    {
        var textColumn = spec.Text("textColumn")
            ?? throw new InvalidOperationException("Text Classification needs the column holding the text.");

        if (!table.Has(textColumn))
        {
            throw new InvalidOperationException($"Column '{textColumn}' is not in the incoming data.");
        }

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ctx.Ml, table, label, MlTask.MulticlassClassification, out _);

        var labelName = MlDataBridge.Sanitise(label);
        var textName = MlDataBridge.Sanitise(textColumn);

        var pipeline = ctx.Ml.Transforms.Conversion.MapValueToKey(PipelineBuilder.Label, labelName)
            .Append(ctx.Ml.MulticlassClassification.Trainers.TextClassification(
                labelColumnName: PipelineBuilder.Label,
                sentence1ColumnName: textName,
                maxEpochs: spec.Int("epochs", 3)))
            .Append(ctx.Ml.Transforms.Conversion.MapKeyToValue(
                Evaluator.PredictedLabel, Evaluator.PredictedLabel));

        ctx.Log(Core.Domain.LogLevel.Info,
            $"Fine-tuning a text classifier over {table.RowCount:N0} rows for {spec.Int("epochs", 3)} epochs.");

        var model = pipeline.Fit(view);

        return new TrainedModelHandle
        {
            Transformer = model,
            InputSchema = view.Schema,
            Task = MlTask.TextClassification,
            Algorithm = spec.DisplayName,
            LabelColumn = label,
            FeatureColumns = [textColumn],
            TrainingSample = table
        };
    }

#else

    private static TrainedModelHandle FitImages(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string label) =>
        throw new InvalidOperationException(string.Format(NotBuiltIn, spec.DisplayName));

    private static TrainedModelHandle FitText(ModuleExecutionContext ctx, TrainerSpec spec,
        TabularData table, string label) =>
        throw new InvalidOperationException(string.Format(NotBuiltIn, spec.DisplayName));

#endif
}
