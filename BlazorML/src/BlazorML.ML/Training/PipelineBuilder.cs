using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace BlazorML.ML.Training;

/// <summary>
/// Turns "these columns, this task, this trainer" into a fitted ML.NET pipeline.
/// <para>
/// Featurisation is derived from the column types the profiler inferred, because the designer
/// never knows the schema ahead of time: numeric columns pass through, categoricals are one-hot
/// encoded, and free text is run through the standard text featuriser.
/// </para>
/// </summary>
public static class PipelineBuilder
{
    public const string Features = "Features";
    public const string Label = "Label";

    /// <summary>
    /// Builds the estimator chain up to but not including the trainer. Returns the feature
    /// columns actually used so the caller can record them on the model.
    /// </summary>
    public static IEstimator<ITransformer> BuildFeaturization(
        MLContext ml, TabularData table, MlTask task, string? labelColumn,
        IReadOnlyCollection<string>? requestedFeatures, out List<string> usedFeatures)
    {
        var label = labelColumn is null ? null : MlDataBridge.Sanitise(labelColumn);

        var candidates = table.Columns
            .Select(c => (Original: c.Name, Name: MlDataBridge.Sanitise(c.Name), c.DataType))
            .Where(c => label is null || !string.Equals(c.Name, label, StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (requestedFeatures is { Count: > 0 })
        {
            var wanted = requestedFeatures.Select(MlDataBridge.Sanitise).ToHashSet(StringComparer.OrdinalIgnoreCase);
            candidates = candidates.Where(c => wanted.Contains(c.Name)).ToList();
        }

        IEstimator<ITransformer>? chain = null;
        var featureParts = new List<string>();
        usedFeatures = new List<string>();

        foreach (var column in candidates)
        {
            switch (column.DataType)
            {
                case ColumnDataType.Numeric:
                    featureParts.Add(column.Name);
                    usedFeatures.Add(column.Original);
                    break;

                case ColumnDataType.Boolean:
                {
                    var output = column.Name + "_num";
                    chain = Append(chain, ml.Transforms.Conversion.ConvertType(output, column.Name, DataKind.Single));
                    featureParts.Add(output);
                    usedFeatures.Add(column.Original);
                    break;
                }

                case ColumnDataType.Categorical:
                {
                    var output = column.Name + "_cat";
                    chain = Append(chain, ml.Transforms.Categorical.OneHotEncoding(output, column.Name));
                    featureParts.Add(output);
                    usedFeatures.Add(column.Original);
                    break;
                }

                case ColumnDataType.Text:
                {
                    var output = column.Name + "_txt";
                    chain = Append(chain, ml.Transforms.Text.FeaturizeText(output, column.Name));
                    featureParts.Add(output);
                    usedFeatures.Add(column.Original);
                    break;
                }

                // DateTime and Unknown columns are left out rather than silently coerced into a
                // number that would mean nothing to the trainer.
            }
        }

        if (featureParts.Count == 0)
        {
            throw new InvalidOperationException(
                "No usable feature columns were found. Check that the columns you selected hold numbers, categories or text.");
        }

        chain = Append(chain, ml.Transforms.Concatenate(Features, featureParts.ToArray()));

        // Trees cope with missing values; linear trainers do not, so replace them everywhere.
        chain = Append(chain, ml.Transforms.ReplaceMissingValues(Features, Features,
            MissingValueReplacingEstimator.ReplacementMode.Mean));

        // Rescale before the trainer sees it. Tree trainers are indifferent — they split on
        // thresholds, and min-max is monotonic — but SDCA, the perceptron and the gradient
        // trainers are not: a column ranging 1..100 beside a one-hot column of 0..1 dominates
        // the gradient and the model badly underfits. Without this the same data scores ~0.95
        // through a tree and ~0.75 through a linear trainer, which reads as "that algorithm is
        // bad" when it is really "we handed it unscaled input".
        chain = Append(chain, ml.Transforms.NormalizeMinMax(Features, Features));

        // Multiclass wants a key-typed label, and the prediction has to be mapped back afterwards.
        if (task == MlTask.MulticlassClassification && label is not null)
        {
            chain = Append(chain, ml.Transforms.Conversion.MapValueToKey(Label, label));
        }
        else if (label is not null && !string.Equals(label, Label, StringComparison.Ordinal))
        {
            chain = Append(chain, ml.Transforms.CopyColumns(Label, label));
        }

        return chain!;
    }

    private static IEstimator<ITransformer> Append(IEstimator<ITransformer>? chain, IEstimator<ITransformer> next) =>
        chain is null ? next : chain.Append(next);

    /// <summary>Instantiates the trainer named by an algorithm module, with the user's settings applied.</summary>
    public static IEstimator<ITransformer> BuildTrainer(MLContext ml, TrainerSpec spec)
    {
        return spec.ModuleId switch
        {
            // ---------------------------------------------------------------- Binary
            "algo.bin.logistic" => ml.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f)),

            "algo.bin.sdca" => ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f),
                maximumNumberOfIterations: spec.Int("maximumNumberOfIterations", 100)),

            "algo.bin.fastTree" => ml.BinaryClassification.Trainers.FastTree(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 20),
                numberOfTrees: spec.Int("numberOfTrees", 100),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2)),

            "algo.bin.fastForest" => ml.BinaryClassification.Trainers.FastForest(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 20),
                numberOfTrees: spec.Int("numberOfTrees", 100),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10)),

            "algo.bin.lightGbm" => ml.BinaryClassification.Trainers.LightGbm(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 31),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2),
                numberOfIterations: spec.Int("numberOfTrees", 100)),

            "algo.bin.perceptron" => ml.BinaryClassification.Trainers.AveragedPerceptron(
                labelColumnName: Label, featureColumnName: Features,
                learningRate: spec.Float("learningRate", 0.1f),
                numberOfIterations: spec.Int("numberOfIterations", 10)),

            "algo.bin.svm" => ml.BinaryClassification.Trainers.LinearSvm(
                labelColumnName: Label, featureColumnName: Features,
                numberOfIterations: spec.Int("numberOfIterations", 10)),

            // ------------------------------------------------------------ Multiclass
            "algo.multi.sdca" => ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f),
                maximumNumberOfIterations: spec.Int("maximumNumberOfIterations", 100)),

            "algo.multi.lbfgs" => ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f)),

            "algo.multi.oneVsAll" => ml.MulticlassClassification.Trainers.OneVersusAll(
                ml.BinaryClassification.Trainers.FastTree(
                    labelColumnName: Label, featureColumnName: Features,
                    numberOfLeaves: spec.Int("numberOfLeaves", 20),
                    numberOfTrees: spec.Int("numberOfTrees", 100),
                    minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                    learningRate: spec.Double("learningRate", 0.2)),
                labelColumnName: Label),

            "algo.multi.lightGbm" => ml.MulticlassClassification.Trainers.LightGbm(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 31),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2),
                numberOfIterations: spec.Int("numberOfTrees", 100)),

            "algo.multi.naiveBayes" => ml.MulticlassClassification.Trainers.NaiveBayes(
                labelColumnName: Label, featureColumnName: Features),

            // ------------------------------------------------------------- Regression
            "algo.reg.sdca" => ml.Regression.Trainers.Sdca(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f),
                maximumNumberOfIterations: spec.Int("maximumNumberOfIterations", 100)),

            "algo.reg.ols" => ml.Regression.Trainers.Ols(
                labelColumnName: Label, featureColumnName: Features),

            "algo.reg.fastTree" => ml.Regression.Trainers.FastTree(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 20),
                numberOfTrees: spec.Int("numberOfTrees", 100),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2)),

            "algo.reg.fastForest" => ml.Regression.Trainers.FastForest(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 20),
                numberOfTrees: spec.Int("numberOfTrees", 100),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10)),

            "algo.reg.lightGbm" => ml.Regression.Trainers.LightGbm(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 31),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2),
                numberOfIterations: spec.Int("numberOfTrees", 100)),

            "algo.reg.tweedie" => ml.Regression.Trainers.FastTreeTweedie(
                labelColumnName: Label, featureColumnName: Features,
                numberOfLeaves: spec.Int("numberOfLeaves", 20),
                numberOfTrees: spec.Int("numberOfTrees", 100),
                minimumExampleCountPerLeaf: spec.Int("minimumExampleCountPerLeaf", 10),
                learningRate: spec.Double("learningRate", 0.2)),

            "algo.reg.poisson" => ml.Regression.Trainers.LbfgsPoissonRegression(
                labelColumnName: Label, featureColumnName: Features,
                l1Regularization: spec.Float("l1Regularization", 0.01f),
                l2Regularization: spec.Float("l2Regularization", 0.01f)),

            "algo.reg.sgd" => ml.Regression.Trainers.OnlineGradientDescent(
                labelColumnName: Label, featureColumnName: Features,
                learningRate: spec.Float("learningRate", 0.1f),
                numberOfIterations: spec.Int("numberOfIterations", 20)),

            // ------------------------------------------------------------- Clustering
            "algo.cluster.kmeans" => ml.Clustering.Trainers.KMeans(
                featureColumnName: Features,
                numberOfClusters: spec.Int("numberOfClusters", 3)),

            // -------------------------------------------------------- Anomaly detection
            "algo.anomaly.pca" => ml.AnomalyDetection.Trainers.RandomizedPca(
                featureColumnName: Features,
                rank: spec.Int("rank", 5),
                oversampling: spec.Int("oversampling", 20)),

            _ => throw new NotSupportedException(
                $"'{spec.DisplayName}' is not trained through the standard pipeline. " +
                "Matrix factorization, forecasting and spike detection have their own paths.")
        };
    }

    /// <summary>
    /// True for algorithms the standard featurise-and-fit path does not handle: they need their
    /// own column wiring rather than a single Features vector.
    /// </summary>
    public static bool NeedsSpecialPath(string moduleId) => moduleId is
        "algo.rec.matrixFactorization" or "algo.forecast.ssa" or "algo.anomaly.spike"
        or "algo.vision.imageClassification" or "algo.nlp.textClassification";
}
