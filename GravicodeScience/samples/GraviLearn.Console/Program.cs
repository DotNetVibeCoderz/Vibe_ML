using System.Diagnostics;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;

Console.WriteLine(GraviInfo.Banner("GraviLearn"));
var screenshots = ResolveScreenshotDirectory();

// ---------------------------------------------------------------- data
Section("1. The Iris dataset");

var iris = Datasets.LoadIris();
Console.WriteLine($"  {iris.SampleCount} samples, {iris.FeatureCount} features, {iris.TargetNames.Count} classes");
Console.WriteLine($"  features: {string.Join(", ", iris.FeatureNames)}");
Console.WriteLine($"  classes : {string.Join(", ", iris.TargetNames)}");

var split = Selection.Split(iris.Features, iris.Target, testSize: 0.3, seed: 42, stratify: true);
Console.WriteLine($"  stratified split: {split.TrainX.Shape[0]} train / {split.TestX.Shape[0]} test");
Console.WriteLine();

// ---------------------------------------------------------------- random forest
Section("2. Training a random forest");

var watch = Stopwatch.StartNew();
var forest = new RandomForestClassifier(nTrees: 200, maxDepth: 0, seed: 42);
forest.Fit(split.TrainX, split.TrainY);
watch.Stop();

Console.WriteLine($"  trained {forest.TreeCount} trees in {watch.ElapsedMilliseconds} ms");
Console.WriteLine($"  training accuracy    : {forest.Score(split.TrainX, split.TrainY):P2}");
Console.WriteLine($"  test accuracy        : {forest.Score(split.TestX, split.TestY):P2}");
Console.WriteLine($"  out-of-bag accuracy  : {forest.OutOfBagScore:P2}  (no held-out data needed)");
Console.WriteLine();

Console.WriteLine("  Feature importances:");
var importances = forest.FeatureImportances;
foreach (var (name, importance) in iris.FeatureNames.Zip(importances.ToArray()).OrderByDescending(t => t.Second))
    Console.WriteLine($"    {name,-16}{importance,8:P2}  {new string('#', (int)(importance * 40))}");
Console.WriteLine();

// ---------------------------------------------------------------- evaluation
Section("3. Evaluation");

var predictions = forest.Predict(split.TestX);
Console.WriteLine(Metrics.ClassificationReport(split.TestY, predictions, iris.LabelNames));
Console.WriteLine();

var confusion = Metrics.ConfusionMatrix(split.TestY, predictions);
Console.WriteLine("  Confusion matrix:");
Console.WriteLine(Metrics.FormatConfusionMatrix(confusion, forest.Classes));

// ---------------------------------------------------------------- model comparison
Section("4. Comparing models with cross-validation");

var candidates = new (string Name, Func<IEstimator> Factory)[]
{
    ("LogisticRegression", () => new LogisticRegression(learningRate: 0.3, maxIterations: 1500)),
    ("DecisionTree", () => new DecisionTree(maxDepth: 4)),
    ("RandomForest", () => new RandomForestClassifier(nTrees: 100, seed: 42)),
    ("GaussianNaiveBayes", () => new GaussianNaiveBayes()),
    ("kNN (k=5)", () => new KNearestNeighborsClassifier(k: 5)),
    ("LinearSVM", () => new LinearSupportVectorClassifier(c: 1.0, maxIterations: 300)),
};

Console.WriteLine($"  {"model",-22}{"5-fold accuracy",20}{"fit time",12}");
Console.WriteLine("  " + new string('-', 54));

foreach (var (name, factory) in candidates)
{
    watch.Restart();
    var result = Selection.CrossValidate(factory, iris.Features, iris.Target, folds: 5, stratified: true, seed: 42);
    watch.Stop();
    Console.WriteLine($"  {name,-22}{result.Mean,12:P2} +/- {result.StandardDeviation,-5:P1}{watch.ElapsedMilliseconds,8} ms");
}
Console.WriteLine();

// ---------------------------------------------------------------- pipeline
Section("5. Pipelines");

var pipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 2))
    .Add(new RandomForestClassifier(nTrees: 100, seed: 42));

pipeline.Fit(split.TrainX, split.TrainY);
Console.WriteLine($"  {pipeline}");
Console.WriteLine($"  test accuracy after reducing 4 features to 2: {pipeline.Score(split.TestX, split.TestY):P2}");

var cv = Selection.CrossValidatePipeline(
    () => new Pipeline()
        .Add(new StandardScaler())
        .Add(new PCA(components: 2))
        .Add(new RandomForestClassifier(nTrees: 100, seed: 42)),
    iris.Features, iris.Target, folds: 5, stratified: true, seed: 42);

Console.WriteLine($"  cross-validated (every step refitted per fold): {cv}");
Console.WriteLine();

// ---------------------------------------------------------------- grid search
Section("6. Hyper-parameter search");

var search = new GridSearch(p => new RandomForestClassifier(
        nTrees: (int)p["nTrees"], maxDepth: (int)p["maxDepth"], seed: 42),
    folds: 5, seed: 42)
    .AddParameter("nTrees", 20, 100)
    .AddParameter("maxDepth", 2, 4, 0);

watch.Restart();
search.Fit(iris.Features, iris.Target);
watch.Stop();

Console.WriteLine($"  evaluated {search.Results.Count} combinations in {watch.ElapsedMilliseconds} ms");
Console.WriteLine(search.Report(6));
Console.WriteLine();

// ---------------------------------------------------------------- dimensionality reduction
Section("7. Principal components");

var pca = new PrincipalComponentAnalysis(4);
var projected = pca.FitTransform(iris.Features);

for (var c = 0; c < 4; c++)
    Console.WriteLine($"  PC{c + 1}: {pca.ExplainedVarianceRatio.At(c),8:P2} of variance " +
                      $"(cumulative {pca.CumulativeExplainedVariance.At(c):P2})");
Console.WriteLine();

// ---------------------------------------------------------------- clustering
Section("8. Unsupervised clustering");

var scaled = new StandardScaler().FitTransform(iris.Features);

Console.WriteLine("  Elbow curve for k-means:");
foreach (var (k, inertia) in KMeans.ElbowCurve(scaled, maxK: 6, seed: 42))
    Console.WriteLine($"    k={k}  inertia {inertia,9:F2}  {new string('#', (int)(inertia / 12))}");

var kmeans = new KMeans(clusters: 3, seed: 42);
var clusters = kmeans.FitPredict(scaled);
Console.WriteLine($"  k=3 silhouette: {Metrics.SilhouetteScore(scaled, clusters):F4}");

var mixture = new GaussianMixture(components: 3, seed: 42);
mixture.Fit(scaled);
Console.WriteLine($"  Gaussian mixture: converged={mixture.Converged}, log-likelihood={mixture.LogLikelihood:F4}, BIC={mixture.Bic(scaled):F1}");

var dbscan = new Dbscan(epsilon: 0.8, minSamples: 5);
dbscan.Fit(scaled);
Console.WriteLine($"  DBSCAN: {dbscan.ClusterCount} clusters, {dbscan.NoiseCount} points labelled noise");
Console.WriteLine();

// ---------------------------------------------------------------- regression
Section("9. Regression on the digits dataset");

var digits = Datasets.LoadDigits();
Console.WriteLine($"  {digits.SampleCount} images of {digits.FeatureCount} pixels");

var digitSplit = Selection.Split(digits.Features, digits.Target, testSize: 0.3, seed: 42, stratify: true);
var digitPipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 30))
    .Add(new KNearestNeighborsClassifier(k: 3));

watch.Restart();
digitPipeline.Fit(digitSplit.TrainX, digitSplit.TrainY);
var digitAccuracy = digitPipeline.Score(digitSplit.TestX, digitSplit.TestY);
watch.Stop();

Console.WriteLine($"  scaler -> PCA(30) -> kNN(3): {digitAccuracy:P2} test accuracy in {watch.ElapsedMilliseconds} ms");
Console.WriteLine();

// ---------------------------------------------------------------- chart
Section("10. Confusion matrix image");

var digitPredictions = digitPipeline.Predict(digitSplit.TestX);
var digitConfusion = Metrics.ConfusionMatrix(digitSplit.TestY, digitPredictions);

var confusionPath = Path.Combine(screenshots, "gravilearn_confusion.png");
var plot = new ScottPlot.Plot();
var heatmap = plot.Add.Heatmap(digitConfusion.To2DArray());
heatmap.Colormap = new ScottPlot.Colormaps.Viridis();
plot.Add.ColorBar(heatmap);
plot.Title($"GraviLearn - digits confusion matrix ({digitAccuracy:P1} accuracy)");
plot.XLabel("predicted digit");
plot.YLabel("true digit");
plot.SavePng(confusionPath, 900, 750);
Console.WriteLine($"  saved {confusionPath}");

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string ResolveScreenshotDirectory()
{
    var directory = new DirectoryInfo(AppContext.BaseDirectory);
    for (var depth = 0; directory is not null && depth < 12; depth++)
    {
        var candidate = Path.Combine(directory.FullName, "docs", "screenshots");
        if (Directory.Exists(candidate)) return candidate;
        directory = directory.Parent;
    }
    var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
    Directory.CreateDirectory(fallback);
    return fallback;
}
