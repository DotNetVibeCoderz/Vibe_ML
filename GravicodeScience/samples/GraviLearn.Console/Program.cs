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
using Gravicode.Science.GraviLearn.Explain;
using Gravicode.Science.GraviLearn.Resampling;
using Gravicode.Science.GraviLearn.Anomaly;

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

// ---------------------------------------------------------------- v0.4: explanations
Section("10. Permutation importance and Shapley values");

var irisData = Datasets.LoadIris();
var explainSplit = Selection.Split(irisData.Features, irisData.Target, testSize: 0.3, seed: 42, stratify: true);

var explainForest = new RandomForestClassifier(nTrees: 60, seed: 42);
explainForest.Fit(explainSplit.TrainX, explainSplit.TrainY);

Console.WriteLine("  Which features does the model rely on? Shuffle each and watch accuracy fall.");
Console.WriteLine("  Measured on the TEST split - on the training set this measures memorisation.");
foreach (var importance in PermutationImportance.Ranked(explainForest, explainSplit.TestX, explainSplit.TestY, repeats: 8))
    Console.WriteLine($"    {irisData.FeatureNames[importance.Feature],-22} {importance.Mean,7:F4} +/- {importance.StandardDeviation:F4}");
Console.WriteLine("  Petal measurements dominate, which is the known structure of this dataset.");
Console.WriteLine();

Console.WriteLine("  Shapley values answer a different question: why THIS row, not which features overall.");
var background = explainSplit.TrainX;
var oneFlower = explainSplit.TestX.Row(0);

// Explaining the probability rather than the hard label: a class index is a step
// function, and attributing changes in a step tells you far less than attributing
// changes in the confidence behind it.
var targetClass = (int)explainForest.Predict(Reshape(oneFlower)).At(0);

NdArray ClassProbability(NdArray batch)
{
    var probabilities = explainForest.PredictProbabilities(batch);
    var column = NdArray.Zeros(batch.Shape[0]);
    for (var i = 0; i < batch.Shape[0]; i++) column.SetAt(i, probabilities[i, targetClass]);
    return column;
}

var attribution = ShapleyValues.Sample(ClassProbability, oneFlower, background, samples: 150);

Console.WriteLine($"    explaining P(class {targetClass} = {irisData.LabelNames[targetClass]}) for one flower");
Console.WriteLine($"    base value (average over the background) = {attribution.BaseValue:F4}");
foreach (var (feature, contribution) in attribution.Ranked)
    Console.WriteLine($"    {irisData.FeatureNames[feature],-22} {contribution,+8:F4}");
Console.WriteLine($"    contributions sum to the prediction: {attribution.Prediction:F4} " +
                  $"(actual {ClassProbability(Reshape(oneFlower)).At(0):F4})");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: calibration
Section("11. Calibration");

// Well-ranked but badly scaled scores: the ordering is perfect, the numbers are not.
var calibrationRng = new GraviRandom(41);
var scores = NdArray.Zeros(4000);
var outcomes = NdArray.Zeros(4000);
for (var i = 0; i < 4000; i++)
{
    var truth = calibrationRng.NextDouble();
    scores.SetAt(i, Math.Sqrt(truth));
    outcomes.SetAt(i, calibrationRng.NextDouble() < truth ? 1 : 0);
}

var before = Calibration.ExpectedError(scores, outcomes);
var isotonic = new IsotonicRegression().Fit(scores, outcomes);
var recalibrated = isotonic.Predict(scores);
var after = Calibration.ExpectedError(recalibrated, outcomes);

Console.WriteLine("  A model can rank perfectly and still be badly calibrated. Accuracy and AUC");
Console.WriteLine("  cannot see it, because neither depends on the numbers themselves.");
Console.WriteLine($"    expected calibration error before = {before:F4}");
Console.WriteLine($"    after isotonic regression         = {after:F4}");
Console.WriteLine($"    Brier score before / after        = {Calibration.BrierScore(scores, outcomes):F4} / " +
                  $"{Calibration.BrierScore(recalibrated, outcomes):F4}");
Console.WriteLine($"    the fit is monotone, so the ranking is untouched and only the numbers move");

var reliabilityPath = Path.Combine(screenshots, "gravilearn_calibration.png");
var reliability = new ScottPlot.Plot();

var rawCurve = Calibration.Curve(scores, outcomes);
var fixedCurve = Calibration.Curve(recalibrated, outcomes);

reliability.Add.Scatter(
    rawCurve.Select(p => p.MeanPredicted).ToArray(),
    rawCurve.Select(p => p.ObservedFraction).ToArray()).LegendText = "before";
reliability.Add.Scatter(
    fixedCurve.Select(p => p.MeanPredicted).ToArray(),
    fixedCurve.Select(p => p.ObservedFraction).ToArray()).LegendText = "after isotonic";

var diagonal = reliability.Add.Scatter(new double[] { 0, 1 }, new double[] { 0, 1 });
diagonal.LegendText = "perfectly calibrated";
diagonal.LinePattern = ScottPlot.LinePattern.Dashed;

reliability.Title("GraviLearn - reliability diagram");
reliability.XLabel("mean predicted probability");
reliability.YLabel("observed frequency");
reliability.ShowLegend();
reliability.SavePng(reliabilityPath, 800, 600);
Console.WriteLine($"  saved {reliabilityPath}");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: imbalance
Section("12. Imbalanced data");

var imbalanceRng = new GraviRandom(7);
var skewedX = NdArray.Zeros(400, 2);
var skewedY = NdArray.Zeros(400);
for (var i = 0; i < 400; i++)
{
    var minority = i >= 370;
    skewedX[i, 0] = (minority ? 6 : 0) + imbalanceRng.Normal() * 0.6;
    skewedX[i, 1] = (minority ? 6 : 0) + imbalanceRng.Normal() * 0.6;
    skewedY.SetAt(i, minority ? 1 : 0);
}

Console.WriteLine("  370 negatives against 30 positives. Predicting 'negative' everywhere scores 92.5%.");
foreach (var (label, count) in Resampler.ClassBalance(skewedY))
    Console.WriteLine($"    class {label}: {count}");

var over = Resampler.OverSample(skewedX, skewedY);
var smote = Resampler.Smote(skewedX, skewedY, neighbours: 5);
Console.WriteLine($"  OverSample -> {over.X.Shape[0]} rows, SMOTE -> {smote.X.Shape[0]} rows");
Console.WriteLine("  SMOTE places new points BETWEEN real neighbours, so the classifier sees a region");
Console.WriteLine("  rather than a set of dots. Duplicates would sit exactly on top of the originals.");

var weights = Resampler.ClassWeights(skewedY);
Console.WriteLine($"  ClassWeights: class 0 = {weights[0]:F4}, class 1 = {weights[1]:F4}");
Console.WriteLine("    the same rebalancing without touching the data - preferable where the learner");
Console.WriteLine("    accepts sample weights, since it discards nothing and invents nothing");
Console.WriteLine("  Resample the TRAINING split only: doing it first puts synthetic rows on both");
Console.WriteLine("  sides of the split and the score comes back optimistic.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: anomaly
Section("13. One-class SVM");

var normalRng = new GraviRandom(11);
var normal = NdArray.Zeros(300, 2);
for (var i = 0; i < 300; i++)
{
    normal[i, 0] = normalRng.Normal();
    normal[i, 1] = normalRng.Normal();
}

var detector = new OneClassSvm(nu: 0.05).Fit(normal);
Console.WriteLine($"  nu = 0.05 says 'about 5% of this data is contamination worth excluding'.");
Console.WriteLine($"  It bounds the outlier fraction above and the support-vector fraction below.");
Console.WriteLine($"    support vectors = {detector.SupportVectorCount} of 300, gamma = {detector.Gamma:F4} (from the data)");

var probes = NdArray.FromArray(new double[,] { { 0, 0 }, { 1.5, 1.5 }, { 6, 6 }, { -8, 3 } });
var probeScores = detector.DecisionFunction(probes);
for (var i = 0; i < probes.Shape[0]; i++)
    Console.WriteLine($"    ({probes[i, 0],5:F1}, {probes[i, 1],5:F1})  score {probeScores.At(i),8:F4}  " +
                      $"{(probeScores.At(i) >= 0 ? "normal" : "ANOMALY")}");
Console.WriteLine("  The score is signed distance, so it ranks alerts; the sign alone only sorts them.");
Console.WriteLine("  The two far-away probes score identically: once every kernel term has decayed to");
Console.WriteLine("  zero the score saturates at -rho, so 'very far' and 'extremely far' look the same.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: HDBSCAN
Section("14. HDBSCAN against a DBSCAN eps sweep");

// Two tight clusters close together, one diffuse cluster far away: no single eps works.
var densityRng = new GraviRandom(5);
var varied = NdArray.Zeros(150, 2);
for (var i = 0; i < 50; i++) { varied[i, 0] = densityRng.Normal() * 0.3; varied[i, 1] = densityRng.Normal() * 0.3; }
for (var i = 50; i < 100; i++) { varied[i, 0] = 3 + densityRng.Normal() * 0.3; varied[i, 1] = densityRng.Normal() * 0.3; }
for (var i = 100; i < 150; i++) { varied[i, 0] = 25 + densityRng.Normal() * 3.0; varied[i, 1] = 25 + densityRng.Normal() * 3.0; }

Console.WriteLine("  Two tight clusters close together, one diffuse cluster far away.");
Console.WriteLine("  DBSCAN applies one eps everywhere, so it has to choose which to get wrong:");

foreach (var epsilon in new[] { 0.5, 1.0, 2.0, 4.0 })
{
    var sweep = new Dbscan(epsilon, minSamples: 5);
    sweep.Fit(varied);

    var found = new HashSet<double>();
    var noise = 0;
    for (var i = 0; i < sweep.Labels.Size; i++)
    {
        if (sweep.Labels.At(i) < 0) noise++;
        else found.Add(sweep.Labels.At(i));
    }

    Console.WriteLine($"    eps = {epsilon,-5} -> {found.Count} clusters, {noise,3} points called noise");
}

var hdbscan = new Hdbscan(minClusterSize: 10).Fit(varied);
Console.WriteLine($"  HDBSCAN, with no density threshold to pick -> {hdbscan.ClusterCount} clusters");
Console.WriteLine("  It runs DBSCAN at every threshold at once and keeps the clusters that persist longest.");

var clusterPath = Path.Combine(screenshots, "gravilearn_hdbscan.png");
var clusterPlot = new ScottPlot.Plot();
for (var label = -1; label < hdbscan.ClusterCount; label++)
{
    var xs = new List<double>();
    var ys = new List<double>();
    for (var i = 0; i < 150; i++)
        if ((int)hdbscan.Labels.At(i) == label) { xs.Add(varied[i, 0]); ys.Add(varied[i, 1]); }

    if (xs.Count == 0) continue;

    var scatter = clusterPlot.Add.Scatter(xs.ToArray(), ys.ToArray());
    scatter.LineWidth = 0;
    scatter.MarkerSize = 7;
    scatter.LegendText = label < 0 ? "noise" : $"cluster {label}";
}

clusterPlot.Title("GraviLearn - HDBSCAN on clusters of differing density");
clusterPlot.ShowLegend();
clusterPlot.SavePng(clusterPath, 800, 600);
Console.WriteLine($"  saved {clusterPath}");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static NdArray Reshape(NdArray row)
{
    var matrix = NdArray.Zeros(1, row.Size);
    for (var i = 0; i < row.Size; i++) matrix[0, i] = row.At(i);
    return matrix;
}

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
