# GraviLearn

*[Bahasa Indonesia](id/GraviLearn.md)* · scikit-learn for .NET — preprocessing, models, pipelines and metrics.

## Conventions

- **Features** `X`: an `NdArray` of shape *(samples, features)*.
- **Target** `y`: an `NdArray` of length *samples*.
- `ITransformer` has `Fit` / `Transform`; `IEstimator` has `Fit` / `Predict`; `IClassifier` adds
  `PredictProbabilities` and `Classes`.
- Calling `Predict` before `Fit` throws with a message naming the model, rather than returning
  garbage.

## Preprocessing

```csharp
using Gravicode.Science.GraviLearn.Preprocessing;

new StandardScaler().FitTransform(x);            // zero mean, unit variance
new MinMaxScaler(0, 1).FitTransform(x);
new RobustScaler().FitTransform(x);              // median and IQR — outliers barely move it
new Normalizer(p: 2).FitTransform(x);            // unit-norm rows, for cosine models

new SimpleImputer(ImputationStrategy.Median).FitTransform(x);
new OneHotEncoder(dropFirst: true).FitTransform(x);
new PolynomialFeatures(degree: 2).FitTransform(x);

var encoder = new LabelEncoder();
var codes = encoder.FitTransform(y);
encoder.InverseTransform(codes);
```

`StandardScaler` leaves a constant column at zero rather than dividing by zero, so a degenerate
feature cannot poison the whole matrix with NaNs.

## Dimensionality reduction

```csharp
using Gravicode.Science.GraviLearn.Decomposition;

var pca = new PrincipalComponentAnalysis(components: 10);   // or: new PCA(10)
var projected = pca.FitTransform(x);
pca.ExplainedVarianceRatio;
pca.CumulativeExplainedVariance;
pca.InverseTransform(projected);

new LinearDiscriminantAnalysis().FitTransform(x, y);        // supervised — maximises class separation
new TStochasticNeighborEmbedding(components: 2, perplexity: 30).FitTransform(x);
```

PCA comes from the SVD of the centred data rather than an eigen decomposition of the covariance
matrix. Both agree in exact arithmetic, but forming `X'X` squares the condition number, so the SVD
route keeps small components accurate on ill-conditioned data.

t-SNE is a **visualisation tool**: no out-of-sample `Transform`, distances in the output are not
meaningful, and cluster sizes carry no information. What it preserves is which points are near
which.

## Supervised models

### Linear

```csharp
using Gravicode.Science.GraviLearn.Linear;

new LinearRegression().Fit(x, y);                        // OLS via pseudo-inverse
new RidgeRegression(alpha: 1.0).Fit(x, y);               // L2 — shrinks, stabilises collinear features
new LassoRegression(alpha: 0.1).Fit(x, y);               // L1 — drives coefficients to exactly zero
new LogisticRegression(learningRate: 0.1, maxIterations: 1000).Fit(x, y);
new LinearSupportVectorClassifier(c: 1.0).Fit(x, y);
```

Lasso doubles as feature selection — check `ZeroCoefficients`. Logistic regression handles
multi-class by one-vs-rest, softmaxing the sub-model scores at prediction time.

### Trees and ensembles

```csharp
using Gravicode.Science.GraviLearn.Trees;

new DecisionTree(SplitCriterion.Gini, maxDepth: 5).Fit(x, y);
new DecisionTreeRegressor(maxDepth: 5).Fit(x, y);

var forest = new RandomForestClassifier(nTrees: 100, maxDepth: 0, seed: 42);
forest.Fit(x, y);
forest.OutOfBagScore;        // honest accuracy estimate with no held-out split
forest.FeatureImportances;

new RandomForestRegressor(nTrees: 100).Fit(x, y);
new GradientBoostingRegressor(nTrees: 100, learningRate: 0.1, maxDepth: 3).Fit(x, y);
new GradientBoostingClassifier(nTrees: 100).Fit(x, y);    // binary
```

A forest and a boosting ensemble solve different problems: averaging decorrelated deep trees cuts
**variance**; adding shallow trees that each fit the previous residual cuts **bias**. The two
randomness sources in a forest also do different jobs — bootstrapping rows decorrelates the data,
restricting features per split decorrelates the structure.

`tree.Render(featureNames)` prints the tree as indented text, which is often enough to explain a
model to a stakeholder.

### Neighbours and Bayes

```csharp
using Gravicode.Science.GraviLearn.Neighbors;

new KNearestNeighborsClassifier(k: 5, DistanceMetric.Euclidean, distanceWeighted: true).Fit(x, y);
new KNearestNeighborsRegressor(k: 5).Fit(x, y);
new GaussianNaiveBayes().Fit(x, y);
new MultinomialNaiveBayes(alpha: 1.0).Fit(counts, y);    // text / count features
```

kNN **requires scaled features** — the distance is otherwise dominated by whichever column has the
largest units. It also trains instantly and predicts slowly, since every query measures a distance
to every training point.

## Unsupervised

```csharp
using Gravicode.Science.GraviLearn.Clustering;

var kmeans = new KMeans(clusters: 3, restarts: 10, seed: 42);
kmeans.FitPredict(x);
kmeans.Centroids;  kmeans.Inertia;
KMeans.ElbowCurve(x, maxK: 10);          // inertia per k, for choosing k

new Dbscan(epsilon: 0.5, minSamples: 5).FitPredict(x);   // label -1 means noise
new AgglomerativeClustering(clusters: 3, Linkage.Average).FitPredict(x);

var gmm = new GaussianMixture(components: 3, seed: 42);
gmm.FitPredict(x);
gmm.Converged;  gmm.LogLikelihood;  gmm.Bic(x);          // BIC picks the component count
```

k-means++ seeds centroids far apart with probability proportional to squared distance, which is
why it beats random seeding; `restarts` covers the rest. DBSCAN finds the cluster count itself and
has an explicit outlier label, but `Epsilon` is a distance — **scale the features first**.

## Metrics

```csharp
Metrics.Accuracy(yTrue, yPredicted);
Metrics.Precision(yTrue, yPredicted, positiveLabel: 1);
Metrics.Recall(...);  Metrics.F1Score(...);  Metrics.Specificity(...);
Metrics.MatthewsCorrelation(...);                       // balanced even on skewed classes
Metrics.F1Average(yTrue, yPredicted, "weighted");       // macro | weighted | micro

Metrics.ConfusionMatrix(yTrue, yPredicted);
Metrics.FormatConfusionMatrix(matrix, classes);
Metrics.ClassificationReport(yTrue, yPredicted, labelNames);

Metrics.RocAucScore(yTrue, scores);                     // rank-based, exact, handles ties
Metrics.RocCurve(yTrue, scores);
Metrics.LogLoss(yTrue, probabilities);

Metrics.MeanSquaredError(...);  Metrics.RootMeanSquaredError(...);
Metrics.MeanAbsoluteError(...); Metrics.R2Score(...);   Metrics.AdjustedR2Score(..., featureCount);
Metrics.RegressionReport(yTrue, yPredicted);

Metrics.SilhouetteScore(x, labels);                     // clustering quality
```

## Pipelines

```csharp
var pipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 10))
    .Add(new RandomForestClassifier(nTrees: 100));

pipeline.Fit(trainX, trainY);
pipeline.Predict(testX);
pipeline.PredictProbabilities(testX);
pipeline.Score(testX, testY);
```

The reason to use a pipeline rather than calling the steps by hand is **leakage**. Inside
cross-validation the whole pipeline is refitted per fold, so the scaler learns its mean from that
fold alone. Scaling first and splitting afterwards leaks test statistics into training and
produces scores that do not survive contact with new data.

Only the last step may be an estimator; anything else throws.

## Model selection

```csharp
using Gravicode.Science.GraviLearn.ModelSelection;

var split = Selection.Split(x, y, testSize: 0.25, seed: 42, stratify: true);
Selection.KFold(samples, folds: 5);
Selection.StratifiedKFold(y, folds: 5);

Selection.CrossValidate(() => new RandomForestClassifier(nTrees: 100), x, y, folds: 5, stratified: true);
Selection.CrossValidatePipeline(() => BuildPipeline(), x, y, folds: 5);

var search = new GridSearch(p => new RandomForestClassifier(
        nTrees: (int)p["nTrees"], maxDepth: (int)p["maxDepth"]), folds: 5)
    .AddParameter("nTrees", 50, 100, 200)
    .AddParameter("maxDepth", 3, 5, 0);

search.Fit(x, y);
search.Best;  search.BestModel;  search.Report(10);
```

`stratify` matters whenever a class is rare: an unstratified split can leave a class out of
training entirely, which makes the resulting score meaningless rather than merely noisy.

The factory must return a **fresh, unfitted** model each call — reusing one instance would carry
the previous fold's fit into the next.

## Datasets

```csharp
Datasets.LoadIris();       // 150 x 4, 3 classes
Datasets.LoadTitanic();    // 891 x 7, encoded and imputed
Datasets.LoadDigits();     // 1797 x 64, 10 classes

Datasets.MakeBlobs(300, features: 2, centers: 3, spread: 1.0, seed: 42);
Datasets.MakeMoons(200, noise: 0.1);     // not linearly separable
Datasets.MakeRegression(200, features: 5, noise: 0.5);
```

## Persistence

```csharp
ModelPersistence.Save(ModelPersistence.Capture(model), "model.json");
var snapshot = ModelPersistence.Load("model.json");
```

Parameters are written explicitly as JSON rather than by serialising the object graph, so a saved
model can be inspected and diffed, and no binary deserialisation is involved.

### Exporting to ONNX

For serving a model outside .NET, a fitted pipeline can be written as ONNX:

```csharp
OnnxExport.Supports(pipeline);          // check before trying
OnnxExport.Save(pipeline, "model.onnx", features: 4);
```

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
session.run(None, {"input": x.astype("float32")})
```

**What exports.** Scalers, PCA and linear models — every step that is an *affine map*, which is why
the whole pipeline collapses into a handful of core ONNX operators (`Sub`, `Div`, `MatMul`, `Add`,
`ArgMax`). Core operators are used rather than the `ai.onnx.ml` set because every runtime
implements them.

**What does not.** A decision tree, a forest or a k-nearest-neighbour model is not an affine map,
and `Save` throws rather than emitting an approximation. A model that loads cleanly and predicts
wrongly is worse than one that refuses to export.

Two details that are easy to get backwards and produce a valid-looking, wrong model:

- The batch dimension is written **symbolically**, so the exported model accepts any number of
  rows. Pinning it to the training set's row count is a common export bug and makes the model
  useless for the single row a serving endpoint actually sends.
- PCA components are stored **transposed**, because the graph multiplies rows of `x` by them.

Everything is written as `float32`: ONNX runtimes support it universally and `float64` only
patchily. Classifications agree exactly; regressions agree to about **1e-6** relative, which is
single precision doing its job.

> Verified against the real tooling rather than against this library's own reader: `onnx.checker`
> confirms the graph is valid, and Python's **onnxruntime** reproduces the .NET predictions —
> 150/150 labels on an Iris pipeline, and 5.9e-07 maximum difference on a regression one.

## Common mistakes

| Symptom | Cause |
|---|---|
| kNN or k-means performs badly | Features are on different scales — add a `StandardScaler` |
| Cross-validated score far below the test score | Preprocessing was fitted before splitting; use a `Pipeline` |
| DBSCAN puts everything in one cluster or all noise | `Epsilon` does not suit the data's scale |
| A rare class scores zero | Use `stratify: true` when splitting |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
