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

## Explaining a model

`PermutationImportance` scores each feature by how much accuracy is lost when its column is shuffled.
Shuffling keeps the column's distribution and destroys its relationship with the target, so the drop
is what that relationship was worth.

```csharp
var ranked = PermutationImportance.Ranked(model, xTest, yTest, repeats: 10);
// feature 0: 0.3125 ± 0.0142
```

**Run it on held-out data.** On the training set it measures what the model memorised rather than
what generalises, and an overfitted model reports every feature as vital. It also measures something
different from a tree's built-in importances, which describe how the tree was *built* and are known
to favour high-cardinality features regardless of whether they predict anything.

Correlated features share the blame and each looks unimportant: shuffling one leaves the other
carrying the same information. That is a real limitation of the method, not a bug — the honest
reading is "these two together matter", and the reported standard deviation is what warns you the
estimate is unstable.

`ShapleyValues` answers a different question: not "which features does this model rely on" but "why
did it say *that*, for *this* row". The two routinely disagree — a feature can be globally
unimportant and decisive for one applicant.

```csharp
var attribution = ShapleyValues.Sample(predict, instance, background, samples: 200);
attribution.BaseValue;        // the model's average over the background
attribution.Contributions;    // they sum to the prediction minus the base
attribution.Ranked;           // largest influence first, in either direction
```

The Shapley value comes from cooperative game theory: add features one at a time in a random order,
record how much each moves the prediction when it joins, average over every order. It is the unique
attribution satisfying efficiency (the parts sum to the whole), symmetry (equal contributors get
equal credit) and the dummy property (a feature that changes nothing gets zero). No cheaper heuristic
has all three.

"Absent" means replaced by a value from the background dataset, so **the background is part of the
explanation.** Explaining a loan refusal against a background of approved applicants answers a
different question from explaining it against all applicants, and reading any attribution honestly
requires knowing which was used.

`Exact` enumerates all 2ⁿ subsets and is refused above twenty features; `Sample` is Monte Carlo over
permutations and is what to use above about fifteen. Because each permutation telescopes, `Sample`
satisfies efficiency *exactly* at any sample size — only the split between features is approximate.

## Calibration

A model can rank perfectly and still be badly calibrated. If everything it calls "90% likely" happens
60% of the time, its ordering is fine and its numbers are not — and any decision made on a threshold,
an expected value or a cost trade-off is then wrong. Accuracy and AUC cannot see this.

```csharp
Calibration.Curve(probabilities, labels, bins: 10);   // a reliability diagram
Calibration.ExpectedError(probabilities, labels);     // ECE: the average claimed-vs-observed gap
Calibration.BrierScore(probabilities, labels);        // mean squared error of the probabilities
```

Empty bins are dropped rather than reported as zero, which would draw a curve through a region where
there is no evidence at all. The expected calibration error is weighted by bin population, so a wild
miss in a bin holding three samples does not outweigh a small one in a bin holding a thousand.

`IsotonicRegression` is the standard fix. It fits the best non-decreasing step function by
pool-adjacent-violators: walk left to right, and whenever a block's mean falls below its
predecessor's, merge them and re-average.

```csharp
var calibrated = new IsotonicRegression().Fit(scores, labels).Predict(scores);
```

The merge repeats *backwards*, because merging can create a new violation with the block before —
that inner loop is the whole algorithm, and leaving it out gives something that looks right on smooth
data and fails on the ragged data this is for. Because the fit is monotone, a classifier's ranking is
preserved and only its numbers change, which is exactly what recalibration should do.

## Imbalanced data

On a dataset that is 99% negative, the cheapest way to minimise total error is to predict "negative"
for everything. That model scores 99% accuracy and is worthless.

```csharp
Resampler.OverSample(x, y);            // duplicate the minority up to the majority
Resampler.UnderSample(x, y);           // drop the majority down to the minority
Resampler.Smote(x, y, neighbours: 5);  // synthesise minority rows by interpolation
Resampler.ClassWeights(y);             // the same effect without touching the data
Resampler.ClassBalance(y);
```

**Resample the training split only.** Rebalancing before splitting puts synthetic points — or
duplicates of real ones — on both sides of the split, so the test set contains rows derived from
training rows and the score comes back optimistic. This is the single most common way to get an
unreproducible result out of an imbalanced problem.

None of it is free. Oversampling makes the minority class look denser than it is; undersampling
throws away real data. SMOTE places new points between real neighbours so the classifier sees a
region rather than a set of dots — but it assumes the segment between two same-class neighbours is
also that class, and where the minority class is not convex that is false. **Scale features first:**
neighbours are found by Euclidean distance, so an unscaled column in the thousands decides every
neighbourhood on its own.

`ClassWeights` is often the better tool where a learner accepts sample weights: it discards nothing,
invents nothing, and adds no rows to train on.

## One-class SVM

Novelty detection is not classification with one class missing. There are no negative examples to
learn a boundary *between*; the task is to find a region containing most of the training data and as
little else as possible.

```csharp
var detector = new OneClassSvm(nu: 0.05).Fit(normalData);

detector.Predict(x);            // 1 for normal, -1 for an anomaly
detector.DecisionFunction(x);   // signed distance — use this for ranking alerts
```

**`nu` is the dial that matters.** It is simultaneously an upper bound on the fraction of training
points outside the boundary and a lower bound on the fraction that become support vectors. So
`nu: 0.05` is a statement that around 5% of the training data is contamination worth excluding — not
a tolerance to be tuned until the answer looks right. Note that `nu` of 0 has no solution, so even
genuinely clean data needs a small positive value.

Scale the features first. An RBF kernel is a function of Euclidean distance, so a column measured in
thousands determines every similarity on its own and `gamma` becomes meaningless.

**Do not judge the model by scoring its own training data.** A support vector appears in its own
decision function, so an isolated training point gets credit for being near itself and always scores
higher than an identical point held out. When `rho` falls below the box bound it lands exactly *on*
the boundary, and which side of zero it reports is then decided by floating-point noise. Evaluate on
data the model has not seen.

The convergence tolerance is visible in the results, not just the runtime: at 1e-3 the nu-property
stops holding, which is why the default is 1e-6.

## HDBSCAN

DBSCAN asks for one `eps` and applies it everywhere. That is fine when every cluster has the same
density and hopeless when they do not: an `eps` tight enough to keep two dense clusters apart shreds
a sparse one into noise, and an `eps` loose enough to hold the sparse one together merges the dense
pair. There is no value that works.

```csharp
var model = new Hdbscan(minClusterSize: 10).Fit(x);

model.Labels;           // -1 is noise
model.ClusterCount;
model.Probabilities;    // membership strength in [0, 1]; zero for noise
model.CoreDistances;    // the local density estimate everything is built on
```

The route around it is to run DBSCAN at *every* threshold at once and ask which clusters survived
longest: compute each point's core distance, define mutual reachability as
`max(core(a), core(b), d(a,b))`, build the minimum spanning tree of that metric, condense away splits
that shed fewer than `minClusterSize` points, and select the non-overlapping clusters maximising
stability.

The result is that `minClusterSize` — "how many points make a cluster worth the name" — is the only
parameter that really needs an answer, and unlike `eps` it is a question about the problem rather
than about the data's scale.

`Probabilities` is often the more useful output than the flat labels: a point at 0.05 is nominally
clustered and practically indistinguishable from noise.

Cost is O(n²) in memory and time — the pairwise distance matrix is materialised. Real implementations
use a space tree, which pays off above a few thousand points and stops helping in high dimensions
anyway.

---

## Visualisations

All three are rendered by `samples/GraviLearn.Console` and reproduced by
`notebooks/GraviLearn.Notebook.ipynb`.

![A confusion matrix](screenshots/gravilearn_confusion.png)

![A reliability diagram before and after isotonic regression](screenshots/gravilearn_calibration.png)

The reliability diagram is the clearest picture of what calibration means. The "before" curve sits
well below the diagonal — everything the model calls 60% likely happens far less often than that —
and isotonic regression pulls it onto the line without touching the ranking.

![HDBSCAN on clusters of differing density](screenshots/gravilearn_hdbscan.png)

Two tight clusters close together and one diffuse cluster far away. No single DBSCAN `eps` separates
all three; HDBSCAN needs no threshold at all.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
