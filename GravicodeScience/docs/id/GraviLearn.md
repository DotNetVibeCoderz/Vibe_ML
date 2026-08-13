# GraviLearn

*[English](../GraviLearn.md)* · scikit-learn untuk .NET — prapemrosesan, model, pipeline, dan metrik.

## Konvensi

- **Fitur** `X`: sebuah `NdArray` berbentuk *(sampel, fitur)*.
- **Target** `y`: sebuah `NdArray` sepanjang *sampel*.
- `ITransformer` punya `Fit` / `Transform`; `IEstimator` punya `Fit` / `Predict`; `IClassifier`
  menambahkan `PredictProbabilities` dan `Classes`.
- Memanggil `Predict` sebelum `Fit` melempar exception yang menyebut nama modelnya, bukan
  mengembalikan nilai sampah.

## Prapemrosesan

```csharp
using Gravicode.Science.GraviLearn.Preprocessing;

new StandardScaler().FitTransform(x);            // rata-rata nol, varians satu
new MinMaxScaler(0, 1).FitTransform(x);
new RobustScaler().FitTransform(x);              // median dan IQR — hampir tak terpengaruh outlier
new Normalizer(p: 2).FitTransform(x);            // baris bernorma satu, untuk model kosinus

new SimpleImputer(ImputationStrategy.Median).FitTransform(x);
new OneHotEncoder(dropFirst: true).FitTransform(x);
new PolynomialFeatures(degree: 2).FitTransform(x);

var encoder = new LabelEncoder();
var codes = encoder.FitTransform(y);
encoder.InverseTransform(codes);
```

`StandardScaler` membiarkan kolom konstan bernilai nol alih-alih membagi dengan nol, sehingga satu
fitur yang merosot tidak meracuni seluruh matriks dengan NaN.

## Reduksi dimensi

```csharp
using Gravicode.Science.GraviLearn.Decomposition;

var pca = new PrincipalComponentAnalysis(components: 10);   // atau: new PCA(10)
var projected = pca.FitTransform(x);
pca.ExplainedVarianceRatio;
pca.CumulativeExplainedVariance;
pca.InverseTransform(projected);

new LinearDiscriminantAnalysis().FitTransform(x, y);        // terbimbing — memaksimalkan pemisahan kelas
new TStochasticNeighborEmbedding(components: 2, perplexity: 30).FitTransform(x);
```

PCA dihitung dari SVD data yang telah dipusatkan, bukan dekomposisi eigen matriks kovarians.
Keduanya sama dalam aritmetika eksak, tetapi membentuk `X'X` mengkuadratkan condition number,
sehingga jalur SVD menjaga komponen kecil tetap akurat pada data berkondisi buruk.

t-SNE adalah **alat visualisasi**: tidak ada `Transform` untuk data baru, jarak pada keluarannya
tidak bermakna, dan ukuran klaster tidak membawa informasi. Yang dipertahankannya adalah titik mana
dekat dengan titik mana.

## Model terbimbing

### Linear

```csharp
using Gravicode.Science.GraviLearn.Linear;

new LinearRegression().Fit(x, y);                        // OLS lewat pseudo-inverse
new RidgeRegression(alpha: 1.0).Fit(x, y);               // L2 — menyusutkan, menstabilkan fitur kolinear
new LassoRegression(alpha: 0.1).Fit(x, y);               // L1 — mendorong koefisien tepat ke nol
new LogisticRegression(learningRate: 0.1, maxIterations: 1000).Fit(x, y);
new LinearSupportVectorClassifier(c: 1.0).Fit(x, y);
```

Lasso sekaligus berfungsi sebagai seleksi fitur — periksa `ZeroCoefficients`. Regresi logistik
menangani multi-kelas dengan one-vs-rest, lalu men-softmax skor sub-modelnya saat prediksi.

### Pohon dan ensemble

```csharp
using Gravicode.Science.GraviLearn.Trees;

new DecisionTree(SplitCriterion.Gini, maxDepth: 5).Fit(x, y);
new DecisionTreeRegressor(maxDepth: 5).Fit(x, y);

var forest = new RandomForestClassifier(nTrees: 100, maxDepth: 0, seed: 42);
forest.Fit(x, y);
forest.OutOfBagScore;        // estimasi akurasi jujur tanpa data uji terpisah
forest.FeatureImportances;

new RandomForestRegressor(nTrees: 100).Fit(x, y);
new GradientBoostingRegressor(nTrees: 100, learningRate: 0.1, maxDepth: 3).Fit(x, y);
new GradientBoostingClassifier(nTrees: 100).Fit(x, y);    // biner
```

Forest dan boosting memecahkan masalah berbeda: merata-ratakan pohon dalam yang terdekorelasi
menekan **varians**; menambah pohon dangkal yang masing-masing menyesuaikan sisa sebelumnya menekan
**bias**. Dua sumber keacakan pada forest juga berbeda perannya — bootstrap baris mendekorelasi
datanya, pembatasan fitur per split mendekorelasi strukturnya.

`tree.Render(featureNames)` mencetak pohon sebagai teks berindentasi, yang sering sudah cukup untuk
menjelaskan model kepada pemangku kepentingan.

### Tetangga dan Bayes

```csharp
using Gravicode.Science.GraviLearn.Neighbors;

new KNearestNeighborsClassifier(k: 5, DistanceMetric.Euclidean, distanceWeighted: true).Fit(x, y);
new KNearestNeighborsRegressor(k: 5).Fit(x, y);
new GaussianNaiveBayes().Fit(x, y);
new MultinomialNaiveBayes(alpha: 1.0).Fit(counts, y);    // fitur teks / hitungan
```

kNN **mewajibkan fitur diskalakan** — jaraknya akan didominasi kolom dengan satuan terbesar. Ia
juga dilatih seketika dan memprediksi dengan lambat, karena setiap kueri mengukur jarak ke setiap
titik latih.

## Tak terbimbing

```csharp
using Gravicode.Science.GraviLearn.Clustering;

var kmeans = new KMeans(clusters: 3, restarts: 10, seed: 42);
kmeans.FitPredict(x);
kmeans.Centroids;  kmeans.Inertia;
KMeans.ElbowCurve(x, maxK: 10);          // inertia per k, untuk memilih k

new Dbscan(epsilon: 0.5, minSamples: 5).FitPredict(x);   // label -1 berarti derau
new AgglomerativeClustering(clusters: 3, Linkage.Average).FitPredict(x);

var gmm = new GaussianMixture(components: 3, seed: 42);
gmm.FitPredict(x);
gmm.Converged;  gmm.LogLikelihood;  gmm.Bic(x);          // BIC memilih jumlah komponen
```

k-means++ menaburkan centroid berjauhan dengan peluang sebanding kuadrat jarak, itulah sebabnya ia
mengungguli penaburan acak; `restarts` menutup sisanya. DBSCAN menemukan sendiri jumlah klaster dan
punya label outlier eksplisit, tetapi `Epsilon` adalah jarak — **skalakan fitur terlebih dahulu**.

## Metrik

```csharp
Metrics.Accuracy(yTrue, yPredicted);
Metrics.Precision(yTrue, yPredicted, positiveLabel: 1);
Metrics.Recall(...);  Metrics.F1Score(...);  Metrics.Specificity(...);
Metrics.MatthewsCorrelation(...);                       // seimbang bahkan pada kelas timpang
Metrics.F1Average(yTrue, yPredicted, "weighted");       // macro | weighted | micro

Metrics.ConfusionMatrix(yTrue, yPredicted);
Metrics.FormatConfusionMatrix(matrix, classes);
Metrics.ClassificationReport(yTrue, yPredicted, labelNames);

Metrics.RocAucScore(yTrue, scores);                     // berbasis peringkat, eksak, menangani seri
Metrics.RocCurve(yTrue, scores);
Metrics.LogLoss(yTrue, probabilities);

Metrics.MeanSquaredError(...);  Metrics.RootMeanSquaredError(...);
Metrics.MeanAbsoluteError(...); Metrics.R2Score(...);   Metrics.AdjustedR2Score(..., featureCount);
Metrics.RegressionReport(yTrue, yPredicted);

Metrics.SilhouetteScore(x, labels);                     // mutu klaster
```

## Pipeline

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

Alasan memakai pipeline alih-alih memanggil langkahnya satu per satu adalah **kebocoran data**. Di
dalam cross-validation seluruh pipeline dilatih ulang per fold, sehingga scaler mempelajari
rata-ratanya hanya dari fold tersebut. Menskalakan dulu lalu membagi kemudian membocorkan statistik
data uji ke pelatihan dan menghasilkan skor yang tidak bertahan pada data baru.

Hanya langkah terakhir yang boleh berupa estimator; selain itu akan melempar exception.

## Pemilihan model

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

`stratify` penting setiap kali ada kelas yang langka: pembagian tanpa stratifikasi bisa membuat
sebuah kelas sama sekali absen dari pelatihan, yang membuat skornya tak bermakna, bukan sekadar
berisik.

Factory harus mengembalikan model **baru dan belum dilatih** setiap kali dipanggil — memakai ulang
satu instance akan membawa hasil pelatihan fold sebelumnya.

## Dataset

```csharp
Datasets.LoadIris();       // 150 x 4, 3 kelas
Datasets.LoadTitanic();    // 891 x 7, sudah dikodekan dan diisi
Datasets.LoadDigits();     // 1797 x 64, 10 kelas

Datasets.MakeBlobs(300, features: 2, centers: 3, spread: 1.0, seed: 42);
Datasets.MakeMoons(200, noise: 0.1);     // tidak terpisah linear
Datasets.MakeRegression(200, features: 5, noise: 0.5);
```

## Penyimpanan model

```csharp
ModelPersistence.Save(ModelPersistence.Capture(model), "model.json");
var snapshot = ModelPersistence.Load("model.json");
```

Parameter ditulis eksplisit sebagai JSON, bukan dengan menserialisasi grafik objek, sehingga model
tersimpan dapat diperiksa dan dibandingkan, dan tidak ada deserialisasi biner yang terlibat.

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| kNN atau k-means berkinerja buruk | Fitur berbeda skala — tambahkan `StandardScaler` |
| Skor cross-validation jauh di bawah skor uji | Prapemrosesan dilatih sebelum pembagian; gunakan `Pipeline` |
| DBSCAN menaruh semuanya dalam satu klaster atau semua derau | `Epsilon` tidak sesuai skala data |
| Kelas langka bernilai nol | Gunakan `stratify: true` saat membagi |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
