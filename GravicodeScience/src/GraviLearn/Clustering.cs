using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Clustering;

/// <summary>
/// k-means clustering with k-means++ initialisation.
/// </summary>
/// <remarks>
/// Lloyd's algorithm only finds a local optimum, and which one depends entirely on where the
/// centroids start. k-means++ seeds them far apart with probability proportional to squared
/// distance, which is why it converges to better solutions than random seeding; running
/// <see cref="Restarts"/> independent attempts and keeping the lowest-inertia one covers the rest.
/// </remarks>
public sealed class KMeans(
    int clusters = 8,
    int maxIterations = 300,
    double tolerance = 1e-4,
    int restarts = 10,
    int seed = 42) : ModelBase, IClusterer
{
    /// <summary>Number of clusters.</summary>
    public int Clusters { get; } = clusters;

    /// <summary>Maximum Lloyd iterations per restart.</summary>
    public int MaxIterations { get; } = maxIterations;

    /// <summary>Convergence threshold on centroid movement.</summary>
    public double Tolerance { get; } = tolerance;

    /// <summary>Independent initialisations tried; the best inertia wins.</summary>
    public int Restarts { get; } = restarts;

    /// <summary>Cluster centres, one per row.</summary>
    public NdArray Centroids { get; private set; } = NdArray.Zeros(0, 0);

    /// <inheritdoc />
    public NdArray Labels { get; private set; } = NdArray.Zeros(0);

    /// <summary>Total squared distance from each point to its centroid.</summary>
    public double Inertia { get; private set; } = double.MaxValue;

    /// <summary>Iterations used by the winning restart.</summary>
    public int IterationsRun { get; private set; }

    /// <inheritdoc />
    public void Fit(NdArray x)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        if (Clusters > samples)
            throw new ArgumentException($"Cannot form {Clusters} clusters from {samples} samples.");

        Inertia = double.MaxValue;

        for (var restart = 0; restart < Restarts; restart++)
        {
            var rng = new GraviRandom(seed + restart * 104729);
            var centroids = InitializePlusPlus(x, samples, rng);
            var labels = new int[samples];
            var iterations = 0;

            // Lloyd's loop runs on flat buffers; the NdArray views are only used at the edges.
            var flatData = x.AsContiguous().ToArray();
            var flatCentres = centroids.ToArray();

            for (; iterations < MaxIterations; iterations++)
            {
                var changed = AssignPointsFast(flatData, flatCentres, samples, labels);
                var shift = UpdateCentroidsFast(flatData, flatCentres, samples, labels, rng);
                if (!changed || shift < Tolerance) { iterations++; break; }
            }

            AssignPointsFast(flatData, flatCentres, samples, labels);
            for (var c = 0; c < Clusters; c++)
                for (var j = 0; j < FeatureCount; j++)
                    centroids[c, j] = flatCentres[c * FeatureCount + j];

            var inertia = ComputeInertia(x, centroids, labels);
            if (inertia < Inertia)
            {
                Inertia = inertia;
                Centroids = centroids.Copy();
                Labels = NdArray.FromValues(labels.Select(l => (double)l));
                IterationsRun = iterations;
            }
        }

        IsFitted = true;
    }

    private NdArray InitializePlusPlus(NdArray x, int samples, GraviRandom rng)
    {
        var centroids = NdArray.Zeros(Clusters, FeatureCount);
        var first = rng.Next(samples);
        for (var j = 0; j < FeatureCount; j++) centroids[0, j] = x[first, j];

        var squaredDistances = new double[samples];
        for (var i = 0; i < samples; i++)
            squaredDistances[i] = SquaredDistance(x, i, centroids, 0);

        for (var c = 1; c < Clusters; c++)
        {
            // Sample the next centre with probability proportional to its squared distance
            // from the nearest existing centre.
            var picked = rng.Categorical(squaredDistances);
            for (var j = 0; j < FeatureCount; j++) centroids[c, j] = x[picked, j];

            for (var i = 0; i < samples; i++)
                squaredDistances[i] = Math.Min(squaredDistances[i], SquaredDistance(x, i, centroids, c));
        }
        return centroids;
    }

    /// <summary>
    /// Assigns every point to its nearest centroid, working on the raw buffers.
    /// </summary>
    /// <remarks>
    /// The assignment step is the whole cost of k-means: it is O(samples x clusters x features)
    /// on every iteration, of every restart. Going through <see cref="NdArray"/>'s two-index
    /// accessor there pays stride arithmetic and a bounds check per feature, which on a
    /// 20,000 x 20 problem is tens of millions of redundant operations per iteration. Reading the
    /// contiguous buffers directly is what closes most of the gap to a BLAS-backed implementation.
    /// </remarks>
    private bool AssignPointsFast(double[] data, double[] centres, int samples, int[] labels)
    {
        var changed = false;
        var features = FeatureCount;
        var clusters = Clusters;

        for (var i = 0; i < samples; i++)
        {
            var rowOffset = i * features;
            var best = 0;
            var bestDistance = double.MaxValue;

            for (var c = 0; c < clusters; c++)
            {
                var centreOffset = c * features;
                var accumulator = 0.0;
                for (var j = 0; j < features; j++)
                {
                    var delta = data[rowOffset + j] - centres[centreOffset + j];
                    accumulator += delta * delta;
                    // Abandoning early once the running distance already exceeds the best
                    // candidate saves most of the inner loop on well-separated data.
                    if (accumulator >= bestDistance) break;
                }
                if (accumulator < bestDistance) { bestDistance = accumulator; best = c; }
            }

            if (labels[i] != best) { labels[i] = best; changed = true; }
        }
        return changed;
    }

    /// <summary>Recomputes centroids from the current assignment, on raw buffers.</summary>
    private double UpdateCentroidsFast(double[] data, double[] centres, int samples, int[] labels,
        GraviRandom rng)
    {
        var features = FeatureCount;
        var sums = new double[Clusters * features];
        var counts = new int[Clusters];

        for (var i = 0; i < samples; i++)
        {
            var label = labels[i];
            counts[label]++;
            var rowOffset = i * features;
            var sumOffset = label * features;
            for (var j = 0; j < features; j++) sums[sumOffset + j] += data[rowOffset + j];
        }

        var shift = 0.0;
        for (var c = 0; c < Clusters; c++)
        {
            var centreOffset = c * features;
            if (counts[c] == 0)
            {
                // An empty cluster is re-seeded rather than left to collapse.
                var replacement = rng.Next(samples) * features;
                for (var j = 0; j < features; j++) centres[centreOffset + j] = data[replacement + j];
                continue;
            }
            for (var j = 0; j < features; j++)
            {
                var updated = sums[centreOffset + j] / counts[c];
                shift += Math.Abs(updated - centres[centreOffset + j]);
                centres[centreOffset + j] = updated;
            }
        }
        return shift;
    }

    private bool AssignPoints(NdArray x, NdArray centroids, int[] labels)
    {
        var changed = false;
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            var bestDistance = double.MaxValue;
            for (var c = 0; c < Clusters; c++)
            {
                var d = SquaredDistance(x, i, centroids, c);
                if (d < bestDistance) { bestDistance = d; best = c; }
            }
            if (labels[i] != best) { labels[i] = best; changed = true; }
        }
        return changed;
    }

    private double UpdateCentroids(NdArray x, int[] labels, NdArray centroids, GraviRandom rng)
    {
        var sums = new double[Clusters, FeatureCount];
        var counts = new int[Clusters];

        for (var i = 0; i < x.Shape[0]; i++)
        {
            counts[labels[i]]++;
            for (var j = 0; j < FeatureCount; j++) sums[labels[i], j] += x[i, j];
        }

        var shift = 0.0;
        for (var c = 0; c < Clusters; c++)
        {
            if (counts[c] == 0)
            {
                // An empty cluster is re-seeded rather than left to collapse.
                var replacement = rng.Next(x.Shape[0]);
                for (var j = 0; j < FeatureCount; j++) centroids[c, j] = x[replacement, j];
                continue;
            }
            for (var j = 0; j < FeatureCount; j++)
            {
                var updated = sums[c, j] / counts[c];
                shift += Math.Abs(updated - centroids[c, j]);
                centroids[c, j] = updated;
            }
        }
        return shift;
    }

    private double ComputeInertia(NdArray x, NdArray centroids, int[] labels)
    {
        var acc = 0.0;
        for (var i = 0; i < x.Shape[0]; i++) acc += SquaredDistance(x, i, centroids, labels[i]);
        return acc;
    }

    private double SquaredDistance(NdArray x, int row, NdArray centroids, int centroid)
    {
        var acc = 0.0;
        for (var j = 0; j < FeatureCount; j++)
        {
            var delta = x[row, j] - centroids[centroid, j];
            acc += delta * delta;
        }
        return acc;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            var bestDistance = double.MaxValue;
            for (var c = 0; c < Clusters; c++)
            {
                var d = SquaredDistance(x, i, Centroids, c);
                if (d < bestDistance) { bestDistance = d; best = c; }
            }
            result.SetAt(i, best);
        }
        return result;
    }

    /// <summary>Fits and returns the training labels in one call.</summary>
    public NdArray FitPredict(NdArray x) { Fit(x); return Labels; }

    /// <summary>
    /// Inertia for a range of cluster counts, the input to the elbow method for choosing k.
    /// </summary>
    public static IReadOnlyList<(int K, double Inertia)> ElbowCurve(NdArray x, int maxK = 10, int seed = 42)
    {
        var curve = new List<(int, double)>();
        for (var k = 1; k <= maxK; k++)
        {
            var model = new KMeans(k, restarts: 3, seed: seed);
            model.Fit(x);
            curve.Add((k, model.Inertia));
        }
        return curve;
    }
}

/// <summary>
/// Density-based clustering: groups points that sit in dense neighbourhoods and labels the rest
/// as noise.
/// </summary>
/// <remarks>
/// Unlike k-means, DBSCAN discovers the number of clusters itself, finds arbitrarily shaped ones,
/// and has an explicit notion of an outlier (label -1). The trade-off is two parameters that must
/// suit the data's scale: <see cref="Epsilon"/> is a distance, so the features must be scaled first.
/// </remarks>
public sealed class Dbscan(
    double epsilon = 0.5,
    int minSamples = 5,
    DistanceMetric metric = DistanceMetric.Euclidean) : ModelBase, IClusterer
{
    private const int Unclassified = -2;
    private const int Noise = -1;

    /// <summary>Neighbourhood radius.</summary>
    public double Epsilon { get; } = epsilon;

    /// <summary>Points required within <see cref="Epsilon"/> for a point to be a core point.</summary>
    public int MinSamples { get; } = minSamples;

    /// <summary>Distance function.</summary>
    public DistanceMetric Metric { get; } = metric;

    /// <inheritdoc />
    public NdArray Labels { get; private set; } = NdArray.Zeros(0);

    /// <summary>Number of clusters found, excluding noise.</summary>
    public int ClusterCount { get; private set; }

    /// <summary>Number of points labelled as noise.</summary>
    public int NoiseCount { get; private set; }

    /// <inheritdoc />
    public void Fit(NdArray x)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];

        var labels = new int[samples];
        Array.Fill(labels, Unclassified);

        var cluster = 0;
        for (var i = 0; i < samples; i++)
        {
            if (labels[i] != Unclassified) continue;

            var neighbours = RegionQuery(x, i);
            if (neighbours.Count < MinSamples)
            {
                labels[i] = Noise;
                continue;
            }

            // Grow the cluster outward from this core point.
            labels[i] = cluster;
            var queue = new Queue<int>(neighbours.Where(n => n != i));

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                // A point previously called noise can still join as a border point.
                if (labels[current] == Noise) labels[current] = cluster;
                if (labels[current] != Unclassified) continue;

                labels[current] = cluster;
                var currentNeighbours = RegionQuery(x, current);
                if (currentNeighbours.Count >= MinSamples)
                    foreach (var n in currentNeighbours)
                        if (labels[n] == Unclassified) queue.Enqueue(n);
            }
            cluster++;
        }

        ClusterCount = cluster;
        NoiseCount = labels.Count(l => l == Noise);
        Labels = NdArray.FromValues(labels.Select(l => (double)l));
        IsFitted = true;
    }

    private List<int> RegionQuery(NdArray x, int row)
    {
        var neighbours = new List<int>();
        for (var j = 0; j < x.Shape[0]; j++)
            if (Distances.Between(x, row, x, j, Metric) <= Epsilon) neighbours.Add(j);
        return neighbours;
    }

    /// <summary>
    /// DBSCAN has no out-of-sample rule; each query point takes the label of its nearest
    /// training point when that point is within <see cref="Epsilon"/>, and noise otherwise.
    /// </summary>
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        throw new NotSupportedException(
            "DBSCAN does not support out-of-sample prediction. Use FitPredict, or refit including the new points.");
    }

    /// <summary>Fits and returns the training labels in one call.</summary>
    public NdArray FitPredict(NdArray x) { Fit(x); return Labels; }
}

/// <summary>How the distance between two clusters is defined during agglomeration.</summary>
public enum Linkage
{
    /// <summary>Distance between the closest pair; produces long, chained clusters.</summary>
    Single,

    /// <summary>Distance between the furthest pair; produces compact clusters.</summary>
    Complete,

    /// <summary>Mean distance over all pairs.</summary>
    Average,
}

/// <summary>
/// Bottom-up hierarchical clustering: every point starts alone and the closest pair of clusters
/// is merged until the requested count remains.
/// </summary>
public sealed class AgglomerativeClustering(
    int clusters = 2,
    Linkage linkage = Linkage.Average,
    DistanceMetric metric = DistanceMetric.Euclidean) : ModelBase, IClusterer
{
    /// <summary>Number of clusters to stop at.</summary>
    public int Clusters { get; } = clusters;

    /// <summary>Cluster-distance definition.</summary>
    public Linkage Linkage { get; } = linkage;

    /// <inheritdoc />
    public NdArray Labels { get; private set; } = NdArray.Zeros(0);

    /// <summary>The merge history as (clusterA, clusterB, distance) triples.</summary>
    public IReadOnlyList<(int A, int B, double Distance)> MergeHistory => _merges;

    private readonly List<(int, int, double)> _merges = [];

    /// <inheritdoc />
    public void Fit(NdArray x)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];
        _merges.Clear();

        var distances = new double[samples, samples];
        for (var i = 0; i < samples; i++)
            for (var j = i + 1; j < samples; j++)
                distances[i, j] = distances[j, i] = Distances.Between(x, i, x, j, metric);

        var members = Enumerable.Range(0, samples).Select(i => new List<int> { i }).ToList();
        var active = Enumerable.Range(0, samples).ToList();

        while (active.Count > Clusters)
        {
            var bestA = -1;
            var bestB = -1;
            var bestDistance = double.MaxValue;

            for (var a = 0; a < active.Count; a++)
                for (var b = a + 1; b < active.Count; b++)
                {
                    var d = ClusterDistance(distances, members[active[a]], members[active[b]]);
                    if (d < bestDistance) { bestDistance = d; bestA = a; bestB = b; }
                }

            if (bestA < 0) break;

            var target = active[bestA];
            var absorbed = active[bestB];
            _merges.Add((target, absorbed, bestDistance));
            members[target].AddRange(members[absorbed]);
            active.RemoveAt(bestB);
        }

        var labels = new double[samples];
        for (var c = 0; c < active.Count; c++)
            foreach (var i in members[active[c]]) labels[i] = c;

        Labels = new NdArray(labels, samples);
        IsFitted = true;
    }

    private double ClusterDistance(double[,] distances, List<int> a, List<int> b) => Linkage switch
    {
        Linkage.Single => a.SelectMany(i => b.Select(j => distances[i, j])).Min(),
        Linkage.Complete => a.SelectMany(i => b.Select(j => distances[i, j])).Max(),
        _ => a.SelectMany(i => b.Select(j => distances[i, j])).Average(),
    };

    /// <summary>Hierarchical clustering has no out-of-sample rule; refit with the new points.</summary>
    public NdArray Predict(NdArray x)
        => throw new NotSupportedException("Agglomerative clustering does not support out-of-sample prediction.");

    /// <summary>Fits and returns the training labels in one call.</summary>
    public NdArray FitPredict(NdArray x) { Fit(x); return Labels; }
}

/// <summary>
/// A Gaussian mixture model fitted by expectation-maximisation.
/// </summary>
/// <remarks>
/// This is the soft-assignment generalisation of k-means: instead of a hard nearest-centroid rule
/// each point gets a responsibility for every component, and components carry a full covariance,
/// so they can be elongated and rotated rather than spherical. The log-likelihood is guaranteed
/// not to decrease between iterations, which is what <see cref="Converged"/> checks.
/// </remarks>
public sealed class GaussianMixture(
    int components = 2,
    int maxIterations = 200,
    double tolerance = 1e-6,
    double regularization = 1e-6,
    int seed = 42) : ModelBase, IClusterer
{
    /// <summary>Number of mixture components.</summary>
    public int Components { get; } = components;

    /// <summary>Mixing proportions, summing to one.</summary>
    public NdArray Weights { get; private set; } = NdArray.Zeros(0);

    /// <summary>Component means, one per row.</summary>
    public NdArray Means { get; private set; } = NdArray.Zeros(0, 0);

    /// <summary>Component covariance matrices.</summary>
    public NdArray[] Covariances { get; private set; } = [];

    /// <summary>Final average log-likelihood per sample.</summary>
    public double LogLikelihood { get; private set; }

    /// <summary>True when the likelihood stopped improving before the iteration cap.</summary>
    public bool Converged { get; private set; }

    /// <inheritdoc />
    public NdArray Labels { get; private set; } = NdArray.Zeros(0);

    /// <inheritdoc />
    public void Fit(NdArray x)
    {
        var samples = ValidateMatrix(x);
        FeatureCount = x.Shape[1];

        // k-means gives EM a sane starting point; random starts often collapse a component.
        var kmeans = new KMeans(Components, restarts: 3, seed: seed);
        kmeans.Fit(x);

        Means = kmeans.Centroids.Copy();
        Weights = NdArray.Full(1.0 / Components, Components);
        Covariances = new NdArray[Components];
        for (var c = 0; c < Components; c++)
        {
            var covariance = NdArray.Eye(FeatureCount);
            for (var j = 0; j < FeatureCount; j++)
                covariance[j, j] = Statistics.Var(x.Column(j).Copy()) + regularization;
            Covariances[c] = covariance;
        }

        var responsibilities = NdArray.Zeros(samples, Components);
        var previousLikelihood = double.NegativeInfinity;
        Converged = false;

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            // E step: responsibility of each component for each point.
            var totalLogLikelihood = 0.0;
            var inverses = Covariances.Select(LinAlg.Inverse).ToArray();
            var logDeterminants = Covariances.Select(c => LinAlg.SlogDet(c).LogAbsDet).ToArray();

            for (var i = 0; i < samples; i++)
            {
                var logProbabilities = new double[Components];
                for (var c = 0; c < Components; c++)
                    logProbabilities[c] = Math.Log(Math.Max(Weights.At(c), 1e-300))
                        + LogGaussian(x, i, Means, c, inverses[c], logDeterminants[c]);

                var normaliser = MathUtil.LogSumExp(logProbabilities);
                totalLogLikelihood += normaliser;
                for (var c = 0; c < Components; c++)
                    responsibilities[i, c] = Math.Exp(logProbabilities[c] - normaliser);
            }

            LogLikelihood = totalLogLikelihood / samples;

            // M step: re-estimate weights, means and covariances from the responsibilities.
            for (var c = 0; c < Components; c++)
            {
                var mass = 0.0;
                for (var i = 0; i < samples; i++) mass += responsibilities[i, c];
                mass = Math.Max(mass, 1e-10);

                Weights.SetAt(c, mass / samples);

                for (var j = 0; j < FeatureCount; j++)
                {
                    var acc = 0.0;
                    for (var i = 0; i < samples; i++) acc += responsibilities[i, c] * x[i, j];
                    Means[c, j] = acc / mass;
                }

                var covariance = NdArray.Zeros(FeatureCount, FeatureCount);
                for (var i = 0; i < samples; i++)
                {
                    var r = responsibilities[i, c];
                    if (r < 1e-12) continue;
                    for (var a = 0; a < FeatureCount; a++)
                    {
                        var da = x[i, a] - Means[c, a];
                        for (var b = 0; b < FeatureCount; b++)
                            covariance[a, b] += r * da * (x[i, b] - Means[c, b]);
                    }
                }
                for (var a = 0; a < FeatureCount; a++)
                {
                    for (var b = 0; b < FeatureCount; b++) covariance[a, b] /= mass;
                    covariance[a, a] += regularization;
                }
                Covariances[c] = covariance;
            }

            if (Math.Abs(LogLikelihood - previousLikelihood) < tolerance) { Converged = true; break; }
            previousLikelihood = LogLikelihood;
        }

        var labels = NdArray.Zeros(samples);
        for (var i = 0; i < samples; i++)
        {
            var best = 0;
            for (var c = 1; c < Components; c++)
                if (responsibilities[i, c] > responsibilities[i, best]) best = c;
            labels.SetAt(i, best);
        }
        Labels = labels;
        IsFitted = true;
    }

    private double LogGaussian(NdArray x, int row, NdArray means, int component,
        NdArray inverse, double logDeterminant)
    {
        var delta = new double[FeatureCount];
        for (var j = 0; j < FeatureCount; j++) delta[j] = x[row, j] - means[component, j];

        var quadratic = 0.0;
        for (var a = 0; a < FeatureCount; a++)
        {
            var acc = 0.0;
            for (var b = 0; b < FeatureCount; b++) acc += inverse[a, b] * delta[b];
            quadratic += delta[a] * acc;
        }

        return -0.5 * (FeatureCount * Math.Log(2 * Math.PI) + logDeterminant + quadratic);
    }

    /// <summary>Posterior probability of each component for each sample.</summary>
    public NdArray PredictProbabilities(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var inverses = Covariances.Select(LinAlg.Inverse).ToArray();
        var logDeterminants = Covariances.Select(c => LinAlg.SlogDet(c).LogAbsDet).ToArray();
        var result = NdArray.Zeros(x.Shape[0], Components);

        for (var i = 0; i < x.Shape[0]; i++)
        {
            var logProbabilities = new double[Components];
            for (var c = 0; c < Components; c++)
                logProbabilities[c] = Math.Log(Math.Max(Weights.At(c), 1e-300))
                    + LogGaussian(x, i, Means, c, inverses[c], logDeterminants[c]);

            var probabilities = MathUtil.Softmax(logProbabilities);
            for (var c = 0; c < Components; c++) result[i, c] = probabilities[c];
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var probabilities = PredictProbabilities(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var best = 0;
            for (var c = 1; c < Components; c++)
                if (probabilities[i, c] > probabilities[i, best]) best = c;
            result.SetAt(i, best);
        }
        return result;
    }

    /// <summary>Bayesian information criterion, for choosing the number of components.</summary>
    public double Bic(NdArray x)
    {
        RequireFitted();
        var parameters = Components - 1
            + Components * FeatureCount
            + Components * FeatureCount * (FeatureCount + 1) / 2;
        return -2 * LogLikelihood * x.Shape[0] + parameters * Math.Log(x.Shape[0]);
    }

    /// <summary>Fits and returns the training labels in one call.</summary>
    public NdArray FitPredict(NdArray x) { Fit(x); return Labels; }
}
