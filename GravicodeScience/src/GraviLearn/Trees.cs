using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Trees;

/// <summary>The impurity measure a decision tree minimises when choosing a split.</summary>
public enum SplitCriterion
{
    /// <summary>Gini impurity (classification). Cheapest and the usual default.</summary>
    Gini,

    /// <summary>Information gain (classification).</summary>
    Entropy,

    /// <summary>Mean squared error (regression).</summary>
    Mse,
}

/// <summary>One node of a decision tree: either a split or a leaf.</summary>
public sealed class TreeNode
{
    /// <summary>Feature tested at this node, or -1 for a leaf.</summary>
    public int Feature { get; init; } = -1;

    /// <summary>Threshold; samples with <c>feature &lt;= threshold</c> go left.</summary>
    public double Threshold { get; init; }

    /// <summary>Left child.</summary>
    public TreeNode? Left { get; set; }

    /// <summary>Right child.</summary>
    public TreeNode? Right { get; set; }

    /// <summary>
    /// Prediction produced at a leaf. Boosting rewrites this after fitting, so it is settable
    /// inside the library while staying read-only to callers.
    /// </summary>
    public double Value { get; internal set; }

    /// <summary>Class distribution at a leaf, for probability estimates.</summary>
    public double[]? ClassProbabilities { get; init; }

    /// <summary>Number of training samples that reached this node.</summary>
    public int Samples { get; init; }

    /// <summary>Impurity at this node before splitting.</summary>
    public double Impurity { get; init; }

    /// <summary>True when this node makes a prediction rather than a decision.</summary>
    public bool IsLeaf => Left is null && Right is null;
}

/// <summary>
/// A CART decision tree, used directly or as the base learner of a forest or boosting ensemble.
/// </summary>
/// <remarks>
/// Splits are chosen greedily: for each candidate feature the samples are sorted once and every
/// midpoint between consecutive distinct values is scored, keeping the split with the largest
/// impurity decrease. <see cref="MaxFeatures"/> restricts how many features each node may consider,
/// which is exactly the decorrelation mechanism a random forest depends on - it is a tree
/// parameter rather than forest bookkeeping.
/// </remarks>
public class DecisionTree(
    SplitCriterion criterion = SplitCriterion.Gini,
    int maxDepth = 0,
    int minSamplesSplit = 2,
    int minSamplesLeaf = 1,
    int maxFeatures = 0,
    int seed = 42) : ModelBase, IClassifier
{
    private TreeNode? _root;
    private double[] _classes = [];
    private double[] _featureImportances = [];
    private GraviRandom _rng = new(seed);

    /// <summary>Impurity measure used for split selection.</summary>
    public SplitCriterion Criterion { get; } = criterion;

    /// <summary>Maximum depth; zero means unlimited.</summary>
    public int MaxDepth { get; } = maxDepth;

    /// <summary>Minimum samples required to consider splitting a node.</summary>
    public int MinSamplesSplit { get; } = minSamplesSplit;

    /// <summary>Minimum samples that must remain in each child.</summary>
    public int MinSamplesLeaf { get; } = minSamplesLeaf;

    /// <summary>Features sampled per split; zero means all of them.</summary>
    public int MaxFeatures { get; } = maxFeatures;

    /// <summary>Seed for the feature subsampling.</summary>
    public int Seed { get; } = seed;

    /// <summary>True when the tree predicts a continuous target.</summary>
    public bool IsRegression => Criterion == SplitCriterion.Mse;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>The fitted tree's root, for inspection or rendering.</summary>
    public TreeNode? Root => _root;

    /// <summary>Depth of the fitted tree.</summary>
    public int Depth => MeasureDepth(_root);

    /// <summary>Number of leaves in the fitted tree.</summary>
    public int LeafCount => CountLeaves(_root);

    /// <summary>
    /// Normalised total impurity decrease contributed by each feature.
    /// </summary>
    public NdArray FeatureImportances
    {
        get
        {
            RequireFitted();
            var total = _featureImportances.Sum();
            var result = NdArray.Zeros(FeatureCount);
            for (var j = 0; j < FeatureCount; j++)
                result.SetAt(j, total > 0 ? _featureImportances[j] / total : 0.0);
            return result;
        }
    }

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _rng = new GraviRandom(Seed);
        _featureImportances = new double[FeatureCount];
        _classes = IsRegression ? [] : DistinctClasses(y);

        var indices = Enumerable.Range(0, samples).ToArray();
        _root = Build(x, y, indices, depth: 0);
        IsFitted = true;
    }

    /// <summary>Fits using sample weights expressed as a bootstrap index list (used by ensembles).</summary>
    internal void FitOnIndices(NdArray x, NdArray y, int[] indices, double[] classes)
    {
        FeatureCount = x.Shape[1];
        _rng = new GraviRandom(Seed);
        _featureImportances = new double[FeatureCount];
        _classes = IsRegression ? [] : classes;
        _root = Build(x, y, indices, depth: 0);
        IsFitted = true;
    }

    private TreeNode Build(NdArray x, NdArray y, int[] indices, int depth)
    {
        var impurity = Impurity(y, indices);
        var leaf = MakeLeaf(y, indices, impurity);

        if (indices.Length < MinSamplesSplit) return leaf;
        if (MaxDepth > 0 && depth >= MaxDepth) return leaf;
        if (impurity <= 1e-12) return leaf;

        var (feature, threshold, gain, left, right) = FindBestSplit(x, y, indices, impurity);
        if (feature < 0 || left is null || right is null) return leaf;

        // Weight the importance by how many samples the split affects.
        _featureImportances[feature] += gain * indices.Length;

        return new TreeNode
        {
            Feature = feature,
            Threshold = threshold,
            Samples = indices.Length,
            Impurity = impurity,
            Value = leaf.Value,
            ClassProbabilities = leaf.ClassProbabilities,
            Left = Build(x, y, left, depth + 1),
            Right = Build(x, y, right, depth + 1),
        };
    }

    private (int Feature, double Threshold, double Gain, int[]? Left, int[]? Right) FindBestSplit(
        NdArray x, NdArray y, int[] indices, double parentImpurity)
    {
        var candidates = SelectFeatures();

        var bestFeature = -1;
        var bestThreshold = 0.0;
        var bestGain = 0.0;
        int[]? bestLeft = null;
        int[]? bestRight = null;

        foreach (var feature in candidates)
        {
            // Sorting once per feature turns threshold search into a single sweep.
            var sorted = indices.OrderBy(i => x[i, feature]).ToArray();

            for (var k = MinSamplesLeaf; k <= sorted.Length - MinSamplesLeaf; k++)
            {
                var lowValue = x[sorted[k - 1], feature];
                var highValue = x[sorted[k], feature];
                if (lowValue == highValue) continue;   // no threshold separates equal values

                var leftIndices = sorted[..k];
                var rightIndices = sorted[k..];

                var leftImpurity = Impurity(y, leftIndices);
                var rightImpurity = Impurity(y, rightIndices);
                var weighted = (leftIndices.Length * leftImpurity + rightIndices.Length * rightImpurity)
                    / sorted.Length;
                var gain = parentImpurity - weighted;

                if (gain > bestGain + 1e-12)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = (lowValue + highValue) / 2.0;
                    bestLeft = leftIndices;
                    bestRight = rightIndices;
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain, bestLeft, bestRight);
    }

    private int[] SelectFeatures()
    {
        if (MaxFeatures <= 0 || MaxFeatures >= FeatureCount)
            return Enumerable.Range(0, FeatureCount).ToArray();
        return _rng.Choice(FeatureCount, MaxFeatures, replace: false);
    }

    private TreeNode MakeLeaf(NdArray y, int[] indices, double impurity)
    {
        if (IsRegression)
        {
            var mean = indices.Length == 0 ? 0.0 : indices.Average(i => y.At(i));
            return new TreeNode { Value = mean, Samples = indices.Length, Impurity = impurity };
        }

        var counts = new double[_classes.Length];
        foreach (var i in indices)
        {
            var index = Array.IndexOf(_classes, y.At(i));
            if (index >= 0) counts[index]++;
        }

        var probabilities = new double[_classes.Length];
        var best = 0;
        for (var c = 0; c < _classes.Length; c++)
        {
            probabilities[c] = indices.Length == 0 ? 0.0 : counts[c] / indices.Length;
            if (counts[c] > counts[best]) best = c;
        }

        return new TreeNode
        {
            Value = _classes.Length == 0 ? 0.0 : _classes[best],
            ClassProbabilities = probabilities,
            Samples = indices.Length,
            Impurity = impurity,
        };
    }

    private double Impurity(NdArray y, int[] indices)
    {
        if (indices.Length == 0) return 0.0;

        if (IsRegression)
        {
            var mean = indices.Average(i => y.At(i));
            var acc = 0.0;
            foreach (var i in indices)
            {
                var d = y.At(i) - mean;
                acc += d * d;
            }
            return acc / indices.Length;
        }

        var counts = new Dictionary<double, int>();
        foreach (var i in indices)
        {
            counts.TryGetValue(y.At(i), out var c);
            counts[y.At(i)] = c + 1;
        }

        if (Criterion == SplitCriterion.Entropy)
        {
            var entropy = 0.0;
            foreach (var count in counts.Values)
            {
                var p = (double)count / indices.Length;
                if (p > 0) entropy -= p * Math.Log2(p);
            }
            return entropy;
        }

        var gini = 1.0;
        foreach (var count in counts.Values)
        {
            var p = (double)count / indices.Length;
            gini -= p * p;
        }
        return gini;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++) result.SetAt(i, Descend(x, i).Value);
        return result;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        RequireFitted();
        if (IsRegression) throw new InvalidOperationException("A regression tree has no class probabilities.");
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var leaf = Descend(x, i);
            var probabilities = leaf.ClassProbabilities;
            if (probabilities is null) continue;
            for (var c = 0; c < _classes.Length; c++) result[i, c] = probabilities[c];
        }
        return result;
    }

    internal TreeNode Descend(NdArray x, int row)
    {
        var node = _root!;
        while (!node.IsLeaf)
            node = x[row, node.Feature] <= node.Threshold ? node.Left! : node.Right!;
        return node;
    }

    /// <summary>
    /// Replaces each leaf's prediction with a value computed from the training rows that land in
    /// it. Boosting uses this to apply a Newton step: the tree structure comes from the gradient,
    /// but the leaf value that actually minimises the loss also needs the curvature.
    /// </summary>
    internal void RecomputeLeafValues(NdArray x, int[] indices, Func<IReadOnlyList<int>, double> valueFor)
    {
        RequireFitted();
        var buckets = new Dictionary<TreeNode, List<int>>();
        foreach (var i in indices)
        {
            var leaf = Descend(x, i);
            if (!buckets.TryGetValue(leaf, out var rows)) buckets[leaf] = rows = [];
            rows.Add(i);
        }
        foreach (var (leaf, rows) in buckets) leaf.Value = valueFor(rows);
    }

    /// <summary>Accuracy (classification) or R-squared (regression) on the given data.</summary>
    public double Score(NdArray x, NdArray y)
        => IsRegression ? Metrics.R2Score(y, Predict(x)) : Metrics.Accuracy(y, Predict(x));

    private static int MeasureDepth(TreeNode? node)
        => node is null || node.IsLeaf ? 0 : 1 + Math.Max(MeasureDepth(node.Left), MeasureDepth(node.Right));

    private static int CountLeaves(TreeNode? node)
        => node is null ? 0 : node.IsLeaf ? 1 : CountLeaves(node.Left) + CountLeaves(node.Right);

    /// <summary>Renders the tree as indented text, which is often enough to explain a model.</summary>
    public string Render(IReadOnlyList<string>? featureNames = null, int maxDepth = 6)
    {
        RequireFitted();
        var sb = new System.Text.StringBuilder();
        RenderNode(_root, sb, "", 0, maxDepth, featureNames);
        return sb.ToString();
    }

    private void RenderNode(TreeNode? node, System.Text.StringBuilder sb, string indent,
        int depth, int maxDepth, IReadOnlyList<string>? names)
    {
        if (node is null) return;
        if (node.IsLeaf)
        {
            sb.AppendLine($"{indent}predict {node.Value:0.####}  (n={node.Samples})");
            return;
        }
        if (depth >= maxDepth)
        {
            sb.AppendLine($"{indent}... (subtree of {node.Samples} samples)");
            return;
        }

        var name = names is not null && node.Feature < names.Count ? names[node.Feature] : $"f{node.Feature}";
        sb.AppendLine($"{indent}if {name} <= {node.Threshold:0.####}  (n={node.Samples}, impurity={node.Impurity:0.####})");
        RenderNode(node.Left, sb, indent + "  ", depth + 1, maxDepth, names);
        sb.AppendLine($"{indent}else");
        RenderNode(node.Right, sb, indent + "  ", depth + 1, maxDepth, names);
    }
}

/// <summary>A decision tree that predicts a continuous target.</summary>
public sealed class DecisionTreeRegressor(
    int maxDepth = 0, int minSamplesSplit = 2, int minSamplesLeaf = 1, int maxFeatures = 0, int seed = 42)
    : DecisionTree(SplitCriterion.Mse, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, seed);

/// <summary>
/// A random forest: many decision trees fitted on bootstrap samples, each split restricted to a
/// random subset of features, with predictions combined by vote or average.
/// </summary>
/// <remarks>
/// The two sources of randomness are doing different jobs. Bootstrapping the rows decorrelates
/// the trees' training data; restricting the features per split decorrelates their structure,
/// which is what stops one dominant predictor from making every tree the same. Averaging
/// decorrelated high-variance trees is what buys the accuracy. Trees are fitted in parallel
/// because they are completely independent.
/// </remarks>
public sealed class RandomForestClassifier(
    int nTrees = 100,
    int maxDepth = 0,
    int minSamplesSplit = 2,
    int minSamplesLeaf = 1,
    int maxFeatures = 0,
    bool bootstrap = true,
    int seed = 42,
    SplitCriterion criterion = SplitCriterion.Gini) : ModelBase, IClassifier
{
    private DecisionTree[] _trees = [];
    private double[] _classes = [];

    /// <summary>Number of trees in the ensemble.</summary>
    public int TreeCount { get; } = nTrees;

    /// <summary>Maximum depth of each tree; zero means unlimited.</summary>
    public int MaxDepth { get; } = maxDepth;

    /// <summary>Minimum samples required to split a node.</summary>
    public int MinSamplesSplit { get; } = minSamplesSplit;

    /// <summary>Minimum samples that must remain in each child.</summary>
    public int MinSamplesLeaf { get; } = minSamplesLeaf;

    /// <summary>
    /// Features considered per split; zero selects the square root of the feature count, the
    /// standard choice for classification.
    /// </summary>
    public int MaxFeatures { get; } = maxFeatures;

    /// <summary>Whether each tree is fitted on a bootstrap resample.</summary>
    public bool Bootstrap { get; } = bootstrap;

    /// <summary>Seed controlling bootstrapping and feature subsampling.</summary>
    public int Seed { get; } = seed;

    /// <summary>Impurity measure used by every tree.</summary>
    public SplitCriterion Criterion { get; } = criterion;

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>The fitted trees.</summary>
    public IReadOnlyList<DecisionTree> Trees => _trees;

    /// <summary>Feature importances averaged across the ensemble.</summary>
    public NdArray FeatureImportances
    {
        get
        {
            RequireFitted();
            var total = NdArray.Zeros(FeatureCount);
            foreach (var tree in _trees) total += tree.FeatureImportances;
            return total / _trees.Length;
        }
    }

    /// <summary>
    /// Accuracy measured on the samples each tree did not see, an honest estimate that needs no
    /// held-out split. Available only when <see cref="Bootstrap"/> is enabled.
    /// </summary>
    public double OutOfBagScore { get; private set; } = double.NaN;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _classes = DistinctClasses(y);

        var featuresPerSplit = MaxFeatures > 0
            ? MaxFeatures
            : Math.Max(1, (int)Math.Sqrt(FeatureCount));

        _trees = new DecisionTree[TreeCount];
        var outOfBagVotes = new Dictionary<double, int>[samples];
        for (var i = 0; i < samples; i++) outOfBagVotes[i] = [];

        var lockObject = new object();

        Parallel.For(0, TreeCount, t =>
        {
            var rng = new GraviRandom(Seed + t * 7919);
            var indices = Bootstrap
                ? rng.Choice(samples, samples, replace: true)
                : Enumerable.Range(0, samples).ToArray();

            var tree = new DecisionTree(Criterion, MaxDepth, MinSamplesSplit, MinSamplesLeaf,
                featuresPerSplit, Seed + t * 7919);
            tree.FitOnIndices(x, y, indices, _classes);
            _trees[t] = tree;

            if (!Bootstrap) return;

            var inBag = new HashSet<int>(indices);
            var outOfBag = Enumerable.Range(0, samples).Where(i => !inBag.Contains(i)).ToArray();
            if (outOfBag.Length == 0) return;

            var predictions = tree.Predict(x.Take(outOfBag));
            lock (lockObject)
            {
                for (var k = 0; k < outOfBag.Length; k++)
                {
                    var votes = outOfBagVotes[outOfBag[k]];
                    votes.TryGetValue(predictions.At(k), out var count);
                    votes[predictions.At(k)] = count + 1;
                }
            }
        });

        if (Bootstrap)
        {
            var scored = 0;
            var correct = 0;
            for (var i = 0; i < samples; i++)
            {
                if (outOfBagVotes[i].Count == 0) continue;
                var winner = outOfBagVotes[i].MaxBy(kv => kv.Value).Key;
                if (winner == y.At(i)) correct++;
                scored++;
            }
            OutOfBagScore = scored == 0 ? double.NaN : (double)correct / scored;
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Zeros(x.Shape[0], _classes.Length);
        foreach (var tree in _trees)
        {
            var probabilities = tree.PredictProbabilities(x);
            for (var i = 0; i < x.Shape[0]; i++)
                for (var c = 0; c < _classes.Length; c++)
                    result[i, c] += probabilities[i, c];
        }

        for (var i = 0; i < x.Shape[0]; i++)
            for (var c = 0; c < _classes.Length; c++)
                result[i, c] /= _trees.Length;
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
            for (var c = 1; c < _classes.Length; c++)
                if (probabilities[i, c] > probabilities[i, best]) best = c;
            result.SetAt(i, _classes[best]);
        }
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}

/// <summary>A random forest that predicts a continuous target by averaging its trees.</summary>
public sealed class RandomForestRegressor(
    int nTrees = 100, int maxDepth = 0, int minSamplesSplit = 2, int minSamplesLeaf = 1,
    int maxFeatures = 0, bool bootstrap = true, int seed = 42) : ModelBase, IEstimator
{
    private DecisionTree[] _trees = [];

    /// <summary>Number of trees.</summary>
    public int TreeCount { get; } = nTrees;

    /// <summary>Feature importances averaged across the ensemble.</summary>
    public NdArray FeatureImportances
    {
        get
        {
            RequireFitted();
            var total = NdArray.Zeros(FeatureCount);
            foreach (var tree in _trees) total += tree.FeatureImportances;
            return total / _trees.Length;
        }
    }

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];

        // Regression forests traditionally use a third of the features per split.
        var featuresPerSplit = maxFeatures > 0 ? maxFeatures : Math.Max(1, FeatureCount / 3);
        _trees = new DecisionTree[TreeCount];

        Parallel.For(0, TreeCount, t =>
        {
            var rng = new GraviRandom(seed + t * 7919);
            var indices = bootstrap
                ? rng.Choice(samples, samples, replace: true)
                : Enumerable.Range(0, samples).ToArray();

            var tree = new DecisionTreeRegressor(maxDepth, minSamplesSplit, minSamplesLeaf,
                featuresPerSplit, seed + t * 7919);
            tree.FitOnIndices(x, y, indices, []);
            _trees[t] = tree;
        });

        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var result = NdArray.Zeros(x.Shape[0]);
        foreach (var tree in _trees) result += tree.Predict(x);
        return result / _trees.Length;
    }

    /// <summary>R-squared on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.R2Score(y, Predict(x));
}

/// <summary>
/// Gradient boosted regression trees: each tree is fitted to the residual error left by the
/// ensemble so far.
/// </summary>
/// <remarks>
/// Where a forest averages independent trees to cut variance, boosting adds dependent shallow
/// trees to cut bias. The learning rate shrinks each tree's contribution; lower rates need more
/// trees but generalise better, which is why the two parameters are always tuned together.
/// </remarks>
public sealed class GradientBoostingRegressor(
    int nTrees = 100,
    double learningRate = 0.1,
    int maxDepth = 3,
    int minSamplesLeaf = 1,
    double subsample = 1.0,
    int seed = 42) : ModelBase, IEstimator
{
    private readonly List<DecisionTree> _trees = [];
    private double _initialPrediction;

    /// <summary>Number of boosting rounds.</summary>
    public int TreeCount { get; } = nTrees;

    /// <summary>Shrinkage applied to each tree's contribution.</summary>
    public double LearningRate { get; } = learningRate;

    /// <summary>Depth of each tree; boosting wants weak learners, so this is small by default.</summary>
    public int MaxDepth { get; } = maxDepth;

    /// <summary>Fraction of rows sampled per round; below 1 this is stochastic gradient boosting.</summary>
    public double Subsample { get; } = subsample;

    /// <summary>Training loss after each round, useful for spotting overfitting.</summary>
    public IReadOnlyList<double> TrainingLoss => _lossHistory;

    private readonly List<double> _lossHistory = [];

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _trees.Clear();
        _lossHistory.Clear();

        // The constant that minimises squared error is the mean; boosting starts there.
        _initialPrediction = Statistics.Mean(y);
        var predictions = new double[samples];
        Array.Fill(predictions, _initialPrediction);

        var rng = new GraviRandom(seed);
        var subsampleSize = Math.Max(1, (int)(samples * Subsample));

        for (var round = 0; round < TreeCount; round++)
        {
            // The negative gradient of squared error is simply the residual.
            var residuals = NdArray.Zeros(samples);
            for (var i = 0; i < samples; i++) residuals.SetAt(i, y.At(i) - predictions[i]);

            var indices = Subsample >= 1.0
                ? Enumerable.Range(0, samples).ToArray()
                : rng.Choice(samples, subsampleSize, replace: false);

            var tree = new DecisionTreeRegressor(MaxDepth, 2, minSamplesLeaf, 0, seed + round);
            tree.FitOnIndices(x, residuals, indices, []);
            _trees.Add(tree);

            var update = tree.Predict(x);
            var loss = 0.0;
            for (var i = 0; i < samples; i++)
            {
                predictions[i] += LearningRate * update.At(i);
                var e = y.At(i) - predictions[i];
                loss += e * e;
            }
            _lossHistory.Add(loss / samples);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);

        var result = NdArray.Full(_initialPrediction, x.Shape[0]);
        foreach (var tree in _trees) result += tree.Predict(x) * LearningRate;
        return result;
    }

    /// <summary>R-squared on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.R2Score(y, Predict(x));
}

/// <summary>
/// Gradient boosting for binary classification, boosting the log-odds with squared-error trees
/// fitted to the residual between the label and the current probability.
/// </summary>
public sealed class GradientBoostingClassifier(
    int nTrees = 100, double learningRate = 0.1, int maxDepth = 3, int seed = 42)
    : ModelBase, IClassifier
{
    private readonly List<DecisionTree> _trees = [];
    private double _initialLogOdds;
    private double[] _classes = [];

    /// <inheritdoc />
    public IReadOnlyList<double> Classes => _classes;

    /// <summary>Number of boosting rounds.</summary>
    public int TreeCount { get; } = nTrees;

    /// <inheritdoc />
    public void Fit(NdArray x, NdArray y)
    {
        var samples = ValidateMatrix(x);
        ValidateTarget(x, y);
        FeatureCount = x.Shape[1];
        _classes = DistinctClasses(y);
        if (_classes.Length != 2)
            throw new ArgumentException("GradientBoostingClassifier supports binary targets only.");

        _trees.Clear();

        var positives = 0;
        for (var i = 0; i < samples; i++) if (y.At(i) == _classes[1]) positives++;
        var baseRate = Math.Clamp((double)positives / samples, 1e-6, 1 - 1e-6);
        _initialLogOdds = Math.Log(baseRate / (1 - baseRate));

        var scores = new double[samples];
        Array.Fill(scores, _initialLogOdds);

        var indices = Enumerable.Range(0, samples).ToArray();

        for (var round = 0; round < TreeCount; round++)
        {
            // Residual of the logistic loss with respect to the score is (label - probability).
            var residuals = NdArray.Zeros(samples);
            for (var i = 0; i < samples; i++)
            {
                var label = y.At(i) == _classes[1] ? 1.0 : 0.0;
                residuals.SetAt(i, label - MathUtil.Sigmoid(scores[i]));
            }

            var tree = new DecisionTreeRegressor(maxDepth, 2, 1, 0, seed + round);
            tree.FitOnIndices(x, residuals, indices, []);

            // Friedman's Newton step. The tree found *where* to split from the gradient; the leaf
            // value that actually minimises the logistic loss is gradient / curvature, and
            // p(1-p) is the curvature. Without this the ensemble converges markedly slower.
            tree.RecomputeLeafValues(x, indices, rows =>
            {
                double numerator = 0, denominator = 0;
                foreach (var i in rows)
                {
                    var p = MathUtil.Sigmoid(scores[i]);
                    numerator += residuals.At(i);
                    denominator += p * (1 - p);
                }
                if (denominator < 1e-12) return 0.0;
                // Clamp so a nearly pure leaf cannot produce an unbounded step.
                return Math.Clamp(numerator / denominator, -10.0, 10.0);
            });
            _trees.Add(tree);

            var update = tree.Predict(x);
            for (var i = 0; i < samples; i++) scores[i] += learningRate * update.At(i);
        }

        IsFitted = true;
    }

    /// <summary>The boosted log-odds for each sample.</summary>
    public NdArray DecisionFunction(NdArray x)
    {
        RequireFitted();
        ValidateForPrediction(x);
        var scores = NdArray.Full(_initialLogOdds, x.Shape[0]);
        foreach (var tree in _trees) scores += tree.Predict(x) * learningRate;
        return scores;
    }

    /// <inheritdoc />
    public NdArray PredictProbabilities(NdArray x)
    {
        var scores = DecisionFunction(x);
        var result = NdArray.Zeros(x.Shape[0], 2);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var p = MathUtil.Sigmoid(scores.At(i));
            result[i, 0] = 1 - p;
            result[i, 1] = p;
        }
        return result;
    }

    /// <inheritdoc />
    public NdArray Predict(NdArray x)
    {
        var scores = DecisionFunction(x);
        var result = NdArray.Zeros(x.Shape[0]);
        for (var i = 0; i < x.Shape[0]; i++)
            result.SetAt(i, scores.At(i) >= 0 ? _classes[1] : _classes[0]);
        return result;
    }

    /// <summary>Accuracy on the given data.</summary>
    public double Score(NdArray x, NdArray y) => Metrics.Accuracy(y, Predict(x));
}
