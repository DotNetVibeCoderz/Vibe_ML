using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn.Clustering;

/// <summary>
/// HDBSCAN: density clustering that does not need a single density threshold.
/// </summary>
/// <remarks>
/// <para>
/// DBSCAN asks for one <c>eps</c> and applies it everywhere. That is fine when every cluster has
/// the same density and hopeless when they do not: an <c>eps</c> tight enough to keep two dense
/// clusters apart shreds a sparse one into noise, and an <c>eps</c> loose enough to hold the sparse
/// one together merges the dense pair. There is no value that works, which is the failure this
/// algorithm exists to fix.
/// </para>
/// <para>
/// The route around it is to run DBSCAN at <em>every</em> threshold at once and then ask which
/// clusters survived longest. Concretely:
/// </para>
/// <list type="number">
/// <item>Compute each point's <b>core distance</b> — the distance to its <c>minSamples</c>-th
/// nearest neighbour. This is a local density estimate: small in a dense region, large in a sparse
/// one.</item>
/// <item>Define <b>mutual reachability</b> between two points as
/// <c>max(core(a), core(b), d(a, b))</c>. Pushing points apart in proportion to how sparse their
/// neighbourhood is, is what puts clusters of different densities on a common footing.</item>
/// <item>Build the minimum spanning tree of that metric, and add edges in increasing order. The
/// order in which components merge is the whole DBSCAN hierarchy.</item>
/// <item><b>Condense</b> the tree: walking down, a split that sheds fewer than
/// <c>minClusterSize</c> points is not a real split — those points just fell out of the cluster as
/// the threshold tightened.</item>
/// <item>Score each surviving cluster by <b>stability</b>, the total density range over which its
/// points persisted, and choose the set of non-overlapping clusters maximising it.</item>
/// </list>
/// <para>
/// The result is that <c>minClusterSize</c> — "how many points make a cluster worth the name" — is
/// the only parameter that really needs an answer, and unlike <c>eps</c> it is a question about the
/// problem rather than about the data's scale.
/// </para>
/// <para>
/// Cost is O(n²) in memory and time: the pairwise distance matrix is materialised. Real
/// implementations use a space tree to avoid that, which pays off above a few thousand points and
/// stops helping in high dimensions anyway.
/// </para>
/// </remarks>
public sealed class Hdbscan
{
    /// <summary>Creates a clusterer.</summary>
    /// <param name="minClusterSize">
    /// The smallest group worth calling a cluster. Anything smaller is absorbed into noise.
    /// </param>
    /// <param name="minSamples">
    /// Neighbours used for the core-distance estimate. Larger values are more conservative — more
    /// points end up as noise. Defaults to <paramref name="minClusterSize"/>.
    /// </param>
    public Hdbscan(int minClusterSize = 5, int? minSamples = null)
    {
        if (minClusterSize < 2) throw new ArgumentOutOfRangeException(nameof(minClusterSize),
            "A cluster needs at least two points.");

        MinClusterSize = minClusterSize;
        MinSamples = minSamples ?? minClusterSize;

        if (MinSamples < 1) throw new ArgumentOutOfRangeException(nameof(minSamples));
    }

    /// <summary>The smallest group counted as a cluster.</summary>
    public int MinClusterSize { get; }

    /// <summary>Neighbours used for the core-distance estimate.</summary>
    public int MinSamples { get; }

    /// <summary>Cluster assignment per training point; <c>-1</c> means noise.</summary>
    public NdArray Labels { get; private set; } = NdArray.Zeros(0);

    /// <summary>How many clusters were found, not counting noise.</summary>
    public int ClusterCount { get; private set; }

    /// <summary>Each point's distance to its <see cref="MinSamples"/>-th nearest neighbour.</summary>
    /// <remarks>A local density estimate, and useful on its own for spotting sparse regions.</remarks>
    public NdArray CoreDistances { get; private set; } = NdArray.Zeros(0);

    /// <summary>
    /// Per-point membership strength in [0, 1]; zero for noise.
    /// </summary>
    /// <remarks>
    /// How deep inside its cluster a point sits — 1 for the densest core, falling towards the edge.
    /// This is what the flat labels throw away, and it is often the more useful output: a point at
    /// 0.05 is nominally clustered and practically indistinguishable from noise.
    /// </remarks>
    public NdArray Probabilities { get; private set; } = NdArray.Zeros(0);

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Clusters <paramref name="x"/> and fills <see cref="Labels"/>.</summary>
    public Hdbscan Fit(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank != 2) throw new ArgumentException("x must be a rank 2 array of shape (samples, features).");

        var n = x.Shape[0];
        if (n == 0) throw new ArgumentException("There is nothing to cluster.");

        if (n <= MinClusterSize)
        {
            // Not enough points for any cluster to qualify, so everything is noise. Saying so is
            // better than returning one cluster that the parameters explicitly rule out.
            Labels = NdArray.Full(-1, n);
            Probabilities = NdArray.Zeros(n);
            CoreDistances = NdArray.Zeros(n);
            ClusterCount = 0;
            IsFitted = true;
            return this;
        }

        var distances = PairwiseDistances(x, n);
        var core = ComputeCoreDistances(distances, n);
        var edges = MinimumSpanningTree(distances, core, n);

        Array.Sort(edges, (a, b) => a.Weight.CompareTo(b.Weight));

        var tree = BuildHierarchy(edges, n);
        var condensed = Condense(tree, n);
        var selected = SelectClusters(condensed);

        AssignLabels(condensed, selected, n);

        CoreDistances = NdArray.FromValues(core);
        IsFitted = true;
        return this;
    }

    /// <summary>Convenience: fit and return the labels in one call.</summary>
    public NdArray FitPredict(NdArray x) => Fit(x).Labels;

    // ---------------------------------------------------------------- density

    private static double[,] PairwiseDistances(NdArray x, int n)
    {
        var features = x.Shape[1];
        var distances = new double[n, n];

        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var sum = 0.0;
                for (var f = 0; f < features; f++)
                {
                    var d = x[i, f] - x[j, f];
                    sum += d * d;
                }

                var distance = Math.Sqrt(sum);
                distances[i, j] = distance;
                distances[j, i] = distance;
            }

        return distances;
    }

    /// <summary>Distance to the k-th nearest neighbour, counting the point itself as the first.</summary>
    private double[] ComputeCoreDistances(double[,] distances, int n)
    {
        var core = new double[n];
        var row = new double[n];

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++) row[j] = distances[i, j];
            Array.Sort(row);

            // row[0] is the point's own zero distance, so the k-th neighbour is at index k.
            core[i] = row[Math.Min(MinSamples, n - 1)];
        }

        return core;
    }

    /// <summary>
    /// Prim's algorithm over the mutual reachability metric.
    /// </summary>
    /// <remarks>
    /// The mutual reachability distance is built on demand rather than stored: it is a simple
    /// function of the raw distance and two core distances, and materialising a second n×n matrix
    /// to hold it would double the memory for nothing.
    /// </remarks>
    private static Edge[] MinimumSpanningTree(double[,] distances, double[] core, int n)
    {
        var inTree = new bool[n];
        var best = new double[n];
        var from = new int[n];

        Array.Fill(best, double.PositiveInfinity);
        Array.Fill(from, -1);

        best[0] = 0;
        var edges = new List<Edge>(n - 1);

        for (var step = 0; step < n; step++)
        {
            var next = -1;
            var cheapest = double.PositiveInfinity;

            for (var i = 0; i < n; i++)
                if (!inTree[i] && best[i] < cheapest) { cheapest = best[i]; next = i; }

            if (next < 0) break;
            inTree[next] = true;

            if (from[next] >= 0) edges.Add(new Edge(from[next], next, best[next]));

            for (var i = 0; i < n; i++)
            {
                if (inTree[i]) continue;

                var reachability = Math.Max(distances[next, i], Math.Max(core[next], core[i]));
                if (reachability < best[i]) { best[i] = reachability; from[i] = next; }
            }
        }

        return [.. edges];
    }

    // ---------------------------------------------------------------- hierarchy

    private readonly record struct Edge(int Left, int Right, double Weight);

    /// <summary>A merge in the single-linkage hierarchy: two children joining at a distance.</summary>
    private readonly record struct Merge(int Left, int Right, double Distance, int Size);

    /// <summary>
    /// Replays the sorted MST edges through a union-find to build the merge hierarchy.
    /// </summary>
    /// <remarks>
    /// Nodes <c>0..n-1</c> are the original points; each merge creates a new node numbered
    /// <c>n + k</c>. That is the standard dendrogram encoding, and it is what makes the condense
    /// step a single downward walk.
    /// </remarks>
    private static Merge[] BuildHierarchy(Edge[] edges, int n)
    {
        var parent = new int[2 * n];
        var size = new int[2 * n];
        var component = new int[2 * n];

        for (var i = 0; i < 2 * n; i++) { parent[i] = i; size[i] = 1; component[i] = i; }

        var merges = new List<Merge>(edges.Length);
        var nextNode = n;

        foreach (var edge in edges)
        {
            var left = component[Find(parent, edge.Left)];
            var right = component[Find(parent, edge.Right)];
            if (left == right) continue;

            merges.Add(new Merge(left, right, edge.Weight, size[left] + size[right]));

            var rootLeft = Find(parent, edge.Left);
            var rootRight = Find(parent, edge.Right);
            parent[rootRight] = rootLeft;

            size[rootLeft] += size[rootRight];
            component[rootLeft] = nextNode;
            size[nextNode] = size[rootLeft];
            nextNode++;
        }

        return [.. merges];
    }

    private static int Find(int[] parent, int node)
    {
        while (parent[node] != node)
        {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        return node;
    }

    // ---------------------------------------------------------------- condensing

    /// <summary>A point or subcluster leaving a cluster at a given density level.</summary>
    /// <param name="Parent">The cluster it fell out of.</param>
    /// <param name="Child">The point (below n) or subcluster node that left.</param>
    /// <param name="Lambda">The density level it left at — the reciprocal of the merge distance.</param>
    /// <param name="Size">How many points left together.</param>
    private readonly record struct Fall(int Parent, int Child, double Lambda, int Size);

    private sealed class CondensedTree
    {
        public List<Fall> Falls { get; } = [];
        public Dictionary<int, double> Births { get; } = [];
        public int Root { get; set; }
    }

    /// <summary>
    /// Turns the full dendrogram into the condensed tree, discarding splits that are just points
    /// falling away.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is where <see cref="MinClusterSize"/> does its work. Descending from the root, each
    /// merge is read backwards as a split. If both sides are large enough, it is a genuine
    /// bifurcation and two new clusters are born. If only one side is, the cluster continues and
    /// the small side is recorded as points falling out. If neither is, the cluster dies.
    /// </para>
    /// <para>
    /// Working in lambda — the reciprocal of distance — rather than distance is what makes the
    /// stability sum below meaningful: lambda rises as the threshold tightens, so "how long a
    /// cluster survived" becomes an area rather than an interval that has to be read backwards.
    /// </para>
    /// </remarks>
    private CondensedTree Condense(Merge[] merges, int n)
    {
        var tree = new CondensedTree();
        if (merges.Length == 0) return tree;

        // Look-up from dendrogram node to the merge that created it.
        var creator = new Dictionary<int, int>();
        for (var i = 0; i < merges.Length; i++) creator[n + i] = i;

        var root = n + merges.Length - 1;
        tree.Root = root;
        tree.Births[root] = 0.0;

        var relabel = new Dictionary<int, int> { [root] = root };
        var nextCluster = root + 1;

        var stack = new Stack<int>();
        stack.Push(root);

        while (stack.Count > 0)
        {
            var node = stack.Pop();
            var cluster = relabel[node];

            if (!creator.TryGetValue(node, out var index)) continue;

            var merge = merges[index];
            var lambda = merge.Distance > 0 ? 1.0 / merge.Distance : double.PositiveInfinity;

            var leftSize = SubtreeSize(merge.Left, creator, merges, n);
            var rightSize = SubtreeSize(merge.Right, creator, merges, n);

            var leftBig = leftSize >= MinClusterSize;
            var rightBig = rightSize >= MinClusterSize;

            if (leftBig && rightBig)
            {
                // A genuine split: two clusters are born and the parent ends here.
                foreach (var (child, childSize) in new[] { (merge.Left, leftSize), (merge.Right, rightSize) })
                {
                    var born = nextCluster++;
                    relabel[child] = born;
                    tree.Births[born] = lambda;
                    tree.Falls.Add(new Fall(cluster, born, lambda, childSize));
                    stack.Push(child);
                }
            }
            else if (leftBig || rightBig)
            {
                // One side is just points falling away; the cluster carries on as the larger side.
                var survivor = leftBig ? merge.Left : merge.Right;
                var lost = leftBig ? merge.Right : merge.Left;

                foreach (var point in Points(lost, creator, merges, n))
                    tree.Falls.Add(new Fall(cluster, point, lambda, 1));

                relabel[survivor] = cluster;
                stack.Push(survivor);
            }
            else
            {
                // Neither side survives: the cluster dissolves here and all its points fall out.
                foreach (var point in Points(node, creator, merges, n))
                    tree.Falls.Add(new Fall(cluster, point, lambda, 1));
            }
        }

        return tree;
    }

    private static int SubtreeSize(int node, Dictionary<int, int> creator, Merge[] merges, int n)
        => node < n ? 1 : merges[creator[node]].Size;

    /// <summary>Every original point beneath a dendrogram node.</summary>
    private static IEnumerable<int> Points(int node, Dictionary<int, int> creator, Merge[] merges, int n)
    {
        var stack = new Stack<int>();
        stack.Push(node);

        while (stack.Count > 0)
        {
            var current = stack.Pop();
            if (current < n) { yield return current; continue; }

            var merge = merges[creator[current]];
            stack.Push(merge.Left);
            stack.Push(merge.Right);
        }
    }

    // ---------------------------------------------------------------- selection

    /// <summary>
    /// Chooses the set of non-overlapping clusters with the greatest total stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A cluster's stability is <c>Σ (λ_fall − λ_birth)</c> over the points that left it: how much
    /// density range it survived, weighted by how many points survived that long. A cluster that
    /// persists across a wide range of thresholds scores high; one that appears and immediately
    /// shatters scores low.
    /// </para>
    /// <para>
    /// The choice is then a single bottom-up pass. A parent is kept only if it beats the combined
    /// stability of its descendants; otherwise the descendants are kept and the parent is
    /// discarded. Because it runs bottom-up, a parent's score is compared against the best its
    /// subtree can do rather than against its immediate children alone — which is what lets a
    /// three-level hierarchy resolve correctly.
    /// </para>
    /// </remarks>
    private static HashSet<int> SelectClusters(CondensedTree tree)
    {
        var selected = new HashSet<int>();
        if (tree.Falls.Count == 0) return selected;

        var stability = new Dictionary<int, double>();
        foreach (var cluster in tree.Births.Keys) stability[cluster] = 0.0;

        foreach (var fall in tree.Falls)
        {
            var birth = tree.Births.GetValueOrDefault(fall.Parent);
            if (double.IsInfinity(fall.Lambda)) continue;   // duplicate points merge at distance 0
            stability[fall.Parent] = stability.GetValueOrDefault(fall.Parent)
                                     + fall.Size * (fall.Lambda - birth);
        }

        var children = new Dictionary<int, List<int>>();
        foreach (var fall in tree.Falls.Where(f => tree.Births.ContainsKey(f.Child)))
        {
            if (!children.TryGetValue(fall.Parent, out var list)) children[fall.Parent] = list = [];
            list.Add(fall.Child);
        }

        // Deepest first, so a node is decided only after its subtree is.
        var order = tree.Births.Keys.OrderByDescending(c => c).ToList();
        var subtreeBest = new Dictionary<int, double>();

        foreach (var cluster in order)
        {
            var own = stability.GetValueOrDefault(cluster);
            var descendants = children.GetValueOrDefault(cluster, [])
                .Sum(child => subtreeBest.GetValueOrDefault(child));

            if (descendants > own)
            {
                // The children are collectively the better explanation; keep whatever they chose.
                subtreeBest[cluster] = descendants;
            }
            else
            {
                subtreeBest[cluster] = own;
                selected.Add(cluster);

                foreach (var descendant in Descendants(cluster, children)) selected.Remove(descendant);
            }
        }

        // The root is never a cluster: it holds everything, which explains nothing.
        selected.Remove(tree.Root);
        return selected;
    }

    private static IEnumerable<int> Descendants(int cluster, Dictionary<int, List<int>> children)
    {
        var stack = new Stack<int>(children.GetValueOrDefault(cluster, []));
        while (stack.Count > 0)
        {
            var current = stack.Pop();
            yield return current;
            foreach (var child in children.GetValueOrDefault(current, [])) stack.Push(child);
        }
    }

    /// <summary>Walks the selected clusters down to the points beneath them.</summary>
    private void AssignLabels(CondensedTree tree, HashSet<int> selected, int n)
    {
        var labels = NdArray.Full(-1, n);
        var probabilities = NdArray.Zeros(n);

        var childrenOf = new Dictionary<int, List<Fall>>();
        foreach (var fall in tree.Falls)
        {
            if (!childrenOf.TryGetValue(fall.Parent, out var list)) childrenOf[fall.Parent] = list = [];
            list.Add(fall);
        }

        var label = 0;
        foreach (var cluster in selected.OrderBy(c => c))
        {
            var members = new List<(int Point, double Lambda)>();
            var stack = new Stack<int>();
            stack.Push(cluster);

            while (stack.Count > 0)
            {
                var node = stack.Pop();
                foreach (var fall in childrenOf.GetValueOrDefault(node, []))
                {
                    if (fall.Child < n) members.Add((fall.Child, fall.Lambda));
                    else stack.Push(fall.Child);
                }
            }

            if (members.Count == 0) continue;

            // Membership strength is how late a point fell out relative to the cluster's most
            // persistent member: the points that held on longest sit deepest in the density peak.
            var finite = members.Where(m => !double.IsInfinity(m.Lambda)).Select(m => m.Lambda).ToArray();
            var deepest = finite.Length > 0 ? finite.Max() : 1.0;

            foreach (var (point, lambda) in members)
            {
                labels.SetAt(point, label);
                probabilities.SetAt(point,
                    deepest > 0 ? Math.Min(1.0, (double.IsInfinity(lambda) ? deepest : lambda) / deepest) : 1.0);
            }

            label++;
        }

        Labels = labels;
        Probabilities = probabilities;
        ClusterCount = label;
    }
}
