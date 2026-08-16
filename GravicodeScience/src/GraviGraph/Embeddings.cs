using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Embeddings;

namespace Gravicode.Science.GraviGraph.Embeddings;

/// <summary>Random walk generation over a graph.</summary>
public static class RandomWalks
{
    /// <summary>Uniform random walks: at each step, pick a neighbour at random.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> Uniform(Graph graph, int walksPerNode = 10,
        int walkLength = 80, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var walks = new List<IReadOnlyList<string>>(graph.NodeCount * walksPerNode);

        for (var round = 0; round < walksPerNode; round++)
            foreach (var start in rng.Permutation(graph.NodeCount))
            {
                var walk = new List<string> { start.ToString() };
                var current = start;
                for (var step = 1; step < walkLength; step++)
                {
                    var neighbours = graph.Neighbors(current);
                    if (neighbours.Count == 0) break;
                    current = neighbours[rng.Next(neighbours.Count)].Target;
                    walk.Add(current.ToString());
                }
                walks.Add(walk);
            }
        return walks;
    }

    /// <summary>
    /// Second-order biased walks, the node2vec sampling strategy.
    /// </summary>
    /// <remarks>
    /// The two parameters control what the walk explores. A small <paramref name="q"/> pushes the
    /// walk outward, producing depth-first walks whose embeddings capture communities
    /// (homophily); a large <paramref name="q"/> keeps it near the origin, producing
    /// breadth-first walks whose embeddings capture structural roles - a hub in one part of the
    /// graph ends up near a hub elsewhere. <paramref name="p"/> controls how eagerly the walk
    /// backtracks. This bias is the entire difference between node2vec and DeepWalk.
    /// </remarks>
    public static IReadOnlyList<IReadOnlyList<string>> Biased(Graph graph, double p = 1.0, double q = 1.0,
        int walksPerNode = 10, int walkLength = 80, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var neighbourSets = new HashSet<int>[graph.NodeCount];
        for (var i = 0; i < graph.NodeCount; i++)
            neighbourSets[i] = [.. graph.Neighbors(i).Select(n => n.Target)];

        var walks = new List<IReadOnlyList<string>>(graph.NodeCount * walksPerNode);

        for (var round = 0; round < walksPerNode; round++)
            foreach (var start in rng.Permutation(graph.NodeCount))
            {
                var walk = new List<int> { start };
                while (walk.Count < walkLength)
                {
                    var current = walk[^1];
                    var neighbours = graph.Neighbors(current);
                    if (neighbours.Count == 0) break;

                    if (walk.Count == 1)
                    {
                        walk.Add(neighbours[rng.Next(neighbours.Count)].Target);
                        continue;
                    }

                    var previous = walk[^2];
                    var weights = new double[neighbours.Count];
                    for (var i = 0; i < neighbours.Count; i++)
                    {
                        var target = neighbours[i].Target;
                        var bias = target == previous ? 1.0 / p                       // step back
                            : neighbourSets[previous].Contains(target) ? 1.0          // stay local
                            : 1.0 / q;                                                // move outward
                        weights[i] = neighbours[i].Weight * bias;
                    }
                    walk.Add(neighbours[rng.Categorical(weights)].Target);
                }
                walks.Add(walk.Select(n => n.ToString()).ToList());
            }
        return walks;
    }
}

/// <summary>
/// Node embeddings learned by treating random walks as sentences and running skip-gram over them.
/// </summary>
/// <remarks>
/// DeepWalk's insight is that node sequences from random walks have the same power-law statistics
/// as natural language, so a word embedding model applies directly: nodes that co-occur in walks
/// end up with similar vectors. node2vec is the same pipeline with biased walks - see
/// <see cref="RandomWalks.Biased"/> for what the bias buys.
/// </remarks>
public sealed class DeepWalk(
    int dimensions = 128,
    int walksPerNode = 10,
    int walkLength = 40,
    int windowSize = 5,
    int epochs = 5,
    int seed = 42)
{
    /// <summary>Learns one vector per node.</summary>
    public NodeEmbeddings Train(Graph graph)
    {
        var walks = RandomWalks.Uniform(graph, walksPerNode, walkLength, seed);
        return Learn(graph, walks, dimensions, windowSize, epochs, seed);
    }

    internal static NodeEmbeddings Learn(Graph graph, IReadOnlyList<IReadOnlyList<string>> walks,
        int dimensions, int windowSize, int epochs, int seed)
    {
        var word2vec = new Word2Vec(dimensions, windowSize, minCount: 1, negativeSamples: 5,
            learningRate: 0.025, epochs: epochs, seed: seed);
        var embeddings = word2vec.Train(walks);

        var matrix = NdArray.Zeros(graph.NodeCount, dimensions);
        for (var node = 0; node < graph.NodeCount; node++)
        {
            var vector = embeddings.TryGet(node.ToString());
            if (vector is null) continue;   // isolated nodes never appear in a walk
            for (var d = 0; d < dimensions; d++) matrix[node, d] = vector.At(d);
        }
        return new NodeEmbeddings(matrix, graph);
    }
}

/// <summary>node2vec: DeepWalk with second-order biased walks.</summary>
public sealed class Node2Vec(
    int dimensions = 128,
    double p = 1.0,
    double q = 1.0,
    int walksPerNode = 10,
    int walkLength = 40,
    int windowSize = 5,
    int epochs = 5,
    int seed = 42)
{
    /// <summary>Return parameter: larger values discourage backtracking.</summary>
    public double P { get; } = p;

    /// <summary>In-out parameter: below 1 explores outward, above 1 stays local.</summary>
    public double Q { get; } = q;

    /// <summary>Learns one vector per node.</summary>
    public NodeEmbeddings Train(Graph graph)
    {
        var walks = RandomWalks.Biased(graph, P, Q, walksPerNode, walkLength, seed);
        return DeepWalk.Learn(graph, walks, dimensions, windowSize, epochs, seed);
    }
}

/// <summary>A learned vector per node, with lookup and similarity helpers.</summary>
public sealed class NodeEmbeddings(NdArray vectors, Graph graph)
{
    /// <summary>The embedding matrix, one row per node.</summary>
    public NdArray Vectors { get; } = vectors;

    /// <summary>Number of nodes.</summary>
    public int NodeCount => Vectors.Shape[0];

    /// <summary>Vector length.</summary>
    public int Dimensions => Vectors.Shape[1];

    /// <summary>The vector for one node.</summary>
    public NdArray this[int node] => Vectors.Row(node).Copy();

    /// <summary>Cosine similarity between two nodes' vectors.</summary>
    public double Similarity(int a, int b)
        => GraviText.Vectorization.Similarity.Cosine(this[a], this[b]);

    /// <summary>The nodes whose vectors are closest to <paramref name="node"/>.</summary>
    public IReadOnlyList<(int Node, string Name, double Score)> MostSimilar(int node, int top = 10)
        => Enumerable.Range(0, NodeCount)
            .Where(i => i != node)
            .Select(i => (Node: i, Name: graph.NodeNames[i], Score: Similarity(node, i)))
            .OrderByDescending(t => t.Score)
            .Take(top)
            .ToList();

    /// <summary>
    /// Scores an unobserved edge by the similarity of its endpoints, the standard link-prediction
    /// use of node embeddings.
    /// </summary>
    public double LinkScore(int source, int target) => Similarity(source, target);

    /// <summary>Writes the vectors in the word2vec text format, keyed by node name.</summary>
    public void Save(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine($"{NodeCount} {Dimensions}");
        for (var i = 0; i < NodeCount; i++)
        {
            var values = string.Join(' ', Enumerable.Range(0, Dimensions).Select(d => Vectors[i, d].ToString("G9")));
            writer.WriteLine($"{graph.NodeNames[i]} {values}");
        }
    }
}
