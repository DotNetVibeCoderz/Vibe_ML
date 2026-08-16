using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviGraph.Embeddings;
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;

BenchmarkSwitcher.FromAssembly(typeof(GcnTrainingBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(1).WithIterationCount(3))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// GCN training cost as the graph grows toward 100k nodes.
/// </summary>
/// <remarks>
/// Each layer is a sparse propagation followed by a dense projection. The sparse half costs
/// <c>O(edges * width)</c> and the dense half <c>O(nodes * features * width)</c>, so on a real
/// sparse network the dense projection dominates - which is the part a GPU would accelerate.
/// This benchmark separates the two so the split is visible rather than assumed.
/// </remarks>
[MemoryDiagnoser]
public class GcnTrainingBenchmark
{
    private Graph _graph = new();
    private NdArray _features = NdArray.Zeros(1, 1);
    private int[] _train = [];

    /// <summary>Number of nodes in the generated graph.</summary>
    [Params(1_000, 10_000, 100_000)]
    public int Nodes { get; set; }

    /// <summary>Builds a scale-free graph with random node features.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _graph = Graph.ScaleFree(Nodes, edgesPerNode: 3, seed: 42);

        var rng = new GraviRandom(42);
        _features = rng.StandardNormal(Nodes, 16);
        for (var i = 0; i < Nodes; i++) _graph.NodeLabels[i] = i % 4;

        _train = rng.Permutation(Nodes).Take(Math.Min(500, Nodes / 4)).ToArray();

        Console.WriteLine($"[setup] {_graph}");
        Console.WriteLine($"[setup] {Compute.DescribeDevices()}");
    }

    /// <summary>Ten epochs of GCN training, forward and backward.</summary>
    [Benchmark(Baseline = true)]
    public double TrainTenEpochs()
    {
        var model = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.05, epochs: 10, seed: 42);
        model.Train(_graph, _train, features: _features);
        return model.History!.Loss[^1];
    }

    /// <summary>The sparse propagation alone, once.</summary>
    [Benchmark]
    public double PropagationOnly()
    {
        var propagation = _graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true);
        return propagation.Multiply(_features, denseIsMatrix: true)[0, 0];
    }

    /// <summary>Building the normalised adjacency matrix, which happens once per training run.</summary>
    [Benchmark]
    public int BuildPropagationMatrix()
        => _graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true).NonZeroCount;
}

/// <summary>The three GNN architectures at a fixed graph size.</summary>
[MemoryDiagnoser]
public class ArchitectureBenchmark
{
    private Graph _graph = new();
    private NdArray _features = NdArray.Zeros(1, 1);
    private int[] _train = [];

    /// <summary>Number of nodes.</summary>
    [Params(2_000)]
    public int Nodes { get; set; }

    /// <summary>Builds a community graph with informative features.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _graph = Graph.Communities(4, Nodes / 4, internalP: 0.02, externalP: 0.001, seed: 42);

        var rng = new GraviRandom(42);
        _features = NdArray.Zeros(_graph.NodeCount, 16);
        for (var i = 0; i < _graph.NodeCount; i++)
        {
            for (var j = 0; j < 16; j++) _features[i, j] = rng.Normal(0, 0.5);
            _features[i, _graph.NodeLabels[i]] += 1.0;
        }
        _train = rng.Permutation(_graph.NodeCount).Take(200).ToArray();
    }

    /// <summary>Graph convolution: sparse propagate then project.</summary>
    [Benchmark(Baseline = true)]
    public double Gcn()
        => new GraphConvolutionalNetwork(epochs: 10, seed: 42).Train(_graph, _train, features: _features)
            .History!.Loss[^1];

    /// <summary>GraphSAGE: concatenate self and neighbourhood mean.</summary>
    [Benchmark]
    public double GraphSage()
        => new GraphSage(epochs: 10, seed: 42).Train(_graph, _train, features: _features).History!.Loss[^1];

    /// <summary>GAT: learned attention over neighbours, four heads.</summary>
    [Benchmark]
    public double Gat()
        => new GraphAttentionNetwork(hiddenSize: 8, heads: 4, epochs: 10, seed: 42)
            .Train(_graph, _train, features: _features).History!.Loss[^1];
}

/// <summary>Classic graph algorithms as the graph grows.</summary>
[MemoryDiagnoser]
public class AlgorithmBenchmark
{
    private Graph _graph = new();

    /// <summary>Number of nodes.</summary>
    [Params(10_000, 100_000)]
    public int Nodes { get; set; }

    /// <summary>Builds a scale-free graph.</summary>
    [GlobalSetup]
    public void Setup() => _graph = Graph.ScaleFree(Nodes, edgesPerNode: 3, seed: 42);

    /// <summary>PageRank to convergence.</summary>
    [Benchmark(Baseline = true)]
    public double PageRank() => GraphAlgorithms.PageRank(_graph).At(0);

    /// <summary>Breadth-first traversal from one node.</summary>
    [Benchmark]
    public int BreadthFirstSearch() => GraphAlgorithms.BreadthFirstSearch(_graph, 0).Count;

    /// <summary>Single-source shortest paths with a binary heap.</summary>
    [Benchmark]
    public double Dijkstra() => GraphAlgorithms.ShortestPaths(_graph, 0).Distances[1];

    /// <summary>Weakly connected components.</summary>
    [Benchmark]
    public int ConnectedComponents() => GraphAlgorithms.ConnectedComponents(_graph).Count;

    /// <summary>Community detection by label propagation.</summary>
    [Benchmark]
    public int LabelPropagation() => GraphAlgorithms.LabelPropagation(_graph, maxIterations: 10, seed: 42).Length;
}

/// <summary>Random-walk embedding cost, which is dominated by the skip-gram training.</summary>
[MemoryDiagnoser]
public class EmbeddingBenchmark
{
    private Graph _graph = new();

    /// <summary>Number of nodes.</summary>
    [Params(500, 2_000)]
    public int Nodes { get; set; }

    /// <summary>Builds an undirected scale-free graph.</summary>
    [GlobalSetup]
    public void Setup() => _graph = Graph.ScaleFree(Nodes, edgesPerNode: 3, seed: 42);

    /// <summary>Generating the walks alone.</summary>
    [Benchmark(Baseline = true)]
    public int UniformWalks() => RandomWalks.Uniform(_graph, walksPerNode: 5, walkLength: 20, seed: 42).Count;

    /// <summary>Second-order biased walks, which inspect the previous step's neighbourhood.</summary>
    [Benchmark]
    public int BiasedWalks() => RandomWalks.Biased(_graph, 1.0, 0.5, 5, 20, seed: 42).Count;

    /// <summary>Walks plus skip-gram training.</summary>
    [Benchmark]
    public int DeepWalk()
        => new DeepWalk(dimensions: 64, walksPerNode: 5, walkLength: 20, epochs: 2, seed: 42).Train(_graph).NodeCount;
}
