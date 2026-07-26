using BlazorML.Core.Designer;

namespace BlazorML.Tests;

public class ExperimentGraphTests
{
    private static GraphNode Node(ExperimentGraph graph, string label)
    {
        var node = new GraphNode { ModuleId = "tf.selectColumns", Label = label };
        graph.Nodes.Add(node);
        return node;
    }

    private static void Connect(ExperimentGraph graph, GraphNode from, GraphNode to) =>
        graph.Edges.Add(new GraphEdge { SourceNodeId = from.Id, TargetNodeId = to.Id });

    [Fact]
    public void TryTopologicalSort_puts_every_node_after_the_ones_it_depends_on()
    {
        var graph = new ExperimentGraph();
        var import = Node(graph, "import");
        var clean = Node(graph, "clean");
        var split = Node(graph, "split");
        var train = Node(graph, "train");

        // Added out of order on purpose: the sort must not depend on insertion order.
        Connect(graph, split, train);
        Connect(graph, import, clean);
        Connect(graph, clean, split);

        Assert.True(graph.TryTopologicalSort(out var ordered, out var error));
        Assert.Null(error);

        var position = ordered.Select((n, i) => (n.Label, i)).ToDictionary(x => x.Label, x => x.i);

        Assert.True(position["import"] < position["clean"]);
        Assert.True(position["clean"] < position["split"]);
        Assert.True(position["split"] < position["train"]);
    }

    [Fact]
    public void TryTopologicalSort_detects_a_cycle_and_names_the_nodes_in_it()
    {
        var graph = new ExperimentGraph();
        var a = Node(graph, "alpha");
        var b = Node(graph, "beta");

        Connect(graph, a, b);
        Connect(graph, b, a);

        Assert.False(graph.TryTopologicalSort(out _, out var error));
        Assert.NotNull(error);
        Assert.Contains("alpha", error);
        Assert.Contains("beta", error);
    }

    [Fact]
    public void TryTopologicalSort_rejects_an_edge_pointing_at_a_node_that_is_gone()
    {
        var graph = new ExperimentGraph();
        var a = Node(graph, "alpha");

        graph.Edges.Add(new GraphEdge { SourceNodeId = a.Id, TargetNodeId = "sudah-dihapus" });

        Assert.False(graph.TryTopologicalSort(out _, out var error));
        Assert.Contains("not on the canvas", error);
    }

    [Fact]
    public void TryTopologicalSort_handles_disconnected_nodes()
    {
        var graph = new ExperimentGraph();
        Node(graph, "lepas-1");
        Node(graph, "lepas-2");

        Assert.True(graph.TryTopologicalSort(out var ordered, out _));
        Assert.Equal(2, ordered.Count);
    }

    [Fact]
    public void Json_round_trip_preserves_nodes_edges_and_parameters()
    {
        var graph = new ExperimentGraph { View = { PanX = 12, PanY = -8, Zoom = 1.4 } };
        var a = Node(graph, "alpha");
        var b = Node(graph, "beta");
        a.Parameters["columns"] = "x, y";
        a.X = 42;
        Connect(graph, a, b);

        var restored = ExperimentGraph.FromJson(graph.ToJson());

        Assert.Equal(2, restored.Nodes.Count);
        Assert.Single(restored.Edges);
        Assert.Equal("x, y", restored.Node(a.Id)!.Parameters["columns"]);
        Assert.Equal(42, restored.Node(a.Id)!.X);
        Assert.Equal(1.4, restored.View.Zoom);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("{}")]
    [InlineData("bukan json sama sekali")]
    public void FromJson_returns_an_empty_graph_rather_than_throwing_on_bad_input(string? json)
    {
        // A graph written by an older version, or corrupted, must not stop the page loading.
        var graph = ExperimentGraph.FromJson(json);

        Assert.Empty(graph.Nodes);
        Assert.Empty(graph.Edges);
    }

    [Fact]
    public void InboundEdges_and_OutboundEdges_only_return_edges_touching_that_node()
    {
        var graph = new ExperimentGraph();
        var a = Node(graph, "a");
        var b = Node(graph, "b");
        var c = Node(graph, "c");

        Connect(graph, a, b);
        Connect(graph, b, c);

        Assert.Single(graph.InboundEdges(b.Id));
        Assert.Single(graph.OutboundEdges(b.Id));
        Assert.Empty(graph.InboundEdges(a.Id));
        Assert.Empty(graph.OutboundEdges(c.Id));
    }
}
