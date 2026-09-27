using System.Collections.Concurrent;
using System.Text.Json;
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Framework.Nodes;

namespace MediaPipeNet.Framework.Tests;

/// <summary>Back edges, executors, tracing and subgraphs (0.3 Graph API).</summary>
public class GraphFeatureTests
{
    [Fact]
    public async Task Back_edge_feeds_the_previous_result_into_the_next_timestamp()
    {
        // running sum: out[t] = in[t] + out[t-1], closed through a loopback back edge.
        var builder = new GraphBuilder().AddInputStream<int>("in");
        builder.AddNode("prev", new PreviousLoopbackNode<int>()).In("MAIN", "in").In("LOOP", "sum", backEdge: true).Out("PREV_LOOP", "prev_sum");
        builder.AddNode("add", new CombineNode<int, int, int>((a, b) => a + b)).In("A", "in").In("B", "prev_sum").Out("OUT", "sum");
        builder.AddOutputStream("sum");
        await using var graph = builder.Build();
        var sums = new List<int>();
        graph.ObserveOutputStream<int>("sum", p => sums.Add(p.Value));
        await graph.StartAsync();
        for (int t = 1; t <= 5; t++)
        {
            graph.AddPacket("in", t, t);
            await graph.WaitUntilIdleAsync(); // sequential frames: the loop value is always ready
        }
        await graph.CloseAsync();
        sums.Should().Equal(1, 3, 6, 10, 15);
        graph.State.Should().Be(GraphState.Done);
    }

    [Fact]
    public void A_cycle_without_back_edge_is_rejected_with_a_hint()
    {
        var builder = new GraphBuilder().AddInputStream<int>("in");
        builder.AddNode("prev", new PreviousLoopbackNode<int>()).In("MAIN", "in").In("LOOP", "sum").Out("PREV_LOOP", "prev_sum");
        builder.AddNode("add", new CombineNode<int, int, int>((a, b) => a + b)).In("A", "in").In("B", "prev_sum").Out("OUT", "sum");
        var act = () => builder.Build();
        act.Should().Throw<GraphValidationException>().WithMessage("*back edge*");
    }

    [Fact]
    public async Task Executor_runs_nodes_on_its_dedicated_threads()
    {
        var threads = new ConcurrentBag<string?>();
        var builder = new GraphBuilder().AddInputStream<int>("in").AddExecutor("serial", threads: 1);
        builder.AddNode("work", new LambdaNode<int, int>(x =>
        {
            threads.Add(Thread.CurrentThread.Name);
            return x;
        })).In("IN", "in").Out("OUT", "out").OnExecutor("serial");
        builder.AddOutputStream("out");
        await using var graph = builder.Build();
        await graph.StartAsync();
        for (int t = 0; t < 20; t++) graph.AddPacket("in", t, t);
        await graph.CloseAsync();
        threads.Should().HaveCount(20).And.OnlyContain(n => n == "mpnet-executor-serial-0");
    }

    [Fact]
    public void Unknown_executor_is_rejected()
    {
        var builder = new GraphBuilder().AddInputStream<int>("in");
        builder.AddNode("n", new PassThroughNode<int>()).In("IN", "in").Out("OUT", "out").OnExecutor("gpu");
        var act = () => builder.Build();
        act.Should().Throw<GraphValidationException>().WithMessage("*executor 'gpu'*");
    }

    [Fact]
    public async Task Tracing_records_every_node_run_and_exports_chrome_json()
    {
        var builder = new GraphBuilder { Options = new GraphOptions { EnableTracing = true } }.AddInputStream<int>("in");
        builder.AddNode("double", new LambdaNode<int, int>(x => x * 2)).In("IN", "in").Out("OUT", "doubled");
        builder.AddNode("inc", new LambdaNode<int, int>(x => x + 1)).In("IN", "doubled").Out("OUT", "out");
        builder.AddOutputStream("out");
        await using var graph = builder.Build();
        await graph.StartAsync();
        for (int t = 0; t < 3; t++) graph.AddPacket("in", t, t * 1000);
        await graph.CloseAsync();

        var events = graph.GetTraceEvents();
        events.Count(e => e.Kind == GraphTraceKind.Process).Should().Be(6);
        events.Where(e => e.Node == "inc" && e.Kind == GraphTraceKind.Process).Select(e => e.Timestamp).Should().BeEquivalentTo(new long[] { 0, 1_000_000, 2_000_000 });
        events.Should().Contain(e => e.Kind == GraphTraceKind.Open).And.Contain(e => e.Kind == GraphTraceKind.Close);

        using var ms = new MemoryStream();
        graph.WriteChromeTrace(ms);
        using var doc = JsonDocument.Parse(ms.ToArray());
        var trace = doc.RootElement.GetProperty("traceEvents");
        trace.GetArrayLength().Should().Be(events.Count);
        trace[0].GetProperty("ph").GetString().Should().Be("X");
    }

    [Fact]
    public async Task Subgraph_is_inlined_with_prefixed_names()
    {
        static GraphBuilder AffineSubgraph(int scale, int offset)
        {
            var sub = new GraphBuilder().AddInputStream<int>("x").AddOutputStream("y");
            sub.AddNode("scale", new LambdaNode<int, int>(v => v * scale)).In("IN", "x").Out("OUT", "scaled");
            sub.AddNode("offset", new LambdaNode<int, int>(v => v + offset)).In("IN", "scaled").Out("OUT", "y");
            return sub;
        }
        var builder = new GraphBuilder().AddInputStream<int>("in").AddOutputStream("out");
        builder.AddSubgraph("first", AffineSubgraph(2, 1), new Dictionary<string, string> { ["x"] = "in" }, new Dictionary<string, string> { ["y"] = "mid" });
        builder.AddSubgraph("second", AffineSubgraph(10, 0), new Dictionary<string, string> { ["x"] = "mid" }, new Dictionary<string, string> { ["y"] = "out" });
        await using var graph = builder.Build();
        graph.NodeNames.Should().Contain(["first/scale", "first/offset", "second/scale", "second/offset"]);
        graph.StreamNames.Should().Contain("first/scaled");
        var results = new List<int>();
        graph.ObserveOutputStream<int>("out", p => results.Add(p.Value));
        await graph.StartAsync();
        graph.AddPacket("in", 3, 0);
        graph.AddPacket("in", 5, 1);
        await graph.CloseAsync();
        results.Should().Equal(70, 110);
    }

    [Fact]
    public void A_subgraph_builder_can_only_be_inlined_once()
    {
        var sub = new GraphBuilder().AddInputStream<int>("x").AddOutputStream("y");
        sub.AddNode("n", new PassThroughNode<int>()).In("IN", "x").Out("OUT", "y");
        var builder = new GraphBuilder().AddInputStream<int>("in");
        var inputs = new Dictionary<string, string> { ["x"] = "in" };
        builder.AddSubgraph("a", sub, inputs, new Dictionary<string, string> { ["y"] = "a_out" });
        var act = () => builder.AddSubgraph("b", sub, inputs, new Dictionary<string, string> { ["y"] = "b_out" });
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Pbtxt_supports_subgraphs_back_edges_executors_and_profiling()
    {
        var registry = new CalculatorRegistry()
            .Register("PassThroughCalculator", _ => new PassThroughNode<object>())
            .Register("PreviousLoopbackCalculator", _ => new PreviousLoopbackNode<object>())
            .RegisterSubgraph("RelaySubgraph", GraphConfig.ParsePbtxt("""
                input_stream: "IN:x"
                output_stream: "OUT:y"
                node { calculator: "PassThroughCalculator" input_stream: "IN:x" output_stream: "OUT:y" }
                """));
        var config = GraphConfig.ParsePbtxt("""
            input_stream: "frames"
            output_stream: "relayed"
            output_stream: "previous"
            profiler_config { trace_enabled: true }
            executor { name: "io" type: "ThreadPoolExecutor" options { [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 2 } } }
            node { calculator: "RelaySubgraph" name: "relay" input_stream: "IN:frames" output_stream: "OUT:relayed" }
            node {
              calculator: "PreviousLoopbackCalculator"
              executor: "io"
              input_stream: "MAIN:frames"
              input_stream: "LOOP:relayed"
              input_stream_info { tag_index: "LOOP" back_edge: true }
              output_stream: "PREV_LOOP:previous"
            }
            """);
        config.Executors.Should().ContainKey("io").WhoseValue.Should().Be(2);
        config.EnableTracing.Should().BeTrue();
        config.Nodes[1].BackEdgeTags.Should().Contain("LOOP");
        config.Nodes[1].Executor.Should().Be("io");
        GraphConfig.ParsePbtxt(config.ToPbtxt()).Nodes[1].BackEdgeTags.Should().Contain("LOOP"); // round trip

        await using var graph = config.Build(registry);
        graph.NodeNames.Should().Contain("relay/PassThroughCalculator_0");
        graph.IsTracing.Should().BeTrue();
        var relayed = new List<object>();
        graph.ObserveOutputStream<object>("relayed", p => relayed.Add(p.Value));
        await graph.StartAsync();
        graph.AddPacket("frames", (object)"a", 1);
        await graph.WaitUntilIdleAsync();
        graph.AddPacket("frames", (object)"b", 2);
        await graph.CloseAsync();
        relayed.Should().Equal("a", "b");
        graph.GetTraceEvents().Should().NotBeEmpty();
    }
}
