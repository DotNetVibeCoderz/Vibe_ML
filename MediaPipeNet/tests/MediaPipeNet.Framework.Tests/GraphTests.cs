using System.Collections.Concurrent;
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Config;
using MediaPipeNet.Framework.Nodes;

namespace MediaPipeNet.Framework.Tests;

public class PacketTests
{
    [Fact]
    public void Typed_and_untyped_access()
    {
        var p = Packet.Create(42, 5);
        p.Timestamp.Should().Be(Timestamp.FromMilliseconds(5));
        p.UntypedValue.Should().Be(42);
        p.PayloadType.Should().Be(typeof(int));
        p.As<int>().Should().BeSameAs(p);
        ((Packet)Packet.Create<object>("x", new Timestamp(1))).As<string>().Value.Should().Be("x");
        p.At(new Timestamp(9)).Timestamp.Should().Be(new Timestamp(9));
        p.WithTimestamp(new Timestamp(3)).Value.Should().Be(42);
        p.ToString().Should().Contain("42");
        var bad = () => p.As<string>();
        bad.Should().Throw<InvalidCastException>();
        Packet.Create<string?>(null, new Timestamp(1)).As<object?>().Value.Should().BeNull();
    }
}

public class GraphValidationTests
{
    [Fact]
    public void Rejects_two_producers_for_one_stream()
    {
        var b = new GraphBuilder().AddInputStream<int>("in");
        b.AddNode("a", new PassThroughNode<int>()).In("IN", "in").Out("OUT", "x");
        b.AddNode("b", new PassThroughNode<int>()).In("IN", "in").Out("OUT", "x");
        b.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*more than one producer*");
    }

    [Fact]
    public void Rejects_unknown_streams_ports_and_type_mismatch()
    {
        var unknown = new GraphBuilder();
        unknown.AddNode("a", new PassThroughNode<int>()).In("IN", "missing");
        unknown.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*unknown stream*");

        var port = new GraphBuilder().AddInputStream<int>("in");
        port.AddNode("a", new PassThroughNode<int>()).In("NOPE", "in");
        port.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*no port*");

        var type = new GraphBuilder().AddInputStream<string>("in");
        type.AddNode("a", new PassThroughNode<int>()).In("IN", "in");
        type.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*expects Int32*");

        var required = new GraphBuilder().AddInputStream<int>("in");
        required.AddNode("a", new PassThroughNode<int>());
        required.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*not connected*");

        new GraphBuilder().AddOutputStream("ghost").Invoking(x => x.Build()).Should().Throw<GraphValidationException>();
        var dup = new GraphBuilder().AddInputStream<int>("in");
        dup.Invoking(x => x.AddInputStream<int>("in")).Should().Throw<ArgumentException>();
        dup.AddNode("n", new PassThroughNode<int>());
        dup.Invoking(x => x.AddNode("n", new PassThroughNode<int>())).Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Detects_cycles()
    {
        var b = new GraphBuilder().AddInputStream<int>("in");
        b.AddNode("a", new CombineNode<int, int, int>((x, y) => x + y)).In("A", "in").In("B", "loop").Out("OUT", "mid");
        b.AddNode("b", new PassThroughNode<int>()).In("IN", "mid").Out("OUT", "loop");
        b.Invoking(x => x.Build()).Should().Throw<GraphValidationException>().WithMessage("*cycle*");
    }

    [Fact]
    public void Exposes_topology_and_mermaid()
    {
        var graph = Chain();
        graph.NodeNames.Should().Equal("square", "plus_one");
        graph.StreamNames.Should().Contain(["numbers", "squares", "result"]);
        graph.ToMermaid().Should().StartWith("flowchart LR").And.Contain("squares");
        graph.State.Should().Be(GraphState.Created);
    }

    internal static CalculatorGraph Chain() => new GraphBuilder()
        .AddInputStream<int>("numbers")
        .AddNode("square", new LambdaNode<int, int>(x => x * x)).In("IN", "numbers").Out("OUT", "squares").Graph
        .AddNode("plus_one", new LambdaNode<int, int>(x => x + 1)).In("IN", "squares").Out("OUT", "result").Graph
        .AddOutputStream("result")
        .Build();
}

public class GraphExecutionTests
{
    [Fact]
    public async Task Pipeline_processes_packets_in_order()
    {
        var graph = GraphValidationTests.Chain();
        var results = new ConcurrentQueue<(long, int)>();
        graph.ObserveOutputStream<int>("result", p => results.Enqueue((p.Timestamp.Value, p.Value)));
        await graph.StartAsync();
        for (int i = 1; i <= 50; i++) graph.AddPacket("numbers", Packet.Create(i, new Timestamp(i))).Should().BeTrue();
        graph.CloseAllInputStreams();
        await graph.WaitUntilDoneAsync();
        graph.State.Should().Be(GraphState.Done);
        results.Select(r => r.Item2).Should().Equal(Enumerable.Range(1, 50).Select(i => i * i + 1));
        results.Select(r => r.Item1).Should().BeInAscendingOrder();
        await graph.DisposeAsync();
    }

    [Fact]
    public async Task Synchronizes_inputs_and_propagates_bounds()
    {
        // Evens are filtered out on one branch; the join must still run for every timestamp.
        var joined = new ConcurrentQueue<string>();
        var graph = new GraphBuilder()
            .AddInputStream<int>("in")
            .AddNode("evens_only", new LambdaNode<int, string>(x => x % 2 == 0 ? $"e{x}" : null)).In("IN", "in").Out("OUT", "evens").Graph
            .AddNode("all", new LambdaNode<int, string>(x => $"a{x}")).In("IN", "in").Out("OUT", "alls").Graph
            .AddNode("join", new CombineNode<string, string, string>((a, b) => $"{a}|{b}")).In("A", "alls").In("B", "evens").Out("OUT", "joined").Graph
            .AddOutputStream("joined")
            .Build();
        graph.ObserveOutputStream<string>("joined", p => joined.Enqueue(p.Value));
        await graph.StartAsync();
        for (int i = 1; i <= 4; i++) graph.AddPacket("in", i, i);
        await graph.CloseAsync();
        joined.Should().Equal("a1|", "a2|e2", "a3|", "a4|e4");
    }

    [Fact]
    public async Task Rejects_non_monotonic_timestamps_and_wrong_types()
    {
        var graph = GraphValidationTests.Chain();
        await graph.StartAsync();
        graph.AddPacket("numbers", 1, 10);
        graph.Invoking(g => g.AddPacket("numbers", 2, 10)).Should().Throw<ArgumentException>().WithMessage("*strictly increase*");
        graph.Invoking(g => g.AddPacket("numbers", Packet.Create("x", 20))).Should().Throw<ArgumentException>();
        graph.Invoking(g => g.AddPacket("nope", 1, 30)).Should().Throw<ArgumentException>();
        graph.Invoking(g => g.AddPacket("numbers", Packet.Create(1, Timestamp.Done))).Should().Throw<ArgumentException>();
        graph.CloseInputStream("numbers");
        graph.Invoking(g => g.CloseInputStream("nope")).Should().Throw<ArgumentException>();
        await graph.WaitUntilDoneAsync();
        graph.Invoking(g => g.AddPacket("numbers", 1, 40)).Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public async Task Max_in_flight_drops_packets_while_busy()
    {
        using var gate = new ManualResetEventSlim(false);
        var seen = new ConcurrentQueue<int>();
        var graph = new GraphBuilder { Options = new GraphOptions { MaxInFlight = 1 } }
            .AddInputStream<int>("in")
            .AddNode("slow", new LambdaNode<int, int>(x => { gate.Wait(); return x; })).In("IN", "in").Out("OUT", "out").Graph
            .AddOutputStream("out")
            .Build();
        graph.ObserveOutputStream<int>("out", p => seen.Enqueue(p.Value));
        await graph.StartAsync();
        graph.AddPacket("in", 1, 1).Should().BeTrue();
        graph.InFlightCount.Should().Be(1);
        graph.AddPacket("in", 2, 2).Should().BeFalse();
        graph.AddPacket("in", 3, 3).Should().BeFalse();
        gate.Set();
        await graph.WaitUntilIdleAsync();
        graph.AddPacket("in", 4, 4).Should().BeTrue();
        await graph.CloseAsync();
        seen.Should().Equal(1, 4);
        graph.DroppedPackets.Should().Be(2);
    }

    [Fact]
    public async Task Bounded_queue_drops_oldest()
    {
        using var gate = new ManualResetEventSlim(false);
        using var started = new ManualResetEventSlim(false);
        var seen = new ConcurrentQueue<int>();
        var graph = new GraphBuilder()
            .AddInputStream<int>("in", new GraphInputOptions(MaxQueueSize: 1))
            .AddNode("slow", new LambdaNode<int, int>(x => { started.Set(); gate.Wait(); return x; })).In("IN", "in").Out("OUT", "out").Graph
            .AddOutputStream("out")
            .Build();
        graph.ObserveOutputStream<int>("out", p => seen.Enqueue(p.Value));
        await graph.StartAsync();
        graph.AddPacket("in", 1, 1);
        started.Wait(TimeSpan.FromSeconds(10)).Should().BeTrue(); // the node holds packet 1 and blocks
        for (int i = 2; i <= 5; i++) graph.AddPacket("in", i, i);
        gate.Set();
        await graph.CloseAsync();
        seen.Should().Equal(1, 5);
        graph.DroppedPackets.Should().Be(3);
    }

    [Fact]
    public async Task Side_packets_and_pollers()
    {
        var graph = new GraphBuilder()
            .AddInputStream<int>("in")
            .AddSidePacket<int>("offset")
            .AddNode("add", new AddSideNode()).In("IN", "in").Out("OUT", "out").Side("OFFSET", "offset").Graph
            .AddOutputStream("out")
            .Build();
        var reader = graph.CreateOutputStreamPoller<int>("out");
        await graph.StartAsync(new Dictionary<string, object?> { ["offset"] = 100 });
        graph.AddPacket("in", 1, 1);
        graph.AddPacket("in", 2, 2);
        graph.CloseAllInputStreams();
        var values = new List<int>();
        await foreach (var p in reader.ReadAllAsync()) values.Add(p.Value);
        values.Should().Equal(101, 102);
    }

    [Fact]
    public async Task Missing_or_wrong_side_packets_fail_at_start()
    {
        var build = () => new GraphBuilder()
            .AddInputStream<int>("in")
            .AddNode("add", new AddSideNode()).In("IN", "in").Out("OUT", "out").Side("OFFSET", "offset").Graph
            .Build();
        await build().Invoking(g => g.StartAsync()).Should().ThrowAsync<GraphValidationException>();
        await build().Invoking(g => g.StartAsync(new Dictionary<string, object?> { ["offset"] = "x" })).Should().ThrowAsync<GraphValidationException>();
        var unbound = new GraphBuilder().AddInputStream<int>("in");
        unbound.AddNode("add", new AddSideNode()).In("IN", "in");
        unbound.Invoking(b => b.Build()).Should().Throw<GraphValidationException>().WithMessage("*side packet*");
    }

    [Fact]
    public async Task Node_failure_fails_the_graph()
    {
        var graph = new GraphBuilder()
            .AddInputStream<int>("in")
            .AddNode("boom", new LambdaNode<int, int>(x => x == 2 ? throw new InvalidOperationException("boom") : x)).In("IN", "in").Out("OUT", "out").Graph
            .AddOutputStream("out")
            .Build();
        Exception? raised = null;
        graph.Failed += (_, e) => raised = e;
        await graph.StartAsync();
        graph.AddPacket("in", 1, 1);
        graph.AddPacket("in", 2, 2);
        await graph.Invoking(g => g.WaitUntilDoneAsync()).Should().ThrowAsync<MediaPipeException>().WithMessage("*boom*");
        graph.State.Should().Be(GraphState.Failed);
        graph.Error.Should().NotBeNull();
        raised.Should().NotBeNull();
        graph.Invoking(g => g.AddPacket("in", 3, 3)).Should().Throw<MediaPipeException>();
        await graph.Invoking(g => g.WaitUntilIdleAsync()).Should().ThrowAsync<MediaPipeException>();
    }

    [Fact]
    public async Task Cancel_and_immediate_policy()
    {
        var counter = new PacketCounterNode();
        var graph = new GraphBuilder()
            .AddInputStream<object>("in")
            .AddNode("count", counter).In("IN", "in").Out("COUNT", "count").Graph
            .AddNode("thin", new PacketThinnerNode(TimeSpan.FromMilliseconds(10))).In("IN", "in").Out("OUT", "thinned").Graph
            .AddNode("immediate", new ImmediateNode()).In("A", "in").In("B", "thinned").Out("OUT", "any").Graph
            .AddOutputStream("any")
            .Build();
        var outputs = new ConcurrentQueue<string>();
        graph.ObserveOutputStream<string>("any", p => outputs.Enqueue(p.Value));
        await graph.StartAsync();
        for (int i = 0; i < 5; i++) graph.AddPacket("in", (object)i, i * 5L);
        await graph.CloseAsync();
        counter.Count.Should().Be(5);
        outputs.Should().NotBeEmpty();
        graph.Cancel(); // no-op once done
        graph.State.Should().Be(GraphState.Done);

        var cancelled = GraphValidationTests.Chain();
        await cancelled.StartAsync();
        cancelled.Cancel();
        cancelled.State.Should().Be(GraphState.Cancelled);
    }

    [Fact]
    public async Task Async_lambda_and_sink_nodes()
    {
        var sunk = new ConcurrentQueue<int>();
        var graph = new GraphBuilder()
            .AddInputStream<int>("in")
            .AddNode("double", new AsyncLambdaNode<int, int>(async (x, ct) => { await Task.Yield(); return x * 2; })).In("IN", "in").Out("OUT", "doubled").Graph
            .AddNode("sink", new SinkNode<int>(p => sunk.Enqueue(p.Value))).In("IN", "doubled").Graph
            .Build();
        await graph.StartAsync();
        graph.AddPacket("in", 21, 1);
        await graph.CloseAsync();
        sunk.Should().Equal(42);
    }

    private sealed class AddSideNode : CalculatorNode
    {
        private int _offset;

        public override void GetContract(CalculatorContract contract) =>
            contract.AddInput<int>("IN").AddOutput<int>("OUT").AddInputSidePacket<int>("OFFSET");

        protected override void Open(CalculatorContext context) => _offset = context.GetSidePacket<int>("OFFSET");

        protected override void Process(CalculatorContext context) => context.Send("OUT", context.GetInput<int>("IN") + _offset);
    }

    private sealed class ImmediateNode : CalculatorNode
    {
        public override void GetContract(CalculatorContract contract)
        {
            contract.AddInput<object>("A").AddInput<object>("B").AddOutput<string>("OUT");
            contract.InputPolicy = InputPolicy.Immediate;
        }

        private Timestamp _last = Timestamp.Unset;

        // With the immediate policy a timestamp can be delivered once per input, so emit at most once per timestamp.
        protected override void Process(CalculatorContext context)
        {
            if (context.InputTimestamp <= _last) return;
            _last = context.InputTimestamp;
            context.Send("OUT", $"{(context.HasInput("A") ? "A" : "")}{(context.HasInput("B") ? "B" : "")}");
        }
    }
}

public class GraphConfigTests
{
    private const string Pbtxt = """
        # A MediaPipe-style graph
        input_stream: "in"
        output_stream: "out"
        max_in_flight: 2

        node {
          calculator: "PassThroughCalculator"
          name: "first"
          input_stream: "IN:in"
          output_stream: "OUT:mid"
        }
        node {
          calculator: "PacketThinnerCalculator"
          input_stream: "IN:mid"
          output_stream: "OUT:out"
          options {
            [mediapipe.PacketThinnerCalculatorOptions.ext] {
              period: 10
            }
          }
        }
        """;

    [Fact]
    public void Parses_and_renders_pbtxt()
    {
        var config = GraphConfig.ParsePbtxt(Pbtxt);
        config.InputStreams.Should().Equal("in");
        config.OutputStreams.Should().Equal("out");
        config.MaxInFlight.Should().Be(2);
        config.Nodes.Should().HaveCount(2);
        config.Nodes[0].Name.Should().Be("first");
        config.Nodes[1].GetOption("period", 0L).Should().Be(10);
        var text = config.ToPbtxt();
        GraphConfig.ParsePbtxt(text).Nodes.Should().HaveCount(2);
        GraphConfig.SplitTagged("TAG:1:name", "X").Should().Be(("TAG", "name"));
        GraphConfig.SplitTagged("name", "X").Should().Be(("X", "name"));
    }

    [Fact]
    public async Task Builds_and_runs_from_config()
    {
        var graph = GraphConfig.ParsePbtxt(Pbtxt).Build();
        var seen = new ConcurrentQueue<object>();
        graph.ObserveOutputStream<object>("out", p => seen.Enqueue(p.Value));
        await graph.StartAsync();
        graph.AddPacket("in", (object)"a", 0);
        await graph.CloseAsync();
        seen.Should().Equal("a");
    }

    [Theory]
    [InlineData("node { input_stream: \"x\" }", "calculator")]
    [InlineData("bogus: 1", "Unknown graph field")]
    [InlineData("node { calculator: \"X\" what: 1 }", "Unknown node field")]
    [InlineData("input_stream: \"unterminated", "unterminated")]
    [InlineData("input_stream: @", "unexpected")]
    public void Reports_syntax_errors(string text, string message) =>
        FluentActions.Invoking(() => GraphConfig.ParsePbtxt(text)).Should().Throw<FormatException>().WithMessage($"*{message}*");

    [Fact]
    public void Registry_rejects_unknown_calculators()
    {
        var registry = new CalculatorRegistry().Register("Custom", _ => new PassThroughNode<int>());
        registry.Names.Should().Contain("Custom");
        registry.Invoking(r => r.Create(new NodeConfig { Calculator = "Nope" })).Should().Throw<GraphValidationException>();
        CalculatorRegistry.Default.Names.Should().Contain("PassThroughCalculator");
    }
}
