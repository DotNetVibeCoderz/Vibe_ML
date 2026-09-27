using System.Globalization;
using System.Text;
using MediaPipeNet.Framework.Nodes;

namespace MediaPipeNet.Framework.Config;

/// <summary>
/// Declarative description of a node, equivalent to a <c>node { ... }</c> block of a MediaPipe
/// <c>CalculatorGraphConfig</c>. Streams are written <c>"TAG:stream_name"</c>.
/// </summary>
public sealed record NodeConfig
{
    /// <summary>Registered calculator name, e.g. <c>FaceDetectorCalculator</c>.</summary>
    public required string Calculator { get; init; }

    /// <summary>Optional node name (defaults to the calculator name plus an index).</summary>
    public string? Name { get; init; }

    /// <summary>Input streams as <c>TAG:stream</c>.</summary>
    public List<string> InputStreams { get; init; } = [];

    /// <summary>Output streams as <c>TAG:stream</c>.</summary>
    public List<string> OutputStreams { get; init; } = [];

    /// <summary>Input side packets as <c>TAG:name</c>.</summary>
    public List<string> InputSidePackets { get; init; } = [];

    /// <summary>Flattened <c>options { key: value }</c> entries.</summary>
    public Dictionary<string, string> Options { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Input tags that are back edges (<c>input_stream_info { tag_index: "LOOP" back_edge: true }</c>).</summary>
    public HashSet<string> BackEdgeTags { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Executor the node runs on (<c>executor: "name"</c>), or null for the shared thread pool.</summary>
    public string? Executor { get; init; }

    /// <summary>Gets an option parsed as <typeparamref name="T"/>, or <paramref name="fallback"/>.</summary>
    public T GetOption<T>(string key, T fallback) where T : IParsable<T> =>
        Options.TryGetValue(key, out var s) && T.TryParse(s, CultureInfo.InvariantCulture, out var v) ? v : fallback;
}

/// <summary>
/// Declarative description of a graph — the .NET counterpart of MediaPipe's
/// <c>CalculatorGraphConfig</c>. Build it in code, deserialize it from JSON, or parse the
/// familiar text-proto (<c>.pbtxt</c>) syntax with <see cref="ParsePbtxt"/>.
/// </summary>
public sealed record GraphConfig
{
    /// <summary>Graph input streams.</summary>
    public List<string> InputStreams { get; init; } = [];

    /// <summary>Graph output streams.</summary>
    public List<string> OutputStreams { get; init; } = [];

    /// <summary>Graph input side packets.</summary>
    public List<string> InputSidePackets { get; init; } = [];

    /// <summary>Maximum in-flight timestamps (0 = unlimited). See <see cref="GraphOptions.MaxInFlight"/>.</summary>
    public int MaxInFlight { get; init; }

    /// <summary>Queue size applied to every graph input (0 = unbounded, drop oldest when full).</summary>
    public int MaxQueueSize { get; init; }

    /// <summary>The nodes.</summary>
    public List<NodeConfig> Nodes { get; init; } = [];

    /// <summary>Executors by name with their thread count (<c>executor { name: "gpu" num_threads: 1 }</c>).</summary>
    public Dictionary<string, int> Executors { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Record a Chrome trace (<c>profiler_config { trace_enabled: true }</c>). See <see cref="GraphOptions.EnableTracing"/>.</summary>
    public bool EnableTracing { get; init; }

    /// <summary>Splits <c>TAG:name</c> (or <c>TAG:index:name</c>) into tag and name. An untagged entry gets tag <c>IN</c>/<c>OUT</c>-style default.</summary>
    public static (string Tag, string Name) SplitTagged(string value, string defaultTag)
    {
        var parts = value.Split(':');
        return parts.Length switch
        {
            1 => (defaultTag, parts[0]),
            2 => (parts[0], parts[1]),
            _ => (parts[0], parts[^1]),
        };
    }

    /// <summary>
    /// Instantiates the nodes through <paramref name="registry"/> and builds a validated graph.
    /// Graph input types are inferred from the first consuming port.
    /// </summary>
    public CalculatorGraph Build(CalculatorRegistry? registry = null, GraphOptions? options = null) =>
        ToBuilder(registry, options).Build();

    /// <summary>
    /// Creates the <see cref="GraphBuilder"/> for this config (nodes instantiated through
    /// <paramref name="registry"/>; calculators registered as subgraphs are inlined).
    /// </summary>
    public GraphBuilder ToBuilder(CalculatorRegistry? registry = null, GraphOptions? options = null) =>
        ToBuilder(registry ?? CalculatorRegistry.Default, options ?? new GraphOptions(), depth: 0);

    private GraphBuilder ToBuilder(CalculatorRegistry registry, GraphOptions options, int depth)
    {
        if (depth > 16) throw new GraphValidationException("Subgraphs are nested more than 16 levels deep (recursive subgraph?).");
        var builder = new GraphBuilder
        {
            Options = options with { MaxInFlight = MaxInFlight, EnableTracing = options.EnableTracing || EnableTracing },
        };
        foreach (var (name, threads) in Executors) builder.AddExecutor(name, threads);
        var inputTypes = new Dictionary<string, Type>(StringComparer.Ordinal);
        var nodes = new List<(NodeConfig Config, ICalculatorNode? Node, CalculatorContract? Contract, GraphConfig? Subgraph)>();
        foreach (var nc in Nodes)
        {
            if (registry.TryGetSubgraph(nc.Calculator, out var subgraph))
            {
                nodes.Add((nc, null, null, subgraph));
                continue;
            }
            var node = registry.Create(nc);
            var contract = new CalculatorContract();
            node.GetContract(contract);
            nodes.Add((nc, node, contract, null));
            foreach (var s in nc.InputStreams)
            {
                var (tag, name) = SplitTagged(s, contract.Inputs.Count > 0 ? contract.Inputs[0].Tag : "IN");
                var spec = contract.Inputs.FirstOrDefault(p => p.Tag == tag)
                           ?? throw new GraphValidationException($"Calculator '{nc.Calculator}' has no input '{tag}'.");
                inputTypes.TryAdd(name, spec.Type);
            }
        }
        // Subgraph configs tag their boundary streams ("IN:x"); the stream itself is the name after the tag.
        foreach (var s in InputStreams.Select(StreamName))
            builder.AddInputStream(s, inputTypes.GetValueOrDefault(s, typeof(object)), new GraphInputOptions(MaxQueueSize));
        foreach (var s in OutputStreams.Select(StreamName)) builder.AddOutputStream(s);
        foreach (var s in InputSidePackets) builder.SidePackets.TryAdd(s, typeof(object));
        int i = 0;
        foreach (var (nc, node, contract, subgraph) in nodes)
        {
            var name = nc.Name ?? $"{nc.Calculator}_{i++}";
            if (subgraph is not null)
            {
                builder.AddSubgraph(name, subgraph.ToBuilder(registry, options, depth + 1),
                    MapPorts(subgraph.InputStreams, nc.InputStreams, name, "input"),
                    MapPorts(subgraph.OutputStreams, nc.OutputStreams, name, "output"));
                continue;
            }
            var nb = builder.AddNode(name, node!);
            foreach (var s in nc.InputStreams)
            {
                var (tag, stream) = SplitTagged(s, contract!.Inputs[0].Tag);
                nb.In(tag, stream, nc.BackEdgeTags.Contains(tag));
            }
            foreach (var s in nc.OutputStreams)
            {
                var (tag, stream) = SplitTagged(s, contract!.Outputs.Count > 0 ? contract.Outputs[0].Tag : "OUT");
                nb.Out(tag, stream);
            }
            foreach (var s in nc.InputSidePackets)
            {
                var (tag, side) = SplitTagged(s, contract!.InputSidePackets.Count > 0 ? contract.InputSidePackets[0].Tag : "SIDE");
                nb.Side(tag, side);
            }
            if (nc.Executor is not null) nb.OnExecutor(nc.Executor);
        }
        return builder;
    }

    private static string StreamName(string declared) => declared.Split(':')[^1];

    /// <summary>
    /// Maps a subgraph's declared streams (<c>TAG:inner</c>) to the streams of the node that uses it
    /// (<c>TAG:outer</c>): by tag when both are tagged, otherwise by position.
    /// </summary>
    private static Dictionary<string, string> MapPorts(List<string> declared, List<string> used, string node, string kind)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        var usedByTag = used.Select(u => u.Split(':')).Where(p => p.Length >= 2).ToDictionary(p => p[0], p => p[^1], StringComparer.Ordinal);
        for (int k = 0; k < declared.Count; k++)
        {
            var parts = declared[k].Split(':');
            string inner = parts[^1];
            string? outer = parts.Length >= 2 && usedByTag.TryGetValue(parts[0], out var byTag) ? byTag
                : k < used.Count ? used[k].Split(':')[^1] : null;
            map[inner] = outer
                ?? throw new GraphValidationException($"Subgraph node '{node}': {kind} '{declared[k]}' is not connected.");
        }
        return map;
    }

    /// <summary>Renders the config in text-proto syntax.</summary>
    public string ToPbtxt()
    {
        var sb = new StringBuilder();
        foreach (var s in InputStreams) sb.Append("input_stream: \"").Append(s).Append("\"\n");
        foreach (var s in OutputStreams) sb.Append("output_stream: \"").Append(s).Append("\"\n");
        foreach (var s in InputSidePackets) sb.Append("input_side_packet: \"").Append(s).Append("\"\n");
        if (MaxInFlight > 0) sb.Append("max_in_flight: ").Append(MaxInFlight).Append('\n');
        if (MaxQueueSize > 0) sb.Append("max_queue_size: ").Append(MaxQueueSize).Append('\n');
        if (EnableTracing) sb.Append("profiler_config {\n  trace_enabled: true\n}\n");
        foreach (var (name, threads) in Executors)
            sb.Append("executor {\n  name: \"").Append(name).Append("\"\n  num_threads: ").Append(threads).Append("\n}\n");
        foreach (var n in Nodes)
        {
            sb.Append("\nnode {\n  calculator: \"").Append(n.Calculator).Append("\"\n");
            if (n.Name is not null) sb.Append("  name: \"").Append(n.Name).Append("\"\n");
            if (n.Executor is not null) sb.Append("  executor: \"").Append(n.Executor).Append("\"\n");
            foreach (var s in n.InputStreams) sb.Append("  input_stream: \"").Append(s).Append("\"\n");
            foreach (var s in n.OutputStreams) sb.Append("  output_stream: \"").Append(s).Append("\"\n");
            foreach (var s in n.InputSidePackets) sb.Append("  input_side_packet: \"").Append(s).Append("\"\n");
            foreach (var tag in n.BackEdgeTags) sb.Append("  input_stream_info {\n    tag_index: \"").Append(tag).Append("\"\n    back_edge: true\n  }\n");
            if (n.Options.Count > 0)
            {
                sb.Append("  options {\n");
                foreach (var (k, v) in n.Options) sb.Append("    ").Append(k).Append(": ").Append(Quote(v)).Append('\n');
                sb.Append("  }\n");
            }
            sb.Append("}\n");
        }
        return sb.ToString();

        static string Quote(string v) => double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out _) || v is "true" or "false" ? v : $"\"{v}\"";
    }

    /// <summary>
    /// Parses MediaPipe's text-proto graph syntax (the subset used by graph configs):
    /// <c>input_stream</c>, <c>output_stream</c>, <c>input_side_packet</c>, <c>max_in_flight</c>,
    /// <c>max_queue_size</c> and <c>node { calculator, name, input_stream, output_stream,
    /// input_side_packet, options { key: value } }</c>. <c>#</c> starts a comment.
    /// </summary>
    public static GraphConfig ParsePbtxt(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var tokens = Tokenize(text);
        int pos = 0;
        var config = new GraphConfig();
        var maxInFlight = 0;
        var maxQueue = 0;
        var tracing = false;
        while (pos < tokens.Count)
        {
            var key = Expect(tokens, ref pos, TokenKind.Identifier);
            if (key == "node")
            {
                config.Nodes.Add(ParseNode(tokens, ref pos));
                continue;
            }
            if (key is "executor" or "profiler_config")
            {
                if (tokens[pos].Kind == TokenKind.Colon) pos++;
                var block = new Dictionary<string, string>(StringComparer.Ordinal);
                ParseOptions(tokens, ref pos, block, prefix: "");
                if (key == "profiler_config")
                {
                    tracing |= block.GetValueOrDefault("trace_enabled") == "true" || block.GetValueOrDefault("enable_profiler") == "true";
                }
                else
                {
                    var threads = block.FirstOrDefault(kv => kv.Key.EndsWith("num_threads", StringComparison.Ordinal)).Value;
                    config.Executors[block.GetValueOrDefault("name", "")] = threads is null ? 1 : int.Parse(threads, CultureInfo.InvariantCulture);
                }
                continue;
            }
            Expect(tokens, ref pos, TokenKind.Colon);
            var value = ExpectValue(tokens, ref pos);
            switch (key)
            {
                case "input_stream": config.InputStreams.Add(value); break;
                case "output_stream": config.OutputStreams.Add(value); break;
                case "input_side_packet": config.InputSidePackets.Add(value); break;
                case "max_in_flight": maxInFlight = int.Parse(value, CultureInfo.InvariantCulture); break;
                case "max_queue_size": maxQueue = int.Parse(value, CultureInfo.InvariantCulture); break;
                default: throw new FormatException($"Unknown graph field '{key}'.");
            }
        }
        return config with { MaxInFlight = maxInFlight, MaxQueueSize = maxQueue, EnableTracing = tracing };
    }

    private static NodeConfig ParseNode(List<Token> tokens, ref int pos)
    {
        Expect(tokens, ref pos, TokenKind.OpenBrace);
        string? calculator = null, name = null, executor = null;
        var node = new NodeConfig { Calculator = "" };
        while (tokens[pos].Kind != TokenKind.CloseBrace)
        {
            var key = Expect(tokens, ref pos, TokenKind.Identifier);
            if (key is "options" or "node_options")
            {
                if (tokens[pos].Kind == TokenKind.Colon) pos++;
                ParseOptions(tokens, ref pos, node.Options, prefix: "");
                continue;
            }
            if (key == "input_stream_info")
            {
                if (tokens[pos].Kind == TokenKind.Colon) pos++;
                var info = new Dictionary<string, string>(StringComparer.Ordinal);
                ParseOptions(tokens, ref pos, info, prefix: "");
                if (info.GetValueOrDefault("back_edge") == "true" && info.TryGetValue("tag_index", out var tagIndex))
                    node.BackEdgeTags.Add(tagIndex.Split(':')[0]);
                continue;
            }
            Expect(tokens, ref pos, TokenKind.Colon);
            var value = ExpectValue(tokens, ref pos);
            switch (key)
            {
                case "calculator": calculator = value; break;
                case "name": name = value; break;
                case "executor": executor = value; break;
                case "input_stream": node.InputStreams.Add(value); break;
                case "output_stream": node.OutputStreams.Add(value); break;
                case "input_side_packet": node.InputSidePackets.Add(value); break;
                default: throw new FormatException($"Unknown node field '{key}'.");
            }
        }
        pos++;
        if (calculator is null) throw new FormatException("A node is missing its 'calculator' field.");
        return node with { Calculator = calculator, Name = name, Executor = executor };
    }

    private static void ParseOptions(List<Token> tokens, ref int pos, Dictionary<string, string> into, string prefix)
    {
        Expect(tokens, ref pos, TokenKind.OpenBrace);
        while (tokens[pos].Kind != TokenKind.CloseBrace)
        {
            string key;
            if (tokens[pos].Kind == TokenKind.OpenBracket)
            {
                // [mediapipe.SomeOptions.ext] { ... } — extension syntax; the type name is ignored.
                while (tokens[pos].Kind != TokenKind.CloseBracket) pos++;
                pos++;
                ParseOptions(tokens, ref pos, into, prefix);
                continue;
            }
            key = Expect(tokens, ref pos, TokenKind.Identifier);
            if (tokens[pos].Kind == TokenKind.Colon) pos++;
            if (tokens[pos].Kind == TokenKind.OpenBrace)
            {
                ParseOptions(tokens, ref pos, into, prefix + key + ".");
                continue;
            }
            into[prefix + key] = ExpectValue(tokens, ref pos);
        }
        pos++;
    }

    private enum TokenKind { Identifier, String, Number, Colon, OpenBrace, CloseBrace, OpenBracket, CloseBracket }

    private readonly record struct Token(TokenKind Kind, string Text, int Line);

    private static string Expect(List<Token> tokens, ref int pos, TokenKind kind)
    {
        if (pos >= tokens.Count) throw new FormatException($"Unexpected end of graph config, expected {kind}.");
        var t = tokens[pos++];
        if (t.Kind != kind) throw new FormatException($"Line {t.Line}: expected {kind} but found '{t.Text}'.");
        return t.Text;
    }

    private static string ExpectValue(List<Token> tokens, ref int pos)
    {
        if (pos >= tokens.Count) throw new FormatException("Unexpected end of graph config, expected a value.");
        var t = tokens[pos++];
        return t.Kind is TokenKind.String or TokenKind.Number or TokenKind.Identifier
            ? t.Text
            : throw new FormatException($"Line {t.Line}: expected a value but found '{t.Text}'.");
    }

    private static List<Token> Tokenize(string text)
    {
        var tokens = new List<Token>();
        int line = 1;
        for (int i = 0; i < text.Length;)
        {
            char c = text[i];
            if (c == '\n') { line++; i++; continue; }
            if (char.IsWhiteSpace(c) || c == ',' || c == ';') { i++; continue; }
            if (c == '#') { while (i < text.Length && text[i] != '\n') i++; continue; }
            switch (c)
            {
                case ':': tokens.Add(new(TokenKind.Colon, ":", line)); i++; continue;
                case '{': tokens.Add(new(TokenKind.OpenBrace, "{", line)); i++; continue;
                case '}': tokens.Add(new(TokenKind.CloseBrace, "}", line)); i++; continue;
                case '[': tokens.Add(new(TokenKind.OpenBracket, "[", line)); i++; continue;
                case ']': tokens.Add(new(TokenKind.CloseBracket, "]", line)); i++; continue;
                case '"' or '\'':
                {
                    var sb = new StringBuilder();
                    char quote = c;
                    i++;
                    while (i < text.Length && text[i] != quote)
                    {
                        if (text[i] == '\\' && i + 1 < text.Length) i++;
                        sb.Append(text[i++]);
                    }
                    if (i >= text.Length) throw new FormatException($"Line {line}: unterminated string.");
                    i++;
                    tokens.Add(new(TokenKind.String, sb.ToString(), line));
                    continue;
                }
            }
            int start = i;
            if (char.IsDigit(c) || c is '-' or '+' or '.')
            {
                while (i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] is '.' or '-' or '+')) i++;
                tokens.Add(new(TokenKind.Number, text[start..i], line));
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                while (i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] is '_' or '.')) i++;
                tokens.Add(new(TokenKind.Identifier, text[start..i], line));
                continue;
            }
            throw new FormatException($"Line {line}: unexpected character '{c}'.");
        }
        return tokens;
    }
}

/// <summary>
/// Maps calculator names used in <see cref="GraphConfig"/> to node factories. The
/// <see cref="Default"/> registry knows the built-in calculators; vision tasks register theirs with
/// <c>CalculatorRegistry.Default.AddVisionCalculators()</c>.
/// </summary>
public sealed class CalculatorRegistry
{
    private readonly Dictionary<string, Func<NodeConfig, ICalculatorNode>> _factories = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GraphConfig> _subgraphs = new(StringComparer.Ordinal);

    /// <summary>The shared registry with the built-in calculators.</summary>
    public static CalculatorRegistry Default { get; } = CreateWithBuiltIns();

    /// <summary>Registered calculator names.</summary>
    public IReadOnlyCollection<string> Names => _factories.Keys;

    /// <summary>Registers (or replaces) a calculator factory.</summary>
    public CalculatorRegistry Register(string name, Func<NodeConfig, ICalculatorNode> factory)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(factory);
        lock (_factories) _factories[name] = factory;
        return this;
    }

    /// <summary>
    /// Registers a subgraph: nodes whose calculator is <paramref name="name"/> are replaced by the nodes of
    /// <paramref name="config"/>. The subgraph's <c>input_stream</c> / <c>output_stream</c> entries
    /// (<c>TAG:name</c>) are matched to the using node's streams by tag, or by position when untagged.
    /// </summary>
    public CalculatorRegistry RegisterSubgraph(string name, GraphConfig config)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(config);
        lock (_factories) _subgraphs[name] = config;
        return this;
    }

    /// <summary>Gets a registered subgraph.</summary>
    public bool TryGetSubgraph(string name, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out GraphConfig? config)
    {
        lock (_factories) return _subgraphs.TryGetValue(name, out config);
    }

    /// <summary>Creates the node described by <paramref name="config"/>.</summary>
    public ICalculatorNode Create(NodeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        Func<NodeConfig, ICalculatorNode>? factory;
        lock (_factories) _factories.TryGetValue(config.Calculator, out factory);
        return factory?.Invoke(config)
               ?? throw new GraphValidationException($"Unknown calculator '{config.Calculator}'. Registered: {string.Join(", ", Names)}.");
    }

    private static CalculatorRegistry CreateWithBuiltIns() => new CalculatorRegistry()
        .Register("PassThroughCalculator", _ => new PassThroughNode<object>())
        .Register("PacketThinnerCalculator", c => new PacketThinnerNode(TimeSpan.FromTicks(c.GetOption("period", 33_333L) * 10)))
        .Register("PacketCounterCalculator", _ => new PacketCounterNode())
        .Register("PreviousLoopbackCalculator", _ => new PreviousLoopbackNode<object>());
}
