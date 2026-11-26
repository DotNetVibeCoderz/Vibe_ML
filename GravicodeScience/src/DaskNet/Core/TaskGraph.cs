using System.Text;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// The tasks one collection operation added, plus the layers they read from. Dask's
/// <c>HighLevelGraph</c>: collections share their ancestors' layers instead of copying their tasks.
/// </summary>
public sealed class GraphLayer
{
    /// <summary>Creates a layer.</summary>
    public GraphLayer(string name, IEnumerable<TaskSpec> tasks, IEnumerable<GraphLayer>? parents = null)
    {
        Name = name;
        Tasks = tasks.ToDictionary(t => t.Key);
        Parents = (parents ?? []).Distinct().ToArray();
    }

    /// <summary>The layer name (shared by its task keys).</summary>
    public string Name { get; }

    /// <summary>The tasks this layer adds.</summary>
    public IReadOnlyDictionary<TaskKey, TaskSpec> Tasks { get; }

    /// <summary>The layers these tasks read from.</summary>
    public IReadOnlyList<GraphLayer> Parents { get; }

    /// <summary>This layer and every ancestor, each once, ancestors first.</summary>
    public IReadOnlyList<GraphLayer> Flatten() => Flatten([this]);

    /// <summary>The union of several layer trees, each layer once, ancestors first.</summary>
    public static IReadOnlyList<GraphLayer> Flatten(IEnumerable<GraphLayer> roots)
    {
        var seen = new HashSet<GraphLayer>(ReferenceEqualityComparer.Instance);
        var ordered = new List<GraphLayer>();
        var stack = new Stack<(GraphLayer Layer, bool Expanded)>();
        foreach (var root in roots.Reverse()) stack.Push((root, false));

        // Iterative post-order: a pipeline of a few thousand operations must not overflow the stack.
        while (stack.Count > 0)
        {
            var (layer, expanded) = stack.Pop();
            if (expanded) { ordered.Add(layer); continue; }
            if (!seen.Add(layer)) continue;
            stack.Push((layer, true));
            for (var i = layer.Parents.Count - 1; i >= 0; i--)
                if (!seen.Contains(layer.Parents[i])) stack.Push((layer.Parents[i], false));
        }
        return ordered;
    }
}

/// <summary>
/// Anything that is a lazy computation: a graph layer and the keys that hold its result.
/// </summary>
public interface IDaskCollection
{
    /// <summary>The newest layer; its ancestors complete the graph.</summary>
    GraphLayer Layer { get; }

    /// <summary>The keys whose results make up this collection.</summary>
    IReadOnlyList<TaskKey> OutputKeys { get; }
}

/// <summary>A flat task graph, ready to be optimised and scheduled.</summary>
public sealed class TaskGraph
{
    private readonly Dictionary<TaskKey, TaskSpec> _tasks;

    /// <summary>Creates a graph from tasks.</summary>
    public TaskGraph(IEnumerable<TaskSpec> tasks)
    {
        _tasks = [];
        foreach (var t in tasks) _tasks[t.Key] = t;
    }

    /// <summary>Merges the layers of one or more collections.</summary>
    public static TaskGraph FromLayers(IEnumerable<GraphLayer> roots)
        => new(GraphLayer.Flatten(roots).SelectMany(l => l.Tasks.Values));

    /// <summary>Merges the layers of one or more collections.</summary>
    public static TaskGraph FromCollections(params IDaskCollection[] collections)
        => FromLayers(collections.Select(c => c.Layer));

    /// <summary>The tasks by key.</summary>
    public IReadOnlyDictionary<TaskKey, TaskSpec> Tasks => _tasks;

    /// <summary>How many tasks.</summary>
    public int Count => _tasks.Count;

    /// <summary>One task.</summary>
    public TaskSpec this[TaskKey key] => _tasks[key];

    /// <summary>Whether a task exists.</summary>
    public bool Contains(TaskKey key) => _tasks.ContainsKey(key);

    /// <summary>Whether every task can run on a cluster worker.</summary>
    public bool IsRemotable => _tasks.Values.All(t => t.IsRemotable);

    /// <summary>For every task, the tasks that read it.</summary>
    public IReadOnlyDictionary<TaskKey, List<TaskKey>> Dependents()
    {
        var result = _tasks.Keys.ToDictionary(k => k, _ => new List<TaskKey>());
        foreach (var t in _tasks.Values)
            foreach (var d in t.Dependencies)
                if (result.TryGetValue(d, out var list)) list.Add(t.Key);
        return result;
    }

    /// <summary>Throws if a dependency is missing or the graph has a cycle.</summary>
    public void Validate()
    {
        foreach (var t in _tasks.Values)
            foreach (var d in t.Dependencies)
                if (!_tasks.ContainsKey(d))
                    throw new InvalidOperationException($"Task {t.Key} depends on {d}, which is not in the graph.");
        _ = TopologicalOrder();
    }

    /// <summary>Every task after all of its dependencies (Kahn's algorithm).</summary>
    /// <exception cref="InvalidOperationException">The graph has a cycle.</exception>
    public IReadOnlyList<TaskKey> TopologicalOrder()
    {
        var remaining = _tasks.Values.ToDictionary(t => t.Key, t => t.Dependencies.Count(_tasks.ContainsKey));
        var dependents = Dependents();
        var ready = new Queue<TaskKey>(remaining.Where(p => p.Value == 0).Select(p => p.Key));
        var order = new List<TaskKey>(_tasks.Count);

        while (ready.Count > 0)
        {
            var key = ready.Dequeue();
            order.Add(key);
            foreach (var d in dependents[key])
                if (--remaining[d] == 0) ready.Enqueue(d);
        }

        if (order.Count != _tasks.Count)
        {
            var stuck = remaining.Where(p => p.Value > 0).Select(p => p.Key.ToString()).Take(5);
            throw new InvalidOperationException($"The task graph has a cycle through {string.Join(", ", stuck)}.");
        }
        return order;
    }

    /// <summary>Only the tasks <paramref name="outputs"/> need (Dask's <c>cull</c>).</summary>
    public TaskGraph Cull(IEnumerable<TaskKey> outputs)
    {
        var keep = new HashSet<TaskKey>();
        var stack = new Stack<TaskKey>(outputs);
        while (stack.Count > 0)
        {
            var key = stack.Pop();
            if (!keep.Add(key)) continue;
            if (!_tasks.TryGetValue(key, out var task))
                throw new KeyNotFoundException($"Output {key} is not in the graph.");
            foreach (var d in task.Dependencies) stack.Push(d);
        }
        return new TaskGraph(_tasks.Values.Where(t => keep.Contains(t.Key)));
    }

    /// <summary>
    /// Fuses linear chains - a task with a single dependency that nobody else reads - into one
    /// task, so a per-partition pipeline costs one scheduling round trip, not one per operation.
    /// </summary>
    /// <remarks>
    /// Only <i>linear</i> links are fused. Folding every single-use dependency into its reader
    /// would also fold the N inputs of a reduction into one task and serialise them.
    /// </remarks>
    public TaskGraph Fuse(IEnumerable<TaskKey> outputs)
    {
        var protectedKeys = outputs.ToHashSet();
        var tasks = new Dictionary<TaskKey, TaskSpec>(_tasks);
        var dependents = Dependents().ToDictionary(p => p.Key, p => p.Value.Distinct().ToList());

        foreach (var key in TopologicalOrder())
        {
            if (!tasks.TryGetValue(key, out var parent)) continue;
            if (parent.Dependencies.Count != 1) continue;

            var childKey = parent.Dependencies[0];
            if (protectedKeys.Contains(childKey)) continue;
            if (!tasks.TryGetValue(childKey, out var child)) continue;
            if (dependents[childKey].Count != 1) continue;
            if (parent.ReadsAsList(childKey)) continue;

            var fused = parent.Absorb(child);
            tasks[key] = fused;
            tasks.Remove(childKey);
            foreach (var d in child.Dependencies)
            {
                var list = dependents[d];
                list.Remove(childKey);
                if (!list.Contains(key)) list.Add(key);
            }
        }
        return new TaskGraph(tasks.Values);
    }

    /// <summary>
    /// A run priority per task: depth-first post-order from the outputs. Lower runs first.
    /// </summary>
    /// <remarks>
    /// Breadth-first execution reads every partition before reducing any of them, so peak memory
    /// is the whole dataset. Finishing one branch before starting the next lets a reduction release
    /// its inputs as it goes - the same reason Dask's <c>order</c> exists.
    /// </remarks>
    public IReadOnlyDictionary<TaskKey, int> Priorities(IReadOnlyList<TaskKey> outputs)
    {
        var priority = new Dictionary<TaskKey, int>(_tasks.Count);
        var visited = new HashSet<TaskKey>();
        var stack = new Stack<(TaskKey Key, bool Expanded)>();
        var next = 0;

        foreach (var output in outputs.Concat(_tasks.Keys))
        {
            if (visited.Contains(output)) continue;
            stack.Push((output, false));
            while (stack.Count > 0)
            {
                var (key, expanded) = stack.Pop();
                if (expanded) { priority[key] = next++; continue; }
                if (!visited.Add(key)) continue;
                stack.Push((key, true));
                var deps = _tasks[key].Dependencies;
                for (var i = deps.Count - 1; i >= 0; i--)
                    if (!visited.Contains(deps[i]) && _tasks.ContainsKey(deps[i])) stack.Push((deps[i], false));
            }
        }
        return priority;
    }

    // ------------------------------------------------------------------ visualisation

    /// <summary>The graph in Graphviz DOT. With <paramref name="highLevel"/>, one node per layer.</summary>
    public string ToDot(bool highLevel = false)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph dask {");
        sb.AppendLine("  rankdir=BT;");
        sb.AppendLine("  node [shape=box, style=rounded, fontname=\"Helvetica\"];");

        if (highLevel)
        {
            foreach (var (layer, label, count) in LayerSummary())
                sb.AppendLine($"  \"{Escape(layer)}\" [label=\"{Escape(label)}\\n{count} task{(count == 1 ? "" : "s")}\"];");
            foreach (var (from, to) in LayerEdges())
                sb.AppendLine($"  \"{Escape(from)}\" -> \"{Escape(to)}\";");
        }
        else
        {
            var ids = NodeIds();
            foreach (var t in _tasks.Values.OrderBy(t => ids[t.Key]))
                sb.AppendLine($"  {ids[t.Key]} [label=\"{Escape(t.Label)}{(t.Key.Index >= 0 ? $" #{t.Key.Index}" : "")}\"];");
            foreach (var t in _tasks.Values.OrderBy(t => ids[t.Key]))
                foreach (var d in t.Dependencies)
                    if (ids.TryGetValue(d, out var from)) sb.AppendLine($"  {from} -> {ids[t.Key]};");
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>The graph as a Mermaid flowchart (renders on GitHub and in notebooks).</summary>
    public string ToMermaid(bool highLevel = false)
    {
        var sb = new StringBuilder();
        sb.AppendLine("flowchart BT");
        if (highLevel)
        {
            var ids = LayerSummary().Select((l, i) => (l, i)).ToDictionary(p => p.l.Layer, p => $"L{p.i}");
            foreach (var (layer, label, count) in LayerSummary())
                sb.AppendLine($"  {ids[layer]}[\"{label} ({count})\"]");
            foreach (var (from, to) in LayerEdges())
                sb.AppendLine($"  {ids[from]} --> {ids[to]}");
        }
        else
        {
            var ids = NodeIds();
            foreach (var t in _tasks.Values.OrderBy(t => ids[t.Key]))
                sb.AppendLine($"  {ids[t.Key]}[\"{t.Label}{(t.Key.Index >= 0 ? $" #{t.Key.Index}" : "")}\"]");
            foreach (var t in _tasks.Values.OrderBy(t => ids[t.Key]))
                foreach (var d in t.Dependencies)
                    if (ids.TryGetValue(d, out var from)) sb.AppendLine($"  {from} --> {ids[t.Key]}");
        }
        return sb.ToString();
    }

    /// <summary>A one-line-per-layer text summary.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"TaskGraph: {Count} tasks in {LayerSummary().Count} layers");
        foreach (var (layer, label, count) in LayerSummary())
            sb.AppendLine($"  {label,-28} {count,6} task{(count == 1 ? "" : "s")}   [{layer}]");
        return sb.ToString();
    }

    private Dictionary<TaskKey, string> NodeIds()
    {
        var order = TopologicalOrder();
        var ids = new Dictionary<TaskKey, string>(order.Count);
        for (var i = 0; i < order.Count; i++) ids[order[i]] = $"t{i}";
        return ids;
    }

    private List<(string Layer, string Label, int Count)> LayerSummary()
    {
        var order = TopologicalOrder();
        var seen = new Dictionary<string, (string Label, int Count, int First)>();
        for (var i = 0; i < order.Count; i++)
        {
            var t = _tasks[order[i]];
            seen[t.Key.Name] = seen.TryGetValue(t.Key.Name, out var s) ? (s.Label, s.Count + 1, s.First) : (t.Label, 1, i);
        }
        return seen.OrderBy(p => p.Value.First).Select(p => (p.Key, p.Value.Label, p.Value.Count)).ToList();
    }

    private List<(string From, string To)> LayerEdges()
    {
        var edges = new List<(string, string)>();
        var seen = new HashSet<(string, string)>();
        foreach (var t in _tasks.Values)
            foreach (var d in t.Dependencies)
                if (d.Name != t.Key.Name && seen.Add((d.Name, t.Key.Name))) edges.Add((d.Name, t.Key.Name));
        return edges;
    }

    private static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
