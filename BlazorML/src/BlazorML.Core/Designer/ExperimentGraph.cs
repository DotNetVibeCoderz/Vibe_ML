using System.Text.Json;
using System.Text.Json.Serialization;
using BlazorML.Core.Domain;

namespace BlazorML.Core.Designer;

/// <summary>
/// The document the designer canvas edits and the run engine executes. Persisted as JSON on
/// <see cref="Experiment.GraphJson"/> so the whole graph versions as one unit.
/// </summary>
public class ExperimentGraph
{
    public List<GraphNode> Nodes { get; set; } = new();
    public List<GraphEdge> Edges { get; set; } = new();

    /// <summary>Canvas pan/zoom, so reopening an experiment restores the view.</summary>
    public CanvasView View { get; set; } = new();

    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
        WriteIndented = false
    };

    public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

    public static ExperimentGraph FromJson(string? json)
    {
        if (string.IsNullOrWhiteSpace(json) || json == "{}")
        {
            return new ExperimentGraph();
        }

        try
        {
            return JsonSerializer.Deserialize<ExperimentGraph>(json, JsonOptions) ?? new ExperimentGraph();
        }
        catch (JsonException)
        {
            return new ExperimentGraph();
        }
    }

    public GraphNode? Node(string id) => Nodes.FirstOrDefault(n => n.Id == id);

    /// <summary>Edges feeding a node, in the order the target's input ports are declared.</summary>
    public IEnumerable<GraphEdge> InboundEdges(string nodeId) => Edges.Where(e => e.TargetNodeId == nodeId);

    public IEnumerable<GraphEdge> OutboundEdges(string nodeId) => Edges.Where(e => e.SourceNodeId == nodeId);

    /// <summary>
    /// Kahn topological sort. Returns false when the graph contains a cycle, which the designer
    /// prevents interactively but an imported or agent-authored graph might still contain.
    /// </summary>
    public bool TryTopologicalSort(out List<GraphNode> ordered, out string? error)
    {
        var sorted = new List<GraphNode>();
        ordered = sorted;
        error = null;

        var indegree = Nodes.ToDictionary(n => n.Id, _ => 0);
        foreach (var edge in Edges)
        {
            if (!indegree.ContainsKey(edge.TargetNodeId) || !indegree.ContainsKey(edge.SourceNodeId))
            {
                error = $"Edge {edge.Id} refers to a node that is not on the canvas.";
                return false;
            }

            indegree[edge.TargetNodeId]++;
        }

        var queue = new Queue<string>(indegree.Where(kv => kv.Value == 0).Select(kv => kv.Key));
        while (queue.Count > 0)
        {
            var id = queue.Dequeue();
            var node = Node(id);
            if (node is not null)
            {
                sorted.Add(node);
            }

            foreach (var edge in OutboundEdges(id))
            {
                if (--indegree[edge.TargetNodeId] == 0)
                {
                    queue.Enqueue(edge.TargetNodeId);
                }
            }
        }

        if (sorted.Count != Nodes.Count)
        {
            var stuck = Nodes.Where(n => !sorted.Any(o => o.Id == n.Id)).Select(n => n.Label);
            error = $"The graph loops back on itself around: {string.Join(", ", stuck)}.";
            return false;
        }

        return true;
    }
}

public class GraphNode
{
    public string Id { get; set; } = Guid.NewGuid().ToString("n")[..8];

    /// <summary>Key into <see cref="Modules.ModuleCatalog"/>.</summary>
    public string ModuleId { get; set; } = string.Empty;

    /// <summary>Defaults to the module name but the user can rename a node.</summary>
    public string Label { get; set; } = string.Empty;

    public double X { get; set; }
    public double Y { get; set; }

    /// <summary>Parameter values keyed by parameter name; types follow the module's parameter specs.</summary>
    public Dictionary<string, string?> Parameters { get; set; } = new();

    public string? Comment { get; set; }
    public bool Collapsed { get; set; }
}

public class GraphEdge
{
    public string Id { get; set; } = Guid.NewGuid().ToString("n")[..8];
    public string SourceNodeId { get; set; } = string.Empty;
    public int SourcePort { get; set; }
    public string TargetNodeId { get; set; } = string.Empty;
    public int TargetPort { get; set; }
}

public class CanvasView
{
    public double PanX { get; set; }
    public double PanY { get; set; }
    public double Zoom { get; set; } = 1;
}

/// <summary>Per-node outcome recorded on a run, replayed onto the canvas as status badges.</summary>
public class NodeRunResult
{
    public NodeRunState State { get; set; } = NodeRunState.Pending;
    public double DurationSeconds { get; set; }
    public string? Message { get; set; }
    public int? RowsOut { get; set; }
    public int? ColumnsOut { get; set; }

    /// <summary>Metrics produced by evaluation nodes, ready to chart.</summary>
    public Dictionary<string, double>? Metrics { get; set; }

    /// <summary>
    /// The full evaluation payload when the node produced one: ROC points, the confusion
    /// matrix, residuals, feature weights. Carried separately from <see cref="Metrics"/>
    /// because these are the shapes the charts need, and a flat number dictionary cannot
    /// hold them.
    /// </summary>
    public Analysis.EvaluationResult? Evaluation { get; set; }

    /// <summary>Preview of the node's output dataset: header row plus a handful of rows.</summary>
    public List<List<string>>? Preview { get; set; }
}
