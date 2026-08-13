namespace Gravicode.Science.GraviNum.Autodiff;

/// <summary>
/// A node in a reverse-mode automatic differentiation graph: a value, its gradient, and the
/// rule for pushing that gradient to whatever produced it.
/// </summary>
/// <remarks>
/// <para>
/// Every operation on a <see cref="Tensor"/> returns a new tensor that remembers its inputs and
/// how to scatter a gradient back to them. Calling <see cref="Backward"/> on a scalar walks that
/// graph in reverse topological order and fills in <see cref="Gradient"/> on every tensor marked
/// <see cref="RequiresGradient"/>.
/// </para>
/// <para>
/// The point of this type is that a backward pass no longer has to be derived by hand. The GNN
/// layers in GraviGraph carry hand-written gradients, which was reasonable for three fixed
/// architectures and does not extend to a fourth; and the variational inference in GraviProb used
/// central finite differences, which costs two extra log-density evaluations <em>per parameter</em>
/// and loses roughly half the available precision.
/// </para>
/// <para>
/// This is deliberately not a deep learning framework. There is no graph optimiser, no fused
/// kernel, no device placement — it is a tape, and it exists so that new layers and new log
/// densities can be written once, forwards.
/// </para>
/// </remarks>
public sealed partial class Tensor
{
    private readonly List<Tensor> _parents;
    private readonly Action<NdArray>? _backward;

    private Tensor(NdArray value, bool requiresGradient, List<Tensor> parents, Action<NdArray>? backward)
    {
        Value = value;
        // NdArray.Shape is a span, which a backward closure cannot capture, so it is materialised
        // once here rather than at every use.
        Shape = value.Shape.ToArray();
        RequiresGradient = requiresGradient;
        _parents = parents;
        _backward = backward;
    }

    /// <summary>The forward value.</summary>
    public NdArray Value { get; }

    /// <summary>
    /// The accumulated derivative of the scalar passed to <see cref="Backward"/> with respect to
    /// this tensor, or null until a backward pass reaches it.
    /// </summary>
    public NdArray? Gradient { get; private set; }

    /// <summary>Whether this tensor participates in the backward pass.</summary>
    public bool RequiresGradient { get; }

    /// <summary>Length of each axis of <see cref="Value"/>.</summary>
    public int[] Shape { get; }

    /// <summary>Number of elements.</summary>
    public int Size => Value.Size;

    /// <summary>The single element of a one-element tensor.</summary>
    public double Item => Size == 1
        ? Value.At(0)
        : throw new InvalidOperationException($"Item requires a single-element tensor, got {Size} elements.");

    // ================================================================ construction

    /// <summary>A leaf whose gradient will be accumulated.</summary>
    public static Tensor Parameter(NdArray value) => new(value, true, [], null);

    /// <summary>A leaf whose gradient will be accumulated.</summary>
    public static Tensor Parameter(double value) => Parameter(NdArray.Scalar(value));

    /// <summary>A leaf treated as constant: gradients stop here.</summary>
    public static Tensor Constant(NdArray value) => new(value, false, [], null);

    /// <summary>A leaf treated as constant: gradients stop here.</summary>
    public static Tensor Constant(double value) => Constant(NdArray.Scalar(value));

    /// <summary>Wraps a raw scalar as a constant, so <c>x * 2.0</c> reads naturally.</summary>
    public static implicit operator Tensor(double value) => Constant(value);

    /// <summary>Builds a derived tensor with its backward rule.</summary>
    /// <param name="value">The forward result.</param>
    /// <param name="parents">Tensors this result was computed from.</param>
    /// <param name="backward">
    /// Receives the gradient flowing into this node and must call
    /// <see cref="AccumulateGradient"/> on each parent that needs one.
    /// </param>
    internal static Tensor Derived(NdArray value, List<Tensor> parents, Action<NdArray> backward)
    {
        var tracked = parents.Any(p => p.RequiresGradient);
        return tracked ? new Tensor(value, true, parents, backward) : Constant(value);
    }

    /// <summary>Adds <paramref name="amount"/> into this tensor's gradient.</summary>
    internal void AccumulateGradient(NdArray amount)
    {
        if (!RequiresGradient) return;

        // A tensor used twice receives a contribution from each use, and they sum — that is the
        // whole reason gradients accumulate rather than overwrite.
        Gradient = Gradient is null ? amount.Copy() : UFunc.Add(Gradient, amount);
    }

    /// <summary>Clears the accumulated gradient on this tensor and everything behind it.</summary>
    public void ZeroGradients()
    {
        foreach (var node in TopologicalOrder()) node.Gradient = null;
    }

    // ================================================================ backward pass

    /// <summary>
    /// Propagates derivatives back from this tensor, which must hold a single value.
    /// </summary>
    /// <remarks>
    /// Reverse topological order is what makes one pass enough: a node is only visited once every
    /// consumer of it has already contributed, so its gradient is complete before it is used.
    /// </remarks>
    public void Backward()
    {
        if (Size != 1)
            throw new InvalidOperationException(
                $"Backward starts from a scalar; this tensor has {Size} elements. Reduce it with Sum or Mean first.");

        var order = TopologicalOrder();

        foreach (var node in order) node.Gradient = null;
        Gradient = NdArray.Ones(Shape);

        // TopologicalOrder lists parents before children, so consuming it backwards visits every
        // node only after all of its consumers.
        for (var i = order.Count - 1; i >= 0; i--)
        {
            var node = order[i];
            if (node.Gradient is null || node._backward is null) continue;
            node._backward(node.Gradient);
        }
    }

    /// <summary>Every tensor this one depends on, parents before children.</summary>
    private List<Tensor> TopologicalOrder()
    {
        var order = new List<Tensor>();

        // Reference identity, not value equality: two nodes holding equal arrays are still two
        // distinct points in the graph, and merging them would drop a gradient path.
        var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);

        // Iterative rather than recursive: a deep chain — an unrolled RNN, a many-layer GNN —
        // would otherwise overflow the stack.
        var stack = new Stack<(Tensor Node, bool Expanded)>();
        stack.Push((this, false));

        while (stack.Count > 0)
        {
            var (node, expanded) = stack.Pop();

            if (expanded)
            {
                order.Add(node);
                continue;
            }

            if (!seen.Add(node)) continue;

            stack.Push((node, true));
            foreach (var parent in node._parents) stack.Push((parent, false));
        }

        return order;
    }
}
