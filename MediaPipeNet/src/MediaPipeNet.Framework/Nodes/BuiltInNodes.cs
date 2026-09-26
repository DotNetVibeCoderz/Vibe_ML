namespace MediaPipeNet.Framework.Nodes;

/// <summary>Forwards every input packet unchanged (MediaPipe's <c>PassThroughCalculator</c>).</summary>
/// <typeparam name="T">Payload type.</typeparam>
public sealed class PassThroughNode<T> : CalculatorNode
{
    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<T>("IN").AddOutput<T>("OUT");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context) => context.SendPacket("OUT", context.GetInputPacket("IN")!);
}

/// <summary>Transforms each input value with a delegate. Returning null emits nothing for that timestamp.</summary>
/// <typeparam name="TIn">Input type.</typeparam>
/// <typeparam name="TOut">Output type.</typeparam>
/// <param name="transform">The transformation.</param>
public sealed class LambdaNode<TIn, TOut>(Func<TIn, TOut?> transform) : CalculatorNode
{
    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<TIn>("IN").AddOutput<TOut>("OUT");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context)
    {
        var result = transform(context.GetInput<TIn>("IN"));
        if (result is not null) context.Send("OUT", result);
    }
}

/// <summary>Transforms each input value with an asynchronous delegate.</summary>
/// <typeparam name="TIn">Input type.</typeparam>
/// <typeparam name="TOut">Output type.</typeparam>
/// <param name="transform">The transformation.</param>
public sealed class AsyncLambdaNode<TIn, TOut>(Func<TIn, CancellationToken, ValueTask<TOut?>> transform) : ICalculatorNode
{
    /// <inheritdoc />
    public void GetContract(CalculatorContract contract) => contract.AddInput<TIn>("IN").AddOutput<TOut>("OUT");

    /// <inheritdoc />
    public ValueTask OpenAsync(CalculatorContext context, CancellationToken cancellationToken) => ValueTask.CompletedTask;

    /// <inheritdoc />
    public async ValueTask ProcessAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        var result = await transform(context.GetInput<TIn>("IN"), cancellationToken).ConfigureAwait(false);
        if (result is not null) context.Send("OUT", result);
    }

    /// <inheritdoc />
    public ValueTask CloseAsync(CalculatorContext context, CancellationToken cancellationToken) => ValueTask.CompletedTask;
}

/// <summary>
/// Joins two synchronized streams: runs once per timestamp with whichever of <c>A</c> and <c>B</c>
/// are present (both inputs are optional per timestamp) and emits the combined value on <c>OUT</c>.
/// </summary>
/// <typeparam name="TA">Type of input A.</typeparam>
/// <typeparam name="TB">Type of input B.</typeparam>
/// <typeparam name="TOut">Output type.</typeparam>
/// <param name="combine">Combines the values; absent inputs are passed as default.</param>
public sealed class CombineNode<TA, TB, TOut>(Func<TA?, TB?, TOut?> combine) : CalculatorNode
{
    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) =>
        contract.AddInput<TA>("A").AddInput<TB>("B").AddOutput<TOut>("OUT");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context)
    {
        context.TryGetInput<TA>("A", out var a);
        context.TryGetInput<TB>("B", out var b);
        var result = combine(a, b);
        if (result is not null) context.Send("OUT", result);
    }
}

/// <summary>Invokes a callback for every packet and forwards nothing (a terminal node).</summary>
/// <typeparam name="T">Payload type.</typeparam>
/// <param name="callback">Receives each packet.</param>
public sealed class SinkNode<T>(Action<Packet<T>> callback) : CalculatorNode
{
    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<T>("IN");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context) => callback(context.GetInputPacket("IN")!.As<T>());
}

/// <summary>
/// Keeps at most one packet per <see cref="Period"/> and drops the rest (MediaPipe's
/// <c>PacketThinnerCalculator</c>, asynchronous mode). Useful to run an expensive branch at a lower rate.
/// </summary>
/// <param name="period">Minimum spacing between forwarded packets.</param>
public sealed class PacketThinnerNode(TimeSpan period) : CalculatorNode
{
    private Timestamp _next = Timestamp.Min;

    /// <summary>Minimum spacing between forwarded packets.</summary>
    public TimeSpan Period { get; } = period;

    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<object>("IN").AddOutput<object>("OUT");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context)
    {
        if (context.InputTimestamp < _next) return;
        context.SendPacket("OUT", context.GetInputPacket("IN")!);
        _next = new Timestamp(context.InputTimestamp.Value + (long)(Period.Ticks / 10));
    }
}

/// <summary>Counts packets and emits the running total on <c>COUNT</c> (handy for diagnostics).</summary>
public sealed class PacketCounterNode : CalculatorNode
{
    private long _count;

    /// <summary>Packets seen so far.</summary>
    public long Count => Interlocked.Read(ref _count);

    /// <inheritdoc />
    public override void GetContract(CalculatorContract contract) => contract.AddInput<object>("IN").AddOutput<long>("COUNT");

    /// <inheritdoc />
    protected override void Process(CalculatorContext context) => context.Send("COUNT", Interlocked.Increment(ref _count));
}
