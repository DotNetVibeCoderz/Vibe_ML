namespace MediaPipeNet.Framework;

/// <summary>
/// A processing unit of a <see cref="CalculatorGraph"/> — MediaPipe's <c>CalculatorBase</c>.
/// A node declares its ports in <see cref="GetContract"/>, is opened once when the graph starts,
/// processes one timestamp at a time (never concurrently with itself) and is closed when all its
/// inputs are done.
/// </summary>
public interface ICalculatorNode
{
    /// <summary>Declares the node's input/output streams and side packets.</summary>
    void GetContract(CalculatorContract contract);

    /// <summary>Called once before any packet is processed; side packets are available.</summary>
    ValueTask OpenAsync(CalculatorContext context, CancellationToken cancellationToken);

    /// <summary>Processes the inputs of <see cref="CalculatorContext.InputTimestamp"/>.</summary>
    ValueTask ProcessAsync(CalculatorContext context, CancellationToken cancellationToken);

    /// <summary>Called once after all inputs are done (or the graph is cancelled).</summary>
    ValueTask CloseAsync(CalculatorContext context, CancellationToken cancellationToken);
}

/// <summary>Convenience base class for nodes with synchronous logic.</summary>
public abstract class CalculatorNode : ICalculatorNode
{
    /// <inheritdoc />
    public abstract void GetContract(CalculatorContract contract);

    /// <summary>Synchronous open hook.</summary>
    protected virtual void Open(CalculatorContext context) { }

    /// <summary>Synchronous process hook.</summary>
    protected abstract void Process(CalculatorContext context);

    /// <summary>Synchronous close hook.</summary>
    protected virtual void Close(CalculatorContext context) { }

    /// <inheritdoc />
    public virtual ValueTask OpenAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        Open(context);
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public virtual ValueTask ProcessAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        Process(context);
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public virtual ValueTask CloseAsync(CalculatorContext context, CancellationToken cancellationToken)
    {
        Close(context);
        return ValueTask.CompletedTask;
    }
}
