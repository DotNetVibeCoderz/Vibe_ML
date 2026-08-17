namespace LocalGen.Kernel.Orchestration;

/// <summary>
/// Something that happened while a team was working.
/// </summary>
/// <remarks>
/// Multi-agent runs are opaque when all you see is the final answer — most of the value of
/// watching one is knowing which agent is speaking and why it was asked. Every event therefore
/// carries the agent it belongs to, and a member's own <see cref="AgentEvent"/> stream is wrapped
/// rather than flattened so the Playground can keep showing tool calls exactly as it does for a
/// single agent.
/// </remarks>
public abstract record OrchestrationEvent
{
    /// <summary>An agent has been given a task.</summary>
    public sealed record AgentStarted(string Agent, string Task) : OrchestrationEvent;

    /// <summary>Something that agent did — text, a tool call, a failure.</summary>
    public sealed record AgentActivity(string Agent, AgentEvent Event) : OrchestrationEvent;

    /// <summary>An agent produced its answer.</summary>
    public sealed record AgentFinished(string Agent, string Result, TimeSpan Duration) : OrchestrationEvent;

    /// <summary>The team produced its answer.</summary>
    public sealed record Completed(string Result) : OrchestrationEvent;

    /// <summary>The run failed and produced no answer.</summary>
    public sealed record Failed(string Message) : OrchestrationEvent;
}
