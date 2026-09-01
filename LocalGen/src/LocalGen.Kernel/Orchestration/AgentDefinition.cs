using System.Text;

namespace LocalGen.Kernel.Orchestration;

/// <summary>
/// One member of a team: what it is for, how it should behave, and what it is allowed to touch.
/// </summary>
/// <remarks>
/// A definition is deliberately declarative — it names a model and a tool selection rather than
/// holding a built agent — so a team can be described in configuration and instantiated per
/// conversation. Tools default to none: an agent that only summarises should not also be able to
/// execute code because a teammate needed to.
/// </remarks>
public sealed record AgentDefinition
{
    /// <summary>
    /// Short identifier, used in transcripts and — for the handoff pattern — in the name of the
    /// function the coordinator calls to reach this agent.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// What this agent is good at. In a handoff team this is the only thing the coordinator has
    /// to route on, so it should read like a job description rather than a label.
    /// </summary>
    public required string Description { get; init; }

    /// <summary>System prompt establishing the agent's role and constraints.</summary>
    public required string Instructions { get; init; }

    /// <summary>Model to run on. Null uses the team's default.</summary>
    public string? Model { get; init; }

    /// <summary>Functions this agent may call. None by default.</summary>
    public ToolSelection Tools { get; init; } = ToolSelection.None;

    /// <summary>Iteration and truncation limits. Null uses the team's default.</summary>
    public AgentOptions? Options { get; init; }

    /// <summary>
    /// The name reduced to characters a function name may contain, for the delegation tool.
    /// </summary>
    /// <remarks>
    /// Tool names travel through a prompt convention that splits a qualified name on its first
    /// hyphen, and a model asked to call <c>ask_data-analyst</c> will sometimes drop the
    /// punctuation it did not expect. Normalising here means the agent can still be called
    /// "Data Analyst" everywhere a person reads it.
    /// </remarks>
    public string FunctionSafeName
    {
        get
        {
            var sb = new StringBuilder(Name.Length);

            foreach (var character in Name)
            {
                sb.Append(char.IsAsciiLetterOrDigit(character) ? char.ToLowerInvariant(character) : '_');
            }

            var normalised = sb.ToString().Trim('_');
            return normalised.Length > 0 ? normalised : "agent";
        }
    }
}
