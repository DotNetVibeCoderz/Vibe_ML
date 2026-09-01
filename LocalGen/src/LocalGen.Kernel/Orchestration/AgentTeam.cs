using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Kernel.Orchestration;

/// <summary>
/// Several agents working on one task.
/// </summary>
/// <remarks>
/// <see cref="LocalGenAgent"/> drives a single agent through its tool calls. A team goes one level
/// up and decides which agent runs when. Three shapes are offered because they fail in different
/// ways, and on a local model that matters more than it does against a frontier one:
///
/// <list type="bullet">
/// <item><description><b>Sequential</b> — a fixed pipeline. Nothing is left to the model's
/// judgement, so a 1.5B model runs it as reliably as a 70B one.</description></item>
/// <item><description><b>Concurrent</b> — every agent answers the same question and a reducer
/// combines them. Also fully deterministic in its routing.</description></item>
/// <item><description><b>Handoff</b> — a coordinator picks specialists itself by calling them as
/// tools. The most flexible and the most demanding: a model that cannot reliably choose a tool
/// cannot reliably choose a colleague either.</description></item>
/// </list>
///
/// Concurrent runs benefit directly from <c>Engine.BatchedInference</c>: without it the members
/// queue behind one another on the same model and the fan-out buys nothing but tidier code.
/// </remarks>
public sealed class AgentTeam
{
    private readonly IReadOnlyList<Member> _members;
    private readonly LocalGenKernelFactory _factory;
    private readonly string _defaultModel;
    private readonly ILogger<AgentTeam> _logger;

    private AgentTeam(
        IReadOnlyList<Member> members,
        LocalGenKernelFactory factory,
        string defaultModel,
        ILogger<AgentTeam> logger)
    {
        _members = members;
        _factory = factory;
        _defaultModel = defaultModel;
        _logger = logger;
    }

    /// <summary>The team's members, in the order they were defined.</summary>
    public IReadOnlyList<AgentDefinition> Members => [.. _members.Select(static m => m.Definition)];

    /// <summary>
    /// Builds every member's agent up front.
    /// </summary>
    /// <remarks>
    /// Done eagerly because building a kernel can mean starting MCP servers as child processes,
    /// and discovering that halfway through a delegation would strand the run.
    /// </remarks>
    public static async Task<AgentTeam> CreateAsync(
        LocalGenKernelFactory factory,
        string defaultModel,
        IEnumerable<AgentDefinition> definitions,
        ILoggerFactory? loggerFactory = null,
        CancellationToken cancellationToken = default)
    {
        var logs = loggerFactory ?? Microsoft.Extensions.Logging.Abstractions.NullLoggerFactory.Instance;
        var members = new List<Member>();

        foreach (var definition in definitions)
        {
            var agent = await factory
                .CreateAgentAsync(
                    definition.Model ?? defaultModel,
                    definition.Tools,
                    definition.Options,
                    cancellationToken)
                .ConfigureAwait(false);

            members.Add(new Member(definition, agent));
        }

        if (members.Count == 0)
        {
            throw new ArgumentException("A team needs at least one agent.", nameof(definitions));
        }

        return new AgentTeam(members, factory, defaultModel, logs.CreateLogger<AgentTeam>());
    }

    /// <summary>
    /// Runs every member in turn, each seeing the original task and the previous agent's answer.
    /// </summary>
    /// <remarks>
    /// The original task is repeated to each member rather than only the running result. A chain
    /// that passes along nothing but the last output drifts: by the third hop the agent is
    /// polishing prose without knowing what was asked.
    /// </remarks>
    public async IAsyncEnumerable<OrchestrationEvent> RunSequentialAsync(
        string task,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var carried = string.Empty;

        for (var i = 0; i < _members.Count; i++)
        {
            var member = _members[i];

            var prompt = i == 0
                ? task
                : $"""
                   Original request:
                   {task}

                   Previous agent ({_members[i - 1].Definition.Name}) produced:
                   {carried}

                   Do your part of the work on this.
                   """;

            await foreach (var evt in RunMemberAsync(member, prompt, cancellationToken).ConfigureAwait(false))
            {
                if (evt is OrchestrationEvent.AgentFinished finished)
                {
                    carried = finished.Result;
                }

                if (evt is OrchestrationEvent.Failed)
                {
                    yield return evt;
                    yield break;
                }

                yield return evt;
            }
        }

        yield return new OrchestrationEvent.Completed(carried);
    }

    /// <summary>
    /// Asks every member the same question at once, then optionally has a reducer combine them.
    /// </summary>
    /// <param name="reducer">
    /// Agent that merges the answers. When null the answers are concatenated under their authors'
    /// names, which is the honest result when no one has been asked to reconcile them.
    /// </param>
    public async IAsyncEnumerable<OrchestrationEvent> RunConcurrentAsync(
        string task,
        AgentDefinition? reducer = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var events = Channel.CreateUnbounded<OrchestrationEvent>();
        var answers = new string[_members.Count];

        var work = Task.Run(async () =>
        {
            try
            {
                await Task.WhenAll(_members.Select(async (member, index) =>
                {
                    await foreach (var evt in RunMemberAsync(member, task, cancellationToken).ConfigureAwait(false))
                    {
                        if (evt is OrchestrationEvent.AgentFinished finished)
                        {
                            answers[index] = finished.Result;
                        }

                        await events.Writer.WriteAsync(evt, cancellationToken).ConfigureAwait(false);
                    }
                })).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                await events.Writer.WriteAsync(new OrchestrationEvent.Failed(ex.Message), CancellationToken.None)
                    .ConfigureAwait(false);
            }
            finally
            {
                events.Writer.TryComplete();
            }
        }, cancellationToken);

        await foreach (var evt in events.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return evt;
        }

        await work.ConfigureAwait(false);

        var combined = new StringBuilder();

        for (var i = 0; i < _members.Count; i++)
        {
            combined.Append("## ").AppendLine(_members[i].Definition.Name)
                    .AppendLine(answers[i] ?? "(no answer)").AppendLine();
        }

        if (reducer is null)
        {
            yield return new OrchestrationEvent.Completed(combined.ToString().TrimEnd());
            yield break;
        }

        var reducerMember = await BuildMemberAsync(reducer, cancellationToken).ConfigureAwait(false);
        var reducerPrompt = $"""
            Original request:
            {task}

            {_members.Count} agents answered it independently:

            {combined}
            Produce one consolidated answer. Resolve disagreements rather than listing them.
            """;

        var final = string.Empty;

        await foreach (var evt in RunMemberAsync(reducerMember, reducerPrompt, cancellationToken)
            .ConfigureAwait(false))
        {
            if (evt is OrchestrationEvent.AgentFinished finished)
            {
                final = finished.Result;
            }

            yield return evt;
        }

        yield return new OrchestrationEvent.Completed(final);
    }

    /// <summary>
    /// Gives a coordinator the other members as callable tools and lets it route the work itself.
    /// </summary>
    /// <remarks>
    /// A delegation is modelled as an ordinary tool call, which is what makes this work at all on
    /// a local model: no new protocol is introduced, so a model that can call a function can
    /// consult a colleague, and the existing stream filter and dialect handling apply unchanged.
    /// </remarks>
    public async IAsyncEnumerable<OrchestrationEvent> RunHandoffAsync(
        string task,
        AgentDefinition? coordinator = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var definition = coordinator ?? DefaultCoordinator(_members);
        var events = Channel.CreateUnbounded<OrchestrationEvent>();

        var kernel = await _factory
            .CreateAsync(definition.Model ?? _defaultModel, definition.Tools, cancellationToken)
            .ConfigureAwait(false);

        kernel.Plugins.Add(BuildDelegationPlugin(events, cancellationToken));

        var agent = new LocalGenAgent(kernel, definition.Options);

        var work = Task.Run(async () =>
        {
            var answer = new StringBuilder();

            try
            {
                var history = new ChatHistory();
                history.AddSystemMessage(definition.Instructions);
                history.AddUserMessage(task);

                await events.Writer
                    .WriteAsync(new OrchestrationEvent.AgentStarted(definition.Name, task), cancellationToken)
                    .ConfigureAwait(false);

                await foreach (var evt in agent.StreamAsync(history, cancellationToken: cancellationToken)
                    .ConfigureAwait(false))
                {
                    if (evt is AgentEvent.TextDelta delta)
                    {
                        answer.Append(delta.Text);
                    }

                    await events.Writer
                        .WriteAsync(new OrchestrationEvent.AgentActivity(definition.Name, evt), cancellationToken)
                        .ConfigureAwait(false);

                    if (evt is AgentEvent.Failed failed)
                    {
                        await events.Writer
                            .WriteAsync(new OrchestrationEvent.Failed(failed.Message), cancellationToken)
                            .ConfigureAwait(false);
                        return;
                    }
                }

                await events.Writer
                    .WriteAsync(new OrchestrationEvent.Completed(answer.ToString().Trim()), cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Handoff orchestration failed.");

                await events.Writer
                    .WriteAsync(new OrchestrationEvent.Failed(ex.Message), CancellationToken.None)
                    .ConfigureAwait(false);
            }
            finally
            {
                events.Writer.TryComplete();
            }
        }, cancellationToken);

        await foreach (var evt in events.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return evt;
        }

        await work.ConfigureAwait(false);
    }

    /// <summary>
    /// Exposes each member as a function the coordinator can call.
    /// </summary>
    /// <remarks>
    /// The specialist's own events are written to the shared channel as the call runs, so the
    /// caller sees the sub-agent working rather than a tool that blocks for thirty seconds and
    /// returns a paragraph.
    /// </remarks>
    private KernelPlugin BuildDelegationPlugin(
        Channel<OrchestrationEvent> events,
        CancellationToken cancellationToken)
    {
        var functions = new List<KernelFunction>(_members.Count);

        foreach (var member in _members)
        {
            var captured = member;

            functions.Add(KernelFunctionFactory.CreateFromMethod(
                async (string task) =>
                {
                    var result = string.Empty;

                    await foreach (var evt in RunMemberAsync(captured, task, cancellationToken)
                        .ConfigureAwait(false))
                    {
                        if (evt is OrchestrationEvent.AgentFinished finished)
                        {
                            result = finished.Result;
                        }

                        await events.Writer.WriteAsync(evt, cancellationToken).ConfigureAwait(false);
                    }

                    return result;
                },
                functionName: $"ask_{captured.Definition.FunctionSafeName}",
                description:
                    $"Delegates a task to {captured.Definition.Name}. {captured.Definition.Description}",
                parameters:
                [
                    new KernelParameterMetadata("task")
                    {
                        Description =
                            "The complete, self-contained instruction for this agent. It cannot " +
                            "see the conversation, so include everything it needs.",
                        ParameterType = typeof(string),
                        IsRequired = true
                    }
                ]));
        }

        return KernelPluginFactory.CreateFromFunctions("Team", functions);
    }

    /// <summary>Runs one member on one task, wrapping its events with its name.</summary>
    private async IAsyncEnumerable<OrchestrationEvent> RunMemberAsync(
        Member member,
        string task,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var answer = new StringBuilder();
        var name = member.Definition.Name;

        yield return new OrchestrationEvent.AgentStarted(name, task);

        var history = new ChatHistory();
        history.AddSystemMessage(member.Definition.Instructions);
        history.AddUserMessage(task);

        await foreach (var evt in member.Agent
            .StreamAsync(history, cancellationToken: cancellationToken)
            .ConfigureAwait(false))
        {
            if (evt is AgentEvent.TextDelta delta)
            {
                answer.Append(delta.Text);
            }

            yield return new OrchestrationEvent.AgentActivity(name, evt);

            if (evt is AgentEvent.Failed failed)
            {
                yield return new OrchestrationEvent.Failed($"{name}: {failed.Message}");
                yield break;
            }
        }

        yield return new OrchestrationEvent.AgentFinished(name, answer.ToString().Trim(), stopwatch.Elapsed);
    }

    private async Task<Member> BuildMemberAsync(
        AgentDefinition definition,
        CancellationToken cancellationToken)
    {
        var agent = await _factory
            .CreateAgentAsync(
                definition.Model ?? _defaultModel,
                definition.Tools,
                definition.Options,
                cancellationToken)
            .ConfigureAwait(false);

        return new Member(definition, agent);
    }

    /// <summary>
    /// The coordinator used when the caller does not supply one.
    /// </summary>
    /// <remarks>
    /// Two things here are concessions to small models, learned by watching one ignore both.
    ///
    /// The roster is written into the prose as well as being visible in the tool descriptions:
    /// a model that skims a tool list will still read its instructions, and the duplication costs
    /// a few dozen tokens.
    ///
    /// Delegating is then stated as a requirement rather than an option. A 1.5B coordinator given
    /// the polite version ("delegate each part to the specialist best suited to it") simply
    /// answers the question itself — it is perfectly capable of producing *an* answer, so the
    /// path of least resistance is to skip the team entirely. Since a coordinator that never
    /// delegates is not a coordinator, the instruction says so outright.
    /// </remarks>
    private static AgentDefinition DefaultCoordinator(IReadOnlyList<Member> members)
    {
        var roster = new StringBuilder();

        foreach (var member in members)
        {
            roster.Append("- ").Append("ask_").Append(member.Definition.FunctionSafeName)
                  .Append(" — ").Append(member.Definition.Name)
                  .Append(": ").AppendLine(member.Definition.Description);
        }

        return new AgentDefinition
        {
            Name = "Coordinator",
            Description = "Routes work to the right specialist and assembles the final answer.",
            Instructions =
                $"""
                 You coordinate a team of specialists. You must not answer the request yourself
                 from your own knowledge — your job is to route it and then assemble what comes
                 back.

                 Your team:

                 {roster}
                 Rules:
                 1. Call at least one ask_* function before writing any final answer.
                 2. Give each specialist a complete, self-contained instruction. They cannot see
                    this conversation and know nothing about the original request.
                 3. Once the specialists have given you what you need, write the final answer
                    yourself, based on their replies rather than on your own knowledge.
                 """,
            Tools = ToolSelection.None
        };
    }

    private sealed record Member(AgentDefinition Definition, LocalGenAgent Agent);
}
