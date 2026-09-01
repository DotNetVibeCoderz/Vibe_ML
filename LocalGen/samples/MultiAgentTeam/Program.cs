// Several agents working on one task, in one process, on one local model.
//
// The three patterns differ in who decides what runs next, which is the thing that actually
// matters when the model is small: sequential and concurrent leave nothing to its judgement,
// while handoff asks it to route the work itself.
//
//   dotnet run --project samples/MultiAgentTeam -- [sequential|concurrent|handoff] [model-id]

using LocalGen.Engines.LlamaSharp;
using LocalGen.Kernel;
using LocalGen.Kernel.Orchestration;
using LocalGen.Runtime;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

var pattern = (args.FirstOrDefault() ?? "sequential").ToLowerInvariant();

if (pattern is not ("sequential" or "concurrent" or "handoff"))
{
    Console.Error.WriteLine("Usage: dotnet run --project samples/MultiAgentTeam -- [sequential|concurrent|handoff] [model-id]");
    return 1;
}

var configuration = new ConfigurationBuilder()
    .AddInMemoryCollection(new Dictionary<string, string?>
    {
        // Concurrent members would otherwise queue behind one another on the same weights, and
        // the fan-out would buy nothing but tidier code.
        ["LocalGen:Engine:BatchedInference"] = pattern == "concurrent" ? "true" : "false",
        ["LocalGen:Engine:MaxBatchedSequences"] = "3"
    })
    .AddEnvironmentVariables("LOCALGEN_")
    .Build();

var services = new ServiceCollection();
services.AddLogging(b => b.AddSimpleConsole(c => c.SingleLine = true).SetMinimumLevel(LogLevel.Warning));
services.AddLocalGenRuntime(configuration);
services.AddLlamaSharpEngine();
services.AddLocalGenKernel();

await using var provider = services.BuildServiceProvider();

var store = provider.GetRequiredService<LocalGen.Core.Models.IModelStore>();
var installed = await store.ListAsync();

if (installed.Count == 0)
{
    Console.Error.WriteLine("No models installed. Try:");
    Console.Error.WriteLine("  localgen pull huggingface:bartowski/Qwen2.5-1.5B-Instruct-GGUF");
    return 1;
}

var model = args.Skip(1).FirstOrDefault()
            ?? installed.FirstOrDefault(m => m.Id.Contains("instruct", StringComparison.OrdinalIgnoreCase))?.Id
            ?? installed[0].Id;

// Each member gets a narrow brief. A vague instruction produces a vague agent, and with several
// of them the vagueness compounds rather than averaging out.
AgentDefinition[] roster =
[
    new()
    {
        Name = "Researcher",
        Description = "Lists the concrete facts and considerations a topic involves.",
        Instructions =
            "You list facts. Given a topic, write 4-6 short bullet points of concrete, " +
            "specific information about it. No preamble, no conclusion, bullets only."
    },
    new()
    {
        Name = "Critic",
        Description = "Finds what is missing, wrong or overstated in a draft.",
        Instructions =
            "You review drafts. Point out anything missing, wrong or overstated, as 2-4 short " +
            "bullet points. Be specific and brief. No preamble."
    },
    new()
    {
        Name = "Writer",
        Description = "Turns notes into one clear, finished paragraph.",
        Instructions =
            "You write final copy. Turn the material you are given into exactly one clear " +
            "paragraph of plain prose. No bullets, no headings, no preamble."
    }
];

const string Task = "Explain why local, on-device AI inference is useful for a small business.";

var factory = provider.GetRequiredService<LocalGenKernelFactory>();
var team = await AgentTeam.CreateAsync(
    factory, model, roster, provider.GetRequiredService<ILoggerFactory>());

Console.WriteLine($"Model   : {model}");
Console.WriteLine($"Pattern : {pattern}");
Console.WriteLine($"Team    : {string.Join(", ", team.Members.Select(m => m.Name))}");
Console.WriteLine($"Task    : {Task}");
Console.WriteLine(new string('─', 78));

var stream = pattern switch
{
    "sequential" => team.RunSequentialAsync(Task),
    "concurrent" => team.RunConcurrentAsync(Task, roster[2]),
    _ => team.RunHandoffAsync(Task)
};

var started = DateTimeOffset.UtcNow;
var final = string.Empty;

await foreach (var evt in stream)
{
    switch (evt)
    {
        case OrchestrationEvent.AgentStarted started2:
            Console.WriteLine();
            Console.WriteLine($"▸ {started2.Agent} ← {Truncate(started2.Task, 90)}");
            break;

        case OrchestrationEvent.AgentActivity { Event: AgentEvent.ToolCallStarted call }:
            Console.WriteLine($"    ↳ calling {call.Tool}");
            break;

        case OrchestrationEvent.AgentFinished finished:
            Console.WriteLine($"  {finished.Agent} finished in {finished.Duration.TotalSeconds:N1}s");
            Console.WriteLine($"  {Indent(finished.Result)}");
            break;

        case OrchestrationEvent.Completed completed:
            final = completed.Result;
            break;

        case OrchestrationEvent.Failed failed:
            Console.Error.WriteLine($"✗ {failed.Message}");
            return 1;
    }
}

Console.WriteLine();
Console.WriteLine(new string('─', 78));
Console.WriteLine($"Final answer ({(DateTimeOffset.UtcNow - started).TotalSeconds:N1}s):");
Console.WriteLine();
Console.WriteLine(final);

// Releases the weights before the process exits, so the native memory is freed deterministically.
await provider.GetRequiredService<ModelSessionManager>().DisposeAsync();
return 0;

static string Truncate(string text, int max)
{
    var flat = text.ReplaceLineEndings(" ").Trim();
    return flat.Length <= max ? flat : flat[..max] + "…";
}

static string Indent(string text) =>
    string.Join("\n  ", Truncate(text, 400).Split('\n'));
