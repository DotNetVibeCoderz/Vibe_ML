using System.Collections.Concurrent;
using ScienceAppGen.Services.Conversion;
using Xunit;

namespace ScienceAppGen.Tests;

/// <summary>An <see cref="ILlmClient"/> that answers from a script, per output file.</summary>
internal sealed class ScriptedLlm(Func<string, int, string> reply) : ILlmClient
{
    private readonly ConcurrentDictionary<string, int> _calls = new();

    public ConcurrentQueue<string> Prompts { get; } = new();

    public string Description => "scripted";

    public Task<string> CompleteAsync(string system, string user, CancellationToken cancellationToken)
    {
        Prompts.Enqueue(user);
        // The target file is named in the prompt: "Convert it to `X.cs`" or "## The current C# (X.cs)".
        var target = user.Contains("## The current C# (", StringComparison.Ordinal)
            ? user.Split("## The current C# (")[1].Split(')')[0]
            : user.Split("Convert it to `")[1].Split('`')[0];
        var n = _calls.AddOrUpdate(target, 1, (_, c) => c + 1);
        return Task.FromResult(reply(target, n));
    }
}

public class PythonConverterTests
{
    // Laid out as a model writes them: one member per line.
    private static string HelpersCs(string body) => $$"""
        ```csharp
        namespace Doubler;

        public static class Helpers
        {
            public static int Double(int x) => {{body}};
        }
        ```
        """;

    private const string ProgramCs = """
        ```csharp
        namespace Doubler;

        public static class Program
        {
            public static void Main(string[] args) => System.Console.WriteLine(Helpers.Double(21));
        }
        ```
        """;

    private static ConversionPlan TwoModulePlan(TempTree tree)
    {
        tree.Write("src/main.py", "from helpers import double\nif __name__ == '__main__':\n    print(double(21))\n");
        tree.Write("src/helpers.py", "def double(x):\n    return 2 * x\n");
        tree.Write("src/data/numbers.csv", "x\n1\n");
        return ConversionPlanner.Plan(PythonProjectScanner.Scan(Path.Combine(tree.Root, "src")), "Doubler");
    }

    [Fact]
    public async Task Writes_the_project_report_and_transcript_without_building()
    {
        using var tree = new TempTree();
        var plan = TwoModulePlan(tree);
        var llm = new ScriptedLlm((file, _) => file == "Helpers.cs" ? HelpersCs("2 * x") : ProgramCs);
        var output = Path.Combine(tree.Root, "out");

        var result = await new PythonConverter(llm).ConvertAsync(plan, new ConversionOptions { OutputDirectory = output, Build = false });

        Assert.True(result.Succeeded);
        Assert.Equal(2, result.LlmCalls);
        Assert.True(File.Exists(Path.Combine(output, "Doubler.csproj")));
        Assert.True(File.Exists(Path.Combine(output, "Helpers.cs")));
        Assert.True(File.Exists(Path.Combine(output, "Program.cs")));
        Assert.True(File.Exists(Path.Combine(output, "data", "numbers.csv")));          // data travels with the code
        Assert.True(File.Exists(Path.Combine(output, "CONVERSION.md")));
        Assert.True(File.Exists(Path.Combine(output, ".conversion", "0-convert-Helpers.cs.reply.md")));

        // The entry point was converted after its dependency and saw the signature it really got.
        var entryPrompt = llm.Prompts.Last();
        Assert.Contains("Convert it to `Program.cs`", entryPrompt);
        Assert.Contains("public static int Double(int x)", entryPrompt);
    }

    [Fact]
    public async Task A_folder_that_is_not_a_previous_conversion_is_refused()
    {
        using var tree = new TempTree();
        var plan = TwoModulePlan(tree);
        tree.Write("busy/notes.txt", "mine");

        await Assert.ThrowsAsync<InvalidOperationException>(() => new PythonConverter(new ScriptedLlm((_, _) => ""))
            .ConvertAsync(plan, new ConversionOptions { OutputDirectory = Path.Combine(tree.Root, "busy"), Build = false }));
        Assert.Equal("mine", File.ReadAllText(Path.Combine(tree.Root, "busy", "notes.txt")));
    }

    /// <summary>
    /// The whole loop with a real <c>dotnet build</c>: the first version of a file does not
    /// compile, the converter sends it back with the compiler's errors, the second compiles, and
    /// the program runs. Uses no Gravicode package (stdlib-only Python), so it needs no NuGet feed.
    /// </summary>
    [Fact]
    public async Task Compiler_errors_go_back_to_the_model_until_the_project_builds()
    {
        using var tree = new TempTree();
        var plan = TwoModulePlan(tree);
        var llm = new ScriptedLlm((file, call) => (file, call) switch
        {
            ("Helpers.cs", 1) => HelpersCs("2 * y"),   // does not compile: there is no y
            ("Helpers.cs", _) => HelpersCs("2 * x"),
            _ => ProgramCs,
        });
        var output = Path.Combine(tree.Root, "out");
        var events = new ConcurrentQueue<ConversionEvent>();

        var result = await new PythonConverter(llm).ConvertAsync(plan,
            new ConversionOptions { OutputDirectory = output, MaxRepairRounds = 2, RunAfterBuild = true },
            new SyncProgress(events.Enqueue));

        Assert.True(result.Succeeded, string.Join("\n", result.RemainingErrors));
        Assert.Equal(1, result.RepairRounds);
        Assert.Equal(3, result.LlmCalls);
        Assert.Equal(0, result.ProgramExitCode);
        Assert.Contains("42", result.ProgramOutput);

        // The repair prompt carried the real compiler error for the real line.
        var repair = llm.Prompts.Single(p => p.Contains("## Compiler errors", StringComparison.Ordinal));
        Assert.Contains("Helpers.cs(5,", repair);
        Assert.Contains("CS0103", repair);                              // the name 'y' does not exist
        Assert.Contains(events, e => e.Status == ModuleStatus.Compiles && e.ModulePath == "helpers.py");
    }

    /// <summary>
    /// A version that compiles and then throws is a repair job too: the Python original ran, so the
    /// conversion must. The model gets the exception and the stack, pinned to the line.
    /// </summary>
    [Fact]
    public async Task A_program_that_builds_but_crashes_is_sent_back_with_its_exception()
    {
        using var tree = new TempTree();
        var plan = TwoModulePlan(tree);
        var llm = new ScriptedLlm((file, call) => (file, call) switch
        {
            ("Helpers.cs", 1) => HelpersCs("throw new System.InvalidOperationException(\"no column named mean\")"),
            ("Helpers.cs", _) => HelpersCs("2 * x"),
            _ => ProgramCs,
        });

        var result = await new PythonConverter(llm).ConvertAsync(plan,
            new ConversionOptions { OutputDirectory = Path.Combine(tree.Root, "out"), MaxRepairRounds = 2, RunAfterBuild = true });

        Assert.Equal(0, result.ProgramExitCode);
        Assert.Contains("42", result.ProgramOutput);
        // Both the throw site and its caller go back: the fix may belong to either.
        var repairs = llm.Prompts.Where(p => p.Contains("## Run-time failure", StringComparison.Ordinal)).ToList();
        Assert.Equal(2, repairs.Count);
        var throwSite = Assert.Single(repairs, p => p.Contains("## The current C# (Helpers.cs)", StringComparison.Ordinal));
        Assert.Contains("no column named mean", throwSite);
        Assert.Contains("compiles but crashes at run time", throwSite);
        Assert.Contains(repairs, p => p.Contains("## The current C# (Program.cs)", StringComparison.Ordinal));
    }

    private sealed class SyncProgress(Action<ConversionEvent> handler) : IProgress<ConversionEvent>
    {
        public void Report(ConversionEvent value) => handler(value);
    }
}
