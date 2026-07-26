using System.Diagnostics;
using System.Text;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Jint;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;

namespace BlazorML.ML.Scripting;

/// <summary>
/// Runs a C# snippet in-process through Roslyn. The script sees <c>rows</c> and <c>rows2</c> as
/// <c>List&lt;Dictionary&lt;string, object&gt;&gt;</c> and returns the rows to pass downstream.
/// </summary>
public sealed class CSharpScriptRunner : IScriptRunner
{
    public ScriptLanguage Language => ScriptLanguage.CSharp;

    public string UnavailableReason => "C# scripting is compiled into the app and is always available.";

    public Task<bool> IsAvailableAsync(CancellationToken ct = default) => Task.FromResult(true);

    public async Task<TabularData> RunAsync(string code, TabularData? input1, TabularData? input2,
        TimeSpan timeout, CancellationToken ct = default)
    {
        var globals = new ScriptGlobals
        {
            rows = input1?.ToDictionaries() ?? [],
            rows2 = input2?.ToDictionaries() ?? []
        };

        var options = ScriptOptions.Default
            .WithImports("System", "System.Linq", "System.Collections.Generic", "System.Math")
            .WithReferences(typeof(Enumerable).Assembly, typeof(object).Assembly);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutSource.CancelAfter(timeout);

        try
        {
            var result = await CSharpScript.EvaluateAsync<object?>(code, options, globals,
                cancellationToken: timeoutSource.Token);

            return Materialise(result, input1);
        }
        catch (CompilationErrorException e)
        {
            throw new InvalidOperationException(
                "The C# script did not compile:\n" + string.Join('\n', e.Diagnostics), e);
        }
    }

    public class ScriptGlobals
    {
        // Lower-case on purpose: these are the names the script author types.
        public List<Dictionary<string, object?>> rows { get; set; } = [];
        public List<Dictionary<string, object?>> rows2 { get; set; } = [];
    }

    internal static TabularData Materialise(object? result, TabularData? fallback)
    {
        return result switch
        {
            null => fallback ?? new TabularData(),
            TabularData table => table,
            IEnumerable<Dictionary<string, object?>> rows => TabularData.FromDictionaries(rows),
            _ => throw new InvalidOperationException(
                "The script must return a list of dictionaries — the same shape as `rows`.")
        };
    }
}

/// <summary>
/// Runs a JavaScript snippet through Jint. The script sees <c>rows</c> and <c>rows2</c> as arrays
/// of objects and returns the array to pass downstream.
/// </summary>
public sealed class JavaScriptRunner : IScriptRunner
{
    public ScriptLanguage Language => ScriptLanguage.JavaScript;

    public string UnavailableReason => "The JavaScript engine ships with the app and is always available.";

    public Task<bool> IsAvailableAsync(CancellationToken ct = default) => Task.FromResult(true);

    public Task<TabularData> RunAsync(string code, TabularData? input1, TabularData? input2,
        TimeSpan timeout, CancellationToken ct = default)
    {
        var engine = new Jint.Engine(options => options
            .TimeoutInterval(timeout)
            .LimitMemory(256 * 1024 * 1024)
            .CancellationToken(ct));

        // Values cross the boundary as JSON so the script sees plain objects rather than
        // .NET types it would have to know about.
        engine.SetValue("rows", ToJsArray(engine, input1));
        engine.SetValue("rows2", ToJsArray(engine, input2));

        // Evaluated once into a global, then serialised from that global. Evaluating the user's
        // code a second time to stringify it would run their side effects twice.
        engine.Execute($"var __result = (function() {{\n{code}\n}})();");

        var json = engine.Evaluate(
            "__result === null || __result === undefined ? '' : JSON.stringify(__result)").ToString();

        if (string.IsNullOrEmpty(json))
        {
            return Task.FromResult(input1 ?? new TabularData());
        }

        var rows = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json) ?? [];

        var converted = rows.Select(r => r.ToDictionary(
            kv => kv.Key,
            kv => (object?)(kv.Value.ValueKind switch
            {
                JsonValueKind.Number => kv.Value.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                _ => kv.Value.ToString()
            })));

        return Task.FromResult(TabularData.FromDictionaries(converted));
    }

    private static Jint.Native.JsValue ToJsArray(Jint.Engine engine, TabularData? table)
    {
        var json = JsonSerializer.Serialize(table?.ToDictionaries() ?? []);
        return engine.Evaluate($"JSON.parse({JsonSerializer.Serialize(json)})");
    }
}

/// <summary>
/// Runs an R script by shelling out to <c>Rscript</c>. Data crosses as CSV files, which every R
/// install can read without extra packages.
/// </summary>
public sealed class RScriptRunner(ScriptingOptions options) : IScriptRunner
{
    public ScriptLanguage Language => ScriptLanguage.R;

    public string UnavailableReason =>
        $"R was not found. Install R and make sure '{options.RscriptPath}' is on the PATH, " +
        "or set the full path under Settings → Scripting.";

    public async Task<bool> IsAvailableAsync(CancellationToken ct = default)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(options.RscriptPath, "--version")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            if (process is null)
            {
                return false;
            }

            await process.WaitForExitAsync(ct);
            return process.ExitCode == 0;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public async Task<TabularData> RunAsync(string code, TabularData? input1, TabularData? input2,
        TimeSpan timeout, CancellationToken ct = default)
    {
        var workspace = Path.Combine(Path.GetTempPath(), $"blazorml-r-{Guid.NewGuid():n}");
        Directory.CreateDirectory(workspace);

        try
        {
            var input1Path = Path.Combine(workspace, "dataset1.csv");
            var input2Path = Path.Combine(workspace, "dataset2.csv");
            var outputPath = Path.Combine(workspace, "result.csv");
            var scriptPath = Path.Combine(workspace, "script.R");

            await File.WriteAllTextAsync(input1Path, (input1 ?? new TabularData()).ToCsv(), ct);
            await File.WriteAllTextAsync(input2Path, (input2 ?? new TabularData()).ToCsv(), ct);

            var harness = new StringBuilder();
            harness.AppendLine($"dataset1 <- read.csv({Quote(input1Path)}, stringsAsFactors = FALSE)");
            harness.AppendLine($"dataset2 <- read.csv({Quote(input2Path)}, stringsAsFactors = FALSE)");
            harness.AppendLine("result <- dataset1");
            harness.AppendLine(code);
            harness.AppendLine($"write.csv(result, {Quote(outputPath)}, row.names = FALSE)");

            await File.WriteAllTextAsync(scriptPath, harness.ToString(), ct);

            var (exitCode, stdout, stderr) = await RunProcessAsync(
                options.RscriptPath, $"--vanilla \"{scriptPath}\"", timeout, ct);

            if (exitCode != 0)
            {
                throw new InvalidOperationException(
                    $"The R script failed (exit code {exitCode}):\n{stderr}\n{stdout}");
            }

            if (!File.Exists(outputPath))
            {
                throw new InvalidOperationException(
                    "The R script finished but wrote no result. Assign your output to `result`.");
            }

            await using var stream = File.OpenRead(outputPath);
            return await Infrastructure.Datasets.TabularSerializer.ReadAsync(
                stream, DatasetFormat.Csv, 0, ct);
        }
        finally
        {
            try
            {
                Directory.Delete(workspace, recursive: true);
            }
            catch (IOException)
            {
                // Leaving a temp directory behind is not worth failing the run over.
            }
        }
    }

    private static string Quote(string path) => "\"" + path.Replace("\\", "/") + "\"";

    internal static async Task<(int ExitCode, string Stdout, string Stderr)> RunProcessAsync(
        string fileName, string arguments, TimeSpan timeout, CancellationToken ct)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo(fileName, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };

        process.Start();

        var stdout = process.StandardOutput.ReadToEndAsync(ct);
        var stderr = process.StandardError.ReadToEndAsync(ct);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutSource.CancelAfter(timeout);

        try
        {
            await process.WaitForExitAsync(timeoutSource.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
                // Already gone.
            }

            throw new TimeoutException($"The script did not finish within {timeout.TotalSeconds:N0} seconds.");
        }

        return (process.ExitCode, await stdout, await stderr);
    }
}
