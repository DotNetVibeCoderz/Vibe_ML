using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using BlazorML.ML.Scripting;

namespace BlazorML.Tests;

/// <summary>
/// Marks a test that needs an interpreter installed on the machine. When it is missing the test
/// is reported as <em>skipped</em>, with the reason, rather than passing silently or failing —
/// an absent runtime is a fact about the environment, not a defect in the code.
/// </summary>
public sealed class RequiresRuntimeFactAttribute : FactAttribute
{
    public RequiresRuntimeFactAttribute(ScriptLanguage language)
    {
        if (!RuntimeProbe.IsAvailable(language))
        {
            Skip = $"{language} is not installed on this machine.";
        }
    }
}

/// <summary>The mirror image: a test that only makes sense when the runtime is <em>absent</em>.</summary>
public sealed class RequiresMissingRuntimeFactAttribute : FactAttribute
{
    public RequiresMissingRuntimeFactAttribute(ScriptLanguage language)
    {
        if (RuntimeProbe.IsAvailable(language))
        {
            Skip = $"{language} is installed, so there is nothing to report.";
        }
    }
}

internal static class RuntimeProbe
{
    // Probed once per run: the check shells out, and xunit constructs an attribute per test.
    private static readonly Dictionary<ScriptLanguage, bool> Cache = new();
    private static readonly Lock Gate = new();

    public static bool IsAvailable(ScriptLanguage language)
    {
        lock (Gate)
        {
            if (Cache.TryGetValue(language, out var cached))
            {
                return cached;
            }

            var options = new ScriptingOptions();

            IScriptRunner runner = language switch
            {
                ScriptLanguage.Python => new PythonScriptRunner(options),
                ScriptLanguage.R => new RScriptRunner(options),
                ScriptLanguage.CSharp => new CSharpScriptRunner(),
                _ => new JavaScriptRunner()
            };

            var available = runner.IsAvailableAsync().GetAwaiter().GetResult();
            Cache[language] = available;

            return available;
        }
    }
}
