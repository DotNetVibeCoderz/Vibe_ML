using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Scripting;

namespace BlazorML.Tests;

/// <summary>
/// The two runtimes that ship with the app. Nothing external is needed, so these always run.
/// </summary>
public class InProcessScriptRunnerTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(20);

    private static TabularData Rows()
    {
        var table = TabularData.WithColumns("nama", "nilai");
        table.AddRow("Ani", 10);
        table.AddRow("Budi", 20);

        return table;
    }

    // ---------------------------------------------------------------------- C#

    [Fact]
    public async Task CSharp_returns_the_rows_it_was_given()
    {
        var runner = new CSharpScriptRunner();
        var output = await runner.RunAsync("return rows;", Rows(), null, Timeout);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("Ani", output.Text(0, "nama"));
    }

    [Fact]
    public async Task CSharp_can_filter_and_reshape()
    {
        var runner = new CSharpScriptRunner();

        var output = await runner.RunAsync("""
            return rows
                .Where(r => Convert.ToDouble(r["nilai"]) > 15)
                .ToList();
            """, Rows(), null, Timeout);

        Assert.Equal(1, output.RowCount);
        Assert.Equal("Budi", output.Text(0, "nama"));
    }

    [Fact]
    public async Task CSharp_sees_both_inputs()
    {
        var runner = new CSharpScriptRunner();
        var output = await runner.RunAsync("return rows.Concat(rows2).ToList();", Rows(), Rows(), Timeout);

        Assert.Equal(4, output.RowCount);
    }

    [Fact]
    public async Task CSharp_reports_a_compile_error_with_the_diagnostics()
    {
        var runner = new CSharpScriptRunner();

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => runner.RunAsync("return ini bukan c#;", Rows(), null, Timeout));

        Assert.Contains("did not compile", error.Message);
    }

    [Fact]
    public async Task CSharp_refuses_a_return_value_that_is_not_a_row_collection()
    {
        var runner = new CSharpScriptRunner();

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => runner.RunAsync("return 42;", Rows(), null, Timeout));

        Assert.Contains("list of dictionaries", error.Message);
    }

    // -------------------------------------------------------------- JavaScript

    [Fact]
    public async Task JavaScript_returns_the_rows_it_was_given()
    {
        var runner = new JavaScriptRunner();
        var output = await runner.RunAsync("return rows;", Rows(), null, Timeout);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("Ani", output.Text(0, "nama"));
    }

    [Fact]
    public async Task JavaScript_can_map_and_add_a_column()
    {
        var runner = new JavaScriptRunner();

        var output = await runner.RunAsync("""
            return rows.map(function (r) {
                return { nama: r.nama, ganda: r.nilai * 2 };
            });
            """, Rows(), null, Timeout);

        Assert.Equal(20d, TabularData.ToNumber(output.Value(0, "ganda")));
        Assert.Equal(40d, TabularData.ToNumber(output.Value(1, "ganda")));
    }

    /// <summary>
    /// Regression test. The runner used to evaluate the user's script twice — once for the
    /// result and once to serialise it — so any side effect happened twice over.
    /// </summary>
    [Fact]
    public async Task JavaScript_runs_the_script_exactly_once()
    {
        var runner = new JavaScriptRunner();

        var output = await runner.RunAsync("""
            var calls = 0;
            function bump() { calls += 1; return calls; }
            return rows.map(function (r) { return { panggilan: bump() }; });
            """, Rows(), null, Timeout);

        // Two rows, so the counter reaches 2 — not 4.
        Assert.Equal(1d, TabularData.ToNumber(output.Value(0, "panggilan")));
        Assert.Equal(2d, TabularData.ToNumber(output.Value(1, "panggilan")));
    }

    [Fact]
    public async Task JavaScript_returning_nothing_passes_the_input_through()
    {
        var runner = new JavaScriptRunner();
        var output = await runner.RunAsync("var x = 1;", Rows(), null, Timeout);

        Assert.Equal(2, output.RowCount);
    }

    [Fact]
    public async Task Both_in_process_runners_report_themselves_as_available()
    {
        Assert.True(await new CSharpScriptRunner().IsAvailableAsync());
        Assert.True(await new JavaScriptRunner().IsAvailableAsync());
    }
}

/// <summary>
/// Python and R run as child processes, so these depend on the machine. Each test skips itself
/// when the runtime is absent rather than failing — a missing interpreter is a fact about the
/// environment, not a defect in the code.
/// </summary>
public class ExternalScriptRunnerTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(60);
    private static readonly ScriptingOptions Options = new();

    private static TabularData Rows()
    {
        var table = TabularData.WithColumns("nama", "nilai");
        table.AddRow("Ani", 10);
        table.AddRow("Budi", 20);

        return table;
    }
    [RequiresRuntimeFact(ScriptLanguage.Python)]
    public async Task Python_runs_a_run_function_over_the_rows()
    {
        var runner = new PythonScriptRunner(Options);
        var output = await runner.RunAsync("""
            def run(dataset1, dataset2):
                return [r for r in dataset1 if float(r['nilai']) > 15]
            """, Rows(), null, Timeout);

        Assert.Equal(1, output.RowCount);
        Assert.Equal("Budi", output.Text(0, "nama"));
    }
    [RequiresRuntimeFact(ScriptLanguage.Python)]
    public async Task Python_accepts_a_result_variable_instead_of_a_function()
    {
        var runner = new PythonScriptRunner(Options);
        var output = await runner.RunAsync(
            "result = [{'a': 1}, {'a': 2}, {'a': 3}]", Rows(), null, Timeout);

        Assert.Equal(3, output.RowCount);
        Assert.True(output.Has("a"));
    }
    [RequiresRuntimeFact(ScriptLanguage.Python)]
    public async Task Python_can_add_a_column()
    {
        var runner = new PythonScriptRunner(Options);
        var output = await runner.RunAsync("""
            def run(dataset1, dataset2):
                for r in dataset1:
                    r['ganda'] = float(r['nilai']) * 2
                return dataset1
            """, Rows(), null, Timeout);

        Assert.Equal(20d, TabularData.ToNumber(output.Value(0, "ganda")));
    }
    [RequiresRuntimeFact(ScriptLanguage.Python)]
    public async Task Python_surfaces_the_interpreter_error_rather_than_a_bare_exit_code()
    {
        var runner = new PythonScriptRunner(Options);
        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => runner.RunAsync("raise ValueError('sengaja gagal')", Rows(), null, Timeout));

        Assert.Contains("sengaja gagal", error.Message);
    }

    /// <summary>
    /// The reason these run out of process: a timeout that actually works. In-process hosting
    /// cannot interrupt a script stuck in native code.
    /// </summary>
    [RequiresRuntimeFact(ScriptLanguage.Python)]
    public async Task Python_that_never_finishes_is_killed_at_the_timeout()
    {
        var runner = new PythonScriptRunner(Options);
        await Assert.ThrowsAsync<TimeoutException>(
            () => runner.RunAsync("while True: pass", Rows(), null, TimeSpan.FromSeconds(3)));
    }
    [RequiresMissingRuntimeFact(ScriptLanguage.R)]
    public async Task R_says_what_is_missing_when_it_is_not_installed()
    {
        var runner = new RScriptRunner(Options);
        Assert.Contains("Install R", runner.UnavailableReason);
        Assert.Contains("Scripting", runner.UnavailableReason);
    }
    [RequiresRuntimeFact(ScriptLanguage.R)]
    public async Task R_transforms_a_data_frame_when_it_is_installed()
    {
        var runner = new RScriptRunner(Options);
        var output = await runner.RunAsync(
            "result <- dataset1[dataset1$nilai > 15, ]", Rows(), null, Timeout);

        Assert.Equal(1, output.RowCount);
    }

    [Theory]
    [InlineData(ScriptLanguage.Python)]
    [InlineData(ScriptLanguage.R)]
    public void An_unavailable_runtime_explains_itself_in_terms_of_what_to_install(ScriptLanguage language)
    {
        // Whatever this machine has, the message must name the runtime and where to configure it
        // — never leave the user with a bare failure.
        IScriptRunner runner = language == ScriptLanguage.Python
            ? new PythonScriptRunner(Options)
            : new RScriptRunner(Options);

        Assert.Contains("Settings", runner.UnavailableReason);
        Assert.Equal(language, runner.Language);
    }
}
