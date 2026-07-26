using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.ML.Execution.Executors;

namespace BlazorML.Tests;

/// <summary>
/// The LLM modules carry real logic around the model call: batching rows, parsing a reply that
/// may or may not be well formed, and recovering when it is not. A stub provider exercises all
/// of it without a network call or an API key — what is being tested is the code, not the model.
/// </summary>
public class LlmExecutorTests
{
    private readonly LlmExecutor _executor = new();

    private static TabularData Reviews(params string[] texts)
    {
        var table = TabularData.WithColumns("Ulasan");

        foreach (var text in texts)
        {
            table.AddRow(text);
        }

        return table;
    }

    private TabularData Run(TabularData input, StubLlm llm,
        params (string Name, string? Value)[] parameters)
    {
        var services = new StubServices(llm);
        var context = TestContext.With("llm.sentiment", services, parameters, input);

        return (TabularData)_executor.ExecuteAsync(context).GetAwaiter().GetResult()[0]!;
    }

    // ------------------------------------------------------------- happy path

    [Fact]
    public void A_well_formed_batch_reply_is_written_back_row_by_row()
    {
        var llm = new StubLlm("""
            [{"Sentiment":"positif","SentimentScore":"0.9"},
             {"Sentiment":"negatif","SentimentScore":"0.8"}]
            """);

        var output = Run(Reviews("Bagus sekali", "Mengecewakan"), llm, ("column", "Ulasan"));

        Assert.Equal("positif", output.Text(0, "Sentiment"));
        Assert.Equal("negatif", output.Text(1, "Sentiment"));
        Assert.Equal("0.9", output.Text(0, "SentimentScore"));
    }

    [Fact]
    public void Rows_are_sent_in_batches_of_the_configured_size()
    {
        // Five rows at two per batch is three requests, not five: the batching is the whole
        // reason this module is usable on a real dataset.
        var llm = StubLlm.Obedient("""{"Sentiment":"positif"}""");

        Run(Reviews("a", "b", "c", "d", "e"), llm,
            ("column", "Ulasan"), ("batchSize", "2"), ("includeScore", "false"));

        Assert.Equal(3, llm.Calls);
    }

    [Fact]
    public void The_row_limit_stops_early_instead_of_processing_everything()
    {
        var llm = StubLlm.Obedient("""{"Sentiment":"positif"}""");

        var output = Run(Reviews("a", "b", "c", "d"), llm,
            ("column", "Ulasan"), ("batchSize", "1"), ("maxRows", "2"), ("includeScore", "false"));

        Assert.Equal(2, llm.Calls);
        // The rows beyond the limit stay in the table, just unlabelled.
        Assert.Equal(4, output.RowCount);
        Assert.Null(output.Value(3, "Sentiment"));
    }

    // --------------------------------------------------------------- parsing

    [Fact]
    public void A_reply_wrapped_in_a_code_fence_is_still_understood()
    {
        // Models fence JSON despite being asked not to, often enough that failing here would
        // make the module unreliable in practice rather than in theory.
        var llm = new StubLlm("""
            ```json
            [{"Sentiment":"positif"}]
            ```
            """);

        var output = Run(Reviews("Bagus"), llm, ("column", "Ulasan"), ("includeScore", "false"));

        Assert.Equal("positif", output.Text(0, "Sentiment"));
    }

    [Fact]
    public void A_batch_reply_that_does_not_line_up_falls_back_to_one_request_per_row()
    {
        // First call returns one object for a two-row batch. The module must not write that
        // answer against the wrong row; it retries individually instead.
        var llm = new StubLlm(
            """[{"Sentiment":"positif"}]""",
            """{"Sentiment":"aaa"}""",
            """{"Sentiment":"bbb"}""");

        var output = Run(Reviews("satu", "dua"), llm, ("column", "Ulasan"), ("includeScore", "false"));

        Assert.Equal(3, llm.Calls);
        Assert.Equal("aaa", output.Text(0, "Sentiment"));
        Assert.Equal("bbb", output.Text(1, "Sentiment"));
    }

    [Fact]
    public void An_unparseable_reply_leaves_the_cells_empty_rather_than_writing_the_raw_text()
    {
        var llm = new StubLlm("maaf, saya tidak bisa membantu");

        var output = Run(Reviews("satu"), llm, ("column", "Ulasan"), ("includeScore", "false"));

        Assert.True(output.Has("Sentiment"));
        Assert.Null(output.Value(0, "Sentiment"));
    }

    // -------------------------------------------------------------- prompting

    [Fact]
    public void Custom_prompt_substitutes_column_values_into_the_template()
    {
        var llm = new StubLlm("""[{"LlmOutput":"ringkas"}]""");

        var input = TabularData.WithColumns("nama", "kota");
        input.AddRow("Ani", "Bandung");

        var context = TestContext.With("llm.prompt", new StubServices(llm),
            [("prompt", "Sapa {{nama}} dari {{kota}}."), ("outputColumn", "LlmOutput")], input);

        _executor.ExecuteAsync(context).GetAwaiter().GetResult();

        Assert.Contains("Sapa Ani dari Bandung.", llm.LastUserPrompt);
    }

    [Fact]
    public void Extract_asks_for_one_column_per_requested_field()
    {
        var llm = new StubLlm("""[{"nama":"Ani","jumlah":"120"}]""");

        var input = TabularData.WithColumns("teks");
        input.AddRow("Ani membayar 120 ribu");

        var context = TestContext.With("llm.extract", new StubServices(llm),
            [("column", "teks"), ("fields", "nama\njumlah")], input);

        var output = (TabularData)_executor.ExecuteAsync(context).GetAwaiter().GetResult()[0]!;

        Assert.Equal("Ani", output.Text(0, "nama"));
        Assert.Equal("120", output.Text(0, "jumlah"));
    }

    [Fact]
    public void Classify_tells_the_model_which_categories_it_may_choose_from()
    {
        var llm = new StubLlm("""[{"Category":"billing"}]""");

        var input = TabularData.WithColumns("tiket");
        input.AddRow("Tagihan saya salah");

        var context = TestContext.With("llm.classify", new StubServices(llm),
            [("column", "tiket"), ("categories", "billing\nshipping\nother")], input);

        _executor.ExecuteAsync(context).GetAwaiter().GetResult();

        Assert.Contains("billing", llm.LastSystemPrompt);
        Assert.Contains("shipping", llm.LastSystemPrompt);
    }

    [Fact]
    public void Format_writes_over_the_source_column_when_no_target_is_named()
    {
        var llm = new StubLlm("""[{"tanggal":"2026-07-26"}]""");

        var input = TabularData.WithColumns("tanggal");
        input.AddRow("26 Juli 2026");

        var context = TestContext.With("llm.format", new StubServices(llm),
            [("column", "tanggal"), ("outputColumn", null)], input);

        var output = (TabularData)_executor.ExecuteAsync(context).GetAwaiter().GetResult()[0]!;

        Assert.Equal(1, output.ColumnCount);
        Assert.Equal("2026-07-26", output.Text(0, "tanggal"));
    }

    // ------------------------------------------------------------- unavailable

    [Fact]
    public void Without_a_configured_provider_the_module_says_what_to_do_about_it()
    {
        var context = TestContext.With("llm.sentiment", new StubServices(new StubLlm { Configured = false }),
            [("column", "Ulasan")], Reviews("satu"));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _executor.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("Settings", error.Message);
        Assert.Contains("Profesor Wicak", error.Message);
    }

    [Fact]
    public void With_no_provider_registered_at_all_the_message_is_the_same()
    {
        // Nothing registered is the same situation from the user's side as nothing configured.
        var context = TestContext.For("llm.sentiment", [("column", "Ulasan")], Reviews("satu"));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _executor.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("needs a language model", error.Message);
    }

    // ------------------------------------------------------------------ stubs

    private sealed class StubLlm : ILlmActionRunner
    {
        private readonly Queue<string> _replies;
        private readonly string? _perRowObject;

        /// <summary>Replies with exactly what it is given, in order; the last one repeats.</summary>
        public StubLlm(params string[] replies)
        {
            _replies = new Queue<string>(replies);
        }

        private StubLlm(string perRowObject)
        {
            _replies = new Queue<string>();
            _perRowObject = perRowObject;
        }

        /// <summary>
        /// A well-behaved model: returns one object per row it was actually asked about. Needed
        /// to test batching, because a fixed-length reply would trip the misalignment fallback on
        /// the final short batch and inflate the call count.
        /// </summary>
        public static StubLlm Obedient(string perRowObject) => new(perRowObject);

        public bool Configured { get; init; } = true;
        public int Calls { get; private set; }
        public string LastSystemPrompt { get; private set; } = string.Empty;
        public string LastUserPrompt { get; private set; } = string.Empty;

        public bool IsConfigured => Configured;

        public Task<string> CompleteAsync(string systemPrompt, string userPrompt, string? provider = null,
            double temperature = 0.2, CancellationToken ct = default)
        {
            Calls++;
            LastSystemPrompt = systemPrompt;
            LastUserPrompt = userPrompt;

            if (_perRowObject is not null)
            {
                // The executor numbers each row "1. ", "2. " … so counting those tells the stub
                // how long its reply has to be.
                var rows = userPrompt.Split('\n')
                    .Count(line => line.Length > 1 && char.IsDigit(line[0]) && line.Contains(". "));

                return Task.FromResult("[" + string.Join(",", Enumerable.Repeat(_perRowObject, Math.Max(1, rows))) + "]");
            }

            return Task.FromResult(_replies.Count > 1 ? _replies.Dequeue() : _replies.Peek());
        }
    }

    private sealed class StubServices(ILlmActionRunner runner) : IServiceProvider
    {
        public object? GetService(Type serviceType) =>
            serviceType == typeof(ILlmActionRunner) ? runner : null;
    }
}
