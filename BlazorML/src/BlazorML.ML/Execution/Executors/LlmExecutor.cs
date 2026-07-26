using System.Text;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.ML.Execution.Executors;

/// <summary>
/// The six LLM modules. All of them share one shape: take a column, send batches of values to a
/// model with a task-specific instruction, and write the answers back as new columns.
/// <para>
/// Rows are sent in batches and the reply is parsed as a JSON array, so one request covers many
/// rows. If a batch comes back unparseable it is retried one row at a time rather than failing
/// the whole node — a single awkward row should not cost the run.
/// </para>
/// </summary>
public sealed class LlmExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("llm.", StringComparison.Ordinal);

    public async Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx)
    {
        var runner = ctx.Services.GetService<ILlmActionRunner>();

        if (runner is null || !runner.IsConfigured)
        {
            throw new InvalidOperationException(
                $"'{ctx.Node.Label}' needs a language model. Add an API key under Settings → Profesor Wicak, " +
                "then run the experiment again.");
        }

        var input = ctx.RequireTable(0, "Dataset");
        var output = input.Clone();

        var maxRows = ctx.ParamInt("maxRows", 0);
        var rowCount = maxRows > 0 ? Math.Min(maxRows, output.RowCount) : output.RowCount;
        var batchSize = Math.Clamp(ctx.ParamInt("batchSize", 20), 1, 200);
        var temperature = ctx.ParamDouble("temperature", 0.2);
        var provider = ctx.Param("provider") is { } p && p != "default" ? p : null;

        var plan = BuildPlan(ctx, output);
        var targets = plan.OutputColumns
            .Select(name => output.Has(name) ? output.IndexOf(name) : output.AddColumn(name))
            .ToList();

        var processed = 0;

        for (var start = 0; start < rowCount; start += batchSize)
        {
            ctx.Ct.ThrowIfCancellationRequested();

            var end = Math.Min(start + batchSize, rowCount);
            var prompts = new List<string>();

            for (var r = start; r < end; r++)
            {
                prompts.Add(plan.RenderRow(output, r));
            }

            var answers = await AskAsync(runner, plan, prompts, provider, temperature, ctx);

            for (var i = 0; i < answers.Count && start + i < end; i++)
            {
                var row = output.Rows[start + i];
                var answer = answers[i];

                for (var c = 0; c < targets.Count; c++)
                {
                    var field = plan.OutputColumns[c];
                    row[targets[c]] = answer.TryGetValue(field, out var value) ? value : null;
                }
            }

            processed += end - start;
            ctx.Log(Core.Domain.LogLevel.Info, $"Processed {processed:N0} of {rowCount:N0} rows.");
        }

        return [output];
    }

    /// <summary>Sends one batch and normalises the reply into one dictionary per row.</summary>
    private static async Task<List<Dictionary<string, string?>>> AskAsync(ILlmActionRunner runner, LlmPlan plan,
        List<string> prompts, string? provider, double temperature, ModuleExecutionContext ctx)
    {
        var body = new StringBuilder();
        body.AppendLine("Here are the inputs, one per line, numbered from 1:");

        for (var i = 0; i < prompts.Count; i++)
        {
            body.AppendLine($"{i + 1}. {prompts[i].Replace('\n', ' ')}");
        }

        body.AppendLine();
        body.AppendLine($"Reply with a JSON array of exactly {prompts.Count} objects, in the same order.");
        body.AppendLine($"Each object has these keys: {string.Join(", ", plan.OutputColumns)}.");
        body.AppendLine("Reply with the JSON array only, no commentary and no code fences.");

        var reply = await runner.CompleteAsync(plan.SystemPrompt, body.ToString(), provider, temperature, ctx.Ct);
        var parsed = TryParse(reply, plan.OutputColumns);

        if (parsed.Count == prompts.Count)
        {
            return parsed;
        }

        // The batch came back malformed or short. Fall back to one request per row so the rows
        // that are fine still get their answers.
        ctx.Log(Core.Domain.LogLevel.Warning,
            $"Batch reply did not line up ({parsed.Count} of {prompts.Count}); retrying row by row.");

        var singles = new List<Dictionary<string, string?>>();
        foreach (var prompt in prompts)
        {
            var single = await runner.CompleteAsync(plan.SystemPrompt,
                $"{prompt}\n\nReply with a single JSON object with these keys: {string.Join(", ", plan.OutputColumns)}. " +
                "JSON only.", provider, temperature, ctx.Ct);

            var one = TryParse(single, plan.OutputColumns);
            singles.Add(one.Count > 0 ? one[0] : new Dictionary<string, string?>());
        }

        return singles;
    }

    private static List<Dictionary<string, string?>> TryParse(string reply, List<string> fields)
    {
        var text = reply.Trim();

        // Models often wrap JSON in a fenced block despite being asked not to.
        if (text.StartsWith("```"))
        {
            var firstBreak = text.IndexOf('\n');
            var lastFence = text.LastIndexOf("```", StringComparison.Ordinal);
            if (firstBreak > 0 && lastFence > firstBreak)
            {
                text = text[(firstBreak + 1)..lastFence].Trim();
            }
        }

        var result = new List<Dictionary<string, string?>>();

        try
        {
            using var document = JsonDocument.Parse(text);
            var root = document.RootElement;

            if (root.ValueKind == JsonValueKind.Object)
            {
                result.Add(ReadObject(root, fields));
                return result;
            }

            if (root.ValueKind != JsonValueKind.Array)
            {
                return result;
            }

            foreach (var element in root.EnumerateArray())
            {
                result.Add(element.ValueKind == JsonValueKind.Object
                    ? ReadObject(element, fields)
                    : new Dictionary<string, string?> { [fields[0]] = element.ToString() });
            }
        }
        catch (JsonException)
        {
            // Unparseable: the caller falls back to per-row requests.
        }

        return result;
    }

    private static Dictionary<string, string?> ReadObject(JsonElement element, List<string> fields)
    {
        var row = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        foreach (var field in fields)
        {
            if (element.TryGetProperty(field, out var value))
            {
                row[field] = value.ValueKind == JsonValueKind.String ? value.GetString() : value.ToString();
            }
        }

        return row;
    }

    // -------------------------------------------------------- per-module prompt shapes

    private sealed class LlmPlan
    {
        public required string SystemPrompt { get; init; }
        public required List<string> OutputColumns { get; init; }
        public required Func<TabularData, int, string> RenderRow { get; init; }
    }

    private static LlmPlan BuildPlan(ModuleExecutionContext ctx, TabularData table)
    {
        switch (ctx.Module.Id)
        {
            case "llm.format":
            {
                var column = ctx.RequireParam("column");
                var target = ctx.Param("outputColumn") ?? column;
                var format = ctx.Param("targetFormat") ?? "a consistent format";

                return new LlmPlan
                {
                    SystemPrompt = $"You reformat values into {format}. Return the reformatted value only. " +
                                   "If a value cannot be reformatted, return it unchanged.",
                    OutputColumns = [target],
                    RenderRow = (t, r) => t.Text(r, column) ?? string.Empty
                };
            }

            case "llm.cleanse":
            {
                var column = ctx.RequireParam("column");
                var target = ctx.Param("outputColumn") ?? column;
                var rules = ctx.Param("rules") ?? "Fix spelling and trim whitespace.";

                return new LlmPlan
                {
                    SystemPrompt = $"You clean up data values. Rules:\n{rules}\nReturn the cleaned value only.",
                    OutputColumns = [target],
                    RenderRow = (t, r) => t.Text(r, column) ?? string.Empty
                };
            }

            case "llm.sentiment":
            {
                var column = ctx.RequireParam("column");
                var target = ctx.Param("outputColumn") ?? "Sentiment";
                var labels = ctx.Param("labels") ?? "positive,neutral,negative";
                var withScore = ctx.ParamBool("includeScore", true);

                var outputs = withScore ? new List<string> { target, target + "Score" } : [target];

                return new LlmPlan
                {
                    SystemPrompt = $"You label sentiment. Use exactly one of: {labels}. " +
                                   (withScore ? $"Also give {target}Score, your confidence from 0 to 1." : string.Empty),
                    OutputColumns = outputs,
                    RenderRow = (t, r) => t.Text(r, column) ?? string.Empty
                };
            }

            case "llm.classify":
            {
                var column = ctx.RequireParam("column");
                var target = ctx.Param("outputColumn") ?? "Category";
                var categories = (ctx.Param("categories") ?? "other")
                    .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

                return new LlmPlan
                {
                    SystemPrompt = $"You sort text into exactly one of these categories: {string.Join(", ", categories)}. " +
                                   "If nothing fits, use the last category.",
                    OutputColumns = [target],
                    RenderRow = (t, r) => t.Text(r, column) ?? string.Empty
                };
            }

            case "llm.extract":
            {
                var column = ctx.RequireParam("column");
                var fields = (ctx.Param("fields") ?? string.Empty)
                    .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .ToList();

                if (fields.Count == 0)
                {
                    throw new InvalidOperationException("List at least one field to extract, one per line.");
                }

                return new LlmPlan
                {
                    SystemPrompt = "You extract structured fields from text. " +
                                   "Leave a field empty when the text does not contain it. Do not invent values.",
                    OutputColumns = fields,
                    RenderRow = (t, r) => t.Text(r, column) ?? string.Empty
                };
            }

            case "llm.prompt":
            {
                var template = ctx.RequireParam("prompt");
                var target = ctx.Param("outputColumn") ?? "LlmOutput";

                return new LlmPlan
                {
                    SystemPrompt = ctx.Param("systemPrompt")
                        ?? "You process one row of data at a time and return only the requested result.",
                    OutputColumns = [target],
                    RenderRow = (t, r) => Render(template, t, r)
                };
            }

            default:
                throw new NotSupportedException($"Unknown LLM module '{ctx.Module.Id}'.");
        }
    }

    /// <summary>Substitutes <c>{{column}}</c> placeholders with the row's values.</summary>
    private static string Render(string template, TabularData table, int row)
    {
        var result = new StringBuilder(template);

        foreach (var column in table.Columns)
        {
            result.Replace("{{" + column.Name + "}}", table.Text(row, column.Name) ?? string.Empty);
        }

        return result.ToString();
    }
}
