// An ASP.NET Core API backed by a local model, generated from the LocalGen template.
//
// The endpoints here are domain operations — summarise, classify, extract — rather than a proxy
// for chat. That is the useful shape for an application service: the prompt is your concern, and
// callers see an API in your own vocabulary.

using System.Text.Json;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddLocalGenClient(options =>
{
    options.Endpoint = builder.Configuration["LocalGen:Endpoint"] ?? "LOCALGEN_ENDPOINT_PLACEHOLDER";
    options.ApiKey = builder.Configuration["LocalGen:ApiKey"];
    options.DefaultModel = builder.Configuration["LocalGen:Model"] ?? "LOCALGEN_MODEL_PLACEHOLDER";
});

var app = builder.Build();

// Reports whether the model behind this API is reachable, so a load balancer does not send
// traffic to an instance that cannot serve it.
app.MapGet("/health", async (LocalGenClient client) =>
    await client.PingAsync()
        ? Results.Ok(new { status = "ok" })
        : Results.StatusCode(StatusCodes.Status503ServiceUnavailable));

app.MapPost("/summarise", async (SummariseRequest request, LocalGenClient client, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
    {
        return Results.BadRequest(new { error = "'text' is required." });
    }

    var summary = await client.ChatAsync(
        prompt: request.Text,
        systemPrompt:
            $"Summarise the text in at most {request.MaxSentences} sentences. " +
            "Preserve figures, dates and names exactly. Add nothing that is not in the text.",
        cancellationToken: ct);

    return Results.Ok(new { summary });
});

app.MapPost("/classify", async (ClassifyRequest request, LocalGenClient client, CancellationToken ct) =>
{
    if (request.Categories.Count == 0)
    {
        return Results.BadRequest(new { error = "'categories' must contain at least one value." });
    }

    var label = await client.ChatAsync(
        prompt: request.Text,
        systemPrompt:
            $"Classify the text into exactly one of: {string.Join(", ", request.Categories)}. " +
            "Reply with the category name alone and nothing else.",
        cancellationToken: ct);

    // The model is asked for a bare label, but constrain the answer anyway — a mismatch is a
    // failed classification, not a new category.
    var match = request.Categories.FirstOrDefault(
        category => label.Trim().Contains(category, StringComparison.OrdinalIgnoreCase));

    return Results.Ok(new { category = match, raw = label.Trim() });
});

app.MapPost("/extract", async (ExtractRequest request, LocalGenClient client, CancellationToken ct) =>
{
    var chat = new ChatCompletionRequest
    {
        Model = client.Options.DefaultModel ?? string.Empty,
        // Near-deterministic: extraction should give the same answer for the same document.
        Temperature = 0.1f,
        // On the LlamaSharp backend this constrains decoding with a JSON grammar, so the reply
        // cannot be prose wrapped around an object.
        ResponseFormat = new ResponseFormat { Type = "json_object" },
        Messages =
        [
            Message("system",
                "Extract the requested fields and reply with JSON only, matching this shape:\n" +
                request.Schema +
                "\nUse null for any field the document does not state."),
            Message("user", request.Text)
        ]
    };

    var response = await client.ChatAsync(chat, ct);
    var content = response.Choices.FirstOrDefault()?.Message?.Content?.GetString() ?? "{}";

    try
    {
        // Returned as parsed JSON so callers get an object, not a string containing one.
        return Results.Ok(JsonSerializer.Deserialize<JsonElement>(content));
    }
    catch (JsonException)
    {
        return Results.UnprocessableEntity(new { error = "The model did not return valid JSON.", raw = content });
    }
});

// Streaming, so a browser or another service can render tokens as they arrive.
app.MapPost("/write", async (HttpContext context, WriteRequest request, LocalGenClient client, CancellationToken ct) =>
{
    context.Response.ContentType = "text/plain; charset=utf-8";
    context.Response.Headers["X-Accel-Buffering"] = "no";

    await foreach (var token in client.StreamTextAsync(request.Prompt, request.Style, cancellationToken: ct))
    {
        await context.Response.WriteAsync(token, ct);
        await context.Response.Body.FlushAsync(ct);
    }
});

app.Run();

static OpenAiMessage Message(string role, string text) => new()
{
    Role = role,
    Content = JsonSerializer.SerializeToElement(text)
};

record SummariseRequest(string Text, int MaxSentences = 3);

record ClassifyRequest(string Text, IReadOnlyList<string> Categories);

record ExtractRequest(string Text, string Schema);

record WriteRequest(string Prompt, string? Style = null);
