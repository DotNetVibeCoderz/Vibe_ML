using System.Text;
using System.Text.Json;
using LocalGen.Core;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Core.Protocol;
using LocalGen.Server.Services;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;

namespace LocalGen.Server.Endpoints;

/// <summary>
/// The OpenAI-compatible surface. Route shapes, field names and the SSE framing follow the
/// OpenAI API exactly so that existing clients switch by changing only their base URL.
/// </summary>
public static class OpenAiEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEndpoints(this IEndpointRouteBuilder app)
    {
        var v1 = app.MapGroup("/v1").WithTags("OpenAI");

        v1.MapPost("/chat/completions", ChatCompletionsAsync)
          .WithName("CreateChatCompletion")
          .WithSummary("Creates a model response for a chat conversation.");

        v1.MapPost("/completions", CompletionsAsync)
          .WithName("CreateCompletion")
          .WithSummary("Legacy text completion endpoint.");

        v1.MapPost("/embeddings", EmbeddingsAsync)
          .WithName("CreateEmbedding")
          .WithSummary("Creates embedding vectors for the given inputs.");

        v1.MapGet("/models", ListModelsAsync)
          .WithName("ListModels")
          .WithSummary("Lists the models available locally.");

        v1.MapGet("/models/{id}", GetModelAsync)
          .WithName("RetrieveModel")
          .WithSummary("Describes a single model.");

        return app;
    }

    private static async Task ChatCompletionsAsync(
        HttpContext context,
        [FromBody] ChatCompletionRequest request,
        InferenceService inference,
        AttachmentResolver attachments,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Model))
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest,
                "'model' is required.", "invalid_request_error").ConfigureAwait(false);
            return;
        }

        if (request.Messages.Count == 0)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest,
                "'messages' must contain at least one message.", "invalid_request_error").ConfigureAwait(false);
            return;
        }

        // Attachments arrive as URLs; images this server stores are loaded into real bytes here,
        // before the request reaches a backend that may be able to see them.
        var domainRequest = await attachments
            .ResolveAsync(OpenAiMapper.ToChatRequest(request), cancellationToken)
            .ConfigureAwait(false);

        var id = $"chatcmpl-{Guid.NewGuid():N}";

        if (request.Stream)
        {
            await StreamChatAsync(context, inference, domainRequest, request, id, cancellationToken)
                .ConfigureAwait(false);
            return;
        }

        try
        {
            var response = await inference.CompleteAsync(domainRequest, null, cancellationToken)
                .ConfigureAwait(false);

            await context.Response
                .WriteAsJsonAsync(
                    OpenAiMapper.ToCompletionResponse(response, id, SystemFingerprint),
                    OpenAiJson.Options,
                    cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (IsClientError(ex))
        {
            await WriteErrorAsync(context, StatusCodes.Status404NotFound, ex.Message, "model_not_found")
                .ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Emits the response as server-sent events. Each chunk is a <c>chat.completion.chunk</c>
    /// object, and the stream is terminated by the literal <c>data: [DONE]</c> sentinel that
    /// OpenAI clients look for.
    /// </summary>
    private static async Task StreamChatAsync(
        HttpContext context,
        InferenceService inference,
        ChatRequest domainRequest,
        ChatCompletionRequest request,
        string id,
        CancellationToken cancellationToken)
    {
        context.Response.Headers.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers.Connection = "keep-alive";
        // Stops reverse proxies such as nginx from buffering the stream into one response.
        context.Response.Headers["X-Accel-Buffering"] = "no";

        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var roleSent = false;

        try
        {
            await foreach (var chunk in inference
                .StreamAsync(domainRequest, null, cancellationToken)
                .ConfigureAwait(false))
            {
                var delta = new OpenAiDelta();

                // OpenAI sends the role once, on the first delta.
                if (!roleSent)
                {
                    delta.Role = "assistant";
                    roleSent = true;
                }

                if (chunk.Delta.Length > 0)
                {
                    delta.Content = chunk.Delta;
                }

                if (chunk.ToolCalls.Count > 0)
                {
                    delta.ToolCalls =
                    [
                        .. chunk.ToolCalls.Select(static (call, index) => new OpenAiToolCall
                        {
                            Id = call.Id,
                            Index = index,
                            Function = new OpenAiFunctionCall
                            {
                                Name = call.Name,
                                Arguments = call.ArgumentsJson
                            }
                        })
                    ];
                }

                var isFinal = chunk.FinishReason != FinishReason.None;

                // Skip empty keep-alive chunks that carry neither content nor a state change.
                if (delta.Content is null && delta.ToolCalls is null && delta.Role is null && !isFinal)
                {
                    continue;
                }

                var payload = new ChatCompletionResponse
                {
                    Id = id,
                    Object = "chat.completion.chunk",
                    Created = created,
                    Model = domainRequest.Model,
                    SystemFingerprint = SystemFingerprint,
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Delta = delta,
                            FinishReason = isFinal ? OpenAiMapper.ToFinishReason(chunk.FinishReason) : null
                        }
                    ],
                    Usage = isFinal && chunk.Usage is not null && request.StreamOptions?.IncludeUsage == true
                        ? OpenAiMapper.ToUsage(chunk.Usage)
                        : null
                };

                await WriteEventAsync(context, payload, cancellationToken).ConfigureAwait(false);
            }

            await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken).ConfigureAwait(false);
            await context.Response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // The client hung up mid-stream; nothing to report.
        }
        catch (Exception ex)
        {
            // Headers are already sent, so the error has to travel inside the stream itself.
            var error = OpenAiErrorResponse.Create(ex.Message, ClassifyError(ex));
            await WriteEventAsync(context, error, CancellationToken.None).ConfigureAwait(false);
            await context.Response.WriteAsync("data: [DONE]\n\n", CancellationToken.None).ConfigureAwait(false);
        }
    }

    private static async Task WriteEventAsync<T>(
        HttpContext context,
        T payload,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(payload, OpenAiJson.Options);
        await context.Response.WriteAsync($"data: {json}\n\n", cancellationToken).ConfigureAwait(false);
        await context.Response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Legacy <c>/v1/completions</c>. The prompt is wrapped in a single user turn, which is the
    /// only sensible mapping onto instruction-tuned chat models.
    /// </summary>
    private static async Task CompletionsAsync(
        HttpContext context,
        [FromBody] CompletionRequest request,
        InferenceService inference,
        CancellationToken cancellationToken)
    {
        var prompt = request.Prompt switch
        {
            { ValueKind: JsonValueKind.String } element => element.GetString() ?? string.Empty,
            { ValueKind: JsonValueKind.Array } element => string.Join(
                "\n",
                element.EnumerateArray()
                    .Where(static e => e.ValueKind == JsonValueKind.String)
                    .Select(static e => e.GetString())),
            _ => string.Empty
        };

        if (string.IsNullOrEmpty(prompt))
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest,
                "'prompt' is required.", "invalid_request_error").ConfigureAwait(false);
            return;
        }

        var domainRequest = new ChatRequest
        {
            Model = request.Model,
            Messages = [ChatMessage.User(prompt)],
            Options = new GenerationOptions
            {
                Temperature = request.Temperature,
                TopP = request.TopP,
                MaxTokens = request.MaxTokens,
                StopSequences = OpenAiMapper.ReadStopSequences(request.Stop),
                Seed = request.Seed is { } seed ? unchecked((uint)seed) : null
            }
        };

        var id = $"cmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (request.Stream)
        {
            context.Response.Headers.ContentType = "text/event-stream";
            context.Response.Headers.CacheControl = "no-cache";

            await foreach (var chunk in inference
                .StreamAsync(domainRequest, null, cancellationToken)
                .ConfigureAwait(false))
            {
                if (chunk.Delta.Length == 0 && chunk.FinishReason == FinishReason.None)
                {
                    continue;
                }

                await WriteEventAsync(context, new CompletionResponse
                {
                    Id = id,
                    Created = created,
                    Model = request.Model,
                    Choices =
                    [
                        new CompletionChoice
                        {
                            Index = 0,
                            Text = chunk.Delta,
                            FinishReason = chunk.FinishReason != FinishReason.None
                                ? OpenAiMapper.ToFinishReason(chunk.FinishReason)
                                : null
                        }
                    ]
                }, cancellationToken).ConfigureAwait(false);
            }

            await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken).ConfigureAwait(false);
            return;
        }

        var response = await inference.CompleteAsync(domainRequest, null, cancellationToken)
            .ConfigureAwait(false);

        await context.Response.WriteAsJsonAsync(new CompletionResponse
        {
            Id = id,
            Created = created,
            Model = response.Model,
            Choices =
            [
                new CompletionChoice
                {
                    Index = 0,
                    Text = response.Message.Text,
                    FinishReason = OpenAiMapper.ToFinishReason(response.FinishReason)
                }
            ],
            Usage = OpenAiMapper.ToUsage(response.Usage)
        }, OpenAiJson.Options, cancellationToken).ConfigureAwait(false);
    }

    private static async Task EmbeddingsAsync(
        HttpContext context,
        [FromBody] EmbeddingsRequest request,
        InferenceService inference,
        CancellationToken cancellationToken)
    {
        var inputs = OpenAiMapper.ReadEmbeddingInputs(request.Input);

        if (inputs.Count == 0)
        {
            await WriteErrorAsync(context, StatusCodes.Status400BadRequest,
                "'input' must be a string or an array of strings.", "invalid_request_error")
                .ConfigureAwait(false);
            return;
        }

        try
        {
            var response = await inference
                .EmbedAsync(new EmbeddingRequest { Model = request.Model, Inputs = inputs }, cancellationToken)
                .ConfigureAwait(false);

            await context.Response.WriteAsJsonAsync(new EmbeddingsResponse
            {
                Model = response.Model,
                Data =
                [
                    .. response.Embeddings.Select((vector, index) => new EmbeddingData
                    {
                        Index = index,
                        Embedding = vector.ToArray()
                    })
                ],
                Usage = OpenAiMapper.ToUsage(response.Usage)
            }, OpenAiJson.Options, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (IsClientError(ex))
        {
            await WriteErrorAsync(context, StatusCodes.Status404NotFound, ex.Message, "model_not_found")
                .ConfigureAwait(false);
        }
    }

    private static async Task<Ok<ModelListResponse>> ListModelsAsync(
        IModelStore store,
        CancellationToken cancellationToken)
    {
        var models = await store.ListAsync(cancellationToken).ConfigureAwait(false);

        return TypedResults.Ok(new ModelListResponse
        {
            Data = [.. models.Select(OpenAiMapper.ToModelData)]
        });
    }

    private static async Task<Results<Ok<ModelData>, NotFound<OpenAiErrorResponse>>> GetModelAsync(
        string id,
        IModelStore store,
        CancellationToken cancellationToken)
    {
        var model = await store.GetAsync(id, cancellationToken).ConfigureAwait(false);

        return model is null
            ? TypedResults.NotFound(OpenAiErrorResponse.Create(
                $"Model '{id}' is not installed.", "invalid_request_error", "model_not_found"))
            : TypedResults.Ok(OpenAiMapper.ToModelData(model));
    }

    /// <summary>Identifies the server build in responses, mirroring OpenAI's fingerprint field.</summary>
    private static string SystemFingerprint { get; } =
        $"localgen-{typeof(OpenAiEndpoints).Assembly.GetName().Version?.ToString(3) ?? "0.1.0"}";

    private static bool IsClientError(Exception ex) =>
        ex is ModelNotFoundException or EngineNotAvailableException;

    private static string ClassifyError(Exception ex) => ex switch
    {
        ModelNotFoundException => "model_not_found",
        EngineNotAvailableException => "engine_unavailable",
        OfflineModeException => "offline_mode",
        _ => "server_error"
    };

    private static async Task WriteErrorAsync(
        HttpContext context,
        int statusCode,
        string message,
        string type)
    {
        context.Response.StatusCode = statusCode;
        await context.Response
            .WriteAsJsonAsync(OpenAiErrorResponse.Create(message, type), OpenAiJson.Options)
            .ConfigureAwait(false);
    }
}
