using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Training;
using BlazorML.Web.Services;
using Microsoft.ML;
using Microsoft.OpenApi.Models;

namespace BlazorML.Web.Endpoints;

/// <summary>
/// The published scoring API. One route per endpoint slug, guarded by an API key, documented in
/// Swagger so a consumer can try it without reading any of this code.
/// </summary>
public static class ScoringApi
{
    public const string KeyHeader = "X-Api-Key";

    public static void ConfigureSwagger(Swashbuckle.AspNetCore.SwaggerGen.SwaggerGenOptions options)
    {
        options.SwaggerDoc("v1", new OpenApiInfo
        {
            Title = "Blazor ML Studio — scoring API",
            Version = "v1",
            Description =
                "Endpoint prediksi untuk model yang sudah diterbitkan dari Blazor ML Studio.\n\n" +
                "Setiap permintaan memerlukan header `X-Api-Key`. Kunci dibuat di halaman Endpoint " +
                "dan hanya ditampilkan satu kali.\n\n" +
                "Dibuat oleh Gravicode Studios, di-lead oleh Kang Fadhil.",
            Contact = new OpenApiContact { Name = "Gravicode Studios" }
        });

        options.AddSecurityDefinition("ApiKey", new OpenApiSecurityScheme
        {
            Name = KeyHeader,
            Type = SecuritySchemeType.ApiKey,
            In = ParameterLocation.Header,
            Description = "Kunci API endpoint, diawali bml_"
        });

        options.AddSecurityRequirement(new OpenApiSecurityRequirement
        {
            [new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference { Type = ReferenceType.SecurityScheme, Id = "ApiKey" }
            }] = Array.Empty<string>()
        });
    }

    public static void MapScoringApi(this WebApplication app)
    {
        var group = app.MapGroup("/api/v1").WithTags("Scoring");

        group.MapGet("/endpoints", async (EndpointService endpoints) =>
            {
                var live = await endpoints.ListAsync();

                return Results.Ok(live
                    .Where(e => e.Status == EndpointStatus.Live)
                    .Select(e => new
                    {
                        e.Name,
                        e.Slug,
                        e.Description,
                        model = e.Model?.Name,
                        task = e.Model?.Task.ToString(),
                        scoreUrl = $"/api/v1/score/{e.Slug}"
                    }));
            })
            .WithName("ListEndpoints")
            .WithSummary("Lists the endpoints that are currently live.")
            .Produces(StatusCodes.Status200OK);

        group.MapPost("/score/{slug}", async (
                string slug,
                ScoreRequest request,
                HttpContext http,
                EndpointService endpoints,
                IModelRegistry models,
                ILoggerFactory loggerFactory,
                CancellationToken ct) =>
            {
                var presented = http.Request.Headers[KeyHeader].FirstOrDefault();
                var endpoint = await endpoints.AuthenticateAsync(slug, presented, ct);

                if (endpoint?.Model is null)
                {
                    // One message for "no such endpoint", "wrong key" and "endpoint stopped": a
                    // caller without a valid key learns nothing about which endpoints exist.
                    return Results.Problem(
                        title: "Unauthorised",
                        detail: "The endpoint was not found, is not live, or the API key is not valid for it.",
                        statusCode: StatusCodes.Status401Unauthorized);
                }

                if (request.Rows is null || request.Rows.Count == 0)
                {
                    return Results.Problem(
                        title: "Nothing to score",
                        detail: "Send at least one row in the 'rows' array.",
                        statusCode: StatusCodes.Status400BadRequest);
                }

                var logger = loggerFactory.CreateLogger("ScoringApi");

                try
                {
                    var table = TabularData.FromDictionaries(
                        request.Rows.Select(r => r.ToDictionary(kv => kv.Key, kv => ReadScalar(kv.Value))));

                    var ml = new MLContext(seed: 42);

                    await using var stream = await models.OpenAsync(endpoint.Model.Id, ct);
                    var transformer = ml.Model.Load(stream, out _);

                    // A caller sends the features and asks for the label; requiring them to send
                    // the label too would make the endpoint useless.
                    table = MlDataBridge.EnsureLabelColumn(table, endpoint.Model.LabelColumn, endpoint.Model.Task);

                    using var bridge = new MlDataBridge();
                    var view = bridge.ToDataView(ml, table, endpoint.Model.LabelColumn, endpoint.Model.Task, out _);
                    var scored = MlDataBridge.FromDataView(transformer.Transform(view));

                    return Results.Ok(new ScoreResponse(
                        endpoint.Model.Name,
                        endpoint.Model.Version,
                        endpoint.Model.Task.ToString(),
                        scored.ToDictionaries()));
                }
                catch (Exception e)
                {
                    logger.LogError(e, "Scoring failed for endpoint {Slug}", slug);

                    return Results.Problem(
                        title: "Scoring failed",
                        detail: e.Message,
                        statusCode: StatusCodes.Status422UnprocessableEntity);
                }
            })
            .WithName("Score")
            .WithSummary("Scores rows against a published model.")
            .Produces<ScoreResponse>()
            .Produces(StatusCodes.Status401Unauthorized)
            .Produces(StatusCodes.Status422UnprocessableEntity);

        group.MapGet("/schema/{slug}", async (
                string slug,
                HttpContext http,
                EndpointService endpoints,
                CancellationToken ct) =>
            {
                var presented = http.Request.Headers[KeyHeader].FirstOrDefault();
                var endpoint = await endpoints.AuthenticateAsync(slug, presented, ct);

                if (endpoint?.Model is null)
                {
                    return Results.Problem(statusCode: StatusCodes.Status401Unauthorized);
                }

                return Results.Ok(new
                {
                    model = endpoint.Model.Name,
                    endpoint.Model.Version,
                    task = endpoint.Model.Task.ToString(),
                    label = endpoint.Model.LabelColumn,
                    inputs = string.IsNullOrWhiteSpace(endpoint.Model.InputSchemaJson)
                        ? (JsonElement?)null
                        : JsonSerializer.Deserialize<JsonElement>(endpoint.Model.InputSchemaJson)
                });
            })
            .WithName("Schema")
            .WithSummary("Describes the columns a model expects.");
    }

    private static object? ReadScalar(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetDouble(out var d) ? d : element.GetRawText(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        _ => element.GetRawText()
    };
}

/// <summary>Rows to score, each a flat object of column name to value.</summary>
public sealed record ScoreRequest(List<Dictionary<string, JsonElement>>? Rows);

public sealed record ScoreResponse(string Model, int Version, string Task,
    List<Dictionary<string, object?>> Predictions);
