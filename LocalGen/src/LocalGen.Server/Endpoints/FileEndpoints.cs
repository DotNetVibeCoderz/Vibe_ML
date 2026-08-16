using LocalGen.Core;
using LocalGen.Runtime.Files;
using Microsoft.AspNetCore.Http.HttpResults;

namespace LocalGen.Server.Endpoints;

/// <summary>
/// Upload and retrieval for conversation attachments.
/// </summary>
/// <remarks>
/// An attached image has to have an address: the OpenAI wire format carries images as
/// <c>image_url</c> parts, and a transcript renders one by URL. Uploading here gives the file a
/// stable local address, so nothing has to be inlined as base64 or fetched from the internet.
/// </remarks>
public static class FileEndpoints
{
    public static IEndpointRouteBuilder MapFileEndpoints(this IEndpointRouteBuilder app)
    {
        var files = app.MapGroup("/api/files").WithTags("Files");

        files.MapPost("/", UploadAsync)
             .WithSummary("Uploads an attachment and returns its URL.")
             .DisableAntiforgery();

        files.MapGet("/{id}", Download)
             .WithSummary("Serves an uploaded attachment.");

        return app;
    }

    private static async Task<Results<Ok<UploadedFile>, BadRequest<string>>> UploadAsync(
        HttpRequest request,
        FileStore store,
        CancellationToken cancellationToken)
    {
        if (!request.HasFormContentType)
        {
            return TypedResults.BadRequest("Send the file as multipart/form-data.");
        }

        var form = await request.ReadFormAsync(cancellationToken).ConfigureAwait(false);
        var file = form.Files.FirstOrDefault();

        if (file is null || file.Length == 0)
        {
            return TypedResults.BadRequest("No file was included in the request.");
        }

        try
        {
            await using var stream = file.OpenReadStream();

            var stored = await store
                .SaveAsync(stream, file.FileName, file.ContentType, cancellationToken)
                .ConfigureAwait(false);

            return TypedResults.Ok(new UploadedFile
            {
                Id = stored.Id,
                FileName = stored.FileName,
                MediaType = stored.MediaType,
                SizeBytes = stored.SizeBytes,
                // Absolute, because the value is pasted into messages and rendered by clients
                // that may not share this server's origin.
                Url = $"{store.BaseUrl}{stored.RelativeUrl}"
            });
        }
        catch (LocalGenException ex)
        {
            return TypedResults.BadRequest(ex.Message);
        }
    }

    private static Results<PhysicalFileHttpResult, NotFound> Download(string id, FileStore store)
    {
        var file = store.Get(id);

        return file is null
            ? TypedResults.NotFound()
            // Range support so a client can seek within an attached video or audio file.
            : TypedResults.PhysicalFile(file.Path, file.MediaType, enableRangeProcessing: true);
    }
}

public sealed record UploadedFile
{
    public required string Id { get; init; }

    public required string FileName { get; init; }

    public required string MediaType { get; init; }

    public required string Url { get; init; }

    public long SizeBytes { get; init; }
}
