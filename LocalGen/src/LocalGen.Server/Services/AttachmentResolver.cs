using LocalGen.Core.Inference;
using LocalGen.Runtime.Files;
using Microsoft.Extensions.Logging;

namespace LocalGen.Server.Services;

/// <summary>
/// Fills in the bytes for attachments this server can serve.
/// </summary>
/// <remarks>
/// The protocol mapper carries an <c>image_url</c> as a source with no data, because reading a
/// file is I/O and that layer performs none. This resolves the ones pointing at this server's own
/// file store, so a vision model receives real pixels rather than a URL it cannot fetch.
///
/// Remote URLs are deliberately left unresolved: fetching them would make inference depend on the
/// network, which offline mode exists to prevent.
/// </remarks>
public sealed class AttachmentResolver
{
    private const string FilesPath = "/api/files/";

    private readonly FileStore _files;
    private readonly ILogger<AttachmentResolver> _logger;

    public AttachmentResolver(FileStore files, ILogger<AttachmentResolver> logger)
    {
        _files = files;
        _logger = logger;
    }

    /// <summary>Returns the request with locally-served images loaded, or unchanged if none are.</summary>
    public async ValueTask<ChatRequest> ResolveAsync(
        ChatRequest request,
        CancellationToken cancellationToken = default)
    {
        if (!request.Messages.Any(HasUnresolvedImage))
        {
            return request;
        }

        var messages = new List<ChatMessage>(request.Messages.Count);

        foreach (var message in request.Messages)
        {
            messages.Add(HasUnresolvedImage(message)
                ? message with { Content = await ResolveContentAsync(message.Content, cancellationToken).ConfigureAwait(false) }
                : message);
        }

        return request with { Messages = messages };
    }

    private static bool HasUnresolvedImage(ChatMessage message) =>
        message.Content.Any(static part =>
            part is ContentPart.Image { Data.IsEmpty: true, Source: not null });

    private async ValueTask<IReadOnlyList<ContentPart>> ResolveContentAsync(
        IReadOnlyList<ContentPart> content,
        CancellationToken cancellationToken)
    {
        var resolved = new List<ContentPart>(content.Count);

        foreach (var part in content)
        {
            if (part is not ContentPart.Image { Data.IsEmpty: true, Source: { } source } image)
            {
                resolved.Add(part);
                continue;
            }

            var id = ExtractFileId(source);

            if (id is null)
            {
                // A remote URL. Left as-is; prompt flattening names it so the model knows an
                // image was attached that it could not read.
                resolved.Add(part);
                continue;
            }

            try
            {
                var bytes = await _files.ReadAsync(id, cancellationToken).ConfigureAwait(false);
                var stored = _files.Get(id);

                resolved.Add(image with
                {
                    Data = bytes,
                    MediaType = stored?.MediaType ?? image.MediaType
                });
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogWarning(ex, "Could not read attachment {Id}", id);
                resolved.Add(part);
            }
        }

        return resolved;
    }

    /// <summary>Pulls the file id out of a URL pointing at this server's file endpoint.</summary>
    private static string? ExtractFileId(string url)
    {
        var index = url.IndexOf(FilesPath, StringComparison.OrdinalIgnoreCase);

        if (index < 0)
        {
            return null;
        }

        var id = url[(index + FilesPath.Length)..].Split('?')[0].Split('#')[0];

        return string.IsNullOrWhiteSpace(id) ? null : id;
    }
}
