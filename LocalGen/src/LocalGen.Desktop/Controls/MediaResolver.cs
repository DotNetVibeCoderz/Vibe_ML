using Avalonia.Media.Imaging;
using Avalonia.Threading;

namespace LocalGen.Desktop.Controls;

/// <summary>What a link points at, which decides how it is rendered.</summary>
public enum MediaKind
{
    Link,
    Image,
    Video,
    Audio,
    Document
}

/// <summary>
/// Classifies and loads media referenced from a markdown document.
/// </summary>
/// <remarks>
/// Bitmaps are cached by URL because a transcript re-renders on every token while a reply
/// streams; decoding the same image on each pass would make the panel stutter.
/// </remarks>
public static class MediaResolver
{
    private static readonly Dictionary<string, Bitmap> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly SemaphoreSlim CacheLock = new(1, 1);

    private static readonly HttpClient Http = new()
    {
        Timeout = TimeSpan.FromSeconds(20)
    };

    /// <summary>Largest image decoded into memory, to keep a huge attachment from stalling the UI.</summary>
    private const int MaxImageBytes = 24 * 1024 * 1024;

    public static MediaKind Classify(string url)
    {
        var path = url.Split('?')[0].Split('#')[0];

        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp" or ".tiff" => MediaKind.Image,
            ".mp4" or ".webm" or ".mov" or ".mkv" or ".avi" => MediaKind.Video,
            ".mp3" or ".wav" or ".ogg" or ".m4a" or ".flac" or ".aac" => MediaKind.Audio,
            ".pdf" or ".docx" or ".xlsx" or ".pptx" or ".csv" or ".zip" => MediaKind.Document,
            _ => MediaKind.Link
        };
    }

    /// <summary>
    /// Loads a bitmap from an http(s) URL or a local path. Returns null rather than throwing —
    /// a broken image in a transcript is a rendering detail, not a failure worth surfacing.
    /// </summary>
    public static async Task<Bitmap?> LoadImageAsync(string url, CancellationToken cancellationToken = default)
    {
        await CacheLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (Cache.TryGetValue(url, out var cached))
            {
                return cached;
            }
        }
        finally
        {
            CacheLock.Release();
        }

        try
        {
            byte[] bytes;

            if (url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                using var response = await Http.GetAsync(url, cancellationToken).ConfigureAwait(false);

                if (!response.IsSuccessStatusCode ||
                    response.Content.Headers.ContentLength > MaxImageBytes)
                {
                    return null;
                }

                bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken).ConfigureAwait(false);
            }
            else
            {
                var path = url.StartsWith("file://", StringComparison.OrdinalIgnoreCase)
                    ? new Uri(url).LocalPath
                    : url;

                if (!File.Exists(path) || new FileInfo(path).Length > MaxImageBytes)
                {
                    return null;
                }

                bytes = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
            }

            using var stream = new MemoryStream(bytes);

            // Bitmap construction touches rendering resources, so it happens on the UI thread.
            var bitmap = await Dispatcher.UIThread.InvokeAsync(() => new Bitmap(stream));

            await CacheLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                Cache[url] = bitmap;
            }
            finally
            {
                CacheLock.Release();
            }

            return bitmap;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or TaskCanceledException
                                       or UriFormatException or ArgumentException or NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>
    /// Hands a URL to the operating system. Used for links, and for video and audio, which
    /// Avalonia has no player for — opening the user's own media application is the honest
    /// alternative to embedding one.
    /// </summary>
    public static void OpenExternally(string url)
    {
        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = url,
                UseShellExecute = true
            });
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            // No handler registered for the scheme; nothing useful to do about it here.
        }
    }
}
