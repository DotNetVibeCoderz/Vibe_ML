using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;

namespace LocalGen.Desktop.Services;

/// <summary>
/// Opens the platform file picker.
/// </summary>
/// <remarks>
/// Behind an interface because picking a file needs a window, and a view model should not reach
/// for one. It also makes the attachment commands testable without a UI.
/// </remarks>
public interface IFileDialogService
{
    Task<IReadOnlyList<string>> PickImagesAsync();

    Task<IReadOnlyList<string>> PickDocumentsAsync();
}

public sealed class FileDialogService : IFileDialogService
{
    private static readonly FilePickerFileType Images = new("Images")
    {
        Patterns = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.bmp"],
        MimeTypes = ["image/*"]
    };

    private static readonly FilePickerFileType Documents = new("Documents and code")
    {
        Patterns =
        [
            "*.pdf", "*.md", "*.markdown", "*.txt", "*.log", "*.csv", "*.json", "*.xml",
            "*.yaml", "*.yml", "*.html", "*.htm",
            "*.cs", "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java", "*.kt", "*.sql", "*.sh", "*.ps1"
        ]
    };

    public Task<IReadOnlyList<string>> PickImagesAsync() =>
        PickAsync("Attach images", Images);

    public Task<IReadOnlyList<string>> PickDocumentsAsync() =>
        PickAsync("Attach documents", Documents);

    private static async Task<IReadOnlyList<string>> PickAsync(string title, FilePickerFileType filter)
    {
        var window = (Avalonia.Application.Current?.ApplicationLifetime
            as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

        if (window?.StorageProvider is not { CanOpen: true } provider)
        {
            return [];
        }

        var files = await provider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = true,
            FileTypeFilter = [filter, FilePickerFileTypes.All]
        });

        // Only local files can be read; a picker may hand back a cloud item with no path.
        return [.. files
            .Select(file => file.TryGetLocalPath())
            .Where(path => !string.IsNullOrEmpty(path))
            .Select(path => path!)];
    }
}
