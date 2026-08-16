using Avalonia;
using Avalonia.Styling;

namespace LocalGen.Desktop.Services;

/// <summary>
/// Switches between the light and dark themes and remembers the choice.
/// </summary>
/// <remarks>
/// The preference is written next to the rest of LocalGen's data rather than to the registry or
/// a platform-specific settings store, so the same file works on every platform Avalonia targets.
/// </remarks>
public sealed class ThemeService
{
    private readonly string _preferencePath;

    public ThemeService(Microsoft.Extensions.Options.IOptions<Core.Configuration.LocalGenOptions> options)
    {
        var directory = options.Value.DataDirectory;
        Directory.CreateDirectory(directory);

        _preferencePath = Path.Combine(directory, "theme.txt");

        IsDark = ReadPreference();
        Apply();
    }

    public bool IsDark { get; private set; }

    public void Toggle()
    {
        IsDark = !IsDark;
        Apply();
        WritePreference();
    }

    private void Apply()
    {
        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = IsDark ? ThemeVariant.Dark : ThemeVariant.Light;
        }
    }

    private bool ReadPreference()
    {
        try
        {
            if (File.Exists(_preferencePath))
            {
                return File.ReadAllText(_preferencePath).Trim()
                    .Equals("dark", StringComparison.OrdinalIgnoreCase);
            }
        }
        catch (IOException)
        {
            // An unreadable preference file is not worth failing startup over.
        }

        // Dark is the default: the palette was designed for it, and it is what an instrument
        // panel looks like.
        return true;
    }

    private void WritePreference()
    {
        try
        {
            File.WriteAllText(_preferencePath, IsDark ? "dark" : "light");
        }
        catch (IOException)
        {
        }
    }
}
