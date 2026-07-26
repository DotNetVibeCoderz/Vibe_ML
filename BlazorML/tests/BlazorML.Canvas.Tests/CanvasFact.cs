namespace BlazorML.Canvas.Tests;

/// <summary>
/// A browser test. Skipped — visibly, with a reason — when Playwright's browser has not been
/// downloaded, because that is a fact about the machine rather than a defect in the code.
/// <para>
/// Anything else is a failure. An earlier version of this suite swallowed every problem into a
/// silent pass, which meant eighteen tests "succeeded" in 191 ms without a browser ever opening.
/// A test that cannot run must say so; it must never report success.
/// </para>
/// </summary>
public sealed class CanvasFactAttribute : FactAttribute
{
    public CanvasFactAttribute()
    {
        if (!BrowserProbe.IsInstalled)
        {
            Skip = "Playwright's browser is not installed. From the test project's output folder run: " +
                   ".playwright/node/<platform>/node .playwright/package/cli.js install chromium";
        }
    }
}

/// <summary>
/// The <see cref="TheoryAttribute"/> counterpart, so a parametric browser test skips for the same
/// reason instead of failing on a browser that was never downloaded.
/// </summary>
public sealed class CanvasTheoryAttribute : TheoryAttribute
{
    public CanvasTheoryAttribute()
    {
        if (!BrowserProbe.IsInstalled)
        {
            Skip = "Playwright's browser is not installed. From the test project's output folder run: " +
                   ".playwright/node/<platform>/node .playwright/package/cli.js install chromium";
        }
    }
}

internal static class BrowserProbe
{
    /// <summary>
    /// Checked by looking for the download rather than by launching one: this runs at discovery,
    /// once per test, and starting a browser there would be far too slow.
    /// </summary>
    public static bool IsInstalled { get; } = Probe();

    private static bool Probe()
    {
        var overridden = Environment.GetEnvironmentVariable("PLAYWRIGHT_BROWSERS_PATH");

        var root = !string.IsNullOrWhiteSpace(overridden)
            ? overridden
            : OperatingSystem.IsWindows()
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ms-playwright")
                : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "ms-playwright");

        return Directory.Exists(root)
               && Directory.EnumerateDirectories(root, "chromium*").Any();
    }
}
