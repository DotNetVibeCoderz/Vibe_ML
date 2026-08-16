using Avalonia;
using Avalonia.Headless;
using Avalonia.Themes.Fluent;

[assembly: AvaloniaTestApplication(typeof(LocalGen.Desktop.Tests.TestApp))]

namespace LocalGen.Desktop.Tests;

/// <summary>
/// The Avalonia application the headless test platform runs.
/// </summary>
/// <remarks>
/// Deliberately minimal: the renderer under test resolves its colours through
/// <c>TryFindResource</c> and falls back to null when a key is missing, so the tests do not need
/// LocalGen's own theme loaded. Fluent is included only because control templates require one.
/// </remarks>
public sealed class TestApp : Application
{
    public override void Initialize() => Styles.Add(new FluentTheme());

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<TestApp>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
}
