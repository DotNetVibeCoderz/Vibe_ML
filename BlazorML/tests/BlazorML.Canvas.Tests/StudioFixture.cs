using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using Microsoft.Playwright;

namespace BlazorML.Canvas.Tests;

/// <summary>
/// Runs the real application as a child process and drives it with a real browser. Nothing else
/// can verify the canvas: it is D3 laying out SVG and HTML, driven by JS interop over a Blazor
/// circuit, and none of that exists until a browser executes it.
/// <para>
/// The app is started rather than hosted in-process because <c>WebApplicationFactory</c>'s test
/// server has no socket for a browser to connect to. Its database and storage go to a temp
/// folder, so a run never touches the developer's own workspace.
/// </para>
/// </summary>
public sealed class StudioFixture : IAsyncLifetime
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"blazorml-canvas-{Guid.NewGuid():n}");
    private Process? _app;
    private IPlaywright? _playwright;

    public IBrowser? Browser { get; private set; }
    public string BaseUrl { get; private set; } = string.Empty;

    /// <summary>
    /// The database the child process is using. Exposed so a test can put a row in front of the
    /// running app — nothing seeds a published endpoint, and the Endpoint page has nothing to show
    /// without one.
    /// </summary>
    public string DatabasePath => Path.Combine(_root, "canvas.db");

    public const string Email = "admin@gravicode.com";
    public const string Password = "StudioML#2026";

    /// <summary>
    /// Whether the browser is present is decided at discovery by <see cref="CanvasFactAttribute"/>,
    /// so by the time this runs the environment is known good. Anything that fails here is a real
    /// problem and is allowed to throw — swallowing it would turn a broken suite into a green one.
    /// </summary>
    public async Task InitializeAsync()
    {
        if (!BrowserProbe.IsInstalled)
        {
            // Every test is skipped, so there is nothing to set up.
            return;
        }

        _playwright = await Playwright.CreateAsync();

        Browser = await _playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
        {
            Headless = true
        });

        BaseUrl = await StartAppAsync();
    }

    private async Task<string> StartAppAsync()
    {
        Directory.CreateDirectory(_root);

        var port = FreePort();
        var url = $"http://127.0.0.1:{port}";

        // The built assembly is launched directly rather than through `dotnet run`, which reads
        // launchSettings.json and overrides ASPNETCORE_URLS with its own port — so the fixture
        // would wait on a port the app was never listening on.
        var output = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "BlazorML.Web", "bin", "Debug", "net10.0"));

        var assembly = Path.Combine(output, "BlazorML.Web.dll");

        if (!File.Exists(assembly))
        {
            throw new FileNotFoundException(
                $"The web app has not been built. Expected it at {assembly}.", assembly);
        }

        var start = new ProcessStartInfo("dotnet")
        {
            Arguments = $"\"{assembly}\"",
            WorkingDirectory = output,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        start.Environment["ASPNETCORE_URLS"] = url;
        start.Environment["ASPNETCORE_ENVIRONMENT"] = "Development";

        // Point the workspace at a temp folder so the suite seeds its own data and leaves the
        // developer's database alone.
        start.Environment["Database__ConnectionStrings__Sqlite"] =
            $"Data Source={Path.Combine(_root, "canvas.db")}";
        start.Environment["Storage__FileSystem__RootPath"] = Path.Combine(_root, "storage");

        // Blank every provider credential. In Development the app also reads user secrets, so a
        // developer who has their own key would otherwise see different behaviour from CI — and
        // the "no provider configured" test would pass or fail depending on whose machine it ran
        // on. Environment variables sit after user secrets in the configuration chain, so this
        // wins. It also guarantees no test can spend money on a real model.
        foreach (var key in (string[])
                 ["Chat__OpenAI__ApiKey", "Chat__Anthropic__ApiKey", "Chat__Gemini__ApiKey",
                  "Tools__TavilyApiKey"])
        {
            start.Environment[key] = string.Empty;
        }

        start.Environment["Chat__Ollama__Enabled"] = "false";

        _app = Process.Start(start)
            ?? throw new InvalidOperationException("dotnet run did not start.");

        // Drained so a full pipe cannot block the child, and kept so a startup failure can be
        // reported with the app's own words instead of a bare timeout.
        var stdout = _app.StandardOutput.ReadToEndAsync();
        var stderr = _app.StandardError.ReadToEndAsync();

        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var deadline = DateTime.UtcNow.AddSeconds(120);

        while (DateTime.UtcNow < deadline)
        {
            if (_app.HasExited)
            {
                throw new InvalidOperationException(
                    $"The app exited with code {_app.ExitCode}.\n{await stdout}\n{await stderr}");
            }

            try
            {
                var response = await client.GetAsync($"{url}/masuk");

                if (response.IsSuccessStatusCode)
                {
                    return url;
                }
            }
            catch (HttpRequestException)
            {
                // Not listening yet.
            }

            await Task.Delay(500);
        }

        _app.Kill(entireProcessTree: true);

        throw new TimeoutException(
            $"The app did not become ready on {url} within two minutes.\n{await stdout}\n{await stderr}");
    }

    private static int FreePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();

        return port;
    }

    /// <summary>
    /// A page plus the browser context that owns it. Playwright's <see cref="IPage"/> is not
    /// disposable, and leaking a context per test would leave browser processes behind for the
    /// whole run.
    /// </summary>
    public sealed class PageSession(IBrowserContext context, IPage page) : IAsyncDisposable
    {
        public IPage Page { get; } = page;

        public async ValueTask DisposeAsync() => await context.CloseAsync();
    }

    /// <summary>A page already signed in, since every canvas route is behind authentication.</summary>
    public async Task<IPage> SignedInPageAsync()
    {
        var context = await Browser!.NewContextAsync(new BrowserNewContextOptions
        {
            ViewportSize = new ViewportSize { Width = 1600, Height = 1000 }
        });

        var page = await context.NewPageAsync();

        await page.GotoAsync($"{BaseUrl}/masuk");
        await page.FillAsync("#email", Email);
        await page.FillAsync("#password", Password);
        await page.ClickAsync("button[type=submit]");

        // Sign-in is a form post that redirects; wait for the shell rather than a URL match.
        await page.WaitForSelectorAsync(".rail", new PageWaitForSelectorOptions { Timeout = 30000 });

        return page;
    }

    /// <summary>
    /// A signed-in page on an ordinary route, for shell-level checks that do not need the canvas
    /// and must not pay for cloning an experiment.
    /// </summary>
    public async Task<PageSession> ShellPageAsync(string route = "/")
    {
        var page = await SignedInPageAsync();

        if (route != "/")
        {
            await page.GotoAsync($"{BaseUrl}{route}");
            await page.WaitForSelectorAsync(".rail", new PageWaitForSelectorOptions { Timeout = 30000 });
        }

        return new PageSession(page.Context, page);
    }

    /// <summary>
    /// Opens the designer on a <em>fresh copy</em> of a seeded sample.
    /// <para>
    /// Every test in this collection shares one application and one database, and most of them
    /// edit the canvas. Pointing them all at the same seeded experiment made them depend on the
    /// order they happened to run in — one test's added node broke the next test's run, and
    /// "this experiment has never been run" stopped being true after the first test that ran it.
    /// Taking a copy through the gallery's own "use this sample" button gives each test its own
    /// experiment, and exercises the clone path into the bargain.
    /// </para>
    /// </summary>
    /// <param name="experimentName">The sample to copy. Defaults to the six-node churn flow.</param>
    public async Task<PageSession> DesignerPageAsync(string experimentName = "Prediksi churn pelanggan")
    {
        var page = await SignedInPageAsync();

        await page.GotoAsync($"{BaseUrl}/galeri");

        var card = page.Locator("article.card")
            .Filter(new LocatorFilterOptions { HasTextString = experimentName });

        Assert.True(await card.CountAsync() > 0, $"No gallery card named '{experimentName}'.");

        var copy = card.First.Locator("button:has-text('Pakai contoh ini')");

        // The click is driven by an interactive circuit, and clicking before it has connected is
        // a no-op that leaves the test waiting on a page it never asked for. Wait for the button
        // to be actionable, then confirm the navigation actually started before waiting on the
        // canvas — otherwise a swallowed click surfaces as a confusing "no nodes" timeout.
        await copy.WaitForAsync(new LocatorWaitForOptions { Timeout = 30000 });

        for (var attempt = 1; attempt <= 3; attempt++)
        {
            await copy.ClickAsync();

            try
            {
                await page.WaitForFunctionAsync(
                    "() => location.pathname.startsWith('/eksperimen/')",
                    null, new PageWaitForFunctionOptions { Timeout = 10000 });

                break;
            }
            catch (TimeoutException) when (attempt < 3)
            {
                // The circuit was not listening yet. Give it a moment and press again.
                await page.WaitForTimeoutAsync(1000);
            }
        }

        Assert.Contains("/eksperimen/", page.Url);

        // Waiting on the canvas rather than the document: cloning navigates through Blazor's
        // enhanced navigation, which swaps content without a load event, and the nodes are then
        // drawn by D3 once the circuit reconnects.
        await page.WaitForSelectorAsync(".node", new PageWaitForSelectorOptions { Timeout = 45000 });

        return new PageSession(page.Context, page);
    }

    public async Task DisposeAsync()
    {
        if (Browser is not null)
        {
            await Browser.CloseAsync();
        }

        _playwright?.Dispose();

        if (_app is { HasExited: false })
        {
            _app.Kill(entireProcessTree: true);
            await _app.WaitForExitAsync();
        }

        _app?.Dispose();

        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp folder is not worth failing a run over.
        }
    }
}

[CollectionDefinition("canvas")]
public sealed class CanvasCollection : ICollectionFixture<StudioFixture>;
