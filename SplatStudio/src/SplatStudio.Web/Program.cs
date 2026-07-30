using Microsoft.AspNetCore.ResponseCompression;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using SplatStudio.Infrastructure;
using SplatStudio.Infrastructure.Data;
using SplatStudio.Infrastructure.Storage;
using SplatStudio.Web;
using SplatStudio.Web.Components;

var builder = WebApplication.CreateBuilder(args);

// ---- Blazor Server (interactive server render mode, global) -------------
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents(options =>
    {
        // DetailedErrors are useful in dev, but leak stack traces to the
        // client in production — only enabled below the environment check.
        options.DetailedErrors = builder.Environment.IsDevelopment();
    });

// Account/login/logout/reset-password endpoints use classic Razor Pages so
// they can write authentication cookies directly (a Blazor Server circuit,
// once the response has started streaming over SignalR, cannot do this).
builder.Services.AddRazorPages();
builder.Services.AddCascadingAuthenticationState();
builder.Services.AddAntiforgery();

// SignalR circuit tuning: keep payloads small since large binary content
// (splat files, full-res images) is served over plain HTTP/static files,
// never pushed through the circuit itself.
builder.Services.AddSignalR(options =>
{
    options.MaximumReceiveMessageSize = 512 * 1024;
});

builder.Services.AddResponseCompression(options =>
{
    options.EnableForHttps = true;
    options.Providers.Add<BrotliCompressionProvider>();
    options.Providers.Add<GzipCompressionProvider>();
    options.MimeTypes = ResponseCompressionDefaults.MimeTypes.Concat(new[]
    {
        "application/octet-stream" // .splat payloads compress very well (lots of repeated bytes)
    });
});

builder.Services.AddSplatInfrastructure(builder.Configuration, builder.Environment.ContentRootPath);

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseResponseCompression();
app.UseHttpsRedirection();

// Default wwwroot static files (css/js/sample-data) — long cache lifetime
// since filenames are fingerprint-free but content rarely changes post-deploy.
app.UseStaticFiles();

// Second static-file mapping for the FileSystem storage backend: serves
// uploaded images/thumbnails/splats from outside wwwroot, so they survive
// a fresh deployment that overwrites wwwroot, and aren't bundled into the
// published app's static-asset manifest.
var storageOptions = app.Services.GetRequiredService<StorageOptions>();
var storageRoot = Path.IsPathRooted(storageOptions.FileSystem.RootPath)
    ? storageOptions.FileSystem.RootPath
    : Path.Combine(app.Environment.ContentRootPath, storageOptions.FileSystem.RootPath);
Directory.CreateDirectory(storageRoot);
// The static-file middleware refuses to serve extensions it has no content type for,
// so ".splat" — the app's primary artifact — 404s unless it is mapped explicitly.
var mediaContentTypes = new FileExtensionContentTypeProvider();
mediaContentTypes.Mappings[".splat"] = "application/octet-stream";

app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(storageRoot),
    RequestPath = storageOptions.FileSystem.PublicBasePath,
    ContentTypeProvider = mediaContentTypes,
    OnPrepareResponse = ctx =>
    {
        ctx.Context.Response.Headers.CacheControl = "public,max-age=2592000,immutable";
    }
});

app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();

// Razor Components endpoints carry anti-forgery metadata, so this middleware is
// mandatory — without it every component route fails with a 500. Must sit after
// UseAuthentication/UseAuthorization and before the endpoint mappings.
app.UseAntiforgery();

app.MapRazorPages();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// ---- First-run database creation + demo data ----------------------------
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
    await db.Database.EnsureCreatedAsync();
    await SeedData.EnsureSeededAsync(scope.ServiceProvider, app.Environment);
}

app.Run();
