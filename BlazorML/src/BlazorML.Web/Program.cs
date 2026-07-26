using BlazorML.Infrastructure.Data;
using BlazorML.Web.Components;
using BlazorML.Web.Endpoints;
using BlazorML.Web.Infrastructure;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents(options => options.DetailedErrors = builder.Environment.IsDevelopment());

builder.Services.AddBlazorMlStudio(builder.Configuration, builder.Environment);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(ScoringApi.ConfigureSwagger);

// Dataset and attachment uploads are the large payloads here; the default 128 MB form limit is
// low for a real training file.
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
    options.MultipartBodyLengthLimit = 512L * 1024 * 1024);

var app = builder.Build();

// Create the schema and seed on first run. EnsureCreated rather than migrations: four database
// providers would otherwise need four separate migration sets kept in step by hand.
await using (var scope = app.Services.CreateAsyncScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    await db.Database.EnsureCreatedAsync();
    await DataSeeder.SeedAsync(scope.ServiceProvider, app.Logger);
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/galat", createScopeForErrors: true);
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseAntiforgery();

app.MapStaticAssets();

app.UseAuthentication();
app.UseAuthorization();

app.UseSwagger();
app.UseSwaggerUI(ui =>
{
    ui.SwaggerEndpoint("/swagger/v1/swagger.json", "Blazor ML Studio scoring API");
    ui.RoutePrefix = "api-docs";
    ui.DocumentTitle = "Blazor ML Studio · API";
});

app.MapScoringApi();
app.MapStorageDownloads();
app.MapAccountEndpoints();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();

/// <summary>
/// Named so the test project can boot this exact application through WebApplicationFactory.
/// Top-level statements otherwise generate an internal entry-point class.
/// </summary>
public partial class Program;
