using LocalGen.Sdk;
using LocalGen.Web.Components;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorComponents().AddInteractiveServerComponents();

// The web UI is a client of the inference server, not a second copy of it: it talks to whatever
// LocalGen instance is configured, which may be on another machine entirely.
builder.Services.AddLocalGenClient(options =>
{
    options.Endpoint = builder.Configuration["LocalGen:Endpoint"] ?? "http://127.0.0.1:11434";
    options.ApiKey = builder.Configuration["LocalGen:ApiKey"];
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>().AddInteractiveServerRenderMode();

app.Run();
