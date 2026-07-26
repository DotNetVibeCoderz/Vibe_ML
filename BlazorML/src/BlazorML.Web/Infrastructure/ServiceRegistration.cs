using BlazorML.Agents.Chat;
using BlazorML.Agents.Kernel;
using BlazorML.Agents.Plugins;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Infrastructure.Data;
using BlazorML.Infrastructure.Datasets;
using BlazorML.Infrastructure.Email;
using BlazorML.Infrastructure.Identity;
using BlazorML.Infrastructure.Settings;
using BlazorML.Infrastructure.Storage;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Models;
using BlazorML.ML.Scripting;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.SemanticKernel;

namespace BlazorML.Web.Infrastructure;

public static class ServiceRegistration
{
    public static IServiceCollection AddBlazorMlStudio(this IServiceCollection services,
        IConfiguration configuration, IWebHostEnvironment environment)
    {
        services.AddDatabase(configuration, environment);
        services.AddIdentityStack();
        services.AddPlatformServices(environment);
        services.AddMachineLearning();
        services.AddProfesorWicak();

        return services;
    }

    private static void AddDatabase(this IServiceCollection services, IConfiguration configuration,
        IWebHostEnvironment environment)
    {
        // The database section is the one piece of configuration that cannot live in the database
        // it configures, so it is read from appsettings.json only. The Settings page edits that
        // file directly and asks for a restart.
        var options = new DatabaseOptions();
        configuration.GetSection(SettingsSections.Database).Bind(options);

        services.AddSingleton(options);
        services.AddDbContext<AppDbContext>(builder =>
            DatabaseProviderSetup.Configure(builder, options, environment.ContentRootPath));
    }

    private static void AddIdentityStack(this IServiceCollection services)
    {
        services.AddIdentity<ApplicationUser, IdentityRole>(identity =>
            {
                identity.Password.RequiredLength = 8;
                identity.Password.RequireNonAlphanumeric = false;
                identity.User.RequireUniqueEmail = true;
                identity.SignIn.RequireConfirmedAccount = false;
            })
            .AddEntityFrameworkStores<AppDbContext>()
            .AddDefaultTokenProviders();

        services.ConfigureApplicationCookie(cookie =>
        {
            cookie.LoginPath = "/masuk";
            cookie.LogoutPath = "/keluar";
            cookie.AccessDeniedPath = "/masuk";
            cookie.ExpireTimeSpan = TimeSpan.FromDays(14);
            cookie.SlidingExpiration = true;
        });

        services.AddCascadingAuthenticationState();
    }

    private static void AddPlatformServices(this IServiceCollection services, IWebHostEnvironment environment)
    {
        services.AddDataProtection().SetApplicationName("BlazorML.Studio");
        services.AddSingleton<ISecretProtector, DataProtectionSecretProtector>();

        services.AddSingleton<ISettingsService, SettingsService>();

        services.AddSingleton<IStorageProviderFactory>(sp =>
            new StorageProviderFactory(sp.GetRequiredService<ISettingsService>(), environment.ContentRootPath));

        services.AddScoped<IDatasetService, DatasetService>();
        services.AddSingleton<IEmailSender, SmtpEmailSender>();

        services.AddHttpClient("modules", client => client.Timeout = TimeSpan.FromMinutes(2));
        services.AddHttpClient("agent", client => client.Timeout = TimeSpan.FromSeconds(60));

        // Trial calls from the Endpoint page. Short timeout because someone is watching it happen;
        // a first call that has to load a large model is the slow case, hence not shorter.
        services.AddHttpClient(Services.EndpointTester.ClientName,
            client => client.Timeout = TimeSpan.FromSeconds(45));

        services.AddScoped<Services.ExperimentService>();
        services.AddScoped<Services.EndpointService>();
        services.AddScoped<Services.EndpointTester>();
        services.AddScoped<Services.MarkdownRenderer>();
        services.AddSingleton<Services.ThemeState>();
        services.AddScoped<Services.RunBroadcaster>();
    }

    private static void AddMachineLearning(this IServiceCollection services)
    {
        services.AddScoped<IModelRegistry, ModelRegistry>();
        services.AddScoped<IExperimentRunner, ExperimentRunner>();

        // Executors are matched to a module by CanExecute, so order here does not matter.
        services.AddScoped<IModuleExecutor, DataInputExecutor>();
        services.AddScoped<IModuleExecutor, TransformExecutor>();
        services.AddScoped<IModuleExecutor, LlmExecutor>();
        services.AddScoped<IModuleExecutor, AlgorithmExecutor>();
        services.AddScoped<IModuleExecutor, TrainingExecutor>();
        services.AddScoped<IModuleExecutor, ScoringExecutor>();
        services.AddScoped<IModuleExecutor, OutputExecutor>();
        services.AddScoped<IModuleExecutor, ScriptExecutor>();

        services.AddScoped<IScriptRunner, CSharpScriptRunner>();
        services.AddScoped<IScriptRunner, JavaScriptRunner>();

        services.AddScoped<IScriptRunner>(sp => new PythonScriptRunner(Scripting(sp)));
        services.AddScoped<IScriptRunner>(sp => new RScriptRunner(Scripting(sp)));
    }

    private static ScriptingOptions Scripting(IServiceProvider sp) =>
        sp.GetRequiredService<ISettingsService>()
            .GetAsync<ScriptingOptions>(SettingsSections.Scripting)
            .GetAwaiter().GetResult();

    private static void AddProfesorWicak(this IServiceCollection services)
    {
        services.AddSingleton<KernelFactory>();
        services.AddScoped<WicakChatService>();
        services.AddSingleton<ILlmActionRunner, LlmActionRunner>();

        // Plugins are registered as KernelPlugin so the factory can attach every one it finds
        // without naming them individually.
        services.AddSingleton<KernelPlugin>(sp =>
            KernelPluginFactory.CreateFromObject(new TimeAndMathPlugin(), "Utilities"));

        services.AddSingleton<KernelPlugin>(sp =>
            KernelPluginFactory.CreateFromObject(
                new WebPlugin(sp.GetRequiredService<ISettingsService>(),
                    sp.GetRequiredService<IHttpClientFactory>()), "Web"));

        services.AddSingleton<KernelPlugin>(sp =>
            KernelPluginFactory.CreateFromObject(
                new DataPlugin(sp.GetRequiredService<IServiceScopeFactory>()), "Datasets"));

        services.AddSingleton<KernelPlugin>(sp =>
            KernelPluginFactory.CreateFromObject(
                new DesignerPlugin(sp.GetRequiredService<IServiceScopeFactory>()), "Designer"));
    }
}

/// <summary>
/// Protects credential-bearing settings before they are written to the database, using the
/// ASP.NET Core key ring so the keys are managed and rotated for us.
/// </summary>
public sealed class DataProtectionSecretProtector(IDataProtectionProvider provider) : ISecretProtector
{
    private readonly IDataProtector _protector = provider.CreateProtector("BlazorML.Settings.v1");

    public string Protect(string plaintext) => _protector.Protect(plaintext);

    public string? Unprotect(string protectedText)
    {
        try
        {
            return _protector.Unprotect(protectedText);
        }
        catch (System.Security.Cryptography.CryptographicException)
        {
            // The key ring was reset or the row was written by another installation. Returning
            // null makes the caller fall back to the appsettings baseline instead of crashing.
            return null;
        }
    }
}
