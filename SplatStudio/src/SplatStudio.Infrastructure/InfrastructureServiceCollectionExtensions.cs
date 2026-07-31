using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.BackgroundProcessing;
using SplatStudio.Infrastructure.Data;
using SplatStudio.Infrastructure.Email;
using SplatStudio.Infrastructure.Splatting;
using SplatStudio.Infrastructure.Storage;

namespace SplatStudio.Infrastructure;

/// <summary>
/// Everything Program.cs needs to call to stand up the persistence,
/// storage, conversion-engine and email subsystems behind their
/// Application-layer interfaces. Keeping this in one method means
/// Program.cs stays a thin, readable wiring file.
/// </summary>
public static class InfrastructureServiceCollectionExtensions
{
    public static IServiceCollection AddSplatInfrastructure(
        this IServiceCollection services,
        IConfiguration configuration,
        string contentRootPath)
    {
        // ---- Database ---------------------------------------------------
        var dbProvider = DatabaseConfiguration.ParseProvider(configuration["Database:Provider"]);
        var connectionString = configuration.GetConnectionString(dbProvider.ToString())
            ?? configuration.GetConnectionString("Default")
            ?? "Data Source=App_Data/splatstudio.db";

        services.AddDbContext<ApplicationDbContext>(options =>
            DatabaseConfiguration.ConfigureProvider(options, dbProvider, connectionString));

        // ---- Identity -----------------------------------------------------
        services.AddIdentity<ApplicationUser, IdentityRole>(options =>
        {
            options.Password.RequiredLength = 8;
            options.Password.RequireNonAlphanumeric = false;
            options.Password.RequireUppercase = false;
            options.User.RequireUniqueEmail = true;
            options.SignIn.RequireConfirmedAccount = false;
        })
        .AddEntityFrameworkStores<ApplicationDbContext>()
        .AddDefaultTokenProviders();

        services.ConfigureApplicationCookie(options =>
        {
            options.LoginPath = "/Account/Login";
            options.LogoutPath = "/Account/Logout";
            options.AccessDeniedPath = "/Account/AccessDenied";
            options.ExpireTimeSpan = TimeSpan.FromDays(14);
            options.SlidingExpiration = true;
        });

        // ---- Storage --------------------------------------------------
        services.AddSplatStorage(configuration, contentRootPath);
        services.AddSingleton<IImageProcessingService, SplatStudio.Infrastructure.Imaging.ImageProcessingService>();

        // ---- Splat conversion engine -----------------------------------
        var splattingOptions = configuration.GetSection(SplattingOptions.SectionName).Get<SplattingOptions>() ?? new SplattingOptions();
        var externalOptions = configuration.GetSection(ExternalSplatEngineOptions.SectionName).Get<ExternalSplatEngineOptions>() ?? new ExternalSplatEngineOptions();
        services.AddSingleton(splattingOptions);
        services.AddSingleton(externalOptions);
        services.AddHttpClient();

        // The GPU engine owns an ILGPU context + accelerator and JIT-compiles its kernels
        // once, so it must be a singleton — building one per conversion would recompile PTX
        // every time. It is registered unconditionally (cheap when unused: the constructor
        // probes for a device and gives up quietly) so diagnostics can report GPU
        // availability regardless of which engine is configured.
        services.AddSingleton<GpuSplatEngine>();

        // Lifetime matters here, not just for efficiency. A *scoped* factory that returns
        // a shared instance makes the container take ownership of it and dispose it when
        // the scope ends — which silently tore down the singleton GPU accelerator after
        // the first conversion and made every later scene fall back to the CPU. Both the
        // local and GPU engines are stateless/thread-safe, so registering them as
        // singletons is both correct and the fix.
        if (string.Equals(splattingOptions.Engine, "ExternalApi", StringComparison.OrdinalIgnoreCase))
        {
            // Scoped: this one holds an HttpClient, which should follow the factory's
            // handler-rotation lifetime rather than living for the life of the process.
            services.AddScoped<IGaussianSplatEngine>(sp =>
                new ExternalApiSplatEngine(sp.GetRequiredService<IHttpClientFactory>().CreateClient(), externalOptions));
        }
        else if (string.Equals(splattingOptions.Engine, "Gpu", StringComparison.OrdinalIgnoreCase))
        {
            services.AddSingleton<IGaussianSplatEngine>(sp =>
            {
                var gpu = sp.GetRequiredService<GpuSplatEngine>();
                // No CUDA/OpenCL device on this machine — degrade to CPU rather than fail
                // every upload. GpuSplatEngine has already logged the reason.
                return gpu.IsAvailable ? gpu : new LocalHeuristicSplatEngine();
            });
        }
        else
        {
            services.AddSingleton<IGaussianSplatEngine>(new LocalHeuristicSplatEngine());
        }

        // ---- Selectable conversion modes --------------------------------
        // The three modes the upload page offers. Registering them all unconditionally (rather
        // than only the configured ones) is what lets the picker show a hosted mode greyed out
        // with the reason, instead of pretending the capability doesn't exist.
        var hostedOptions = configuration.GetSection(HostedEnginesOptions.SectionName).Get<HostedEnginesOptions>()
                            ?? new HostedEnginesOptions();
        services.AddSingleton(hostedOptions);

        // Hosted jobs run for minutes, so this client gets a long timeout of its own rather
        // than fighting HttpClient's 100-second default.
        services.AddHttpClient(nameof(HostedGenerationClient), c =>
            c.Timeout = Timeout.InfiniteTimeSpan);

        services.AddSingleton<IConversionEngine, HeuristicConversionEngine>();
        services.AddSingleton<IConversionEngine, HostedPhotorealConversionEngine>();
        services.AddSingleton<IConversionEngine, HostedMeshConversionEngine>();
        services.AddSingleton<IConversionEngineCatalog, ConversionEngineCatalog>();

        // ---- Background conversion queue -------------------------------
        services.AddSingleton<IConversionQueue, ChannelConversionQueue>();
        services.AddSingleton<ISceneUpdateNotifier, SceneUpdateNotifier>();
        services.AddHostedService<ConversionBackgroundService>();

        // ---- Email ------------------------------------------------------
        var emailOptions = configuration.GetSection(EmailOptions.SectionName).Get<EmailOptions>() ?? new EmailOptions();
        services.AddSingleton(emailOptions.Smtp);
        services.AddSingleton<IAppEmailSender>(sp =>
            string.Equals(emailOptions.Provider, "Smtp", StringComparison.OrdinalIgnoreCase)
                ? new SmtpEmailSender(emailOptions.Smtp)
                : new FileEmailSender(contentRootPath, sp.GetRequiredService<Microsoft.Extensions.Logging.ILogger<FileEmailSender>>()));

        return services;
    }
}
