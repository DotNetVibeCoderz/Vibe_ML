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

        services.AddScoped<IGaussianSplatEngine>(sp =>
            string.Equals(splattingOptions.Engine, "ExternalApi", StringComparison.OrdinalIgnoreCase)
                ? new ExternalApiSplatEngine(sp.GetRequiredService<IHttpClientFactory>().CreateClient(), externalOptions)
                : new LocalHeuristicSplatEngine());

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
