using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Infrastructure.Settings;

/// <summary>
/// Two-layer configuration: <c>appsettings.json</c> provides the baseline, and anything the user
/// changes on the Settings page is stored as one JSON row per section that overlays it. That is
/// what makes "semua konfigurasi bisa diubah dari aplikasi" true without anyone editing a file
/// on the server.
/// </summary>
public sealed class SettingsService(
    IConfiguration configuration,
    IServiceScopeFactory scopeFactory,
    ISecretProtector? protector = null) : ISettingsService
{
    private readonly ConcurrentDictionary<string, object> _cache = new();

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        Converters = { new JsonStringEnumConverter() }
    };

    public event Action<string>? SectionChanged;

    public async Task<T> GetAsync<T>(string section, CancellationToken ct = default) where T : class, new()
    {
        if (_cache.TryGetValue(section, out var cached) && cached is T hit)
        {
            return hit;
        }

        // Baseline from appsettings.json.
        var value = new T();
        configuration.GetSection(section).Bind(value);

        // Overlay whatever the user saved in the app.
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var row = await db.AppSettings.AsNoTracking()
            .FirstOrDefaultAsync(s => s.Key == section, ct);

        if (row?.Value is { Length: > 0 })
        {
            var payload = row.IsSecret && protector is not null
                ? protector.Unprotect(row.Value)
                : row.Value;

            if (payload is not null)
            {
                try
                {
                    var stored = JsonSerializer.Deserialize<T>(payload, Json);
                    if (stored is not null)
                    {
                        value = stored;
                    }
                }
                catch (JsonException)
                {
                    // A settings row written by an older shape of this class should not stop the
                    // app from starting; the appsettings baseline stands in until it is re-saved.
                }
            }
        }

        _cache[section] = value;
        return value;
    }

    public async Task SaveAsync<T>(string section, T value, CancellationToken ct = default) where T : class
    {
        var json = JsonSerializer.Serialize(value, Json);
        var secret = CarriesSecrets(section);
        var payload = secret && protector is not null ? protector.Protect(json) : json;

        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var row = await db.AppSettings.FirstOrDefaultAsync(s => s.Key == section, ct);
        if (row is null)
        {
            db.AppSettings.Add(new AppSetting
            {
                Key = section,
                Section = section,
                Value = payload,
                IsSecret = secret
            });
        }
        else
        {
            row.Value = payload;
            row.IsSecret = secret;
        }

        await db.SaveChangesAsync(ct);

        _cache[section] = value;
        SectionChanged?.Invoke(section);
    }

    public void Invalidate(string? section = null)
    {
        if (section is null)
        {
            _cache.Clear();
        }
        else
        {
            _cache.TryRemove(section, out _);
        }
    }

    /// <summary>Sections whose payload contains credentials and so is protected before storage.</summary>
    private static bool CarriesSecrets(string section) =>
        section is Core.Configuration.SettingsSections.Chat
            or Core.Configuration.SettingsSections.Storage
            or Core.Configuration.SettingsSections.Database
            or Core.Configuration.SettingsSections.Tools;
}
