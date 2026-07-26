using System.Security.Cryptography;
using System.Text;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace BlazorML.Web.Services;

/// <summary>Publishes trained models as REST endpoints and manages the keys that open them.</summary>
public sealed class EndpointService(AppDbContext db)
{
    public async Task<List<ServiceEndpoint>> ListAsync(CancellationToken ct = default) =>
        await db.Endpoints.AsNoTracking()
            .Include(e => e.Model)
            .OrderByDescending(e => e.CreatedAt)
            .ToListAsync(ct);

    public async Task<(ServiceEndpoint Endpoint, string PlainKey)> PublishAsync(string modelId, string name,
        string? description, string? ownerId, CancellationToken ct = default)
    {
        var model = await db.TrainedModels.FirstOrDefaultAsync(m => m.Id == modelId, ct)
            ?? throw new InvalidOperationException("That model no longer exists.");

        var endpoint = new ServiceEndpoint
        {
            Name = name,
            Description = description,
            ModelId = modelId,
            OwnerId = ownerId,
            Status = EndpointStatus.Live,
            Slug = await UniqueSlugAsync(name, ct)
        };

        db.Endpoints.Add(endpoint);

        var (apiKey, plain) = CreateKey("Kunci pertama", endpoint.Id, ownerId);
        db.ApiKeys.Add(apiKey);

        await db.SaveChangesAsync(ct);
        return (endpoint, plain);
    }

    public async Task<(ApiKey Key, string PlainKey)> AddKeyAsync(string endpointId, string name,
        string? ownerId, DateTimeOffset? expiresAt = null, CancellationToken ct = default)
    {
        var (apiKey, plain) = CreateKey(name, endpointId, ownerId);
        apiKey.ExpiresAt = expiresAt;

        db.ApiKeys.Add(apiKey);
        await db.SaveChangesAsync(ct);

        return (apiKey, plain);
    }

    public async Task RevokeKeyAsync(string keyId, CancellationToken ct = default)
    {
        var key = await db.ApiKeys.FirstOrDefaultAsync(k => k.Id == keyId, ct);
        if (key is null)
        {
            return;
        }

        key.IsRevoked = true;
        await db.SaveChangesAsync(ct);
    }

    public async Task SetStatusAsync(string endpointId, EndpointStatus status, CancellationToken ct = default)
    {
        var endpoint = await db.Endpoints.FirstOrDefaultAsync(e => e.Id == endpointId, ct);
        if (endpoint is null)
        {
            return;
        }

        endpoint.Status = status;
        await db.SaveChangesAsync(ct);
    }

    /// <summary>
    /// Checks a presented key against an endpoint. Returns the endpoint when the key is valid,
    /// live and unexpired, and null otherwise — the caller turns that into a 401 without saying
    /// which of those conditions failed.
    /// </summary>
    public async Task<ServiceEndpoint?> AuthenticateAsync(string slug, string? presentedKey,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(presentedKey))
        {
            return null;
        }

        var endpoint = await db.Endpoints
            .Include(e => e.Model)
            .FirstOrDefaultAsync(e => e.Slug == slug, ct);

        if (endpoint is null || endpoint.Status != EndpointStatus.Live)
        {
            return null;
        }

        var hash = Hash(presentedKey);

        var key = await db.ApiKeys.FirstOrDefaultAsync(k =>
            k.KeyHash == hash && k.EndpointId == endpoint.Id && !k.IsRevoked, ct);

        if (key is null || (key.ExpiresAt is not null && key.ExpiresAt < DateTimeOffset.UtcNow))
        {
            return null;
        }

        key.LastUsedAt = DateTimeOffset.UtcNow;
        endpoint.LastCalledAt = key.LastUsedAt;
        endpoint.CallCount++;
        await db.SaveChangesAsync(ct);

        return endpoint;
    }

    public async Task<List<ApiKey>> KeysAsync(string endpointId, CancellationToken ct = default) =>
        await db.ApiKeys.AsNoTracking()
            .Where(k => k.EndpointId == endpointId)
            .OrderByDescending(k => k.CreatedAt)
            .ToListAsync(ct);

    /// <summary>
    /// Generates a key and returns it in plaintext exactly once. Only the hash is stored, so a
    /// leak of the database does not leak working keys.
    /// </summary>
    private static (ApiKey Key, string Plain) CreateKey(string name, string endpointId, string? ownerId)
    {
        var secret = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32))
            .Replace("+", string.Empty).Replace("/", string.Empty).Replace("=", string.Empty);

        var plain = $"bml_{secret}";

        var key = new ApiKey
        {
            Name = name,
            EndpointId = endpointId,
            OwnerId = ownerId,
            KeyHash = Hash(plain),
            Prefix = plain[..12]
        };

        return (key, plain);
    }

    private static string Hash(string value) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)));

    private async Task<string> UniqueSlugAsync(string name, CancellationToken ct)
    {
        var basis = new string(name.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '-').ToArray()).Trim('-');

        while (basis.Contains("--"))
        {
            basis = basis.Replace("--", "-");
        }

        if (string.IsNullOrWhiteSpace(basis))
        {
            basis = "model";
        }

        var slug = basis;
        var suffix = 2;

        while (await db.Endpoints.AnyAsync(e => e.Slug == slug, ct))
        {
            slug = $"{basis}-{suffix++}";
        }

        return slug;
    }
}
