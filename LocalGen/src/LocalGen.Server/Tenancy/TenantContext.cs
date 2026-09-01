namespace LocalGen.Server.Tenancy;

/// <summary>
/// Exposes the tenant behind the request currently being served.
/// </summary>
/// <remarks>
/// The middleware authenticates a key long before the code that spends the quota runs, and the
/// spending happens in <c>InferenceService</c>, which is a singleton shared with the in-process
/// desktop host. Flowing the tenant through the ambient <see cref="HttpContext"/> keeps the
/// inference path from having to take a tenant parameter it would have nothing to pass for when
/// there is no HTTP request at all — which is exactly the Admin Control's Playground.
/// </remarks>
public sealed class TenantContext(IHttpContextAccessor accessor)
{
    private const string ItemKey = "localgen.tenant";

    /// <summary>The current tenant, or null when running outside a request or with keys disabled.</summary>
    public ApiTenant? Current => accessor.HttpContext?.Items.TryGetValue(ItemKey, out var value) == true
        ? value as ApiTenant
        : null;

    internal static void Attach(HttpContext context, ApiTenant tenant) =>
        context.Items[ItemKey] = tenant;
}
