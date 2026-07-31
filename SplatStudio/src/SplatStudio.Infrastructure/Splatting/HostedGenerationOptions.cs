namespace SplatStudio.Infrastructure.Splatting;

/// <summary>
/// Configuration for one hosted image-to-3D provider.
///
/// Every provider worth using here (Rodin/Hyper3D, Tripo, Tencent Hunyuan3D, a self-hosted
/// TRELLIS runner) works the same way at the protocol level: POST the image, get a job id
/// back, poll until the job reaches a terminal state, then download the produced asset. What
/// differs between them is only naming — the paths, the multipart field name, and where the
/// job id, status and result URL sit in the JSON.
///
/// So rather than shipping three hand-written vendor clients that would rot the moment any of
/// them revises an endpoint, this describes that one shape as configuration. Point it at your
/// provider by filling in the paths below; nothing in the code assumes a particular vendor.
/// The defaults spell out the contract but are not any real provider's API, so
/// <see cref="IsConfigured"/> stays false until you set at least <see cref="BaseUrl"/> and
/// <see cref="ApiKey"/>.
/// </summary>
public class HostedGenerationOptions
{
    /// <summary>Root URL of the provider's API, e.g. "https://api.example.com".</summary>
    public string BaseUrl { get; set; } = string.Empty;

    /// <summary>Provider credential. Keep it in user secrets or the environment, not appsettings.json.</summary>
    public string ApiKey { get; set; } = string.Empty;

    /// <summary>Human-readable provider name, shown in the UI next to the mode.</summary>
    public string ProviderName { get; set; } = "hosted provider";

    // ---- Submitting the job -------------------------------------------------

    public string SubmitPath { get; set; } = "/v1/generate";

    /// <summary>Multipart field the image is uploaded under.</summary>
    public string ImageFieldName { get; set; } = "image";

    /// <summary>
    /// Extra multipart form fields sent with every submission — this is where provider-specific
    /// knobs live (model name, quality tier, output format, texture resolution...).
    /// </summary>
    public Dictionary<string, string> SubmitFields { get; set; } = new();

    public string AuthHeaderName { get; set; } = "Authorization";

    /// <summary>Prefix for the credential, e.g. "Bearer". Empty sends the key bare.</summary>
    public string AuthScheme { get; set; } = "Bearer";

    /// <summary>Dotted path to the job id in the submit response, e.g. "data.job_id".</summary>
    public string JobIdPath { get; set; } = "job_id";

    // ---- Polling ------------------------------------------------------------

    /// <summary>Status endpoint; "{jobId}" is substituted.</summary>
    public string StatusPath { get; set; } = "/v1/generate/{jobId}";

    /// <summary>Dotted path to the job state in the status response.</summary>
    public string StatusFieldPath { get; set; } = "status";

    /// <summary>State values that mean the job finished successfully (case-insensitive).</summary>
    public List<string> SuccessStates { get; set; } = new() { "succeeded", "success", "completed", "done" };

    /// <summary>State values that mean the job will never finish (case-insensitive).</summary>
    public List<string> FailureStates { get; set; } = new() { "failed", "error", "cancelled", "canceled" };

    /// <summary>Dotted path to the finished asset's download URL in the status response.</summary>
    public string ResultUrlPath { get; set; } = "result.url";

    /// <summary>Dotted path to a provider error message, used to explain failures to the user.</summary>
    public string ErrorMessagePath { get; set; } = "error";

    public int PollIntervalSeconds { get; set; } = 3;

    /// <summary>Give up after this long. These jobs run minutes, not seconds.</summary>
    public int TimeoutSeconds { get; set; } = 600;

    /// <summary>
    /// True once there is enough configuration to attempt a call. The upload page uses this to
    /// decide whether to offer the mode at all.
    /// </summary>
    public bool IsConfigured =>
        !string.IsNullOrWhiteSpace(BaseUrl) && !string.IsNullOrWhiteSpace(ApiKey);
}

/// <summary>
/// The two hosted modes, configured independently — photorealistic splatting and mesh
/// generation are usually different products even at the same vendor, and a deployment may
/// well have credentials for one and not the other.
/// </summary>
public class HostedEnginesOptions
{
    public const string SectionName = "Splatting:Hosted";

    public HostedGenerationOptions Photoreal { get; set; } = new()
    {
        ProviderName = "photorealistic 3DGS provider"
    };

    public HostedGenerationOptions Mesh { get; set; } = new()
    {
        ProviderName = "image-to-3D provider",
        SubmitFields = new Dictionary<string, string>()
    };
}
