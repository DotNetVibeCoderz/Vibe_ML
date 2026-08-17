namespace LocalGen.Core;

/// <summary>Base type for errors LocalGen raises deliberately, as opposed to unexpected faults.</summary>
public class LocalGenException(string message, Exception? innerException = null)
    : Exception(message, innerException);

/// <summary>A model was requested that is not installed locally.</summary>
public sealed class ModelNotFoundException(string modelId)
    : LocalGenException($"Model '{modelId}' is not installed. Pull it first with: localgen pull {modelId}")
{
    public string ModelId { get; } = modelId;
}

/// <summary>No configured engine can serve the requested model.</summary>
public sealed class EngineNotAvailableException(string engine, string reason)
    : LocalGenException($"Engine '{engine}' is unavailable: {reason}")
{
    public string Engine { get; } = engine;
}

/// <summary>Loading weights failed — bad file, unsupported format, or not enough memory.</summary>
public sealed class ModelLoadException(string modelId, string reason, Exception? inner = null)
    : LocalGenException($"Failed to load model '{modelId}': {reason}", inner)
{
    public string ModelId { get; } = modelId;
}

/// <summary>An operation needed the network while offline mode was enabled.</summary>
public sealed class OfflineModeException(string operation)
    : LocalGenException($"'{operation}' requires network access, but LocalGen is running in offline mode.");

/// <summary>An API key exceeded one of its configured quotas.</summary>
/// <remarks>
/// Carries <see cref="RetryAfter"/> so the transport can set the header OpenAI clients already
/// honour when they back off, rather than leaving the caller to guess how long to wait.
/// </remarks>
public sealed class QuotaExceededException(string message, TimeSpan? retryAfter = null)
    : LocalGenException(message)
{
    public TimeSpan? RetryAfter { get; } = retryAfter;
}

/// <summary>A key addressed a model outside the set it is allowed to use.</summary>
public sealed class ModelForbiddenException(string modelId, string tenant)
    : LocalGenException($"Model '{modelId}' is not available to '{tenant}'.")
{
    public string ModelId { get; } = modelId;
}

/// <summary>A tool call was rejected by policy, e.g. a path outside the allowed roots.</summary>
public sealed class ToolPermissionException(string tool, string reason)
    : LocalGenException($"Tool '{tool}' refused the call: {reason}")
{
    public string Tool { get; } = tool;
}
