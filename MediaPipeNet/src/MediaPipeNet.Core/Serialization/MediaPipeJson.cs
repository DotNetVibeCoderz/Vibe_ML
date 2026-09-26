using System.Text.Json;
using System.Text.Json.Serialization;

namespace MediaPipeNet.Serialization;

/// <summary>
/// JSON settings shared by every MediaPipe.NET result type: camelCase names, enums as strings,
/// null members omitted and non-finite floats allowed.
/// </summary>
public static class MediaPipeJson
{
    /// <summary>Compact options (no indentation).</summary>
    public static JsonSerializerOptions Options { get; } = Create(indented: false);

    /// <summary>Indented options for human-readable output.</summary>
    public static JsonSerializerOptions IndentedOptions { get; } = Create(indented: true);

    /// <summary>Creates a fresh copy of the MediaPipe.NET serializer options.</summary>
    public static JsonSerializerOptions Create(bool indented)
    {
        var o = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = indented,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        };
        o.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        o.MakeReadOnly(populateMissingResolver: true);
        return o;
    }

    /// <summary>Serializes a result to JSON.</summary>
    public static string Serialize<T>(T value, bool indented = false) =>
        JsonSerializer.Serialize(value, indented ? IndentedOptions : Options);

    /// <summary>Deserializes a result from JSON.</summary>
    public static T? Deserialize<T>(string json) => JsonSerializer.Deserialize<T>(json, Options);
}
