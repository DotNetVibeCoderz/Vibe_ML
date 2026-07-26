using System.Text.Json;
using System.Text.Json.Serialization;

namespace BlazorML.Core.Domain;

/// <summary>
/// One feature a published model expects, with enough detail to write a request that actually
/// works: its name, what kind of value it takes, and a real example drawn from the data the model
/// was fitted on.
/// <para>
/// The example matters more than it looks. A sample payload of <c>{"kolom1": 123}</c> is a shape,
/// not a request — an integrator has to go and find the real column names before anything runs.
/// Carrying one row of real values turns the generated code on the Endpoint page into something
/// that can be pasted and run.
/// </para>
/// </summary>
public sealed record ModelInputField
{
    [JsonPropertyName("name")] public required string Name { get; init; }

    /// <summary>Absent on models registered before the schema carried types.</summary>
    [JsonPropertyName("type")] public ColumnDataType? Type { get; init; }

    /// <summary>A value seen during training, or null when none was recorded.</summary>
    [JsonPropertyName("example")] public JsonElement? Example { get; init; }
}

/// <summary>
/// Reads <see cref="TrainedModel.InputSchemaJson"/>.
/// <para>
/// Tolerant by necessity: models registered before this type existed stored a bare
/// <c>[{"name":"x"}]</c>, and they must keep working. A caller gets the names either way and the
/// types and examples only when they are there.
/// </para>
/// </summary>
public static class ModelInputSchema
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new JsonStringEnumConverter() }
    };

    public static IReadOnlyList<ModelInputField> Parse(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return [];
        }

        try
        {
            return JsonSerializer.Deserialize<List<ModelInputField>>(json, Options) ?? [];
        }
        catch (JsonException)
        {
            // A schema we cannot read is not a reason to fail the page it appears on.
            return [];
        }
    }

    public static string Serialise(IEnumerable<ModelInputField> fields) =>
        JsonSerializer.Serialize(fields, Options);

    /// <summary>
    /// A value to stand in for a field with no recorded example. Typed rather than uniform, so the
    /// payload still parses as the right shape once someone edits the numbers.
    /// </summary>
    public static object PlaceholderFor(ColumnDataType? type) => type switch
    {
        ColumnDataType.Numeric => 0,
        ColumnDataType.Boolean => false,
        ColumnDataType.DateTime => DateTime.UtcNow.ToString("yyyy-MM-dd"),
        _ => "teks"
    };
}
