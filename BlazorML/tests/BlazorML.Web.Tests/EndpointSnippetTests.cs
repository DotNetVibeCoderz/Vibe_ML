using System.Text.Json;
using BlazorML.Core.Domain;
using BlazorML.Web.Services;

namespace BlazorML.Web.Tests;

/// <summary>
/// The code the Endpoint page hands an integrator.
/// <para>
/// The claim these make is narrow and worth checking: the snippets are built from the model's own
/// input schema, so the column names and values in them are the real ones. A snippet that is
/// merely well formed but says <c>kolom1</c> has failed at the only job it has.
/// </para>
/// </summary>
public class EndpointSnippetTests
{
    private const string BaseUri = "https://studio.contoh.id/";
    private const string Slug = "prediksi-churn";

    private static ModelInputField Field(string name, ColumnDataType type, object? example) => new()
    {
        Name = name,
        Type = type,
        Example = example is null ? null : JsonSerializer.SerializeToElement(example)
    };

    private static readonly ModelInputField[] Churn =
    [
        Field("pendapatan", ColumnDataType.Numeric, 72.5),
        Field("usia", ColumnDataType.Numeric, 34),
        Field("kota", ColumnDataType.Categorical, "Bandung"),
        Field("langganan", ColumnDataType.Boolean, true)
    ];

    private static readonly SnippetLanguage[] All =
        [SnippetLanguage.Curl, SnippetLanguage.CSharp, SnippetLanguage.Python, SnippetLanguage.Node];

    public static TheoryData<SnippetLanguage> Languages()
    {
        var data = new TheoryData<SnippetLanguage>();

        foreach (var language in All)
        {
            data.Add(language);
        }

        return data;
    }

    // ------------------------------------------------------------------ payload

    [Fact]
    public void The_sample_payload_uses_the_model_own_columns_and_values()
    {
        var payload = EndpointSnippets.SamplePayload(Churn);

        using var document = JsonDocument.Parse(payload);
        var row = document.RootElement.GetProperty("rows")[0];

        Assert.Equal(4, row.EnumerateObject().Count());
        Assert.Equal(72.5, row.GetProperty("pendapatan").GetDouble());
        Assert.Equal(34, row.GetProperty("usia").GetInt32());
        Assert.Equal("Bandung", row.GetProperty("kota").GetString());
        Assert.True(row.GetProperty("langganan").GetBoolean());
    }

    [Fact]
    public void A_column_with_no_recorded_example_still_gets_a_value_of_the_right_kind()
    {
        var payload = EndpointSnippets.SamplePayload([
            Field("jumlah", ColumnDataType.Numeric, null),
            Field("nama", ColumnDataType.Text, null),
            Field("aktif", ColumnDataType.Boolean, null)
        ]);

        using var document = JsonDocument.Parse(payload);
        var row = document.RootElement.GetProperty("rows")[0];

        // Typed placeholders, so the body still parses as the right shape after someone edits it.
        Assert.Equal(JsonValueKind.Number, row.GetProperty("jumlah").ValueKind);
        Assert.Equal(JsonValueKind.String, row.GetProperty("nama").ValueKind);
        Assert.Equal(JsonValueKind.False, row.GetProperty("aktif").ValueKind);
    }

    [Fact]
    public void A_model_with_no_recorded_schema_still_produces_a_body_that_parses()
    {
        // Models registered before the schema carried anything land here. An empty rows object is
        // wrong for the model but right as JSON, which is what the console needs to start from.
        var payload = EndpointSnippets.SamplePayload([]);

        using var document = JsonDocument.Parse(payload);

        Assert.Equal(1, document.RootElement.GetProperty("rows").GetArrayLength());
    }

    // ----------------------------------------------------------------- snippets

    [Theory]
    [MemberData(nameof(Languages))]
    public void Every_snippet_carries_the_url_the_key_header_and_all_the_columns(SnippetLanguage language)
    {
        var snippet = EndpointSnippets.For(language, BaseUri, Slug, Churn);

        Assert.Contains($"https://studio.contoh.id/api/v1/score/{Slug}", snippet);
        Assert.Contains("X-Api-Key", snippet);
        Assert.Contains(EndpointSnippets.KeyPlaceholder, snippet);

        foreach (var field in Churn)
        {
            Assert.Contains(field.Name, snippet);
        }
    }

    [Theory]
    [MemberData(nameof(Languages))]
    public void No_snippet_ever_contains_a_real_key(SnippetLanguage language)
    {
        // Nothing in the generator is given a key, and nothing should be able to introduce one:
        // these snippets are meant to be pasted into chats and issue trackers.
        var snippet = EndpointSnippets.For(language, BaseUri, Slug, Churn);

        Assert.DoesNotContain("bml_", snippet);
    }

    [Theory]
    [MemberData(nameof(Languages))]
    public void A_trailing_slash_on_the_base_address_does_not_double_up(SnippetLanguage language)
    {
        var withSlash = EndpointSnippets.For(language, "https://studio.contoh.id/", Slug, Churn);
        var without = EndpointSnippets.For(language, "https://studio.contoh.id", Slug, Churn);

        Assert.DoesNotContain("id//api", withSlash);
        Assert.Equal(without, withSlash);
    }

    [Fact]
    public void The_curl_body_is_valid_json_and_matches_the_sample_payload()
    {
        var snippet = EndpointSnippets.For(SnippetLanguage.Curl, BaseUri, Slug, Churn);

        var start = snippet.IndexOf("-d '", StringComparison.Ordinal) + 4;
        var body = snippet[start..snippet.LastIndexOf('\'')];

        using var sent = JsonDocument.Parse(body);
        using var expected = JsonDocument.Parse(EndpointSnippets.SamplePayload(Churn));

        // Compact vs indented, so compare the values rather than the text.
        Assert.Equal(
            JsonSerializer.Serialize(expected.RootElement),
            JsonSerializer.Serialize(sent.RootElement));
    }

    [Fact]
    public void Python_gets_python_booleans_and_the_others_get_json_ones()
    {
        var python = EndpointSnippets.For(SnippetLanguage.Python, BaseUri, Slug, Churn);

        Assert.Contains("\"langganan\": True", python);

        foreach (var language in (SnippetLanguage[])[SnippetLanguage.CSharp, SnippetLanguage.Node])
        {
            Assert.Contains("true", EndpointSnippets.For(language, BaseUri, Slug, Churn));
        }
    }

    /// <summary>
    /// Column names come out of spreadsheets, so they are not identifiers and they are not always
    /// quote-free. A name that closes its own string turns a snippet into something that will not
    /// even parse, which is worse than no snippet at all.
    /// </summary>
    [Theory]
    [MemberData(nameof(Languages))]
    public void An_awkward_column_name_does_not_break_the_snippet(SnippetLanguage language)
    {
        var awkward = new[]
        {
            Field("harga rumah", ColumnDataType.Numeric, 250.0),
            Field("kolom \"aneh\"", ColumnDataType.Text, "nilai \"dikutip\""),
            Field("jalur\\balik", ColumnDataType.Text, "a\\b")
        };

        var snippet = EndpointSnippets.For(language, BaseUri, Slug, awkward);

        Assert.Contains("harga rumah", snippet);

        // Every quote in the body is either escaped or a delimiter — never a bare one that would
        // end a string early.
        Assert.DoesNotContain("\"kolom \"aneh\"\"", snippet);
    }

    /// <summary>
    /// The body is wrapped in single quotes for the shell, so an apostrophe in the data is the
    /// classic way a copied curl command breaks. What matters is that the quoting cannot be
    /// closed early — not which mechanism achieves it, since the JSON writer already escapes an
    /// apostrophe to <c>'</c> and the explicit escape is the backstop behind that.
    /// </summary>
    [Fact]
    public void The_curl_body_survives_a_value_containing_a_single_quote()
    {
        var fields = new[]
        {
            Field("kota", ColumnDataType.Categorical, "Ma'rifat"),
            Field("pemilik's", ColumnDataType.Text, "O'Brien")
        };

        var snippet = EndpointSnippets.For(SnippetLanguage.Curl, BaseUri, Slug, fields);

        var start = snippet.IndexOf("-d '", StringComparison.Ordinal) + 4;
        var body = snippet[start..snippet.LastIndexOf('\'')];

        // Nothing inside the quoted body may be a bare apostrophe.
        Assert.DoesNotContain('\'', body);

        // And it still means what it should once the shell has handed it over.
        using var parsed = JsonDocument.Parse(body);
        var row = parsed.RootElement.GetProperty("rows")[0];

        Assert.Equal("Ma'rifat", row.GetProperty("kota").GetString());
        Assert.Equal("O'Brien", row.GetProperty("pemilik's").GetString());
    }

    [Theory]
    [MemberData(nameof(Languages))]
    public void Every_language_has_a_label_that_is_not_the_enum_name(SnippetLanguage language)
    {
        var label = EndpointSnippets.Label(language);

        Assert.False(string.IsNullOrWhiteSpace(label));
        Assert.Contains(label, (string[])["cURL", "C#", "Python", "Node.js"]);
    }
}
