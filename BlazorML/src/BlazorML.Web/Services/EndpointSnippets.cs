using System.Text;
using System.Text.Json;
using BlazorML.Core.Domain;

namespace BlazorML.Web.Services;

/// <summary>The languages the Endpoint page offers ready-made code for.</summary>
public enum SnippetLanguage
{
    Curl,
    CSharp,
    Python,
    Node
}

/// <summary>
/// Turns a published endpoint into code an integrator can paste and run.
/// <para>
/// The point of generating these rather than printing a generic template is the payload: the
/// column names and the example values come from the model's own input schema, so the snippet
/// works on the first try instead of after someone goes and finds out what the columns are called.
/// </para>
/// </summary>
public static class EndpointSnippets
{
    public static string Label(SnippetLanguage language) => language switch
    {
        SnippetLanguage.Curl => "cURL",
        SnippetLanguage.CSharp => "C#",
        SnippetLanguage.Python => "Python",
        SnippetLanguage.Node => "Node.js",
        _ => language.ToString()
    };

    /// <summary>Placeholder for the key, so a snippet never carries a real one out of the page.</summary>
    public const string KeyPlaceholder = "KUNCI_API_KAMU";

    // ------------------------------------------------------------------ payload

    /// <summary>
    /// The request body, pretty-printed, with one row of real values.
    /// <para>
    /// One row rather than several: the surrounding array already says the endpoint takes a batch,
    /// and a second row would double the height of every snippet to repeat what the brackets
    /// already show.
    /// </para>
    /// </summary>
    public static string SamplePayload(IReadOnlyList<ModelInputField> fields)
    {
        using var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteStartArray("rows");
            writer.WriteStartObject();

            foreach (var field in fields)
            {
                writer.WritePropertyName(field.Name);

                if (field.Example is { } example)
                {
                    example.WriteTo(writer);
                }
                else
                {
                    JsonSerializer.SerializeToElement(
                        ModelInputSchema.PlaceholderFor(field.Type)).WriteTo(writer);
                }
            }

            writer.WriteEndObject();
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    // ----------------------------------------------------------------- snippets

    public static string For(SnippetLanguage language, string baseUri, string slug,
        IReadOnlyList<ModelInputField> fields)
    {
        var url = $"{baseUri.TrimEnd('/')}/api/v1/score/{slug}";

        return language switch
        {
            SnippetLanguage.Curl => Curl(url, fields),
            SnippetLanguage.CSharp => CSharp(url, fields),
            SnippetLanguage.Python => Python(url, fields),
            SnippetLanguage.Node => Node(url, fields),
            _ => throw new ArgumentOutOfRangeException(nameof(language))
        };
    }

    private static string Curl(string url, IReadOnlyList<ModelInputField> fields)
    {
        // Compact JSON on one line: a shell-quoted multi-line body is where pasted curl commands
        // usually break. Single quotes inside are closed, escaped and reopened the POSIX way.
        var body = Compact(fields).Replace("'", "'\\''");

        return $"""
                curl -X POST {url} \
                  -H "X-Api-Key: {KeyPlaceholder}" \
                  -H "Content-Type: application/json" \
                  -d '{body}'
                """;
    }

    private static string CSharp(string url, IReadOnlyList<ModelInputField> fields)
    {
        var rows = string.Join(",\n", fields.Select(f =>
            $"                [{Quote(f.Name)}] = {Value(f, SnippetLanguage.CSharp)}"));

        // A Dictionary rather than an anonymous object: column names come from a spreadsheet, and
        // "harga rumah" or "col-1" is not a C# identifier.
        return $$"""
                 using System.Net.Http.Json;

                 using var http = new HttpClient();
                 http.DefaultRequestHeaders.Add("X-Api-Key", "{{KeyPlaceholder}}");

                 var payload = new
                 {
                     rows = new[]
                     {
                         new Dictionary<string, object?>
                         {
                 {{rows}}
                         }
                     }
                 };

                 var response = await http.PostAsJsonAsync("{{url}}", payload);
                 response.EnsureSuccessStatusCode();

                 Console.WriteLine(await response.Content.ReadAsStringAsync());
                 """;
    }

    private static string Python(string url, IReadOnlyList<ModelInputField> fields)
    {
        var rows = string.Join(",\n", fields.Select(f =>
            $"        {Quote(f.Name)}: {Value(f, SnippetLanguage.Python)}"));

        // Doubled $ so Python's own braces stay literal and only {{ }} interpolates.
        return $$"""
                 import requests

                 response = requests.post(
                     "{{url}}",
                     headers={"X-Api-Key": "{{KeyPlaceholder}}"},
                     json={"rows": [{
                 {{rows}}
                     }]},
                     timeout=30,
                 )
                 response.raise_for_status()

                 print(response.json()["predictions"])
                 """;
    }

    private static string Node(string url, IReadOnlyList<ModelInputField> fields)
    {
        var rows = string.Join(",\n", fields.Select(f =>
            $"      {Quote(f.Name)}: {Value(f, SnippetLanguage.Node)}"));

        return $$"""
                 const response = await fetch("{{url}}", {
                   method: "POST",
                   headers: {
                     "X-Api-Key": "{{KeyPlaceholder}}",
                     "Content-Type": "application/json",
                   },
                   body: JSON.stringify({ rows: [{
                 {{rows}}
                   }] }),
                 });

                 if (!response.ok) throw new Error(`Skor gagal: ${response.status}`);

                 const { predictions } = await response.json();
                 console.log(predictions);
                 """;
    }

    // ------------------------------------------------------------------ helpers

    private static string Compact(IReadOnlyList<ModelInputField> fields)
    {
        using var document = JsonDocument.Parse(SamplePayload(fields));
        return JsonSerializer.Serialize(document.RootElement);
    }

    /// <summary>A double-quoted string literal, valid in all four languages.</summary>
    private static string Quote(string value) =>
        '"' + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + '"';

    private static string Value(ModelInputField field, SnippetLanguage language)
    {
        var example = field.Example
                      ?? JsonSerializer.SerializeToElement(ModelInputSchema.PlaceholderFor(field.Type));

        return example.ValueKind switch
        {
            JsonValueKind.String => Quote(example.GetString() ?? string.Empty),
            JsonValueKind.Number => example.GetRawText(),
            JsonValueKind.True => language == SnippetLanguage.Python ? "True" : "true",
            JsonValueKind.False => language == SnippetLanguage.Python ? "False" : "false",
            JsonValueKind.Null => language switch
            {
                SnippetLanguage.Python => "None",
                SnippetLanguage.Node => "null",
                _ => "null"
            },
            // Anything structured is rendered as its JSON text; a feature column should never be
            // one, and quietly dropping it would be worse than showing it.
            _ => Quote(example.GetRawText())
        };
    }
}
