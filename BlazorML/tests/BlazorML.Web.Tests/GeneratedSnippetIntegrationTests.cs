using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using BlazorML.Core.Domain;
using BlazorML.Web.Endpoints;
using BlazorML.Web.Services;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Web.Tests;

/// <summary>
/// The end of the promise the Endpoint page makes: paste this and it runs.
/// <para>
/// The unit tests check the snippets are built from the model's schema. This one closes the loop —
/// it takes the payload the page would show for a real published model and sends it to the real
/// running API. Everything in between can be individually correct and still not add up; only the
/// round trip proves it does.
/// </para>
/// </summary>
public class GeneratedSnippetIntegrationTests(ScoringApiFixture fixture) : IClassFixture<ScoringApiFixture>
{
    private IReadOnlyList<ModelInputField> Fields()
    {
        using var scope = fixture.Services.CreateScope();
        var endpoints = scope.ServiceProvider.GetRequiredService<EndpointService>();

        var endpoint = endpoints.ListAsync().GetAwaiter().GetResult()
            .First(e => e.Slug == fixture.Slug);

        return ModelInputSchema.Parse(endpoint.Model!.InputSchemaJson);
    }

    [Fact]
    public void The_published_model_records_the_columns_and_a_real_example_for_each()
    {
        var fields = Fields();

        Assert.NotEmpty(fields);
        Assert.All(fields, f => Assert.False(string.IsNullOrWhiteSpace(f.Name)));

        // Without examples the generated payload is a shape rather than a working request.
        Assert.All(fields, f => Assert.NotNull(f.Example));
    }

    [Fact]
    public async Task The_payload_the_page_shows_scores_successfully_against_the_running_api()
    {
        var payload = EndpointSnippets.SamplePayload(Fields());

        var client = fixture.CreateClient();
        client.DefaultRequestHeaders.Add(ScoringApi.KeyHeader, fixture.ApiKey);

        var response = await client.PostAsync($"/api/v1/score/{fixture.Slug}",
            new StringContent(payload, Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(1, body.GetProperty("predictions").GetArrayLength());
        Assert.True(body.GetProperty("predictions")[0].TryGetProperty("PredictedLabel", out _));
    }

    /// <summary>
    /// The same body, taken out of the generated cURL command rather than from the payload helper,
    /// because that is the one an integrator actually copies.
    /// </summary>
    [Fact]
    public async Task The_body_inside_the_generated_curl_command_scores_too()
    {
        var snippet = EndpointSnippets.For(SnippetLanguage.Curl, "https://ignored.test",
            fixture.Slug, Fields());

        var start = snippet.IndexOf("-d '", StringComparison.Ordinal) + 4;
        var body = snippet[start..snippet.LastIndexOf('\'')];

        var client = fixture.CreateClient();
        client.DefaultRequestHeaders.Add(ScoringApi.KeyHeader, fixture.ApiKey);

        var response = await client.PostAsync($"/api/v1/score/{fixture.Slug}",
            new StringContent(body, Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task The_trial_service_reaches_the_same_endpoint_the_snippets_point_at()
    {
        // EndpointTester goes over HTTP, so it needs a factory whose client speaks to the test
        // host rather than the network.
        var tester = new EndpointTester(new FixtureClients(fixture));

        var attempt = await tester.ScoreAsync("http://localhost", fixture.Slug, fixture.ApiKey,
            EndpointSnippets.SamplePayload(Fields()));

        Assert.True(attempt.Ok, $"Trial call returned {attempt.Status}: {attempt.Failure ?? attempt.Body}");
        Assert.Equal(200, attempt.Status);
        Assert.Contains("predictions", attempt.Body);
    }

    [Fact]
    public async Task A_wrong_key_comes_back_as_a_401_the_console_can_show()
    {
        var tester = new EndpointTester(new FixtureClients(fixture));

        var attempt = await tester.ScoreAsync("http://localhost", fixture.Slug, "bml_bukan_kunci",
            EndpointSnippets.SamplePayload(Fields()));

        Assert.False(attempt.Ok);
        Assert.Equal(401, attempt.Status);

        // A refusal from the API, not a transport failure: the console renders the two differently
        // and sending someone to look for a network problem would waste their time.
        Assert.Null(attempt.Failure);
    }

    private sealed class FixtureClients(ScoringApiFixture fixture) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) => fixture.CreateClient();
    }
}
