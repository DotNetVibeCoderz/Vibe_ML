using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using BlazorML.Web.Endpoints;

namespace BlazorML.Web.Tests;

/// <summary>
/// Exercises the published scoring API against the real application. These cover the path a
/// customer's integration takes, which is the one place a break is invisible until someone
/// else's system stops working.
/// </summary>
public class ScoringApiTests(ScoringApiFixture fixture) : IClassFixture<ScoringApiFixture>
{
    private HttpClient Client(string? apiKey = null)
    {
        var client = fixture.CreateClient();

        if (apiKey is not null)
        {
            client.DefaultRequestHeaders.Add(ScoringApi.KeyHeader, apiKey);
        }

        return client;
    }

    private static object Rows(params object[] rows) => new { rows };

    private static object SampleRow(int income = 90) =>
        new { pendapatan = income, usia = 40 };

    // ------------------------------------------------------------ happy path

    [Fact]
    public async Task Scoring_with_a_valid_key_returns_a_prediction_per_row()
    {
        var response = await Client(fixture.ApiKey)
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", Rows(SampleRow(90), SampleRow(10)));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal("uji-churn", body.GetProperty("model").GetString());
        Assert.Equal(1, body.GetProperty("version").GetInt32());
        Assert.Equal("BinaryClassification", body.GetProperty("task").GetString());
        Assert.Equal(2, body.GetProperty("predictions").GetArrayLength());
    }

    [Fact]
    public async Task The_prediction_actually_follows_the_signal_the_model_was_trained_on()
    {
        // A response shaped correctly but full of constant predictions would pass every other
        // test here; this is the one that says the model is really being applied.
        var response = await Client(fixture.ApiKey)
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", Rows(SampleRow(95), SampleRow(5)));

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        var predictions = body.GetProperty("predictions");

        var high = predictions[0].GetProperty("Probability").GetDouble();
        var low = predictions[1].GetProperty("Probability").GetDouble();

        Assert.True(high > low,
            $"High income scored {high:F3} and low income {low:F3}; the model is not being applied.");
    }

    [Fact]
    public async Task Schema_describes_the_columns_the_model_expects()
    {
        var response = await Client(fixture.ApiKey).GetAsync($"/api/v1/schema/{fixture.Slug}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal("beli", body.GetProperty("label").GetString());
        Assert.Equal("BinaryClassification", body.GetProperty("task").GetString());
        Assert.True(body.GetProperty("inputs").GetArrayLength() > 0);
    }

    [Fact]
    public async Task Listing_endpoints_shows_the_live_one_and_hides_the_stopped_one()
    {
        var response = await Client().GetAsync("/api/v1/endpoints");
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        var slugs = body.EnumerateArray().Select(e => e.GetProperty("slug").GetString()).ToList();

        Assert.Contains(fixture.Slug, slugs);
        Assert.DoesNotContain(fixture.StoppedSlug, slugs);
    }

    // -------------------------------------------------------------- refusals

    [Fact]
    public async Task Scoring_without_a_key_is_refused()
    {
        var response = await Client().PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", Rows(SampleRow()));

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task Scoring_with_the_wrong_key_is_refused()
    {
        var response = await Client("bml_bukan_kunci_yang_benar")
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", Rows(SampleRow()));

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task A_stopped_endpoint_refuses_even_a_valid_key()
    {
        var response = await Client(fixture.ApiKey)
            .PostAsJsonAsync($"/api/v1/score/{fixture.StoppedSlug}", Rows(SampleRow()));

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    /// <summary>
    /// A caller without a valid key must not be able to tell an endpoint that exists from one
    /// that does not — otherwise the API becomes a directory of what is deployed.
    /// </summary>
    [Fact]
    public async Task An_unknown_slug_is_indistinguishable_from_a_wrong_key()
    {
        var unknown = await Client("bml_apa_saja")
            .PostAsJsonAsync("/api/v1/score/endpoint-yang-tidak-ada", Rows(SampleRow()));

        var wrongKey = await Client("bml_apa_saja")
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", Rows(SampleRow()));

        Assert.Equal(wrongKey.StatusCode, unknown.StatusCode);
        Assert.Equal(
            await wrongKey.Content.ReadAsStringAsync(),
            await unknown.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task An_empty_row_list_is_a_bad_request_not_an_empty_success()
    {
        var response = await Client(fixture.ApiKey)
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}", new { rows = Array.Empty<object>() });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("at least one row", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Rows_that_do_not_match_the_schema_report_why_rather_than_returning_a_500()
    {
        var response = await Client(fixture.ApiKey)
            .PostAsJsonAsync($"/api/v1/score/{fixture.Slug}",
                Rows(new { kolom_yang_tidak_dikenal = "abc" }));

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        Assert.Contains("Scoring failed", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Schema_is_refused_without_a_key_too()
    {
        var response = await Client().GetAsync($"/api/v1/schema/{fixture.Slug}");

        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }
}
