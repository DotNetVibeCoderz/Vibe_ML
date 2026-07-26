using System.Net;
using BlazorML.Web.Services;

namespace BlazorML.Web.Tests;

/// <summary>
/// The trial-call service behind the Endpoint page's console.
/// <para>
/// The refusals are the interesting part. A console that forwards obviously broken input and then
/// reports the server's 400 back has made the user do the diagnosing; these tests pin the messages
/// that say what is wrong instead.
/// </para>
/// </summary>
public class EndpointTesterTests
{
    /// <summary>A factory whose client never reaches the network, so a forwarded request fails loudly.</summary>
    private sealed class NeverCalled : IHttpClientFactory
    {
        public bool WasCalled { get; private set; }

        public HttpClient CreateClient(string name)
        {
            WasCalled = true;
            return new HttpClient(new BlockingHandler());
        }

        private sealed class BlockingHandler : HttpMessageHandler
        {
            protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request,
                CancellationToken cancellationToken) =>
                throw new HttpRequestException("The test's handler refuses to make real calls.");
        }
    }

    /// <summary>Answers with a canned response and records what it was sent.</summary>
    private sealed class Recording(HttpStatusCode status, string body) : IHttpClientFactory
    {
        public HttpRequestMessage? Seen { get; private set; }
        public string? SeenBody { get; private set; }

        public HttpClient CreateClient(string name) => new(new Handler(this, status, body));

        private sealed class Handler(Recording owner, HttpStatusCode status, string body) : HttpMessageHandler
        {
            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request,
                CancellationToken cancellationToken)
            {
                owner.Seen = request;
                owner.SeenBody = request.Content is null
                    ? null
                    : await request.Content.ReadAsStringAsync(cancellationToken);

                return new HttpResponseMessage(status) { Content = new StringContent(body) };
            }
        }
    }

    private const string Payload = """{"rows":[{"usia":34}]}""";

    // ------------------------------------------------------------- refusals

    [Theory]
    [InlineData("", "Isi dulu")]
    [InlineData("   ", "Isi dulu")]
    [InlineData("bukan json", "belum valid")]
    [InlineData("[1,2,3]", "objek JSON")]
    [InlineData("""{"baris":[]}""", "\"rows\"")]
    [InlineData("""{"rows":{}}""", "\"rows\"")]
    [InlineData("""{"rows":[]}""", "masih kosong")]
    public async Task A_body_that_cannot_work_is_refused_before_anything_is_sent(string body, string expected)
    {
        var factory = new NeverCalled();
        var attempt = await new EndpointTester(factory).ScoreAsync("https://x.test", "s", "bml_k", body);

        Assert.False(attempt.Ok);
        Assert.NotNull(attempt.Failure);
        Assert.Contains(expected, attempt.Failure);

        // The point of checking locally is not making the round trip at all.
        Assert.False(factory.WasCalled);
    }

    // -------------------------------------------------------------- sending

    [Fact]
    public async Task The_key_goes_in_the_header_the_api_actually_reads()
    {
        var factory = new Recording(HttpStatusCode.OK, """{"predictions":[]}""");

        await new EndpointTester(factory).ScoreAsync("https://x.test/", "churn", "bml_rahasia", Payload);

        Assert.NotNull(factory.Seen);
        Assert.Equal("https://x.test/api/v1/score/churn", factory.Seen!.RequestUri!.ToString());
        Assert.Equal(HttpMethod.Post, factory.Seen.Method);
        Assert.Equal("bml_rahasia", factory.Seen.Headers.GetValues("X-Api-Key").Single());
        Assert.Equal(Payload, factory.SeenBody);
    }

    [Fact]
    public async Task A_successful_call_reports_the_status_and_indents_the_response()
    {
        var factory = new Recording(HttpStatusCode.OK, """{"model":"churn","version":1}""");

        var attempt = await new EndpointTester(factory).ScoreAsync("https://x.test", "s", "bml_k", Payload);

        Assert.True(attempt.Ok);
        Assert.Equal(200, attempt.Status);
        Assert.Null(attempt.Failure);
        Assert.Contains("\n", attempt.Body);
        Assert.Contains("\"model\": \"churn\"", attempt.Body);
    }

    [Fact]
    public async Task A_refusal_is_reported_as_a_response_not_as_a_failure()
    {
        // 401 is the API answering correctly. Showing it as a transport failure would send the
        // reader looking for a network problem that is not there.
        var factory = new Recording(HttpStatusCode.Unauthorized, """{"title":"Unauthorised"}""");

        var attempt = await new EndpointTester(factory).ScoreAsync("https://x.test", "s", "salah", Payload);

        Assert.False(attempt.Ok);
        Assert.Equal(401, attempt.Status);
        Assert.Null(attempt.Failure);
        Assert.Contains("Unauthorised", attempt.Body);
    }

    [Fact]
    public async Task A_response_that_is_not_json_is_shown_exactly_as_it_came_back()
    {
        // A proxy's HTML error page is precisely the thing worth seeing verbatim.
        const string html = "<html><body>502 Bad Gateway</body></html>";
        var factory = new Recording(HttpStatusCode.BadGateway, html);

        var attempt = await new EndpointTester(factory).ScoreAsync("https://x.test", "s", "bml_k", Payload);

        Assert.Equal(html, attempt.Body);
    }

    [Fact]
    public async Task An_unreachable_endpoint_says_so_rather_than_throwing()
    {
        var attempt = await new EndpointTester(new NeverCalled())
            .ScoreAsync("https://x.test", "s", "bml_k", Payload);

        Assert.False(attempt.Ok);
        Assert.Contains("Tidak bisa menghubungi", attempt.Failure);
    }
}
