using System.Net;
using System.Text;
using FluentAssertions;
using LocalGen.Runtime.Downloads;
using LocalGen.Runtime.Models;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Exercises the real downloader against a stubbed Hub. Sharded models in the wild are tens of
/// gigabytes, so the only practical way to prove every part is fetched — and that a shard published
/// inside a subdirectory lands flat next to its siblings, which is where llama.cpp looks — is to
/// serve the repository locally.
/// </summary>
public sealed class ShardedDownloadTests : IDisposable
{
    private readonly string _destination = Path.Combine(
        Path.GetTempPath(), "localgen-download-tests", Guid.NewGuid().ToString("N"));

    private readonly List<string> _requested = [];

    [Fact]
    public async Task Every_shard_is_downloaded_and_flattened_beside_its_siblings()
    {
        // Bartowski publishes split quantizations under a per-quantization folder, so the
        // repository path has a directory component the local file name must not inherit.
        var downloader = CreateDownloader(
            "Q8_0/model-00001-of-00003.gguf",
            "Q8_0/model-00002-of-00003.gguf",
            "Q8_0/model-00003-of-00003.gguf");

        var result = await downloader.DownloadAsync(
            ModelReference.Parse("huggingface:owner/repo/Q8_0/model-00001-of-00003.gguf"),
            _destination);

        Directory.EnumerateFiles(_destination, "*.gguf")
            .Select(Path.GetFileName)
            .Should().BeEquivalentTo(
                "model-00001-of-00003.gguf",
                "model-00002-of-00003.gguf",
                "model-00003-of-00003.gguf");

        // The manifest points at the first shard; llama.cpp resolves the rest from it.
        Path.GetFileName(result.FilePath).Should().Be("model-00001-of-00003.gguf");

        _requested.Should().Contain("/owner/repo/resolve/main/Q8_0/model-00003-of-00003.gguf");
    }

    [Fact]
    public async Task Naming_a_later_shard_still_fetches_the_whole_set()
    {
        var downloader = CreateDownloader(
            "model-00001-of-00002.gguf",
            "model-00002-of-00002.gguf");

        await downloader.DownloadAsync(
            ModelReference.Parse("huggingface:owner/repo/model-00002-of-00002.gguf"),
            _destination);

        Directory.EnumerateFiles(_destination, "*.gguf").Should().HaveCount(2);
    }

    [Fact]
    public async Task Shards_already_on_disk_are_not_downloaded_again()
    {
        Directory.CreateDirectory(_destination);
        await File.WriteAllTextAsync(
            Path.Combine(_destination, "model-00001-of-00002.gguf"), "already here");

        var downloader = CreateDownloader(
            "model-00001-of-00002.gguf",
            "model-00002-of-00002.gguf");

        await downloader.DownloadAsync(
            ModelReference.Parse("huggingface:owner/repo/model-00001-of-00002.gguf"),
            _destination);

        // Re-fetching a shard already present would cost hours on a real model.
        _requested.Should().NotContain(r => r.EndsWith("model-00001-of-00002.gguf"));
        _requested.Should().Contain(r => r.EndsWith("model-00002-of-00002.gguf"));
    }

    [Fact]
    public async Task An_unnamed_file_resolves_to_the_first_shard_of_a_split_model()
    {
        // The repository offers only a split model, so automatic selection has to accept one.
        var downloader = CreateDownloader(
            "model-Q4_K_M-00001-of-00002.gguf",
            "model-Q4_K_M-00002-of-00002.gguf");

        var result = await downloader.DownloadAsync(
            ModelReference.Parse("huggingface:owner/repo"),
            _destination);

        Path.GetFileName(result.FilePath).Should().Be("model-Q4_K_M-00001-of-00002.gguf");
        Directory.EnumerateFiles(_destination, "*.gguf").Should().HaveCount(2);
    }

    private HuggingFaceDownloader CreateDownloader(params string[] repositoryFiles) =>
        new(new StubFactory(new StubHandler(repositoryFiles, _requested)),
            NullLogger<HuggingFaceDownloader>.Instance);

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_destination))
            {
                Directory.Delete(_destination, recursive: true);
            }
        }
        catch (IOException)
        {
            // Leaving a temp directory behind is not worth failing a test over.
        }
    }

    private sealed class StubFactory(HttpMessageHandler handler) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name) =>
            new(handler, disposeHandler: false) { BaseAddress = null };
    }

    /// <summary>Answers the two Hub calls the downloader makes: the file listing and each blob.</summary>
    private sealed class StubHandler(IReadOnlyList<string> files, List<string> requested)
        : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            var path = request.RequestUri!.AbsolutePath;
            requested.Add(path);

            if (path.StartsWith("/api/models/", StringComparison.Ordinal))
            {
                var siblings = string.Join(",", files.Select(f => $$"""{"rfilename":"{{f}}"}"""));

                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(
                        $$"""{"id":"owner/repo","siblings":[{{siblings}}]}""",
                        Encoding.UTF8,
                        "application/json")
                });
            }

            // Stand-in weights; content does not matter, only that each shard arrives.
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(new byte[4096])
            });
        }
    }
}
