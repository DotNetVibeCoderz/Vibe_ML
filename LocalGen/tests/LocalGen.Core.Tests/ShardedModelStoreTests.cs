using FluentAssertions;
using LocalGen.Core.Configuration;
using LocalGen.Core.Models;
using LocalGen.Runtime.Downloads;
using LocalGen.Runtime.Models;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// A split model lives in several files but is one model. The store has to present it that way:
/// listed once, sized as a whole, and deleted completely.
/// </summary>
public sealed class ShardedModelStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "localgen-shard-tests", Guid.NewGuid().ToString("N"));

    private readonly LocalGenOptions _options;

    public ShardedModelStoreTests()
    {
        _options = new LocalGenOptions { DataDirectory = _root };
        LocalGenPaths.EnsureCreated(_options);
    }

    private string ModelsDirectory => _options.ModelsDirectory;

    private FileModelStore CreateStore() => new(
        Options.Create(_options),
        new UnusedDownloader(),
        NullLogger<FileModelStore>.Instance);

    /// <summary>Writes a file of the given size; GGUF metadata is unreadable, which the store tolerates.</summary>
    private void WriteWeights(string fileName, int bytes)
    {
        File.WriteAllBytes(Path.Combine(ModelsDirectory, fileName), new byte[bytes]);
    }

    [Fact]
    public async Task A_split_model_is_listed_once()
    {
        WriteWeights("big-Q4_K_M-00001-of-00003.gguf", 100);
        WriteWeights("big-Q4_K_M-00002-of-00003.gguf", 200);
        WriteWeights("big-Q4_K_M-00003-of-00003.gguf", 300);

        var models = await CreateStore().ListAsync();

        models.Should().HaveCount(1);
        models[0].Name.Should().Be("big-Q4_K_M-00001-of-00003");
    }

    [Fact]
    public async Task A_split_model_reports_the_size_of_every_shard()
    {
        WriteWeights("big-00001-of-00003.gguf", 100);
        WriteWeights("big-00002-of-00003.gguf", 200);
        WriteWeights("big-00003-of-00003.gguf", 300);

        var models = await CreateStore().ListAsync();

        // Not 100 — the first shard alone would badly understate the model.
        models[0].SizeBytes.Should().Be(600);
    }

    [Fact]
    public async Task Unsharded_models_beside_a_split_one_are_untouched()
    {
        WriteWeights("plain-Q8_0.gguf", 50);
        WriteWeights("big-00001-of-00002.gguf", 100);
        WriteWeights("big-00002-of-00002.gguf", 100);

        var models = await CreateStore().ListAsync();

        models.Should().HaveCount(2);
        models.Should().ContainSingle(m => m.SizeBytes == 50);
        models.Should().ContainSingle(m => m.SizeBytes == 200);
    }

    [Fact]
    public async Task Removing_a_split_model_deletes_all_of_its_shards()
    {
        WriteWeights("big-00001-of-00003.gguf", 100);
        WriteWeights("big-00002-of-00003.gguf", 100);
        WriteWeights("big-00003-of-00003.gguf", 100);

        var store = CreateStore();
        var model = (await store.ListAsync())[0];

        (await store.RemoveAsync(model.Id)).Should().BeTrue();

        // Orphaned shards would be gigabytes of unreferenced disk in practice.
        Directory.EnumerateFiles(ModelsDirectory, "*.gguf").Should().BeEmpty();
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
        catch (IOException)
        {
            // A temp directory that outlives the test run is not worth failing over.
        }
    }

    /// <summary>The store requires a downloader; these tests never pull.</summary>
    private sealed class UnusedDownloader : IModelDownloader
    {
        public ValueTask<DownloadResult> DownloadAsync(
            ModelReference reference,
            string destinationDirectory,
            IProgress<DownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();
    }
}
