using System.IO.Compression;
using System.Net;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tests;

namespace MediaPipeNet.Core.Tests;

public class ExecutionProviderTests
{
    [Fact]
    public void Cpu_is_available_and_auto_resolves()
    {
        ExecutionProviderSelector.GetAvailableProviders().Should().Contain(ExecutionProvider.Cpu);
        ExecutionProviderSelector.GetRuntimeVersion().Should().NotBeNullOrEmpty();
        ExecutionProviderSelector.GetCandidates(ExecutionProvider.Auto, true).Should().EndWith(ExecutionProvider.Cpu);
        ExecutionProviderSelector.GetCandidates(ExecutionProvider.Cuda, true).Should().Equal(ExecutionProvider.Cuda, ExecutionProvider.Cpu);
        ExecutionProviderSelector.GetCandidates(ExecutionProvider.Cuda, false).Should().Equal(ExecutionProvider.Cuda);
        ExecutionProviderSelector.FromOrtName("DmlExecutionProvider").Should().Be(ExecutionProvider.DirectML);
        ExecutionProviderSelector.FromOrtName("Foo").Should().BeNull();
    }

    [Fact]
    public void Session_options_can_be_created()
    {
        using var so = ExecutionProviderSelector.CreateSessionOptions(ExecutionProvider.Cpu, new InferenceOptions { IntraOpThreads = 2, InterOpThreads = 1 });
        so.IntraOpNumThreads.Should().Be(2);
    }

    [Fact]
    public void Unavailable_provider_falls_back_to_cpu()
    {
        using var model = OnnxModel.Load(Path.Combine(TestPaths.Models, "canned_gesture_classifier.onnx"),
            new InferenceOptions { Provider = ExecutionProvider.Cuda, FallbackToCpu = true });
        model.Provider.Should().Be(ExecutionProvider.Cpu);
    }
}

public class OnnxModelTests
{
    [Fact]
    public void Loads_and_runs_with_pooled_contexts()
    {
        using var model = OnnxModel.Load(Path.Combine(TestPaths.Models, "face_detection_short_range.onnx"));
        model.Name.Should().Be("face_detection_short_range");
        model.Inputs.Should().ContainSingle().Which.Shape.Should().Equal(1, 128, 128, 3);
        model.Outputs.Should().HaveCount(2);
        model.GetOutputIndex("classificators").Should().Be(1);
        model.Inputs[0].ToString().Should().Contain("1x128x128x3");

        InferenceContext first;
        using (var ctx = model.RentContext())
        {
            first = ctx;
            ctx.GetInput(0).Fill(0.1f);
            ctx.GetInput("input").Length.Should().Be(128 * 128 * 3);
            ctx.Run();
            ctx.GetOutput(0).Length.Should().Be(896 * 16);
            ctx.GetOutput("classificators").Length.Should().Be(896);
            ctx.Model.Should().BeSameAs(model);
        }
        using var again = model.RentContext();
        again.Should().BeSameAs(first); // returned to and taken from the pool
    }

    [Fact]
    public void Loads_from_bytes_and_rejects_missing_files()
    {
        var bytes = File.ReadAllBytes(Path.Combine(TestPaths.Models, "canned_gesture_classifier.onnx"));
        using var model = OnnxModel.Load(bytes, "classifier");
        model.Outputs[0].ElementCount.Should().Be(8);
        var act = () => OnnxModel.Load("does-not-exist.onnx");
        act.Should().Throw<ModelNotFoundException>();
        var bad = () => model.GetInputIndex("nope");
        bad.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void Concurrent_runs_are_safe()
    {
        using var model = OnnxModel.Load(Path.Combine(TestPaths.Models, "canned_gesture_classifier.onnx"));
        Parallel.For(0, 32, i =>
        {
            using var ctx = model.RentContext();
            ctx.GetInput(0).Fill(i / 32f);
            ctx.Run();
            ctx.GetOutput(0).ToArray().Sum().Should().BeApproximately(1f, 1e-3f);
        });
    }
}

public class ModelStoreTests
{
    [Fact]
    public void Catalog_lists_every_shipped_model()
    {
        ModelCatalog.All.Should().HaveCount(13);
        ModelCatalog.All.Select(m => m.Id).Should().OnlyHaveUniqueItems();
        ModelCatalog.Find("palm_detection.onnx").Should().Be(ModelCatalog.PalmDetection);
        ModelCatalog.Find("unknown").Should().BeNull();
        ModelCatalog.PalmDetection.Attribution.Should().Contain("Google");
        foreach (var m in ModelCatalog.All)
        {
            var path = Path.Combine(TestPaths.Models, m.FileName);
            File.Exists(path).Should().BeTrue(m.FileName);
            new FileInfo(path).Length.Should().Be(m.SizeBytes, m.Id);
        }
    }

    [Fact]
    public async Task Resolves_and_verifies_from_directory()
    {
        var store = new ModelStore([new DirectoryModelProvider(TestPaths.Models)]);
        var path = await store.GetModelPathAsync(ModelCatalog.CannedGestureClassifier);
        path.Should().EndWith("canned_gesture_classifier.onnx");
        (await ModelStore.ComputeSha256Async(path)).Should().Be(ModelCatalog.CannedGestureClassifier.Sha256);
        store.GetModelPath(ModelCatalog.CannedGestureClassifier).Should().Be(path);
        store.FindLocal(ModelCatalog.CannedGestureClassifier).Should().Be(path);
        await store.EnsureModelsAsync([ModelCatalog.CannedGestureClassifier, ModelCatalog.GestureEmbedder]);
    }

    [Fact]
    public async Task Checksum_mismatch_and_missing_models_throw()
    {
        var dir = Directory.CreateTempSubdirectory("mpn-models").FullName;
        var fake = ModelCatalog.CannedGestureClassifier;
        await File.WriteAllBytesAsync(Path.Combine(dir, fake.FileName), new byte[fake.SizeBytes]);
        var store = new ModelStore([new DirectoryModelProvider(dir)]);
        var act = () => store.GetModelPathAsync(fake).AsTask();
        (await act.Should().ThrowAsync<ModelNotFoundException>()).Which.Message.Should().Contain("checksum");
        var missing = () => store.GetModelPathAsync(ModelCatalog.EfficientNetLite0).AsTask();
        await missing.Should().ThrowAsync<ModelNotFoundException>();
        new ModelStore([new DirectoryModelProvider(dir)], verifyChecksums: false).GetModelPath(fake).Should().StartWith(dir);
        Directory.Delete(dir, true);
    }

    [Fact]
    public async Task Nuget_provider_downloads_and_extracts_package()
    {
        var model = ModelCatalog.CannedGestureClassifier;
        var nupkg = new MemoryStream();
        using (var zip = new ZipArchive(nupkg, ZipArchiveMode.Create, leaveOpen: true))
        {
            zip.CreateEntryFromFile(Path.Combine(TestPaths.Models, model.FileName), "contentFiles/any/any/models/" + model.FileName);
            zip.CreateEntryFromFile(Path.Combine(TestPaths.Models, "gesture_embedder.onnx"), "contentFiles/any/any/models/gesture_embedder.onnx");
        }
        Uri? requested = null;
        using var http = new HttpClient(new StubHandler(req =>
        {
            requested = req.RequestUri;
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(nupkg.ToArray()) };
        }));
        var cache = Directory.CreateTempSubdirectory("mpn-cache").FullName;
        var provider = new NuGetModelProvider(cache, "9.9.9", new Uri("https://feed.test/v3/"), http);
        provider.IsRemote.Should().BeTrue();
        var reports = new List<ModelDownloadProgress>();
        var store = new ModelStore([provider]);
        var path = await store.GetModelPathAsync(model, new SyncProgress(reports.Add));
        requested!.ToString().Should().Be("https://feed.test/v3/gravicode.mediapipenet.models.hand/9.9.9/gravicode.mediapipenet.models.hand.9.9.9.nupkg");
        File.Exists(path).Should().BeTrue();
        File.Exists(Path.Combine(cache, "gesture_embedder.onnx")).Should().BeTrue();
        reports.Should().Contain(r => r.Stage == "downloading");
        Directory.Delete(cache, true);
    }

    [Fact]
    public async Task Http_and_embedded_providers()
    {
        var model = ModelCatalog.CannedGestureClassifier;
        var bytes = await File.ReadAllBytesAsync(Path.Combine(TestPaths.Models, model.FileName));
        using var http = new HttpClient(new StubHandler(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) }));
        var cache = Directory.CreateTempSubdirectory("mpn-http").FullName;
        var p = new HttpModelProvider(new Uri("https://mirror.test/models"), cache, http);
        (await p.TryGetModelPathAsync(model, null, default)).Should().NotBeNull();
        var embedded = new EmbeddedResourceModelProvider(typeof(ModelStoreTests).Assembly, cache);
        (await embedded.TryGetModelPathAsync(model, null, default)).Should().BeNull();
        new BundledModelProvider().Name.Should().Contain("models");
        Directory.Delete(cache, true);
        var progress = new ModelDownloadProgress(model, 50, 100, "downloading");
        progress.Fraction.Should().Be(0.5);
        ModelStore.CreateDefault(allowDownload: false).Providers.Should().NotContain(x => x.IsRemote);
    }

    private sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(respond(request));
    }

    private sealed class SyncProgress(Action<ModelDownloadProgress> report) : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value) => report(value);
    }
}
