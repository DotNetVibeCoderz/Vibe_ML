using FluentAssertions;
using LocalGen.Core.Configuration;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Desktop.ViewModels;
using LocalGen.Runtime.Engines;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace LocalGen.Desktop.Tests;

/// <summary>
/// The multi-GPU section of the Engine screen, driven by a backend that reports two accelerators.
/// </summary>
/// <remarks>
/// This machine has one GPU, so the two-card path cannot be reached by probing real hardware — and
/// the screen's whole purpose is to be usable on the machine that does have two, where nobody will
/// be watching a debugger. A stub backend is the only way to see the panel appear, the summary
/// name both devices and the preview resolve a split before any weights are loaded.
/// </remarks>
public class EngineViewModelTests
{
    [Fact]
    public async Task Two_accelerators_reveal_the_split_settings()
    {
        var viewModel = await CreateAsync(acceleratorCount: 2);

        viewModel.MultiGpuAvailable.Should().BeTrue();
        viewModel.AcceleratorSummary.Should().Be("2 GPUs registered: CUDA0, CUDA1.");
    }

    [Fact]
    public async Task One_accelerator_hides_them_and_explains_why()
    {
        // The explanation is shown whether or not the controls are: why they are missing is the
        // question someone opens this screen with.
        var viewModel = await CreateAsync(acceleratorCount: 1);

        viewModel.MultiGpuAvailable.Should().BeFalse();
        viewModel.AcceleratorSummary.Should().Contain("needs a second one");
    }

    [Fact]
    public async Task No_accelerator_says_the_backend_registered_none()
    {
        var viewModel = await CreateAsync(acceleratorCount: 0);

        viewModel.MultiGpuAvailable.Should().BeFalse();
        viewModel.AcceleratorSummary.Should().Contain("registered no GPU");
    }

    [Fact]
    public async Task The_preview_resolves_a_split_against_the_devices_present()
    {
        var viewModel = await CreateAsync(acceleratorCount: 2);

        viewModel.TensorSplitText = "0.6, 0.4";

        viewModel.SplitPreviewIsWarning.Should().BeFalse();
        viewModel.SplitPreview.Should().Be("Weights: CUDA0 60% · CUDA1 40%");
    }

    [Fact]
    public async Task The_preview_warns_before_a_model_is_loaded_with_a_bad_split()
    {
        var viewModel = await CreateAsync(acceleratorCount: 2);

        viewModel.TensorSplitText = "0.4, 0.3, 0.3";

        viewModel.SplitPreviewIsWarning.Should().BeTrue();
        viewModel.SplitPreview.Should().Contain("names 3 devices but only 2 are visible");
    }

    [Fact]
    public async Task Text_that_is_not_a_list_of_numbers_is_called_out()
    {
        var viewModel = await CreateAsync(acceleratorCount: 2);

        viewModel.TensorSplitText = "60% / 40%";

        viewModel.SplitPreviewIsWarning.Should().BeTrue();
        viewModel.SplitPreview.Should().Contain("one weight per GPU");
    }

    [Fact]
    public async Task Apply_writes_the_split_into_the_live_options()
    {
        var options = NewOptions();
        var viewModel = await CreateAsync(acceleratorCount: 2, options);

        viewModel.TensorSplitText = "3, 1";
        viewModel.SplitMode = GpuSplitMode.Row;
        viewModel.MainGpu = 1;
        viewModel.ApplyCommand.Execute(null);

        options.Value.Engine.TensorSplit.Should().Equal(3f, 1f);
        options.Value.Engine.SplitMode.Should().Be(GpuSplitMode.Row);
        options.Value.Engine.MainGpu.Should().Be(1);
    }

    [Fact]
    public async Task Apply_leaves_an_unreadable_split_alone()
    {
        // Writing an empty split would look like the typed text had been accepted.
        var options = NewOptions();
        options.Value.Engine.TensorSplit = [0.5f, 0.5f];

        var viewModel = await CreateAsync(acceleratorCount: 2, options);

        viewModel.TensorSplitText = "not numbers";
        viewModel.ApplyCommand.Execute(null);

        options.Value.Engine.TensorSplit.Should().Equal(0.5f, 0.5f);
    }

    private static async Task<EngineViewModel> CreateAsync(
        int acceleratorCount,
        IOptions<LocalGenOptions>? options = null)
    {
        options ??= NewOptions();

        var registry = new EngineRegistry(
            [new StubEngine(acceleratorCount)],
            options,
            NullLogger<EngineRegistry>.Instance);

        var viewModel = new EngineViewModel(registry, options);
        await viewModel.InitializeAsync();

        return viewModel;
    }

    private static IOptions<LocalGenOptions> NewOptions() =>
        Options.Create(new LocalGenOptions());

    /// <summary>A backend that reports the accelerators the test needs and nothing else.</summary>
    private sealed class StubEngine(int acceleratorCount) : IInferenceEngine
    {
        public EngineDescriptor Descriptor { get; } = new()
        {
            Kind = EngineKind.LlamaSharp,
            DisplayName = "Stub",
            Description = "Reports a fixed set of accelerators.",
            Capabilities = new EngineCapabilities { SupportsMultiGpu = true }
        };

        public ValueTask<EngineAvailability> ProbeAsync(CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(new EngineAvailability
            {
                IsAvailable = true,
                AvailableDevices = [DeviceKind.Auto, DeviceKind.Cpu, DeviceKind.Cuda],
                Accelerators =
                [
                    .. Enumerable.Range(0, acceleratorCount)
                        .Select(i => new AcceleratorDevice { Index = i, Name = $"CUDA{i}" })
                ]
            });

        public bool CanServe(ModelDescriptor model) => true;

        public ValueTask<IModelSession> LoadAsync(
            ModelDescriptor model,
            ModelLoadOptions options,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The Engine screen never loads a model.");
    }
}
