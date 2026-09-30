using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>Settings for one generation.</summary>
/// <param name="Steps">How many denoising steps to take.</param>
/// <param name="GuidanceScale">
/// How strongly to push the sample towards the prompt. 1 disables guidance; 7.5 is the usual value.
/// </param>
/// <param name="Width">Image width in pixels. Must be a multiple of the VAE's scale factor, 8 for Stable Diffusion.</param>
/// <param name="Height">Image height in pixels. Same rule as <paramref name="Width"/>.</param>
/// <param name="Seed">
/// Seed for the initial latent, drawn as <c>np.random.RandomState(seed).randn</c> - so the same seed
/// gives the same starting noise as diffusers' ONNX pipelines in Python.
/// </param>
/// <param name="NegativePrompt">What to steer away from. Empty is the usual unconditional prompt.</param>
public readonly record struct GenerationOptions(
    int Steps = 25,
    double GuidanceScale = 7.5,
    int Width = 512,
    int Height = 512,
    int Seed = 42,
    string NegativePrompt = "")
{
    /// <summary>The settings a first run should use.</summary>
    public static GenerationOptions Default => new(25, 7.5, 512, 512, 42, "");

    /// <summary>Whether classifier-free guidance is in play.</summary>
    /// <remarks>
    /// Guidance means running the model twice per step, once conditioned and once not, so a scale
    /// of 1 is not merely a weak setting - it halves the work.
    /// </remarks>
    public bool UsesGuidance => GuidanceScale > 1.0;
}

/// <summary>Which sampler a pipeline builds from the repository's scheduler configuration.</summary>
public enum Sampler
{
    /// <summary>
    /// The one the repository's scheduler configuration names: Euler for <c>EulerDiscreteScheduler</c>,
    /// DDPM for <c>DDPMScheduler</c>, and DDIM for DDIM and for PNDM, which this does not implement
    /// and for which DDIM is the usual substitute.
    /// </summary>
    Auto,

    /// <summary>DDIM: deterministic, the usual default.</summary>
    Ddim,

    /// <summary>Euler: sigma-parameterised, often sharper at few steps.</summary>
    Euler,

    /// <summary>DDPM: the original process, for hundreds of steps.</summary>
    Ddpm,
}

/// <summary>
/// Stable Diffusion over ONNX: text to image, image to image and inpainting.
/// </summary>
/// <remarks>
/// <para>
/// This is the blueprint's <c>DiffusionPipeline.Generate(prompt)</c>. The networks - text encoder,
/// UNet and the VAE's two halves - run through ONNX Runtime, because a diffusion run evaluates the
/// UNet tens of times and the managed <see cref="double"/> path in this stack is the wrong tool for
/// that by two orders of magnitude.
/// </para>
/// <para>
/// Everything the numbers depend on is read from the repository rather than assumed: the VAE's
/// scale factor and latent scaling from <c>vae_decoder/config.json</c>, the UNet's input channels
/// from <c>unet/config.json</c>, the sampler from <c>scheduler/scheduler_config.json</c>, and the
/// prompt tokenizer as CLIP's. The steps follow diffusers' ONNX pipelines line for line, and the
/// initial noise comes from NumPy's generator, so the same seed and settings reproduce diffusers'
/// image.
/// </para>
/// <para>
/// <b>It needs an ONNX export laid out as diffusers lays one out:</b> <c>text_encoder/model.onnx</c>,
/// <c>unet/model.onnx</c>, <c>vae_decoder/model.onnx</c>, and for image to image or inpainting
/// <c>vae_encoder/model.onnx</c>. A repository holding only PyTorch weights is refused on load -
/// converting one is a job for <c>optimum-cli export onnx</c>.
/// </para>
/// </remarks>
public sealed class DiffusionPipeline : IDisposable
{
    private readonly string _directory;
    private readonly ExecutionTarget _target;
    private readonly OnnxSession _vaeDecoder;
    private readonly List<DiffusionLora.Adapter> _loras = [];
    private OnnxSession _textEncoder;
    private OnnxSession _unet;
    private OnnxSession? _vaeEncoder;
    private bool _disposed;

    private DiffusionPipeline(
        string repoId, string directory, ExecutionTarget target, HfTokenizer tokenizer,
        OnnxSession textEncoder, OnnxSession unet, OnnxSession vaeDecoder, SchedulerConfig schedulerConfig, IScheduler scheduler)
    {
        RepoId = repoId;
        _directory = directory;
        _target = target;
        Tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        SchedulerConfig = schedulerConfig;
        Scheduler = scheduler;

        var vae = ReadConfig(Path.Combine(directory, "vae_decoder", "config.json"));
        var blocks = vae is { } v && v.TryGetProperty("block_out_channels", out var channels) ? channels.GetArrayLength() : 4;
        VaeScaleFactor = 1 << Math.Max(0, blocks - 1);
        LatentChannels = vae is { } l && l.TryGetProperty("latent_channels", out var latent) ? latent.GetInt32() : 4;
        LatentScaling = vae is { } s && s.TryGetProperty("scaling_factor", out var scaling) ? scaling.GetDouble() : 0.18215;

        var unetConfig = ReadConfig(Path.Combine(directory, "unet", "config.json"));
        UnetInputChannels = unetConfig is { } u && u.TryGetProperty("in_channels", out var inputs)
            ? inputs.GetInt32()
            : unet.Inputs.FirstOrDefault(i => i.Name == "sample").Shape is { Length: 4 } shape && shape[1] > 0 ? shape[1] : LatentChannels;
    }

    /// <summary>The repository this pipeline came from.</summary>
    public string RepoId { get; }

    /// <summary>The prompt tokenizer.</summary>
    public HfTokenizer Tokenizer { get; }

    /// <summary>The sampler the reverse process walks with.</summary>
    public IScheduler Scheduler { get; set; }

    /// <summary>The repository's scheduler configuration.</summary>
    public SchedulerConfig SchedulerConfig { get; }

    /// <summary>How many times smaller the latent grid is than the image, from the VAE's config.</summary>
    public int VaeScaleFactor { get; }

    /// <summary>Latent channels, from the VAE's config.</summary>
    public int LatentChannels { get; }

    /// <summary>The factor latents are scaled by between the VAE and the UNet.</summary>
    public double LatentScaling { get; }

    /// <summary>The UNet's input channels: 4 for text to image, 9 for an inpainting model.</summary>
    public int UnetInputChannels { get; }

    /// <summary>How many tokens the text encoder takes.</summary>
    public int MaxPromptTokens { get; init; } = 77;

    /// <summary>
    /// Whether prompts are read with the <c>(word)</c> / <c>[word]</c> / <c>(word:1.4)</c> emphasis
    /// syntax. Off by default, so a plain prompt is encoded exactly as diffusers encodes it.
    /// </summary>
    /// <remarks>
    /// Turned on, a prompt is parsed with <see cref="PromptWeights"/>, each piece is tokenized on its
    /// own, the text encoder's output is multiplied by each token's weight, and the result is
    /// rescaled to its original mean - AUTOMATIC1111's scheme. It applies to the negative prompt too.
    /// </remarks>
    public bool PromptWeighting { get; set; }

    /// <summary>Whether a VAE encoder is present, which image to image and inpainting need.</summary>
    public bool SupportsImageToImage => File.Exists(Path.Combine(_directory, "vae_encoder", "model.onnx"));

    /// <summary>The file layout text to image needs inside a repository.</summary>
    public static IReadOnlyList<string> RequiredFiles { get; } =
    [
        "text_encoder/model.onnx",
        "unet/model.onnx",
        "vae_decoder/model.onnx",
    ];

    // ------------------------------------------------------------------ loading

    /// <summary>Downloads an ONNX Stable Diffusion repository and builds a pipeline.</summary>
    /// <param name="repoId">A model id whose repository holds the ONNX exports.</param>
    /// <param name="scheduler">A sampler to use as it is; when null one is built from the repository's configuration.</param>
    /// <param name="target">Which execution provider to run on.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives download progress.</param>
    /// <param name="sampler">Which sampler to build when <paramref name="scheduler"/> is null.</param>
    /// <exception cref="HubException">The repository has no ONNX export.</exception>
    public static DiffusionPipeline FromPretrained(
        string repoId,
        IScheduler? scheduler = null,
        ExecutionTarget target = ExecutionTarget.Auto,
        string revision = "main",
        IProgress<TransferProgress>? progress = null,
        Sampler sampler = Sampler.Auto)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var info = Hub.ModelInfo(repoId, revision);
        var available = info.Files.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);

        var missing = RequiredFiles.Where(f => !available.Contains(f)).ToList();
        if (missing.Count > 0)
        {
            throw new HubException(
                $"'{repoId}' is missing {string.Join(", ", missing)}. GraviDiffusers runs the ONNX "
                + "export of a diffusion model, not the PyTorch weights. Convert one with "
                + "`optimum-cli export onnx --model <id> <out>`, or point at a repository that "
                + "already publishes an onnx layout.") { RepoId = repoId };
        }

        var directory = Hub.Shared.SnapshotAsync(
            repoId,
            RepoKind.Model,
            revision,
            allowPatterns: ["*/model.onnx", "*/model.onnx_data", "*/*.onnx_data", "*/weights.pb", "tokenizer/*", "*.json", "*.txt"],
            ignorePatterns: null,
            progress).GetAwaiter().GetResult();

        return Open(directory, repoId, scheduler, target, sampler);
    }

    /// <summary>Builds a pipeline from an ONNX export on disk, laid out as diffusers lays one out.</summary>
    /// <param name="directory">The export's root, holding <c>unet/</c>, <c>text_encoder/</c> and the rest.</param>
    /// <param name="scheduler">A sampler to use as it is; when null one is built from the configuration.</param>
    /// <param name="target">Which execution provider to run on.</param>
    /// <param name="sampler">Which sampler to build when <paramref name="scheduler"/> is null.</param>
    public static DiffusionPipeline FromDirectory(
        string directory, IScheduler? scheduler = null, ExecutionTarget target = ExecutionTarget.Auto, Sampler sampler = Sampler.Auto)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        var missing = RequiredFiles.Where(f => !File.Exists(Path.Combine(directory, f))).ToList();
        if (missing.Count > 0)
        {
            throw new FileNotFoundException(
                $"'{directory}' is missing {string.Join(", ", missing)}; it is not an ONNX diffusion export.");
        }

        return Open(directory, Path.GetFileName(Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar)), scheduler, target, sampler);
    }

    private static DiffusionPipeline Open(string directory, string repoId, IScheduler? scheduler, ExecutionTarget target, Sampler sampler)
    {
        var schedulerPath = Path.Combine(directory, "scheduler", "scheduler_config.json");
        var config = File.Exists(schedulerPath) ? SchedulerConfig.Load(schedulerPath) : SchedulerConfig.StableDiffusion;

        if (sampler == Sampler.Auto)
        {
            sampler = config.ClassName switch
            {
                "EulerDiscreteScheduler" => Sampler.Euler,
                "DDPMScheduler" => Sampler.Ddpm,
                _ => Sampler.Ddim,
            };
        }

        scheduler ??= sampler switch
        {
            Sampler.Euler => new EulerScheduler(config),
            Sampler.Ddpm => new DdpmScheduler(config),
            _ => new DdimScheduler(config),
        };

        return new DiffusionPipeline(
            repoId,
            directory,
            target,
            LoadTokenizer(directory),
            OnnxSession.Open(Path.Combine(directory, "text_encoder", "model.onnx"), target),
            OnnxSession.Open(Path.Combine(directory, "unet", "model.onnx"), target),
            OnnxSession.Open(Path.Combine(directory, "vae_decoder", "model.onnx"), target),
            config,
            scheduler);
    }

    // ------------------------------------------------------------------ text to image

    /// <summary>Generates an image from a prompt.</summary>
    /// <param name="prompt">What to draw.</param>
    /// <param name="options">Steps, guidance, size and seed.</param>
    /// <param name="onStep">Called after each denoising step, with the index and the total.</param>
    /// <returns>The generated image.</returns>
    public Image<Rgb24> Generate(string prompt, GenerationOptions? options = null, Action<int, int>? onStep = null)
    {
        var settings = options ?? GenerationOptions.Default;
        CheckSize(settings);

        var shape = new[] { 1, LatentChannels, settings.Height / VaeScaleFactor, settings.Width / VaeScaleFactor };
        return Generate(prompt, InitialLatents(shape, settings.Seed), settings, onStep);
    }

    /// <summary>Generates an image from a prompt, starting from given latents.</summary>
    /// <param name="prompt">What to draw.</param>
    /// <param name="latents">
    /// The starting noise, <c>[1, channels, height / scale, width / scale]</c>, before the scheduler's
    /// initial scaling - what diffusers' <c>latents=</c> argument takes.
    /// </param>
    /// <param name="options">Steps and guidance; size and seed are ignored.</param>
    /// <param name="onStep">Called after each denoising step, with the index and the total.</param>
    public Image<Rgb24> Generate(string prompt, NdArray latents, GenerationOptions? options = null, Action<int, int>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(latents);

        if (UnetInputChannels != LatentChannels)
        {
            throw new InvalidOperationException(
                $"This UNet takes {UnetInputChannels} input channels - it is an inpainting model. Use Inpaint.");
        }

        var settings = options ?? GenerationOptions.Default;
        var embeddings = PromptEmbeddings(prompt, settings);

        Scheduler.SetTimesteps(settings.Steps);
        var sample = Scale(latents, Scheduler.InitialNoiseScale);

        sample = Denoise(sample, embeddings, settings, 0, extra: null, onStep);
        return Decode(sample);
    }

    // ------------------------------------------------------------------ image to image

    /// <summary>Redraws an image to follow a prompt, keeping its composition.</summary>
    /// <param name="prompt">What the result should show.</param>
    /// <param name="image">The starting image. Its sides are trimmed to multiples of 64, as diffusers does.</param>
    /// <param name="strength">
    /// How far to move from the original: 0 returns it, 1 ignores it. The run starts at that
    /// fraction of the schedule.
    /// </param>
    /// <param name="options">Steps, guidance, seed and negative prompt.</param>
    /// <param name="onStep">Called after each denoising step.</param>
    /// <remarks>
    /// <para>
    /// Follows diffusers' <c>OnnxStableDiffusionImg2ImgPipeline</c>: the image is encoded, noised to
    /// the timestep <c>strength</c> of the way through the schedule, and denoised from there.
    /// </para>
    /// <para>
    /// The encoder uses the latent distribution's <b>mean</b>. The usual export samples inside the
    /// graph with a <c>RandomNormalLike</c> that has no seed, so it returns a different latent on
    /// every call - in Python as well - and no seed could reproduce a result. The export is copied
    /// once, beside the original, with that node's scale set to the smallest normal float: the noise
    /// it then draws is about 1e-38, which vanishes when added to any latent value. Not zero, because
    /// the libstdc++ that ONNX Runtime is built against on Linux asserts on a zero deviation and aborts
    /// the process.
    /// </para>
    /// </remarks>
    public Image<Rgb24> ImageToImage(
        string prompt, Image<Rgb24> image, double strength = 0.8, GenerationOptions? options = null, Action<int, int>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(image);
        if (strength is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(strength), "Strength is between 0 and 1.");

        var settings = options ?? GenerationOptions.Default;
        var embeddings = PromptEmbeddings(prompt, settings);

        var original = EncodeImage(Preprocess(image, 64));

        Scheduler.SetTimesteps(settings.Steps);
        var offset = Scheduler.StepsOffset;
        var initial = Math.Min((int)(settings.Steps * strength) + offset, settings.Steps);

        // diffusers draws this noise first and only, from the same generator.
        var noise = new NdArray(Float32(new NumpyRandom((uint)settings.Seed).Normal(original.Size)), original.Shape.ToArray());
        var start = settings.Steps - initial;
        var sample = start < Scheduler.Timesteps.Count ? Scheduler.AddNoise(original, noise, start) : original;

        // diffusers starts the loop one step later than it noised, by steps_offset. Kept as it is,
        // so the same settings give the same picture.
        var first = Math.Max(settings.Steps - initial + offset, 0);
        sample = Denoise(sample, embeddings, settings, first, extra: null, onStep);
        return Decode(sample);
    }

    // ------------------------------------------------------------------ inpainting

    /// <summary>Repaints the masked part of an image to follow a prompt.</summary>
    /// <param name="prompt">What the masked area should show.</param>
    /// <param name="image">The image.</param>
    /// <param name="mask">White where to repaint, black where to keep; the same size as the image.</param>
    /// <param name="options">Steps, guidance, size, seed and negative prompt.</param>
    /// <param name="onStep">Called after each denoising step.</param>
    /// <remarks>
    /// <para>
    /// With an inpainting UNet - nine input channels - this is diffusers'
    /// <c>OnnxStableDiffusionInpaintPipeline</c>: the UNet sees the latents, the mask at latent
    /// resolution, and the latents of the image with the masked area blacked out.
    /// </para>
    /// <para>
    /// With an ordinary four-channel UNet it blends instead: after every step, the area outside the
    /// mask is replaced by the original image noised to that step, so only the masked area is
    /// generated. Softer at the edge than a trained inpainting model, and available for any
    /// checkpoint.
    /// </para>
    /// </remarks>
    public Image<Rgb24> Inpaint(
        string prompt, Image<Rgb24> image, Image<Rgb24> mask, GenerationOptions? options = null, Action<int, int>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(mask);

        var settings = (options ?? GenerationOptions.Default) with { Width = image.Width, Height = image.Height };
        CheckSize(settings);

        var embeddings = PromptEmbeddings(prompt, settings);
        var latentHeight = settings.Height / VaeScaleFactor;
        var latentWidth = settings.Width / VaeScaleFactor;

        var random = new NumpyRandom((uint)settings.Seed);
        var latents = new NdArray(Float32(random.Normal(LatentChannels * latentHeight * latentWidth)), 1, LatentChannels, latentHeight, latentWidth);

        var (pixels, keep) = PrepareMaskedImage(image, mask);
        var latentMask = LatentMask(mask, latentWidth, latentHeight);

        Scheduler.SetTimesteps(settings.Steps);
        var sample = Scale(latents, Scheduler.InitialNoiseScale);

        if (UnetInputChannels == LatentChannels + 1 + LatentChannels)
        {
            var maskedLatents = EncodeImage(Masked(pixels, keep));
            sample = Denoise(sample, embeddings, settings, 0, (latentMask, maskedLatents), onStep);
            return Decode(sample);
        }

        if (UnetInputChannels != LatentChannels)
        {
            throw new NotSupportedException(
                $"The UNet takes {UnetInputChannels} input channels; inpainting runs on 4 (blended) or 9 (an inpainting model).");
        }

        // Blended inpainting on an ordinary UNet.
        var original = EncodeImage(pixels);
        var noise = new NdArray(Float32(random.Normal(original.Size)), original.Shape.ToArray());
        sample = Denoise(sample, embeddings, settings, 0, extra: null, (step, total) =>
        {
            onStep?.Invoke(step, total);
        }, blend: (original, noise, latentMask));

        return Decode(sample);
    }

    // ------------------------------------------------------------------ LoRA

    /// <summary>Merges a LoRA adapter into the UNet and text encoder.</summary>
    /// <param name="source">A <c>.safetensors</c> file, a directory holding one, or a Hub repository id.</param>
    /// <param name="scale">How strongly to apply it; 1 is the adapter as trained.</param>
    /// <returns>Which layers it changed, and any module it named that the graphs do not have.</returns>
    /// <exception cref="InvalidOperationException">No module of the adapter matched a layer.</exception>
    /// <remarks>
    /// Adapters accumulate: loading a second one adds its update to the first's. The sessions are
    /// rebuilt with the merged weights substituted, so a large UNet takes seconds to reload.
    /// </remarks>
    public LoraReport LoadLora(string source, double scale = 1.0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(source);

        var adapters = DiffusionLora.Read(DiffusionLora.Resolve(source), scale);
        if (adapters.Count == 0) throw new InvalidDataException($"'{source}' holds no LoRA pairs this reader recognises.");

        _loras.AddRange(adapters);
        var report = Rebuild();

        if (report.UnetLayers + report.TextEncoderLayers == 0)
        {
            _loras.RemoveRange(_loras.Count - adapters.Count, adapters.Count);
            Rebuild();
            throw new InvalidOperationException(
                $"None of the {adapters.Count} adapter modules matched a layer of this model, for example "
                + $"'{report.Unmatched.FirstOrDefault()}'. The LoRA was probably trained for a different base model.");
        }

        return report;
    }

    /// <summary>Removes every loaded LoRA, restoring the base model.</summary>
    public void UnloadLoras()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _loras.Clear();
        Rebuild();
    }

    private LoraReport Rebuild()
    {
        var unmatched = new List<string>();
        var unetPath = Path.Combine(_directory, "unet", "model.onnx");
        var textPath = Path.Combine(_directory, "text_encoder", "model.onnx");

        var unetOverrides = DiffusionLora.Merge(OnnxModelFile.Read(unetPath), _loras.Where(a => a.Component == "unet"), unmatched, out var unetLayers);
        var textOverrides = DiffusionLora.Merge(OnnxModelFile.Read(textPath), _loras.Where(a => a.Component == "text_encoder"), unmatched, out var textLayers);

        _unet.Dispose();
        _textEncoder.Dispose();
        _unet = OnnxSession.Open(unetPath, _target, unetOverrides);
        _textEncoder = OnnxSession.Open(textPath, _target, textOverrides);

        return new LoraReport(unetLayers, textLayers, unmatched);
    }

    // ------------------------------------------------------------------ the loop

    /// <summary>
    /// Runs the reverse process from <paramref name="first"/> to the end, with classifier-free
    /// guidance batched as diffusers batches it: unconditional first, conditional second.
    /// </summary>
    private NdArray Denoise(
        NdArray sample, NdArray embeddings, GenerationOptions settings, int first,
        (NdArray Mask, NdArray MaskedLatents)? extra, Action<int, int>? onStep,
        (NdArray Original, NdArray Noise, NdArray Mask)? blend = null)
    {
        var total = Scheduler.Timesteps.Count;
        var guided = settings.UsesGuidance;
        var integerTimestep = _unet.Inputs.FirstOrDefault(i => i.Name == "timestep").ElementType?.Contains("Int", StringComparison.OrdinalIgnoreCase) == true;

        for (var step = first; step < total; step++)
        {
            var input = guided ? Stack(sample, sample) : sample;
            input = Scheduler.ScaleInput(input, step);

            if (extra is { } e)
            {
                input = Concatenate(input, guided ? Stack(e.Mask, e.Mask) : e.Mask, guided ? Stack(e.MaskedLatents, e.MaskedLatents) : e.MaskedLatents);
            }

            // numpy casts a float timestep to an integer input by truncation.
            var timestep = Scheduler.ModelTimestep(step);
            if (integerTimestep) timestep = Math.Truncate(timestep);

            var prediction = PredictNoise(input, timestep, embeddings);

            if (guided)
            {
                var half = prediction.Size / 2;
                var noise = NdArray.Zeros([1, .. prediction.Shape.ToArray()[1..]]);
                for (var i = 0; i < half; i++)
                {
                    var unconditional = prediction.At(i);
                    noise.SetAt(i, unconditional + settings.GuidanceScale * (prediction.At(half + i) - unconditional));
                }

                prediction = noise;
            }

            sample = Scheduler.Step(prediction, step, sample);

            if (blend is { } b)
            {
                // Outside the mask, the original noised to the level of the next step.
                var known = step + 1 < total ? Scheduler.AddNoise(b.Original, b.Noise, step + 1) : b.Original;
                sample = Blend(sample, known, b.Mask);
            }

            onStep?.Invoke(step + 1, total);
        }

        return sample;
    }

    /// <summary>Runs the UNet for one timestep.</summary>
    private NdArray PredictNoise(NdArray latents, double timestep, NdArray textStates)
    {
        var feeds = new Dictionary<string, NdArray>(StringComparer.Ordinal)
        {
            ["sample"] = latents,
            ["timestep"] = new NdArray([timestep], 1),
            ["encoder_hidden_states"] = textStates,
        };

        // Exports disagree on the timestep input's name and rank, so only the names the graph
        // actually declares are fed.
        var declared = _unet.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var key in feeds.Keys.ToList())
        {
            if (!declared.Contains(key)) feeds.Remove(key);
        }

        return _unet.Run(feeds).Values.First();
    }

    // ------------------------------------------------------------------ prompts

    /// <summary>The text states the UNet cross-attends to: <c>[uncond; cond]</c> with guidance, <c>cond</c> without.</summary>
    private NdArray PromptEmbeddings(string prompt, GenerationOptions settings)
    {
        var conditional = EncodePrompt(prompt);
        return settings.UsesGuidance ? Stack(EncodePrompt(settings.NegativePrompt), conditional) : conditional;
    }

    /// <summary>Runs the text encoder over a prompt.</summary>
    private NdArray EncodePrompt(string prompt)
    {
        if (!PromptWeighting || !PromptWeights.HasEmphasis(prompt))
        {
            var encoding = Tokenizer.EncodeBatch([prompt], BatchOptions.Exactly(MaxPromptTokens));
            return RunTextEncoder(encoding.Ids);
        }

        // Each weighted piece tokenized on its own, then assembled into BOS, pieces, EOS, padding.
        var bounds = Tokenizer.Encode("").Ids;
        var ids = new List<int>();
        var weights = new List<double>();

        foreach (var (text, weight) in PromptWeights.Parse(prompt))
        {
            foreach (var id in Tokenizer.Encode(text, addSpecialTokens: false).Ids)
            {
                ids.Add(id);
                weights.Add(weight);
            }
        }

        var room = MaxPromptTokens - 2;
        var sequence = new List<int> { bounds[0] };
        var multipliers = new List<double> { 1.0 };
        sequence.AddRange(ids.Take(room));
        multipliers.AddRange(weights.Take(room));
        sequence.Add(bounds[^1]);
        multipliers.Add(1.0);

        var pad = Tokenizer.PadId >= 0 ? Tokenizer.PadId : bounds[^1];
        while (sequence.Count < MaxPromptTokens)
        {
            sequence.Add(pad);
            multipliers.Add(1.0);
        }

        var states = RunTextEncoder(new NdArray([.. sequence.Select(i => (double)i)], 1, sequence.Count));
        var width = states.Size / MaxPromptTokens;
        var values = states.ToArray();

        var before = values.Average();
        for (var t = 0; t < MaxPromptTokens; t++)
        {
            for (var d = 0; d < width; d++) values[t * width + d] *= multipliers[t];
        }

        var after = values.Average();
        var correction = after == 0 ? 1 : before / after;
        for (var i = 0; i < values.Length; i++) values[i] *= correction;

        return new NdArray(values, states.Shape.ToArray());
    }

    private NdArray RunTextEncoder(NdArray ids)
    {
        var feeds = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var input in _textEncoder.Inputs)
        {
            feeds[input.Name] = input.Name == "attention_mask" ? NdArray.Ones(ids.Shape.ToArray()) : ids;
        }

        var outputs = _textEncoder.Run(feeds);

        // CLIP's text encoder returns the per-token states first and a pooled vector second; the
        // UNet cross-attends to the per-token states, so the first output is the one wanted.
        return outputs.TryGetValue("last_hidden_state", out var hidden) ? hidden : outputs.Values.First();
    }

    // ------------------------------------------------------------------ images and latents

    /// <summary>Decodes a latent grid into pixels.</summary>
    private Image<Rgb24> Decode(NdArray latents)
        => ToImage(_vaeDecoder.Run(new Dictionary<string, NdArray>
        {
            [_vaeDecoder.Inputs[0].Name] = Scale(latents, 1 / LatentScaling),
        }).Values.First());

    /// <summary>The VAE encoder's latents for pixels in <c>[-1, 1]</c>, scaled for the UNet.</summary>
    private NdArray EncodeImage(NdArray pixels)
    {
        _vaeEncoder ??= OpenMeanEncoder();
        var latents = _vaeEncoder.Run(new Dictionary<string, NdArray> { [_vaeEncoder.Inputs[0].Name] = pixels }).Values.First();
        return Scale(latents, LatentScaling);
    }

    /// <summary>The smallest normal float32: a deviation too small to move any latent, and not zero.</summary>
    private const float SmallestNormal = 1.17549435E-38f;

    /// <summary>The VAE encoder with any in-graph sampling turned into the distribution's mean.</summary>
    private OnnxSession OpenMeanEncoder()
    {
        var path = Path.Combine(_directory, "vae_encoder", "model.onnx");
        if (!File.Exists(path))
        {
            throw new NotSupportedException(
                $"'{RepoId}' has no vae_encoder/model.onnx, which image to image and inpainting need. Export the encoder too.");
        }

        var model = OnnxModelFile.Read(path);
        if (!model.Nodes.Any(n => n.OpType is "RandomNormalLike" or "RandomNormal")) return OnnxSession.Open(path, _target);

        // A new name for the patched copy: an earlier release wrote model.mean.onnx with a scale of 0,
        // which aborts ONNX Runtime on Linux, and it must not be picked up again.
        var mean = Path.Combine(_directory, "vae_encoder", "model.deterministic.onnx");
        if (!File.Exists(mean) || File.GetLastWriteTimeUtc(mean) < File.GetLastWriteTimeUtc(path))
        {
            var (bytes, _) = OnnxModelFile.SetFloatAttribute(path, "RandomNormalLike", "scale", SmallestNormal);
            File.WriteAllBytes(mean, bytes);
        }

        return OnnxSession.Open(mean, _target);
    }

    /// <summary>
    /// An image as <c>[1, 3, H, W]</c> in <c>[-1, 1]</c>, its sides trimmed to multiples of
    /// <paramref name="multiple"/> - diffusers' <c>preprocess</c>.
    /// </summary>
    internal static NdArray Preprocess(Image<Rgb24> image, int multiple)
    {
        var width = image.Width - image.Width % multiple;
        var height = image.Height - image.Height % multiple;
        if (width == 0 || height == 0) throw new ArgumentException($"The image is smaller than {multiple} pixels on a side.", nameof(image));

        using var sized = width == image.Width && height == image.Height
            ? image.Clone()
            : image.Clone(c => c.Resize(width, height, KnownResamplers.Lanczos3));

        var values = new double[3 * width * height];
        var plane = width * height;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixel = sized[x, y];
                var at = y * width + x;

                // float32 arithmetic, as numpy's astype(np.float32) / 255.0 and 2.0 * image - 1.0.
                values[at] = 2f * (pixel.R / 255f) - 1f;
                values[plane + at] = 2f * (pixel.G / 255f) - 1f;
                values[2 * plane + at] = 2f * (pixel.B / 255f) - 1f;
            }
        }

        return new NdArray(values, 1, 3, height, width);
    }

    /// <summary>
    /// The image as <c>[-1, 1]</c> pixels (<c>/ 127.5 - 1</c>) and which pixels the mask keeps, as
    /// diffusers' inpainting <c>prepare_mask_and_masked_image</c> computes them.
    /// </summary>
    private static (NdArray Pixels, bool[] Keep) PrepareMaskedImage(Image<Rgb24> image, Image<Rgb24> mask)
    {
        if (mask.Width != image.Width || mask.Height != image.Height)
        {
            throw new ArgumentException($"The mask is {mask.Width}x{mask.Height} but the image is {image.Width}x{image.Height}.", nameof(mask));
        }

        var width = image.Width;
        var height = image.Height;
        var plane = width * height;
        var values = new double[3 * plane];
        var keep = new bool[plane];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                var at = y * width + x;
                values[at] = pixel.R / 127.5f - 1f;
                values[plane + at] = pixel.G / 127.5f - 1f;
                values[2 * plane + at] = pixel.B / 127.5f - 1f;
                keep[at] = Luminance(mask[x, y]) < 127.5;
            }
        }

        return (new NdArray(values, 1, 3, height, width), keep);
    }

    private static NdArray Masked(NdArray pixels, bool[] keep)
    {
        var values = pixels.ToArray();
        var plane = keep.Length;
        for (var c = 0; c < 3; c++)
        {
            for (var i = 0; i < plane; i++)
            {
                if (!keep[i]) values[c * plane + i] = 0;
            }
        }

        return new NdArray(values, pixels.Shape.ToArray());
    }

    /// <summary>The mask at latent resolution, 1 to repaint: nearest-neighbour, thresholded at one half.</summary>
    private static NdArray LatentMask(Image<Rgb24> mask, int width, int height)
    {
        var values = new double[width * height];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                // PIL's NEAREST samples the source at the centre of each target pixel.
                var sx = Math.Min(mask.Width - 1, (int)((x + 0.5) * mask.Width / width));
                var sy = Math.Min(mask.Height - 1, (int)((y + 0.5) * mask.Height / height));
                values[y * width + x] = (float)(Luminance(mask[sx, sy]) / 255.0) >= 0.5f ? 1 : 0;
            }
        }

        return new NdArray(values, 1, 1, height, width);
    }

    /// <summary>PIL's <c>convert("L")</c>: <c>(299 R + 587 G + 114 B + 500) / 1000</c>, in integers.</summary>
    private static int Luminance(Rgb24 pixel) => (pixel.R * 299 + pixel.G * 587 + pixel.B * 114 + 500) / 1000;

    private static NdArray Blend(NdArray generated, NdArray known, NdArray mask)
    {
        var shape = generated.Shape.ToArray();
        var plane = shape[2] * shape[3];
        var result = NdArray.Zeros(shape);
        for (var i = 0; i < generated.Size; i++)
        {
            var m = mask.At(i % plane);
            result.SetAt(i, m * generated.At(i) + (1 - m) * known.At(i));
        }

        return result;
    }

    /// <summary>Turns the VAE's <c>[1, 3, H, W]</c> output in <c>[-1, 1]</c> into an image.</summary>
    internal static Image<Rgb24> ToImage(NdArray tensor)
    {
        var shape = tensor.Shape.ToArray();
        if (shape.Length != 4 || shape[1] != 3)
        {
            throw new InvalidDataException(
                $"Expected a [1, 3, H, W] image tensor, got [{string.Join("x", shape)}].");
        }

        var height = shape[2];
        var width = shape[3];
        var image = new Image<Rgb24>(width, height);
        var plane = height * width;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var offset = y * width + x;
                image[x, y] = new Rgb24(
                    ToByte(tensor.At(offset)),
                    ToByte(tensor.At(plane + offset)),
                    ToByte(tensor.At(2 * plane + offset)));
            }
        }

        return image;
    }

    /// <summary>
    /// A VAE output value in <c>[-1, 1]</c> as a byte: <c>round(clip(v / 2 + 0.5, 0, 1) * 255)</c>,
    /// rounding half to even as numpy does. Truncating instead darkens every pixel by half a level.
    /// </summary>
    private static byte ToByte(double value)
    {
        var unit = (float)Math.Clamp((float)value / 2f + 0.5f, 0f, 1f);
        return (byte)Math.Round(unit * 255f, MidpointRounding.ToEven);
    }

    /// <summary>Draws initial latent noise as <c>np.random.RandomState(seed).randn(*shape)</c>, in float32.</summary>
    internal static NdArray InitialLatents(int[] shape, int seed)
    {
        var count = shape.Aggregate(1, (a, b) => a * b);
        return new NdArray(Float32(new NumpyRandom((uint)seed).Normal(count)), shape);
    }

    /// <summary>Draws a <c>[1, 4, height, width]</c> latent as <see cref="InitialLatents(int[], int)"/>, scaled.</summary>
    internal static NdArray InitialLatents(int height, int width, int seed, double scale)
        => Scale(InitialLatents([1, 4, height, width], seed), scale);

    private static double[] Float32(double[] values)
    {
        for (var i = 0; i < values.Length; i++) values[i] = (float)values[i];
        return values;
    }

    private static NdArray Scale(NdArray values, double factor)
    {
        var result = NdArray.Zeros(values.Shape.ToArray());
        for (var i = 0; i < values.Size; i++) result.SetAt(i, values.At(i) * factor);
        return result;
    }

    /// <summary>Two tensors stacked along the batch axis.</summary>
    private static NdArray Stack(NdArray first, NdArray second)
    {
        var shape = first.Shape.ToArray();
        shape[0] += second.Shape[0];
        return new NdArray([.. first.ToArray(), .. second.ToArray()], shape);
    }

    /// <summary>Tensors joined along the channel axis, batch by batch.</summary>
    private static NdArray Concatenate(params NdArray[] parts)
    {
        var batch = parts[0].Shape[0];
        var height = parts[0].Shape[2];
        var width = parts[0].Shape[3];
        var channels = parts.Sum(p => p.Shape[1]);
        var plane = height * width;

        var values = new double[batch * channels * plane];
        var cursor = 0;
        for (var b = 0; b < batch; b++)
        {
            foreach (var part in parts)
            {
                var size = part.Shape[1] * plane;
                var source = part.ToArray();
                Array.Copy(source, b * size, values, cursor, size);
                cursor += size;
            }
        }

        return new NdArray(values, batch, channels, height, width);
    }

    private void CheckSize(GenerationOptions settings)
    {
        if (settings.Width % VaeScaleFactor != 0 || settings.Height % VaeScaleFactor != 0)
        {
            throw new ArgumentException(
                $"Width and height must be multiples of {VaeScaleFactor}; the latent grid is that much "
                + $"smaller than the image. Got {settings.Width}x{settings.Height}.", nameof(settings));
        }
    }

    private static JsonElement? ReadConfig(string path)
    {
        if (!File.Exists(path)) return null;
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    /// <summary>The prompt tokenizer: CLIP's, from <c>tokenizer/</c>, with the configured padding.</summary>
    /// <remarks>The padding token is the model's choice: <c>&lt;|endoftext|&gt;</c> for SD 1.x, <c>!</c> for SD 2.x.</remarks>
    private static HfTokenizer LoadTokenizer(string directory) => HfTokenizer.FromDirectory(Path.Combine(directory, "tokenizer"));

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _textEncoder.Dispose();
        _unet.Dispose();
        _vaeDecoder.Dispose();
        _vaeEncoder?.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"DiffusionPipeline({RepoId}, {Scheduler})";
}
