using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>Settings for one generation.</summary>
/// <param name="Steps">How many denoising steps to take.</param>
/// <param name="GuidanceScale">
/// How strongly to push the sample towards the prompt. 1 disables guidance; 7.5 is the usual value.
/// </param>
/// <param name="Width">Image width in pixels. Must be a multiple of 8.</param>
/// <param name="Height">Image height in pixels. Must be a multiple of 8.</param>
/// <param name="Seed">Seed for the initial latent, so a result can be reproduced.</param>
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

/// <summary>
/// Text to image generation with a Stable Diffusion checkpoint exported to ONNX.
/// </summary>
/// <remarks>
/// <para>
/// This is the blueprint's <c>DiffusionPipeline.Generate(prompt)</c>. The three networks - text
/// encoder, UNet and VAE decoder - run through ONNX Runtime, because a diffusion run evaluates the
/// UNet tens of times and the managed <see cref="double"/> path in this stack is the wrong tool for
/// that by two orders of magnitude.
/// </para>
/// <para>
/// <b>It needs a repository laid out the way the ONNX exports are:</b> <c>text_encoder/model.onnx</c>,
/// <c>unet/model.onnx</c> and <c>vae_decoder/model.onnx</c>. A repository holding only the PyTorch
/// weights will not work here and says so on load rather than failing later - converting one is a
/// job for <c>optimum-cli export onnx</c>.
/// </para>
/// <para>
/// The latent space is eight times smaller than the image in each dimension, which is where the
/// <c>Width / 8</c> arithmetic throughout comes from, and why both dimensions must be multiples of 8.
/// </para>
/// </remarks>
public sealed class DiffusionPipeline : IDisposable
{
    private const double LatentScaling = 0.18215;
    private const int LatentChannels = 4;
    private const int VaeScaleFactor = 8;

    private readonly OnnxSession _textEncoder;
    private readonly OnnxSession _unet;
    private readonly OnnxSession _vaeDecoder;
    private bool _disposed;

    private DiffusionPipeline(
        string repoId,
        HfTokenizer tokenizer,
        OnnxSession textEncoder,
        OnnxSession unet,
        OnnxSession vaeDecoder,
        IScheduler scheduler)
    {
        RepoId = repoId;
        Tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _unet = unet;
        _vaeDecoder = vaeDecoder;
        Scheduler = scheduler;
    }

    /// <summary>The repository this pipeline came from.</summary>
    public string RepoId { get; }

    /// <summary>The prompt tokenizer.</summary>
    public HfTokenizer Tokenizer { get; }

    /// <summary>The sampler the reverse process walks with.</summary>
    public IScheduler Scheduler { get; set; }

    /// <summary>How many tokens the text encoder takes.</summary>
    public int MaxPromptTokens { get; init; } = 77;

    /// <summary>The file layout this pipeline expects inside a repository.</summary>
    public static IReadOnlyList<string> RequiredFiles { get; } =
    [
        "text_encoder/model.onnx",
        "unet/model.onnx",
        "vae_decoder/model.onnx",
    ];

    /// <summary>Downloads an ONNX Stable Diffusion repository and builds a pipeline.</summary>
    /// <param name="repoId">A model id whose repository holds the ONNX exports.</param>
    /// <param name="scheduler">The sampler to use. DDIM when null.</param>
    /// <param name="target">Which execution provider to run on.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives download progress.</param>
    /// <exception cref="HubException">The repository has no ONNX export.</exception>
    public static DiffusionPipeline FromPretrained(
        string repoId,
        IScheduler? scheduler = null,
        ExecutionTarget target = ExecutionTarget.Auto,
        string revision = "main",
        IProgress<TransferProgress>? progress = null)
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
            allowPatterns: ["*/model.onnx", "*/model.onnx_data", "*/*.onnx_data", "tokenizer/*", "*.json", "*.txt"],
            ignorePatterns: null,
            progress).GetAwaiter().GetResult();

        var tokenizer = LoadTokenizer(directory, repoId, revision);

        return new DiffusionPipeline(
            repoId,
            tokenizer,
            OnnxSession.Open(Path.Combine(directory, "text_encoder", "model.onnx"), target),
            OnnxSession.Open(Path.Combine(directory, "unet", "model.onnx"), target),
            OnnxSession.Open(Path.Combine(directory, "vae_decoder", "model.onnx"), target),
            scheduler ?? new DdimScheduler());
    }

    /// <summary>Generates an image from a prompt.</summary>
    /// <param name="prompt">What to draw.</param>
    /// <param name="options">Steps, guidance, size and seed.</param>
    /// <param name="onStep">Called after each denoising step, with the index and the total.</param>
    /// <returns>The generated image.</returns>
    public Image<Rgb24> Generate(
        string prompt,
        GenerationOptions? options = null,
        Action<int, int>? onStep = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);

        var settings = options ?? GenerationOptions.Default;

        if (settings.Width % VaeScaleFactor != 0 || settings.Height % VaeScaleFactor != 0)
        {
            throw new ArgumentException(
                $"Width and height must be multiples of {VaeScaleFactor}; the latent grid is that much "
                + $"smaller than the image. Got {settings.Width}x{settings.Height}.", nameof(options));
        }

        var conditional = EncodePrompt(prompt);
        var unconditional = settings.UsesGuidance ? EncodePrompt(settings.NegativePrompt) : null;

        var timesteps = Scheduler.SetTimesteps(settings.Steps);

        var latentWidth = settings.Width / VaeScaleFactor;
        var latentHeight = settings.Height / VaeScaleFactor;
        var latents = InitialLatents(latentHeight, latentWidth, settings.Seed, Scheduler.InitialNoiseScale);

        for (var step = 0; step < timesteps.Count; step++)
        {
            var scaled = Scheduler.ScaleInput(latents, step);
            var noise = PredictNoise(scaled, timesteps[step], conditional);

            if (unconditional is not null)
            {
                // Classifier-free guidance: push away from what the model would have drawn with no
                // prompt, by the guidance scale. This is the second UNet evaluation per step.
                var unguided = PredictNoise(scaled, timesteps[step], unconditional);

                for (var i = 0; i < noise.Size; i++)
                {
                    var guided = unguided.At(i) + settings.GuidanceScale * (noise.At(i) - unguided.At(i));
                    noise.SetAt(i, guided);
                }
            }

            latents = Scheduler.Step(noise, step, latents);
            onStep?.Invoke(step + 1, timesteps.Count);
        }

        return Decode(latents);
    }

    /// <summary>Runs the text encoder over a prompt.</summary>
    private NdArray EncodePrompt(string prompt)
    {
        var encoding = Tokenizer.EncodeBatch([prompt], BatchOptions.Exactly(MaxPromptTokens));

        var feeds = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var input in _textEncoder.Inputs)
        {
            feeds[input.Name] = input.Name switch
            {
                "attention_mask" => encoding.AttentionMask,
                _ => encoding.Ids,
            };
        }

        var outputs = _textEncoder.Run(feeds);

        // CLIP's text encoder returns the per-token states first and a pooled vector second; the
        // UNet cross-attends to the per-token states, so the first output is the one wanted.
        return outputs.TryGetValue("last_hidden_state", out var hidden) ? hidden : outputs.Values.First();
    }

    /// <summary>Runs the UNet for one timestep.</summary>
    private NdArray PredictNoise(NdArray latents, int timestep, NdArray textStates)
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

    /// <summary>Decodes a latent grid into pixels.</summary>
    private Image<Rgb24> Decode(NdArray latents)
    {
        var scaled = NdArray.Zeros(latents.Shape.ToArray());
        for (var i = 0; i < latents.Size; i++) scaled.SetAt(i, latents.At(i) / LatentScaling);

        var decoded = _vaeDecoder.Run(new Dictionary<string, NdArray>
        {
            [_vaeDecoder.Inputs[0].Name] = scaled,
        }).Values.First();

        return ToImage(decoded);
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

    /// <summary>Maps a VAE output value in <c>[-1, 1]</c> onto a byte.</summary>
    private static byte ToByte(double value) => (byte)Math.Clamp((value / 2 + 0.5) * 255, 0, 255);

    /// <summary>Draws the initial latent noise.</summary>
    /// <remarks>
    /// Seeded from <see cref="GraviRandom"/> so a prompt and a seed reproduce an image exactly -
    /// which is the whole basis of sharing a generation rather than just a picture of one.
    /// </remarks>
    internal static NdArray InitialLatents(int height, int width, int seed, double scale)
    {
        var random = new GraviRandom(seed);
        var latents = NdArray.Zeros(1, LatentChannels, height, width);

        for (var i = 0; i < latents.Size; i++) latents.SetAt(i, random.Normal() * scale);

        return latents;
    }

    private static HfTokenizer LoadTokenizer(string directory, string repoId, string revision)
    {
        foreach (var candidate in (string[])
        [
            Path.Combine(directory, "tokenizer", "tokenizer.json"),
            Path.Combine(directory, "tokenizer.json"),
        ])
        {
            if (File.Exists(candidate)) return HfTokenizer.Load(candidate);
        }

        foreach (var candidate in (string[])
        [
            Path.Combine(directory, "tokenizer", "vocab.json"),
            Path.Combine(directory, "vocab.json"),
        ])
        {
            var merges = Path.Combine(Path.GetDirectoryName(candidate)!, "merges.txt");
            if (File.Exists(candidate) && File.Exists(merges))
            {
                return HfTokenizer.FromGpt2Files(candidate, merges);
            }
        }

        // Falling back to the Hub covers a repository that keeps its tokenizer outside the patterns
        // the snapshot fetched.
        return HfTokenizer.FromPretrained(repoId, revision);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _textEncoder.Dispose();
        _unet.Dispose();
        _vaeDecoder.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"DiffusionPipeline({RepoId}, {Scheduler})";
}
