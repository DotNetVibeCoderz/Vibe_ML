using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace MediaPipeNet.Extensions.DependencyInjection;

/// <summary>Global MediaPipe.NET settings for dependency injection.</summary>
public sealed class MediaPipeNetOptions
{
    /// <summary>Extra directory searched first for model files.</summary>
    public string? ModelDirectory { get; set; }

    /// <summary>Model cache directory (default: the per-user cache).</summary>
    public string? CacheDirectory { get; set; }

    /// <summary>Allow downloading missing models from nuget.org. Default true.</summary>
    public bool AllowModelDownload { get; set; } = true;

    /// <summary>Execution provider and threading.</summary>
    public InferenceOptions Inference { get; set; } = InferenceOptions.Default;
}

/// <summary>Fluent builder returned by <see cref="ServiceCollectionExtensions.AddMediaPipeNet"/>.</summary>
public interface IMediaPipeNetBuilder
{
    /// <summary>The service collection.</summary>
    IServiceCollection Services { get; }
}

internal sealed class MediaPipeNetBuilder(IServiceCollection services) : IMediaPipeNetBuilder
{
    public IServiceCollection Services { get; } = services;
}

/// <summary>DI registration for MediaPipe.NET tasks.</summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers MediaPipe.NET core services: <see cref="ModelStore"/> and the shared <see cref="BaseOptions"/>
    /// (wired to the container's <see cref="ILoggerFactory"/>). Chain <c>AddFaceDetector()</c> etc. to register tasks.
    /// </summary>
    /// <example>
    /// <code>
    /// builder.Services.AddMediaPipeNet(o =&gt; o.Inference = new() { Provider = ExecutionProvider.Cpu })
    ///     .AddFaceDetector()
    ///     .AddHandLandmarker(o =&gt; o with { NumHands = 2 });
    /// </code>
    /// </example>
    public static IMediaPipeNetBuilder AddMediaPipeNet(this IServiceCollection services, Action<MediaPipeNetOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddOptions<MediaPipeNetOptions>();
        if (configure is not null) services.Configure(configure);
        services.TryAddSingleton(sp =>
        {
            var o = sp.GetRequiredService<IOptions<MediaPipeNetOptions>>().Value;
            return ModelStore.CreateDefault(o.ModelDirectory, o.CacheDirectory, o.AllowModelDownload,
                sp.GetService<ILoggerFactory>()?.CreateLogger<ModelStore>());
        });
        services.TryAddSingleton(sp =>
        {
            var o = sp.GetRequiredService<IOptions<MediaPipeNetOptions>>().Value;
            return new BaseOptions
            {
                ModelStore = sp.GetRequiredService<ModelStore>(),
                ModelDirectory = o.ModelDirectory,
                Inference = o.Inference,
                LoggerFactory = sp.GetService<ILoggerFactory>(),
            };
        });
        return new MediaPipeNetBuilder(services);
    }

    /// <summary>Registers a <see cref="FaceDetector"/> singleton (image mode, thread-safe).</summary>
    public static IMediaPipeNetBuilder AddFaceDetector(this IMediaPipeNetBuilder builder, Func<FaceDetectorOptions, FaceDetectorOptions>? configure = null) =>
        Add(builder, b => FaceDetector.Create(Configure(new FaceDetectorOptions { BaseOptions = b }, configure)));

    /// <summary>Registers a <see cref="FaceLandmarker"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddFaceLandmarker(this IMediaPipeNetBuilder builder, Func<FaceLandmarkerOptions, FaceLandmarkerOptions>? configure = null) =>
        Add(builder, b => FaceLandmarker.Create(Configure(new FaceLandmarkerOptions { BaseOptions = b }, configure)));

    /// <summary>Registers a <see cref="HandLandmarker"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddHandLandmarker(this IMediaPipeNetBuilder builder, Func<HandLandmarkerOptions, HandLandmarkerOptions>? configure = null) =>
        Add(builder, b => HandLandmarker.Create(Configure(new HandLandmarkerOptions { BaseOptions = b }, configure)));

    /// <summary>Registers a <see cref="GestureRecognizer"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddGestureRecognizer(this IMediaPipeNetBuilder builder, Func<GestureRecognizerOptions, GestureRecognizerOptions>? configure = null) =>
        Add(builder, b => GestureRecognizer.Create(Configure(new GestureRecognizerOptions { BaseOptions = b }, configure)));

    /// <summary>Registers a <see cref="PoseLandmarker"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddPoseLandmarker(this IMediaPipeNetBuilder builder, Func<PoseLandmarkerOptions, PoseLandmarkerOptions>? configure = null) =>
        Add(builder, b => PoseLandmarker.Create(Configure(new PoseLandmarkerOptions { BaseOptions = b }, configure)));

    /// <summary>Registers a <see cref="HolisticLandmarker"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddHolisticLandmarker(this IMediaPipeNetBuilder builder, Func<HolisticLandmarkerOptions, HolisticLandmarkerOptions>? configure = null) =>
        Add(builder, b => HolisticLandmarker.Create(Configure(new HolisticLandmarkerOptions { BaseOptions = b }, configure)));

    /// <summary>Registers an <see cref="ImageSegmenter"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddImageSegmenter(this IMediaPipeNetBuilder builder, Func<ImageSegmenterOptions, ImageSegmenterOptions>? configure = null) =>
        Add(builder, b => ImageSegmenter.Create(Configure(new ImageSegmenterOptions { BaseOptions = b }, configure)));

    /// <summary>Registers an <see cref="ObjectDetector"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddObjectDetector(this IMediaPipeNetBuilder builder, Func<ObjectDetectorOptions, ObjectDetectorOptions>? configure = null) =>
        Add(builder, b => ObjectDetector.Create(Configure(new ObjectDetectorOptions { BaseOptions = b }, configure)));

    /// <summary>Registers an <see cref="ImageClassifier"/> singleton.</summary>
    public static IMediaPipeNetBuilder AddImageClassifier(this IMediaPipeNetBuilder builder, Func<ImageClassifierOptions, ImageClassifierOptions>? configure = null) =>
        Add(builder, b => ImageClassifier.Create(Configure(new ImageClassifierOptions { BaseOptions = b }, configure)));

    // Registered tasks are shared singletons, so they are always created in RunningMode.Image (thread-safe).
    private static T Configure<T>(T options, Func<T, T>? configure) where T : class => configure?.Invoke(options) ?? options;

    private static IMediaPipeNetBuilder Add<TTask>(IMediaPipeNetBuilder builder, Func<BaseOptions, TTask> factory) where TTask : class
    {
        ArgumentNullException.ThrowIfNull(builder);
        builder.Services.TryAddSingleton(sp => factory(sp.GetRequiredService<BaseOptions>()));
        return builder;
    }
}
