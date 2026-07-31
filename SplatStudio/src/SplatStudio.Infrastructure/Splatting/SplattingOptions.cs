namespace SplatStudio.Infrastructure.Splatting;

public class SplattingOptions
{
    public const string SectionName = "Splatting";

    /// <summary>
    /// LocalHeuristic | Gpu | ExternalApi. "Gpu" runs the same heuristic on a CUDA/OpenCL
    /// device and silently falls back to LocalHeuristic when no such device is present.
    /// </summary>
    public string Engine { get; set; } = "LocalHeuristic";

    /// <summary>Upper bound on splats per scene — the main lever for conversion speed vs. visual density.</summary>
    public int MaxPoints { get; set; } = 40000;

    /// <summary>
    /// Point budget used when the GPU engine is active. Separate from <see cref="MaxPoints"/>
    /// because the GPU can afford a far denser cloud at the same wall-clock cost; the only
    /// real ceiling is the .splat bytes the browser has to download (32 bytes per point).
    /// </summary>
    public int GpuMaxPoints { get; set; } = 250000;
}
