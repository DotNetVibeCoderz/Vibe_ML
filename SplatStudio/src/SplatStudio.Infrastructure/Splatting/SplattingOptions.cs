namespace SplatStudio.Infrastructure.Splatting;

public class SplattingOptions
{
    public const string SectionName = "Splatting";

    /// <summary>LocalHeuristic | ExternalApi</summary>
    public string Engine { get; set; } = "LocalHeuristic";

    /// <summary>Upper bound on splats per scene — the main lever for conversion speed vs. visual density.</summary>
    public int MaxPoints { get; set; } = 40000;
}
