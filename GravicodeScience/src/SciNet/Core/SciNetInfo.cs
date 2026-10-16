using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Identity of the SciNet library and the map from SciPy modules to SciNet entry points.
/// </summary>
public static class SciNetInfo
{
    /// <summary>Library name as shown in banners.</summary>
    public const string Name = "SciNet";

    /// <summary>The attribution line shared by the whole Gravicode.Science stack.</summary>
    public const string Attribution = GraviInfo.Attribution;

    /// <summary>A banner for the top of a console sample.</summary>
    public static string Banner() => GraviInfo.Banner(Name);
}
