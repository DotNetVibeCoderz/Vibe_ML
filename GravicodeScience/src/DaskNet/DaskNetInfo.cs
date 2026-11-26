using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet;

/// <summary>Identity of the Dask.Net library.</summary>
public static class DaskNetInfo
{
    /// <summary>Library name as shown in banners.</summary>
    public const string Name = "Dask.Net";

    /// <summary>The attribution line shared by the whole Gravicode.Science stack.</summary>
    public const string Attribution = GraviInfo.Attribution;

    /// <summary>A banner for the top of a console sample.</summary>
    public static string Banner() => GraviInfo.Banner(Name);
}
