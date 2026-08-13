using System.Numerics;
using System.Reflection;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// Identity and runtime capability reporting for the Gravicode.Science stack.
/// Samples and notebooks print this so a benchmark result can always be tied to the machine
/// and build that produced it.
/// </summary>
public static class GraviInfo
{
    /// <summary>Product name.</summary>
    public const string Product = "Gravicode.Science";

    /// <summary>The studio behind the project.</summary>
    public const string Vendor = "Gravicode Studios";

    /// <summary>Project lead.</summary>
    public const string Lead = "Kang Fadhil";

    /// <summary>The attribution line shown by every sample application and document.</summary>
    public const string Attribution = "Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil";

    /// <summary>Assembly version of the core library.</summary>
    public static string Version =>
        typeof(GraviInfo).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(GraviInfo).Assembly.GetName().Version?.ToString()
        ?? "0.1.0";

    /// <summary>A banner suitable for the top of a console sample.</summary>
    public static string Banner(string module) => $"""
        ==========================================================
         {Product} - {module} v{Version}
         {Attribution}
        ==========================================================
        """;

    /// <summary>Describes the hardware the numeric kernels will actually use.</summary>
    public static string HardwareReport() => $"""
         Runtime      : {Environment.Version} on {Environment.OSVersion.VersionString}
         Processors   : {Environment.ProcessorCount}
         SIMD         : {(Vector.IsHardwareAccelerated ? $"enabled, Vector<double> width {Vector<double>.Count}" : "not accelerated")}
         Backends     : {Compute.Compute.DescribeDevices()}
        """;
}
