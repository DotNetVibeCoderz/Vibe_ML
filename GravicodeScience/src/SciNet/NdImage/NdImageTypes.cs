using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// How <see cref="NdImage"/> extends an array beyond its edges, mirroring scipy.ndimage's
/// <c>mode</c> strings. For a line <c>a b c d</c>:
/// </summary>
/// <remarks>
/// <para>
/// Filters and interpolation read these differently, exactly as scipy does. For filters the
/// <c>Grid*</c> names are aliases: <see cref="GridMirror"/> is <see cref="Reflect"/>,
/// <see cref="GridConstant"/> is <see cref="Constant"/> and <see cref="GridWrap"/> is
/// <see cref="Wrap"/>.
/// </para>
/// <para>
/// For interpolation they are not. <see cref="Constant"/> returns <c>cval</c> for any sample
/// outside <c>[0, n-1]</c> and interpolates inside using mirrored taps, whereas
/// <see cref="GridConstant"/> treats every tap outside the grid as <c>cval</c> and interpolates
/// everywhere. <see cref="Wrap"/> is scipy's historical 'wrap', whose period is <c>n-1</c> (the
/// first and last samples coincide) and whose spline taps are mirrored; <see cref="GridWrap"/>
/// is true periodic extension with period <c>n</c>.
/// </para>
/// </remarks>
public enum NdImageMode
{
    /// <summary><c>d c b a | a b c d | d c b a</c> — reflect about the edge of the last pixel (half-sample symmetric).</summary>
    Reflect,
    /// <summary><c>d c b | a b c d | c b a</c> — reflect about the centre of the last pixel (whole-sample symmetric).</summary>
    Mirror,
    /// <summary><c>a a a a | a b c d | d d d d</c> — repeat the edge value.</summary>
    Nearest,
    /// <summary><c>k k k k | a b c d | k k k k</c> with <c>k = cval</c>; see the remarks for the interpolation meaning.</summary>
    Constant,
    /// <summary><c>a b c d | a b c d | a b c d</c> for filters; scipy's period-(n-1) 'wrap' for interpolation.</summary>
    Wrap,
    /// <summary>Synonym of <see cref="Reflect"/>.</summary>
    GridMirror,
    /// <summary>Constant extension that interpolation also applies to taps outside the grid.</summary>
    GridConstant,
    /// <summary>Periodic extension with period n (filters treat it as <see cref="Wrap"/>).</summary>
    GridWrap,
}

/// <summary>Metric for the chamfer and brute-force distance transforms.</summary>
public enum NdImageDistanceMetric
{
    /// <summary>Euclidean distance (brute force only).</summary>
    Euclidean,
    /// <summary>City-block / Manhattan distance (connectivity 1).</summary>
    Taxicab,
    /// <summary>Chessboard distance (full connectivity).</summary>
    Chessboard,
}

/// <summary>
/// Result of a distance transform that may also return the feature transform.
/// </summary>
/// <param name="Distances">Distance of each element to the nearest background element, or null when not requested.</param>
/// <param name="Indices">
/// Shape <c>(rank, ...input shape)</c>: for each element, the coordinates of the background element
/// it is nearest to, or null when not requested.
/// </param>
public sealed record NdImageDistanceResult(NdArray? Distances, NdArray? Indices);

/// <summary>Minimum, maximum and their first positions (raster order) inside one labelled region.</summary>
/// <param name="Minimum">Smallest value.</param>
/// <param name="Maximum">Largest value.</param>
/// <param name="MinimumPosition">Coordinates of the first occurrence of the minimum.</param>
/// <param name="MaximumPosition">Coordinates of the first occurrence of the maximum.</param>
public sealed record NdImageExtrema(double Minimum, double Maximum, int[] MinimumPosition, int[] MaximumPosition);

/// <summary>Function evaluated by <c>NdImage.GenericFilter</c>
/// on the values under the footprint, in the footprint's row-major order.</summary>
/// <param name="values">The neighbourhood values; the span is reused, so copy it if you keep it.</param>
/// <returns>The output value for this position.</returns>
public delegate double NdImageFootprintFunction(ReadOnlySpan<double> values);

/// <summary>Function evaluated by <see cref="NdImage.GenericFilter1d"/> on one line.</summary>
/// <param name="input">The line, extended by <c>filterSize - 1</c> boundary values (shifted by the origin).</param>
/// <param name="output">Receives the filtered line; its length is the original line length.</param>
public delegate void NdImageLineFunction(ReadOnlySpan<double> input, Span<double> output);

/// <summary>A 1-D derivative used by the generic Laplace / gradient-magnitude filters.</summary>
/// <param name="input">The array to differentiate.</param>
/// <param name="axis">The axis along which to differentiate.</param>
/// <param name="mode">Boundary mode.</param>
/// <param name="cval">Constant for <see cref="NdImageMode.Constant"/>.</param>
/// <returns>The derivative, same shape as <paramref name="input"/>.</returns>
public delegate NdArray NdImageDerivative(NdArray input, int axis, NdImageMode mode, double cval);
