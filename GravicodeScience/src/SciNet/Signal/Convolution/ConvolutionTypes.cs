namespace Gravicode.Science.SciNet;

/// <summary>Output size of a convolution or correlation (scipy's <c>mode</c>).</summary>
public enum ConvolveMode
{
    /// <summary>The full discrete linear convolution, length <c>n + m − 1</c> per axis.</summary>
    Full,
    /// <summary>Same size as the first input, centred with respect to the full output.</summary>
    Same,
    /// <summary>Only the points that do not rely on zero padding, length <c>|n − m| + 1</c>.</summary>
    Valid,
}

/// <summary>How <see cref="Signal.Convolve(double[], double[], ConvolveMode, ConvolveMethod)"/> computes its result.</summary>
public enum ConvolveMethod
{
    /// <summary>Pick the faster of direct and FFT from an operation-count model measured on this engine.</summary>
    Auto,
    /// <summary>Direct sums — exact for integer-valued data, O(n·m).</summary>
    Direct,
    /// <summary>Through the real FFT — O(N log N), rounding error relative to the largest output.</summary>
    Fft,
}

/// <summary>How <c>convolve2d</c>/<c>correlate2d</c> extend the first input past its edges.</summary>
public enum Convolve2dBoundary
{
    /// <summary>Pad with a constant (<c>fillvalue</c>, default 0).</summary>
    Fill,
    /// <summary>Circular (periodic) extension.</summary>
    Wrap,
    /// <summary>Symmetric reflection including the edge sample: <c>d c b a | a b c d | d c b a</c>.</summary>
    Symm,
}

/// <summary>
/// Edge extension used by <c>upfirdn</c> and <c>resample_poly</c> (scipy's <c>mode</c>/<c>padtype</c>).
/// The last four are <c>resample_poly</c>-only: the signal is shifted so that the named statistic
/// becomes zero, zero-padded, and shifted back.
/// </summary>
public enum UpfirdnPad
{
    /// <summary>A constant value (<c>cval</c>, default 0).</summary>
    Constant,
    /// <summary>Periodic extension.</summary>
    Wrap,
    /// <summary>Repeat the edge sample.</summary>
    Edge,
    /// <summary>Continue the line through the two samples nearest each edge.</summary>
    Smooth,
    /// <summary>Reflect including the edge sample (<c>b a | a b c</c>).</summary>
    Symmetric,
    /// <summary>Reflect about the edge sample (<c>c b | a b c</c>).</summary>
    Reflect,
    /// <summary>Symmetric reflection with the sign flipped.</summary>
    Antisymmetric,
    /// <summary>Point reflection about the edge sample (odd extension).</summary>
    Antireflect,
    /// <summary>Continue the line through the first and last samples.</summary>
    Line,
    /// <summary><c>resample_poly</c> only: pad with the mean.</summary>
    Mean,
    /// <summary><c>resample_poly</c> only: pad with the median.</summary>
    Median,
    /// <summary><c>resample_poly</c> only: pad with the minimum.</summary>
    Minimum,
    /// <summary><c>resample_poly</c> only: pad with the maximum.</summary>
    Maximum,
}
