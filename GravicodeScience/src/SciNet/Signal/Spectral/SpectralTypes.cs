namespace Gravicode.Science.SciNet;

/// <summary>
/// A window for the spectral estimators: either a name with parameters (as accepted by
/// <see cref="Signal.GetWindow"/>, generated periodic at the segment length) or explicit values
/// whose length fixes the segment length — scipy's "string, tuple or array_like" argument.
/// </summary>
public sealed class SpectralWindow
{
    private SpectralWindow(WindowSpec? spec, double[]? values)
    {
        Spec = spec;
        Values = values;
    }

    /// <summary>The named window, or null when explicit values were given.</summary>
    public WindowSpec? Spec { get; }

    /// <summary>The explicit window values, or null for a named window.</summary>
    public double[]? Values { get; }

    /// <summary>A named window.</summary>
    public static implicit operator SpectralWindow(string name) => new(new WindowSpec(name), null);

    /// <summary>A named window with parameters.</summary>
    public static implicit operator SpectralWindow(WindowSpec spec) => new(spec ?? throw new ArgumentNullException(nameof(spec)), null);

    /// <summary>A named window with one parameter, e.g. <c>("tukey", 0.25)</c>.</summary>
    public static implicit operator SpectralWindow((string Name, double P) spec) => new(new WindowSpec(spec.Name, spec.P), null);

    /// <summary>A named window with two parameters.</summary>
    public static implicit operator SpectralWindow((string Name, double P1, double P2) spec) => new(new WindowSpec(spec.Name, spec.P1, spec.P2), null);

    /// <summary>Explicit window values.</summary>
    public static implicit operator SpectralWindow(double[] values) => new(null, values ?? throw new ArgumentNullException(nameof(values)));

    /// <summary>
    /// scipy's <c>_triage_segments</c>: resolves the window and segment length. A named window
    /// defaults to 256 samples and is shortened to the input length; explicit values must not be
    /// longer than the input and must agree with <paramref name="nperseg"/> when both are given.
    /// </summary>
    internal (double[] Win, int Nperseg) Resolve(int? nperseg, int inputLength, int defaultLength = 256)
    {
        if (nperseg is < 1) throw new ArgumentException("nperseg must be a positive integer.", nameof(nperseg));
        if (Values is null)
        {
            var n = nperseg ?? defaultLength;
            if (n > inputLength) n = inputLength;
            if (n < 1) throw new ArgumentException("Input is empty.");
            return (Signal.GetWindow(Spec!, n), n);
        }
        if (inputLength < Values.Length) throw new ArgumentException("window is longer than input signal.");
        if (nperseg is { } np && np != Values.Length) throw new ArgumentException("value specified for nperseg is different from length of window.");
        return ((double[])Values.Clone(), Values.Length);
    }

    /// <summary>Window values for a known segment length (COLA/NOLA checks, istft).</summary>
    internal double[] ValuesFor(int nperseg)
    {
        if (Values is null) return Signal.GetWindow(Spec!, nperseg);
        if (Values.Length != nperseg) throw new ArgumentException($"window must have length of {nperseg}.");
        return (double[])Values.Clone();
    }
}

/// <summary>Units of a spectral estimate.</summary>
public enum SpectralScaling
{
    /// <summary>Power spectral density, V²/Hz.</summary>
    Density,
    /// <summary>Power spectrum, V² (squared magnitude of each bin).</summary>
    Spectrum,
}

/// <summary>How Welch-type estimators combine segments.</summary>
public enum SpectralAverage
{
    /// <summary>Arithmetic mean.</summary>
    Mean,
    /// <summary>Median, divided by the bias of the median of χ²₂ samples so it estimates the mean.</summary>
    Median,
}

/// <summary>What <see cref="Signal.Spectrogram"/> returns.</summary>
public enum SpectrogramMode
{
    /// <summary>Power spectral density (or power spectrum).</summary>
    Psd,
    /// <summary>The complex STFT.</summary>
    Complex,
    /// <summary>|STFT|.</summary>
    Magnitude,
    /// <summary>The wrapped phase angle.</summary>
    Angle,
    /// <summary>The phase unwrapped along the frequency axis (scipy's behaviour).</summary>
    Phase,
}

/// <summary>How <see cref="Signal.Stft"/> extends the signal by half a segment at each end.</summary>
public enum StftBoundary
{
    /// <summary>No extension.</summary>
    None,
    /// <summary>Zeros (scipy's default).</summary>
    Zeros,
    /// <summary>Even reflection about the edge sample.</summary>
    Even,
    /// <summary>Odd (point) reflection about the edge sample.</summary>
    Odd,
    /// <summary>Repeat the edge sample.</summary>
    Constant,
}

/// <summary>Normalisation of <see cref="Signal.Lombscargle"/>.</summary>
public enum LombscargleNormalization
{
    /// <summary>Unnormalised power (scipy's <c>normalize=False</c>).</summary>
    Power,
    /// <summary>Divided by the weighted signal power (<c>normalize=True</c>), in [0, 1].</summary>
    Normalize,
}
