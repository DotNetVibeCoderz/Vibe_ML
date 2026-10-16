namespace Gravicode.Science.SciNet;

/// <summary>A CODATA physical constant: value, unit and standard uncertainty (scipy.constants.physical_constants).</summary>
/// <param name="Value">Value in the stated unit.</param>
/// <param name="Unit">Unit, written as SciPy writes it (e.g. "J K^-1"); empty for dimensionless constants.</param>
/// <param name="Uncertainty">Absolute standard uncertainty; zero for exact (SI-defining or derived-exact) values.</param>
public sealed record PhysicalConstant(double Value, string Unit, double Uncertainty);

/// <summary>Temperature scales understood by <see cref="Constants.ConvertTemperature(double, TemperatureScale, TemperatureScale)"/>.</summary>
public enum TemperatureScale
{
    /// <summary>Degrees Celsius.</summary>
    Celsius,

    /// <summary>Kelvin.</summary>
    Kelvin,

    /// <summary>Degrees Fahrenheit.</summary>
    Fahrenheit,

    /// <summary>Degrees Rankine.</summary>
    Rankine,
}

/// <summary>
/// Physical and mathematical constants and unit conversion factors, mirroring
/// <c>scipy.constants</c>: every module-level SciPy name as a PascalCase <c>const double</c>
/// (in SI units), the full CODATA 2022 table as <see cref="PhysicalConstants"/>, and the
/// temperature and wavelength/frequency conversions.
/// </summary>
/// <remarks>
/// The values are generated from SciPy 1.17.1, not typed in: the CODATA set changes every four
/// years and a hand-copied table drifts. SciPy's one-letter names that would collide in
/// PascalCase are resolved as: <c>G</c> is the gravitational constant and <c>g</c> becomes
/// <see cref="StandardGravity"/>; <c>e</c> is only available as <see cref="ElementaryCharge"/>
/// (an <c>E</c> would read as Euler's number); <c>m_e</c>, <c>m_p</c>, <c>m_n</c>, <c>m_u</c> and
/// <c>N_A</c> become <see cref="MElectron"/>, <see cref="MProton"/>, <see cref="MNeutron"/>,
/// <see cref="MU"/> and <see cref="NA"/>.
/// <para>Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil.</para>
/// </remarks>
public static partial class Constants
{
    private static readonly IReadOnlyDictionary<string, PhysicalConstant> Table = BuildTable();
    private static readonly HashSet<string> Current = BuildCurrent();

    /// <summary>
    /// The CODATA 2022 recommended values keyed by SciPy's names (e.g. "electron mass",
    /// "Planck constant in eV/Hz"). Lookups are exact and case-sensitive, as in SciPy. Like
    /// SciPy's dictionary it also holds the obsolete spellings of older CODATA releases
    /// ("... magn. moment", "inverse meters") with the values of the release that used them;
    /// <see cref="Find"/> lists only the current names, so a search never lands on a stale value.
    /// </summary>
    public static IReadOnlyDictionary<string, PhysicalConstant> PhysicalConstants => Table;

    /// <summary>The value of a physical constant (scipy.constants.value).</summary>
    /// <exception cref="KeyNotFoundException">The key is not a CODATA constant name.</exception>
    public static double Value(string key) => Get(key).Value;

    /// <summary>The unit of a physical constant (scipy.constants.unit).</summary>
    public static string Unit(string key) => Get(key).Unit;

    /// <summary>Relative precision, uncertainty / |value|, of a physical constant (scipy.constants.precision).</summary>
    public static double Precision(string key)
    {
        var c = Get(key);
        return c.Uncertainty / Math.Abs(c.Value);
    }

    /// <summary>
    /// Current CODATA 2022 keys containing <paramref name="sub"/>, ignoring case, in ordinal order
    /// (scipy.constants.find); every current key when <paramref name="sub"/> is null.
    /// </summary>
    public static IReadOnlyList<string> Find(string? sub = null)
    {
        var keys = new List<string>();
        foreach (var k in Current)
            if (sub is null || k.Contains(sub, StringComparison.OrdinalIgnoreCase)) keys.Add(k);
        keys.Sort(StringComparer.Ordinal);
        return keys;
    }

    private static PhysicalConstant Get(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (Table.TryGetValue(key, out var c)) return c;
        throw new KeyNotFoundException($"'{key}' is not in the CODATA 2022 table; use Constants.Find to search it.");
    }

    /// <summary>
    /// Converts a temperature between scales given by name (scipy.constants.convert_temperature).
    /// Accepts SciPy's spellings: "Celsius"/"celsius"/"C"/"c", "Kelvin"/"kelvin"/"K"/"k",
    /// "Fahrenheit"/"fahrenheit"/"F"/"f" and "Rankine"/"rankine"/"R"/"r".
    /// </summary>
    /// <exception cref="ArgumentException">An unrecognised scale name.</exception>
    public static double ConvertTemperature(double value, string fromScale, string toScale)
        => ConvertTemperature(value, ParseScale(fromScale, nameof(fromScale)), ParseScale(toScale, nameof(toScale)));

    /// <summary>Converts a temperature between scales, via kelvin.</summary>
    public static double ConvertTemperature(double value, TemperatureScale fromScale, TemperatureScale toScale)
    {
        var kelvin = fromScale switch
        {
            TemperatureScale.Celsius => value + ZeroCelsius,
            TemperatureScale.Kelvin => value,
            TemperatureScale.Fahrenheit => (value - 32) * 5 / 9 + ZeroCelsius,
            TemperatureScale.Rankine => value * 5 / 9,
            _ => throw new ArgumentOutOfRangeException(nameof(fromScale)),
        };
        return toScale switch
        {
            TemperatureScale.Celsius => kelvin - ZeroCelsius,
            TemperatureScale.Kelvin => kelvin,
            TemperatureScale.Fahrenheit => (kelvin - ZeroCelsius) * 9 / 5 + 32,
            TemperatureScale.Rankine => kelvin * 9 / 5,
            _ => throw new ArgumentOutOfRangeException(nameof(toScale)),
        };
    }

    /// <summary>Converts each element of <paramref name="values"/>; a new array is returned.</summary>
    public static double[] ConvertTemperature(ReadOnlySpan<double> values, TemperatureScale fromScale, TemperatureScale toScale)
    {
        var r = new double[values.Length];
        for (var i = 0; i < r.Length; i++) r[i] = ConvertTemperature(values[i], fromScale, toScale);
        return r;
    }

    private static TemperatureScale ParseScale(string s, string param)
    {
        ArgumentNullException.ThrowIfNull(s, param);
        return s switch
        {
            "Celsius" or "celsius" or "C" or "c" => TemperatureScale.Celsius,
            "Kelvin" or "kelvin" or "K" or "k" => TemperatureScale.Kelvin,
            "Fahrenheit" or "fahrenheit" or "F" or "f" => TemperatureScale.Fahrenheit,
            "Rankine" or "rankine" or "R" or "r" => TemperatureScale.Rankine,
            _ => throw new ArgumentException($"'{s}' is not a supported temperature scale (Celsius, Kelvin, Fahrenheit, Rankine).", param),
        };
    }

    /// <summary>Optical frequency (Hz) of a wavelength (m): c/λ (scipy.constants.lambda2nu).</summary>
    public static double Lambda2Nu(double lambda) => SpeedOfLight / lambda;

    /// <summary>Wavelength (m) of an optical frequency (Hz): c/ν (scipy.constants.nu2lambda).</summary>
    public static double Nu2Lambda(double nu) => SpeedOfLight / nu;
}
