using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Plumbing shared by the distribution classes: seeding and element-wise mapping.</summary>
internal static class StatsDistributions
{
    /// <summary>A generator for <paramref name="seed"/>, or a freshly seeded one when it is null.</summary>
    public static GraviRandom MakeRng(int? seed)
        => seed is int s ? new GraviRandom(s) : new GraviRandom((ulong)Random.Shared.NextInt64());

    /// <summary>Applies <paramref name="f"/> element-wise, in parallel above <see cref="Special.ParallelThreshold"/>.</summary>
    public static double[] Map(double[] x, Func<double, double> f)
    {
        ArgumentNullException.ThrowIfNull(x);
        var r = new double[x.Length];
        Special.Map(x, r, f);
        return r;
    }

    /// <summary>Validates a sample: non-empty and finite.</summary>
    public static void CheckData(double[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length == 0) throw new ArgumentException("data must not be empty.", nameof(data));
        foreach (var x in data)
            if (!double.IsFinite(x)) throw new ArgumentException("data must be finite.", nameof(data));
    }

    /// <summary>Throws unless <paramref name="value"/> is strictly positive and finite.</summary>
    public static double Positive(double value, string name)
    {
        if (!(value > 0) || double.IsPositiveInfinity(value)) throw new ArgumentException($"{name} must be positive and finite.", name);
        return value;
    }

    /// <summary>Throws unless <paramref name="value"/> lies in [lo, hi].</summary>
    public static double InRange(double value, double lo, double hi, string name)
    {
        if (!(value >= lo && value <= hi)) throw new ArgumentException($"{name} must lie in [{lo}, {hi}].", name);
        return value;
    }

    /// <summary>Throws unless <paramref name="value"/> is finite.</summary>
    public static double Finite(double value, string name)
    {
        if (!double.IsFinite(value)) throw new ArgumentException($"{name} must be finite.", name);
        return value;
    }

    /// <summary>The value of a fixed parameter if one was supplied.</summary>
    public static double? Fixed(double?[]? fixedShapes, int index)
        => fixedShapes != null && index < fixedShapes.Length ? fixedShapes[index] : null;
}
