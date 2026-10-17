namespace Gravicode.Science.SciNet;

public static partial class Stats
{
    /// <summary>
    /// Spearman's ρ of two paired samples with its t-approximation p-value, scipy's <c>spearmanrho</c>
    /// (new in scipy 1.17: the one-pair form of <c>spearmanr</c>, with the same statistic and p-value).
    /// </summary>
    public static SignificanceResult Spearmanrho(double[] x, double[] y, Alternative alternative = Alternative.TwoSided)
        => Spearmanr(x, y, alternative);
}
