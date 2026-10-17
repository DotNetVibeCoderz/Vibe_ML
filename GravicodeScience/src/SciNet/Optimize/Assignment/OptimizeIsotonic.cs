namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Optimize.IsotonicRegression"/>.</summary>
/// <param name="X">The fitted monotone sequence, same length as the input.</param>
/// <param name="Weights">Total weight of each block (pooled run of equal fitted values).</param>
/// <param name="Blocks">Block boundaries: block k covers indices Blocks[k] .. Blocks[k+1]−1; length = blocks + 1.</param>
public sealed record IsotonicRegressionResult(double[] X, double[] Weights, int[] Blocks);

public static partial class Optimize
{
    /// <summary>
    /// Weighted least-squares isotonic regression (<c>scipy.optimize.isotonic_regression</c>):
    /// minimises <c>Σ wᵢ (yᵢ − xᵢ)²</c> subject to x non-decreasing (or non-increasing), by the
    /// pool-adjacent-violators algorithm in O(n).
    /// </summary>
    /// <remarks>
    /// The merge must cascade backwards: pooling a violating pair lowers the pooled value, which can
    /// newly violate against the block before it. A forward-only pass looks right on smooth data and
    /// fails on exactly the ragged data isotonic regression exists for. Equal neighbours are pooled
    /// too, as SciPy does, so a run of ties is reported as one block.
    /// </remarks>
    /// <param name="y">Observations.</param>
    /// <param name="weights">Strictly positive weights; null means all ones.</param>
    /// <param name="increasing">Fit a non-decreasing (true) or non-increasing (false) sequence.</param>
    public static IsotonicRegressionResult IsotonicRegression(ReadOnlySpan<double> y, ReadOnlySpan<double> weights = default, bool increasing = true)
    {
        var n = y.Length;
        if (!weights.IsEmpty && weights.Length != n) throw new ArgumentException("Input arrays y and w must have one dimension of equal length.");
        var x = new double[n];
        var w = new double[n];
        for (var i = 0; i < n; i++)
        {
            var src = increasing ? i : n - 1 - i;
            if (double.IsNaN(y[src])) throw new ArgumentException("y must not contain NaN.");
            x[i] = y[src];
            w[i] = weights.IsEmpty ? 1.0 : weights[src];
            if (!(w[i] > 0) || double.IsInfinity(w[i])) throw new ArgumentException("Weights w must be strictly positive.");
        }

        // Stack of blocks: start index, weighted mean, total weight.
        var start = new int[n + 1];
        var mean = new double[n];
        var wt = new double[n];
        var b = 0;
        for (var i = 0; i < n; i++)
        {
            start[b] = i; mean[b] = x[i]; wt[b] = w[i];
            b++;
            while (b > 1 && mean[b - 2] >= mean[b - 1])
            {
                var tw = wt[b - 2] + wt[b - 1];
                mean[b - 2] = (wt[b - 2] * mean[b - 2] + wt[b - 1] * mean[b - 1]) / tw;
                wt[b - 2] = tw;
                b--;
            }
        }
        start[b] = n;

        var fit = new double[n];
        for (var k = 0; k < b; k++)
            for (var i = start[k]; i < start[k + 1]; i++) fit[i] = mean[k];
        var blockW = wt.AsSpan(0, b).ToArray();
        var blocks = start.AsSpan(0, b + 1).ToArray();
        if (!increasing)
        {
            Array.Reverse(fit);
            Array.Reverse(blockW);
            var r = new int[b + 1];
            for (var k = 0; k <= b; k++) r[k] = n - blocks[b - k];
            blocks = r;
        }
        return new IsotonicRegressionResult(fit, blockW, blocks);
    }
}
