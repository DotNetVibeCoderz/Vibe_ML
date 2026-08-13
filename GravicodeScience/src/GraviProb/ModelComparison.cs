using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>An information criterion together with its uncertainty.</summary>
/// <param name="Estimate">The criterion on the deviance scale — lower is better.</param>
/// <param name="EffectiveParameters">
/// The estimated model complexity. Not a count of coefficients: a parameter constrained by a tight
/// prior contributes less than one.
/// </param>
/// <param name="StandardError">
/// Uncertainty in the estimate, from the spread of the pointwise terms. Two models whose difference
/// is inside a couple of these are not distinguishable by this data.
/// </param>
/// <param name="Pointwise">The per-observation contributions, which is where the outliers show up.</param>
public readonly record struct InformationCriterion(
    double Estimate, double EffectiveParameters, double StandardError, NdArray Pointwise)
{
    /// <summary>The expected log pointwise predictive density — the criterion before the ×-2.</summary>
    public double ExpectedLogDensity => -Estimate / 2;

    /// <inheritdoc />
    public override string ToString()
        => $"{Estimate:F2} ± {StandardError:F2} (p_eff = {EffectiveParameters:F2})";
}

/// <summary>
/// Out-of-sample predictive accuracy estimated from the posterior, without a held-out set.
/// </summary>
/// <remarks>
/// <para>
/// The question these answer is "how well would this model predict data it has not seen", and the
/// reason they exist is that the obvious in-sample answer is systematically optimistic — a more
/// flexible model always fits the data it was fitted on better. Both methods start from the same
/// input: the log likelihood of every observation under every posterior draw.
/// </para>
/// <para>
/// <b>WAIC</b> estimates the optimism as the posterior variance of each observation's log
/// likelihood. An observation the model is uncertain about contributes more, which is what makes the
/// penalty a measure of effective complexity rather than a parameter count.
/// </para>
/// <para>
/// <b>PSIS-LOO</b> asks the same question by importance sampling: reweighting the posterior to
/// approximate what it would have been with one observation removed. It is generally preferred, not
/// because it is more accurate on well-behaved problems — the two agree closely there — but because
/// it comes with a diagnostic. The Pareto <c>k</c> for each observation says whether the reweighting
/// was trustworthy, and WAIC has no equivalent: it fails silently in exactly the cases LOO reports.
/// </para>
/// <para>
/// <b>Both are on the deviance scale, so lower is better</b>, and neither means anything in
/// absolute terms — only differences between models fitted to the same observations are
/// interpretable.
/// </para>
/// </remarks>
public static class ModelComparison
{
    /// <summary>The Pareto shape above which importance sampling is considered unreliable.</summary>
    /// <remarks>
    /// The usual threshold. Above 0.7 the importance weights have infinite variance and the LOO
    /// estimate for that observation should not be trusted — refit without it, or use a model that
    /// does not depend so heavily on one point.
    /// </remarks>
    public const double PairwiseReliabilityThreshold = 0.7;

    /// <summary>
    /// The widely applicable information criterion.
    /// </summary>
    /// <param name="logLikelihood">
    /// A (draws × observations) matrix: entry <c>[s, i]</c> is the log likelihood of observation
    /// <c>i</c> under posterior draw <c>s</c>.
    /// </param>
    /// <remarks>
    /// <c>WAIC = −2(lppd − p_waic)</c>, where the first term is the log of the posterior-mean
    /// predictive density per observation and the second is the posterior variance of the log
    /// density. The order of operations in the first term matters: it is the log of a mean, not the
    /// mean of logs, and computing it the wrong way round gives a number that looks reasonable and
    /// is not WAIC.
    /// </remarks>
    public static InformationCriterion Waic(NdArray logLikelihood)
    {
        var (draws, observations) = Validate(logLikelihood);

        var pointwise = NdArray.Zeros(observations);
        var lppd = 0.0;
        var penalty = 0.0;

        for (var i = 0; i < observations; i++)
        {
            var column = Column(logLikelihood, draws, i);

            var logMean = LogMeanExp(column);
            var variance = Variance(column);

            lppd += logMean;
            penalty += variance;
            pointwise.SetAt(i, -2 * (logMean - variance));
        }

        return new InformationCriterion(
            -2 * (lppd - penalty), penalty, StandardError(pointwise), pointwise);
    }

    /// <summary>The result of a PSIS-LOO computation, including its diagnostics.</summary>
    /// <param name="Criterion">The criterion itself.</param>
    /// <param name="ParetoK">
    /// Per-observation Pareto shape estimates. Values above
    /// <see cref="PairwiseReliabilityThreshold"/> mean the estimate for that observation is not
    /// trustworthy.
    /// </param>
    public readonly record struct LooResult(InformationCriterion Criterion, NdArray ParetoK)
    {
        /// <summary>Observations whose importance sampling was unreliable.</summary>
        public IReadOnlyList<int> UnreliableObservations
        {
            get
            {
                var found = new List<int>();
                for (var i = 0; i < ParetoK.Size; i++)
                    if (ParetoK.At(i) > PairwiseReliabilityThreshold) found.Add(i);
                return found;
            }
        }

        /// <summary>True when every observation's estimate is trustworthy.</summary>
        public bool IsReliable => UnreliableObservations.Count == 0;
    }

    /// <summary>
    /// Leave-one-out cross-validation by Pareto-smoothed importance sampling.
    /// </summary>
    /// <param name="logLikelihood">A (draws × observations) matrix of pointwise log likelihoods.</param>
    /// <remarks>
    /// <para>
    /// The importance weight for dropping observation <c>i</c> is <c>1/p(yᵢ|θₛ)</c>, so the raw
    /// weights are the negated log likelihoods. Those weights are heavy-tailed — an observation the
    /// model fits badly under one draw gets an enormous weight — and the plain estimate is dominated
    /// by a handful of draws.
    /// </para>
    /// <para>
    /// Pareto smoothing fixes that by fitting a generalised Pareto distribution to the largest
    /// weights and replacing them with the fitted quantiles. The fitted shape <c>k</c> is then a
    /// free diagnostic: it says how heavy the tail was, and therefore whether the answer can be
    /// believed. That diagnostic is the reason to prefer this over WAIC.
    /// </para>
    /// </remarks>
    public static LooResult Loo(NdArray logLikelihood)
    {
        var (draws, observations) = Validate(logLikelihood);

        var pointwise = NdArray.Zeros(observations);
        var shapes = NdArray.Zeros(observations);
        var elpd = 0.0;
        var lppd = 0.0;

        for (var i = 0; i < observations; i++)
        {
            var column = Column(logLikelihood, draws, i);

            // Raw log importance weights: dropping observation i means dividing by its likelihood.
            var logWeights = new double[draws];
            for (var s = 0; s < draws; s++) logWeights[s] = -column[s];

            var k = SmoothTail(logWeights);
            shapes.SetAt(i, k);

            // The LOO predictive density is the weighted harmonic-style mean: the weights and the
            // likelihood combine so that the numerator is simply the normalising constant.
            var numerator = LogSumExp(logWeights, column);
            var denominator = LogSumExp(logWeights);

            var value = numerator - denominator;
            elpd += value;
            lppd += LogMeanExp(column);
            pointwise.SetAt(i, -2 * value);
        }

        var criterion = new InformationCriterion(
            -2 * elpd, lppd - elpd, StandardError(pointwise), pointwise);

        return new LooResult(criterion, shapes);
    }

    /// <summary>
    /// Compares models, best first, with the difference from the best and its standard error.
    /// </summary>
    /// <remarks>
    /// The standard error of the <em>difference</em> is computed from the paired pointwise terms,
    /// not from the two models' individual errors. That matters: the models are evaluated on the
    /// same observations, so their errors are strongly correlated, and treating them as independent
    /// makes every difference look insignificant.
    /// </remarks>
    public static IReadOnlyList<(string Name, double Estimate, double Difference, double DifferenceError)>
        Compare(IReadOnlyDictionary<string, InformationCriterion> models)
    {
        ArgumentNullException.ThrowIfNull(models);
        if (models.Count == 0) throw new ArgumentException("There is nothing to compare.", nameof(models));

        var ordered = models.OrderBy(m => m.Value.Estimate).ToArray();
        var best = ordered[0].Value;

        var result = new List<(string, double, double, double)>(ordered.Length);

        foreach (var (name, criterion) in ordered)
        {
            if (ReferenceEquals(criterion.Pointwise, best.Pointwise) || criterion.Estimate == best.Estimate)
            {
                result.Add((name, criterion.Estimate, 0.0, 0.0));
                continue;
            }

            if (criterion.Pointwise.Size != best.Pointwise.Size)
                throw new ArgumentException(
                    $"'{name}' was scored on {criterion.Pointwise.Size} observations but the best model " +
                    $"on {best.Pointwise.Size}. Only models fitted to the same data are comparable.");

            var differences = NdArray.Zeros(best.Pointwise.Size);
            for (var i = 0; i < best.Pointwise.Size; i++)
                differences.SetAt(i, criterion.Pointwise.At(i) - best.Pointwise.At(i));

            result.Add((name, criterion.Estimate, criterion.Estimate - best.Estimate, StandardError(differences)));
        }

        return result;
    }

    // ------------------------------------------------------------------ helpers

    private static (int Draws, int Observations) Validate(NdArray logLikelihood)
    {
        ArgumentNullException.ThrowIfNull(logLikelihood);

        if (logLikelihood.Rank != 2)
            throw new ArgumentException(
                "The log likelihood must be a rank 2 array of shape (draws, observations).", nameof(logLikelihood));

        var draws = logLikelihood.Shape[0];
        var observations = logLikelihood.Shape[1];

        if (draws < 2) throw new ArgumentException("At least two posterior draws are needed.", nameof(logLikelihood));
        if (observations < 1) throw new ArgumentException("At least one observation is needed.", nameof(logLikelihood));

        return (draws, observations);
    }

    private static double[] Column(NdArray matrix, int draws, int index)
    {
        var column = new double[draws];
        for (var s = 0; s < draws; s++) column[s] = matrix[s, index];
        return column;
    }

    /// <summary>log(mean(exp(x))), computed by shifting out the maximum so nothing overflows.</summary>
    private static double LogMeanExp(double[] values) => LogSumExp(values) - Math.Log(values.Length);

    private static double LogSumExp(double[] values)
    {
        var max = double.NegativeInfinity;
        foreach (var value in values) if (value > max) max = value;
        if (double.IsNegativeInfinity(max)) return max;

        var sum = 0.0;
        foreach (var value in values) sum += Math.Exp(value - max);
        return max + Math.Log(sum);
    }

    /// <summary>log(Σ exp(a + b)) over paired entries.</summary>
    private static double LogSumExp(double[] a, double[] b)
    {
        var combined = new double[a.Length];
        for (var i = 0; i < a.Length; i++) combined[i] = a[i] + b[i];
        return LogSumExp(combined);
    }

    private static double Variance(double[] values)
    {
        var mean = values.Average();
        var total = 0.0;
        foreach (var value in values) total += (value - mean) * (value - mean);
        return total / (values.Length - 1);
    }

    /// <summary>
    /// The standard error of a summed criterion: <c>sqrt(n · Var(pointwise))</c>.
    /// </summary>
    private static double StandardError(NdArray pointwise)
    {
        var n = pointwise.Size;
        if (n < 2) return 0.0;

        var mean = 0.0;
        for (var i = 0; i < n; i++) mean += pointwise.At(i);
        mean /= n;

        var variance = 0.0;
        for (var i = 0; i < n; i++)
        {
            var d = pointwise.At(i) - mean;
            variance += d * d;
        }
        variance /= n - 1;

        return Math.Sqrt(n * variance);
    }

    /// <summary>
    /// Replaces the largest log weights with fitted generalised-Pareto quantiles, in place, and
    /// returns the fitted shape.
    /// </summary>
    /// <remarks>
    /// The tail is taken as the largest <c>min(⌊S/5⌋, ⌊3√S⌋)</c> weights, the usual choice. Fewer
    /// than five and there is nothing to fit, so the weights are left alone and the shape is
    /// reported as zero — which is honest rather than a claim that the tail was light.
    /// </remarks>
    private static double SmoothTail(double[] logWeights)
    {
        var count = logWeights.Length;
        var tailSize = Math.Min(count / 5, (int)(3 * Math.Sqrt(count)));
        if (tailSize < 5) return 0.0;

        var order = Enumerable.Range(0, count).OrderBy(i => logWeights[i]).ToArray();
        var cutoff = logWeights[order[count - tailSize - 1]];

        // Fit on the exceedances above the cutoff, in the original (non-log) scale.
        var exceedances = new double[tailSize];
        for (var i = 0; i < tailSize; i++)
            exceedances[i] = Math.Exp(logWeights[order[count - tailSize + i]] - cutoff) - 1;

        var (k, sigma) = FitGeneralisedPareto(exceedances);

        if (double.IsNaN(k)) return 0.0;

        // Replace each tail weight with the fitted quantile at its plotting position, which is what
        // pulls the extreme values in without discarding them.
        for (var i = 0; i < tailSize; i++)
        {
            var p = (i + 0.5) / tailSize;
            var quantile = Math.Abs(k) < 1e-10
                ? -sigma * Math.Log(1 - p)
                : sigma * (Math.Pow(1 - p, -k) - 1) / k;

            logWeights[order[count - tailSize + i]] = Math.Log(1 + quantile) + cutoff;
        }

        return k;
    }

    /// <summary>
    /// Fits a generalised Pareto distribution by the empirical-Bayes method of Zhang and Stephens.
    /// </summary>
    /// <remarks>
    /// Chosen over maximum likelihood because it has a closed form given a small profile grid, and
    /// because ML for the GPD is badly behaved for exactly the heavy tails this is used on — it
    /// routinely fails to converge on the samples that matter most.
    /// </remarks>
    private static (double Shape, double Scale) FitGeneralisedPareto(double[] exceedances)
    {
        var n = exceedances.Length;
        var sorted = (double[])exceedances.Clone();
        Array.Sort(sorted);

        if (sorted[^1] <= 0) return (double.NaN, double.NaN);

        var prior = 3;
        var m = 30 + (int)Math.Sqrt(n);

        var theta = new double[m];
        var logLikelihood = new double[m];

        var quartile = sorted[(int)(n / 4.0 + 0.5) - 1 < 0 ? 0 : (int)(n / 4.0 + 0.5) - 1];
        if (quartile <= 0) return (double.NaN, double.NaN);

        for (var i = 0; i < m; i++)
        {
            theta[i] = 1.0 / sorted[^1] + (1 - Math.Sqrt((m + 0.0) / (i + 0.5))) / (prior * quartile);

            var k = 0.0;
            for (var j = 0; j < n; j++) k += Math.Log(1 - theta[i] * sorted[j]);
            k /= n;

            logLikelihood[i] = n * (Math.Log(-theta[i] / k) - k - 1);
        }

        // Normalised weights over the profile grid, then a weighted average of theta.
        var max = logLikelihood.Max();
        var weights = new double[m];
        var total = 0.0;

        for (var i = 0; i < m; i++)
        {
            weights[i] = Math.Exp(logLikelihood[i] - max);
            total += weights[i];
        }

        var thetaHat = 0.0;
        for (var i = 0; i < m; i++) thetaHat += theta[i] * weights[i] / total;

        var shape = 0.0;
        for (var j = 0; j < n; j++) shape += Math.Log(1 - thetaHat * sorted[j]);
        shape /= n;

        var scale = -shape / thetaHat;

        return double.IsNaN(shape) || double.IsNaN(scale) || scale <= 0
            ? (double.NaN, double.NaN)
            : (shape, scale);
    }
}
