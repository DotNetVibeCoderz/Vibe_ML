using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>
/// A distribution over vectors rather than scalars.
/// </summary>
/// <remarks>
/// Kept separate from <see cref="Distribution"/> rather than generalising it. The scalar interface
/// is used everywhere — priors, likelihoods, the samplers — and widening it to vectors would make
/// every implementation carry a dimension it does not have. What these share with the scalar family
/// is the contract, not the signature: a log density, a sampler, a mean, and a support that says
/// which vectors are admissible.
/// </remarks>
public abstract class MultivariateDistribution
{
    /// <summary>Name used in reports.</summary>
    public abstract string Name { get; }

    /// <summary>Number of components in a draw.</summary>
    public abstract int Dimension { get; }

    /// <summary>Natural log of the density (or mass) at <paramref name="x"/>.</summary>
    public abstract double LogDensity(NdArray x);

    /// <summary>Draws one vector.</summary>
    public abstract NdArray Sample(GraviRandom rng);

    /// <summary>The mean vector.</summary>
    public abstract NdArray Mean { get; }

    /// <summary>The covariance matrix.</summary>
    public abstract NdArray Covariance { get; }

    /// <summary>True when <paramref name="x"/> lies in the support.</summary>
    public abstract bool Supports(NdArray x);

    /// <summary>Density at <paramref name="x"/>.</summary>
    public double Density(NdArray x) => Math.Exp(LogDensity(x));

    /// <summary>Draws <paramref name="count"/> vectors, one per row.</summary>
    public NdArray Sample(GraviRandom rng, int count)
    {
        var result = NdArray.Zeros(count, Dimension);
        for (var i = 0; i < count; i++)
        {
            var draw = Sample(rng);
            for (var j = 0; j < Dimension; j++) result[i, j] = draw.At(j);
        }
        return result;
    }

    /// <summary>Total log density of a dataset with one observation per row.</summary>
    public double LogLikelihood(NdArray data)
    {
        var total = 0.0;
        for (var i = 0; i < data.Shape[0]; i++) total += LogDensity(data.Row(i));
        return total;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name}(dim={Dimension})";
}

/// <summary>
/// The multivariate normal, parameterised by a mean vector and a covariance matrix.
/// </summary>
/// <remarks>
/// <para>
/// The distribution that makes correlated uncertainty tractable. A vector of independent normals
/// cannot express "these two quantities are uncertain but move together", which is most of what a
/// posterior over several parameters looks like.
/// </para>
/// <para>
/// Everything here goes through the Cholesky factor <c>Σ = LLᵀ</c>, computed once at construction.
/// That single factorisation gives all three of the things needed: the quadratic form
/// <c>(x−μ)ᵀΣ⁻¹(x−μ)</c> by forward substitution instead of an explicit inverse, the log determinant
/// as twice the sum of the log diagonal, and sampling as <c>μ + Lz</c> from standard normals.
/// Inverting Σ directly would be slower and markedly less accurate for an ill-conditioned
/// covariance, which is exactly when it matters.
/// </para>
/// <para>
/// A covariance that is not positive definite is rejected at construction. Singular covariance is a
/// real modelling situation — perfectly correlated components — but the density is then unbounded
/// on a lower-dimensional subspace and does not exist as written, so failing loudly beats returning
/// infinities later.
/// </para>
/// </remarks>
public sealed class MultivariateNormal : MultivariateDistribution
{
    private readonly double[] _mean;
    private readonly NdArray _covariance;
    private readonly NdArray _cholesky;
    private readonly double _logNormaliser;

    /// <summary>Creates a multivariate normal.</summary>
    /// <param name="mean">The mean vector.</param>
    /// <param name="covariance">A symmetric positive-definite covariance matrix.</param>
    public MultivariateNormal(NdArray mean, NdArray covariance)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(covariance);

        Dimension = mean.Size;

        if (covariance.Rank != 2 || covariance.Shape[0] != Dimension || covariance.Shape[1] != Dimension)
            throw new ArgumentException(
                $"Covariance must be {Dimension}x{Dimension} to match the mean.", nameof(covariance));

        for (var i = 0; i < Dimension; i++)
            for (var j = i + 1; j < Dimension; j++)
                if (Math.Abs(covariance[i, j] - covariance[j, i]) > 1e-9)
                    throw new ArgumentException(
                        $"Covariance is not symmetric: [{i},{j}] is {covariance[i, j]} but [{j},{i}] is {covariance[j, i]}.",
                        nameof(covariance));

        _mean = mean.ToArray();
        _covariance = covariance.Copy();

        try
        {
            _cholesky = Decomposition.Cholesky(covariance);
        }
        catch (Exception error)
        {
            throw new ArgumentException(
                "Covariance is not positive definite, so the density does not exist. " +
                "A singular covariance means some components are perfectly determined by others.",
                nameof(covariance), error);
        }

        // log|Σ| = 2 Σ log Lᵢᵢ — free from the factor, where computing the determinant directly
        // would overflow for even a moderately large dimension.
        var logDeterminant = 0.0;
        for (var i = 0; i < Dimension; i++) logDeterminant += Math.Log(_cholesky[i, i]);
        logDeterminant *= 2;

        _logNormaliser = -0.5 * (Dimension * Math.Log(2 * Math.PI) + logDeterminant);
    }

    /// <summary>A standard multivariate normal of the given dimension.</summary>
    public static MultivariateNormal Standard(int dimension)
        => new(NdArray.Zeros(dimension), NdArray.Eye(dimension));

    /// <summary>A multivariate normal with no correlation between components.</summary>
    public static MultivariateNormal Diagonal(NdArray mean, NdArray variances)
    {
        var covariance = NdArray.Zeros(mean.Size, mean.Size);
        for (var i = 0; i < mean.Size; i++) covariance[i, i] = variances.At(i);
        return new MultivariateNormal(mean, covariance);
    }

    /// <inheritdoc />
    public override string Name => "MultivariateNormal";

    /// <inheritdoc />
    public override int Dimension { get; }

    /// <summary>The lower-triangular Cholesky factor of the covariance.</summary>
    public NdArray CholeskyFactor => _cholesky.Copy();

    /// <inheritdoc />
    public override NdArray Mean => NdArray.FromValues(_mean);

    /// <inheritdoc />
    public override NdArray Covariance => _covariance.Copy();

    /// <inheritdoc />
    public override bool Supports(NdArray x)
    {
        if (x.Size != Dimension) return false;
        for (var i = 0; i < Dimension; i++) if (double.IsNaN(x.At(i))) return false;
        return true;
    }

    /// <inheritdoc />
    /// <remarks>
    /// The quadratic form is computed as <c>‖L⁻¹(x−μ)‖²</c> by forward substitution. That is the
    /// same number as <c>(x−μ)ᵀΣ⁻¹(x−μ)</c>, reached without ever forming <c>Σ⁻¹</c>.
    /// </remarks>
    public override double LogDensity(NdArray x)
    {
        if (x.Size != Dimension)
            throw new ArgumentException($"Expected a vector of length {Dimension} but got {x.Size}.", nameof(x));

        var deviation = new double[Dimension];
        for (var i = 0; i < Dimension; i++) deviation[i] = x.At(i) - _mean[i];

        // Forward substitution: solve L y = (x - μ).
        var y = new double[Dimension];
        for (var i = 0; i < Dimension; i++)
        {
            var sum = deviation[i];
            for (var j = 0; j < i; j++) sum -= _cholesky[i, j] * y[j];
            y[i] = sum / _cholesky[i, i];
        }

        var quadratic = 0.0;
        for (var i = 0; i < Dimension; i++) quadratic += y[i] * y[i];

        return _logNormaliser - 0.5 * quadratic;
    }

    /// <inheritdoc />
    /// <remarks>
    /// <c>μ + Lz</c> where <c>z</c> is standard normal. Its covariance is
    /// <c>L·I·Lᵀ = Σ</c> by construction, which is why the Cholesky factor is the right thing to
    /// sample through.
    /// </remarks>
    public override NdArray Sample(GraviRandom rng)
    {
        var z = new double[Dimension];
        for (var i = 0; i < Dimension; i++) z[i] = rng.Normal();

        var result = NdArray.Zeros(Dimension);
        for (var i = 0; i < Dimension; i++)
        {
            var sum = _mean[i];
            for (var j = 0; j <= i; j++) sum += _cholesky[i, j] * z[j];
            result.SetAt(i, sum);
        }

        return result;
    }

    /// <summary>
    /// The conditional distribution of the components in <paramref name="unknown"/> given values for
    /// the rest.
    /// </summary>
    /// <param name="unknown">Indices of the components being predicted.</param>
    /// <param name="observed">Indices of the components whose values are known.</param>
    /// <param name="values">The known values, in the order of <paramref name="observed"/>.</param>
    /// <remarks>
    /// The property that makes Gaussians useful for prediction: conditioning a normal on part of
    /// itself gives another normal, in closed form. This is the whole mechanism behind Gaussian
    /// process regression — see <see cref="GaussianProcess"/>.
    /// </remarks>
    public MultivariateNormal Conditional(int[] unknown, int[] observed, NdArray values)
    {
        ArgumentNullException.ThrowIfNull(unknown);
        ArgumentNullException.ThrowIfNull(observed);

        if (observed.Length != values.Size)
            throw new ArgumentException("There must be one value per observed index.", nameof(values));

        var sigmaUU = Submatrix(unknown, unknown);
        var sigmaUO = Submatrix(unknown, observed);
        var sigmaOO = Submatrix(observed, observed);

        var deviation = NdArray.Zeros(observed.Length, 1);
        for (var i = 0; i < observed.Length; i++) deviation[i, 0] = values.At(i) - _mean[observed[i]];

        // Σ_uo Σ_oo⁻¹ (x_o − μ_o), computed as a solve rather than an inverse.
        var weights = LinAlg.Solve(sigmaOO, deviation);
        var shift = LinAlg.Dot(sigmaUO, weights);

        var mean = NdArray.Zeros(unknown.Length);
        for (var i = 0; i < unknown.Length; i++) mean.SetAt(i, _mean[unknown[i]] + shift[i, 0]);

        var correction = LinAlg.Dot(sigmaUO, LinAlg.Solve(sigmaOO, Transpose(sigmaUO)));
        var covariance = NdArray.Zeros(unknown.Length, unknown.Length);
        for (var i = 0; i < unknown.Length; i++)
            for (var j = 0; j < unknown.Length; j++)
                covariance[i, j] = sigmaUU[i, j] - correction[i, j];

        return new MultivariateNormal(mean, covariance);
    }

    private NdArray Submatrix(int[] rows, int[] columns)
    {
        var result = NdArray.Zeros(rows.Length, columns.Length);
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < columns.Length; j++)
                result[i, j] = _covariance[rows[i], columns[j]];
        return result;
    }

    private static NdArray Transpose(NdArray a)
    {
        var result = NdArray.Zeros(a.Shape[1], a.Shape[0]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[j, i] = a[i, j];
        return result;
    }
}

/// <summary>
/// The Dirichlet distribution: a distribution over probability vectors.
/// </summary>
/// <remarks>
/// <para>
/// A draw is a vector of non-negative numbers summing to one, which makes this the natural prior
/// over the parameters of a <see cref="Categorical"/> — mixing weights, topic proportions,
/// transition probabilities. It is the multivariate generalisation of the <see cref="Beta"/>, and
/// with two components it <em>is</em> a Beta: the first component of a Dirichlet(α₁, α₂) is
/// distributed exactly as Beta(α₁, α₂).
/// </para>
/// <para>
/// The concentration vector controls both location and spread. Its normalised value is the mean, and
/// its total governs how tightly draws cluster around that mean: large α gives vectors close to the
/// mean, α of all ones gives the uniform distribution over the simplex, and α below one pushes mass
/// into the corners — draws that are nearly one-hot. That last regime is what makes a sparse prior
/// sparse, and it is the reason "uninformative" is not the same as "α = 1" for every purpose.
/// </para>
/// </remarks>
public sealed class Dirichlet : MultivariateDistribution
{
    private readonly double[] _alpha;
    private readonly double _total;
    private readonly double _logNormaliser;

    /// <summary>Creates a Dirichlet from its concentration parameters, which must all be positive.</summary>
    public Dirichlet(params double[] alpha)
    {
        ArgumentNullException.ThrowIfNull(alpha);
        if (alpha.Length < 2) throw new ArgumentException("A Dirichlet needs at least two components.", nameof(alpha));

        foreach (var value in alpha)
            if (value <= 0 || double.IsNaN(value))
                throw new ArgumentException($"Concentration parameters must be positive; got {value}.", nameof(alpha));

        _alpha = (double[])alpha.Clone();
        _total = alpha.Sum();

        // The log multivariate beta function, which is the normalising constant.
        var logBeta = 0.0;
        foreach (var value in alpha) logBeta += MathUtil.LogGamma(value);
        _logNormaliser = MathUtil.LogGamma(_total) - logBeta;
    }

    /// <summary>A symmetric Dirichlet, the same concentration on every component.</summary>
    public static Dirichlet Symmetric(int dimension, double concentration = 1.0)
    {
        var alpha = new double[dimension];
        Array.Fill(alpha, concentration);
        return new Dirichlet(alpha);
    }

    /// <summary>Uniform over the simplex — every concentration one.</summary>
    public static Dirichlet Uniform(int dimension) => Symmetric(dimension);

    /// <inheritdoc />
    public override string Name => "Dirichlet";

    /// <inheritdoc />
    public override int Dimension => _alpha.Length;

    /// <summary>The concentration parameters.</summary>
    public IReadOnlyList<double> Alpha => _alpha;

    /// <summary>The sum of the concentrations — how tightly draws cluster around the mean.</summary>
    public double Concentration => _total;

    /// <inheritdoc />
    public override NdArray Mean
    {
        get
        {
            var result = NdArray.Zeros(Dimension);
            for (var i = 0; i < Dimension; i++) result.SetAt(i, _alpha[i] / _total);
            return result;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Every off-diagonal entry is negative, and necessarily so: the components sum to a constant,
    /// so one going up means another comes down. A Dirichlet cannot express positively correlated
    /// proportions, which is the main reason to reach for a logistic normal instead.
    /// </remarks>
    public override NdArray Covariance
    {
        get
        {
            var result = NdArray.Zeros(Dimension, Dimension);
            var denominator = _total * _total * (_total + 1);

            for (var i = 0; i < Dimension; i++)
                for (var j = 0; j < Dimension; j++)
                    result[i, j] = i == j
                        ? _alpha[i] * (_total - _alpha[i]) / denominator
                        : -_alpha[i] * _alpha[j] / denominator;

            return result;
        }
    }

    /// <inheritdoc />
    public override bool Supports(NdArray x)
    {
        if (x.Size != Dimension) return false;

        var sum = 0.0;
        for (var i = 0; i < Dimension; i++)
        {
            var value = x.At(i);
            if (double.IsNaN(value) || value < 0) return false;
            sum += value;
        }

        return Math.Abs(sum - 1.0) < 1e-9;
    }

    /// <inheritdoc />
    public override double LogDensity(NdArray x)
    {
        if (!Supports(x))
            throw new ArgumentException("A Dirichlet is only defined on the probability simplex.", nameof(x));

        var total = _logNormaliser;
        for (var i = 0; i < Dimension; i++)
        {
            var value = x.At(i);

            // A zero component has zero density unless its concentration is exactly one, where the
            // exponent vanishes and the term is well defined.
            if (value <= 0) return _alpha[i] > 1 ? double.NegativeInfinity : total;
            total += (_alpha[i] - 1) * Math.Log(value);
        }

        return total;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Draw one Gamma per component and normalise. This works because a Gamma is the unnormalised
    /// building block of a Dirichlet: independent Gamma(αᵢ, 1) variables divided by their sum are
    /// exactly Dirichlet(α), which is both the standard method and considerably simpler than any
    /// direct construction on the simplex.
    /// </remarks>
    public override NdArray Sample(GraviRandom rng)
    {
        var draws = new double[Dimension];
        var sum = 0.0;

        for (var i = 0; i < Dimension; i++)
        {
            draws[i] = new Gamma(_alpha[i], 1.0).Sample(rng);
            sum += draws[i];
        }

        var result = NdArray.Zeros(Dimension);
        for (var i = 0; i < Dimension; i++) result.SetAt(i, draws[i] / sum);
        return result;
    }

    /// <summary>
    /// The posterior after observing category counts.
    /// </summary>
    /// <remarks>
    /// The Dirichlet is conjugate to the categorical and multinomial, so the update is addition:
    /// the posterior is Dirichlet(α + counts). That is why it is the default prior for anything
    /// proportion-shaped — the whole inference is one vector sum.
    /// </remarks>
    public Dirichlet Posterior(IReadOnlyList<double> counts)
    {
        ArgumentNullException.ThrowIfNull(counts);
        if (counts.Count != Dimension)
            throw new ArgumentException($"Expected {Dimension} counts but got {counts.Count}.", nameof(counts));

        var updated = new double[Dimension];
        for (var i = 0; i < Dimension; i++)
        {
            if (counts[i] < 0) throw new ArgumentException("Counts cannot be negative.", nameof(counts));
            updated[i] = _alpha[i] + counts[i];
        }

        return new Dirichlet(updated);
    }

    /// <summary>The marginal distribution of one component, which is a Beta.</summary>
    /// <remarks>
    /// Every marginal of a Dirichlet is Beta(αᵢ, α₀ − αᵢ). Useful in its own right, and the
    /// cleanest independent check that the sampler and the density agree.
    /// </remarks>
    public Beta Marginal(int component)
    {
        if ((uint)component >= (uint)Dimension)
            throw new ArgumentOutOfRangeException(nameof(component));

        return new Beta(_alpha[component], _total - _alpha[component]);
    }
}

/// <summary>
/// The multinomial distribution: counts from repeated categorical draws.
/// </summary>
/// <remarks>
/// The generalisation of the <see cref="Binomial"/> to more than two outcomes, and with two
/// categories it is exactly a binomial. Its conjugate prior is the <see cref="Dirichlet"/>.
/// </remarks>
public sealed class Multinomial : MultivariateDistribution
{
    private readonly double[] _probabilities;

    /// <summary>Creates a multinomial.</summary>
    /// <param name="trials">Number of draws.</param>
    /// <param name="probabilities">Category probabilities; must be non-negative and sum to one.</param>
    public Multinomial(int trials, params double[] probabilities)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        if (trials < 0) throw new ArgumentOutOfRangeException(nameof(trials));
        if (probabilities.Length < 2)
            throw new ArgumentException("A multinomial needs at least two categories.", nameof(probabilities));

        var sum = 0.0;
        foreach (var p in probabilities)
        {
            if (p < 0 || double.IsNaN(p))
                throw new ArgumentException($"Probabilities must be non-negative; got {p}.", nameof(probabilities));
            sum += p;
        }

        if (Math.Abs(sum - 1.0) > 1e-9)
            throw new ArgumentException($"Probabilities must sum to one; they sum to {sum}.", nameof(probabilities));

        Trials = trials;
        _probabilities = (double[])probabilities.Clone();
    }

    /// <summary>Number of draws.</summary>
    public int Trials { get; }

    /// <summary>The category probabilities.</summary>
    public IReadOnlyList<double> Probabilities => _probabilities;

    /// <inheritdoc />
    public override string Name => "Multinomial";

    /// <inheritdoc />
    public override int Dimension => _probabilities.Length;

    /// <inheritdoc />
    public override NdArray Mean
    {
        get
        {
            var result = NdArray.Zeros(Dimension);
            for (var i = 0; i < Dimension; i++) result.SetAt(i, Trials * _probabilities[i]);
            return result;
        }
    }

    /// <inheritdoc />
    public override NdArray Covariance
    {
        get
        {
            var result = NdArray.Zeros(Dimension, Dimension);
            for (var i = 0; i < Dimension; i++)
                for (var j = 0; j < Dimension; j++)
                    result[i, j] = i == j
                        ? Trials * _probabilities[i] * (1 - _probabilities[i])
                        : -Trials * _probabilities[i] * _probabilities[j];
            return result;
        }
    }

    /// <inheritdoc />
    public override bool Supports(NdArray x)
    {
        if (x.Size != Dimension) return false;

        var sum = 0.0;
        for (var i = 0; i < Dimension; i++)
        {
            var value = x.At(i);
            if (value < 0 || Math.Abs(value - Math.Round(value)) > 1e-9) return false;
            sum += value;
        }

        return Math.Abs(sum - Trials) < 1e-9;
    }

    /// <inheritdoc />
    public override double LogDensity(NdArray x)
    {
        if (!Supports(x))
            throw new ArgumentException(
                $"A multinomial is defined on non-negative integer vectors summing to {Trials}.", nameof(x));

        var total = MathUtil.LogFactorial(Trials);
        for (var i = 0; i < Dimension; i++)
        {
            var count = (int)Math.Round(x.At(i));
            total -= MathUtil.LogFactorial(count);

            // A category with zero probability contributes nothing unless it was observed, in which
            // case the whole outcome is impossible.
            if (_probabilities[i] <= 0) { if (count > 0) return double.NegativeInfinity; continue; }
            total += count * Math.Log(_probabilities[i]);
        }

        return total;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Drawn as a chain of binomials on the remaining trials: the first category is Binomial(n, p₁),
    /// then the second is Binomial(n − k₁, p₂/(1 − p₁)) and so on. That keeps the counts summing to
    /// <see cref="Trials"/> exactly, where sampling each category independently would not.
    /// </remarks>
    public override NdArray Sample(GraviRandom rng)
    {
        var result = NdArray.Zeros(Dimension);
        var remaining = Trials;
        var remainingProbability = 1.0;

        for (var i = 0; i < Dimension - 1 && remaining > 0; i++)
        {
            if (remainingProbability <= 0) break;

            var p = Math.Clamp(_probabilities[i] / remainingProbability, 0.0, 1.0);
            var count = (int)new Binomial(remaining, p).Sample(rng);

            result.SetAt(i, count);
            remaining -= count;
            remainingProbability -= _probabilities[i];
        }

        result.SetAt(Dimension - 1, remaining);
        return result;
    }
}
