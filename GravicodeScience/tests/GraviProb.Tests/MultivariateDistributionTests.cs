using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Xunit;

namespace Gravicode.Science.Tests.GraviProb;

/// <summary>
/// Tests for the multivariate distributions.
/// </summary>
/// <remarks>
/// Each one reduces to a scalar distribution that is already tested independently — a Dirichlet's
/// marginal is a Beta, a two-category multinomial is a binomial, a one-dimensional multivariate
/// normal is a normal — so those reductions are the references rather than any number written down
/// here.
/// </remarks>
public class MultivariateDistributionTests
{
    // ---------------------------------------------------------------- Dirichlet

    [Fact]
    public void DirichletMeanIsTheNormalisedConcentration()
    {
        var dirichlet = new Dirichlet(2, 3, 5);
        var mean = dirichlet.Mean;

        Assert.Equal(0.2, mean.At(0), 12);
        Assert.Equal(0.3, mean.At(1), 12);
        Assert.Equal(0.5, mean.At(2), 12);
    }

    [Fact]
    public void ATwoComponentDirichletIsExactlyABeta()
    {
        // The cleanest available reference: Dirichlet(a, b) restricted to its first component has
        // the Beta(a, b) density, up to the Jacobian of dropping the redundant second component —
        // which is one, because the second component is determined.
        var dirichlet = new Dirichlet(2.5, 4.0);
        var beta = new Beta(2.5, 4.0);

        foreach (var x in new[] { 0.1, 0.35, 0.5, 0.8, 0.95 })
        {
            var point = NdArray.FromValues([x, 1 - x]);
            Assert.Equal(beta.LogDensity(x), dirichlet.LogDensity(point), 10);
        }
    }

    [Fact]
    public void EveryDirichletMarginalIsABeta()
    {
        var dirichlet = new Dirichlet(2, 3, 5);

        for (var i = 0; i < 3; i++)
        {
            var marginal = dirichlet.Marginal(i);
            Assert.Equal(dirichlet.Mean.At(i), marginal.Mean, 12);
            Assert.Equal(dirichlet.Covariance[i, i], marginal.Variance, 12);
        }
    }

    [Fact]
    public void DirichletDrawsLieOnTheSimplexAndMatchTheirMean()
    {
        var rng = new GraviRandom(3);
        var dirichlet = new Dirichlet(2, 3, 5);
        var draws = dirichlet.Sample(rng, 20000);

        for (var i = 0; i < 200; i++)
        {
            var sum = draws[i, 0] + draws[i, 1] + draws[i, 2];
            Assert.Equal(1.0, sum, 10);
            for (var j = 0; j < 3; j++) Assert.True(draws[i, j] >= 0);
        }

        for (var j = 0; j < 3; j++)
        {
            var mean = 0.0;
            for (var i = 0; i < 20000; i++) mean += draws[i, j];
            mean /= 20000;

            Assert.Equal(dirichlet.Mean.At(j), mean, 2);
        }
    }

    [Fact]
    public void DirichletSampleCovarianceMatchesTheClosedForm()
    {
        // Includes the negative off-diagonals, which are the structurally interesting part.
        var rng = new GraviRandom(5);
        var dirichlet = new Dirichlet(3, 4, 5);
        var draws = dirichlet.Sample(rng, 40000);

        var means = new double[3];
        for (var j = 0; j < 3; j++)
        {
            for (var i = 0; i < 40000; i++) means[j] += draws[i, j];
            means[j] /= 40000;
        }

        for (var a = 0; a < 3; a++)
            for (var b = 0; b < 3; b++)
            {
                var covariance = 0.0;
                for (var i = 0; i < 40000; i++) covariance += (draws[i, a] - means[a]) * (draws[i, b] - means[b]);
                covariance /= 40000;

                Assert.True(Math.Abs(covariance - dirichlet.Covariance[a, b]) < 5e-4,
                    $"covariance[{a},{b}] was {covariance:F6} against {dirichlet.Covariance[a, b]:F6}");
            }
    }

    [Fact]
    public void DirichletCovarianceIsNegativeOffTheDiagonal()
    {
        // Structural, not incidental: the components sum to a constant, so one rising means another
        // falling. A Dirichlet cannot represent positively correlated proportions at all.
        var covariance = new Dirichlet(2, 3, 5).Covariance;

        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                if (i != j) Assert.True(covariance[i, j] < 0, $"[{i},{j}] was {covariance[i, j]}");
    }

    [Fact]
    public void AUniformDirichletHasConstantDensityOnTheSimplex()
    {
        var dirichlet = Dirichlet.Uniform(3);

        var a = dirichlet.LogDensity(NdArray.FromValues([1 / 3.0, 1 / 3.0, 1 / 3.0]));
        var b = dirichlet.LogDensity(NdArray.FromValues([0.8, 0.1, 0.1]));

        Assert.Equal(a, b, 12);
        Assert.Equal(Math.Log(2.0), a, 12);      // the density is 1/area, and the 2-simplex has area 1/2
    }

    [Fact]
    public void TheDirichletPosteriorIsTheConcentrationPlusTheCounts()
    {
        // Conjugacy, which is the whole reason this is the standard prior for proportions.
        var posterior = new Dirichlet(1, 1, 1).Posterior([10.0, 5.0, 0.0]);

        Assert.Equal([11.0, 6.0, 1.0], posterior.Alpha.ToArray());
        Assert.Equal(11 / 18.0, posterior.Mean.At(0), 12);
    }

    [Fact]
    public void ADirichletRejectsWhatIsNotOnTheSimplex()
    {
        var dirichlet = new Dirichlet(1, 1, 1);

        Assert.False(dirichlet.Supports(NdArray.FromValues([0.5, 0.5, 0.5])));    // sums to 1.5
        Assert.False(dirichlet.Supports(NdArray.FromValues([-0.1, 0.6, 0.5])));   // negative
        Assert.False(dirichlet.Supports(NdArray.FromValues([0.5, 0.5])));         // wrong length

        Assert.Throws<ArgumentException>(() => dirichlet.LogDensity(NdArray.FromValues([1.0, 1.0, 1.0])));
        Assert.Throws<ArgumentException>(() => new Dirichlet(1, -1));
        Assert.Throws<ArgumentException>(() => new Dirichlet(1));
    }

    // -------------------------------------------------------- multivariate normal

    [Fact]
    public void AOneDimensionalMultivariateNormalIsANormal()
    {
        var mvn = new MultivariateNormal(NdArray.FromValues([2.0]), NdArray.Full(9.0, 1, 1));
        var normal = new Normal(2.0, 3.0);

        foreach (var x in new[] { -4.0, 0.0, 2.0, 5.5 })
            Assert.Equal(normal.LogDensity(x), mvn.LogDensity(NdArray.FromValues([x])), 10);
    }

    [Fact]
    public void ADiagonalCovarianceFactorsIntoIndependentNormals()
    {
        // With no correlation the joint log density is the sum of the marginals, which is an
        // independent statement of what a diagonal covariance means.
        var mvn = MultivariateNormal.Diagonal(
            NdArray.FromValues([1.0, -2.0]), NdArray.FromValues([4.0, 0.25]));

        var point = NdArray.FromValues([2.5, -1.0]);
        var expected = new Normal(1.0, 2.0).LogDensity(2.5) + new Normal(-2.0, 0.5).LogDensity(-1.0);

        Assert.Equal(expected, mvn.LogDensity(point), 10);
    }

    [Fact]
    public void TheDensityPeaksAtTheMean()
    {
        var mean = NdArray.FromValues([1.0, 2.0]);
        var covariance = NdArray.FromArray(new double[,] { { 2.0, 0.8 }, { 0.8, 1.0 } });
        var mvn = new MultivariateNormal(mean, covariance);

        var atMean = mvn.LogDensity(mean);
        foreach (var offset in new[] { 0.1, 0.5, 2.0 })
            Assert.True(mvn.LogDensity(NdArray.FromValues([1 + offset, 2.0])) < atMean);
    }

    [Fact]
    public void SamplesReproduceTheMeanAndCovariance()
    {
        // The check that the Cholesky factor is being applied the right way round: using Lᵀ instead
        // of L gives draws with the correct marginal variances and the wrong correlation.
        var mean = NdArray.FromValues([1.0, -2.0]);
        var covariance = NdArray.FromArray(new double[,] { { 4.0, 1.5 }, { 1.5, 2.0 } });
        var mvn = new MultivariateNormal(mean, covariance);

        var draws = mvn.Sample(new GraviRandom(7), 50000);

        var sampleMean = new double[2];
        for (var j = 0; j < 2; j++)
        {
            for (var i = 0; i < 50000; i++) sampleMean[j] += draws[i, j];
            sampleMean[j] /= 50000;
            Assert.Equal(mean.At(j), sampleMean[j], 1);
        }

        for (var a = 0; a < 2; a++)
            for (var b = 0; b < 2; b++)
            {
                var c = 0.0;
                for (var i = 0; i < 50000; i++) c += (draws[i, a] - sampleMean[a]) * (draws[i, b] - sampleMean[b]);
                c /= 50000;

                Assert.True(Math.Abs(c - covariance[a, b]) < 0.1,
                    $"covariance[{a},{b}] was {c:F4} against {covariance[a, b]:F4}");
            }
    }

    [Fact]
    public void ConditioningOnOneComponentGivesTheTextbookFormulas()
    {
        // For a bivariate normal the conditional mean is μ₁ + ρ(σ₁/σ₂)(x₂ − μ₂) and the conditional
        // variance is σ₁²(1 − ρ²) — both written down independently of the implementation.
        var mean = NdArray.FromValues([1.0, 2.0]);
        var (sigma1, sigma2, rho) = (2.0, 3.0, 0.6);

        var covariance = NdArray.FromArray(new double[,]
        {
            { sigma1 * sigma1, rho * sigma1 * sigma2 },
            { rho * sigma1 * sigma2, sigma2 * sigma2 },
        });

        var conditional = new MultivariateNormal(mean, covariance)
            .Conditional([0], [1], NdArray.FromValues([5.0]));

        Assert.Equal(1.0 + rho * (sigma1 / sigma2) * (5.0 - 2.0), conditional.Mean.At(0), 10);
        Assert.Equal(sigma1 * sigma1 * (1 - rho * rho), conditional.Covariance[0, 0], 10);
    }

    [Fact]
    public void ConditioningReducesUncertainty()
    {
        var covariance = NdArray.FromArray(new double[,] { { 4.0, 1.5 }, { 1.5, 2.0 } });
        var mvn = new MultivariateNormal(NdArray.Zeros(2), covariance);

        var conditional = mvn.Conditional([0], [1], NdArray.FromValues([1.0]));
        Assert.True(conditional.Covariance[0, 0] < covariance[0, 0]);
    }

    [Fact]
    public void ANonPositiveDefiniteCovarianceIsRejected()
    {
        // Perfectly correlated components make the density undefined; failing here beats returning
        // infinities from somewhere far away.
        var singular = NdArray.FromArray(new double[,] { { 1.0, 1.0 }, { 1.0, 1.0 } });
        Assert.Throws<ArgumentException>(() => new MultivariateNormal(NdArray.Zeros(2), singular));

        var negative = NdArray.FromArray(new double[,] { { -1.0, 0.0 }, { 0.0, 1.0 } });
        Assert.Throws<ArgumentException>(() => new MultivariateNormal(NdArray.Zeros(2), negative));
    }

    [Fact]
    public void AnAsymmetricCovarianceIsRejected()
    {
        var asymmetric = NdArray.FromArray(new double[,] { { 2.0, 0.5 }, { 0.9, 2.0 } });
        Assert.Throws<ArgumentException>(() => new MultivariateNormal(NdArray.Zeros(2), asymmetric));
    }

    // -------------------------------------------------------------- multinomial

    [Fact]
    public void ATwoCategoryMultinomialIsABinomial()
    {
        var multinomial = new Multinomial(10, 0.3, 0.7);
        var binomial = new Binomial(10, 0.3);

        for (var k = 0; k <= 10; k++)
            Assert.Equal(binomial.LogDensity(k), multinomial.LogDensity(NdArray.FromValues([k, 10 - k])), 10);
    }

    [Fact]
    public void MultinomialProbabilitiesSumToOneAcrossEveryOutcome()
    {
        // A normalising-constant error is invisible in any single density and obvious here.
        var multinomial = new Multinomial(4, 0.2, 0.3, 0.5);
        var total = 0.0;

        for (var a = 0; a <= 4; a++)
            for (var b = 0; a + b <= 4; b++)
                total += multinomial.Density(NdArray.FromValues([a, b, 4 - a - b]));

        Assert.Equal(1.0, total, 10);
    }

    [Fact]
    public void MultinomialDrawsAlwaysSumToTheTrialCount()
    {
        // Sampling each category independently would not guarantee this, which is why the sampler
        // walks a chain of binomials on the remaining trials.
        var rng = new GraviRandom(11);
        var multinomial = new Multinomial(20, 0.2, 0.3, 0.5);

        for (var i = 0; i < 500; i++)
        {
            var draw = multinomial.Sample(rng);
            var sum = 0.0;
            for (var j = 0; j < 3; j++)
            {
                Assert.True(draw.At(j) >= 0);
                sum += draw.At(j);
            }
            Assert.Equal(20.0, sum);
        }
    }

    [Fact]
    public void MultinomialDrawsMatchTheirMean()
    {
        var rng = new GraviRandom(13);
        var multinomial = new Multinomial(20, 0.2, 0.3, 0.5);
        var draws = multinomial.Sample(rng, 20000);

        for (var j = 0; j < 3; j++)
        {
            var mean = 0.0;
            for (var i = 0; i < 20000; i++) mean += draws[i, j];
            mean /= 20000;

            Assert.True(Math.Abs(mean - multinomial.Mean.At(j)) < 0.1,
                $"category {j} averaged {mean:F3} against {multinomial.Mean.At(j):F3}");
        }
    }

    [Fact]
    public void MultinomialRejectsMalformedInputs()
    {
        Assert.Throws<ArgumentException>(() => new Multinomial(5, 0.5, 0.4));       // does not sum to 1
        Assert.Throws<ArgumentException>(() => new Multinomial(5, 1.5, -0.5));      // negative
        Assert.Throws<ArgumentException>(() => new Multinomial(5, 1.0));            // one category

        var multinomial = new Multinomial(5, 0.5, 0.5);
        Assert.Throws<ArgumentException>(() => multinomial.LogDensity(NdArray.FromValues([2, 2])));   // sums to 4
        Assert.Throws<ArgumentException>(() => multinomial.LogDensity(NdArray.FromValues([2.5, 2.5])));
    }
}
