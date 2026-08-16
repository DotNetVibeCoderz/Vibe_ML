namespace Gravicode.Science.GraviNum;

/// <summary>
/// A reproducible random number generator with the distributions the rest of the stack needs.
/// </summary>
/// <remarks>
/// The core is xoshiro256++ seeded through SplitMix64: fast, statistically solid and, crucially
/// for regression tests and benchmarks, byte-for-byte reproducible from a seed on every platform.
/// Sampling algorithms are the standard exact ones (Marsaglia-Tsang for gamma, recursive
/// beta-splitting for binomial, Knuth then transformed rejection for Poisson) rather than normal
/// approximations, so distributions stay correct in the tails.
/// </remarks>
public sealed class GraviRandom
{
    private ulong _s0, _s1, _s2, _s3;
    private double _cachedNormal;
    private bool _hasCachedNormal;

    /// <summary>Creates a generator seeded from <paramref name="seed"/>.</summary>
    public GraviRandom(int seed = 42) : this((ulong)seed) { }

    /// <summary>Creates a generator seeded from a 64-bit value.</summary>
    public GraviRandom(ulong seed)
    {
        Seed = seed;
        var state = seed;
        _s0 = SplitMix64(ref state);
        _s1 = SplitMix64(ref state);
        _s2 = SplitMix64(ref state);
        _s3 = SplitMix64(ref state);
    }

    /// <summary>The seed this generator was constructed with.</summary>
    public ulong Seed { get; }

    /// <summary>A shared, non-deterministic instance for casual use.</summary>
    public static GraviRandom Shared { get; } = new((ulong)Environment.TickCount64);

    private static ulong SplitMix64(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        var z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

    /// <summary>The raw 64-bit xoshiro256++ output.</summary>
    public ulong NextUInt64()
    {
        var result = Rotl(_s0 + _s3, 23) + _s0;
        var t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = Rotl(_s3, 45);
        return result;
    }

    /// <summary>A uniform double in <c>[0, 1)</c>.</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>A uniform integer in <c>[0, maxExclusive)</c>.</summary>
    public int Next(int maxExclusive)
    {
        if (maxExclusive <= 0) throw new ArgumentOutOfRangeException(nameof(maxExclusive));
        return (int)(NextDouble() * maxExclusive);
    }

    /// <summary>A uniform integer in <c>[min, maxExclusive)</c>.</summary>
    public int Next(int min, int maxExclusive) => min + Next(maxExclusive - min);

    /// <summary>A coin flip with probability <paramref name="p"/> of true.</summary>
    public bool NextBernoulli(double p) => NextDouble() < p;

    // ---------------------------------------------------------------- scalars

    /// <summary>A uniform double in <c>[low, high)</c>.</summary>
    public double Uniform(double low = 0.0, double high = 1.0) => low + (high - low) * NextDouble();

    /// <summary>A normal deviate, by the Marsaglia polar method (two per pair, one cached).</summary>
    public double Normal(double mean = 0.0, double stdDev = 1.0)
    {
        if (_hasCachedNormal)
        {
            _hasCachedNormal = false;
            return mean + stdDev * _cachedNormal;
        }

        double u, v, s;
        do
        {
            u = 2.0 * NextDouble() - 1.0;
            v = 2.0 * NextDouble() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        var factor = Math.Sqrt(-2.0 * Math.Log(s) / s);
        _cachedNormal = v * factor;
        _hasCachedNormal = true;
        return mean + stdDev * u * factor;
    }

    /// <summary>An exponential deviate with the given rate.</summary>
    public double Exponential(double rate = 1.0) => -Math.Log(1.0 - NextDouble()) / rate;

    /// <summary>A gamma deviate (shape/scale parameterisation), Marsaglia-Tsang.</summary>
    public double Gamma(double shape, double scale = 1.0)
    {
        if (shape <= 0) throw new ArgumentOutOfRangeException(nameof(shape), "Shape must be positive.");

        // Boost shapes below 1 into the valid range and correct with a power of a uniform.
        if (shape < 1.0)
            return Gamma(shape + 1.0, scale) * Math.Pow(NextDouble(), 1.0 / shape);

        var d = shape - 1.0 / 3.0;
        var c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = Normal();
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            var u = NextDouble();
            var x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2) return d * v * scale;
            if (Math.Log(u) < 0.5 * x2 + d * (1.0 - v + Math.Log(v))) return d * v * scale;
        }
    }

    /// <summary>A beta deviate, built from two gamma deviates.</summary>
    public double Beta(double alpha, double beta)
    {
        var x = Gamma(alpha);
        var y = Gamma(beta);
        return x + y == 0 ? 0.5 : x / (x + y);
    }

    /// <summary>A chi-squared deviate.</summary>
    public double ChiSquared(double degreesOfFreedom) => Gamma(degreesOfFreedom / 2.0, 2.0);

    /// <summary>A Student-t deviate.</summary>
    public double StudentT(double degreesOfFreedom)
        => Normal() / Math.Sqrt(ChiSquared(degreesOfFreedom) / degreesOfFreedom);

    /// <summary>A log-normal deviate.</summary>
    public double LogNormal(double mu = 0.0, double sigma = 1.0) => Math.Exp(Normal(mu, sigma));

    /// <summary>
    /// A binomial deviate. Small <paramref name="trials"/> use direct Bernoulli counting; larger
    /// ones use the exact recursive beta-splitting method, which is logarithmic in the trial count.
    /// </summary>
    public int Binomial(int trials, double probability)
    {
        if (trials < 0) throw new ArgumentOutOfRangeException(nameof(trials));
        if (probability <= 0) return 0;
        if (probability >= 1) return trials;
        if (trials == 0) return 0;

        if (trials < 30)
        {
            var count = 0;
            for (var i = 0; i < trials; i++) if (NextDouble() < probability) count++;
            return count;
        }

        var a = 1 + trials / 2;
        var b = trials - a + 1;
        var x = Beta(a, b);
        if (x >= probability)
            return Binomial(a - 1, probability / x);
        return a + Binomial(b - 1, (probability - x) / (1.0 - x));
    }

    /// <summary>
    /// A Poisson deviate: Knuth's product method for small means, transformed rejection above 30.
    /// </summary>
    public int Poisson(double lambda)
    {
        if (lambda <= 0) return 0;

        if (lambda < 30.0)
        {
            var limit = Math.Exp(-lambda);
            var product = 1.0;
            var k = 0;
            do
            {
                k++;
                product *= NextDouble();
            } while (product > limit);
            return k - 1;
        }

        // Hoermann's PTRS transformed rejection.
        var b = 0.931 + 2.53 * Math.Sqrt(lambda);
        var a = -0.059 + 0.02483 * b;
        var invAlpha = 1.1239 + 1.1328 / (b - 3.4);
        var vr = 0.9277 - 3.6224 / (b - 2.0);

        while (true)
        {
            var u = NextDouble() - 0.5;
            var v = NextDouble();
            var us = 0.5 - Math.Abs(u);
            var k = (int)Math.Floor((2.0 * a / us + b) * u + lambda + 0.43);

            if (us >= 0.07 && v <= vr) return k;
            if (k < 0 || (us < 0.013 && v > us)) continue;

            if (Math.Log(v * invAlpha / (a / (us * us) + b)) <=
                -lambda + k * Math.Log(lambda) - MathUtil.LogGamma(k + 1.0))
                return k;
        }
    }

    /// <summary>A geometric deviate: the number of failures before the first success.</summary>
    public int Geometric(double probability)
        => (int)Math.Floor(Math.Log(1.0 - NextDouble()) / Math.Log(1.0 - probability));

    /// <summary>Draws an index according to <paramref name="weights"/> (normalised internally).</summary>
    public int Categorical(ReadOnlySpan<double> weights)
    {
        var total = 0.0;
        foreach (var w in weights) total += w;
        var target = NextDouble() * total;
        var acc = 0.0;
        for (var i = 0; i < weights.Length; i++)
        {
            acc += weights[i];
            if (target < acc) return i;
        }
        return weights.Length - 1;
    }

    // ---------------------------------------------------------------- arrays

    /// <summary>An array of uniform values in <c>[0, 1)</c>.</summary>
    public NdArray Random(params int[] shape) => Build(shape, () => NextDouble());

    /// <summary>An array of uniform values in <c>[low, high)</c>.</summary>
    public NdArray Uniform(double low, double high, params int[] shape) => Build(shape, () => Uniform(low, high));

    /// <summary>An array of normal deviates.</summary>
    public NdArray Normal(double mean, double stdDev, params int[] shape) => Build(shape, () => Normal(mean, stdDev));

    /// <summary>An array of standard normal deviates.</summary>
    public NdArray StandardNormal(params int[] shape) => Build(shape, () => Normal());

    /// <summary>An array of binomial counts.</summary>
    public NdArray Binomial(int trials, double probability, params int[] shape)
        => Build(shape, () => Binomial(trials, probability));

    /// <summary>An array of Poisson counts.</summary>
    public NdArray Poisson(double lambda, params int[] shape) => Build(shape, () => Poisson(lambda));

    /// <summary>An array of gamma deviates.</summary>
    public NdArray Gamma(double shape, double scale, params int[] dims) => Build(dims, () => Gamma(shape, scale));

    /// <summary>An array of beta deviates.</summary>
    public NdArray Beta(double alpha, double beta, params int[] shape) => Build(shape, () => Beta(alpha, beta));

    /// <summary>An array of exponential deviates.</summary>
    public NdArray Exponential(double rate, params int[] shape) => Build(shape, () => Exponential(rate));

    /// <summary>An array of uniform integers in <c>[low, high)</c>, stored as doubles.</summary>
    public NdArray Integers(int low, int high, params int[] shape) => Build(shape, () => Next(low, high));

    private NdArray Build(int[] shape, Func<double> sampler)
    {
        if (shape.Length == 0) shape = [1];
        var data = new double[Shapes.Size(shape)];
        for (var i = 0; i < data.Length; i++) data[i] = sampler();
        return new NdArray(data, shape);
    }

    /// <summary>A random permutation of <c>[0, n)</c>.</summary>
    public int[] Permutation(int n)
    {
        var order = new int[n];
        for (var i = 0; i < n; i++) order[i] = i;
        Shuffle(order);
        return order;
    }

    /// <summary>Fisher-Yates shuffle, in place.</summary>
    public void Shuffle<T>(IList<T> items)
    {
        for (var i = items.Count - 1; i > 0; i--)
        {
            var j = Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    /// <summary>Draws <paramref name="count"/> indices from <c>[0, n)</c>.</summary>
    public int[] Choice(int n, int count, bool replace = true)
    {
        if (!replace)
        {
            if (count > n) throw new ArgumentException("Cannot draw more items than available without replacement.");
            return Permutation(n)[..count];
        }
        var result = new int[count];
        for (var i = 0; i < count; i++) result[i] = Next(n);
        return result;
    }

    /// <summary>
    /// Draws from a multivariate normal with the given mean and covariance,
    /// using the Cholesky factor of the covariance.
    /// </summary>
    public NdArray MultivariateNormal(NdArray mean, NdArray covariance, int samples = 1)
    {
        var d = mean.Size;
        var l = Decomposition.Cholesky(covariance);
        var result = NdArray.Zeros(samples, d);
        for (var s = 0; s < samples; s++)
        {
            var z = StandardNormal(d);
            for (var i = 0; i < d; i++)
            {
                var acc = mean.At(i);
                for (var j = 0; j <= i; j++) acc += l[i, j] * z.At(j);
                result[s, i] = acc;
            }
        }
        return samples == 1 ? result.Reshape(d) : result;
    }
}
