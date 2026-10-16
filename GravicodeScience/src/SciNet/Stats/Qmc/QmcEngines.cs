using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Base class of the quasi-Monte Carlo engines (scipy's <c>qmc.QMCEngine</c>): draws points in the unit
/// hypercube [0, 1)^d and remembers how many it has produced, so successive calls continue the sequence.
/// </summary>
public abstract class QmcEngine
{
    /// <summary>Initialises the common state.</summary>
    protected QmcEngine(int d, int? seed)
    {
        if (d < 0) throw new ArgumentException("d must be a non-negative integer value", nameof(d));
        D = d;
        Seed = seed ?? System.Random.Shared.Next();
        Rng = new GraviRandom(Seed);
    }

    /// <summary>Dimension of the points.</summary>
    public int D { get; }

    /// <summary>Number of points already generated.</summary>
    public long NumGenerated { get; protected set; }

    /// <summary>The seed the engine was created (and is reset) with.</summary>
    protected int Seed { get; }

    /// <summary>The engine's random stream (scrambling, permutations).</summary>
    protected GraviRandom Rng { get; private set; }

    /// <summary>Draws the next <paramref name="n"/> points as an n×d array.</summary>
    public NdArray Random(int n = 1)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        return RandomCore(n);
    }

    /// <summary>Draws and counts <paramref name="n"/> points; engines with their own bookkeeping override it.</summary>
    protected virtual NdArray RandomCore(int n)
    {
        var buffer = new double[(long)n * D];
        Generate(n, buffer);
        NumGenerated += n;
        return new NdArray(buffer, n, D);
    }

    /// <summary>Writes the next <paramref name="n"/> points into <paramref name="destination"/> (row-major) without advancing the counter.</summary>
    protected abstract void Generate(int n, double[] destination);

    /// <summary>
    /// Integer points in [l, u) (or [l, u] with <paramref name="endpoint"/>), scipy's <c>integers</c>: QMC points
    /// scaled to the bounds and floored. With <paramref name="uBounds"/> null, the range is [0, lBounds).
    /// </summary>
    public long[,] Integers(long[] lBounds, long[]? uBounds = null, int n = 1, bool endpoint = false)
    {
        ArgumentNullException.ThrowIfNull(lBounds);
        if (uBounds is null) { uBounds = lBounds; lBounds = new long[uBounds.Length]; }
        if (lBounds.Length == 1 && D > 1) lBounds = Enumerable.Repeat(lBounds[0], D).ToArray();
        if (uBounds.Length == 1 && D > 1) uBounds = Enumerable.Repeat(uBounds[0], D).ToArray();
        if (lBounds.Length != D || uBounds.Length != D) throw new ArgumentException("bounds must have length d.");
        var sample = Random(n).AsContiguous().ToArray();
        var r = new long[n, D];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < D; j++)
            {
                var hi = uBounds[j] + (endpoint ? 1 : 0);
                if (hi <= lBounds[j]) throw new ArgumentException("Bounds are not consistent 'l_bounds' < 'u_bounds'");
                r[i, j] = (long)Math.Floor(sample[i * D + j] * (hi - lBounds[j]) + lBounds[j]);
            }
        return r;
    }

    /// <summary>Returns the engine to its initial state; scrambled engines regenerate the same scrambling.</summary>
    public virtual QmcEngine Reset()
    {
        Rng = new GraviRandom(Seed);
        NumGenerated = 0;
        Initialize();
        return this;
    }

    /// <summary>Skips <paramref name="n"/> points.</summary>
    public virtual QmcEngine FastForward(long n)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        if (n > 0) Random(checked((int)n));
        return this;
    }

    /// <summary>Recreates the engine's random structures (scrambles, permutations) from <see cref="Rng"/>.</summary>
    protected abstract void Initialize();
}

/// <summary>
/// Sobol' low-discrepancy sequence (scipy's <c>qmc.Sobol</c>) with Joe–Kuo direction numbers for up to
/// 4096 dimensions, optionally scrambled by Owen-type linear matrix scrambling plus a digital shift.
/// </summary>
/// <remarks>
/// <para>Unscrambled points reproduce scipy's bit for bit: point k is the XOR of the direction numbers
/// selected by the bits of the Gray code of k. Because any point can be computed directly from k, large
/// draws are split across threads and <see cref="FastForward"/> costs nothing.</para>
/// <para>The scrambled sequence uses this library's random stream, so it differs from scipy's for the
/// same seed while keeping the same (t, m, s)-net structure.</para>
/// </remarks>
public sealed class SobolEngine : QmcEngine
{
    private static readonly Lazy<(int[] Degree, uint[] Poly, uint[][] Init)> Table = new(DecodeTable);
    private readonly bool _scramble;
    private ulong[] _sv = [];   // d × bits
    private ulong[] _shift = [];

    /// <summary>Creates a Sobol' engine.</summary>
    /// <param name="d">Dimension, 1…4096.</param>
    /// <param name="scramble">Apply LMS + digital-shift scrambling (scipy's default).</param>
    /// <param name="bits">Bits of precision per coordinate, 1…64 (scipy's default 30); at most 2^bits points.</param>
    /// <param name="seed">Seed of the scrambling.</param>
    public SobolEngine(int d, bool scramble = true, int bits = 30, int? seed = null) : base(d, seed)
    {
        if (d > StatsSobolDirectionNumbers.MaxDim) throw new ArgumentException($"Maximum supported dimensionality is {StatsSobolDirectionNumbers.MaxDim}.", nameof(d));
        if (bits < 1 || bits > 64) throw new ArgumentException("Maximum supported 'bits' is 64", nameof(bits));
        Bits = bits;
        _scramble = scramble;
        Initialize();
    }

    /// <summary>Bits of precision.</summary>
    public int Bits { get; }

    /// <summary>Maximum number of points, 2^bits.</summary>
    public double MaxN => Math.Pow(2, Bits);

    /// <inheritdoc/>
    protected override void Initialize()
    {
        _sv = InitializeDirections(D, Bits);
        _shift = new ulong[D];
        if (_scramble) Scramble();
    }

    private static ulong[] InitializeDirections(int dim, int bits)
    {
        var v = new ulong[dim * bits];
        if (dim == 0) return v;
        var (deg, poly, init) = Table.Value;
        for (var i = 0; i < bits; i++) v[i] = 1;
        for (var d = 1; d < dim; d++)
        {
            var m = deg[d];
            ulong p = poly[d];
            var row = d * bits;
            for (var j = 0; j < Math.Min(m, bits); j++) v[row + j] = init[d][j];
            // Bratley & Fox (1988), Algorithm 659: vⱼ = 2a₁vⱼ₋₁ ⊕ 4a₂vⱼ₋₂ ⊕ … ⊕ 2^m vⱼ₋ₘ ⊕ vⱼ₋ₘ.
            for (var j = m; j < bits; j++)
            {
                var newv = v[row + j - m];
                ulong pow2 = 1;
                for (var k = 0; k < m; k++)
                {
                    pow2 <<= 1;
                    if (((p >> (m - 1 - k)) & 1) != 0) newv ^= pow2 * v[row + j - k - 1];
                }
                v[row + j] = newv;
            }
        }
        // Scale column j by 2^(bits − 1 − j) so the numbers become bits-wide binary fractions.
        for (var j = 0; j < bits; j++)
        {
            var sh = bits - 1 - j;
            for (var d = 0; d < dim; d++) v[d * bits + j] <<= sh;
        }
        return v;
    }

    private void Scramble()
    {
        // Digital shift: a random bits-wide integer per dimension.
        for (var d = 0; d < D; d++) _shift[d] = RandomBits(Bits);
        // Linear matrix scrambling: a random lower-triangular (in MSB-first digit order) binary matrix
        // with unit diagonal per dimension, applied to every direction number. Each output digit is the
        // input digit XOR a random parity of the more significant ones, which preserves the net structure.
        var masks = new ulong[Bits];
        for (var d = 0; d < D; d++)
        {
            for (var r = 0; r < Bits; r++)
            {
                var pos = Bits - 1 - r;                   // bit position of the r-th most significant digit
                var higher = r == 0 ? 0UL : RandomBits(r) << (pos + 1);
                masks[r] = higher;
            }
            for (var j = 0; j < Bits; j++)
            {
                var vdj = _sv[d * Bits + j];
                ulong outv = 0;
                for (var r = 0; r < Bits; r++)
                {
                    var pos = Bits - 1 - r;
                    var bit = (vdj >> pos) & 1;
                    bit ^= (ulong)(BitOperations.PopCount(vdj & masks[r]) & 1);
                    outv |= bit << pos;
                }
                _sv[d * Bits + j] = outv;
            }
        }
    }

    private ulong RandomBits(int count) => count >= 64 ? Rng.NextUInt64() : Rng.NextUInt64() & ((1UL << count) - 1);

    /// <summary>
    /// Draws 2^m points (scipy's <c>random_base2</c>), keeping the balance properties: the total number
    /// drawn so far must remain a power of two.
    /// </summary>
    public NdArray RandomBase2(int m)
    {
        if (m < 0) throw new ArgumentException("m must be non-negative.", nameof(m));
        var n = 1L << m;
        var total = NumGenerated + n;
        if ((total & (total - 1)) != 0)
            throw new ArgumentException($"The balance properties of Sobol' points require n to be a power of 2. {NumGenerated} points have been previously generated, then: n={NumGenerated}+2**{m}={total}. If you still want to do this, the function 'Sobol.random()' can be used.");
        return Random(checked((int)n));
    }

    /// <inheritdoc/>
    protected override void Generate(int n, double[] destination)
    {
        if (NumGenerated + n > MaxN) throw new ArgumentException($"At most 2**{Bits}={MaxN} distinct points can be generated. {NumGenerated} points have been previously generated, then: n={NumGenerated}+{n}={NumGenerated + n}. Consider increasing `bits`.");
        if (n == 0 || D == 0) return;
        var scale = Math.Pow(2, -Bits);
        var start = NumGenerated;
        var chunk = n >= 1 << 15 ? Math.Max(1 << 13, n / (Environment.ProcessorCount * 4)) : n;
        var chunks = (n + chunk - 1) / chunk;
        void Run(int c)
        {
            var lo = c * chunk;
            var hi = Math.Min(n, lo + chunk);
            // Point k = shift ⊕ (⊕ over set bits b of gray(k) of v[·, b]); then step along the Gray code.
            var quasi = new ulong[D];
            var k = (ulong)(start + lo);
            var gray = k ^ (k >> 1);
            for (var j = 0; j < D; j++)
            {
                var q = _shift[j];
                var g = gray;
                var b = 0;
                while (g != 0)
                {
                    if ((g & 1) != 0) q ^= _sv[j * Bits + b];
                    g >>= 1;
                    b++;
                }
                quasi[j] = q;
            }
            for (var i = lo; i < hi; i++)
            {
                var row = (long)i * D;
                for (var j = 0; j < D; j++) destination[row + j] = quasi[j] * scale;
                if (i + 1 < hi)
                {
                    var l = BitOperations.TrailingZeroCount(~k); // lowest zero bit of k: the Gray-code bit that flips
                    for (var j = 0; j < D; j++) quasi[j] ^= _sv[j * Bits + l];
                    k++;
                }
            }
        }
        if (chunks > 1) Parallel.For(0, chunks, Run);
        else Run(0);
    }

    /// <summary>Skips <paramref name="n"/> points in O(1): the next point is computed from its index.</summary>
    public override QmcEngine FastForward(long n)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        NumGenerated += n;
        return this;
    }

    private static (int[] Degree, uint[] Poly, uint[][] Init) DecodeTable()
    {
        var bytes = Convert.FromBase64String(StatsSobolDirectionNumbers.Packed);
        var max = StatsSobolDirectionNumbers.MaxDim;
        var deg = new int[max];
        var poly = new uint[max];
        var init = new uint[max][];
        init[0] = [];
        long bitPos = 0;
        uint Read(int width)
        {
            uint v = 0;
            for (var b = 0; b < width; b++, bitPos++)
                v |= (uint)((bytes[bitPos >> 3] >> (int)(bitPos & 7)) & 1) << b;
            return v;
        }
        for (var d = 1; d < max; d++)
        {
            var m = (int)Read(5);
            var inner = Read(m - 1);
            deg[d] = m;
            poly[d] = (1u << m) | (inner << 1) | 1u;
            init[d] = new uint[m];
            for (var j = 0; j < m; j++) init[d][j] = (Read(j) << 1) | 1u;
        }
        return (deg, poly, init);
    }
}

/// <summary>
/// Halton sequence (scipy's <c>qmc.Halton</c>): the van der Corput sequence in the i-th prime base for
/// dimension i, optionally scrambled with an independent random digit permutation per digit position
/// (Owen-type permutation scrambling, which removes the correlation between high dimensions).
/// </summary>
public sealed class HaltonEngine : QmcEngine
{
    private readonly bool _scramble;
    private readonly int[] _bases;
    private int[][]?[] _perms = [];

    /// <summary>Creates a Halton engine.</summary>
    public HaltonEngine(int d, bool scramble = true, int? seed = null) : base(d, seed)
    {
        _scramble = scramble;
        _bases = Qmc.FirstPrimes(d);
        Initialize();
    }

    /// <inheritdoc/>
    protected override void Initialize()
    {
        _perms = new int[D][][];
        if (!_scramble) return;
        for (var i = 0; i < D; i++)
        {
            var b = _bases[i];
            var count = (int)Math.Ceiling(54 / Math.Log2(b)) - 1;
            var digitPerms = new int[count][];
            _perms[i] = digitPerms;
            for (var k = 0; k < count; k++)
            {
                var p = Enumerable.Range(0, b).ToArray();
                for (var t = b - 1; t > 0; t--)
                {
                    var j = Rng.Next(t + 1);
                    (p[t], p[j]) = (p[j], p[t]);
                }
                digitPerms[k] = p;
            }
        }
    }

    /// <inheritdoc/>
    protected override void Generate(int n, double[] destination)
    {
        var start = NumGenerated;
        // Work items are (dimension, block of rows) so that even a handful of dimensions keeps every core busy.
        const int block = 8192;
        var blocks = (n + block - 1) / block;
        void Item(int t)
        {
            var j = t % D;
            var lo = t / D * block;
            var hi = Math.Min(n, lo + block);
            var b = _bases[j];
            var perms = _perms[j];
            for (var i = lo; i < hi; i++)
            {
                var q = start + i;
                var b2r = 1.0 / b;
                var s = 0.0;
                if (perms is null)
                {
                    while (q > 0)
                    {
                        var (quot, rem) = Math.DivRem(q, b);
                        s += rem * b2r;
                        b2r /= b;
                        q = quot;
                    }
                }
                else if (q <= int.MaxValue)
                {
                    var qi = (int)q;
                    for (var k = 0; k < perms.Length; k++)
                    {
                        var (quot, rem) = Math.DivRem(qi, b);
                        s += perms[k][rem] * b2r;
                        b2r /= b;
                        qi = quot;
                    }
                }
                else
                    for (var k = 0; k < perms.Length; k++)
                    {
                        var (quot, rem) = Math.DivRem(q, b);
                        s += perms[k][(int)rem] * b2r;
                        b2r /= b;
                        q = quot;
                    }
                destination[(long)i * D + j] = s;
            }
        }
        var items = blocks * D;
        if ((long)n * D >= 1 << 15) Parallel.For(0, items, Item);
        else for (var t = 0; t < items; t++) Item(t);
    }
}

/// <summary>
/// Latin hypercube sampling (scipy's <c>qmc.LatinHypercube</c>): each of the n strata of every axis holds
/// exactly one point. Strength 2 builds an orthogonal-array LHS (n = p², p prime, d ≤ p + 1), balanced
/// in every pair of dimensions too. <c>optimization: "random-cd"</c> swaps coordinates while the centred
/// discrepancy improves (Damblin et al. 2013), keeping the Latin hypercube property.
/// </summary>
public sealed class LatinHypercubeEngine : QmcEngine
{
    private readonly bool _scramble;
    private readonly int _strength;
    private readonly string? _optimization;

    /// <summary>Creates a Latin hypercube engine.</summary>
    /// <param name="d">Dimension.</param>
    /// <param name="scramble">Random positions within the strata (false puts points at the strata centres).</param>
    /// <param name="strength">1 (plain LHS) or 2 (orthogonal-array LHS).</param>
    /// <param name="optimization">null or <c>"random-cd"</c>.</param>
    /// <param name="seed">Seed.</param>
    public LatinHypercubeEngine(int d, bool scramble = true, int strength = 1, string? optimization = null, int? seed = null) : base(d, seed)
    {
        if (strength is not (1 or 2)) throw new ArgumentException("strength=" + strength + " is not supported. Strength must be 1 or 2.", nameof(strength));
        if (optimization is not (null or "random-cd")) throw new ArgumentException($"{optimization} is not a valid optimization method. It must be one of random-cd.", nameof(optimization));
        _scramble = scramble;
        _strength = strength;
        _optimization = optimization;
        Initialize();
    }

    /// <inheritdoc/>
    protected override void Initialize() { }

    /// <inheritdoc/>
    protected override void Generate(int n, double[] destination)
    {
        var s = _strength == 1 ? RandomLhs(n, D) : RandomOaLhs(n);
        if (_optimization == "random-cd") Qmc.RandomCdOptimize(s, n, D, Rng);
        Array.Copy(s, destination, s.Length);
    }

    private double[] RandomLhs(int n, int d)
    {
        var s = new double[n * d];
        for (var j = 0; j < d; j++)
        {
            var perm = Enumerable.Range(1, n).ToArray();
            for (var t = n - 1; t > 0; t--)
            {
                var k = Rng.Next(t + 1);
                (perm[t], perm[k]) = (perm[k], perm[t]);
            }
            for (var i = 0; i < n; i++)
            {
                var u = _scramble ? Rng.NextDouble() : 0.5;
                s[i * d + j] = (perm[i] - u) / n;
            }
        }
        return s;
    }

    private double[] RandomOaLhs(int n)
    {
        var p = (int)Math.Sqrt(n);
        if (p * p != n || !Qmc.IsPrime(p)) throw new ArgumentException("n is not the square of a prime number.", nameof(n));
        if (D > p + 1) throw new ArgumentException("n is too small for d. Must be n > (d-1)**2");
        var cols = p + 1;
        var oa = new int[n, cols];
        for (var i = 0; i < n; i++)
        {
            oa[i, 0] = i % p;          // meshgrid ordering as scipy: first column varies fastest
            oa[i, 1] = i / p;
            for (var q = 1; q < p; q++) oa[i, 1 + q] = (oa[i, 0] + q * oa[i, 1]) % p;
        }
        for (var j = 0; j < cols; j++)
        {
            var perm = Enumerable.Range(0, p).ToArray();
            for (var t = p - 1; t > 0; t--)
            {
                var k = Rng.Next(t + 1);
                (perm[t], perm[k]) = (perm[k], perm[t]);
            }
            for (var i = 0; i < n; i++) oa[i, j] = perm[oa[i, j]];
        }
        var outp = new double[n * D];
        for (var j = 0; j < D; j++)
            for (var level = 0; level < p; level++)
            {
                var lhs = RandomLhs(p, 1);
                var t = 0;
                for (var i = 0; i < n; i++)
                    if (oa[i, j] == level) outp[i * D + j] = (lhs[t++] + level) / p;
            }
        return outp;
    }
}

/// <summary>
/// Poisson disk sampling (scipy's <c>qmc.PoissonDisk</c>, Bridson's algorithm): points at least
/// <c>radius</c> apart, grown from random candidates in a hypersphere around existing points. Fewer than
/// n points come back once the space is full; <see cref="FillSpace"/> keeps going until it is.
/// </summary>
public sealed class PoissonDiskEngine : QmcEngine
{
    private readonly double _radius;
    private readonly bool _volume;
    private readonly int _ncandidates;
    private readonly double[] _lower;
    private readonly double[] _upper;
    private readonly double _cell;
    private readonly int[] _gridSize;
    private List<double[]> _pool = [];
    private Dictionary<long, double[]> _grid = [];

    /// <summary>Creates a Poisson disk engine.</summary>
    public PoissonDiskEngine(int d, double radius = 0.05, string hypersphere = "volume", int ncandidates = 30, int? seed = null,
        double[]? lBounds = null, double[]? uBounds = null) : base(d, seed)
    {
        if (!(radius > 0)) throw new ArgumentException("radius must be positive.", nameof(radius));
        if (hypersphere is not ("volume" or "surface")) throw new ArgumentException($"'{hypersphere}' is not a valid hypersphere sampling method. It must be one of volume, surface.", nameof(hypersphere));
        _radius = radius;
        _volume = hypersphere == "volume";
        _ncandidates = ncandidates;
        _lower = lBounds ?? new double[d];
        _upper = uBounds ?? Enumerable.Repeat(1.0, d).ToArray();
        if (_lower.Length != d || _upper.Length != d) throw new ArgumentException("bounds must have length d.");
        for (var i = 0; i < d; i++) if (!(_lower[i] < _upper[i])) throw new ArgumentException("Bounds are not consistent 'l_bounds' < 'u_bounds'");
        _cell = radius / Math.Sqrt(d);
        _gridSize = new int[d];
        for (var i = 0; i < d; i++) _gridSize[i] = (int)Math.Ceiling((_upper[i] - _lower[i]) / _cell);
        Initialize();
    }

    /// <inheritdoc/>
    protected override void Initialize()
    {
        _pool = [];
        _grid = [];
    }

    private long CellKey(ReadOnlySpan<int> idx)
    {
        long key = 0;
        for (var i = 0; i < D; i++) key = key * (_gridSize[i] + 1) + idx[i];
        return key;
    }

    private int[] CellOf(double[] p)
    {
        var idx = new int[D];
        for (var i = 0; i < D; i++) idx[i] = Math.Min((int)((p[i] - _lower[i]) / _cell), _gridSize[i] - 1);
        return idx;
    }

    private bool TooClose(double[] c)
    {
        var idx = CellOf(c);
        var lo = new int[D];
        var hi = new int[D];
        for (var i = 0; i < D; i++) { lo[i] = Math.Max(idx[i] - 2, 0); hi[i] = Math.Min(idx[i] + 2, _gridSize[i] - 1); }
        var cur = (int[])lo.Clone();
        var r2 = _radius * _radius;
        while (true)
        {
            if (_grid.TryGetValue(CellKey(cur), out var q))
            {
                var s = 0.0;
                for (var i = 0; i < D; i++) s += (c[i] - q[i]) * (c[i] - q[i]);
                if (s < r2) return true;
            }
            var k = D - 1;
            while (k >= 0 && ++cur[k] > hi[k]) { cur[k] = lo[k]; k--; }
            if (k < 0) return false;
        }
    }

    private void Add(double[] c, List<double[]> output)
    {
        _pool.Add(c);
        _grid[CellKey(CellOf(c))] = c;
        output.Add(c);
    }

    /// <summary>Draws up to <paramref name="n"/> new points (fewer when the space is full).</summary>
    protected override NdArray RandomCore(int n) => Draw(n);

    /// <summary>Adds points until no candidate fits anywhere.</summary>
    public NdArray FillSpace() => Draw(int.MaxValue);

    private NdArray Draw(int n)
    {
        var output = new List<double[]>();
        if (n <= 0 || D == 0) return new NdArray(new double[0], 0, D);
        var drawn = 0;
        if (_pool.Count == 0 && _grid.Count == 0)
        {
            var first = new double[D];
            for (var i = 0; i < D; i++) first[i] = _lower[i] + (_upper[i] - _lower[i]) * Rng.NextDouble();
            Add(first, output);
            drawn = 1;
        }
        while (_pool.Count > 0 && drawn < n)
        {
            var ci = Rng.Next(_pool.Count);
            var center = _pool[ci];
            _pool.RemoveAt(ci);
            for (var c = 0; c < _ncandidates && drawn < n; c++)
            {
                var cand = Candidate(center);
                var inside = true;
                for (var i = 0; i < D; i++) if (cand[i] > _upper[i] || cand[i] < _lower[i]) { inside = false; break; }
                if (!inside || TooClose(cand)) continue;
                Add(cand, output);
                drawn++;
            }
        }
        NumGenerated += drawn;
        var data = new double[output.Count * D];
        for (var i = 0; i < output.Count; i++) Array.Copy(output[i], 0, data, i * D, D);
        return new NdArray(data, output.Count, D);
    }

    private double[] Candidate(double[] center)
    {
        var x = new double[D];
        var ssq = 0.0;
        for (var i = 0; i < D; i++) { x[i] = Rng.Normal(); ssq += x[i] * x[i]; }
        double f;
        if (_volume)
        {
            // Uniform in the ball of radius 2r: the radial CDF is the regularised incomplete gamma of |x|²/2.
            var radius = 2 * _radius;
            f = radius * Math.Pow(Special.GammaInc(D / 2.0, ssq / 2), 1.0 / D) / Math.Sqrt(ssq);
        }
        else f = 1.001 * _radius / Math.Sqrt(ssq);
        var c = new double[D];
        for (var i = 0; i < D; i++) c[i] = center[i] + x[i] * f;
        return c;
    }

    /// <inheritdoc/>
    protected override void Generate(int n, double[] destination) => throw new NotSupportedException("PoissonDiskEngine draws through Random/FillSpace.");
}
