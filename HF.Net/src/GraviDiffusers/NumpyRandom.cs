namespace Gravicode.HFNet.GraviDiffusers;

/// <summary>
/// NumPy's legacy <c>RandomState</c>: MT19937 and the polar Gaussian, bit for bit.
/// </summary>
/// <remarks>
/// <para>
/// diffusers' ONNX pipelines draw their initial latents with
/// <c>np.random.RandomState(seed).randn(...)</c>. Reproducing that generator exactly is what makes a
/// seed mean the same thing here as in Python: the same seed, prompt and scheduler give the same
/// starting noise, and so the same image.
/// </para>
/// <para>
/// The integer seed goes through <c>init_genrand</c>, as <c>RandomState(int)</c> does. Doubles are
/// built from two 32-bit draws (27 and 26 bits), and <see cref="NextGaussian"/> is the Marsaglia
/// polar method, returning the second value of each pair first and caching the other - the order
/// <c>legacy_gauss</c> uses.
/// </para>
/// </remarks>
public sealed class NumpyRandom
{
    private const int StateLength = 624;
    private const int Period = 397;

    private readonly uint[] _state = new uint[StateLength];
    private int _position;
    private bool _hasGaussian;
    private double _gaussian;

    /// <summary>Seeds the generator as <c>np.random.RandomState(seed)</c> does.</summary>
    /// <param name="seed">A seed in <c>[0, 2^32)</c>.</param>
    public NumpyRandom(uint seed)
    {
        _state[0] = seed;
        for (var i = 1; i < StateLength; i++)
        {
            _state[i] = 1812433253u * (_state[i - 1] ^ (_state[i - 1] >> 30)) + (uint)i;
        }

        _position = StateLength;
    }

    /// <summary>The next 32-bit output.</summary>
    public uint NextUInt32()
    {
        if (_position >= StateLength) Twist();

        var y = _state[_position++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680u;
        y ^= (y << 15) & 0xEFC60000u;
        y ^= y >> 18;
        return y;
    }

    /// <summary>A double in <c>[0, 1)</c> with 53 random bits.</summary>
    public double NextDouble()
    {
        var a = NextUInt32() >> 5;
        var b = NextUInt32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    /// <summary>A standard normal draw, as <c>legacy_gauss</c> computes it.</summary>
    public double NextGaussian()
    {
        if (_hasGaussian)
        {
            _hasGaussian = false;
            return _gaussian;
        }

        double x1, x2, r2;
        do
        {
            x1 = 2.0 * NextDouble() - 1.0;
            x2 = 2.0 * NextDouble() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        var f = Math.Sqrt(-2.0 * Math.Log(r2) / r2);
        _gaussian = f * x1;
        _hasGaussian = true;
        return f * x2;
    }

    /// <summary><c>randn(count)</c>: standard normal draws in order.</summary>
    public double[] Normal(int count)
    {
        var values = new double[count];
        for (var i = 0; i < count; i++) values[i] = NextGaussian();
        return values;
    }

    private void Twist()
    {
        for (var i = 0; i < StateLength; i++)
        {
            var y = (_state[i] & 0x80000000u) | (_state[(i + 1) % StateLength] & 0x7FFFFFFFu);
            var next = _state[(i + Period) % StateLength] ^ (y >> 1);
            if ((y & 1) != 0) next ^= 0x9908B0DFu;
            _state[i] = next;
        }

        _position = 0;
    }
}
