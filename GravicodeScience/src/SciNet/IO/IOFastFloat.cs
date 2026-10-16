using System.Buffers.Text;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Gravicode.Science.SciNet.IO;

/// <summary>
/// Correctly rounded decimal → double parsing for text formats, several times faster than
/// <see cref="Utf8Parser"/> (~100 ns per 17-digit value on .NET 10, which dominated Matrix Market
/// reading). Clinger's exact fast path handles short mantissas with small exponents; the
/// Eisel-Lemire algorithm (Lemire, "Number Parsing at a Gigabyte per Second", 2021; the
/// formulation of Go's <c>strconv</c>) handles the rest of the 19-digit range; anything it cannot
/// decide — more than 19 significant digits, exact halfway cases, subnormal or overflowing
/// results, inf/nan, Fortran D exponents — falls back to the BCL parser. The fallback is what
/// makes it safe: the fast paths either return the correctly rounded value or decline.
/// </summary>
internal static class IOFastFloat
{
    private const int MinExp10 = -348, MaxExp10 = 347;

    // 128-bit mantissas of 10^e, rounded down, normalised so the top bit is set: [hi, lo] per e.
    private static readonly ulong[] Pow10Table = BuildTable();

    private static readonly double[] ExactPow10 =
    [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
        1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
    ];

    private static ulong[] BuildTable()
    {
        var t = new ulong[(MaxExp10 - MinExp10 + 1) * 2];
        var two128 = BigInteger.One << 128;
        var two127 = BigInteger.One << 127;
        for (var e = MinExp10; e <= MaxExp10; e++)
        {
            BigInteger m;
            if (e >= 0)
            {
                var v = BigInteger.Pow(10, e);
                var bits = (int)v.GetBitLength();
                m = bits > 128 ? v >> (bits - 128) : v << (128 - bits);
            }
            else
            {
                var d = BigInteger.Pow(10, -e);
                var k = 128 + (int)d.GetBitLength() - 1;
                m = (BigInteger.One << k) / d;
                while (m >= two128) { k--; m = (BigInteger.One << k) / d; }
                while (m < two127) { k++; m = (BigInteger.One << k) / d; }
            }
            var i = (e - MinExp10) * 2;
            t[i] = (ulong)(m >> 64);
            t[i + 1] = (ulong)(m & ulong.MaxValue);
        }
        return t;
    }

    /// <summary>Parses a whole token; false when it is not a number.</summary>
    public static bool TryParse(ReadOnlySpan<byte> s, out double value)
    {
        if (TryFast(s, out value)) return true;
        // Not Utf8Parser: it misrounds exact halfway cases written with trailing zeros
        // ("17965106369622265.000" → ...266; ties-to-even gives ...264). double.TryParse is right.
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryFast(ReadOnlySpan<byte> s, out double value)
    {
        value = 0;
        var p = 0;
        var n = s.Length;
        if (n == 0) return false;
        var neg = false;
        if (s[0] == (byte)'-') { neg = true; p++; }
        else if (s[0] == (byte)'+') p++;
        ulong w = 0;
        var digits = 0; // significant digits accumulated (leading zeros excluded)
        var exp10 = 0;
        var any = false;
        while (p < n && (uint)(s[p] - '0') <= 9)
        {
            any = true;
            if (w != 0 || s[p] != (byte)'0')
            {
                if (digits == 19) return false; // too many digits for the fast paths
                w = w * 10 + (uint)(s[p] - '0');
                digits++;
            }
            p++;
        }
        if (p < n && s[p] == (byte)'.')
        {
            p++;
            while (p < n && (uint)(s[p] - '0') <= 9)
            {
                any = true;
                if (w != 0 || s[p] != (byte)'0')
                {
                    if (digits == 19) return false;
                    w = w * 10 + (uint)(s[p] - '0');
                    digits++;
                }
                exp10--;
                p++;
            }
        }
        if (!any) return false;
        if (p < n && (s[p] | 0x20) == (byte)'e')
        {
            p++;
            var eneg = false;
            if (p < n && s[p] == (byte)'-') { eneg = true; p++; }
            else if (p < n && s[p] == (byte)'+') p++;
            var start = p;
            var e = 0;
            while (p < n && (uint)(s[p] - '0') <= 9)
            {
                if (e < 100000) e = e * 10 + (s[p] - '0');
                p++;
            }
            if (p == start) return false;
            exp10 += eneg ? -e : e;
        }
        if (p != n) return false;

        if (w == 0)
        {
            value = neg ? -0.0 : 0.0;
            return true;
        }
        // Clinger: both w and 10^|q| are exact doubles, so one IEEE operation rounds correctly.
        if (w <= (1UL << 53) && exp10 is >= -22 and <= 22)
        {
            double d = w;
            d = exp10 < 0 ? d / ExactPow10[-exp10] : d * ExactPow10[exp10];
            value = neg ? -d : d;
            return true;
        }
        return EiselLemire(w, exp10, neg, out value);
    }

    private static bool EiselLemire(ulong man, int exp10, bool neg, out double f)
    {
        f = 0;
        if (exp10 < MinExp10 || exp10 > MaxExp10) return false;
        var clz = BitOperations.LeadingZeroCount(man);
        man <<= clz;
        var retExp2 = (ulong)(((217706L * exp10) >> 16) + 64 + 1023) - (ulong)clz;
        var idx = (exp10 - MinExp10) * 2;
        var hi = Math.BigMul(man, Pow10Table[idx], out var lo);
        if ((hi & 0x1FF) == 0x1FF && lo + man < man)
        {
            var yHi = Math.BigMul(man, Pow10Table[idx + 1], out var yLo);
            var mergedHi = hi;
            var mergedLo = lo + yHi;
            if (mergedLo < lo) mergedHi++;
            if ((mergedHi & 0x1FF) == 0x1FF && mergedLo + 1 == 0 && yLo + man < man) return false;
            hi = mergedHi;
            lo = mergedLo;
        }
        var msb = hi >> 63;
        var retMantissa = hi >> (int)(msb + 9);
        retExp2 -= 1 ^ msb;
        if (lo == 0 && (hi & 0x1FF) == 0 && (retMantissa & 3) == 1) return false; // exact halfway: let the slow path decide
        retMantissa += retMantissa & 1;
        retMantissa >>= 1;
        if (retMantissa >> 53 > 0)
        {
            retMantissa >>= 1;
            retExp2++;
        }
        if (retExp2 - 1 >= 0x7FF - 1) return false; // subnormal, zero or overflow: slow path
        var bits = (retExp2 << 52) | (retMantissa & 0x000FFFFFFFFFFFFFUL);
        if (neg) bits |= 0x8000000000000000UL;
        f = BitConverter.UInt64BitsToDouble(bits);
        return true;
    }
}
