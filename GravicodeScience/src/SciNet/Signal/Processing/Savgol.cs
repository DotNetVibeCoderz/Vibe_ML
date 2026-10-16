using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Whether <see cref="Signal.SavgolCoeffs"/> returns convolution or dot-product order.</summary>
public enum SavgolUse
{
    /// <summary>Reversed, for use with a convolution.</summary>
    Conv,
    /// <summary>In window order, for a dot product with the window samples.</summary>
    Dot,
}

/// <summary>Edge handling for <see cref="Signal.SavgolFilter(double[], int, int, int, double, SavgolMode, double)"/>.</summary>
public enum SavgolMode
{
    /// <summary>Fit a polynomial to the first/last window and evaluate it at the edge samples (scipy's default).</summary>
    Interp,
    /// <summary>Reflect about the edge sample (<c>c b | a b c</c>), ndimage's "mirror".</summary>
    Mirror,
    /// <summary>Repeat the edge sample.</summary>
    Nearest,
    /// <summary>Pad with <c>cval</c>.</summary>
    Constant,
    /// <summary>Periodic extension.</summary>
    Wrap,
}

public static partial class Signal
{
    /// <summary>
    /// Savitzky–Golay filter coefficients (<c>scipy.signal.savgol_coeffs</c>): the weights that
    /// evaluate the <paramref name="deriv"/>-th derivative of the least-squares polynomial fit at
    /// <paramref name="pos"/>.
    /// </summary>
    /// <remarks>
    /// The system <c>A c = e_d · d!/Δ^d</c> (A is the Vandermonde matrix of the window positions,
    /// one row per power) is underdetermined, and the minimum-norm solution is taken, as scipy's
    /// <c>lstsq</c> does — computed by Householder QR of <c>Aᵀ</c> after scaling each power by the
    /// half-width, which keeps the Vandermonde system far better conditioned than the raw powers.
    /// </remarks>
    /// <param name="windowLength">Window length.</param>
    /// <param name="polyorder">Polynomial order, less than <paramref name="windowLength"/>.</param>
    /// <param name="deriv">Derivative order (0 = smoothing).</param>
    /// <param name="delta">Sample spacing, used only when <paramref name="deriv"/> &gt; 0.</param>
    /// <param name="pos">Evaluation point within the window; default the centre.</param>
    /// <param name="use">Convolution (reversed) or dot-product order.</param>
    public static double[] SavgolCoeffs(int windowLength, int polyorder, int deriv = 0, double delta = 1.0, double? pos = null, SavgolUse use = SavgolUse.Conv)
    {
        if (windowLength < 1) throw new ArgumentException("window_length must be positive.", nameof(windowLength));
        if (polyorder < 0) throw new ArgumentException("polyorder must be non-negative.", nameof(polyorder));
        if (deriv < 0) throw new ArgumentException("deriv must be non-negative.", nameof(deriv));
        if (polyorder >= windowLength) throw new ArgumentException("polyorder must be less than window_length.", nameof(polyorder));
        var half = windowLength / 2;
        var p = pos ?? (windowLength % 2 == 0 ? half - 0.5 : half);
        if (!(p >= 0 && p < windowLength)) throw new ArgumentException("pos must be nonnegative and less than window_length.", nameof(pos));
        var w = windowLength;
        if (deriv > polyorder) return new double[w];
        var x = new double[w];
        for (var i = 0; i < w; i++) x[i] = i - p;
        if (use == SavgolUse.Conv) Array.Reverse(x);
        var scale = 0.0;
        foreach (var v in x) scale = Math.Max(scale, Math.Abs(v));
        if (scale == 0) scale = 1;
        var cols = polyorder + 1;
        // M = (D A)ᵀ, w × cols, column-major in a jagged layout: M[j][i] = (x_i / s)^j.
        var m = new double[cols][];
        for (var j = 0; j < cols; j++)
        {
            m[j] = new double[w];
            for (var i = 0; i < w; i++) m[j][i] = Math.Pow(x[i] / scale, j);
        }
        // Right-hand side D y: only entry `deriv` is non-zero.
        var rhs = new double[cols];
        rhs[deriv] = Factorial(deriv) / Math.Pow(delta, deriv) / Math.Pow(scale, deriv);
        return MinNormSolve(m, w, rhs);
    }

    private static double Factorial(int n)
    {
        var f = 1.0;
        for (var i = 2; i <= n; i++) f *= i;
        return f;
    }

    /// <summary>
    /// Minimum-norm solution of <c>Mᵀ c = r</c> with M (rows × cols, cols ≤ rows) given by columns:
    /// QR-factor M = QR, then c = Q R⁻ᵀ r.
    /// </summary>
    private static double[] MinNormSolve(double[][] mCols, int rows, double[] r)
    {
        var cols = mCols.Length;
        var a = mCols.Select(c => (double[])c.Clone()).ToArray();
        var vs = new double[cols][];
        var rdiag = new double[cols];
        for (var k = 0; k < cols; k++)
        {
            var col = a[k];
            double norm = 0;
            for (var i = k; i < rows; i++) norm += col[i] * col[i];
            norm = Math.Sqrt(norm);
            var alpha = col[k] > 0 ? -norm : norm;
            var v = new double[rows];
            for (var i = k; i < rows; i++) v[i] = col[i];
            v[k] -= alpha;
            double vn = 0;
            for (var i = k; i < rows; i++) vn += v[i] * v[i];
            if (vn > 0)
                for (var j = k; j < cols; j++)
                {
                    double d = 0;
                    for (var i = k; i < rows; i++) d += v[i] * a[j][i];
                    var f = 2 * d / vn;
                    for (var i = k; i < rows; i++) a[j][i] -= f * v[i];
                }
            vs[k] = v;
            rdiag[k] = a[k][k];
            if (rdiag[k] == 0) throw new ArgumentException("Savitzky–Golay system is rank deficient.");
        }
        // Solve Rᵀ z = r (forward substitution); R[i][j] = a[j][i] for i ≤ j.
        var z = new double[cols];
        for (var i = 0; i < cols; i++)
        {
            var s = r[i];
            for (var j = 0; j < i; j++) s -= a[i][j] * z[j];
            z[i] = s / a[i][i];
        }
        // c = Q [z; 0] = H_0 H_1 … H_{k-1} [z; 0].
        var c = new double[rows];
        Array.Copy(z, c, cols);
        for (var k = cols - 1; k >= 0; k--)
        {
            var v = vs[k];
            double vn = 0, d = 0;
            for (var i = k; i < rows; i++) { vn += v[i] * v[i]; d += v[i] * c[i]; }
            if (vn == 0) continue;
            var f = 2 * d / vn;
            for (var i = k; i < rows; i++) c[i] -= f * v[i];
        }
        return c;
    }

    /// <summary>
    /// Savitzky–Golay smoothing or differentiation (<c>scipy.signal.savgol_filter</c>).
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="windowLength">Window length (≤ len x for <see cref="SavgolMode.Interp"/>).</param>
    /// <param name="polyorder">Polynomial order.</param>
    /// <param name="deriv">Derivative order.</param>
    /// <param name="delta">Sample spacing for derivatives.</param>
    /// <param name="mode">Edge handling.</param>
    /// <param name="cval">Pad value for <see cref="SavgolMode.Constant"/>.</param>
    public static double[] SavgolFilter(double[] x, int windowLength, int polyorder, int deriv = 0, double delta = 1.0,
        SavgolMode mode = SavgolMode.Interp, double cval = 0.0)
    {
        ArgumentNullException.ThrowIfNull(x);
        var coeffs = SavgolCoeffs(windowLength, polyorder, deriv, delta);
        var y = new double[x.Length];
        SavgolLine(x, y, coeffs, windowLength, polyorder, deriv, delta, mode, cval);
        return y;
    }

    /// <summary><see cref="SavgolFilter(double[], int, int, int, double, SavgolMode, double)"/> along one axis.</summary>
    public static NdArray SavgolFilter(NdArray x, int windowLength, int polyorder, int deriv = 0, double delta = 1.0, int axis = -1,
        SavgolMode mode = SavgolMode.Interp, double cval = 0.0)
    {
        var coeffs = SavgolCoeffs(windowLength, polyorder, deriv, delta);
        var n = SignalAxisLines.Length(x, axis);
        return SignalAxisLines.Map(x, axis, n, (s, d) => SavgolLine(s, d, coeffs, windowLength, polyorder, deriv, delta, mode, cval));
    }

    private static void SavgolLine(ReadOnlySpan<double> x, Span<double> y, double[] coeffs, int wl, int polyorder, int deriv, double delta, SavgolMode mode, double cval)
    {
        var n = x.Length;
        if (n == 0) return;
        if (mode == SavgolMode.Interp && wl > n)
            throw new ArgumentException("If mode is Interp, window_length must be less than or equal to the size of x.");
        var padMode = mode switch
        {
            SavgolMode.Mirror => UpfirdnPad.Reflect,
            SavgolMode.Nearest => UpfirdnPad.Edge,
            SavgolMode.Wrap => UpfirdnPad.Wrap,
            _ => UpfirdnPad.Constant,
        };
        var cv = mode == SavgolMode.Constant ? cval : 0.0;
        // ndimage.convolve1d: y[i] = Σ_k w[k] x[i + L/2 − k], i.e. a dot product of the padded
        // window starting at i with the reversed weights.
        var left = wl - 1 - wl / 2;
        var right = wl / 2;
        var xp = new double[n + left + right];
        for (var i = 0; i < left; i++) xp[i] = SignalPadding.Extend(x, i - left, padMode, cv);
        x.CopyTo(xp.AsSpan(left));
        for (var i = 0; i < right; i++) xp[left + n + i] = SignalPadding.Extend(x, n + i, padMode, cv);
        var wr = SignalConvKernels.Reversed(coeffs);
        for (var i = 0; i < n; i++) y[i] = SignalConvKernels.Dot(xp.AsSpan(i, wl), wr);
        if (mode != SavgolMode.Interp) return;
        // Edges: evaluate the polynomial fitted to the first/last window — a fixed linear map of
        // the window, so it is a dot product with savgol_coeffs at the shifted position.
        var half = wl / 2;
        for (var i = 0; i < half; i++)
        {
            var c = SavgolCoeffs(wl, polyorder, deriv, delta, pos: i, use: SavgolUse.Dot);
            y[i] = SignalConvKernels.Dot(x[..wl], c);
            var c2 = SavgolCoeffs(wl, polyorder, deriv, delta, pos: wl - half + i, use: SavgolUse.Dot);
            y[n - half + i] = SignalConvKernels.Dot(x[(n - wl)..], c2);
        }
    }
}
