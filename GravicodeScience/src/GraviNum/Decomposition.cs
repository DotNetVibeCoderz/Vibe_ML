namespace Gravicode.Science.GraviNum;

/// <summary>LU factorisation with partial pivoting: <c>P A = L U</c>.</summary>
/// <param name="Lower">Unit lower-triangular factor.</param>
/// <param name="Upper">Upper-triangular factor.</param>
/// <param name="Pivot">Row permutation; <c>Pivot[i]</c> is the source row of output row <c>i</c>.</param>
/// <param name="Sign">Determinant sign contributed by the row swaps (+1 or -1).</param>
public sealed record LuResult(NdArray Lower, NdArray Upper, int[] Pivot, double Sign)
{
    /// <summary>Solves <c>A x = b</c> by forward then back substitution. <paramref name="b"/> may be a matrix.</summary>
    public NdArray Solve(NdArray b)
    {
        var n = Upper.Shape[0];
        var asMatrix = b.Rank == 2 ? b : b.Reshape(b.Size, 1);
        if (asMatrix.Shape[0] != n)
            throw new InvalidOperationException($"Right-hand side has {asMatrix.Shape[0]} rows, expected {n}.");

        var columns = asMatrix.Shape[1];
        var x = NdArray.Zeros(n, columns);

        for (var c = 0; c < columns; c++)
        {
            // Forward substitution through L, applying the pivot as we read b.
            var y = new double[n];
            for (var i = 0; i < n; i++)
            {
                var acc = asMatrix[Pivot[i], c];
                for (var j = 0; j < i; j++) acc -= Lower[i, j] * y[j];
                y[i] = acc;
            }

            // Back substitution through U.
            for (var i = n - 1; i >= 0; i--)
            {
                var acc = y[i];
                for (var j = i + 1; j < n; j++) acc -= Upper[i, j] * x[j, c];
                var pivot = Upper[i, i];
                if (pivot == 0.0)
                    throw new InvalidOperationException("Matrix is singular to working precision.");
                x[i, c] = acc / pivot;
            }
        }

        return b.Rank == 2 ? x : x.Reshape(n);
    }

    /// <summary>The permutation matrix <c>P</c>.</summary>
    public NdArray PermutationMatrix()
    {
        var n = Pivot.Length;
        var p = NdArray.Zeros(n, n);
        for (var i = 0; i < n; i++) p[i, Pivot[i]] = 1.0;
        return p;
    }
}

/// <summary>QR factorisation <c>A = Q R</c> with orthonormal <c>Q</c>.</summary>
public sealed record QrResult(NdArray Q, NdArray R)
{
    /// <summary>Least-squares solve of <c>A x ~ b</c> using the factors.</summary>
    public NdArray Solve(NdArray b)
    {
        var qtb = LinAlg.Dot(Q.T, b);
        var n = R.Shape[1];
        var asMatrix = qtb.Rank == 2 ? qtb : qtb.Reshape(qtb.Size, 1);
        var columns = asMatrix.Shape[1];
        var x = NdArray.Zeros(n, columns);

        for (var c = 0; c < columns; c++)
            for (var i = n - 1; i >= 0; i--)
            {
                var acc = asMatrix[i, c];
                for (var j = i + 1; j < n; j++) acc -= R[i, j] * x[j, c];
                x[i, c] = R[i, i] == 0 ? 0 : acc / R[i, i];
            }

        return b.Rank == 2 ? x : x.Reshape(n);
    }
}

/// <summary>Singular value decomposition <c>A = U diag(S) V^T</c>, singular values descending.</summary>
public sealed record SvdResult(NdArray U, NdArray SingularValues, NdArray V)
{
    /// <summary>Rebuilds the original matrix from the factors; useful as a sanity check.</summary>
    public NdArray Reconstruct()
    {
        var k = SingularValues.Size;
        var s = NdArray.Zeros(k, k);
        for (var i = 0; i < k; i++) s[i, i] = SingularValues.At(i);
        return LinAlg.Dot(LinAlg.Dot(U, s), V.T);
    }
}

/// <summary>Eigen decomposition of a symmetric matrix, eigenvalues descending.</summary>
/// <param name="Values">Eigenvalues, largest first.</param>
/// <param name="Vectors">Eigenvectors as columns, aligned with <paramref name="Values"/>.</param>
public sealed record EigenResult(NdArray Values, NdArray Vectors);

/// <summary>
/// Matrix factorisations: LU, QR, Cholesky, SVD and eigen decomposition.
/// </summary>
/// <remarks>
/// Everything here is a managed implementation with no native dependency, chosen for numerical
/// robustness over raw speed: Householder reflections for QR, one-sided Jacobi for SVD, cyclic
/// Jacobi for symmetric eigenproblems and Hessenberg reduction plus shifted QR for general ones.
/// </remarks>
public static class Decomposition
{
    /// <summary>LU factorisation with partial pivoting.</summary>
    public static LuResult Lu(NdArray a)
    {
        LinAlg.RequireSquare(a, nameof(Lu));
        var n = a.Shape[0];
        var work = a.Copy();
        var pivot = new int[n];
        for (var i = 0; i < n; i++) pivot[i] = i;
        var sign = 1.0;

        for (var col = 0; col < n; col++)
        {
            // Partial pivoting: pull the largest magnitude entry onto the diagonal.
            var best = col;
            var bestValue = Math.Abs(work[col, col]);
            for (var row = col + 1; row < n; row++)
            {
                var v = Math.Abs(work[row, col]);
                if (v > bestValue) { bestValue = v; best = row; }
            }

            if (best != col)
            {
                for (var j = 0; j < n; j++) (work[col, j], work[best, j]) = (work[best, j], work[col, j]);
                (pivot[col], pivot[best]) = (pivot[best], pivot[col]);
                sign = -sign;
            }

            var diag = work[col, col];
            if (diag == 0.0) continue;

            for (var row = col + 1; row < n; row++)
            {
                var factor = work[row, col] / diag;
                work[row, col] = factor;
                for (var j = col + 1; j < n; j++) work[row, j] -= factor * work[col, j];
            }
        }

        var lower = NdArray.Eye(n);
        var upper = NdArray.Zeros(n, n);
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < i; j++) lower[i, j] = work[i, j];
            for (var j = i; j < n; j++) upper[i, j] = work[i, j];
        }
        return new LuResult(lower, upper, pivot, sign);
    }

    /// <summary>
    /// QR factorisation via Householder reflections. With <paramref name="reduced"/> the
    /// factors are <c>m×n</c> and <c>n×n</c>; otherwise <c>m×m</c> and <c>m×n</c>.
    /// </summary>
    public static QrResult Qr(NdArray a, bool reduced = true)
    {
        if (a.Rank != 2) throw new ArgumentException("Qr expects a rank 2 array.");
        var m = a.Shape[0];
        var n = a.Shape[1];
        var r = a.Copy();
        var q = NdArray.Eye(m);

        var steps = Math.Min(m - 1, n);
        for (var k = 0; k < steps; k++)
        {
            // Build the Householder vector that zeroes column k below the diagonal.
            var normX = 0.0;
            for (var i = k; i < m; i++) normX += r[i, k] * r[i, k];
            normX = Math.Sqrt(normX);
            if (normX == 0.0) continue;

            var alpha = r[k, k] >= 0 ? -normX : normX;
            var v = new double[m];
            for (var i = k; i < m; i++) v[i] = r[i, k];
            v[k] -= alpha;

            var vNorm = 0.0;
            for (var i = k; i < m; i++) vNorm += v[i] * v[i];
            if (vNorm < 1e-300) continue;

            // Apply H = I - 2 v v^T / (v^T v) to R from the left and accumulate into Q.
            for (var j = k; j < n; j++)
            {
                var dot = 0.0;
                for (var i = k; i < m; i++) dot += v[i] * r[i, j];
                var scale = 2.0 * dot / vNorm;
                for (var i = k; i < m; i++) r[i, j] -= scale * v[i];
            }

            for (var j = 0; j < m; j++)
            {
                var dot = 0.0;
                for (var i = k; i < m; i++) dot += v[i] * q[j, i];
                var scale = 2.0 * dot / vNorm;
                for (var i = k; i < m; i++) q[j, i] -= scale * v[i];
            }
        }

        // Scrub the numerical dust below the diagonal.
        for (var i = 0; i < m; i++)
            for (var j = 0; j < Math.Min(i, n); j++)
                r[i, j] = 0.0;

        if (!reduced || m <= n) return new QrResult(q, r);

        var qReduced = q.Slice(Slice.All, Slice.Range(0, n)).Copy();
        var rReduced = r.Slice(Slice.Range(0, n), Slice.All).Copy();
        return new QrResult(qReduced, rReduced);
    }

    /// <summary>
    /// Cholesky factorisation of a symmetric positive-definite matrix, returning lower-triangular
    /// <c>L</c> with <c>A = L L^T</c>.
    /// </summary>
    public static NdArray Cholesky(NdArray a)
    {
        LinAlg.RequireSquare(a, nameof(Cholesky));
        var n = a.Shape[0];
        var l = NdArray.Zeros(n, n);

        for (var i = 0; i < n; i++)
            for (var j = 0; j <= i; j++)
            {
                var acc = a[i, j];
                for (var k = 0; k < j; k++) acc -= l[i, k] * l[j, k];

                if (i == j)
                {
                    if (acc <= 0)
                        throw new InvalidOperationException(
                            $"Matrix is not positive definite (pivot {acc:G6} at index {i}).");
                    l[i, j] = Math.Sqrt(acc);
                }
                else
                {
                    l[i, j] = acc / l[j, j];
                }
            }
        return l;
    }

    /// <summary>
    /// Singular value decomposition by one-sided Jacobi rotations.
    /// </summary>
    public static SvdResult Svd(NdArray a, int maxSweeps = 60, double tolerance = 1e-14)
    {
        if (a.Rank != 2) throw new ArgumentException("Svd expects a rank 2 array.");

        // The algorithm wants at least as many rows as columns; otherwise work on the transpose.
        if (a.Shape[0] < a.Shape[1])
        {
            var flipped = Svd(a.T.Copy(), maxSweeps, tolerance);
            return new SvdResult(flipped.V, flipped.SingularValues, flipped.U);
        }

        var m = a.Shape[0];
        var n = a.Shape[1];
        var u = a.Copy();
        var v = NdArray.Eye(n);

        for (var sweep = 0; sweep < maxSweeps; sweep++)
        {
            var converged = true;
            for (var p = 0; p < n - 1; p++)
                for (var q = p + 1; q < n; q++)
                {
                    double alpha = 0, beta = 0, gamma = 0;
                    for (var i = 0; i < m; i++)
                    {
                        var up = u[i, p];
                        var uq = u[i, q];
                        alpha += up * up;
                        beta += uq * uq;
                        gamma += up * uq;
                    }

                    if (Math.Abs(gamma) <= tolerance * Math.Sqrt(alpha * beta) || gamma == 0.0) continue;
                    converged = false;

                    // Rotate columns p and q so they become orthogonal.
                    var zeta = (beta - alpha) / (2.0 * gamma);
                    var t = Math.Sign(zeta) / (Math.Abs(zeta) + Math.Sqrt(1.0 + zeta * zeta));
                    if (zeta == 0.0) t = 1.0;
                    var c = 1.0 / Math.Sqrt(1.0 + t * t);
                    var s = c * t;

                    for (var i = 0; i < m; i++)
                    {
                        var up = u[i, p];
                        var uq = u[i, q];
                        u[i, p] = c * up - s * uq;
                        u[i, q] = s * up + c * uq;
                    }
                    for (var i = 0; i < n; i++)
                    {
                        var vp = v[i, p];
                        var vq = v[i, q];
                        v[i, p] = c * vp - s * vq;
                        v[i, q] = s * vp + c * vq;
                    }
                }
            if (converged) break;
        }

        // Column norms of U are the singular values; normalise, then sort descending.
        var singular = new double[n];
        for (var j = 0; j < n; j++)
        {
            var acc = 0.0;
            for (var i = 0; i < m; i++) acc += u[i, j] * u[i, j];
            singular[j] = Math.Sqrt(acc);
        }

        var order = Enumerable.Range(0, n).OrderByDescending(j => singular[j]).ToArray();
        var uOut = NdArray.Zeros(m, n);
        var vOut = NdArray.Zeros(n, n);
        var sOut = NdArray.Zeros(n);

        for (var rank = 0; rank < n; rank++)
        {
            var src = order[rank];
            var sv = singular[src];
            sOut.SetAt(rank, sv);
            var inv = sv > 1e-300 ? 1.0 / sv : 0.0;
            for (var i = 0; i < m; i++) uOut[i, rank] = u[i, src] * inv;
            for (var i = 0; i < n; i++) vOut[i, rank] = v[i, src];
        }

        return new SvdResult(uOut, sOut, vOut);
    }

    /// <summary>
    /// Eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi rotations,
    /// returned in descending eigenvalue order.
    /// </summary>
    public static EigenResult SymmetricEigen(NdArray a, int maxSweeps = 100, double tolerance = 1e-14)
    {
        LinAlg.RequireSquare(a, nameof(SymmetricEigen));
        var n = a.Shape[0];
        var work = a.Copy();
        var vectors = NdArray.Eye(n);

        for (var sweep = 0; sweep < maxSweeps; sweep++)
        {
            var off = 0.0;
            for (var p = 0; p < n - 1; p++)
                for (var q = p + 1; q < n; q++)
                    off += work[p, q] * work[p, q];
            if (Math.Sqrt(off) < tolerance) break;

            for (var p = 0; p < n - 1; p++)
                for (var q = p + 1; q < n; q++)
                {
                    var apq = work[p, q];
                    if (Math.Abs(apq) < 1e-300) continue;

                    var theta = (work[q, q] - work[p, p]) / (2.0 * apq);
                    var t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta == 0.0) t = 1.0;
                    var c = 1.0 / Math.Sqrt(t * t + 1.0);
                    var s = t * c;

                    for (var k = 0; k < n; k++)
                    {
                        var akp = work[k, p];
                        var akq = work[k, q];
                        work[k, p] = c * akp - s * akq;
                        work[k, q] = s * akp + c * akq;
                    }
                    for (var k = 0; k < n; k++)
                    {
                        var apk = work[p, k];
                        var aqk = work[q, k];
                        work[p, k] = c * apk - s * aqk;
                        work[q, k] = s * apk + c * aqk;
                    }
                    for (var k = 0; k < n; k++)
                    {
                        var vkp = vectors[k, p];
                        var vkq = vectors[k, q];
                        vectors[k, p] = c * vkp - s * vkq;
                        vectors[k, q] = s * vkp + c * vkq;
                    }
                }
        }

        var values = new double[n];
        for (var i = 0; i < n; i++) values[i] = work[i, i];

        var order = Enumerable.Range(0, n).OrderByDescending(i => values[i]).ToArray();
        var valuesOut = NdArray.Zeros(n);
        var vectorsOut = NdArray.Zeros(n, n);
        for (var rank = 0; rank < n; rank++)
        {
            valuesOut.SetAt(rank, values[order[rank]]);
            // Fix the sign so the largest-magnitude component is positive; keeps results stable.
            var col = order[rank];
            var maxAbs = 0.0;
            var sign = 1.0;
            for (var i = 0; i < n; i++)
            {
                var v = vectors[i, col];
                if (Math.Abs(v) > maxAbs) { maxAbs = Math.Abs(v); sign = v >= 0 ? 1.0 : -1.0; }
            }
            for (var i = 0; i < n; i++) vectorsOut[i, rank] = vectors[i, col] * sign;
        }

        return new EigenResult(valuesOut, vectorsOut);
    }

    /// <summary>
    /// Eigenvalues of a general (possibly non-symmetric) real matrix, as parallel real and
    /// imaginary parts. Symmetric inputs are routed to <see cref="SymmetricEigen"/>.
    /// </summary>
    public static (NdArray Real, NdArray Imaginary) Eigenvalues(NdArray a)
    {
        LinAlg.RequireSquare(a, nameof(Eigenvalues));
        if (LinAlg.IsSymmetric(a))
            return (SymmetricEigen(a).Values, NdArray.Zeros(a.Shape[0]));

        var n = a.Shape[0];
        var h = a.To2DArray();
        ReduceToHessenberg(h, n);
        var (re, im) = FrancisQr(h, n);
        return (new NdArray(re, n), new NdArray(im, n));
    }

    /// <summary>Reduces a matrix to upper Hessenberg form by elimination with pivoting.</summary>
    private static void ReduceToHessenberg(double[,] a, int n)
    {
        for (var m = 1; m < n - 1; m++)
        {
            var x = 0.0;
            var i = m;
            for (var j = m; j < n; j++)
                if (Math.Abs(a[j, m - 1]) > Math.Abs(x)) { x = a[j, m - 1]; i = j; }

            if (i != m)
            {
                for (var j = m - 1; j < n; j++) (a[i, j], a[m, j]) = (a[m, j], a[i, j]);
                for (var j = 0; j < n; j++) (a[j, i], a[j, m]) = (a[j, m], a[j, i]);
            }

            if (x == 0.0) continue;
            for (var k = m + 1; k < n; k++)
            {
                var y = a[k, m - 1];
                if (y == 0.0) continue;
                y /= x;
                a[k, m - 1] = y;
                for (var j = m; j < n; j++) a[k, j] -= y * a[m, j];
                for (var j = 0; j < n; j++) a[j, m] += y * a[j, k];
            }
        }

        for (var i = 2; i < n; i++)
            for (var j = 0; j < i - 1; j++)
                a[i, j] = 0.0;
    }

    /// <summary>Shifted QR iteration on an upper Hessenberg matrix (the classic <c>hqr</c>).</summary>
    private static (double[] Real, double[] Imaginary) FrancisQr(double[,] a, int n)
    {
        const double eps = 2.22e-16;
        var wr = new double[n];
        var wi = new double[n];

        var anorm = 0.0;
        for (var i = 0; i < n; i++)
            for (var j = Math.Max(i - 1, 0); j < n; j++)
                anorm += Math.Abs(a[i, j]);

        var nn = n - 1;
        var t = 0.0;
        double p = 0, q = 0, r = 0, z = 0, x, y, w, s, u, v;

        while (nn >= 0)
        {
            var its = 0;
            int l;
            do
            {
                // Look for a small subdiagonal element that splits the problem.
                for (l = nn; l > 0; l--)
                {
                    s = Math.Abs(a[l - 1, l - 1]) + Math.Abs(a[l, l]);
                    if (s == 0.0) s = anorm;
                    if (Math.Abs(a[l, l - 1]) <= eps * s) { a[l, l - 1] = 0.0; break; }
                }

                x = a[nn, nn];
                if (l == nn)
                {
                    wr[nn] = x + t;
                    wi[nn] = 0.0;
                    nn--;
                }
                else
                {
                    y = a[nn - 1, nn - 1];
                    w = a[nn, nn - 1] * a[nn - 1, nn];
                    if (l == nn - 1)
                    {
                        // A 2x2 block resolves to a real pair or a complex conjugate pair.
                        p = 0.5 * (y - x);
                        q = p * p + w;
                        z = Math.Sqrt(Math.Abs(q));
                        x += t;
                        if (q >= 0.0)
                        {
                            z = p + (p >= 0 ? Math.Abs(z) : -Math.Abs(z));
                            wr[nn - 1] = wr[nn] = x + z;
                            if (z != 0.0) wr[nn] = x - w / z;
                            wi[nn - 1] = wi[nn] = 0.0;
                        }
                        else
                        {
                            wr[nn] = wr[nn - 1] = x + p;
                            wi[nn] = -z;
                            wi[nn - 1] = z;
                        }
                        nn -= 2;
                    }
                    else
                    {
                        if (its == 60) throw new InvalidOperationException("QR iteration did not converge.");
                        if (its == 10 || its == 20 || its == 30 || its == 40 || its == 50)
                        {
                            // Exceptional shift to break a cycle.
                            t += x;
                            for (var i = 0; i <= nn; i++) a[i, i] -= x;
                            s = Math.Abs(a[nn, nn - 1]) + Math.Abs(a[nn - 1, nn - 2]);
                            y = x = 0.75 * s;
                            w = -0.4375 * s * s;
                        }
                        its++;

                        int m;
                        for (m = nn - 2; m >= l; m--)
                        {
                            z = a[m, m];
                            r = x - z;
                            s = y - z;
                            p = (r * s - w) / a[m + 1, m] + a[m, m + 1];
                            q = a[m + 1, m + 1] - z - r - s;
                            r = a[m + 2, m + 1];
                            s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                            p /= s; q /= s; r /= s;
                            if (m == l) break;
                            u = Math.Abs(a[m, m - 1]) * (Math.Abs(q) + Math.Abs(r));
                            v = Math.Abs(p) * (Math.Abs(a[m - 1, m - 1]) + Math.Abs(z) + Math.Abs(a[m + 1, m + 1]));
                            if (u <= eps * v) break;
                        }

                        for (var i = m; i <= nn - 2; i++)
                        {
                            a[i + 2, i] = 0.0;
                            if (i != m) a[i + 2, i - 1] = 0.0;
                        }

                        for (var k = m; k <= nn - 1; k++)
                        {
                            if (k != m)
                            {
                                p = a[k, k - 1];
                                q = a[k + 1, k - 1];
                                r = 0.0;
                                if (k + 1 != nn) r = a[k + 2, k - 1];
                                x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                                if (x != 0.0) { p /= x; q /= x; r /= x; }
                            }

                            s = Math.Sqrt(p * p + q * q + r * r);
                            if (p < 0) s = -s;
                            if (s == 0.0) continue;

                            if (k == m)
                            {
                                if (l != m) a[k, k - 1] = -a[k, k - 1];
                            }
                            else
                            {
                                a[k, k - 1] = -s * x;
                            }

                            p += s;
                            x = p / s; y = q / s; z = r / s;
                            q /= p; r /= p;

                            for (var j = k; j <= nn; j++)
                            {
                                var pp = a[k, j] + q * a[k + 1, j];
                                if (k + 1 != nn) { pp += r * a[k + 2, j]; a[k + 2, j] -= pp * z; }
                                a[k + 1, j] -= pp * y;
                                a[k, j] -= pp * x;
                            }

                            var mmin = nn < k + 3 ? nn : k + 3;
                            for (var i = l; i <= mmin; i++)
                            {
                                var pp = x * a[i, k] + y * a[i, k + 1];
                                if (k + 1 != nn) { pp += z * a[i, k + 2]; a[i, k + 2] -= pp * r; }
                                a[i, k + 1] -= pp * q;
                                a[i, k] -= pp;
                            }
                        }
                    }
                }
            } while (l + 1 < nn);
        }

        return (wr, wi);
    }
}
