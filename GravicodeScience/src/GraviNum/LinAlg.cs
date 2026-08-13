using System.Numerics;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// Dense linear algebra: products, solves, inverses and norms.
/// </summary>
/// <remarks>
/// <see cref="Dot"/> is the hot path of the whole ecosystem, so it is written as an <c>i-k-j</c>
/// loop over row-major storage: the innermost loop is a contiguous <c>y += a * x</c> that the
/// JIT turns into <see cref="Vector{T}"/> FMA work, and rows are handed to the thread pool once
/// the matrix is big enough to amortise it. Decompositions live in <see cref="Decomposition"/>.
/// </remarks>
public static class LinAlg
{
    /// <summary>Rows above which matrix products are parallelised.</summary>
    public const int ParallelRowThreshold = 64;

    /// <summary>
    /// Matrix product. Accepts matrix·matrix, matrix·vector, vector·matrix and vector·vector
    /// (which returns a 1-element array holding the inner product).
    /// </summary>
    public static NdArray Dot(NdArray a, NdArray b)
    {
        if (a.Rank == 1 && b.Rank == 1)
        {
            if (a.Size != b.Size)
                throw new InvalidOperationException($"Inner product needs equal lengths, got {a.Size} and {b.Size}.");
            return NdArray.Scalar(Inner(a, b));
        }

        if (a.Rank == 2 && b.Rank == 1)
        {
            if (a.Shape[1] != b.Size)
                throw new InvalidOperationException($"Cannot multiply ({Shapes.Describe(a.Shape)}) by a vector of length {b.Size}.");
            return MatVec(a, b);
        }

        if (a.Rank == 1 && b.Rank == 2)
        {
            if (a.Size != b.Shape[0])
                throw new InvalidOperationException($"Cannot multiply a vector of length {a.Size} by ({Shapes.Describe(b.Shape)}).");
            return MatMul(a.Reshape(1, a.Size), b).Reshape(b.Shape[1]);
        }

        if (a.Rank != 2 || b.Rank != 2)
            throw new InvalidOperationException($"Dot supports rank 1 and 2 arrays, got {a.Rank} and {b.Rank}.");

        if (a.Shape[1] != b.Shape[0])
            throw new InvalidOperationException(
                $"Shapes ({Shapes.Describe(a.Shape)}) and ({Shapes.Describe(b.Shape)}) are not aligned for a matrix product.");

        return MatMul(a, b);
    }

    /// <summary>Inner (dot) product of two 1-D arrays.</summary>
    public static double Inner(NdArray a, NdArray b)
    {
        if (a.Size != b.Size) throw new InvalidOperationException("Inner product needs equal lengths.");
        if (a.IsContiguous && b.IsContiguous) return InnerSpan(a.AsSpan(), b.AsSpan());

        var acc = 0.0;
        for (var i = 0; i < a.Size; i++) acc += a.At(i) * b.At(i);
        return acc;
    }

    /// <summary>SIMD inner product over two contiguous spans.</summary>
    public static double InnerSpan(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        var n = x.Length;
        var acc = 0.0;
        var i = 0;

        if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            var width = Vector<double>.Count;
            var vsum = Vector<double>.Zero;
            for (; i <= n - width; i += width)
                vsum += new Vector<double>(x.Slice(i, width)) * new Vector<double>(y.Slice(i, width));
            acc = Vector.Sum(vsum);
        }

        for (; i < n; i++) acc += x[i] * y[i];
        return acc;
    }

    /// <summary>Outer product of two 1-D arrays.</summary>
    public static NdArray Outer(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Size, b.Size);
        for (var i = 0; i < a.Size; i++)
        {
            var ai = a.At(i);
            for (var j = 0; j < b.Size; j++) result[i, j] = ai * b.At(j);
        }
        return result;
    }

    private static NdArray MatVec(NdArray a, NdArray x)
    {
        var m = a.Shape[0];
        var k = a.Shape[1];
        var ac = a.AsContiguous();
        var xc = x.AsContiguous();
        var src = ac.Buffer;
        var srcOffset = ac.Offset;
        var vec = xc.Buffer.AsSpan(xc.Offset, k).ToArray();

        var result = NdArray.Zeros(m);
        var dst = result.Buffer;

        if (m >= ParallelRowThreshold)
        {
            Parallel.For(0, m, i => dst[i] = InnerSpan(src.AsSpan(srcOffset + i * k, k), vec));
        }
        else
        {
            for (var i = 0; i < m; i++) dst[i] = InnerSpan(src.AsSpan(srcOffset + i * k, k), vec);
        }
        return result;
    }

    private static NdArray MatMul(NdArray a, NdArray b)
    {
        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];

        var ac = a.AsContiguous();
        var bc = b.AsContiguous();
        var av = ac.Buffer;
        var bv = bc.Buffer;
        var aOff = ac.Offset;
        var bOff = bc.Offset;

        var result = NdArray.Zeros(m, n);
        var cv = result.Buffer;

        // Hand off to a native BLAS when the machine has one and the product is big enough to
        // repay the call. The managed kernel below stays as the reference the tests check against.
        if (Compute.NativeBlas.ShouldUse(m, n, k) && aOff == 0 && bOff == 0 && cv.Length == (long)m * n)
        {
            Compute.NativeBlas.Multiply(av, bv, cv, m, n, k);
            return result;
        }

        // No native library: pack the operands and use the blocked kernel, which is worth its
        // copying cost once the product is large enough that cache misses dominate.
        if (PackedMatMul.ShouldUse(m, n, k))
        {
            PackedMatMul.Multiply(av, aOff, bv, bOff, cv, m, n, k);
            return result;
        }

        // A 4-row by one-vector tile of C is held in registers across a slice of k, then added
        // into memory once per slice. Two separate problems drive that shape, and fixing either
        // one alone makes things worse:
        //
        //   Traffic. With the column loop innermost and k outside it, every element of C is
        //   loaded and stored once per step of k - nine memory operations per four FMAs at
        //   1024x1024. Accumulating in registers instead touches C once per k slice.
        //
        //   Locality. Walking k inside a fixed column means striding through B by a whole row
        //   each step. Left unbounded that sweeps k*n bytes per tile and falls out of cache
        //   entirely; slicing k keeps the live part of B to KC*n, which stays resident.
        //
        // Four rows also means one load of B feeds four independent FMA chains, which hides the
        // multiply latency - a single-row kernel stalls on the dependency between iterations.
        var slice = KSlice(k, n);

        void ComputeRowBlock(int block, int p0, int pn)
        {
            var i0 = block * 4;
            var rows = Math.Min(4, m - i0);

            unsafe
            {
                fixed (double* aPtr = av, bPtr = bv, cPtr = cv)
                {
                    var width = Vector<double>.Count;
                    var aBase = aPtr + aOff + (long)i0 * k + p0;
                    var bBase = bPtr + bOff + (long)p0 * n;
                    var c0 = cPtr + (long)i0 * n;
                    var vectorised = Vector.IsHardwareAccelerated && n >= width;

                    var j = 0;
                    if (vectorised && rows == 4)
                    {
                        // The common case gets its own loop with no per-iteration row test.
                        for (; j <= n - width; j += width)
                        {
                            Vector<double> acc0 = default, acc1 = default, acc2 = default, acc3 = default;
                            var bp = bBase + j;

                            for (var p = 0; p < pn; p++, bp += n)
                            {
                                var bVec = Vector.Load(bp);
                                acc0 += new Vector<double>(aBase[p]) * bVec;
                                acc1 += new Vector<double>(aBase[k + p]) * bVec;
                                acc2 += new Vector<double>(aBase[2 * k + p]) * bVec;
                                acc3 += new Vector<double>(aBase[3 * k + p]) * bVec;
                            }

                            Vector.Store(Vector.Load(c0 + j) + acc0, c0 + j);
                            Vector.Store(Vector.Load(c0 + n + j) + acc1, c0 + n + j);
                            Vector.Store(Vector.Load(c0 + 2 * n + j) + acc2, c0 + 2 * n + j);
                            Vector.Store(Vector.Load(c0 + 3 * n + j) + acc3, c0 + 3 * n + j);
                        }
                    }
                    else if (vectorised)
                    {
                        // Ragged final block: same shape, guarded.
                        for (; j <= n - width; j += width)
                        {
                            Vector<double> acc0 = default, acc1 = default, acc2 = default;
                            var bp = bBase + j;

                            for (var p = 0; p < pn; p++, bp += n)
                            {
                                var bVec = Vector.Load(bp);
                                acc0 += new Vector<double>(aBase[p]) * bVec;
                                if (rows > 1) acc1 += new Vector<double>(aBase[k + p]) * bVec;
                                if (rows > 2) acc2 += new Vector<double>(aBase[2 * k + p]) * bVec;
                            }

                            Vector.Store(Vector.Load(c0 + j) + acc0, c0 + j);
                            if (rows > 1) Vector.Store(Vector.Load(c0 + n + j) + acc1, c0 + n + j);
                            if (rows > 2) Vector.Store(Vector.Load(c0 + 2 * n + j) + acc2, c0 + 2 * n + j);
                        }
                    }

                    // Columns left over when n is not a multiple of the vector width.
                    for (; j < n; j++)
                    {
                        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        var bp = bBase + j;

                        for (var p = 0; p < pn; p++, bp += n)
                        {
                            var bValue = *bp;
                            s0 += aBase[p] * bValue;
                            if (rows > 1) s1 += aBase[k + p] * bValue;
                            if (rows > 2) s2 += aBase[2 * k + p] * bValue;
                            if (rows > 3) s3 += aBase[3 * k + p] * bValue;
                        }

                        c0[j] += s0;
                        if (rows > 1) c0[n + j] += s1;
                        if (rows > 2) c0[2 * n + j] += s2;
                        if (rows > 3) c0[3 * n + j] += s3;
                    }
                }
            }
        }

        var blocks = (m + 3) / 4;
        var parallel = m >= ParallelRowThreshold || (long)m * n * k > 1_000_000;

        for (var p0 = 0; p0 < k; p0 += slice)
        {
            var pn = Math.Min(slice, k - p0);
            var start = p0;

            if (parallel)
                Parallel.For(0, blocks, block => ComputeRowBlock(block, start, pn));
            else
                for (var block = 0; block < blocks; block++) ComputeRowBlock(block, start, pn);
        }

        return result;
    }

    /// <summary>
    /// Rows of B processed per pass, sized so the live block stays in cache.
    /// </summary>
    /// <remarks>
    /// The inner loop strides through B by a full row, so the block it touches is
    /// <c>slice * n * 8</c> bytes. Holding that near 512 KB keeps it in L2 while leaving room for
    /// the row block of A and the tile of C. A small <c>k</c> is left as one slice, since splitting
    /// it would only add passes over C for no locality gain.
    /// </remarks>
    private static int KSlice(int k, int n)
    {
        const long target = 4 * 1024 * 1024;
        if (k <= 8) return k;

        return (int)Math.Clamp(target / Math.Max(1, (long)n * 8), 8, k);
    }

    /// <summary>Vectorised <c>y += alpha * x</c> over contiguous spans.</summary>
    public static void AxpySpan(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        var n = x.Length;
        var i = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            var width = Vector<double>.Count;
            var va = new Vector<double>(alpha);
            for (; i <= n - width; i += width)
            {
                var acc = new Vector<double>(y.Slice(i, width)) + va * new Vector<double>(x.Slice(i, width));
                acc.CopyTo(y.Slice(i, width));
            }
        }
        for (; i < n; i++) y[i] += alpha * x[i];
    }

    /// <summary>Transpose of a 2-D array as a zero-copy view.</summary>
    public static NdArray Transpose(NdArray a) => a.T;

    /// <summary>Sum of the main diagonal.</summary>
    public static double Trace(NdArray a)
    {
        RequireSquare(a, nameof(Trace));
        var acc = 0.0;
        for (var i = 0; i < a.Shape[0]; i++) acc += a[i, i];
        return acc;
    }

    /// <summary>The main diagonal as a 1-D array.</summary>
    public static NdArray Diagonal(NdArray a)
    {
        if (a.Rank != 2) throw new ArgumentException("Diagonal expects a rank 2 array.");
        var n = Math.Min(a.Shape[0], a.Shape[1]);
        var result = NdArray.Zeros(n);
        for (var i = 0; i < n; i++) result.SetAt(i, a[i, i]);
        return result;
    }

    /// <summary>Determinant, computed from the LU factorisation.</summary>
    public static double Determinant(NdArray a)
    {
        RequireSquare(a, nameof(Determinant));
        var lu = Decomposition.Lu(a);
        var det = lu.Sign;
        for (var i = 0; i < a.Shape[0]; i++) det *= lu.Upper[i, i];
        return det;
    }

    /// <summary>Natural log of the absolute determinant, safe for large matrices.</summary>
    public static (double LogAbsDet, int Sign) SlogDet(NdArray a)
    {
        RequireSquare(a, nameof(SlogDet));
        var lu = Decomposition.Lu(a);
        var sign = (int)lu.Sign;
        var acc = 0.0;
        for (var i = 0; i < a.Shape[0]; i++)
        {
            var d = lu.Upper[i, i];
            if (d == 0) return (double.NegativeInfinity, 0);
            if (d < 0) sign = -sign;
            acc += Math.Log(Math.Abs(d));
        }
        return (acc, sign);
    }

    /// <summary>Solves <c>A x = b</c> for a square <c>A</c> using LU with partial pivoting.</summary>
    /// <remarks>
    /// Goes to a native <c>dgesv</c> when one is available and the system is big enough to repay
    /// the call. That is by far the largest native win in this library — a 1024-square system
    /// measured about 70x faster — because the managed LU walks the array through its indexer
    /// while LAPACK runs a blocked factorisation on top of a tuned BLAS.
    /// </remarks>
    public static NdArray Solve(NdArray a, NdArray b)
    {
        RequireSquare(a, nameof(Solve));
        var n = a.Shape[0];

        if (Compute.NativeLapack.ShouldUse(n))
        {
            var columns = b.Rank == 2 ? b.Shape[1] : 1;
            var matrix = a.ToArray();
            var rhs = b.AsContiguous().ToArray();

            // A non-zero info means the matrix is singular to working precision; fall through so
            // the managed path raises the error it already documents.
            if (Compute.NativeLapack.Solve(matrix, rhs, n, columns) == 0)
                return b.Rank == 2 ? new NdArray(rhs, n, columns) : new NdArray(rhs, n);
        }

        var lu = Decomposition.Lu(a);
        return lu.Solve(b);
    }

    /// <summary>Matrix inverse.</summary>
    /// <remarks>
    /// Routed through <see cref="Solve"/> with an identity right-hand side rather than calling the
    /// LU factorisation directly, so it picks up the native path along with everything else. The
    /// arithmetic is the same either way — an inverse *is* n simultaneous solves — but going
    /// through the LU result kept this on the managed path while `Solve` had already moved.
    /// </remarks>
    public static NdArray Inverse(NdArray a)
    {
        RequireSquare(a, nameof(Inverse));
        return Solve(a, NdArray.Eye(a.Shape[0]));
    }

    /// <summary>Moore-Penrose pseudo-inverse, computed from the SVD.</summary>
    public static NdArray PseudoInverse(NdArray a, double tolerance = 1e-12)
    {
        var svd = Decomposition.Svd(a);
        var k = svd.SingularValues.Size;
        var cutoff = tolerance * (svd.SingularValues.Size > 0 ? Statistics.Max(svd.SingularValues) : 0.0);

        var sInv = NdArray.Zeros(k, k);
        for (var i = 0; i < k; i++)
        {
            var s = svd.SingularValues.At(i);
            sInv[i, i] = s > cutoff ? 1.0 / s : 0.0;
        }
        return Dot(Dot(svd.V, sInv), svd.U.T);
    }

    /// <summary>Least-squares solution of <c>A x ~ b</c> via the pseudo-inverse.</summary>
    public static NdArray LeastSquares(NdArray a, NdArray b) => Dot(PseudoInverse(a), b);

    /// <summary>Numerical rank, counted as singular values above a relative tolerance.</summary>
    public static int MatrixRank(NdArray a, double tolerance = 1e-10)
    {
        var s = Decomposition.SingularValues(a);
        if (s.Size == 0) return 0;
        var cutoff = Statistics.Max(s) * tolerance;
        var rank = 0;
        for (var i = 0; i < s.Size; i++) if (s.At(i) > cutoff) rank++;
        return rank;
    }

    /// <summary>Ratio of the largest to the smallest singular value.</summary>
    public static double ConditionNumber(NdArray a)
    {
        var s = Decomposition.SingularValues(a);
        var min = Statistics.Min(s);
        return min == 0 ? double.PositiveInfinity : Statistics.Max(s) / min;
    }

    /// <summary>Frobenius norm (or the Euclidean norm for a vector).</summary>
    public static double Norm(NdArray a)
    {
        var acc = 0.0;
        for (var i = 0; i < a.Size; i++) { var v = a.At(i); acc += v * v; }
        return Math.Sqrt(acc);
    }

    /// <summary>General p-norm; use <c>p = double.PositiveInfinity</c> for the max norm.</summary>
    public static double Norm(NdArray a, double p)
    {
        if (double.IsPositiveInfinity(p))
        {
            var m = 0.0;
            for (var i = 0; i < a.Size; i++) m = Math.Max(m, Math.Abs(a.At(i)));
            return m;
        }
        if (p == 1)
        {
            var s = 0.0;
            for (var i = 0; i < a.Size; i++) s += Math.Abs(a.At(i));
            return s;
        }
        var acc = 0.0;
        for (var i = 0; i < a.Size; i++) acc += Math.Pow(Math.Abs(a.At(i)), p);
        return Math.Pow(acc, 1.0 / p);
    }

    /// <summary>Raises a square matrix to a non-negative integer power.</summary>
    public static NdArray MatrixPower(NdArray a, int power)
    {
        RequireSquare(a, nameof(MatrixPower));
        if (power < 0) return MatrixPower(Inverse(a), -power);

        var result = NdArray.Eye(a.Shape[0]);
        var baseMatrix = a.Copy();
        while (power > 0)
        {
            if ((power & 1) == 1) result = Dot(result, baseMatrix);
            baseMatrix = Dot(baseMatrix, baseMatrix);
            power >>= 1;
        }
        return result;
    }

    /// <summary>Kronecker product of two 2-D arrays.</summary>
    public static NdArray Kron(NdArray a, NdArray b)
    {
        if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("Kron expects rank 2 arrays.");
        var (m, n) = (a.Shape[0], a.Shape[1]);
        var (p, q) = (b.Shape[0], b.Shape[1]);
        var result = NdArray.Zeros(m * p, n * q);
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var aij = a[i, j];
                for (var r = 0; r < p; r++)
                    for (var c = 0; c < q; c++)
                        result[i * p + r, j * q + c] = aij * b[r, c];
            }
        return result;
    }

    /// <summary>True when the matrix equals its transpose within <paramref name="tolerance"/>.</summary>
    public static bool IsSymmetric(NdArray a, double tolerance = 1e-10)
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1]) return false;
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = i + 1; j < a.Shape[1]; j++)
                if (Math.Abs(a[i, j] - a[j, i]) > tolerance) return false;
        return true;
    }

    internal static void RequireSquare(NdArray a, string caller)
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException($"{caller} requires a square matrix, got ({Shapes.Describe(a.Shape)}).");
    }
}
