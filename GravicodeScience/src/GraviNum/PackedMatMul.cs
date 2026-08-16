using System.Numerics;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// A cache-blocked matrix product with packed operands, for machines with no native BLAS.
/// </summary>
/// <remarks>
/// <para>
/// The simple kernel in <see cref="LinAlg"/> holds a tile of C in registers across a slice of k,
/// which is most of the arithmetic win available without moving data. What it cannot fix is the
/// walk through B: the inner loop strides a whole row per step of k, so consecutive loads land in
/// different cache lines and most of the bandwidth is spent fetching lines to use eight bytes of.
/// </para>
/// <para>
/// Packing fixes that by rewriting both operands into the order the kernel reads them. A panel of
/// B is copied so that the values one micro-kernel iteration needs sit next to each other; the
/// copy costs one pass over the panel and is repaid many times over, because that panel is then
/// streamed by every row block. This is the idea behind every tuned BLAS, in its simplest form —
/// two levels of blocking and one micro-kernel, rather than the four levels and hand-written
/// assembly a real one uses.
/// </para>
/// </remarks>
public static class PackedMatMul
{
    /// <summary>
    /// Whether the packed kernel is used, when no native BLAS is present.
    /// </summary>
    /// <remarks>
    /// Public for the same reason <see cref="Compute.NativeBlas.Enabled"/> is: turning a fast path
    /// off is how it gets compared against the simpler one it replaced, and the simpler one stays
    /// in place as that reference.
    /// </remarks>
    public static bool Enabled { get; set; } = true;

    /// <summary>Rows of the micro-kernel tile.</summary>
    private const int Mr = 4;

    /// <summary>Rows of B held in the packed panel — chosen so the panel stays in L2.</summary>
    private const int Kc = 256;

    /// <summary>Columns of B per packed panel.</summary>
    private const int Nc = 256;

    /// <summary>Rows of A per block.</summary>
    private const int Mc = 128;

    /// <summary>Whether a product of this shape is worth the packing cost.</summary>
    /// <remarks>
    /// <para>
    /// Packing costs a pass over each operand before any arithmetic happens, so there is a size
    /// below which it cannot pay for itself. Measured on this machine the packed kernel loses at a
    /// 128-cube, is within noise up to about 192, and wins from roughly 224 upward — 1.65x at 256
    /// and 2.08x at 2048. The threshold sits at eight million multiply-adds, just above the noisy
    /// region.
    /// </para>
    /// <para>
    /// It is set deliberately conservatively. Missing a marginal win on a medium product costs
    /// almost nothing; being slower than the kernel this replaced would be a regression.
    /// </para>
    /// </remarks>
    public static bool ShouldUse(int m, int n, int k)
        => Enabled && (long)m * n * k >= 8_000_000;

    /// <summary>Computes <c>C = A B</c> for row-major operands.</summary>
    public static unsafe void Multiply(
        double[] a, int aOffset, double[] b, int bOffset, double[] c, int m, int n, int k)
    {
        var width = Vector<double>.Count;
        var nr = width * 2;   // micro-kernel tile width, two vectors

        // The packed B panel is shared across every row block, which is the whole point of packing
        // it: one copy, many readers. Allocated once per call rather than per panel.
        var packedB = new double[Kc * Nc];

        // Row blocks are independent — each writes its own rows of C — so they run in parallel.
        // Each worker needs its own packed A, because that buffer holds *its* rows.
        var blocks = (m + Mc - 1) / Mc;
        var scratch = new double[blocks][];

        fixed (double* aBase = a, bBase = b, cBase = c, pbBase = packedB)
        {
            // Copied out of the fixed locals: a lambda cannot capture those directly, and the
            // buffers stay pinned for the whole block regardless.
            var aPtr = aBase + aOffset;
            var bPtr = bBase + bOffset;
            var cPtr = cBase;
            var packedBPtr = pbBase;

            for (var jc = 0; jc < n; jc += Nc)
            {
                var nn = Math.Min(Nc, n - jc);
                var columnStart = jc;

                for (var pc = 0; pc < k; pc += Kc)
                {
                    var kk = Math.Min(Kc, k - pc);
                    var sliceStart = pc;

                    PackB(bPtr, pbBase, n, pc, jc, kk, nn, nr);

                    Parallel.For(0, blocks, block =>
                    {
                        var ic = block * Mc;
                        var mm = Math.Min(Mc, m - ic);
                        if (mm <= 0) return;

                        var buffer = scratch[block] ??= new double[Mc * Kc];

                        fixed (double* paBase = buffer)
                        {
                            PackA(aPtr, paBase, k, ic, sliceStart, mm, kk);

                            // The first k slice writes C; later slices accumulate into it.
                            MacroKernel(paBase, packedBPtr, cPtr, n, ic, columnStart, mm, nn, kk, nr,
                                width, accumulate: sliceStart > 0);
                        }
                    });
                }
            }
        }
    }

    /// <summary>
    /// Copies a <c>kk x nn</c> block of B into panels of <paramref name="nr"/> columns, so the
    /// micro-kernel reads one contiguous run per step of k.
    /// </summary>
    private static unsafe void PackB(double* b, double* packed, int n, int pc, int jc,
        int kk, int nn, int nr)
    {
        var write = packed;
        for (var j = 0; j < nn; j += nr)
        {
            var span = Math.Min(nr, nn - j);
            for (var p = 0; p < kk; p++)
            {
                var row = b + (long)(pc + p) * n + jc + j;
                for (var t = 0; t < span; t++) *write++ = row[t];
                // Ragged tail padded with zeros so the kernel needs no edge case.
                for (var t = span; t < nr; t++) *write++ = 0.0;
            }
        }
    }

    /// <summary>
    /// Copies an <c>mm x kk</c> block of A into panels of <see cref="Mr"/> rows, transposed so
    /// the micro-kernel's k loop reads consecutive doubles.
    /// </summary>
    private static unsafe void PackA(double* a, double* packed, int k, int ic, int pc, int mm, int kk)
    {
        var write = packed;
        for (var i = 0; i < mm; i += Mr)
        {
            var span = Math.Min(Mr, mm - i);
            for (var p = 0; p < kk; p++)
            {
                for (var t = 0; t < span; t++) *write++ = a[(long)(ic + i + t) * k + pc + p];
                for (var t = span; t < Mr; t++) *write++ = 0.0;
            }
        }
    }

    /// <summary>Runs the micro-kernel over every tile of the packed block.</summary>
    private static unsafe void MacroKernel(double* packedA, double* packedB, double* c, int n,
        int ic, int jc, int mm, int nn, int kk, int nr, int width, bool accumulate)
    {
        for (var j = 0; j < nn; j += nr)
        {
            var bPanel = packedB + (long)(j / nr) * kk * nr;

            for (var i = 0; i < mm; i += Mr)
            {
                var aPanel = packedA + (long)(i / Mr) * kk * Mr;
                var rows = Math.Min(Mr, mm - i);
                var columns = Math.Min(nr, nn - j);

                MicroKernel(aPanel, bPanel, c + (long)(ic + i) * n + jc + j, n, kk,
                    rows, columns, width, accumulate);
            }
        }
    }

    /// <summary>
    /// The innermost tile: <see cref="Mr"/> rows by two vectors of columns, accumulated in
    /// registers across the whole packed k slice.
    /// </summary>
    /// <remarks>
    /// Both operands are now contiguous, so each step of k reads <see cref="Mr"/> consecutive
    /// doubles from A and <c>nr</c> consecutive doubles from B — the loads the previous kernel
    /// scattered across a whole row of memory.
    /// </remarks>
    private static unsafe void MicroKernel(double* a, double* b, double* c, int n, int kk,
        int rows, int columns, int width, bool accumulate)
    {
        Vector<double> c00 = default, c01 = default;
        Vector<double> c10 = default, c11 = default;
        Vector<double> c20 = default, c21 = default;
        Vector<double> c30 = default, c31 = default;

        for (var p = 0; p < kk; p++)
        {
            var b0 = Vector.Load(b);
            var b1 = Vector.Load(b + width);
            b += width * 2;

            c00 += new Vector<double>(a[0]) * b0;
            c01 += new Vector<double>(a[0]) * b1;
            c10 += new Vector<double>(a[1]) * b0;
            c11 += new Vector<double>(a[1]) * b1;
            c20 += new Vector<double>(a[2]) * b0;
            c21 += new Vector<double>(a[2]) * b1;
            c30 += new Vector<double>(a[3]) * b0;
            c31 += new Vector<double>(a[3]) * b1;
            a += Mr;
        }

        Store(c, 0, c00, c01);
        Store(c, 1, c10, c11);
        Store(c, 2, c20, c21);
        Store(c, 3, c30, c31);

        void Store(double* target, int row, Vector<double> left, Vector<double> right)
        {
            if (row >= rows) return;
            var destination = target + (long)row * n;

            // A tile at the edge of C is written element by element: the packed panels were
            // zero-padded, so the extra lanes hold zeros, but writing them would run off C.
            WriteLane(destination, left, 0);
            WriteLane(destination, right, width);
        }

        void WriteLane(double* destination, Vector<double> value, int offset)
        {
            if (offset + width <= columns)
            {
                if (accumulate) Vector.Store(Vector.Load(destination + offset) + value, destination + offset);
                else Vector.Store(value, destination + offset);
                return;
            }

            for (var t = 0; t < width && offset + t < columns; t++)
            {
                if (accumulate) destination[offset + t] += value[t];
                else destination[offset + t] = value[t];
            }
        }
    }
}
