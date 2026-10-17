namespace Gravicode.Science.SciNet.Sparse;

/// <summary>A compressed (row- or column-major) sparse structure: the raw arrays every kernel works on.</summary>
internal readonly record struct CompressedArrays(int Major, int Minor, int[] Ptr, int[] Idx, double[] Val);

/// <summary>A binary operation applied entry-wise by the merge kernels; static so the JIT specialises each one.</summary>
internal interface ISparseBinaryOp
{
    static abstract double Apply(double a, double b);
}

internal struct SparseAddOp : ISparseBinaryOp { public static double Apply(double a, double b) => a + b; }
internal struct SparseSubOp : ISparseBinaryOp { public static double Apply(double a, double b) => a - b; }
internal struct SparseMulOp : ISparseBinaryOp { public static double Apply(double a, double b) => a * b; }
internal struct SparseMaxOp : ISparseBinaryOp { public static double Apply(double a, double b) => Math.Max(a, b); }
internal struct SparseMinOp : ISparseBinaryOp { public static double Apply(double a, double b) => Math.Min(a, b); }

/// <summary>
/// The compressed-format kernels shared by CSR and CSC (a CSC matrix is the CSR structure of its
/// transpose, so every kernel here is written once, for rows).
/// </summary>
internal static unsafe class SparseKernels
{
    /// <summary>Non-zeros above which SpMV goes parallel; below it the fork/join costs more than the loop.</summary>
    internal const int ParallelSpMVThreshold = 60_000;

    /// <summary>Estimated multiply-adds above which SpGEMM goes parallel.</summary>
    internal const long ParallelSpGemmThreshold = 200_000;

    /// <summary>Transposes a compressed structure in O(nnz + n) by counting sort. The output's minor indices are sorted.</summary>
    internal static CompressedArrays Transpose(in CompressedArrays a)
    {
        var nnz = a.Ptr[a.Major];
        var ptr = new int[a.Minor + 1];
        var idx = new int[nnz];
        var val = new double[nnz];
        var aIdx = a.Idx;
        for (var k = 0; k < nnz; k++) ptr[aIdx[k] + 1]++;
        for (var j = 0; j < a.Minor; j++) ptr[j + 1] += ptr[j];
        var next = new int[a.Minor];
        Array.Copy(ptr, next, a.Minor);
        for (var i = 0; i < a.Major; i++)
            for (var k = a.Ptr[i]; k < a.Ptr[i + 1]; k++)
            {
                var j = aIdx[k];
                var dst = next[j]++;
                idx[dst] = i;
                val[dst] = a.Val[k];
            }
        return new CompressedArrays(a.Minor, a.Major, ptr, idx, val);
    }

    /// <summary>True when every row's minor indices are strictly increasing (sorted, no duplicates).</summary>
    internal static bool IsCanonical(int major, int[] ptr, int[] idx)
    {
        for (var i = 0; i < major; i++)
            for (var k = ptr[i] + 1; k < ptr[i + 1]; k++)
                if (idx[k] <= idx[k - 1]) return false;
        return true;
    }

    /// <summary>True when every row's minor indices are non-decreasing.</summary>
    internal static bool IsSorted(int major, int[] ptr, int[] idx)
    {
        for (var i = 0; i < major; i++)
            for (var k = ptr[i] + 1; k < ptr[i + 1]; k++)
                if (idx[k] < idx[k - 1]) return false;
        return true;
    }

    /// <summary>Sorts each row by minor index and sums duplicates. Explicit zeros are kept (as SciPy does).</summary>
    internal static CompressedArrays Canonicalize(in CompressedArrays a)
    {
        if (IsCanonical(a.Major, a.Ptr, a.Idx)) return a;
        var idx = (int[])a.Idx.Clone();
        var val = (double[])a.Val.Clone();
        for (var i = 0; i < a.Major; i++)
        {
            int s = a.Ptr[i], len = a.Ptr[i + 1] - s;
            if (len > 1) Array.Sort(idx, val, s, len);
        }
        var ptr = new int[a.Major + 1];
        var w = 0;
        for (var i = 0; i < a.Major; i++)
        {
            var k = a.Ptr[i];
            var end = a.Ptr[i + 1];
            while (k < end)
            {
                var j = idx[k];
                var sum = val[k++];
                while (k < end && idx[k] == j) sum += val[k++];
                idx[w] = j;
                val[w++] = sum;
            }
            ptr[i + 1] = w;
        }
        Array.Resize(ref idx, w);
        Array.Resize(ref val, w);
        return a with { Ptr = ptr, Idx = idx, Val = val };
    }

    /// <summary>Removes stored entries for which <paramref name="drop"/> is true.</summary>
    internal static CompressedArrays Filter(in CompressedArrays a, Func<int, int, double, bool> drop)
    {
        var nnz = a.Ptr[a.Major];
        var ptr = new int[a.Major + 1];
        var idx = new int[nnz];
        var val = new double[nnz];
        var w = 0;
        for (var i = 0; i < a.Major; i++)
        {
            for (var k = a.Ptr[i]; k < a.Ptr[i + 1]; k++)
            {
                if (drop(i, a.Idx[k], a.Val[k])) continue;
                idx[w] = a.Idx[k];
                val[w++] = a.Val[k];
            }
            ptr[i + 1] = w;
        }
        Array.Resize(ref idx, w);
        Array.Resize(ref val, w);
        return a with { Ptr = ptr, Idx = idx, Val = val };
    }

    /// <summary>
    /// Merges two canonical structures of the same shape entry-wise. With <paramref name="union"/> a
    /// missing entry reads as zero (add, subtract, max, min); without it only shared positions survive
    /// (Hadamard product). Exact-zero results are not stored, matching SciPy's binop.
    /// </summary>
    internal static CompressedArrays Merge<TOp>(in CompressedArrays a, in CompressedArrays b, bool union)
        where TOp : ISparseBinaryOp
    {
        var cap = union ? a.Ptr[a.Major] + b.Ptr[b.Major] : Math.Min(a.Ptr[a.Major], b.Ptr[b.Major]);
        var ptr = new int[a.Major + 1];
        var idx = new int[cap];
        var val = new double[cap];
        var w = 0;
        for (var i = 0; i < a.Major; i++)
        {
            int p = a.Ptr[i], pe = a.Ptr[i + 1], q = b.Ptr[i], qe = b.Ptr[i + 1];
            while (p < pe || q < qe)
            {
                int ja = p < pe ? a.Idx[p] : int.MaxValue;
                int jb = q < qe ? b.Idx[q] : int.MaxValue;
                double r;
                int j;
                if (ja == jb) { j = ja; r = TOp.Apply(a.Val[p++], b.Val[q++]); }
                else if (ja < jb)
                {
                    j = ja;
                    var v = a.Val[p++];
                    if (!union) continue;
                    r = TOp.Apply(v, 0.0);
                }
                else
                {
                    j = jb;
                    var v = b.Val[q++];
                    if (!union) continue;
                    r = TOp.Apply(0.0, v);
                }
                if (r == 0.0) continue;
                idx[w] = j;
                val[w++] = r;
            }
            ptr[i + 1] = w;
        }
        Array.Resize(ref idx, w);
        Array.Resize(ref val, w);
        return new CompressedArrays(a.Major, a.Minor, ptr, idx, val);
    }

    /// <summary>
    /// <c>y = A x</c> for a compressed-row structure. Rows are split into chunks of roughly equal
    /// non-zero count (not equal row count: a power-law row would otherwise serialise one worker),
    /// and each chunk runs a pointer loop with two accumulators.
    /// </summary>
    internal static void SpMV(in CompressedArrays a, ReadOnlySpan<double> x, Span<double> y)
    {
        var nnz = a.Ptr[a.Major];
        fixed (int* ptr = a.Ptr)
        fixed (int* idx = a.Idx)
        fixed (double* val = a.Val)
        fixed (double* px = x)
        fixed (double* py = y)
        {
            if (nnz < ParallelSpMVThreshold || Environment.ProcessorCount == 1)
            {
                SpMVRange(0, a.Major, ptr, idx, val, px, py);
                return;
            }
            nint iptr = (nint)ptr, iidx = (nint)idx, ival = (nint)val, ix = (nint)px, iy = (nint)py;
            var bounds = BalancedRowBlocks(a.Ptr, a.Major, Environment.ProcessorCount * 4);
            Parallel.For(0, bounds.Length - 1, b =>
                SpMVRange(bounds[b], bounds[b + 1], (int*)iptr, (int*)iidx, (double*)ival, (double*)ix, (double*)iy));
        }
    }

    /// <summary>The scalar reference SpMV, kept so the tests can check the unrolled/parallel path against it.</summary>
    internal static void SpMVReference(in CompressedArrays a, ReadOnlySpan<double> x, Span<double> y)
    {
        for (var i = 0; i < a.Major; i++)
        {
            var acc = 0.0;
            for (var k = a.Ptr[i]; k < a.Ptr[i + 1]; k++) acc += a.Val[k] * x[a.Idx[k]];
            y[i] = acc;
        }
    }

    private static void SpMVRange(int r0, int r1, int* ptr, int* idx, double* val, double* x, double* y)
    {
        for (var i = r0; i < r1; i++)
        {
            double acc0 = 0, acc1 = 0;
            int k = ptr[i], end = ptr[i + 1];
            for (; k + 1 < end; k += 2)
            {
                acc0 += val[k] * x[idx[k]];
                acc1 += val[k + 1] * x[idx[k + 1]];
            }
            if (k < end) acc0 += val[k] * x[idx[k]];
            y[i] = acc0 + acc1;
        }
    }

    /// <summary>Row boundaries splitting the structure into about <paramref name="blocks"/> chunks of equal non-zero count.</summary>
    internal static int[] BalancedRowBlocks(int[] ptr, int major, int blocks)
    {
        var nnz = ptr[major];
        blocks = Math.Max(1, Math.Min(blocks, major));
        var bounds = new List<int>(blocks + 1) { 0 };
        for (var b = 1; b < blocks; b++)
        {
            var target = (long)nnz * b / blocks;
            var lo = Array.BinarySearch(ptr, 0, major + 1, (int)target);
            if (lo < 0) lo = ~lo;
            lo = Math.Clamp(lo, bounds[^1], major);
            if (lo > bounds[^1]) bounds.Add(lo);
        }
        if (bounds[^1] != major) bounds.Add(major);
        return bounds.ToArray();
    }

    /// <summary>
    /// <c>y = Aᵀ x</c> for a compressed-row structure: a scatter, so serial. A parallel version with
    /// per-thread buffers and a column-wise reduction was measured and lost at every size (0.55–0.89×
    /// from 450k to 1.8M non-zeros): clearing and summing the buffers costs more memory traffic
    /// than the scatter it splits.
    /// </summary>
    internal static void SpMVTransposed(in CompressedArrays a, ReadOnlySpan<double> x, Span<double> y)
    {
        y.Clear();
        fixed (int* ptr = a.Ptr)
        fixed (int* idx = a.Idx)
        fixed (double* val = a.Val)
        fixed (double* py = y)
        {
            for (var i = 0; i < a.Major; i++)
            {
                var xi = x[i];
                if (xi == 0.0) continue;
                for (int k = ptr[i], end = ptr[i + 1]; k < end; k++) py[idx[k]] += val[k] * xi;
            }
        }
    }

    /// <summary>
    /// <c>C = A B</c> (both compressed-row) by Gustavson's row-by-row algorithm in two passes: a
    /// symbolic pass counts each output row so the result is allocated exactly once, then a
    /// numeric pass fills it. Both run in parallel over row blocks with a per-thread dense
    /// accumulator and marker. Output rows are sorted; exact-zero results are dropped like SciPy.
    /// </summary>
    internal static CompressedArrays SpGemm(in CompressedArrays a, in CompressedArrays b)
    {
        int m = a.Major, n = b.Minor;
        var aPtr = a.Ptr; var aIdx = a.Idx; var aVal = a.Val;
        var bPtr = b.Ptr; var bIdx = b.Idx; var bVal = b.Val;

        long flops = 0;
        for (var k = 0; k < aPtr[m]; k++) flops += bPtr[aIdx[k] + 1] - bPtr[aIdx[k]];

        var rowCount = new int[m + 1];
        var parallel = flops >= ParallelSpGemmThreshold && Environment.ProcessorCount > 1 && m > 1;
        var blocks = parallel ? BalancedRowBlocks(aPtr, m, Environment.ProcessorCount * 4) : [0, m];

        void Symbolic(int r0, int r1, int[] mask)
        {
            for (var i = r0; i < r1; i++)
            {
                var count = 0;
                for (var p = aPtr[i]; p < aPtr[i + 1]; p++)
                {
                    var j = aIdx[p];
                    for (var q = bPtr[j]; q < bPtr[j + 1]; q++)
                    {
                        var c = bIdx[q];
                        if (mask[c] != i) { mask[c] = i; count++; }
                    }
                }
                rowCount[i + 1] = count;
            }
        }

        if (parallel)
            Parallel.For(0, blocks.Length - 1, () => NewMask(n), (blk, _, mask) => { Symbolic(blocks[blk], blocks[blk + 1], mask); return mask; }, _ => { });
        else
            Symbolic(0, m, NewMask(n));

        for (var i = 0; i < m; i++) rowCount[i + 1] += rowCount[i];
        var ptr = rowCount;
        var idx = new int[ptr[m]];
        var val = new double[ptr[m]];
        var kept = new int[m];

        void Numeric(int r0, int r1, (int[] Mask, double[] Acc) ws)
        {
            var (mask, acc) = ws;
            for (var i = r0; i < r1; i++)
            {
                var w = ptr[i];
                for (var p = aPtr[i]; p < aPtr[i + 1]; p++)
                {
                    var j = aIdx[p];
                    var av = aVal[p];
                    for (var q = bPtr[j]; q < bPtr[j + 1]; q++)
                    {
                        var c = bIdx[q];
                        if (mask[c] != i) { mask[c] = i; idx[w++] = c; acc[c] = av * bVal[q]; }
                        else acc[c] += av * bVal[q];
                    }
                }
                var row = idx.AsSpan(ptr[i], w - ptr[i]);
                row.Sort();
                var write = ptr[i];
                foreach (var c in row)
                {
                    var v = acc[c];
                    if (v == 0.0) continue;
                    idx[write] = c;
                    val[write++] = v;
                }
                kept[i] = write - ptr[i];
            }
        }

        if (parallel)
            Parallel.For(0, blocks.Length - 1, () => (NewMask(n), new double[n]), (blk, _, ws) => { Numeric(blocks[blk], blocks[blk + 1], ws); return ws; }, _ => { });
        else
            Numeric(0, m, (NewMask(n), new double[n]));

        // Compact rows that lost exact-zero results (rare: needs exact cancellation).
        var total = 0;
        for (var i = 0; i < m; i++) total += kept[i];
        if (total == ptr[m]) return new CompressedArrays(m, n, ptr, idx, val);

        var nptr = new int[m + 1];
        var nidx = new int[total];
        var nval = new double[total];
        for (var i = 0; i < m; i++)
        {
            Array.Copy(idx, ptr[i], nidx, nptr[i], kept[i]);
            Array.Copy(val, ptr[i], nval, nptr[i], kept[i]);
            nptr[i + 1] = nptr[i] + kept[i];
        }
        return new CompressedArrays(m, n, nptr, nidx, nval);
    }

    private static int[] NewMask(int n)
    {
        var mask = new int[n];
        Array.Fill(mask, -1);
        return mask;
    }

    /// <summary><c>C = A B</c> for compressed-row <c>A</c> and a dense row-major <c>B</c> with <paramref name="k"/> columns.</summary>
    internal static void SpMM(in CompressedArrays a, double[] b, int k, double[] c)
    {
        var ptr = a.Ptr; var idx = a.Idx; var val = a.Val;
        void Rows(int r0, int r1)
        {
            for (var i = r0; i < r1; i++)
            {
                var row = c.AsSpan(i * k, k);
                row.Clear();
                for (var p = ptr[i]; p < ptr[i + 1]; p++)
                    Gravicode.Science.GraviNum.LinAlg.AxpySpan(val[p], b.AsSpan(idx[p] * k, k), row);
            }
        }
        if ((long)ptr[a.Major] * k < ParallelSpMVThreshold || Environment.ProcessorCount == 1) { Rows(0, a.Major); return; }
        var bounds = BalancedRowBlocks(ptr, a.Major, Environment.ProcessorCount * 4);
        Parallel.For(0, bounds.Length - 1, blk => Rows(bounds[blk], bounds[blk + 1]));
    }

    /// <summary>Builds a canonical compressed structure from coordinate triplets (duplicates summed).</summary>
    internal static CompressedArrays FromTriplets(int major, int minor, ReadOnlySpan<int> rows, ReadOnlySpan<int> cols, ReadOnlySpan<double> vals)
    {
        var nnz = rows.Length;
        var ptr = new int[major + 1];
        for (var k = 0; k < nnz; k++)
        {
            var r = rows[k];
            if ((uint)r >= (uint)major) throw new ArgumentException($"Row index {r} out of range for {major} rows.");
            if ((uint)cols[k] >= (uint)minor) throw new ArgumentException($"Column index {cols[k]} out of range for {minor} columns.");
            ptr[r + 1]++;
        }
        for (var i = 0; i < major; i++) ptr[i + 1] += ptr[i];
        var next = new int[major];
        Array.Copy(ptr, next, major);
        var idx = new int[nnz];
        var val = new double[nnz];
        for (var k = 0; k < nnz; k++)
        {
            var dst = next[rows[k]]++;
            idx[dst] = cols[k];
            val[dst] = vals[k];
        }
        return Canonicalize(new CompressedArrays(major, minor, ptr, idx, val));
    }
}
