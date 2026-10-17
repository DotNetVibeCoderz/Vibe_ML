namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Symbolic analysis shared by the supernodal LU and Cholesky: the elimination tree of a
/// symmetric pattern under a fill-reducing order, its postorder, column counts, fundamental
/// supernodes with relaxed amalgamation, and the row structure ("border") of every front.
/// </summary>
/// <remarks>
/// Everything here is in the final elimination order: variable <c>v</c> is the v-th eliminated,
/// and <see cref="Order"/> maps it back to the index of the matrix that was analysed. Supernodes
/// are contiguous ranges of variables, numbered in postorder, so a parent always comes after all
/// of its children and the children's contribution blocks are the top of a stack when the parent
/// is assembled.
/// </remarks>
internal sealed class SparseSymbolic
{
    /// <summary>Order of the matrix.</summary>
    internal int N;

    /// <summary><c>Order[v]</c> = analysed-matrix index of variable v.</summary>
    internal int[] Order = [];

    /// <summary>Inverse of <see cref="Order"/>.</summary>
    internal int[] OrderInv = [];

    /// <summary>Supernode s owns variables <c>SnStart[s] .. SnStart[s+1]−1</c>.</summary>
    internal int[] SnStart = [];

    /// <summary>Parent supernode, −1 at a root.</summary>
    internal int[] SnParent = [];

    /// <summary>Number of child supernodes.</summary>
    internal int[] SnChildren = [];

    /// <summary>Border rows of supernode s: <c>BorderIdx[BorderPtr[s] .. BorderPtr[s+1])</c>, ascending variables.</summary>
    internal int[] BorderPtr = [];

    /// <summary>See <see cref="BorderPtr"/>.</summary>
    internal int[] BorderIdx = [];

    /// <summary>Number of supernodes.</summary>
    internal int Count => SnParent.Length;

    /// <summary>Entries of the lower-triangular factor including the diagonal and amalgamation zeros.</summary>
    internal long FactorEntries;

    /// <summary>
    /// Analyses the symmetric pattern <c>(sp, si)</c> (CSC, both triangles; diagonal ignored) under
    /// the ordering <paramref name="order"/> (<c>order[k]</c> = index eliminated k-th).
    /// </summary>
    internal static SparseSymbolic Analyze(int n, int[] sp, int[] si, int[] order, bool relax = true)
    {
        // First pass: etree in the given order, then postorder it and re-permute, so every
        // subtree is a contiguous range (needed for supernodes and the contribution-block stack).
        var (cp, ci) = Permute(n, sp, si, order);
        var parent = Etree(n, cp, ci);
        var post = Postorder(n, parent);
        var order2 = new int[n];
        for (var k = 0; k < n; k++) order2[k] = order[post[k]];
        (cp, ci) = Permute(n, sp, si, order2);
        parent = Etree(n, cp, ci);

        // Column counts (off-diagonal entries of each column of L) from row subtrees: row k of L
        // is the union of etree paths from each i < k with C(i, k) ≠ 0 up to k. O(nnz(L)).
        var count = new int[n];
        var mark = new int[n];
        Array.Fill(mark, -1);
        for (var k = 0; k < n; k++)
        {
            mark[k] = k;
            for (var p = cp[k]; p < cp[k + 1]; p++)
            {
                var j = ci[p];
                if (j >= k) continue;
                while (mark[j] != k)
                {
                    mark[j] = k;
                    count[j]++;
                    j = parent[j];
                }
            }
        }
        var nchild = new int[n];
        for (var j = 0; j < n; j++) if (parent[j] >= 0) nchild[parent[j]]++;

        // Fundamental supernodes, merged on the fly with relaxed amalgamation.
        var starts = new List<int>();
        var nzReal = new List<long>(); // true entries of L in the supernode's columns (incl. diagonal)
        var cols = new List<int>();
        var j0 = 0;
        while (j0 < n)
        {
            var j1 = j0;
            while (j1 + 1 < n && parent[j1] == j1 + 1 && nchild[j1 + 1] == 1 && count[j1] == count[j1 + 1] + 1) j1++;
            long nz = 0;
            for (var j = j0; j <= j1; j++) nz += count[j] + 1;
            var ncol = j1 - j0 + 1;
            var border = count[j1];
            if (relax)
            {
                // Merge the previous supernode (the last child, adjacent in postorder) while the
                // extra explicit zeros stay acceptable (CHOLMOD's nrelax/zrelax defaults).
                while (starts.Count > 0)
                {
                    var s = starts.Count - 1;
                    var sLast = starts[s] + cols[s] - 1;
                    if (sLast != j0 - 1 || parent[sLast] < j0 || parent[sLast] > j1) break;
                    var nc = cols[s] + ncol;
                    var front = nc + border;
                    var entries = (long)nc * front - (long)nc * (nc - 1) / 2;
                    var real = nzReal[s] + nz;
                    var zeros = (double)(entries - real) / entries;
                    var ok = nc <= 4 || (nc <= 16 && zeros < 0.8) || (nc <= 48 && zeros < 0.1) || zeros < 0.05;
                    if (!ok) break;
                    j0 = starts[s];
                    ncol = nc;
                    nz = real;
                    starts.RemoveAt(s); cols.RemoveAt(s); nzReal.RemoveAt(s);
                }
            }
            starts.Add(j0); cols.Add(ncol); nzReal.Add(nz);
            j0 = j1 + 1;
        }

        var ns = starts.Count;
        var snStart = new int[ns + 1];
        for (var s = 0; s < ns; s++) snStart[s] = starts[s];
        snStart[ns] = n;
        var snOf = new int[n];
        for (var s = 0; s < ns; s++)
            for (var j = snStart[s]; j < snStart[s + 1]; j++) snOf[j] = s;
        var snParent = new int[ns];
        var snChildren = new int[ns];
        var borderPtr = new int[ns + 1];
        long entriesTotal = 0;
        for (var s = 0; s < ns; s++)
        {
            var last = snStart[s + 1] - 1;
            snParent[s] = parent[last] < 0 ? -1 : snOf[parent[last]];
            if (snParent[s] >= 0) snChildren[snParent[s]]++;
            borderPtr[s + 1] = borderPtr[s] + count[last];
            var nc = snStart[s + 1] - snStart[s];
            var front = nc + count[last];
            entriesTotal += (long)nc * front - (long)nc * (nc - 1) / 2;
        }

        // Border rows: original entries below the supernode plus the children's borders.
        var borderIdx = new int[borderPtr[ns]];
        var childHead = new int[ns];
        var childNext = new int[ns];
        Array.Fill(childHead, -1);
        for (var s = ns - 1; s >= 0; s--)
            if (snParent[s] >= 0) { childNext[s] = childHead[snParent[s]]; childHead[snParent[s]] = s; }
        Array.Fill(mark, -1);
        for (var s = 0; s < ns; s++)
        {
            int f = snStart[s], l = snStart[s + 1] - 1;
            var q = borderPtr[s];
            for (var j = f; j <= l; j++)
                for (var p = cp[j]; p < cp[j + 1]; p++)
                {
                    var i = ci[p];
                    if (i > l && mark[i] != s) { mark[i] = s; borderIdx[q++] = i; }
                }
            for (var c = childHead[s]; c != -1; c = childNext[c])
                for (var p = borderPtr[c]; p < borderPtr[c + 1]; p++)
                {
                    var i = borderIdx[p];
                    if (i > l && mark[i] != s) { mark[i] = s; borderIdx[q++] = i; }
                }
            if (q != borderPtr[s + 1]) throw new InvalidOperationException("Supernodal analysis: border size disagrees with the column count.");
            Array.Sort(borderIdx, borderPtr[s], q - borderPtr[s]);
        }

        var orderInv = new int[n];
        for (var k = 0; k < n; k++) orderInv[order2[k]] = k;
        return new SparseSymbolic
        {
            N = n, Order = order2, OrderInv = orderInv, SnStart = snStart, SnParent = snParent, SnChildren = snChildren,
            BorderPtr = borderPtr, BorderIdx = borderIdx, FactorEntries = entriesTotal,
        };
    }

    /// <summary><c>C = P S Pᵀ</c> for the symmetric pattern S, with <c>order[k]</c> = old index of new k.</summary>
    private static (int[] Ptr, int[] Idx) Permute(int n, int[] sp, int[] si, int[] order)
    {
        var pinv = new int[n];
        for (var k = 0; k < n; k++) pinv[order[k]] = k;
        var cp = new int[n + 1];
        for (var k = 0; k < n; k++) cp[k + 1] = cp[k] + sp[order[k] + 1] - sp[order[k]];
        var ci = new int[cp[n]];
        for (var k = 0; k < n; k++)
        {
            var j = order[k];
            var q = cp[k];
            for (var p = sp[j]; p < sp[j + 1]; p++) ci[q++] = pinv[si[p]];
        }
        return (cp, ci);
    }

    /// <summary>Elimination tree (Liu's algorithm with path compression; CSparse <c>cs_etree</c>).</summary>
    internal static int[] Etree(int n, int[] cp, int[] ci)
    {
        var parent = new int[n];
        var ancestor = new int[n];
        for (var k = 0; k < n; k++)
        {
            parent[k] = -1;
            ancestor[k] = -1;
            for (var p = cp[k]; p < cp[k + 1]; p++)
            {
                var i = ci[p];
                while (i != -1 && i < k)
                {
                    var inext = ancestor[i];
                    ancestor[i] = k;
                    if (inext == -1) parent[i] = k;
                    i = inext;
                }
            }
        }
        return parent;
    }

    /// <summary>Postorder of a forest (children visited in increasing order).</summary>
    internal static int[] Postorder(int n, int[] parent)
    {
        var head = new int[n];
        var next = new int[n];
        var stack = new int[n];
        var post = new int[n];
        Array.Fill(head, -1);
        for (var j = n - 1; j >= 0; j--)
        {
            if (parent[j] == -1) continue;
            next[j] = head[parent[j]];
            head[parent[j]] = j;
        }
        var k = 0;
        for (var j = 0; j < n; j++)
        {
            if (parent[j] != -1) continue;
            var top = 0;
            stack[0] = j;
            while (top >= 0)
            {
                var p = stack[top];
                var i = head[p];
                if (i == -1) { top--; post[k++] = p; }
                else { head[p] = next[i]; stack[++top] = i; }
            }
        }
        return post;
    }
}
