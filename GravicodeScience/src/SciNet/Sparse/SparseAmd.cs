namespace Gravicode.Science.SciNet.Sparse;

/// <summary>
/// Approximate minimum degree ordering (Amestoy, Davis &amp; Duff 1996) on a symmetric pattern,
/// following the structure of Davis's CSparse <c>cs_amd</c>: a quotient graph held in one integer
/// workspace with garbage collection, approximate external degrees from set differences
/// <c>|Lₑ \ Lₖ|</c>, aggressive element absorption, mass elimination, supervariable detection by
/// hashing, and dense rows (degree above <c>10√n</c>) removed up front and ordered last.
/// </summary>
/// <remarks>
/// The output is a postorder of the assembly tree, which keeps each subtree's columns contiguous
/// — what the supernodal factorisations need anyway.
/// </remarks>
internal static class SparseAmd
{
    private static int Flip(int i) => -i - 2;

    /// <summary>
    /// Orders a symmetric pattern given as CSC (both triangles; any diagonal entries are ignored).
    /// Returns <c>p</c> with <c>p[k]</c> = node eliminated k-th.
    /// </summary>
    internal static int[] Order(int n, int[] ap, int[] ai)
    {
        if (n == 0) return [];
        // C = pattern without the diagonal, with elbow room.
        var cnz0 = 0;
        for (var j = 0; j < n; j++)
            for (var p = ap[j]; p < ap[j + 1]; p++) if (ai[p] != j) cnz0++;
        var nzmax = cnz0 + cnz0 / 5 + 2 * n + 16;
        var ci = new int[nzmax];
        var cp = new int[n + 1];
        var cnz = 0;
        for (var j = 0; j < n; j++)
        {
            cp[j] = cnz;
            for (var p = ap[j]; p < ap[j + 1]; p++) if (ai[p] != j) ci[cnz++] = ai[p];
        }
        cp[n] = cnz;

        var dense = (int)Math.Max(16, 10 * Math.Sqrt(n));
        dense = Math.Min(n - 2, dense);
        var len = new int[n + 1];
        var nv = new int[n + 1];
        var next = new int[n + 1];
        var head = new int[n + 1];
        var elen = new int[n + 1];
        var degree = new int[n + 1];
        var w = new int[n + 1];
        var hhead = new int[n + 1];
        var last = new int[n + 1];
        var perm = new int[n + 1];
        for (var k = 0; k < n; k++) len[k] = cp[k + 1] - cp[k];
        len[n] = 0;
        for (var i = 0; i <= n; i++)
        {
            head[i] = -1; last[i] = -1; next[i] = -1; hhead[i] = -1;
            nv[i] = 1; w[i] = 1; elen[i] = 0; degree[i] = len[i];
        }
        var mark = WClear(0, 0, w, n);
        elen[n] = -2;
        cp[n] = -1;
        w[n] = 0;
        var nel = 0;
        for (var i = 0; i < n; i++)
        {
            var d = degree[i];
            if (d == 0)
            {
                elen[i] = -2; nel++; cp[i] = -1; w[i] = 0;
            }
            else if (d > dense)
            {
                nv[i] = 0; elen[i] = -1; nel++; cp[i] = Flip(n); nv[n]++;
            }
            else
            {
                if (head[d] != -1) last[head[d]] = i;
                next[i] = head[d];
                head[d] = i;
            }
        }
        int mindeg = 0, lemax = 0;
        while (nel < n)
        {
            // ---- select node of minimum approximate degree
            var k = -1;
            for (; mindeg < n && (k = head[mindeg]) == -1; mindeg++) { }
            if (next[k] != -1) last[next[k]] = -1;
            head[mindeg] = next[k];
            var elenk = elen[k];
            var nvk = nv[k];
            nel += nvk;

            // ---- garbage collection
            if (elenk > 0 && cnz + mindeg >= nzmax)
            {
                for (var j = 0; j < n; j++)
                {
                    int p;
                    if ((p = cp[j]) >= 0)
                    {
                        cp[j] = ci[p];
                        ci[p] = Flip(j);
                    }
                }
                int q = 0, pp = 0;
                while (pp < cnz)
                {
                    var j = Flip(ci[pp++]);
                    if (j >= 0)
                    {
                        ci[q] = cp[j];
                        cp[j] = q++;
                        for (var k3 = 0; k3 < len[j] - 1; k3++) ci[q++] = ci[pp++];
                    }
                }
                cnz = q;
                if (cnz + mindeg >= nzmax)
                {
                    // Still no room for the new element: grow (CSparse would fail here).
                    nzmax = 2 * nzmax + n;
                    Array.Resize(ref ci, nzmax);
                }
            }

            // ---- construct new element
            var dk = 0;
            nv[k] = -nvk;
            var p0 = cp[k];
            var pk1 = elenk == 0 ? p0 : cnz;
            var pk2 = pk1;
            if (elenk != 0 && pk1 + mindeg + n >= ci.Length) Array.Resize(ref ci, Math.Max(2 * ci.Length, pk1 + 2 * n + 16));
            var pcur = p0;
            for (var k1 = 1; k1 <= elenk + 1; k1++)
            {
                int e, pj, ln;
                if (k1 > elenk)
                {
                    e = k; pj = pcur; ln = len[k] - elenk;
                }
                else
                {
                    e = ci[pcur++]; pj = cp[e]; ln = len[e];
                }
                for (var k2 = 1; k2 <= ln; k2++)
                {
                    var i = ci[pj++];
                    int nvi;
                    if ((nvi = nv[i]) <= 0) continue;
                    dk += nvi;
                    nv[i] = -nvi;
                    ci[pk2++] = i;
                    if (next[i] != -1) last[next[i]] = last[i];
                    if (last[i] != -1) next[last[i]] = next[i];
                    else head[degree[i]] = next[i];
                }
                if (e != k)
                {
                    cp[e] = Flip(k);
                    w[e] = 0;
                }
            }
            if (elenk != 0) cnz = pk2;
            nzmax = ci.Length;
            degree[k] = dk;
            cp[k] = pk1;
            len[k] = pk2 - pk1;
            elen[k] = -2;

            // ---- find set differences |Le \ Lk|
            mark = WClear(mark, lemax, w, n);
            for (var pk = pk1; pk < pk2; pk++)
            {
                var i = ci[pk];
                int eln;
                if ((eln = elen[i]) <= 0) continue;
                var nvi = -nv[i];
                var wnvi = mark - nvi;
                for (var p = cp[i]; p <= cp[i] + eln - 1; p++)
                {
                    var e = ci[p];
                    if (w[e] >= mark) w[e] -= nvi;
                    else if (w[e] != 0) w[e] = degree[e] + wnvi;
                }
            }

            // ---- degree update and element absorption
            for (var pk = pk1; pk < pk2; pk++)
            {
                var i = ci[pk];
                var p1 = cp[i];
                var p2 = p1 + elen[i] - 1;
                var pn = p1;
                long h = 0;
                var d = 0;
                for (var p = p1; p <= p2; p++)
                {
                    var e = ci[p];
                    if (w[e] != 0)
                    {
                        var dext = w[e] - mark;
                        if (dext > 0)
                        {
                            d += dext;
                            ci[pn++] = e;
                            h += e;
                        }
                        else
                        {
                            cp[e] = Flip(k); // aggressive absorption
                            w[e] = 0;
                        }
                    }
                }
                elen[i] = pn - p1 + 1;
                var p3 = pn;
                var p4 = p1 + len[i];
                for (var p = p2 + 1; p < p4; p++)
                {
                    var j = ci[p];
                    int nvj;
                    if ((nvj = nv[j]) <= 0) continue;
                    d += nvj;
                    ci[pn++] = j;
                    h += j;
                }
                if (d == 0)
                {
                    // mass elimination
                    cp[i] = Flip(k);
                    var nvi = -nv[i];
                    dk -= nvi;
                    nvk += nvi;
                    nel += nvi;
                    nv[i] = 0;
                    elen[i] = -1;
                }
                else
                {
                    degree[i] = Math.Min(degree[i], d);
                    ci[pn] = ci[p3];
                    ci[p3] = ci[p1];
                    ci[p1] = k;
                    len[i] = pn - p1 + 1;
                    var hh = (int)(h % n);
                    next[i] = hhead[hh];
                    hhead[hh] = i;
                    last[i] = hh;
                }
            }
            degree[k] = dk;
            lemax = Math.Max(lemax, dk);
            mark = WClear(mark + lemax, lemax, w, n);

            // ---- supernode detection
            for (var pk = pk1; pk < pk2; pk++)
            {
                var i = ci[pk];
                if (nv[i] >= 0) continue;
                var hh = last[i];
                i = hhead[hh];
                hhead[hh] = -1;
                for (; i != -1 && next[i] != -1; i = next[i], mark++)
                {
                    var ln = len[i];
                    var eln = elen[i];
                    for (var p = cp[i] + 1; p <= cp[i] + ln - 1; p++) w[ci[p]] = mark;
                    var jlast = i;
                    for (var j = next[i]; j != -1;)
                    {
                        var ok = len[j] == ln && elen[j] == eln;
                        for (var p = cp[j] + 1; ok && p <= cp[j] + ln - 1; p++)
                            if (w[ci[p]] != mark) ok = false;
                        if (ok)
                        {
                            cp[j] = Flip(i);
                            nv[i] += nv[j];
                            nv[j] = 0;
                            elen[j] = -1;
                            j = next[j];
                            next[jlast] = j;
                        }
                        else
                        {
                            jlast = j;
                            j = next[j];
                        }
                    }
                }
            }

            // ---- finalise new element
            var pout = pk1;
            for (var pk = pk1; pk < pk2; pk++)
            {
                var i = ci[pk];
                int nvi;
                if ((nvi = -nv[i]) <= 0) continue;
                nv[i] = nvi;
                var d = degree[i] + dk - nvi;
                d = Math.Min(d, n - nel - nvi);
                if (head[d] != -1) last[head[d]] = i;
                next[i] = head[d];
                last[i] = -1;
                head[d] = i;
                mindeg = Math.Min(mindeg, d);
                degree[i] = d;
                ci[pout++] = i;
            }
            nv[k] = nvk;
            if ((len[k] = pout - pk1) == 0)
            {
                cp[k] = -1;
                w[k] = 0;
            }
            if (elenk != 0) cnz = pout;
        }

        // ---- postorder the assembly tree
        for (var i = 0; i < n; i++) cp[i] = Flip(cp[i]);
        for (var j = 0; j <= n; j++) head[j] = -1;
        for (var j = n; j >= 0; j--)
        {
            if (nv[j] > 0) continue;
            next[j] = head[cp[j]];
            head[cp[j]] = j;
        }
        for (var e = n; e >= 0; e--)
        {
            if (nv[e] <= 0) continue;
            if (cp[e] != -1)
            {
                next[e] = head[cp[e]];
                head[cp[e]] = e;
            }
        }
        var kk = 0;
        for (var i = 0; i <= n; i++)
            if (cp[i] == -1) kk = TreeDfs(i, kk, head, next, perm, w);
        var result = new int[n];
        Array.Copy(perm, result, n);
        return result;
    }

    private static int WClear(int mark, int lemax, int[] w, int n)
    {
        if (mark < 2 || mark + lemax < 0 || mark + lemax > int.MaxValue / 2)
        {
            for (var k = 0; k < n; k++) if (w[k] != 0) w[k] = 1;
            mark = 2;
        }
        return mark;
    }

    private static int TreeDfs(int j, int k, int[] head, int[] next, int[] post, int[] stack)
    {
        var top = 0;
        stack[0] = j;
        while (top >= 0)
        {
            var p = stack[top];
            var i = head[p];
            if (i == -1)
            {
                top--;
                post[k++] = p;
            }
            else
            {
                head[p] = next[i];
                stack[++top] = i;
            }
        }
        return k;
    }

    /// <summary>The pattern of <c>A + Aᵀ</c> (no diagonal) of a square CSC/CSR pattern, as CSC.</summary>
    internal static (int[] Ptr, int[] Idx) SymmetricPattern(int n, ReadOnlySpan<int> ptr, ReadOnlySpan<int> idx)
    {
        var cnt = new int[n + 1];
        for (var j = 0; j < n; j++)
            for (var p = ptr[j]; p < ptr[j + 1]; p++)
            {
                var i = idx[p];
                if (i == j) continue;
                cnt[i]++; cnt[j]++;
            }
        var sp = new int[n + 1];
        for (var j = 0; j < n; j++) sp[j + 1] = sp[j] + cnt[j];
        var si = new int[sp[n]];
        var pos = new int[n];
        Array.Copy(sp, pos, n);
        for (var j = 0; j < n; j++)
            for (var p = ptr[j]; p < ptr[j + 1]; p++)
            {
                var i = idx[p];
                if (i == j) continue;
                si[pos[j]++] = i;
                si[pos[i]++] = j;
            }
        return Dedupe(n, sp, si);
    }

    /// <summary>Removes duplicate indices within each column (in place, compacting).</summary>
    internal static (int[] Ptr, int[] Idx) Dedupe(int n, int[] sp, int[] si)
    {
        var mark = new int[n];
        Array.Fill(mark, -1);
        var q = 0;
        var newP = new int[n + 1];
        for (var j = 0; j < n; j++)
        {
            newP[j] = q;
            for (var p = sp[j]; p < sp[j + 1]; p++)
            {
                var i = si[p];
                if (mark[i] == j) continue;
                mark[i] = j;
                si[q++] = i;
            }
        }
        newP[n] = q;
        return (newP, si);
    }

    /// <summary>
    /// The pattern of <c>AᵀA</c> (no diagonal) from the CSR pattern of an m×n matrix, dropping rows
    /// with more than <c>max(16, 10√n)</c> entries as COLAMD and <c>cs_amd</c> do: a dense row makes
    /// <c>AᵀA</c> dense and says nothing useful about the ordering.
    /// </summary>
    internal static (int[] Ptr, int[] Idx) AtAPattern(int m, int n, ReadOnlySpan<int> rowPtr, ReadOnlySpan<int> colIdx)
    {
        var dense = Math.Max(16, (int)(10 * Math.Sqrt(n)));
        // Columns -> rows (CSC pattern) so each column j can visit the rows it touches.
        var cp = new int[n + 1];
        for (var i = 0; i < m; i++)
        {
            if (rowPtr[i + 1] - rowPtr[i] > dense) continue;
            for (var p = rowPtr[i]; p < rowPtr[i + 1]; p++) cp[colIdx[p] + 1]++;
        }
        for (var j = 0; j < n; j++) cp[j + 1] += cp[j];
        var ri = new int[cp[n]];
        var pos = (int[])cp.Clone();
        for (var i = 0; i < m; i++)
        {
            if (rowPtr[i + 1] - rowPtr[i] > dense) continue;
            for (var p = rowPtr[i]; p < rowPtr[i + 1]; p++) ri[pos[colIdx[p]]++] = i;
        }
        var mark = new int[n];
        Array.Fill(mark, -1);
        var outP = new int[n + 1];
        var outI = new List<int>(cp[n] * 4);
        for (var j = 0; j < n; j++)
        {
            outP[j] = outI.Count;
            mark[j] = j;
            for (var p = cp[j]; p < cp[j + 1]; p++)
            {
                var i = ri[p];
                for (var q = rowPtr[i]; q < rowPtr[i + 1]; q++)
                {
                    var c = colIdx[q];
                    if (mark[c] == j) continue;
                    mark[c] = j;
                    outI.Add(c);
                }
            }
        }
        outP[n] = outI.Count;
        return (outP, outI.ToArray());
    }
}
