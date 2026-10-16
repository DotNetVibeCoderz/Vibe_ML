using System.Diagnostics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Options of <c>Optimize.Milp</c>.</summary>
public sealed record MilpOptions
{
    /// <summary>Wall-clock limit in seconds. Default unlimited.</summary>
    public double? TimeLimit { get; init; }
    /// <summary>Branch-and-bound node limit. Default 1 000 000.</summary>
    public int NodeLimit { get; init; } = 1_000_000;
    /// <summary>Relative gap at which a node is pruned against the incumbent. Default 1e-4 (HiGHS's default).</summary>
    public double MipRelGap { get; init; } = 1e-4;
    /// <summary>Distance from an integer within which a value counts as integral. Default 1e-6.</summary>
    public double IntegralityTol { get; init; } = 1e-6;
}

/// <summary>Result of <c>Optimize.Milp</c>.</summary>
public sealed class MilpResult
{
    /// <summary>Best solution found (null when none).</summary>
    public double[]? X { get; init; }
    /// <summary>Objective at <see cref="X"/>.</summary>
    public double? Fun { get; init; }
    /// <summary>Status 0.</summary>
    public bool Success { get; init; }
    /// <summary>0 optimal, 1 limit reached, 2 infeasible, 3 unbounded, 4 other.</summary>
    public int Status { get; init; }
    /// <summary>Message.</summary>
    public string Message { get; init; } = "";
    /// <summary>Nodes explored.</summary>
    public int MipNodeCount { get; init; }
    /// <summary>Lower bound on the optimum.</summary>
    public double? MipDualBound { get; init; }
    /// <summary>(Fun − bound) / |Fun|.</summary>
    public double? MipGap { get; init; }
}

public static partial class Optimize
{
    /// <summary>
    /// Mixed-integer linear programming, <c>scipy.optimize.milp</c>: minimise cᵀx subject to the
    /// linear constraints, bounds (default 0 ≤ x &lt; ∞) and integrality (0 continuous, 1 integer).
    /// Branch and bound over the bounded simplex: depth-first until an incumbent exists, then best
    /// bound; branching on the most fractional variable.
    /// </summary>
    public static MilpResult Milp(double[] c, int[]? integrality = null, Bounds? bounds = null, IReadOnlyList<LinearConstraint>? constraints = null, MilpOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(c);
        var n = c.Length;
        var integ = integrality ?? new int[n];
        if (integ.Length != n) throw new ArgumentException("integrality must have one entry per variable.");
        if (integ.Any(v => v is 2 or 3)) throw new NotSupportedException("Semi-continuous variables are not supported.");
        if (integ.Any(v => v is not (0 or 1))) throw new ArgumentException("integrality entries must be 0 or 1.");
        var o = options ?? new MilpOptions();
        var lb = (double[])(bounds?.Lb ?? new double[n]).Clone();
        var ub = (double[])(bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray()).Clone();
        for (var j = 0; j < n; j++) if (integ[j] == 1) { lb[j] = Math.Ceiling(lb[j] - o.IntegralityTol); ub[j] = Math.Floor(ub[j] + o.IntegralityTol); }

        var aUb = new List<double[]>(); var bUb = new List<double>();
        var aEq = new List<double[]>(); var bEq = new List<double>();
        foreach (var lc in constraints ?? [])
        {
            if (lc.A.Shape[1] != n) throw new ArgumentException($"Constraint matrix must have {n} columns.");
            var rows = lc.A.Shape[0];
            var a = lc.A.AsContiguous().ToArray();
            for (var i = 0; i < rows; i++)
            {
                var row = a.AsSpan(i * n, n).ToArray();
                if (lc.Lb[i] == lc.Ub[i]) { aEq.Add(row); bEq.Add(lc.Ub[i]); continue; }
                if (double.IsFinite(lc.Ub[i])) { aUb.Add(row); bUb.Add(lc.Ub[i]); }
                if (double.IsFinite(lc.Lb[i])) { aUb.Add(row.Select(v => -v).ToArray()); bUb.Add(-lc.Lb[i]); }
            }
        }
        var au = aUb.SelectMany(r => r).ToArray();
        var ae = aEq.SelectMany(r => r).ToArray();
        return OptimizeMilp.Solve(c, integ, lb, ub, au, bUb.ToArray(), ae, bEq.ToArray(), o);
    }

    /// <summary>Convenience overload taking a single constraint.</summary>
    public static MilpResult Milp(double[] c, int[] integrality, Bounds? bounds, LinearConstraint constraint, MilpOptions? options = null)
        => Milp(c, integrality, bounds, [constraint], options);
}

/// <summary>Branch and bound for <c>Optimize.Milp</c>.</summary>
internal static class OptimizeMilp
{
    private sealed record Node(double[] Lb, double[] Ub, double Bound, int Depth);

    public static MilpResult Solve(double[] c, int[] integ, double[] lb, double[] ub, double[] au, double[] bu, double[] ae, double[] be, MilpOptions o)
    {
        var n = c.Length;
        var sw = Stopwatch.StartNew();
        double[]? best = null;
        var bestF = double.PositiveInfinity;
        var nodes = 0;
        var stack = new Stack<Node>();
        var heap = new PriorityQueue<Node, double>();
        var limitHit = false;
        var unbounded = false;
        stack.Push(new Node(lb, ub, double.NegativeInfinity, 0));

        OptimizeLp.Solution Lp(double[] l, double[] u)
            => OptimizeLpSimplex.Solve(new OptimizeLp.Problem(c, au, bu, bu.Length, ae, be, be.Length, l, u), 1e-9, 50000, false);

        bool Prune(double bound) => best is not null && bound >= bestF - Math.Max(1e-9, o.MipRelGap * Math.Abs(bestF));

        while (stack.Count > 0 || heap.Count > 0)
        {
            if (nodes >= o.NodeLimit || (o.TimeLimit is { } tl && sw.Elapsed.TotalSeconds > tl)) { limitHit = true; break; }
            var node = best is null && stack.Count > 0 ? stack.Pop() : heap.Count > 0 ? heap.Dequeue() : stack.Pop();
            if (Prune(node.Bound)) continue;
            nodes++;
            var sol = Lp(node.Lb, node.Ub);
            if (sol.Status == 2) continue;
            if (sol.Status == 3)
            {
                // An unbounded relaxation with integer variables still feasible means an unbounded MILP
                // as far as this solver can tell (SciPy/HiGHS report it the same way).
                unbounded = true; break;
            }
            if (sol.Status != 0) continue;
            double f = 0;
            for (var j = 0; j < n; j++) f += c[j] * sol.X[j];
            if (Prune(f)) continue;
            var bj = -1; double bestFrac = 0;
            for (var j = 0; j < n; j++)
            {
                if (integ[j] == 0) continue;
                var frac = Math.Abs(sol.X[j] - Math.Round(sol.X[j]));
                if (frac > o.IntegralityTol && frac > bestFrac) { bestFrac = frac; bj = j; }
            }
            if (bj < 0)
            {
                if (f < bestF)
                {
                    bestF = f;
                    best = (double[])sol.X.Clone();
                    for (var j = 0; j < n; j++) if (integ[j] == 1) best[j] = Math.Round(best[j]);
                    // Move the remaining depth-first nodes to the best-bound queue.
                    while (stack.Count > 0) { var s = stack.Pop(); heap.Enqueue(s, s.Bound); }
                }
                continue;
            }
            var v = sol.X[bj];
            var downU = (double[])node.Ub.Clone(); downU[bj] = Math.Floor(v);
            var upL = (double[])node.Lb.Clone(); upL[bj] = Math.Ceiling(v);
            var down = new Node(node.Lb, downU, f, node.Depth + 1);
            var up = new Node(upL, node.Ub, f, node.Depth + 1);
            if (best is null)
            {
                // Dive towards the nearer integer first.
                if (v - Math.Floor(v) < 0.5) { stack.Push(up); stack.Push(down); }
                else { stack.Push(down); stack.Push(up); }
            }
            else { heap.Enqueue(down, f); heap.Enqueue(up, f); }
        }

        if (unbounded && best is null)
            return new MilpResult { Status = 3, Message = "The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is At upper bound)", MipNodeCount = nodes };
        double? dual = null;
        if (best is not null)
        {
            var open = stack.Select(s => s.Bound).Concat(heap.UnorderedItems.Select(t => t.Element.Bound)).Where(bv => bv < bestF).ToList();
            dual = limitHit && open.Count > 0 ? Math.Min(bestF, open.Min()) : bestF;
        }
        if (best is null)
            return limitHit
                ? new MilpResult { Status = 1, Message = "Time or node limit reached without a feasible solution.", MipNodeCount = nodes }
                : new MilpResult { Status = 2, Message = "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)", MipNodeCount = nodes };
        var gap = Math.Abs(bestF) > 0 ? (bestF - dual!.Value) / Math.Abs(bestF) : bestF - dual!.Value;
        return new MilpResult
        {
            X = best, Fun = bestF, Success = !limitHit, Status = limitHit ? 1 : 0,
            Message = limitHit ? "Time or node limit reached; the best feasible solution found is returned." : "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
            MipNodeCount = nodes, MipDualBound = dual, MipGap = gap,
        };
    }
}

public static partial class Optimize
{
    /// <summary>
    /// Solves the linear sum assignment problem (<c>scipy.optimize.linear_sum_assignment</c>) with
    /// Crouse's shortest augmenting path variant of Jonker–Volgenant, exactly as SciPy's C++. A
    /// rectangular matrix assigns every row (or column) of the smaller dimension.
    /// </summary>
    /// <returns>Row indices (ascending) and the column assigned to each.</returns>
    /// <exception cref="ArgumentException">NaN or −∞ entries (+∞ when maximising), or no finite assignment exists.</exception>
    public static (int[] RowInd, int[] ColInd) LinearSumAssignment(NdArray cost, bool maximize = false)
    {
        ArgumentNullException.ThrowIfNull(cost);
        if (cost.Rank != 2) throw new ArgumentException("expected a matrix (2-D array)");
        return OptimizeLsa.Solve(cost.AsContiguous().ToArray(), cost.Shape[0], cost.Shape[1], maximize);
    }

    /// <summary><see cref="LinearSumAssignment(NdArray, bool)"/> on a row-major buffer.</summary>
    public static (int[] RowInd, int[] ColInd) LinearSumAssignment(double[,] cost, bool maximize = false)
        => LinearSumAssignment(NdArray.FromArray(cost), maximize);
}

/// <summary>Port of SciPy's <c>rectangular_lsap.cpp</c>.</summary>
internal static class OptimizeLsa
{
    public static (int[] RowInd, int[] ColInd) Solve(double[] input, int nr, int nc, bool maximize)
    {
        if (nr == 0 || nc == 0) return ([], []);
        var transpose = nc < nr;
        double[] cost;
        if (transpose)
        {
            cost = new double[nr * nc];
            for (var i = 0; i < nr; i++) for (var j = 0; j < nc; j++) cost[j * nr + i] = input[i * nc + j];
            (nr, nc) = (nc, nr);
        }
        else cost = (double[])input.Clone();
        if (maximize) for (var i = 0; i < cost.Length; i++) cost[i] = -cost[i];
        foreach (var v in cost) if (double.IsNaN(v) || double.IsNegativeInfinity(v)) throw new ArgumentException("matrix contains invalid numeric entries");

        var u = new double[nr]; var v2 = new double[nc];
        var shortest = new double[nc];
        var path = new int[nc]; Array.Fill(path, -1);
        var col4row = new int[nr]; Array.Fill(col4row, -1);
        var row4col = new int[nc]; Array.Fill(row4col, -1);
        var sr = new bool[nr]; var sc = new bool[nc];
        var remaining = new int[nc];

        for (var curRow = 0; curRow < nr; curRow++)
        {
            var sink = AugmentingPath(nc, cost, u, v2, path, row4col, shortest, curRow, sr, sc, remaining, out var minVal);
            if (sink < 0) throw new ArgumentException("cost matrix is infeasible");
            u[curRow] += minVal;
            for (var i = 0; i < nr; i++) if (sr[i] && i != curRow) u[i] += minVal - shortest[col4row[i]];
            for (var j = 0; j < nc; j++) if (sc[j]) v2[j] -= minVal - shortest[j];
            var jj = sink;
            while (true)
            {
                var i = path[jj];
                row4col[jj] = i;
                (col4row[i], jj) = (jj, col4row[i]);
                if (i == curRow) break;
            }
        }
        var a = new int[nr]; var b = new int[nr];
        if (transpose)
        {
            var order = Enumerable.Range(0, nr).OrderBy(k => col4row[k]).ToArray();
            for (var k = 0; k < nr; k++) { a[k] = col4row[order[k]]; b[k] = order[k]; }
        }
        else for (var i = 0; i < nr; i++) { a[i] = i; b[i] = col4row[i]; }
        return (a, b);
    }

    private static int AugmentingPath(int nc, double[] cost, double[] u, double[] v, int[] path, int[] row4col, double[] shortest, int i,
        bool[] sr, bool[] sc, int[] remaining, out double pMinVal)
    {
        double minVal = 0;
        var numRemaining = nc;
        for (var it = 0; it < nc; it++) remaining[it] = nc - it - 1;
        Array.Clear(sr); Array.Clear(sc);
        Array.Fill(shortest, double.PositiveInfinity);
        var sink = -1;
        while (sink == -1)
        {
            var index = -1;
            var lowest = double.PositiveInfinity;
            sr[i] = true;
            var rowOff = i * nc;
            var ui = u[i];
            for (var it = 0; it < numRemaining; it++)
            {
                var j = remaining[it];
                var r = minVal + cost[rowOff + j] - ui - v[j];
                if (r < shortest[j]) { path[j] = i; shortest[j] = r; }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) { lowest = shortest[j]; index = it; }
            }
            minVal = lowest;
            if (double.IsPositiveInfinity(minVal)) { pMinVal = minVal; return -1; }
            var jx = remaining[index];
            if (row4col[jx] == -1) sink = jx;
            else i = row4col[jx];
            sc[jx] = true;
            remaining[index] = remaining[--numRemaining];
        }
        pMinVal = minVal;
        return sink;
    }
}
