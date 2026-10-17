using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Stats
{
    /// <summary>
    /// The 1-Wasserstein distance between two empirical distributions in d dimensions with the Euclidean
    /// ground metric, scipy's <c>wasserstein_distance_nd</c>: the optimal-transport linear programme
    /// min Σ π_ij ‖u_i − v_j‖ subject to the row sums being u's weights and the column sums v's.
    /// </summary>
    /// <param name="uValues">Shape (m, d) (or (m) for d = 1).</param>
    /// <param name="vValues">Shape (n, d) (or (n)).</param>
    /// <param name="uWeights">Non-negative weights of u (normalised); uniform when null.</param>
    /// <param name="vWeights">Non-negative weights of v; uniform when null.</param>
    /// <remarks>
    /// With equal sample sizes and uniform weights an optimal plan is a permutation (Birkhoff–von Neumann),
    /// so it is solved exactly as an assignment problem (<see cref="Optimize.LinearSumAssignment(NdArray, bool)"/>,
    /// O(n³)). Otherwise the m·n-variable LP goes to <see cref="Optimize.Linprog"/>, which is only
    /// practical for m·n up to a few thousand. One-dimensional input is reduced to
    /// <see cref="WassersteinDistance"/> as in scipy.
    /// </remarks>
    public static double WassersteinDistanceNd(NdArray uValues, NdArray vValues, double[]? uWeights = null, double[]? vWeights = null)
    {
        ArgumentNullException.ThrowIfNull(uValues);
        ArgumentNullException.ThrowIfNull(vValues);
        if (uValues.Shape.Length == 1 && vValues.Shape.Length == 1)
            return WassersteinDistance(uValues.AsContiguous().ToArray(), vValues.AsContiguous().ToArray(), uWeights, vWeights);
        var u = uValues.Shape.Length == 1 ? new NdArray(uValues.AsContiguous().ToArray(), uValues.Shape[0], 1) : uValues;
        var v = vValues.Shape.Length == 1 ? new NdArray(vValues.AsContiguous().ToArray(), vValues.Shape[0], 1) : vValues;
        if (u.Shape.Length != 2 || v.Shape.Length != 2) throw new ArgumentException("Invalid input values. The inputs must have either one dimension or two dimensions.");
        if (u.Shape[1] != v.Shape[1]) throw new ArgumentException("Invalid input values. Dimensions of inputs must be equal.");
        int m = u.Shape[0], n = v.Shape[0], d = u.Shape[1];
        if (m == 0 || n == 0) throw new ArgumentException("Distribution can't be empty.");
        if (d == 1 && uWeights == null && vWeights == null)
            return WassersteinDistance(u.AsContiguous().ToArray(), v.AsContiguous().ToArray());
        var us = u.AsContiguous().ToArray();
        var vs = v.AsContiguous().ToArray();
        foreach (var x in us.Concat(vs)) if (!double.IsFinite(x)) throw new ArgumentException("Values must be finite.");
        var cost = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < d; k++) { var t = us[i * d + k] - vs[j * d + k]; s += t * t; }
                cost[i * n + j] = Math.Sqrt(s);
            }
        if (uWeights == null && vWeights == null && m == n)
        {
            var (rows, cols) = Optimize.LinearSumAssignment(new NdArray(cost, m, n));
            var total = 0.0;
            for (var i = 0; i < rows.Length; i++) total += cost[rows[i] * n + cols[i]];
            return total / m;
        }
        var a = Normalise(uWeights, m, nameof(uWeights));
        var b = Normalise(vWeights, n, nameof(vWeights));
        // Equality constraints: m row sums and the first n − 1 column sums (the last is implied, and a
        // redundant row would make the constraint matrix rank deficient).
        var rowsEq = m + n - 1;
        var aEq = new double[rowsEq * m * n];
        var bEq = new double[rowsEq];
        for (var i = 0; i < m; i++)
        {
            for (var j = 0; j < n; j++) aEq[i * (m * n) + i * n + j] = 1;
            bEq[i] = a[i];
        }
        for (var j = 0; j < n - 1; j++)
        {
            for (var i = 0; i < m; i++) aEq[(m + j) * (m * n) + i * n + j] = 1;
            bEq[m + j] = b[j];
        }
        var res = Optimize.Linprog(cost, aEq: new NdArray(aEq, rowsEq, m * n), bEq: bEq);
        if (!res.Success) throw new ArithmeticException($"The transport linear programme failed: {res.Message}");
        return res.Fun;
    }

    private static double[] Normalise(double[]? w, int n, string name)
    {
        if (w == null) return Enumerable.Repeat(1.0 / n, n).ToArray();
        if (w.Length != n) throw new ArgumentException("Value and weight array-likes for the same empirical distribution must be of the same size.", name);
        var s = 0.0;
        foreach (var x in w)
        {
            if (!(x >= 0) || !double.IsFinite(x)) throw new ArgumentException("All weights must be non-negative and finite.", name);
            s += x;
        }
        if (!(s > 0)) throw new ArgumentException("Weight array-like sums must be positive.", name);
        return w.Select(x => x / s).ToArray();
    }
}
