using System.Buffers.Binary;

namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// Estimates the facial transformation matrix (the rigid pose plus scale that maps MediaPipe's
/// canonical face model into the camera's metric space) from face mesh landmarks. A port of
/// MediaPipe's <c>GeometryPipeline</c> (<c>ScreenToMetricSpaceConverter</c> with a perspective camera of
/// 63° vertical field of view, landmark input source, top-left origin) and its weighted orthogonal
/// Procrustes solver.
/// </summary>
public sealed class FaceGeometry
{
    private const float VerticalFovDegrees = 63f;
    private const float NearPlane = 1f;

    private static readonly Lazy<FaceGeometry> s_default = new(() =>
    {
        using var stream = typeof(FaceGeometry).Assembly.GetManifestResourceStream("MediaPipeNet.Tasks.Vision.Resources.face_geometry.bin")
            ?? throw new MediaPipeException("Embedded resource face_geometry.bin is missing.");
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return Parse(ms.ToArray());
    });

    private readonly double[] _canonical; // 3 x N, row-major (x row, y row, z row)
    private readonly double[] _sqrtWeights;

    private FaceGeometry(double[] canonical, double[] weights)
    {
        LandmarkCount = weights.Length;
        _canonical = canonical;
        _sqrtWeights = weights.Select(Math.Sqrt).ToArray();
    }

    /// <summary>The canonical face model shipped with MediaPipe (468 vertices, 33 weighted Procrustes landmarks).</summary>
    public static FaceGeometry Default => s_default.Value;

    /// <summary>Number of landmarks used (the first 468 face mesh points; iris points are ignored).</summary>
    public int LandmarkCount { get; }

    /// <summary>
    /// Parses the binary geometry produced by <c>tools/model-conversion/extract_face_geometry.py</c>:
    /// int32 count, count × (x, y, z) float32 canonical landmarks in centimeters, count float32 weights.
    /// </summary>
    public static FaceGeometry Parse(ReadOnlySpan<byte> data)
    {
        int n = BinaryPrimitives.ReadInt32LittleEndian(data);
        if (n <= 0 || data.Length < 4 + n * 16) throw new MediaPipeException("Invalid face geometry data.");
        var canonical = new double[3 * n];
        var weights = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++)
                canonical[c * n + i] = BinaryPrimitives.ReadSingleLittleEndian(data[(4 + (3 * i + c) * 4)..]);
            weights[i] = BinaryPrimitives.ReadSingleLittleEndian(data[(4 + 12 * n + 4 * i)..]);
        }
        return new FaceGeometry(canonical, weights);
    }

    /// <summary>
    /// Computes the 4×4 facial transformation matrix (row-major, column-vector convention: the last
    /// column holds the translation in centimeters) for one face, or null when the landmarks are
    /// degenerate. <paramref name="landmarks"/> must hold at least <see cref="LandmarkCount"/> points.
    /// </summary>
    public float[]? EstimateTransform(IReadOnlyList<NormalizedLandmark> landmarks, int imageWidth, int imageHeight)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        int n = LandmarkCount;
        if (landmarks.Count < n) throw new ArgumentException($"Expected at least {n} landmarks.", nameof(landmarks));
        if (imageWidth <= 0 || imageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(imageWidth));
        if (IsTooCompact(landmarks, n)) return null;

        // Perspective camera frustum at the near plane.
        double heightAtNear = 2.0 * NearPlane * Math.Tan(0.5 * VerticalFovDegrees * Math.PI / 180.0);
        double widthAtNear = imageWidth * heightAtNear / imageHeight;
        double left = -0.5 * widthAtNear, bottom = -0.5 * heightAtNear;

        // ProjectXY (origin at the top-left corner, so y is flipped).
        var screen = new double[3 * n];
        for (int i = 0; i < n; i++)
        {
            var l = landmarks[i];
            screen[i] = l.X * widthAtNear + left;
            screen[n + i] = (1.0 - l.Y) * heightAtNear + bottom;
            screen[2 * n + i] = l.Z * widthAtNear;
        }
        double depthOffset = 0;
        for (int i = 0; i < n; i++) depthOffset += screen[2 * n + i];
        depthOffset /= n;

        var intermediate = (double[])screen.Clone();
        ChangeHandedness(intermediate, n);
        double firstScale = EstimateScale(intermediate) ?? double.NaN;
        if (double.IsNaN(firstScale)) return null;

        intermediate = (double[])screen.Clone();
        MoveAndRescaleZ(intermediate, n, depthOffset, firstScale);
        UnprojectXY(intermediate, n);
        ChangeHandedness(intermediate, n);
        double secondScale = EstimateScale(intermediate) ?? double.NaN;
        if (double.IsNaN(secondScale)) return null;

        MoveAndRescaleZ(screen, n, depthOffset, firstScale * secondScale);
        UnprojectXY(screen, n);
        ChangeHandedness(screen, n);
        var transform = SolveWeightedOrthogonal(screen);
        if (transform is null) return null;
        var result = new float[16];
        for (int i = 0; i < 16; i++) result[i] = (float)transform[i];
        return result;
    }

    private double? EstimateScale(double[] landmarks)
    {
        var m = SolveWeightedOrthogonal(landmarks);
        return m is null ? null : Math.Sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
    }

    private static void MoveAndRescaleZ(double[] p, int n, double depthOffset, double scale)
    {
        for (int i = 0; i < n; i++) p[2 * n + i] = (p[2 * n + i] - depthOffset + NearPlane) / scale;
    }

    private static void UnprojectXY(double[] p, int n)
    {
        for (int i = 0; i < n; i++)
        {
            p[i] = p[i] * p[2 * n + i] / NearPlane;
            p[n + i] = p[n + i] * p[2 * n + i] / NearPlane;
        }
    }

    private static void ChangeHandedness(double[] p, int n)
    {
        for (int i = 0; i < n; i++) p[2 * n + i] = -p[2 * n + i];
    }

    private static bool IsTooCompact(IReadOnlyList<NormalizedLandmark> l, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++)
        {
            mx += (l[i].X - mx) / (i + 1);
            my += (l[i].Y - my) / (i + 1);
        }
        double max = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = l[i].X - mx, dy = l[i].Y - my;
            max = Math.Max(max, dx * dx + dy * dy);
        }
        return Math.Sqrt(max) <= 1e-3;
    }

    /// <summary>
    /// Weighted orthogonal Procrustes (Akca 2003, §2.4): finds scale·R and t minimizing
    /// Σ wᵢ‖s·R·srcᵢ + t − dstᵢ‖² from the canonical model to <paramref name="targets"/> (3 × N, row-major).
    /// Returns a row-major 4×4 matrix, or null for a degenerate problem.
    /// </summary>
    private double[]? SolveWeightedOrthogonal(double[] targets)
    {
        int n = LandmarkCount;
        var sw = _sqrtWeights;
        var src = _canonical;
        double totalWeight = 0;
        for (int i = 0; i < n; i++) totalWeight += sw[i] * sw[i];

        // Weighted source center of mass.
        Span<double> com = stackalloc double[3];
        for (int r = 0; r < 3; r++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) s += src[r * n + i] * sw[i] * sw[i];
            com[r] = s / totalWeight;
        }

        // Design matrix: weighted_targets * centered_weighted_sources^T (3 × 3).
        var design = new double[9];
        double denominator = 0;
        for (int i = 0; i < n; i++)
        {
            double w = sw[i];
            if (w == 0) continue;
            for (int r = 0; r < 3; r++)
            {
                double ws = src[r * n + i] * w;
                double cws = ws - com[r] * w;
                denominator += cws * ws;
                for (int c = 0; c < 3; c++) design[c * 3 + r] += targets[c * n + i] * w * cws;
            }
        }
        double norm = Math.Sqrt(design.Sum(v => v * v));
        if (norm <= 1e-9 || denominator <= 1e-9) return null;

        Svd3(design, out var u, out var v);
        // Disallow reflections: det(U)·det(V) < 0 → flip U's column of the smallest singular value.
        if (Det3(u) * Det3(v) < 0)
            for (int r = 0; r < 3; r++) u[r * 3 + 2] = -u[r * 3 + 2];
        var rot = new double[9]; // U · Vᵀ
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                rot[r * 3 + c] = u[r * 3] * v[c * 3] + u[r * 3 + 1] * v[c * 3 + 1] + u[r * 3 + 2] * v[c * 3 + 2];

        // Optimal scale: Σ (R · centered_weighted_sources) ∘ weighted_targets / denominator.
        double numerator = 0;
        for (int i = 0; i < n; i++)
        {
            double w = sw[i];
            if (w == 0) continue;
            double c0 = (src[i] - com[0]) * w, c1 = (src[n + i] - com[1]) * w, c2 = (src[2 * n + i] - com[2]) * w;
            for (int r = 0; r < 3; r++)
                numerator += (rot[r * 3] * c0 + rot[r * 3 + 1] * c1 + rot[r * 3 + 2] * c2) * targets[r * n + i] * w;
        }
        double scale = numerator / denominator;
        if (scale <= 1e-9) return null;

        // Translation: weighted mean of target − scale·R·source.
        var m = new double[16];
        for (int r = 0; r < 3; r++)
        {
            double t = 0;
            for (int i = 0; i < n; i++)
            {
                double w2 = sw[i] * sw[i];
                if (w2 == 0) continue;
                double rs = rot[r * 3] * src[i] + rot[r * 3 + 1] * src[n + i] + rot[r * 3 + 2] * src[2 * n + i];
                t += (targets[r * n + i] - scale * rs) * w2;
            }
            for (int c = 0; c < 3; c++) m[r * 4 + c] = scale * rot[r * 3 + c];
            m[r * 4 + 3] = t / totalWeight;
        }
        m[15] = 1;
        return m;
    }

    private static double Det3(double[] a) =>
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);

    /// <summary>
    /// SVD of a 3×3 matrix A = U·diag(σ)·Vᵀ by one-sided Jacobi rotations, singular values sorted in
    /// descending order (row-major U and V).
    /// </summary>
    internal static void Svd3(double[] a, out double[] u, out double[] v)
    {
        var w = (double[])a.Clone();
        v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        for (int sweep = 0; sweep < 60; sweep++)
        {
            double off = 0;
            for (int p = 0; p < 2; p++)
            {
                for (int q = p + 1; q < 3; q++)
                {
                    double alpha = 0, beta = 0, gamma = 0;
                    for (int r = 0; r < 3; r++)
                    {
                        alpha += w[r * 3 + p] * w[r * 3 + p];
                        beta += w[r * 3 + q] * w[r * 3 + q];
                        gamma += w[r * 3 + p] * w[r * 3 + q];
                    }
                    if (Math.Abs(gamma) <= 1e-15 * Math.Sqrt(alpha * beta)) continue;
                    off = Math.Max(off, Math.Abs(gamma) / Math.Sqrt(alpha * beta));
                    double zeta = (beta - alpha) / (2 * gamma);
                    double t = Math.Sign(zeta) / (Math.Abs(zeta) + Math.Sqrt(1 + zeta * zeta));
                    if (zeta == 0) t = 1;
                    double c = 1 / Math.Sqrt(1 + t * t), s = c * t;
                    for (int r = 0; r < 3; r++)
                    {
                        double wp = w[r * 3 + p], wq = w[r * 3 + q];
                        w[r * 3 + p] = c * wp - s * wq;
                        w[r * 3 + q] = s * wp + c * wq;
                        double vp = v[r * 3 + p], vq = v[r * 3 + q];
                        v[r * 3 + p] = c * vp - s * vq;
                        v[r * 3 + q] = s * vp + c * vq;
                    }
                }
            }
            if (off < 1e-15) break;
        }

        // Singular values are the column norms; sort descending.
        Span<double> sigma = stackalloc double[3];
        for (int c = 0; c < 3; c++) sigma[c] = Math.Sqrt(w[c] * w[c] + w[3 + c] * w[3 + c] + w[6 + c] * w[6 + c]);
        int[] order = [0, 1, 2];
        var sig = sigma.ToArray();
        Array.Sort(order, (x, y) => sig[y].CompareTo(sig[x]));

        u = new double[9];
        var vs = new double[9];
        for (int k = 0; k < 3; k++)
        {
            int c = order[k];
            for (int r = 0; r < 3; r++)
            {
                vs[r * 3 + k] = v[r * 3 + c];
                u[r * 3 + k] = sig[c] > 1e-12 ? w[r * 3 + c] / sig[c] : 0;
            }
        }
        v = vs;
        // A rank-deficient input leaves U's last column undefined: complete the orthonormal basis.
        if (sig[order[2]] <= 1e-12)
        {
            u[2] = u[3] * u[7] - u[6] * u[4];
            u[5] = u[6] * u[1] - u[0] * u[7];
            u[8] = u[0] * u[4] - u[3] * u[1];
        }
    }
}
