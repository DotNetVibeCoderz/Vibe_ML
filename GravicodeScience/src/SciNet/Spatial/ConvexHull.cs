using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Convex hull of a point set in any dimension ≥ 2 (scipy.spatial.ConvexHull, triangulated output).
/// </summary>
/// <remarks>
/// <para><b>2-D</b> uses Andrew's monotone chain; <b>3-D</b> uses quickhull with conflict lists
/// (Barber, Dobkin and Huhdanpaa 1996 — the algorithm behind Qhull). Every visibility decision
/// goes through an exact orientation predicate (<c>SpatialPredicates</c>), so coplanar and
/// collinear inputs cannot leave the hull inconsistent; a point exactly on a face counts as
/// inside, which is why only extreme points become <see cref="Vertices"/>, as with Qhull.</para>
/// <para><b>Conventions follow scipy:</b> in 2-D, <see cref="Area"/> is the perimeter and
/// <see cref="Volume"/> the enclosed area; 2-D <see cref="Vertices"/> run counter-clockwise
/// (scipy does not fix the starting vertex; here it is the lexicographically smallest (x, y));
/// in 3-D they are sorted ascending. <see cref="Equations"/>
/// rows are <c>[normal, offset]</c> with a unit outward normal, so <c>normal·x + offset &lt;= 0</c>
/// inside. 3-D facets are triangles; a flat face of the hull (a cube side) is split into
/// several, as with scipy's default <c>Qt</c> option, though not necessarily into the same
/// triangles.</para>
/// <para><b>4-D and above</b> use the general quickhull (<c>QuickhullNd</c>) with the exact
/// minor-expansion orientation predicate; facets are (d−1)-simplices, <see cref="Vertices"/> are
/// ascending, <see cref="Area"/> is the total (d−1)-volume of the boundary and
/// <see cref="Volume"/> the d-volume, as scipy reports them. The output is simplicial (Qhull's
/// <c>Qt</c>), so a flat face of the hull is split into simplices, not necessarily scipy's.</para>
/// <para>A flat input (all points in a lower-dimensional affine subspace) throws, as Qhull
/// does.</para>
/// </remarks>
public sealed class ConvexHull
{
    private readonly double[] _pts;

    /// <summary>The input points, shape (n, d).</summary>
    public NdArray Points => new((double[])_pts.Clone(), NPoints, NDim);

    /// <summary>Number of input points.</summary>
    public int NPoints { get; }

    /// <summary>Dimension of the points.</summary>
    public int NDim { get; }

    /// <summary>Indices of the hull vertices (2-D: counter-clockwise; otherwise ascending).</summary>
    public int[] Vertices { get; }

    /// <summary>Facets as point indices, [nfacet, NDim]: edges in 2-D, triangles in 3-D (outward
    /// counter-clockwise), (d−1)-simplices above (no orientation guarantee, as in scipy).</summary>
    public int[,] Simplices { get; }

    /// <summary><c>Neighbors[f, k]</c> is the facet opposite vertex <c>Simplices[f, k]</c>.</summary>
    public int[,] Neighbors { get; }

    /// <summary>Facet hyperplanes [unit outward normal, offset], shape (nfacet, NDim + 1).</summary>
    public NdArray Equations { get; }

    /// <summary>Perimeter in 2-D, surface area in 3-D, boundary (d−1)-volume above.</summary>
    public double Area { get; }

    /// <summary>Enclosed area in 2-D, volume in 3-D, d-volume above.</summary>

    /// <summary>Lower corner of the points' bounding box (scipy's <c>min_bound</c>).</summary>
    public double[] MinBound
    {
        get
        {
            var r = new double[NDim];
            Array.Fill(r, double.PositiveInfinity);
            for (var i = 0; i < NPoints; i++) for (var j = 0; j < NDim; j++) r[j] = Math.Min(r[j], _pts[i * NDim + j]);
            return r;
        }
    }

    /// <summary>Upper corner of the points' bounding box (scipy's <c>max_bound</c>).</summary>
    public double[] MaxBound
    {
        get
        {
            var r = new double[NDim];
            Array.Fill(r, double.NegativeInfinity);
            for (var i = 0; i < NPoints; i++) for (var j = 0; j < NDim; j++) r[j] = Math.Max(r[j], _pts[i * NDim + j]);
            return r;
        }
    }
    public double Volume { get; }

    /// <summary>Builds the hull of the rows of <paramref name="points"/>, shape (n, d), d ≥ 2.</summary>
    public ConvexHull(NdArray points) : this(SpatialInput.Copy(points, out var n, out var d, nameof(points)), n, d) { }

    /// <summary>Builds the hull of a rectangular array of points.</summary>
    public ConvexHull(double[,] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    /// <summary>Builds from a jagged array of points.</summary>
    public ConvexHull(double[][] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    /// <summary>Builds through the general N-d quickhull even in 3-D (tests cross-check the two).</summary>
    internal static ConvexHull GeneralNd(NdArray points) => new(SpatialInput.Copy(points, out var n, out var d, nameof(points)), n, d, forceNd: true);

    private ConvexHull(double[] pts, int n, int d, bool forceNd = false)
    {
        SpatialInput.RequireFinite(pts, "Hull points");
        _pts = pts; NPoints = n; NDim = d;
        if (d == 2)
        {
            Vertices = Hull2D(pts, n);
            var h = Vertices.Length;
            Simplices = new int[h, 2];
            Neighbors = new int[h, 2];
            var eq = new double[h * 3];
            double per = 0, area = 0;
            for (var k = 0; k < h; k++)
            {
                int a = Vertices[k], b = Vertices[(k + 1) % h];
                Simplices[k, 0] = a; Simplices[k, 1] = b;
                Neighbors[k, 0] = (k + 1) % h;      // opposite a: the edge that contains b
                Neighbors[k, 1] = (k - 1 + h) % h;  // opposite b: the edge that contains a
                double ax = pts[2 * a], ay = pts[2 * a + 1], bx = pts[2 * b], by = pts[2 * b + 1];
                double dx = bx - ax, dy = by - ay, len = Math.Sqrt(dx * dx + dy * dy);
                double nx = dy / len, ny = -dx / len;
                eq[3 * k] = nx; eq[3 * k + 1] = ny; eq[3 * k + 2] = -(nx * ax + ny * ay);
                per += len;
                area += ax * by - bx * ay;
            }
            Equations = new NdArray(eq, h, 3);
            Area = per;
            Volume = 0.5 * area;
        }
        else if (d == 3 && !forceNd)
        {
            var faces = Hull3D(pts, n);
            var f = faces.Count;
            Simplices = new int[f, 3];
            Neighbors = new int[f, 3];
            var eq = new double[f * 4];
            var used = new SortedSet<int>();
            for (var i = 0; i < f; i++)
                for (var k = 0; k < 3; k++) { Simplices[i, k] = faces[i].V[k]; Neighbors[i, k] = faces[i].N[k]; used.Add(faces[i].V[k]); }
            Vertices = [.. used];
            double cx = 0, cy = 0, cz = 0;
            foreach (var v in Vertices) { cx += pts[3 * v]; cy += pts[3 * v + 1]; cz += pts[3 * v + 2]; }
            cx /= Vertices.Length; cy /= Vertices.Length; cz /= Vertices.Length;
            double area = 0, vol = 0;
            for (var i = 0; i < f; i++)
            {
                int a = Simplices[i, 0], b = Simplices[i, 1], c = Simplices[i, 2];
                double ux = pts[3 * b] - pts[3 * a], uy = pts[3 * b + 1] - pts[3 * a + 1], uz = pts[3 * b + 2] - pts[3 * a + 2];
                double vx = pts[3 * c] - pts[3 * a], vy = pts[3 * c + 1] - pts[3 * a + 1], vz = pts[3 * c + 2] - pts[3 * a + 2];
                double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
                var len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                area += 0.5 * len;
                double ox = pts[3 * a] - cx, oy = pts[3 * a + 1] - cy, oz = pts[3 * a + 2] - cz;
                vol += (nx * ox + ny * oy + nz * oz) / 6.0;
                nx /= len; ny /= len; nz /= len;
                eq[4 * i] = nx; eq[4 * i + 1] = ny; eq[4 * i + 2] = nz;
                eq[4 * i + 3] = -(nx * pts[3 * a] + ny * pts[3 * a + 1] + nz * pts[3 * a + 2]);
            }
            Equations = new NdArray(eq, f, 4);
            Area = area;
            Volume = vol;
        }
        else if (d >= 3)
        {
            if (d > SpatialPredicates.MaxNdDimension) throw new ArgumentException($"ConvexHull supports at most {SpatialPredicates.MaxNdDimension} dimensions, got {d}.");
            var qh = new QuickhullNd(pts, n, d, lifted: false, excess: null, floatCoords: pts);
            qh.Run(Enumerable.Range(0, n).ToArray());
            var f = qh.FacetCount;
            Simplices = new int[f, d];
            Neighbors = new int[f, d];
            var used = new SortedSet<int>();
            for (var i = 0; i < f; i++)
                for (var k = 0; k < d; k++)
                {
                    Simplices[i, k] = qh.Vertices[i * d + k];
                    Neighbors[i, k] = qh.Neighbors[i * d + k];
                    used.Add(Simplices[i, k]);
                }
            Vertices = [.. used];
            var c = new double[d];
            foreach (var v in Vertices) for (var j = 0; j < d; j++) c[j] += pts[v * d + j];
            for (var j = 0; j < d; j++) c[j] /= Vertices.Length;
            double fact = 1; for (var k = 2; k < d; k++) fact *= k; // (d-1)!
            var eq = new double[f * (d + 1)];
            double area = 0, vol = 0;
            for (var i = 0; i < f; i++)
            {
                var o = i * (d + 1);
                double len = 0;
                for (var j = 0; j < d; j++) len += qh.Planes[o + j] * qh.Planes[o + j];
                len = Math.Sqrt(len);
                area += len / fact;
                var v0 = Simplices[i, 0];
                double h = 0, off = 0;
                for (var j = 0; j < d; j++)
                {
                    h += qh.Planes[o + j] * (pts[v0 * d + j] - c[j]);
                    var nj = qh.Planes[o + j] / len;
                    eq[o + j] = nj;
                    off -= nj * pts[v0 * d + j];
                }
                eq[o + d] = off;
                vol += h / (fact * d);
            }
            Equations = new NdArray(eq, f, d + 1);
            Area = area;
            Volume = vol;
        }
        else throw new ArgumentException($"ConvexHull needs points of dimension 2 or more, got {d}-D.");
    }

    // ------------------------------------------------------------------ 2-D

    internal static int[] Hull2D(double[] p, int n)
    {
        if (n < 3) throw new ArgumentException("A 2-D convex hull needs at least 3 points.");
        var idx = new int[n];
        for (var i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) =>
        {
            var c = p[2 * a].CompareTo(p[2 * b]);
            if (c != 0) return c;
            c = p[2 * a + 1].CompareTo(p[2 * b + 1]);
            return c != 0 ? c : a.CompareTo(b);
        });
        var hull = new int[2 * n];
        var k = 0;
        // Lower chain, then upper chain; a strictly positive turn is required to keep a point,
        // so collinear and duplicate points never become vertices.
        for (var t = 0; t < n; t++)
        {
            var i = idx[t];
            while (k >= 2 && Turn(p, hull[k - 2], hull[k - 1], i) <= 0) k--;
            hull[k++] = i;
        }
        var lower = k + 1;
        for (var t = n - 2; t >= 0; t--)
        {
            var i = idx[t];
            while (k >= lower && Turn(p, hull[k - 2], hull[k - 1], i) <= 0) k--;
            hull[k++] = i;
        }
        k--; // the last point repeats the first
        if (k < 3) throw new ArgumentException("Initial simplex is flat: all points are collinear (Qhull QH6154).");
        return hull[..k];
    }

    private static double Turn(double[] p, int a, int b, int c)
        => SpatialPredicates.Orient2D(p[2 * a], p[2 * a + 1], p[2 * b], p[2 * b + 1], p[2 * c], p[2 * c + 1]);

    // ------------------------------------------------------------------ 3-D quickhull

    internal sealed class Face
    {
        public readonly int[] V = new int[3];
        public readonly int[] N = new int[3];
        public List<int> Conflict = [];
        public bool Alive = true;
        public int Mark = -1;
        public double Nx, Ny, Nz, Off; // floating plane, only used to rank conflict points
    }

    private static ReadOnlySpan<double> P3(double[] p, int i) => p.AsSpan(3 * i, 3);

    /// <summary>True when q is strictly on the outer side of face f.</summary>
    private static bool Visible(double[] p, Face f, int q)
        => SpatialPredicates.Orient3D(P3(p, f.V[0]), P3(p, f.V[1]), P3(p, f.V[2]), P3(p, q)) < 0;

    private static void Plane(double[] p, Face f)
    {
        int a = f.V[0], b = f.V[1], c = f.V[2];
        double ux = p[3 * b] - p[3 * a], uy = p[3 * b + 1] - p[3 * a + 1], uz = p[3 * b + 2] - p[3 * a + 2];
        double vx = p[3 * c] - p[3 * a], vy = p[3 * c + 1] - p[3 * a + 1], vz = p[3 * c + 2] - p[3 * a + 2];
        f.Nx = uy * vz - uz * vy; f.Ny = uz * vx - ux * vz; f.Nz = ux * vy - uy * vx;
        f.Off = -(f.Nx * p[3 * a] + f.Ny * p[3 * a + 1] + f.Nz * p[3 * a + 2]);
    }

    private static double Height(double[] p, Face f, int q) => f.Nx * p[3 * q] + f.Ny * p[3 * q + 1] + f.Nz * p[3 * q + 2] + f.Off;

    internal static List<Face> Hull3D(double[] p, int n)
    {
        if (n < 4) throw new ArgumentException("A 3-D convex hull needs at least 4 points.");
        // Initial tetrahedron: extremes in x, furthest from that line, furthest from that plane.
        int i0 = 0, i1 = 0;
        for (var i = 1; i < n; i++)
        {
            if (Less(p, i, i0)) i0 = i;
            if (Less(p, i1, i)) i1 = i;
        }
        if (Same(p, i0, i1)) throw new ArgumentException("Initial simplex is flat: all points coincide (Qhull QH6154).");
        int i2 = -1; double best = 0;
        for (var i = 0; i < n; i++)
        {
            var d = CrossNorm2(p, i0, i1, i);
            if (d > best) { best = d; i2 = i; }
        }
        if (i2 < 0) throw new ArgumentException("Initial simplex is flat: all points are collinear (Qhull QH6154).");
        int i3 = -1; best = 0;
        for (var i = 0; i < n; i++)
        {
            var o = SpatialPredicates.Orient3D(P3(p, i0), P3(p, i1), P3(p, i2), P3(p, i));
            if (o != 0 && Math.Abs(o) > best) { best = Math.Abs(o); i3 = i; }
        }
        if (i3 < 0) throw new ArgumentException("Initial simplex is flat: all points are coplanar (Qhull QH6154).");
        // Orient so that i3 is inside face (i0, i1, i2): Orient3D(face, i3) must be positive.
        if (SpatialPredicates.Orient3D(P3(p, i0), P3(p, i1), P3(p, i2), P3(p, i3)) < 0) (i1, i2) = (i2, i1);

        var faces = new List<Face>();
        Face Make(int a, int b, int c) { var f = new Face(); f.V[0] = a; f.V[1] = b; f.V[2] = c; Plane(p, f); faces.Add(f); return f; }
        // Faces of the tetrahedron, each outward.
        var f0 = Make(i0, i1, i2);
        var f1 = Make(i0, i3, i1);
        var f2 = Make(i1, i3, i2);
        var f3 = Make(i2, i3, i0);
        LinkAll(faces);

        var initial = new HashSet<int> { i0, i1, i2, i3 };
        for (var q = 0; q < n; q++)
        {
            if (initial.Contains(q)) continue;
            foreach (var f in faces)
                if (Visible(p, f, q)) { f.Conflict.Add(q); break; }
        }

        var mark = 0;
        var stack = new Stack<int>();
        var visible = new List<int>();
        var startOf = new Dictionary<int, int>();
        var endOf = new Dictionary<int, int>();
        var pending = new Queue<int>();
        for (var i = 0; i < faces.Count; i++) pending.Enqueue(i);
        while (pending.Count > 0)
        {
            var fi = pending.Dequeue();
            var face = faces[fi];
            if (!face.Alive || face.Conflict.Count == 0) continue;
            // Eye point: the furthest conflict point (floating height only ranks candidates).
            var eye = face.Conflict[0]; var h = Height(p, face, eye);
            foreach (var q in face.Conflict) { var hq = Height(p, face, q); if (hq > h) { h = hq; eye = q; } }

            mark++;
            visible.Clear();
            stack.Push(fi); face.Mark = mark;
            while (stack.Count > 0)
            {
                var g = stack.Pop();
                visible.Add(g);
                foreach (var nb in faces[g].N)
                {
                    var fn = faces[nb];
                    if (fn.Mark == mark || !fn.Alive) continue;
                    if (Visible(p, fn, eye)) { fn.Mark = mark; stack.Push(nb); }
                }
            }
            var orphans = new List<int>();
            startOf.Clear(); endOf.Clear();
            var created = new List<int>();
            foreach (var g in visible)
            {
                var vf = faces[g];
                for (var k = 0; k < 3; k++)
                {
                    var nb = vf.N[k];
                    if (faces[nb].Mark == mark) continue;
                    int u = vf.V[(k + 1) % 3], w = vf.V[(k + 2) % 3];
                    var nf = Make(u, w, eye);
                    var id = faces.Count - 1;
                    nf.N[2] = nb;
                    var outside = faces[nb];
                    for (var j = 0; j < 3; j++) if (outside.N[j] == g) outside.N[j] = id;
                    startOf[u] = id; endOf[w] = id;
                    created.Add(id);
                }
            }
            foreach (var id in created)
            {
                var nf = faces[id];
                nf.N[0] = startOf[nf.V[1]];
                nf.N[1] = endOf[nf.V[0]];
            }
            foreach (var g in visible)
            {
                faces[g].Alive = false;
                foreach (var q in faces[g].Conflict) if (q != eye) orphans.Add(q);
                faces[g].Conflict = [];
            }
            foreach (var q in orphans)
                foreach (var id in created)
                    if (Visible(p, faces[id], q)) { faces[id].Conflict.Add(q); break; }
            foreach (var id in created) pending.Enqueue(id);
        }

        var alive = new List<Face>();
        var remap = new int[faces.Count];
        for (var i = 0; i < faces.Count; i++)
            if (faces[i].Alive) { remap[i] = alive.Count; alive.Add(faces[i]); }
        foreach (var f in alive)
            for (var k = 0; k < 3; k++) f.N[k] = remap[f.N[k]];
        return alive;
    }

    private static void LinkAll(List<Face> faces)
    {
        var edge = new Dictionary<(int, int), int>();
        for (var i = 0; i < faces.Count; i++)
            for (var k = 0; k < 3; k++) edge[(faces[i].V[(k + 1) % 3], faces[i].V[(k + 2) % 3])] = i;
        for (var i = 0; i < faces.Count; i++)
            for (var k = 0; k < 3; k++) faces[i].N[k] = edge[(faces[i].V[(k + 2) % 3], faces[i].V[(k + 1) % 3])];
    }

    private static bool Less(double[] p, int a, int b)
    {
        for (var k = 0; k < 3; k++)
        {
            if (p[3 * a + k] < p[3 * b + k]) return true;
            if (p[3 * a + k] > p[3 * b + k]) return false;
        }
        return false;
    }

    private static bool Same(double[] p, int a, int b)
        => p[3 * a] == p[3 * b] && p[3 * a + 1] == p[3 * b + 1] && p[3 * a + 2] == p[3 * b + 2];

    private static double CrossNorm2(double[] p, int a, int b, int c)
    {
        double ux = p[3 * b] - p[3 * a], uy = p[3 * b + 1] - p[3 * a + 1], uz = p[3 * b + 2] - p[3 * a + 2];
        double vx = p[3 * c] - p[3 * a], vy = p[3 * c + 1] - p[3 * a + 1], vz = p[3 * c + 2] - p[3 * a + 2];
        double x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
        return x * x + y * y + z * z;
    }
}
