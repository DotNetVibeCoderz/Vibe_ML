using System.Numerics;
using System.Text;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

namespace SplatStudio.Web;

/// <summary>
/// Builds the sample meshes the demo gallery is seeded with, as binary glTF written from
/// scratch.
///
/// Generated in code for the same reason <see cref="SampleImageFactory"/> is: the repository
/// stays free of binary blobs, there is no third-party model licence to track, and every
/// deployment produces the same gallery. Each recipe is a pure function of its key.
///
/// These are demo geometry, not the output of any image-to-3D model. The seed labels them
/// as such — see <see cref="SplatStudio.Domain.Enums.SplatEngineType.SampleData"/> — because
/// presenting hand-written parametric surfaces as generative results would misrepresent
/// what mode 3 actually does.
/// </summary>
public static class SampleMeshFactory
{
    public record Recipe(string Key, string Title, string Description);

    public static readonly Recipe[] Catalogue =
    {
        new("torus-knot", "Trefoil knot",
            "A (2,3) torus knot swept as a tube. Nothing about it is flat, which makes it a good test of whether the viewer's orbit and depth really work."),
        new("vase", "Turned vase",
            "A profile curve revolved around its axis, the way a lathe would cut it. Smooth shading over a silhouette that reads instantly as an object."),
        new("gem", "Cut gem",
            "A subdivided icosahedron with per-face normals, so every facet catches the light separately instead of blending into its neighbours."),
        new("shell", "Nautilus",
            "A logarithmic spiral swept with a growing tube radius. The self-occlusion near the centre is the interesting part."),
        new("terrain", "Ridge field",
            "A height field coloured by elevation. Closest of these to what a photogrammetry scan of landscape would give you."),
        new("mobius", "Möbius band",
            "A single-sided surface with a half twist, rendered double-sided. Useful for checking that back faces are not being culled away.")
    };

    // ---- Depth-ramp palette, shared with the stylesheet ------------------------
    private static readonly Vector3 Near = new(1.00f, 0.702f, 0.361f);   // #FFB35C
    private static readonly Vector3 Mid = new(0.894f, 0.337f, 0.431f);   // #E4566E
    private static readonly Vector3 Far = new(0.333f, 0.400f, 0.847f);   // #5566D8

    /// <summary>Blends the three ramp stops. t runs 0 (near) to 1 (far).</summary>
    private static Vector3 Ramp(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return t < 0.5f
            ? Vector3.Lerp(Near, Mid, t * 2f)
            : Vector3.Lerp(Mid, Far, (t - 0.5f) * 2f);
    }

    public sealed class MeshData
    {
        public List<Vector3> Positions { get; } = new();
        public List<Vector3> Normals { get; } = new();
        public List<Vector4> Colors { get; } = new();
        public List<uint> Indices { get; } = new();
    }

    public static byte[] Render(string key) => WriteGlb(Build(key));

    public static MeshData Build(string key) => key switch
    {
        "torus-knot" => TorusKnot(),
        "vase" => Vase(),
        "gem" => Gem(),
        "shell" => Shell(),
        "terrain" => Terrain(),
        "mobius" => Mobius(),
        _ => Vase()
    };

    // ---- Recipes ---------------------------------------------------------------

    private static MeshData TorusKnot(int curveSegments = 220, int tubeSegments = 28)
    {
        const float tubeRadius = 0.28f;

        // Frenet frame along the knot curve, with the tube swept around it.
        Vector3 Curve(float t)
        {
            float p = 2f, q = 3f;
            float r = MathF.Cos(q * t) + 2f;
            return new Vector3(r * MathF.Cos(p * t), -MathF.Sin(q * t), r * MathF.Sin(p * t));
        }

        return Sweep(curveSegments, tubeSegments, closed: true, Curve, (_, _) => tubeRadius,
            (_, v, along) => Ramp(0.5f + 0.5f * MathF.Sin(along * MathF.Tau * 2f + v)));
    }

    private static MeshData Shell(int curveSegments = 260, int tubeSegments = 26)
    {
        // A logarithmic spiral that also climbs, with the tube widening as it goes.
        // t arrives in radians over [0, tau] like every other curve here, so `progress`
        // is what runs 0..1 along the shell.
        Vector3 Curve(float t)
        {
            const float turns = 3.6f;
            float progress = t / MathF.Tau;
            float angle = t * turns;
            float r = 0.09f * MathF.Exp(2.5f * progress);
            // A shallow climb keeps the whorls readable as a spiral from the side.
            return new Vector3(r * MathF.Cos(angle), progress * 0.62f - 0.31f, r * MathF.Sin(angle));
        }

        // Open, not closed: joining the wide end back to the tip would draw a band straight
        // through the middle of the shell.
        //
        // The tube stays thin relative to the spiral's radius on purpose — grow it faster and
        // successive whorls merge into what reads as a fat torus instead of a shell.
        return Sweep(curveSegments, tubeSegments, closed: false, Curve,
            (along, _) => 0.018f + 0.135f * MathF.Pow(along, 2f),
            (_, v, along) => Ramp(0.15f + 0.75f * along + 0.08f * MathF.Sin(v)));
    }

    private static MeshData Vase(int rings = 96, int segments = 64)
    {
        // Profile radius as a function of height: a foot, a belly, a neck, a flared lip.
        float Profile(float t)
        {
            float belly = 0.55f * MathF.Sin(MathF.PI * MathF.Pow(t, 0.85f));
            float neck = 0.16f * MathF.Exp(-MathF.Pow((t - 0.86f) / 0.10f, 2f));
            float lip = 0.13f * MathF.Exp(-MathF.Pow((t - 0.99f) / 0.035f, 2f));
            float foot = 0.10f * MathF.Exp(-MathF.Pow(t / 0.06f, 2f));
            return 0.13f + belly + neck + lip + foot;
        }

        return Grid(rings, segments, wrapU: false, wrapV: true,
            (u, v) =>
            {
                float angle = v * MathF.Tau;
                float r = Profile(u);
                return new Vector3(r * MathF.Cos(angle), u * 1.7f - 0.85f, r * MathF.Sin(angle));
            },
            (_, u, _) => Ramp(1f - u));
    }

    private static MeshData Terrain(int nx = 110, int nz = 110)
    {
        float Height(float x, float z)
        {
            // Layered ridges: abs() of a sine gives creases rather than rolling hills.
            float h = 0.34f * (1f - MathF.Abs(MathF.Sin(x * 2.6f + MathF.Sin(z * 1.3f) * 0.8f)));
            h += 0.16f * (1f - MathF.Abs(MathF.Sin(z * 3.7f + 0.6f)));
            h += 0.07f * MathF.Sin(x * 9.1f) * MathF.Cos(z * 8.3f);
            return h;
        }

        return Grid(nx, nz, wrapU: false, wrapV: false,
            (u, v) =>
            {
                float x = u * 2f - 1f, z = v * 2f - 1f;
                return new Vector3(x, Height(x * 2f, z * 2f) - 0.2f, z);
            },
            (p, _, _) => Ramp(1f - Math.Clamp((p.Y + 0.25f) / 0.62f, 0f, 1f)));
    }

    private static MeshData Mobius(int nu = 220, int nv = 18)
    {
        return Grid(nu, nv, wrapU: true, wrapV: false,
            (u, v) =>
            {
                float a = u * MathF.Tau;
                float w = (v - 0.5f) * 0.62f;
                float half = a / 2f;
                float r = 1f + w * MathF.Cos(half);
                return new Vector3(r * MathF.Cos(a), w * MathF.Sin(half), r * MathF.Sin(a));
            },
            (_, u, _) => Ramp(0.5f + 0.5f * MathF.Sin(u * MathF.Tau)));
    }

    /// <summary>
    /// Subdivided icosahedron projected onto its sphere, then split so every triangle owns
    /// its vertices — that is what gives the flat, faceted shading.
    /// </summary>
    /// <param name="subdivisions">
    /// Two levels gives 320 facets — enough to read as a cut stone. Go higher and the faces
    /// get small enough that flat shading blurs back into a sphere.
    /// </param>
    private static MeshData Gem(int subdivisions = 2)
    {
        float t = (1f + MathF.Sqrt(5f)) / 2f;
        var verts = new List<Vector3>
        {
            new(-1, t, 0), new(1, t, 0), new(-1, -t, 0), new(1, -t, 0),
            new(0, -1, t), new(0, 1, t), new(0, -1, -t), new(0, 1, -t),
            new(t, 0, -1), new(t, 0, 1), new(-t, 0, -1), new(-t, 0, 1)
        };
        for (int i = 0; i < verts.Count; i++) verts[i] = Vector3.Normalize(verts[i]);

        var faces = new List<(int A, int B, int C)>
        {
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)
        };

        for (int s = 0; s < subdivisions; s++)
        {
            var next = new List<(int, int, int)>(faces.Count * 4);
            var midpoints = new Dictionary<(int, int), int>();

            int Midpoint(int a, int b)
            {
                var key = a < b ? (a, b) : (b, a);
                if (midpoints.TryGetValue(key, out var existing)) return existing;
                verts.Add(Vector3.Normalize((verts[a] + verts[b]) * 0.5f));
                midpoints[key] = verts.Count - 1;
                return verts.Count - 1;
            }

            foreach (var (a, b, c) in faces)
            {
                int ab = Midpoint(a, b), bc = Midpoint(b, c), ca = Midpoint(c, a);
                next.Add((a, ab, ca));
                next.Add((b, bc, ab));
                next.Add((c, ca, bc));
                next.Add((ab, bc, ca));
            }
            faces = next;
        }

        var mesh = new MeshData();
        foreach (var (a, b, c) in faces)
        {
            // Push each facet out to a different radius so the silhouette is a cut stone
            // rather than a smooth ball. The displacement is quantised into a few steps so
            // neighbouring facets share a plane and read as one cut face, and it varies
            // slowly across the sphere rather than per-triangle.
            var centre = Vector3.Normalize(verts[a] + verts[b] + verts[c]);
            // Kept small on purpose: displacing adjacent facets to different radii pulls their
            // shared edges apart, and a large step opens visible cracks through to the inside.
            float band = MathF.Sin(centre.Y * 4.2f) * 0.5f + MathF.Cos(centre.X * 3.1f + centre.Z * 2.4f) * 0.5f;
            float facet = 0.94f + 0.06f * MathF.Round((band * 0.5f + 0.5f) * 4f) / 4f;

            Vector3 p0 = verts[a] * facet, p1 = verts[b] * facet, p2 = verts[c] * facet;
            var normal = Vector3.Normalize(Vector3.Cross(p1 - p0, p2 - p0));
            var colour = Ramp(0.5f - 0.5f * centre.Y);

            uint baseIndex = (uint)mesh.Positions.Count;
            foreach (var p in new[] { p0, p1, p2 })
            {
                mesh.Positions.Add(p);
                mesh.Normals.Add(normal);
                mesh.Colors.Add(new Vector4(colour, 1f));
            }
            mesh.Indices.Add(baseIndex);
            mesh.Indices.Add(baseIndex + 1);
            mesh.Indices.Add(baseIndex + 2);
        }
        return mesh;
    }

    // ---- Surface builders --------------------------------------------------------

    /// <summary>
    /// Tessellates a parametric surface over a (u,v) grid. Normals are accumulated from the
    /// faces that touch each vertex, which handles the seams of wrapped surfaces correctly.
    /// </summary>
    private static MeshData Grid(
        int nu, int nv, bool wrapU, bool wrapV,
        Func<float, float, Vector3> surface,
        Func<Vector3, float, float, Vector3> colour)
    {
        var mesh = new MeshData();
        int countU = wrapU ? nu : nu + 1;
        int countV = wrapV ? nv : nv + 1;

        for (int i = 0; i < countU; i++)
        {
            float u = i / (float)nu;
            for (int j = 0; j < countV; j++)
            {
                float v = j / (float)nv;
                var p = surface(u, v);
                mesh.Positions.Add(p);
                mesh.Normals.Add(Vector3.Zero);
                mesh.Colors.Add(new Vector4(colour(p, u, v), 1f));
            }
        }

        int Index(int i, int j) => (i % countU) * countV + (j % countV);
        int lastU = wrapU ? countU : countU - 1;
        int lastV = wrapV ? countV : countV - 1;

        for (int i = 0; i < lastU; i++)
        {
            for (int j = 0; j < lastV; j++)
            {
                int a = Index(i, j), b = Index(i + 1, j), c = Index(i + 1, j + 1), d = Index(i, j + 1);
                AddTriangle(mesh, a, b, c);
                AddTriangle(mesh, a, c, d);
            }
        }

        NormaliseNormals(mesh);
        return mesh;
    }

    /// <summary>
    /// Sweeps a circular tube along a space curve using a parallel-transport frame. The curve
    /// receives its parameter in radians over [0, tau].
    /// </summary>
    /// <param name="closed">
    /// True joins the last ring back to the first, which is right for a knot and wrong for a
    /// spiral — an open curve would get a band running from its end back to its start.
    /// </param>
    private static MeshData Sweep(
        int curveSegments, int tubeSegments, bool closed,
        Func<float, Vector3> curve,
        Func<float, float, float> radius,
        Func<Vector3, float, float, Vector3> colour)
    {
        // A Frenet frame flips at inflection points; carrying the previous normal forward
        // instead keeps the tube from twisting where the curvature changes sign.
        var points = new Vector3[curveSegments];
        var tangents = new Vector3[curveSegments];
        for (int i = 0; i < curveSegments; i++)
        {
            // A closed curve must not repeat its first point as its last.
            float t = closed
                ? i / (float)curveSegments * MathF.Tau
                : i / (float)(curveSegments - 1) * MathF.Tau;
            points[i] = curve(t);
            float dt = 0.0015f;
            tangents[i] = Vector3.Normalize(curve(t + dt) - curve(t - dt));
        }

        var normals = new Vector3[curveSegments];
        var reference = Vector3.UnitY;
        if (MathF.Abs(Vector3.Dot(reference, tangents[0])) > 0.9f) reference = Vector3.UnitX;
        normals[0] = Vector3.Normalize(Vector3.Cross(tangents[0], reference));
        for (int i = 1; i < curveSegments; i++)
        {
            var projected = normals[i - 1] - tangents[i] * Vector3.Dot(normals[i - 1], tangents[i]);
            normals[i] = projected.LengthSquared() < 1e-8f
                ? Vector3.Normalize(Vector3.Cross(tangents[i], reference))
                : Vector3.Normalize(projected);
        }

        var mesh = new MeshData();
        for (int i = 0; i < curveSegments; i++)
        {
            float along = closed ? i / (float)curveSegments : i / (float)(curveSegments - 1);
            var binormal = Vector3.Cross(tangents[i], normals[i]);

            for (int j = 0; j < tubeSegments; j++)
            {
                float v = j / (float)tubeSegments * MathF.Tau;
                var offset = normals[i] * MathF.Cos(v) + binormal * MathF.Sin(v);
                var p = points[i] + offset * radius(along, v);

                mesh.Positions.Add(p);
                mesh.Normals.Add(Vector3.Zero);
                mesh.Colors.Add(new Vector4(colour(p, v, along), 1f));
            }
        }

        int Index(int i, int j) => (i % curveSegments) * tubeSegments + (j % tubeSegments);
        int lastRing = closed ? curveSegments : curveSegments - 1;

        for (int i = 0; i < lastRing; i++)
        {
            for (int j = 0; j < tubeSegments; j++)
            {
                int a = Index(i, j), b = Index(i + 1, j), c = Index(i + 1, j + 1), d = Index(i, j + 1);
                AddTriangle(mesh, a, b, c);
                AddTriangle(mesh, a, c, d);
            }
        }

        NormaliseNormals(mesh);
        return mesh;
    }

    private static void AddTriangle(MeshData mesh, int a, int b, int c)
    {
        mesh.Indices.Add((uint)a);
        mesh.Indices.Add((uint)b);
        mesh.Indices.Add((uint)c);

        // Cross product magnitude is proportional to triangle area, so summing unnormalised
        // face normals weights each contribution by area for free.
        var faceNormal = Vector3.Cross(mesh.Positions[b] - mesh.Positions[a],
                                       mesh.Positions[c] - mesh.Positions[a]);
        mesh.Normals[a] += faceNormal;
        mesh.Normals[b] += faceNormal;
        mesh.Normals[c] += faceNormal;
    }

    private static void NormaliseNormals(MeshData mesh)
    {
        for (int i = 0; i < mesh.Normals.Count; i++)
        {
            mesh.Normals[i] = mesh.Normals[i].LengthSquared() < 1e-12f
                ? Vector3.UnitY
                : Vector3.Normalize(mesh.Normals[i]);
        }
    }

    // ---- Binary glTF writer --------------------------------------------------------

    /// <summary>
    /// Serialises a mesh as .glb: a 12-byte header, a JSON chunk describing the structure,
    /// and a binary chunk holding the vertex data. Both chunks must be padded to a 4-byte
    /// boundary — JSON with spaces, binary with zeroes — or strict loaders reject the file.
    /// </summary>
    public static byte[] WriteGlb(MeshData mesh)
    {
        var positions = ToBytes(mesh.Positions, (w, p) => { w.Write(p.X); w.Write(p.Y); w.Write(p.Z); });
        var normals = ToBytes(mesh.Normals, (w, n) => { w.Write(n.X); w.Write(n.Y); w.Write(n.Z); });
        var colors = ToBytes(mesh.Colors, (w, c) => { w.Write(c.X); w.Write(c.Y); w.Write(c.Z); w.Write(c.W); });
        var indices = ToBytes(mesh.Indices, (w, i) => w.Write(i));

        var bin = new byte[positions.Length + normals.Length + colors.Length + indices.Length];
        int offset = 0;
        foreach (var block in new[] { positions, normals, colors, indices })
        {
            Buffer.BlockCopy(block, 0, bin, offset, block.Length);
            offset += block.Length;
        }

        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var p in mesh.Positions)
        {
            min = Vector3.Min(min, p);
            max = Vector3.Max(max, p);
        }

        var gltf = new
        {
            asset = new { version = "2.0", generator = "SplatStudio SampleMeshFactory" },
            scene = 0,
            scenes = new[] { new { nodes = new[] { 0 } } },
            nodes = new[] { new { mesh = 0 } },
            meshes = new[]
            {
                new
                {
                    primitives = new[]
                    {
                        new
                        {
                            attributes = new Dictionary<string, int>
                            {
                                ["POSITION"] = 0, ["NORMAL"] = 1, ["COLOR_0"] = 2
                            },
                            indices = 3,
                            material = 0
                        }
                    }
                }
            },
            materials = new[]
            {
                new
                {
                    // Vertex colours carry the look; the factor stays white so it multiplies
                    // through unchanged. Double-sided because the Möbius band has no inside.
                    pbrMetallicRoughness = new
                    {
                        baseColorFactor = new[] { 1f, 1f, 1f, 1f },
                        metallicFactor = 0.05f,
                        roughnessFactor = 0.62f
                    },
                    doubleSided = true
                }
            },
            buffers = new[] { new { byteLength = bin.Length } },
            bufferViews = new[]
            {
                new { buffer = 0, byteOffset = 0, byteLength = positions.Length, target = 34962 },
                new { buffer = 0, byteOffset = positions.Length, byteLength = normals.Length, target = 34962 },
                new { buffer = 0, byteOffset = positions.Length + normals.Length, byteLength = colors.Length, target = 34962 },
                new { buffer = 0, byteOffset = positions.Length + normals.Length + colors.Length, byteLength = indices.Length, target = 34963 }
            },
            accessors = new object[]
            {
                new
                {
                    bufferView = 0, componentType = 5126, count = mesh.Positions.Count, type = "VEC3",
                    min = new[] { min.X, min.Y, min.Z }, max = new[] { max.X, max.Y, max.Z }
                },
                new { bufferView = 1, componentType = 5126, count = mesh.Normals.Count, type = "VEC3" },
                new { bufferView = 2, componentType = 5126, count = mesh.Colors.Count, type = "VEC4" },
                new { bufferView = 3, componentType = 5125, count = mesh.Indices.Count, type = "SCALAR" }
            }
        };

        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(gltf));
        var jsonChunk = PadTo4(json, 0x20);   // spaces
        var binChunk = PadTo4(bin, 0x00);     // zeroes

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(Encoding.ASCII.GetBytes("glTF"));
        writer.Write(2u);
        writer.Write((uint)(12 + 8 + jsonChunk.Length + 8 + binChunk.Length));

        writer.Write((uint)jsonChunk.Length);
        writer.Write(0x4E4F534Au); // "JSON"
        writer.Write(jsonChunk);

        writer.Write((uint)binChunk.Length);
        writer.Write(0x004E4942u); // "BIN"
        writer.Write(binChunk);

        writer.Flush();
        return ms.ToArray();
    }

    private static byte[] ToBytes<T>(List<T> items, Action<BinaryWriter, T> write)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        foreach (var item in items) write(writer, item);
        writer.Flush();
        return ms.ToArray();
    }

    private static byte[] PadTo4(byte[] data, byte filler)
    {
        int remainder = data.Length % 4;
        if (remainder == 0) return data;

        var padded = new byte[data.Length + (4 - remainder)];
        Buffer.BlockCopy(data, 0, padded, 0, data.Length);
        for (int i = data.Length; i < padded.Length; i++) padded[i] = filler;
        return padded;
    }

    // ---- Thumbnail ------------------------------------------------------------------

    /// <summary>
    /// Renders the mesh to a JPEG for the gallery card.
    ///
    /// A gallery of mesh scenes needs thumbnails that show the actual model — reusing an
    /// unrelated source photo would misrepresent what the scene contains. That means
    /// rasterising here, so this is a small z-buffered triangle filler: orthographic
    /// projection, Lambert shading plus a rim term, no textures or shadows.
    /// </summary>
    public static byte[] RenderThumbnail(MeshData mesh, int size = 640)
    {
        var background = new Vector3(0.055f, 0.067f, 0.102f); // matches --ground-lift
        var pixels = new Vector3[size * size];
        var depth = new float[size * size];
        Array.Fill(depth, float.MaxValue);
        Array.Fill(pixels, background);

        // Fixed three-quarter view, so every recipe is framed the same way.
        var view = Matrix4x4.CreateRotationY(-0.6f) * Matrix4x4.CreateRotationX(0.42f);

        var transformed = new Vector3[mesh.Positions.Count];
        var viewNormals = new Vector3[mesh.Normals.Count];
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);

        for (int i = 0; i < mesh.Positions.Count; i++)
        {
            transformed[i] = Vector3.Transform(mesh.Positions[i], view);
            viewNormals[i] = Vector3.TransformNormal(mesh.Normals[i], view);
            min = Vector3.Min(min, transformed[i]);
            max = Vector3.Max(max, transformed[i]);
        }

        var centre = (min + max) * 0.5f;
        var extent = max - min;
        float longest = MathF.Max(extent.X, extent.Y);
        if (longest <= 0f) longest = 1f;
        float scale = size * 0.80f / longest;

        Vector2 Project(Vector3 p) => new(
            (p.X - centre.X) * scale + size / 2f,
            // Screen Y grows downward; world Y grows up.
            -(p.Y - centre.Y) * scale + size / 2f);

        var light = Vector3.Normalize(new Vector3(-0.45f, 0.75f, 0.85f));

        for (int t = 0; t < mesh.Indices.Count; t += 3)
        {
            int i0 = (int)mesh.Indices[t], i1 = (int)mesh.Indices[t + 1], i2 = (int)mesh.Indices[t + 2];
            Vector2 s0 = Project(transformed[i0]), s1 = Project(transformed[i1]), s2 = Project(transformed[i2]);

            float area = (s1.X - s0.X) * (s2.Y - s0.Y) - (s2.X - s0.X) * (s1.Y - s0.Y);
            if (MathF.Abs(area) < 1e-6f) continue;

            int minX = Math.Max(0, (int)MathF.Floor(MathF.Min(s0.X, MathF.Min(s1.X, s2.X))));
            int maxX = Math.Min(size - 1, (int)MathF.Ceiling(MathF.Max(s0.X, MathF.Max(s1.X, s2.X))));
            int minY = Math.Max(0, (int)MathF.Floor(MathF.Min(s0.Y, MathF.Min(s1.Y, s2.Y))));
            int maxY = Math.Min(size - 1, (int)MathF.Ceiling(MathF.Max(s0.Y, MathF.Max(s1.Y, s2.Y))));

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    float px = x + 0.5f, py = y + 0.5f;
                    float w0 = ((s1.X - px) * (s2.Y - py) - (s2.X - px) * (s1.Y - py)) / area;
                    float w1 = ((s2.X - px) * (s0.Y - py) - (s0.X - px) * (s2.Y - py)) / area;
                    float w2 = 1f - w0 - w1;
                    if (w0 < 0f || w1 < 0f || w2 < 0f) continue;

                    // Larger Z is nearer in this view space, so keep the maximum.
                    float z = w0 * transformed[i0].Z + w1 * transformed[i1].Z + w2 * transformed[i2].Z;
                    int index = y * size + x;
                    if (depth[index] != float.MaxValue && z <= -depth[index]) continue;
                    depth[index] = -z;

                    var normal = Vector3.Normalize(
                        w0 * viewNormals[i0] + w1 * viewNormals[i1] + w2 * viewNormals[i2]);
                    var albedo = new Vector3(
                        w0 * mesh.Colors[i0].X + w1 * mesh.Colors[i1].X + w2 * mesh.Colors[i2].X,
                        w0 * mesh.Colors[i0].Y + w1 * mesh.Colors[i1].Y + w2 * mesh.Colors[i2].Y,
                        w0 * mesh.Colors[i0].Z + w1 * mesh.Colors[i1].Z + w2 * mesh.Colors[i2].Z);

                    // Two-sided: back faces are lit as if their normal pointed at us.
                    float lambert = MathF.Abs(Vector3.Dot(normal, light));
                    float rim = MathF.Pow(1f - MathF.Abs(normal.Z), 3f) * 0.35f;
                    pixels[index] = albedo * (0.22f + 0.78f * lambert) + new Vector3(rim * 0.5f, rim * 0.55f, rim);
                }
            }
        }

        using var image = new Image<Rgba32>(size, size);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    var c = pixels[y * size + x];
                    row[x] = new Rgba32(
                        (byte)Math.Clamp(c.X * 255f, 0f, 255f),
                        (byte)Math.Clamp(c.Y * 255f, 0f, 255f),
                        (byte)Math.Clamp(c.Z * 255f, 0f, 255f),
                        (byte)255);
                }
            }
        });

        using var ms = new MemoryStream();
        image.Save(ms, new JpegEncoder { Quality = 90 });
        return ms.ToArray();
    }
}
