using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;

namespace SplatStudio.Web;

/// <summary>
/// Draws the synthetic images the demo gallery is seeded with. Generating them in code
/// rather than shipping JPEGs keeps the repository free of binary blobs and lets the seed
/// scale to as many scenes as it wants; every recipe is a pure function of its index, so a
/// given deployment always produces the same gallery.
///
/// The recipes are deliberately biased toward what the depth heuristic can actually do
/// well: one bright, roughly centred subject against a darker falloff. That is also why
/// they read as a fair demo rather than a flattering one.
/// </summary>
public static class SampleImageFactory
{
    public record Recipe(string Key, string Title, string Description);

    /// <summary>The gallery's seed catalogue, in display order.</summary>
    public static readonly Recipe[] Catalogue =
    {
        new("orb", "Drifting orb",
            "A single soft sphere against a dark field — the clear, centred subject the depth heuristic handles best."),
        new("studio", "Studio sphere",
            "Product-photo lighting on a plain backdrop, with a contact shadow to give the solver something to anchor on."),
        new("colour-study", "Colour study",
            "Three overlapping colour blobs. More abstract, and a harder test: no single subject for the radial prior to latch onto."),
        new("nebula", "Nebula drift",
            "Layered noise clouds lit from within. Depth comes almost entirely from luminance here."),
        new("crystal", "Faceted crystal",
            "Hard-edged facets with sharp luminance steps — the depth field ends up terraced rather than smooth."),
        new("torus", "Ring toss",
            "A torus with a hole in the middle, which the centre-weighted prior gets confidently wrong. Kept in as an honest failure case."),
        new("ripples", "Interference",
            "Concentric wave interference. The result is a rippled relief rather than a solid object."),
        new("bokeh", "Night bokeh",
            "Scattered defocused highlights. Each blob floats at its own depth, which reads surprisingly well in orbit."),
        new("dunes", "Dune field",
            "A soft horizontal gradient with layered ridges — nearly flat, and a good demonstration of the technique's limits."),
        new("portrait-bust", "Silhouette bust",
            "A portrait-shaped silhouette with rim lighting, the closest recipe to a real photograph subject."),
        new("prism", "Prism split",
            "A bright wedge dispersing into colour bands across a dark ground."),
        new("moss", "Moss macro",
            "Dense organic texture with no dominant subject — the point cloud comes out as a rolling surface.")
    };

    public static byte[] Render(string key, int size = 768)
    {
        using var image = new Image<Rgba32>(size, size);
        var rng = new Random(StableSeed(key));

        // Precompute per-recipe random features so the inner loop stays branch-light.
        var blobs = CreateBlobs(key, rng, size);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                float v = y / (float)(size - 1);

                for (int x = 0; x < size; x++)
                {
                    float u = x / (float)(size - 1);
                    row[x] = Shade(key, u, v, blobs);
                }
            }
        });

        using var ms = new MemoryStream();
        image.Save(ms, new JpegEncoder { Quality = 90 });
        return ms.ToArray();
    }

    private readonly record struct Blob(float X, float Y, float Radius, float R, float G, float B);

    private static Blob[] CreateBlobs(string key, Random rng, int size) => key switch
    {
        "colour-study" =>
        [
            new(0.38f, 0.42f, 0.26f, 0.92f, 0.34f, 0.45f),
            new(0.58f, 0.50f, 0.24f, 0.30f, 0.78f, 0.86f),
            new(0.48f, 0.64f, 0.22f, 0.62f, 0.45f, 0.95f)
        ],
        "bokeh" => Enumerable.Range(0, 14).Select(_ => new Blob(
            (float)rng.NextDouble(),
            (float)rng.NextDouble(),
            0.04f + (float)rng.NextDouble() * 0.10f,
            0.55f + (float)rng.NextDouble() * 0.45f,
            0.45f + (float)rng.NextDouble() * 0.45f,
            0.60f + (float)rng.NextDouble() * 0.40f)).ToArray(),
        "nebula" => Enumerable.Range(0, 6).Select(_ => new Blob(
            0.25f + (float)rng.NextDouble() * 0.5f,
            0.25f + (float)rng.NextDouble() * 0.5f,
            0.18f + (float)rng.NextDouble() * 0.22f,
            0.40f + (float)rng.NextDouble() * 0.55f,
            0.25f + (float)rng.NextDouble() * 0.45f,
            0.65f + (float)rng.NextDouble() * 0.35f)).ToArray(),
        _ => []
    };

    private static Rgba32 Shade(string key, float u, float v, Blob[] blobs)
    {
        // Centre-relative coordinates, +/-1 across the frame.
        float cx = (u - 0.5f) * 2f;
        float cy = (v - 0.5f) * 2f;
        float radius = MathF.Sqrt(cx * cx + cy * cy);

        switch (key)
        {
            case "orb":
            {
                float d = Dist(u, v, 0.47f, 0.44f);
                float body = Falloff(d, 0.30f);
                float glow = Falloff(d, 0.62f) * 0.35f;
                return Rgb(
                    0.06f + body * 0.94f + glow * 0.35f,
                    0.05f + body * 0.74f + glow * 0.30f,
                    0.12f + body * 0.52f + glow * 0.55f);
            }

            case "studio":
            {
                float d = Dist(u, v, 0.50f, 0.44f);
                float body = Falloff(d, 0.28f);
                // Key light from upper-left.
                float key1 = Falloff(Dist(u, v, 0.42f, 0.36f), 0.30f);
                // Elliptical contact shadow beneath the sphere.
                float shadow = Falloff(EllipseDist(u, v, 0.50f, 0.76f, 1.0f, 3.2f), 0.20f) * 0.45f;
                float bg = 0.72f - v * 0.18f - shadow;
                float lum = body > 0.01f ? 0.35f + key1 * 0.75f : bg;
                return Rgb(lum * 0.98f, lum * 0.96f, lum * 0.93f);
            }

            case "colour-study":
            case "bokeh":
            case "nebula":
            {
                float r = key == "bokeh" ? 0.02f : 0.05f;
                float g = key == "bokeh" ? 0.02f : 0.04f;
                float b = key == "bokeh" ? 0.06f : 0.10f;

                foreach (var blob in blobs)
                {
                    float d = Dist(u, v, blob.X, blob.Y);
                    // Bokeh discs have a hard edge and a bright rim; the others are soft.
                    float w = key == "bokeh"
                        ? (d < blob.Radius ? 0.55f + 0.45f * Smooth(d / blob.Radius) : 0f)
                        : Falloff(d, blob.Radius);
                    r += blob.R * w;
                    g += blob.G * w;
                    b += blob.B * w;
                }
                return Rgb(r, g, b);
            }

            case "crystal":
            {
                // Quantising the angle into wedges gives hard facet boundaries.
                float angle = MathF.Atan2(cy, cx);
                int facet = (int)MathF.Floor((angle + MathF.PI) / (MathF.PI * 2f) * 7f);
                float facetShade = 0.35f + (facet % 3) * 0.22f;
                float body = radius < 0.62f ? 1f : 0f;
                float edge = Falloff(MathF.Abs(radius - 0.62f), 0.05f) * 0.6f;
                float lum = body * facetShade + edge;
                return Rgb(lum * 0.72f + 0.03f, lum * 0.88f + 0.04f, lum * 1.0f + 0.09f);
            }

            case "torus":
            {
                float ring = MathF.Abs(radius - 0.48f);
                float body = Falloff(ring, 0.17f);
                float highlight = Falloff(Dist(u, v, 0.36f, 0.34f), 0.22f) * 0.5f;
                return Rgb(
                    0.04f + body * 0.95f + highlight * 0.4f,
                    0.05f + body * 0.62f + highlight * 0.5f,
                    0.09f + body * 0.30f + highlight * 0.6f);
            }

            case "ripples":
            {
                float w1 = MathF.Sin(Dist(u, v, 0.36f, 0.40f) * 46f);
                float w2 = MathF.Sin(Dist(u, v, 0.66f, 0.58f) * 46f);
                float interference = (w1 + w2) * 0.25f + 0.5f;
                float vignette = 1f - Smooth(Math.Clamp(radius / 1.25f, 0f, 1f));
                float lum = interference * vignette;
                return Rgb(lum * 0.42f + 0.02f, lum * 0.82f + 0.03f, lum * 0.95f + 0.07f);
            }

            case "dunes":
            {
                float ridge = MathF.Sin(v * 13f + MathF.Sin(u * 4f) * 1.4f) * 0.5f + 0.5f;
                float horizon = 1f - v * 0.55f;
                float lum = 0.25f + ridge * 0.35f * horizon + horizon * 0.3f;
                return Rgb(lum * 1.0f, lum * 0.78f, lum * 0.55f);
            }

            case "portrait-bust":
            {
                // Head over shoulders, as two merged ellipses.
                float head = EllipseDist(u, v, 0.50f, 0.38f, 1.35f, 1.0f);
                float shoulders = EllipseDist(u, v, 0.50f, 0.95f, 0.75f, 1.6f);
                bool inside = head < 0.20f || shoulders < 0.34f;
                // Rim light from the right edge of the silhouette.
                float rim = Falloff(MathF.Abs(head - 0.20f), 0.035f) * (u > 0.5f ? 1f : 0.25f);
                if (inside)
                {
                    float fill = 0.16f + Falloff(Dist(u, v, 0.42f, 0.34f), 0.34f) * 0.30f;
                    return Rgb(fill * 1.0f + rim * 0.9f, fill * 0.82f + rim * 0.85f, fill * 0.74f + rim * 1.0f);
                }
                float bg = 0.05f + (1f - v) * 0.06f;
                return Rgb(bg + rim * 0.9f, bg + rim * 0.85f, bg * 1.6f + rim);
            }

            case "prism":
            {
                // A wedge that fans out into spectral bands toward the lower right.
                float band = Math.Clamp((u - 0.42f) * 2.1f, 0f, 1f);
                float spread = Falloff(MathF.Abs(cy - (u - 0.5f) * 0.7f), 0.34f);
                float hue = band * 5.0f;
                float wedge = Falloff(Dist(u, v, 0.30f, 0.48f), 0.16f);
                var (r, g, b) = SpectrumAt(hue);
                float lum = spread * band;
                return Rgb(
                    0.03f + wedge * 0.85f + r * lum,
                    0.03f + wedge * 0.88f + g * lum,
                    0.07f + wedge * 0.95f + b * lum);
            }

            case "moss":
            {
                // Layered sine noise, no dominant subject.
                float n = MathF.Sin(u * 37f) * MathF.Sin(v * 31f)
                        + MathF.Sin(u * 71f + 1.3f) * MathF.Sin(v * 67f + 0.7f) * 0.5f
                        + MathF.Sin(u * 131f) * MathF.Sin(v * 127f) * 0.25f;
                float lum = 0.34f + n * 0.20f;
                return Rgb(lum * 0.36f, lum * 0.92f, lum * 0.42f);
            }

            default:
            {
                float body = Falloff(radius, 0.55f);
                return Rgb(body, body * 0.8f, body * 0.6f);
            }
        }
    }

    // ---- Small shading helpers -------------------------------------------------

    private static float Dist(float u, float v, float cx, float cy)
    {
        float dx = u - cx, dy = v - cy;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private static float EllipseDist(float u, float v, float cx, float cy, float sx, float sy)
    {
        float dx = (u - cx) * sx, dy = (v - cy) * sy;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>Smooth 1-at-centre, 0-at-radius falloff with a smoothstep profile.</summary>
    private static float Falloff(float distance, float radius)
    {
        if (distance >= radius) return 0f;
        return Smooth(1f - distance / radius);
    }

    private static float Smooth(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        return t * t * (3f - 2f * t);
    }

    private static (float R, float G, float B) SpectrumAt(float t)
    {
        // Cheap rainbow ramp; phase-shifted cosines rather than a real HSV conversion.
        float r = 0.5f + 0.5f * MathF.Cos(t + 0.0f);
        float g = 0.5f + 0.5f * MathF.Cos(t + 2.1f);
        float b = 0.5f + 0.5f * MathF.Cos(t + 4.2f);
        return (r, g, b);
    }

    private static Rgba32 Rgb(float r, float g, float b) => new(
        (byte)Math.Clamp(r * 255f, 0f, 255f),
        (byte)Math.Clamp(g * 255f, 0f, 255f),
        (byte)Math.Clamp(b * 255f, 0f, 255f),
        (byte)255);

    /// <summary>Deterministic across runs and processes, unlike string.GetHashCode().</summary>
    private static int StableSeed(string key)
    {
        unchecked
        {
            int hash = 17;
            foreach (var c in key) hash = hash * 31 + c;
            return hash;
        }
    }

    /// <summary>A small circular avatar built from the same palette language as the gallery.</summary>
    public static byte[] RenderAvatar(string seedKey, int size = 256)
    {
        using var image = new Image<Rgba32>(size, size);
        var rng = new Random(StableSeed(seedKey));

        float hueA = (float)rng.NextDouble() * 6.28f;
        float hueB = hueA + 1.6f + (float)rng.NextDouble();
        var (ar, ag, ab) = SpectrumAt(hueA);
        var (br, bg, bb) = SpectrumAt(hueB);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                float v = y / (float)(size - 1);
                for (int x = 0; x < size; x++)
                {
                    float u = x / (float)(size - 1);
                    float t = Smooth((u + v) * 0.5f);
                    float blobA = Falloff(Dist(u, v, 0.34f, 0.36f), 0.42f);
                    float blobB = Falloff(Dist(u, v, 0.68f, 0.66f), 0.38f);

                    row[x] = Rgb(
                        0.06f + ar * blobA + br * blobB + t * 0.05f,
                        0.07f + ag * blobA + bg * blobB + t * 0.05f,
                        0.12f + ab * blobA + bb * blobB + t * 0.08f);
                }
            }
        });

        using var ms = new MemoryStream();
        image.Save(ms, new JpegEncoder { Quality = 88 });
        return ms.ToArray();
    }
}
