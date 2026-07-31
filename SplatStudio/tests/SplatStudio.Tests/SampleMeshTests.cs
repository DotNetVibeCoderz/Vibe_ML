using System.Text;
using System.Text.Json;
using SplatStudio.Web;
using Xunit.Abstractions;

namespace SplatStudio.Tests;

/// <summary>
/// Validates the generated sample geometry against the .glb container rules, because a file
/// that is subtly malformed fails inside GLTFLoader in the browser with nothing useful on the
/// server side.
/// </summary>
public class SampleMeshTests
{
    private readonly ITestOutputHelper _output;

    public SampleMeshTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> Recipes =>
        SampleMeshFactory.Catalogue.Select(r => new object[] { r.Key });

    [Fact]
    public void Catalogue_has_a_varied_set_of_recipes()
    {
        Assert.True(SampleMeshFactory.Catalogue.Length >= 5);
        Assert.Equal(
            SampleMeshFactory.Catalogue.Select(r => r.Key).Distinct().Count(),
            SampleMeshFactory.Catalogue.Length);
        Assert.All(SampleMeshFactory.Catalogue, r =>
        {
            Assert.False(string.IsNullOrWhiteSpace(r.Title));
            Assert.False(string.IsNullOrWhiteSpace(r.Description));
        });
    }

    [Theory]
    [MemberData(nameof(Recipes))]
    public void Mesh_geometry_is_well_formed(string key)
    {
        var mesh = SampleMeshFactory.Build(key);

        Assert.True(mesh.Positions.Count > 100, $"{key} produced only {mesh.Positions.Count} vertices.");
        Assert.Equal(mesh.Positions.Count, mesh.Normals.Count);
        Assert.Equal(mesh.Positions.Count, mesh.Colors.Count);
        Assert.True(mesh.Indices.Count % 3 == 0, "Indices must describe whole triangles.");
        Assert.All(mesh.Indices, i => Assert.InRange(i, 0u, (uint)mesh.Positions.Count - 1));

        // Degenerate normals would render as black patches, and NaNs propagate silently.
        Assert.All(mesh.Normals, n =>
        {
            Assert.False(float.IsNaN(n.X) || float.IsNaN(n.Y) || float.IsNaN(n.Z));
            Assert.InRange(n.Length(), 0.99f, 1.01f);
        });
        Assert.All(mesh.Positions, p =>
            Assert.False(float.IsNaN(p.X) || float.IsNaN(p.Y) || float.IsNaN(p.Z)));

        // A flat result would mean the parametric surface collapsed.
        var min = mesh.Positions.Aggregate(System.Numerics.Vector3.Min);
        var max = mesh.Positions.Aggregate(System.Numerics.Vector3.Max);
        var extent = max - min;
        Assert.True(extent.X > 0.1f && extent.Y > 0.1f && extent.Z > 0.1f,
            $"{key} is degenerate: extent {extent}.");
    }

    [Theory]
    [MemberData(nameof(Recipes))]
    public void Glb_container_is_valid(string key)
    {
        var glb = SampleMeshFactory.Render(key);

        Assert.Equal("glTF", Encoding.ASCII.GetString(glb, 0, 4));
        Assert.Equal(2u, BitConverter.ToUInt32(glb, 4));
        Assert.Equal((uint)glb.Length, BitConverter.ToUInt32(glb, 8));

        // Chunk 0 must be JSON, chunk 1 BIN, and both lengths must be 4-byte aligned.
        var jsonLength = BitConverter.ToUInt32(glb, 12);
        Assert.Equal(0x4E4F534Au, BitConverter.ToUInt32(glb, 16));
        Assert.True(jsonLength % 4 == 0, "JSON chunk must be padded to 4 bytes.");

        int binHeader = 20 + (int)jsonLength;
        var binLength = BitConverter.ToUInt32(glb, binHeader);
        Assert.Equal(0x004E4942u, BitConverter.ToUInt32(glb, binHeader + 4));
        Assert.True(binLength % 4 == 0, "BIN chunk must be padded to 4 bytes.");
        Assert.Equal(glb.Length, binHeader + 8 + (int)binLength);

        _output.WriteLine($"{key}: {glb.Length / 1024.0:0.#} KB");
    }

    [Theory]
    [MemberData(nameof(Recipes))]
    public void Glb_accessors_match_the_binary_chunk(string key)
    {
        var glb = SampleMeshFactory.Render(key);
        var jsonLength = (int)BitConverter.ToUInt32(glb, 12);
        using var doc = JsonDocument.Parse(Encoding.UTF8.GetString(glb, 20, jsonLength).TrimEnd());
        var root = doc.RootElement;

        var declaredBufferLength = root.GetProperty("buffers")[0].GetProperty("byteLength").GetInt32();
        var binLength = (int)BitConverter.ToUInt32(glb, 20 + jsonLength);
        // The BIN chunk is padded, so it can exceed the buffer it carries but never fall short.
        Assert.True(binLength >= declaredBufferLength);

        // Every bufferView must sit inside the buffer.
        foreach (var view in root.GetProperty("bufferViews").EnumerateArray())
        {
            int offset = view.GetProperty("byteOffset").GetInt32();
            int length = view.GetProperty("byteLength").GetInt32();
            Assert.True(offset % 4 == 0, "bufferView offsets must be 4-byte aligned.");
            Assert.True(offset + length <= declaredBufferLength);
        }

        var accessors = root.GetProperty("accessors");
        var mesh = SampleMeshFactory.Build(key);
        Assert.Equal(mesh.Positions.Count, accessors[0].GetProperty("count").GetInt32());
        Assert.Equal(mesh.Indices.Count, accessors[3].GetProperty("count").GetInt32());

        // POSITION requires min/max; loaders use it for bounds and some reject the file without it.
        Assert.True(accessors[0].TryGetProperty("min", out _));
        Assert.True(accessors[0].TryGetProperty("max", out _));

        var attributes = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        Assert.True(attributes.TryGetProperty("POSITION", out _));
        Assert.True(attributes.TryGetProperty("NORMAL", out _));
        Assert.True(attributes.TryGetProperty("COLOR_0", out _));
    }

    [Theory]
    [MemberData(nameof(Recipes))]
    public void Thumbnail_renders_the_mesh_rather_than_an_empty_frame(string key)
    {
        var mesh = SampleMeshFactory.Build(key);
        var jpeg = SampleMeshFactory.RenderThumbnail(mesh, 256);

        Assert.True(jpeg.Length > 1000);
        Assert.Equal(0xFF, jpeg[0]);
        Assert.Equal(0xD8, jpeg[1]); // JPEG SOI

        using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(jpeg);
        Assert.Equal(256, image.Width);

        // Count pixels that differ from the background, so a blank render fails here rather
        // than silently shipping an empty card to the gallery.
        int lit = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    if (row[x].R > 40 || row[x].G > 40 || row[x].B > 45) lit++;
            }
        });

        var coverage = lit / (256.0 * 256.0);
        _output.WriteLine($"{key}: {coverage:P1} of the thumbnail is subject");
        Assert.InRange(coverage, 0.04, 0.95);
    }

    [Fact]
    public void Output_is_deterministic_so_every_deployment_seeds_the_same_gallery()
    {
        foreach (var recipe in SampleMeshFactory.Catalogue)
            Assert.Equal(SampleMeshFactory.Render(recipe.Key), SampleMeshFactory.Render(recipe.Key));
    }
}
