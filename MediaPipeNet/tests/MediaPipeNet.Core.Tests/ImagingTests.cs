using MediaPipeNet.Imaging;
using MediaPipeNet.Tests;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MediaPipeNet.Core.Tests;

public class MPImageTests
{
    private static MPImage Gradient(int w, int h)
    {
        var img = MPImage.Create(w, h);
        var px = img.GetPixelSpan();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                px[y * w + x] = new Rgba32((byte)(x * 255 / Math.Max(1, w - 1)), (byte)(y * 255 / Math.Max(1, h - 1)), 128, 255);
        return img;
    }

    [Theory]
    [InlineData(PixelFormat.Rgba32, 4)]
    [InlineData(PixelFormat.Bgra32, 4)]
    [InlineData(PixelFormat.Rgb24, 3)]
    [InlineData(PixelFormat.Bgr24, 3)]
    [InlineData(PixelFormat.Gray8, 1)]
    public void Pixel_formats_round_trip(PixelFormat format, int bpp)
    {
        using var src = Gradient(7, 5);
        var bytes = new byte[7 * 5 * bpp];
        src.CopyTo(bytes, format);
        using var back = MPImage.FromPixelData(bytes, 7, 5, format);
        if (format == PixelFormat.Gray8)
        {
            back[3, 2].R.Should().Be(back[3, 2].G);
            return;
        }
        back[6, 4].Should().Be(src[6, 4] with { A = back[6, 4].A });
        back[0, 0].R.Should().Be(src[0, 0].R);
    }

    [Fact]
    public void Stride_is_honored()
    {
        var data = new byte[2 * 8]; // 2 rows, 2 pixels of RGB + 2 padding bytes
        data[0] = 10; data[3] = 20; data[8] = 30;
        using var img = MPImage.FromPixelData(data, 2, 2, PixelFormat.Rgb24, stride: 8);
        img[0, 0].R.Should().Be(10);
        img[1, 0].R.Should().Be(20);
        img[0, 1].R.Should().Be(30);
        var act = () => MPImage.FromPixelData(new byte[3], 2, 2, PixelFormat.Rgb24);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Copy_clone_flip_and_reuse()
    {
        using var img = Gradient(4, 3);
        using var clone = img.Clone();
        clone.Pixels.ToArray().Should().Equal(img.Pixels.ToArray());
        using var flipped = img.FlipHorizontal();
        flipped[0, 1].Should().Be(img[3, 1]);

        using var reused = MPImage.Create(1, 1);
        reused.CopyFrom(img);
        reused.Size.Should().Be(new ImageSize(4, 3));
        reused.CopyFrom(new byte[2 * 2 * 4], 2, 2, PixelFormat.Rgba32);
        reused.PixelCount.Should().Be(4);
        reused.AsBytes().Length.Should().Be(16);
        reused.GetRow(1).Length.Should().Be(2);
    }

    [Fact]
    public void Load_from_file_stream_and_bytes()
    {
        var path = TestPaths.Image("burger.jpg");
        using var a = MPImage.Load(path);
        using var s = File.OpenRead(path);
        using var b = MPImage.Load(s);
        using var c = MPImage.Load(File.ReadAllBytes(path));
        a.Size.Should().Be(b.Size).And.Be(c.Size);
        using var sharp = a.ToImage();
        sharp.Width.Should().Be(a.Width);
        using var fromSharp = MPImage.FromImage((Image)sharp);
        fromSharp[5, 5].Should().Be(a[5, 5]);
    }

    [Fact]
    public async Task Load_async_and_save()
    {
        using var img = await MPImage.LoadAsync(TestPaths.Image("burger.jpg"));
        var png = Path.Combine(Path.GetTempPath(), $"mpn-{Guid.NewGuid():N}.png");
        var jpg = Path.ChangeExtension(png, ".jpg");
        img.SaveAsPng(png);
        img.SaveAsJpeg(jpg);
        await using (var fs = File.OpenRead(png))
        {
            using var back = await MPImage.LoadAsync(fs);
            back[10, 10].Should().Be(img[10, 10]);
        }
        File.Delete(png);
        File.Delete(jpg);
    }

    [Fact]
    public void Dispose_returns_buffer()
    {
        var img = MPImage.Create(2, 2);
        img.Dispose();
        img.IsDisposed.Should().BeTrue();
        Action act = () => img.GetPixelSpan();
        act.Should().Throw<ObjectDisposedException>();
        img.Dispose();
    }
}

public class ImageToTensorTests
{
    [Fact]
    public void Full_image_identity_resize_normalizes()
    {
        using var img = MPImage.Create(4, 4);
        img.GetPixelSpan().Fill(new Rgba32(255, 0, 51, 255));
        var t = new float[4 * 4 * 3];
        var map = ImageToTensor.Convert(img, new ImageToTensorOptions(4, 4, -1, 1), t);
        t[0].Should().BeApproximately(1, 1e-5f);
        t[1].Should().BeApproximately(-1, 1e-5f);
        t[2].Should().BeApproximately(-0.6f, 1e-5f);
        map.Padding.Should().Be(LetterboxPadding.None);
    }

    [Fact]
    public void Letterbox_pads_and_mapping_round_trips()
    {
        using var img = MPImage.Create(200, 100);
        img.GetPixelSpan().Fill(new Rgba32(255, 255, 255, 255));
        var t = new float[64 * 64 * 3];
        var map = ImageToTensor.Convert(img, new ImageToTensorOptions(64, 64, 0, 1, KeepAspectRatio: true), t);
        map.Padding.Top.Should().BeApproximately(0.25f, 1e-4f);
        t[0].Should().Be(0); // padded area is black (border zero)
        t[(32 * 64 + 32) * 3].Should().BeApproximately(1, 1e-4f);
        var (x, y) = map.TensorToImage(0.5f, 0.25f);
        x.Should().BeApproximately(0.5f, 1e-4f);
        y.Should().BeApproximately(0f, 1e-4f);
        var (u, v) = map.ImageToTensor(x, y);
        u.Should().BeApproximately(0.5f, 1e-4f);
        v.Should().BeApproximately(0.25f, 1e-4f);
    }

    [Fact]
    public void Rotated_roi_samples_rotated_content()
    {
        // Left half red, right half blue. A 90° clockwise ROI sees red at the top.
        using var img = MPImage.Create(100, 100);
        var px = img.GetPixelSpan();
        for (int y = 0; y < 100; y++)
            for (int x = 0; x < 100; x++)
                px[y * 100 + x] = x < 50 ? new Rgba32(255, 0, 0, 255) : new Rgba32(0, 0, 255, 255);
        var t = new float[10 * 10 * 3];
        var roi = new NormalizedRect(0.5f, 0.5f, 0.8f, 0.8f, MathF.PI / 2);
        var map = ImageToTensor.Convert(img, roi, new ImageToTensorOptions(10, 10, Antialias: false), t);
        t[(0 * 10 + 5) * 3 + 2].Should().BeApproximately(1, 1e-3f); // top center: blue side after rotation
        t[(9 * 10 + 5) * 3 + 0].Should().BeApproximately(1, 1e-3f); // bottom center: red
        map.ScaleZ(1f).Should().BeApproximately(0.8f, 1e-5f);
        var (ix, iy) = map.TensorToImage(0.5f, 0f);
        ix.Should().BeApproximately(0.9f, 1e-4f);
        iy.Should().BeApproximately(0.5f, 1e-4f);
    }

    [Fact]
    public void Replicate_border_and_antialias()
    {
        using var img = MPImage.Create(8, 8);
        img.GetPixelSpan().Fill(new Rgba32(255, 255, 255, 255));
        var t = new float[4 * 4 * 3];
        ImageToTensor.Convert(img, new NormalizedRect(0.5f, 0.5f, 2f, 2f), new ImageToTensorOptions(4, 4, BorderMode: BorderMode.Replicate), t);
        t.Should().OnlyContain(v => Math.Abs(v - 1) < 1e-4f);
        var big = new float[2 * 2 * 3];
        using var large = MPImage.Create(400, 400);
        large.GetPixelSpan().Fill(new Rgba32(10, 20, 30, 255));
        ImageToTensor.Convert(large, new ImageToTensorOptions(2, 2, 0, 255), big);
        big[0].Should().BeApproximately(10, 1e-3f);
        var act = () => ImageToTensor.Convert(img, new ImageToTensorOptions(4, 4), new float[3]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Tensor_warp_projects_mask_back()
    {
        var tensor = new float[4 * 4];
        Array.Fill(tensor, 1f);
        var map = new TensorMapping(new NormalizedRect(0.25f, 0.25f, 0.5f, 0.5f), default, 100, 100, 4, 4);
        var dst = new float[10 * 10];
        TensorWarp.ProjectToImage(tensor, 4, 4, map, dst, 10, 10);
        dst[1 * 10 + 1].Should().BeApproximately(1, 1e-4f);
        dst[8 * 10 + 8].Should().Be(0);
        var resized = new float[8 * 8];
        TensorWarp.Resize(tensor, 4, 4, resized, 8, 8);
        resized.Should().OnlyContain(v => Math.Abs(v - 1) < 1e-5f);
    }
}

public class FrameSourceTests
{
    [Fact]
    public async Task Image_file_source_timestamps_frames()
    {
        await using var src = new ImageFileFrameSource([TestPaths.Image("burger.jpg"), TestPaths.Image("victory.jpg")], frameRate: 10);
        src.IsLive.Should().BeFalse();
        src.Name.Should().Contain("2");
        var frames = new List<(long, long, int)>();
        await foreach (var f in src.ReadFramesAsync()) frames.Add((f.TimestampMs, f.Index, f.Image.Width));
        frames.Select(f => f.Item1).Should().Equal(0, 100);
        frames.Select(f => f.Item2).Should().Equal(0, 1);
    }

    [Fact]
    public async Task Directory_and_memory_sources()
    {
        var dir = ImageFileFrameSource.FromDirectory(Path.GetDirectoryName(TestPaths.Image("burger.jpg"))!);
        int n = 0;
        await foreach (var _ in dir.ReadFramesAsync()) n++;
        n.Should().BeGreaterThanOrEqualTo(7);

        using var a = MPImage.Create(2, 2);
        await using var mem = new MemoryFrameSource([a, a, a], frameRate: 100, realTime: true);
        mem.FrameSize.Should().Be(new ImageSize(2, 2));
        mem.IsLive.Should().BeTrue();
        var ts = new List<long>();
        await foreach (var f in mem.ReadFramesAsync()) ts.Add(f.TimestampMs);
        ts.Should().Equal(0, 10, 20);
    }
}
