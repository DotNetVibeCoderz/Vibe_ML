// Renders HF.Net's NuGet package icon: assets/icon.svg, assets/icon.png (128 px, what the packages
// carry) and assets/icon-512.png. One drawing routine produces all three, so they cannot drift.
//
//     dotnet run --project tools/IconGen -- assets
//
// The mark is an attention map. Five by five query/key cells, shaded by weight, and the heaviest
// weights spell an H: close up it is what a transformer computes, at 32 px it is a letter. Under it
// runs HFAppGen's offset rail - eight spans, one per library, each as wide as that library's share of
// the source - in the rail's own colours, on HFAppGen's graphite. Amber is identity, as in the app.
//
// Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil.
using SkiaSharp;

var output = args.Length > 0 ? args[0] : "assets";
Directory.CreateDirectory(output);

using (var svg = File.Create(Path.Combine(output, "icon.svg")))
using (var canvas = SKSvgCanvas.Create(new SKRect(0, 0, 512, 512), svg))
{
    Draw(canvas);
}

File.WriteAllBytes(Path.Combine(output, "icon-512.png"), Png(512));
File.WriteAllBytes(Path.Combine(output, "icon.png"), Png(128));
Console.WriteLine($"Icon written to {Path.GetFullPath(output)}");

static byte[] Png(int size)
{
    using var surface = SKSurface.Create(new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Premul));
    surface.Canvas.Clear(SKColors.Transparent);
    surface.Canvas.Scale(size / 512f);
    Draw(surface.Canvas);

    using var image = surface.Snapshot();
    using var data = image.Encode(SKEncodedImageFormat.Png, 100);
    return data.ToArray();
}

static void Draw(SKCanvas c)
{
    var amber = SKColor.Parse("#FFB454");
    var tile = new SKRoundRect(new SKRect(8, 8, 504, 504), 108, 108);

    // Ground: HFAppGen's blue-leaning graphite, lit from the top left.
    using (var ground = new SKPaint { IsAntialias = true })
    {
        ground.Shader = SKShader.CreateLinearGradient(new SKPoint(8, 8), new SKPoint(504, 504),
            [SKColor.Parse("#262E38"), SKColor.Parse("#171C22"), SKColor.Parse("#0E1216")], [0f, 0.5f, 1f], SKShaderTileMode.Clamp);
        c.DrawRoundRect(tile, ground);
    }

    c.Save();
    c.ClipRoundRect(tile, antialias: true);

    // A low amber warmth behind the H, so the mark sits in light rather than on a flat field.
    using (var warmth = new SKPaint { IsAntialias = true })
    {
        warmth.Shader = SKShader.CreateRadialGradient(new SKPoint(256, 236), 250,
            [amber.WithAlpha(46), amber.WithAlpha(0)], SKShaderTileMode.Clamp);
        c.DrawRect(0, 0, 512, 512, warmth);
    }

    // The attention map. Weights are fixed, not random: the H at full strength (varied a little, as
    // real weights are), the diagonal - each token attending to itself - at a third, the rest faint.
    const float Left = 98, Top = 74, Cell = 52, Gap = 13;
    double[,] weight =
    {
        { 1.00, 0.10, 0.16, 0.08, 0.94 },
        { 0.96, 0.34, 0.12, 0.10, 0.90 },
        { 0.92, 0.88, 0.97, 0.86, 1.00 },
        { 0.98, 0.14, 0.09, 0.32, 0.93 },
        { 0.95, 0.07, 0.13, 0.11, 0.99 },
    };

    for (var row = 0; row < 5; row++)
    {
        for (var column = 0; column < 5; column++)
        {
            var x = Left + column * (Cell + Gap);
            var y = Top + row * (Cell + Gap);
            var w = weight[row, column];
            var cell = new SKRoundRect(new SKRect(x, y, x + Cell, y + Cell), 11, 11);

            using var fill = new SKPaint { IsAntialias = true };
            if (w > 0.5)
            {
                // Strong cells: amber, warming toward the rail's coral at the bottom of each cell.
                fill.Shader = SKShader.CreateLinearGradient(new SKPoint(x, y), new SKPoint(x, y + Cell),
                    [SKColor.Parse("#FFC87A").WithAlpha((byte)(255 * w)), amber.WithAlpha((byte)(255 * w))], SKShaderTileMode.Clamp);
            }
            else
            {
                // Weak cells: a cool tint of the text colour, with amber mixed in by weight, so the
                // diagonal warms a little and the faintest stay graphite rather than turning brown.
                var cool = SKColor.Parse("#C2CBD6");
                var t = (float)Math.Clamp((w - 0.07) / 0.3, 0, 1);
                var mixed = new SKColor(
                    (byte)(cool.Red + (amber.Red - cool.Red) * t),
                    (byte)(cool.Green + (amber.Green - cool.Green) * t),
                    (byte)(cool.Blue + (amber.Blue - cool.Blue) * t));
                fill.Color = mixed.WithAlpha((byte)(14 + 120 * w));
            }

            c.DrawRoundRect(cell, fill);

            if (w > 0.5)
            {
                // A one-pixel top highlight keeps the strong cells crisp at 128 px.
                using var edge = new SKPaint { IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 2.5f, Color = SKColors.White.WithAlpha(70) };
                c.Save();
                c.ClipRect(new SKRect(x, y, x + Cell, y + 10));
                c.DrawRoundRect(new SKRoundRect(new SKRect(x + 1.25f, y + 1.25f, x + Cell - 1.25f, y + Cell - 1.25f), 10, 10), edge);
                c.Restore();
            }
        }
    }

    // The offset rail: Hub, Tokenizers, Datasets, Transformers (with Vision), PEFT, Accelerate,
    // Optimum, Diffusers - widths from each library's C# source in kilobytes at 0.11.0.
    (double Share, string Colour)[] rail =
    [
        (132, "#FFB454"), (170, "#F2915C"), (29, "#E2726A"), (366, "#C86089"),
        (159, "#9E63A8"), (12, "#7370BE"), (58, "#4E86C4"), (94, "#57C7E3"),
    ];

    const float RailLeft = Left, RailWidth = 5 * Cell + 4 * Gap, RailTop = 420, RailHeight = 22, Seam = 5;
    var total = rail.Sum(r => r.Share);
    var room = RailWidth - Seam * (rail.Length - 1);
    var at = RailLeft;
    foreach (var (share, colour) in rail)
    {
        var width = (float)(room * share / total);
        using var span = new SKPaint { IsAntialias = true, Color = SKColor.Parse(colour) };
        c.DrawRoundRect(new SKRoundRect(new SKRect(at, RailTop, at + width, RailTop + RailHeight), 5, 5), span);
        at += width + Seam;
    }

    c.Restore();

    // A hairline rim, brighter at the top left, so the tile holds its edge on a dark listing.
    using var rim = new SKPaint { IsAntialias = true, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
    rim.Shader = SKShader.CreateLinearGradient(new SKPoint(8, 8), new SKPoint(504, 504),
        [SKColors.White.WithAlpha(64), SKColors.White.WithAlpha(10)], SKShaderTileMode.Clamp);
    c.DrawRoundRect(tile, rim);
}
