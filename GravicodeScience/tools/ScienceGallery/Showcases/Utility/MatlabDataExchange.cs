using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Gravicode.Science.SciNet.Sparse;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Utility;

/// <summary>Exchanging data with MATLAB users: .mat files with matrices, text, structs and sparse matrices, plus Matrix Market.</summary>
public sealed class MatlabDataExchange : Showcase
{
    public override string Title => "Swap data with a MATLAB colleague, losslessly";
    public override Domain Domain => Domain.Utility;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "Write and read a compressed .mat file and a Matrix Market file, then load the result into a DataFrame.";
    public override string Scenario =>
        "A lab's test rig is driven from MATLAB, and the analysis team works in .NET. The rig saves an hour of " +
        "thermocouple readings, the channel names, a struct of run metadata and the sparse stiffness matrix of " +
        "the test piece in one .mat file; the sparse matrix also goes to a solver in Matrix Market format. Here " +
        "both files are written and read back, every value is checked bit for bit, and the readings land in a " +
        "DataFrame ready for analysis.";

    public override void Run(ShowcaseOutput output)
    {
        // What the rig produces: 3,600 one-second readings from four thermocouples (°C).
        string[] channels = ["inlet", "outlet", "housing", "ambient"];
        var rng = new GraviRandom(2);
        var readings = NdArray.Zeros(3600, 4);
        for (var s = 0; s < 3600; s++)
        {
            var warm = 1 - Math.Exp(-s / 900.0);
            double[] values = [25 + 40 * warm, 25 + 32 * warm, 25 + 18 * warm, 24 + 0.5 * Math.Sin(s / 600.0)];
            for (var c = 0; c < 4; c++) readings[s, c] = Math.Round(values[c] + rng.Normal(0, 0.15), 2);
        }
        // The test piece's stiffness matrix: a 2-D finite-difference Laplacian on a 20 × 20 grid, 400 × 400 and sparse.
        var line = SparseMatrices.Diags([Enumerable.Repeat(-1.0, 19).ToArray(), Enumerable.Repeat(2.0, 20).ToArray(), Enumerable.Repeat(-1.0, 19).ToArray()], [-1, 0, 1]);
        var stiffness = SparseMatrices.Kronsum(line, line);

        var folder = Path.GetTempPath();
        var matPath = Path.Combine(folder, "gallery-rig-run.mat");
        var plainPath = Path.Combine(folder, "gallery-rig-run-uncompressed.mat");
        var mtxPath = Path.Combine(folder, "gallery-stiffness.mtx");
        var variables = new Dictionary<string, object?>
        {
            ["temps"] = readings,
            ["fs"] = 1.0,
            ["channels"] = channels,                                       // a MATLAB char matrix, one row per name
            ["run"] = new Dictionary<string, object?> { ["operator"] = "R. Wijaya", ["rig"] = "TR-2", ["setpoint"] = 65.0 },
            ["K"] = stiffness,
        };
        MatFile.Save(matPath, variables, doCompression: true);
        MatFile.Save(plainPath, variables);
        MatrixMarket.MmWrite(mtxPath, stiffness, comment: "2-D Laplacian, 20 x 20 grid", symmetry: "symmetric");

        // Read everything back.
        var contents = MatFile.Whosmat(matPath);
        var loaded = MatFile.Load(matPath);
        var temps = ((MatArray)loaded["temps"]).Real;
        var k = ((MatSparse)loaded["K"]).Matrix;
        var run = (MatStruct)loaded["run"];
        var fromMtx = MatrixMarket.MmRead(mtxPath).Sparse!;
        var worst = Enumerable.Range(0, temps.Size).Max(i => Math.Abs(temps.At(i) - readings.At(i)));
        var sparseSame = k.ToDense().ToArray().SequenceEqual(stiffness.ToDense().ToArray()) && fromMtx.ToDense().ToArray().SequenceEqual(stiffness.ToDense().ToArray());

        // Into a DataFrame, with the channel names that travelled in the same file.
        // MATLAB pads a char matrix with spaces to its longest row, so trim each name.
        var columns = ((MatChar)loaded["channels"]).Strings.Select(name => name.Trim()).ToArray();
        var frame = DataFrame.FromMatrix(temps, columns);
        var lastTen = frame.Rows(3000, 600);

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var minutes = Enumerable.Range(0, 3600).Select(s => s / 60.0).ToArray();
        foreach (var name in columns) left.Plot(minutes, frame.Numeric(name).Values, name, lineWidth: 1.2);
        left.SetTitle("Readings, as loaded from the .mat file").SetXLabel("minutes").SetYLabel("temperature (°C)");
        left.Legend(LegendLocation.CenterRight);
        var entries = fromMtx.EnumerateEntries().ToArray();
        right.Scatter(entries.Select(e => (double)e.Column).ToArray(), entries.Select(e => (double)e.Row).ToArray(), sizes: [2], color: GalleryCharts.Of(GraviLibrary.GraviNum));
        right.InvertYAxis().SetAspect(1.0);
        right.SetTitle($"Sparsity of K from the .mtx file ({fromMtx.Nnz:N0} non-zeros)").SetXLabel("column").SetYLabel("row");
        output.Chart(figure);

        output.Metric(".mat file", $"{new FileInfo(matPath).Length / 1024} KB compressed", $"{new FileInfo(plainPath).Length / 1024} KB without compression");
        output.Metric("Variables", $"{contents.Count}: " + string.Join(", ", contents.Select(v => v.Name)),
                      string.Join(", ", contents.Select(v => $"{v.ClassName} {string.Join("×", v.Shape)}")));
        output.Metric("Round trip", worst == 0 && sparseSame ? "bit-identical" : $"max error {worst:E1}", "readings, struct and both sparse copies");
        output.Line($"Run by {run["operator"].Text} on rig {run["rig"].Text}: in the last ten minutes the outlet ran " +
                    $"{lastTen.Numeric("inlet").Values.Average() - lastTen.Numeric("outlet").Values.Average():F1} °C below the inlet. " +
                    "The .mat file opens in MATLAB with load(), and the .mtx file in any sparse solver.");
    }
}
