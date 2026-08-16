using System.Diagnostics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Gravicode.Science.GraviNum.Io;
using Gravicode.Science.GraviNum.Signal;

Console.WriteLine(GraviInfo.Banner("GraviNum"));
Console.WriteLine(GraviInfo.HardwareReport());
Console.WriteLine();

var screenshots = SampleSupport.ResolveScreenshotDirectory();

// ---------------------------------------------------------------- arrays
Section("1. Creating and reshaping arrays");

var range = NdArray.Arange(12);
Console.WriteLine($"  Arange(12)          -> {string.Join(", ", range.ToArray())}");

var matrix = range.Reshape(3, 4);
Console.WriteLine($"  Reshape(3, 4)       -> shape [{Shapes.Describe(matrix.Shape)}]");
Console.WriteLine(matrix.ToString(2));

Console.WriteLine("  Transpose is a view, not a copy - no data is moved:");
Console.WriteLine($"    matrix[1, 2] = {matrix[1, 2]}   transposed[2, 1] = {matrix.T[2, 1]}");

var slice = matrix.Slice(Slice.All, Slice.Range(1, 3));
Console.WriteLine($"  matrix[:, 1:3]      -> shape [{Shapes.Describe(slice.Shape)}], contiguous = {slice.IsContiguous}");
Console.WriteLine();

// ---------------------------------------------------------------- broadcasting
Section("2. Broadcasting and element-wise maths");

var rows = NdArray.Ones(3, 4);
var offsets = NdArray.Arange(4);
Console.WriteLine("  Ones(3,4) + Arange(4) stretches the vector across every row:");
Console.WriteLine((rows + offsets).ToString(2));

var angles = NdArray.Linspace(0, Math.PI, 5);
Console.WriteLine($"  sin(linspace(0, pi, 5)) = {Format(UFunc.Sin(angles))}");
Console.WriteLine($"  SIMD enabled: {UFunc.IsSimdAccelerated} (Vector<double> width {UFunc.VectorWidth})");
Console.WriteLine();

// ---------------------------------------------------------------- linear algebra
Section("3. Linear algebra");

var a = NdArray.FromArray(new double[,] { { 4, 7, 2 }, { 3, 6, 1 }, { 2, 5, 9 } });
Console.WriteLine("  A =");
Console.WriteLine(a.ToString(2));

Console.WriteLine($"  det(A)       = {LinAlg.Determinant(a):F6}");
Console.WriteLine($"  trace(A)     = {LinAlg.Trace(a):F6}");
Console.WriteLine($"  rank(A)      = {LinAlg.MatrixRank(a)}");
Console.WriteLine($"  cond(A)      = {LinAlg.ConditionNumber(a):F4}");

var inverse = LinAlg.Inverse(a);
var identity = LinAlg.Dot(a, inverse);
Console.WriteLine($"  A * inv(A) == I within 1e-10: {UFunc.AllClose(identity, NdArray.Eye(3), 1e-10)}");

var b = NdArray.FromValues([10.0, 8.0, 26.0]);
var solution = LinAlg.Solve(a, b);
Console.WriteLine($"  solve(A, b)  = {Format(solution)}");
Console.WriteLine($"  residual     = {LinAlg.Norm(LinAlg.Dot(a, solution) - b):E3}");
Console.WriteLine();

Section("4. Decompositions");

var lu = Decomposition.Lu(a);
Console.WriteLine($"  LU  : P A == L U            -> {UFunc.AllClose(LinAlg.Dot(lu.PermutationMatrix(), a), LinAlg.Dot(lu.Lower, lu.Upper), 1e-10)}");

var qr = Decomposition.Qr(a);
Console.WriteLine($"  QR  : Q orthonormal          -> {UFunc.AllClose(LinAlg.Dot(qr.Q.T, qr.Q), NdArray.Eye(3), 1e-10)}");
Console.WriteLine($"        Q R == A               -> {UFunc.AllClose(LinAlg.Dot(qr.Q, qr.R), a, 1e-10)}");

var svd = Decomposition.Svd(a);
Console.WriteLine($"  SVD : singular values        -> {Format(svd.SingularValues)}");
Console.WriteLine($"        U S V' == A            -> {UFunc.AllClose(svd.Reconstruct(), a, 1e-9)}");

var symmetric = (a + a.T) * 0.5;
var eigen = Decomposition.SymmetricEigen(symmetric);
Console.WriteLine($"  Eigen (symmetric part)       -> {Format(eigen.Values)}");
Console.WriteLine();

// ---------------------------------------------------------------- statistics
Section("5. Random numbers and statistics");

var rng = new GraviRandom(seed: 42);
var samples = rng.Normal(mean: 100, stdDev: 15, 100_000);

Console.WriteLine($"  100,000 draws from Normal(100, 15)");
foreach (var (key, value) in Statistics.Describe(samples))
    Console.WriteLine($"    {key,-8}{value,12:F4}");

var poisson = rng.Poisson(4.0, 50_000);
Console.WriteLine($"  Poisson(4): mean {Statistics.Mean(poisson):F3}, variance {Statistics.Var(poisson):F3} (both should equal lambda)");

var x = rng.StandardNormal(2000);
var y = x * 0.8 + rng.Normal(0, 0.6, 2000);
Console.WriteLine($"  correlation(x, 0.8x + noise) = {Statistics.Correlation(x, y):F4}");
Console.WriteLine();

// ---------------------------------------------------------------- sparse
Section("6. Sparse matrices");

var dense = NdArray.Zeros(500, 500);
for (var i = 0; i < 500; i++)
    for (var k = 0; k < 4; k++) dense[i, rng.Next(500)] = rng.Normal();

var sparse = SparseMatrix.FromDense(dense);
Console.WriteLine($"  {sparse}");
Console.WriteLine($"  dense storage : {500 * 500 * 8 / 1024.0:F1} KB");
Console.WriteLine($"  sparse storage: {(sparse.NonZeroCount * 12 + 501 * 4) / 1024.0:F1} KB");

var vector = rng.StandardNormal(500);
Console.WriteLine($"  sparse * v == dense * v: {UFunc.AllClose(sparse.Multiply(vector), LinAlg.Dot(dense, vector), 1e-10)}");
Console.WriteLine();

// ---------------------------------------------------------------- performance
Section("7. Matrix multiplication performance");

foreach (var size in new[] { 128, 256, 512 })
{
    var left = rng.StandardNormal(size, size);
    var right = rng.StandardNormal(size, size);

    var watch = Stopwatch.StartNew();
    var product = LinAlg.Dot(left, right);
    watch.Stop();

    // 2*n^3 flops for an n-cubed matrix product.
    var gflops = 2.0 * size * size * size / watch.Elapsed.TotalSeconds / 1e9;
    Console.WriteLine($"  {size,4} x {size,-4}  {watch.ElapsedMilliseconds,6} ms   {gflops,7:F2} GFLOP/s   (checksum {product[0, 0]:F4})");
}

Console.WriteLine($"  Backend selected for a 512x512 product: {Compute.Best(512 * 512).Name}");
Console.WriteLine();

// ---------------------------------------------------------------- io
Section("8. Persistence and memory mapping");

var temporary = Directory.CreateTempSubdirectory("gravinum-sample");
try
{
    var payload = rng.StandardNormal(1000, 10);

    var binaryPath = Path.Combine(temporary.FullName, "matrix.gnb");
    NdIO.SaveBinary(payload, binaryPath);
    var reloaded = NdIO.LoadBinary(binaryPath);
    Console.WriteLine($"  binary round trip ({new FileInfo(binaryPath).Length / 1024} KB): {UFunc.AllClose(payload, reloaded, 0)}");

    var mappedPath = Path.Combine(temporary.FullName, "matrix.gmm");
    using (MemoryMappedArray.Persist(payload, mappedPath)) { }
    using var mapped = MemoryMappedArray.Open(mappedPath);

    Console.WriteLine($"  memory mapped: {mapped.Length:N0} elements, reading row 42 without loading the file");
    Console.WriteLine($"    row 42 sum = {mapped.ReadRow(42).Sum():F6} (expected {payload.Row(42).Sum():F6})");
}
finally { temporary.Delete(recursive: true); }
Console.WriteLine();

// ---------------------------------------------------------------- visualisation
Section("9. Heatmap");

var heat = NdArray.Zeros(40, 40);
for (var i = 0; i < 40; i++)
    for (var j = 0; j < 40; j++)
    {
        // A smooth two-dimensional function, so the heatmap shows structure rather than noise.
        var u = (i - 20) / 8.0;
        var v = (j - 20) / 8.0;
        heat[i, j] = Math.Exp(-(u * u + v * v) / 2) * Math.Cos(u * 2) * Math.Sin(v * 2);
    }

var heatmapPath = Path.Combine(screenshots, "gravinum_heatmap.png");
var plot = new ScottPlot.Plot();
plot.Add.Heatmap(heat.To2DArray());
plot.Title("GraviNum - 40x40 matrix heatmap");
plot.SavePng(heatmapPath, 900, 700);
Console.WriteLine($"  saved {heatmapPath}");

// ---------------------------------------------------------------- v0.4: einsum
Section("10. Einstein summation");

var ea = rng.StandardNormal(4, 5);
var eb = rng.StandardNormal(5, 3);

Console.WriteLine("  One notation covers products, transposes, traces and contractions.");
Console.WriteLine($"  \"ij,jk->ik\" matches LinAlg.Dot: {UFunc.AllClose(Einsum.Evaluate("ij,jk->ik", ea, eb), LinAlg.Dot(ea, eb), 1e-12)}");
Console.WriteLine($"  \"ji,jk->ik\" is Dot(a.T, b), but the axes are written down rather than implied");

var square = NdArray.FromArray(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
Console.WriteLine($"  \"ii->i\"  diagonal = {Format(Einsum.Evaluate("ii->i", square))}");
Console.WriteLine($"  \"ii->\"   trace    = {Einsum.Evaluate("ii->", square).At(0):F1}");
Console.WriteLine($"  \"ij->j\"  col sums = {Format(Einsum.Evaluate("ij->j", square))}");

// Rank 3 is where writing this out by hand stops being readable.
var batchA = rng.StandardNormal(3, 4, 5);
var batchB = rng.StandardNormal(3, 5, 2);
Console.WriteLine($"  \"bij,bjk->bik\" batched product -> shape [{Shapes.Describe(Einsum.Evaluate("bij,bjk->bik", batchA, batchB).Shape)}]");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: complex
Section("11. Complex arrays and the 2-D transform");

var signal = ComplexNdArray.FromParts(rng.StandardNormal(64), rng.StandardNormal(64));
Console.WriteLine($"  round trip through Fft/Ifft holds: {signal.Fft().Ifft().AllClose(signal, 1e-10)}");

// Parseval: energy is conserved, which pins the scaling convention.
var timeEnergy = 0.0;
for (var i = 0; i < signal.Size; i++) timeEnergy += signal.Power().At(i);
var spectrum = signal.Fft();
var freqEnergy = 0.0;
for (var i = 0; i < spectrum.Size; i++) freqEnergy += spectrum.Power().At(i);
Console.WriteLine($"  Parseval:  time {timeEnergy:F4}  vs  frequency/N {freqEnergy / signal.Size:F4}");

var hermitian = ComplexNdArray.FromParts(rng.StandardNormal(3, 3), rng.StandardNormal(3, 3));
var gram = ComplexNdArray.Dot(hermitian.ConjugateTranspose(), hermitian);
Console.WriteLine("  A^H A has a real, non-negative diagonal; A^T A would not:");
Console.WriteLine($"    diag = [{gram[0, 0].Real:F4}, {gram[1, 1].Real:F4}, {gram[2, 2].Real:F4}]  " +
                  $"max |imag| = {Math.Max(Math.Abs(gram[0, 0].Imaginary), Math.Abs(gram[2, 2].Imaginary)):E1}");

var image = ComplexNdArray.FromParts(rng.StandardNormal(16, 16));
Console.WriteLine($"  separable 2-D transform round trips: {image.Fft2().Ifft2().AllClose(image, 1e-10)}");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: slicing
Section("12. Slice ergonomics");

var grid = NdArray.Arange(12).Reshape(3, 4).Copy();
Console.WriteLine("  grid =");
Console.WriteLine(grid.ToString(0));

// Slice returns a view, so writing through it changes the original.
grid.Slice(Slice.Range(1, 2)).Assign(0.0);
Console.WriteLine("  after grid[1:2] = 0 - the write went through the view:");
Console.WriteLine(grid.ToString(0));

var volume = NdArray.Arange(24).Reshape(2, 3, 4);
Console.WriteLine($"  SliceEllipsis([], [Slice.At(3)]) takes the last axis without counting the leading ones");
Console.WriteLine($"    -> shape [{Shapes.Describe(volume.SliceEllipsis([], [Slice.At(3)]).Shape)}], and stays right if the rank grows");

var table = NdArray.Arange(12).Reshape(3, 4);
Console.WriteLine($"  TakeAlong([3, 0], axis: 1) picks columns: [{Shapes.Describe(table.TakeAlong([3, 0], axis: 1).Shape)}]");
Console.WriteLine($"  IndicesWhere(v => v > 8)   = [{string.Join(", ", table.IndicesWhere(v => v > 8))}]");
Console.WriteLine($"  Clip(2, 8) first row       = {Format(table.Clip(2, 8).Row(0))}");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: spectrum plot
Section("13. Power spectrum");

// Two tones plus noise: the transform should show two peaks and a floor.
const int sampleCount = 512;
const double sampleRate = 256.0;
var tone = NdArray.Zeros(sampleCount);
for (var i = 0; i < sampleCount; i++)
{
    var t = i / sampleRate;
    tone.SetAt(i, Math.Sin(2 * Math.PI * 12 * t) + 0.5 * Math.Sin(2 * Math.PI * 40 * t) + 0.15 * rng.Normal());
}

var bins = Fft.FrequencyBins(sampleCount, sampleRate);
var power = Fft.Magnitude(tone.ToArray());

var spectrumPath = Path.Combine(screenshots, "gravinum_spectrum.png");
var spectrumPlot = new ScottPlot.Plot();
var half = sampleCount / 2;
spectrumPlot.Add.Scatter(bins.ToArray().Take(half).ToArray(), power.ToArray().Take(half).ToArray());
spectrumPlot.Title("GraviNum - power spectrum: 12 Hz and 40 Hz recovered from noise");
spectrumPlot.XLabel("frequency (Hz)");
spectrumPlot.YLabel("magnitude");
spectrumPlot.SavePng(spectrumPath, 900, 500);
Console.WriteLine($"  two tones at 12 Hz and 40 Hz, buried in noise, recovered by the transform");
Console.WriteLine($"  saved {spectrumPath}");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
{
    Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));
}

static string Format(NdArray array, int max = 6)
{
    var values = array.ToArray().Take(max).Select(v => v.ToString("F4"));
    return "[" + string.Join(", ", values) + (array.Size > max ? ", ...]" : "]");
}

/// <summary>Locates the repository's screenshot directory so samples can write their output there.</summary>
internal static class SampleSupport
{
    public static string ResolveScreenshotDirectory()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, "docs", "screenshots");
            if (Directory.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }

        var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
        Directory.CreateDirectory(fallback);
        return fallback;
    }
}
