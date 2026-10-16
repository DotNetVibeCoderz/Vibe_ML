using System.Diagnostics;
using System.Globalization;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Gravicode.Science.SciNet.IO;
using Gravicode.Science.SciNet.Sparse;
using Gravicode.Science.SciNet.Spatial;

Console.WriteLine(SciNetInfo.Banner());

var datasets = Resolve("datasets") ?? throw new DirectoryNotFoundException("datasets directory not found.");
var screenshots = ResolveScreenshots();
var watch = new Stopwatch();

// ---------------------------------------------------------------- special
Section("1. Special functions and constants");

Console.WriteLine($"  Gamma(0.5)       = {Special.Gamma(0.5):R}   (sqrt(pi) = {Math.Sqrt(Math.PI):R})");
Console.WriteLine($"  Jv(2.5, 10)      = {Special.Jv(2.5, 10):R}");
Console.WriteLine($"  Erfc(6)          = {Special.Erfc(6):E6}   (computed directly, not as 1 - Erf)");
Console.WriteLine($"  LogNdtr(-40)     = {Special.LogNdtr(-40):F6}");
Console.WriteLine($"  EllipK(0.99)     = {Special.EllipK(0.99):R}");
var gauss = Special.RootsLegendre(5);
Console.WriteLine($"  5-point Gauss-Legendre of x^8 on [-1,1]: {gauss.Integrate(x => Math.Pow(x, 8)):R} (exact 2/9 = {2.0 / 9:R})");
Console.WriteLine($"  speed of light   = {Constants.SpeedOfLight} m/s");
Console.WriteLine($"  Planck constant  = {Constants.Value("Planck constant")} {Constants.Unit("Planck constant")}");
Console.WriteLine($"  98.6 F in C      = {Constants.ConvertTemperature(98.6, "F", "C"):F2}");
Console.WriteLine();

// ---------------------------------------------------------------- optimize
Section("2. Optimisation");

double Rosen(double[] x) => Optimize.Rosen(x);
double[] RosenDer(double[] x) => Optimize.RosenDer(x);
// scipy's tutorial start point. From (-1.2, 1, ...) a 10-D Rosenbrock has a genuine local minimum
// near (-1, 1, ..., 1) with f = 3.99, which BFGS legitimately finds.
double[] x0 = [1.3, 0.7, 0.8, 1.9, 1.2];
foreach (var method in new[] { MinimizeMethod.NelderMead, MinimizeMethod.Bfgs, MinimizeMethod.LBfgsB })
{
    watch.Restart();
    var r = Optimize.Minimize(Rosen, x0, method, jac: method == MinimizeMethod.NelderMead ? null : RosenDer,
        options: method == MinimizeMethod.NelderMead ? new NelderMeadOptions { Xatol = 1e-8 } : null);
    watch.Stop();
    Console.WriteLine($"  Rosenbrock-5, {method,-11}: f = {r.Fun:E2}, nit {r.Nit,5}, nfev {r.Nfev,6}, {watch.Elapsed.TotalMilliseconds,7:F2} ms");
}

// curve fitting: exponential decay with deterministic noise
var tData = Enumerable.Range(0, 50).Select(i => i * 0.1).ToArray();
var yData = tData.Select((t, i) => 2.5 * Math.Exp(-1.3 * t) + 0.5 + 0.05 * Math.Sin(17.0 * i)).ToArray();
var fit = Optimize.CurveFit((t, p) => p[0] * Math.Exp(-p[1] * t) + p[2], tData, yData, p0: [1, 1, 1]);
Console.WriteLine($"  curve fit a*exp(-b t)+c: a = {fit.Popt[0]:F4} +/- {fit.Perr[0]:F4}, b = {fit.Popt[1]:F4} +/- {fit.Perr[1]:F4}, c = {fit.Popt[2]:F4} +/- {fit.Perr[2]:F4}");

var root = Optimize.Brentq(x => Math.Cos(x) - x, 0, 1);
Console.WriteLine($"  root of cos(x) = x: {root.Root:R} in {root.Iterations} iterations");

var assignmentCost = NdArray.FromRows([[4, 1, 3], [2, 0, 5], [3, 2, 2]]);
var (rows, cols) = Optimize.LinearSumAssignment(assignmentCost);
Console.WriteLine($"  assignment: {string.Join(", ", rows.Zip(cols, (r, c) => $"{r}->{c}"))}, cost {rows.Zip(cols, (r, c) => assignmentCost[r, c]).Sum()}");

var lp = Optimize.Linprog([-1, 4], aUb: NdArray.FromRows([[-3, 1], [1, 2]]), bUb: [6, 4]);
Console.WriteLine($"  linprog: x = [{string.Join(", ", lp.X.Select(v => v.ToString("F4")))}], objective {lp.Fun:F4}");

var de = Optimize.DifferentialEvolution(x => 10 * x.Length + x.Sum(v => v * v - 10 * Math.Cos(2 * Math.PI * v)),
    new Bounds([-5.12, -5.12, -5.12], [5.12, 5.12, 5.12]), seed: 7);
Console.WriteLine($"  differential evolution on Rastrigin-3: f = {de.Fun:E2} at [{string.Join(", ", de.X.Select(v => v.ToString("F4")))}]");
Console.WriteLine();

// ---------------------------------------------------------------- integrate
Section("3. Integration and ODEs");

var q = Integrate.Quad(x => Math.Log(x) / Math.Sqrt(x), 0, 1);
Console.WriteLine($"  quad log(x)/sqrt(x) on [0,1] = {q.Value:R} (exact -4), abserr {q.AbsErr:E2}, neval {q.Neval}");
var qi = Integrate.Quad(x => Math.Exp(-x * x), double.NegativeInfinity, double.PositiveInfinity);
Console.WriteLine($"  quad exp(-x^2) over R     = {qi.Value:R} (sqrt(pi) = {Math.Sqrt(Math.PI):R})");

void Lorenz(double t, ReadOnlySpan<double> y, Span<double> dy)
{
    dy[0] = 10 * (y[1] - y[0]);
    dy[1] = y[0] * (28 - y[2]) - y[1];
    dy[2] = y[0] * y[1] - 8.0 / 3.0 * y[2];
}

watch.Restart();
var lorenz = Integrate.SolveIvp(Lorenz, 0, 40, [1.0, 1.0, 1.0], OdeMethod.Dop853, rtol: 1e-9, atol: 1e-12, denseOutput: true);
watch.Stop();
Console.WriteLine($"  Lorenz 0..40 with DOP853: {lorenz.T.Length} steps, nfev {lorenz.Nfev}, {watch.Elapsed.TotalMilliseconds:F2} ms");

void VanDerPol(double t, ReadOnlySpan<double> y, Span<double> dy)
{
    dy[0] = y[1];
    dy[1] = 1000 * (1 - y[0] * y[0]) * y[1] - y[0];
}

watch.Restart();
var stiff = Integrate.SolveIvp(VanDerPol, 0, 3000, [2.0, 0.0], OdeMethod.Bdf, rtol: 1e-6);
watch.Stop();
Console.WriteLine($"  stiff Van der Pol mu=1000 with BDF: {stiff.T.Length} steps, nlu {stiff.Nlu}, {watch.Elapsed.TotalMilliseconds:F2} ms");

var dense = lorenz.Sol!.Evaluate(Enumerable.Range(0, 8000).Select(i => i * 40.0 / 7999).ToArray());
var lorenzPlot = new ScottPlot.Plot();
var trajectory = lorenzPlot.Add.Scatter(dense.Row(0).ToArray(), dense.Row(2).ToArray());
trajectory.MarkerSize = 0;
trajectory.LineWidth = 1;
lorenzPlot.Title("SciNet - Lorenz attractor, DOP853 with dense output");
lorenzPlot.XLabel("x");
lorenzPlot.YLabel("z");
Save(lorenzPlot, "scinet_lorenz.png", 900, 650);
Console.WriteLine();

// ---------------------------------------------------------------- interpolate
Section("4. Interpolation");

var knotsX = Enumerable.Range(0, 11).Select(i => i * Math.PI / 5).ToArray();
var knotsY = knotsX.Select(Math.Sin).ToArray();
var spline = new CubicSpline(knotsX, knotsY);
Console.WriteLine($"  cubic spline of sin on 11 knots: s(1.0) = {spline.Evaluate(1.0):F6}, sin(1.0) = {Math.Sin(1.0):F6}");
Console.WriteLine($"  integral of the spline over [0, pi] = {spline.Integrate(0, Math.PI):F6} (exact 2)");
var pchip = new PchipInterpolator([0, 1, 2, 3, 4], [0, 0, 1, 1, 1]);
Console.WriteLine($"  pchip keeps a step monotone: p(1.5) = {pchip.Evaluate(1.5):F4}, p(2.5) = {pchip.Evaluate(2.5):F4}");
Console.WriteLine();

// ---------------------------------------------------------------- linalg
Section("5. Linear algebra beyond numpy");

var rotation = NdArray.FromRows([[0, -1], [1, 0]]);
var expRotation = Linalg.Expm(rotation * (Math.PI / 2));
Console.WriteLine($"  expm of a quarter-turn generator: [[{expRotation[0, 0]:F6}, {expRotation[0, 1]:F6}], [{expRotation[1, 0]:F6}, {expRotation[1, 1]:F6}]]");
var a = NdArray.FromRows([[4, 1, 0], [1, 3, 1], [0, 1, 2]]);
var sqrtA = Linalg.Sqrtm(a);
Console.WriteLine($"  sqrtm(A)^2 - A, max abs error: {MaxAbs(LinAlg.Dot(sqrtA, sqrtA) - a):E2}");
var sylvesterX = Linalg.SolveSylvester(a, a, NdArray.Eye(3));
Console.WriteLine($"  Sylvester AX + XA = I residual: {MaxAbs(LinAlg.Dot(a, sylvesterX) + LinAlg.Dot(sylvesterX, a) - NdArray.Eye(3)):E2}");
var schur = Linalg.Schur(NdArray.FromRows([[0, 2, 1], [-2, 0, 3], [0, 0, 1]]));
Console.WriteLine($"  real Schur form keeps the complex pair as a 2x2 block: T[1,0] = {schur.T[1, 0]:F4}");
Console.WriteLine($"  derivative of sin at 1 (Richardson): {Differentiate.Derivative(Math.Sin, 1.0).Df:R} vs cos(1) = {Math.Cos(1.0):R}");
Console.WriteLine();

// ---------------------------------------------------------------- sparse
Section("6. Sparse matrices and graphs");

const int grid = 200;
var d1 = SparseMatrices.Diags([Enumerable.Repeat(-1.0, grid - 1).ToArray(), Enumerable.Repeat(2.0, grid).ToArray(),
    Enumerable.Repeat(-1.0, grid - 1).ToArray()], [-1, 0, 1]);
var poisson = SparseMatrices.Kronsum(d1, d1);
var rhs = Enumerable.Repeat(1.0, grid * grid).ToArray();
watch.Restart();
var cg = SparseLinalg.Cg(poisson, rhs, rtol: 1e-8, maxiter: 5000);
watch.Stop();
Console.WriteLine($"  2-D Poisson {grid}x{grid} ({poisson.Nnz:N0} non-zeros): CG converged = {cg.Converged} in {cg.Iterations} iterations, {watch.ElapsedMilliseconds} ms");
var lowest = SparseLinalg.Eigsh(poisson, k: 3, sigma: 0);
Console.WriteLine($"  three smallest eigenvalues (shift-invert): {string.Join(", ", lowest.Values.Select(v => v.ToString("F6")))}");
var road = CsrMatrix.FromDense(NdArray.FromRows([
    [0, 7, 9, 0, 0, 14], [7, 0, 10, 15, 0, 0], [9, 10, 0, 11, 0, 2],
    [0, 15, 11, 0, 6, 0], [0, 0, 0, 6, 0, 9], [14, 0, 2, 0, 9, 0]]));
var shortest = CsGraph.Dijkstra(road, directed: false, indices: [0]);
Console.WriteLine($"  Dijkstra from node 0: [{string.Join(", ", shortest.Distances.Row(0).ToArray())}]");
Console.WriteLine();

// ---------------------------------------------------------------- spatial and cluster
Section("7. Spatial algorithms and clustering");

var iris = File.ReadAllLines(Path.Combine(datasets, "iris.csv")).Skip(1).Where(l => l.Length > 0)
    .Select(l => l.Split(',').Take(4).Select(v => double.Parse(v, CultureInfo.InvariantCulture)).ToArray()).ToArray();
var irisArray = NdArray.FromRows(iris);
var tree = new KDTree(irisArray);
var (nearDist, nearIdx) = tree.Query(iris[0], k: 4);
Console.WriteLine($"  KD-tree: the 3 nearest neighbours of iris[0] are {string.Join(", ", nearIdx.Skip(1))} at distances {string.Join(", ", nearDist.Skip(1).Select(v => v.ToString("F3")))}");
var rng = new GraviRandom(3);
var cloud = rng.Uniform(0, 1, 400, 2);
var triangulation = new Delaunay(cloud);
var hull = new ConvexHull(cloud);
Console.WriteLine($"  Delaunay of 400 random points: {triangulation.Simplices.GetLength(0)} triangles, hull of {hull.Vertices.Length} vertices, area {hull.Volume:F4}");

var linkage = Cluster.Linkage(irisArray, LinkageMethod.Ward);
var labels = Cluster.Fcluster(linkage, 3, FclusterCriterion.Maxclust);
Console.WriteLine($"  Ward linkage on Iris cut into 3 clusters: sizes {string.Join(", ", labels.GroupBy(v => v).OrderBy(g => g.Key).Select(g => g.Count()))}");
var Tab10 = Tab10Colors();
var tree30 = Cluster.Dendrogram(linkage, p: 30, truncateMode: DendrogramTruncateMode.LastP);
var dendroPlot = new ScottPlot.Plot();
for (var i = 0; i < tree30.Icoord.Count; i++)
{
    var link = dendroPlot.Add.Scatter(tree30.Icoord[i], tree30.Dcoord[i]);
    link.MarkerSize = 0;
    link.LineWidth = 2;
    // ColorList holds matplotlib's "C0".."C9" codes, exactly as scipy returns them.
    link.Color = ScottPlot.Color.FromHex(Tab10[int.Parse(tree30.ColorList[i][1..]) % Tab10.Length]);
}
dendroPlot.Title("SciNet - Ward dendrogram of Iris (last 30 merges)");
dendroPlot.YLabel("merge distance");
Save(dendroPlot, "scinet_dendrogram.png", 1000, 550);
Console.WriteLine();

// ---------------------------------------------------------------- signal and fft
Section("8. Signal processing and FFT");

const double fs = 1000.0;
var time = Enumerable.Range(0, 2000).Select(i => i / fs).ToArray();
var noisy = time.Select((t, i) => Math.Sin(2 * Math.PI * 5 * t) + 0.4 * Math.Sin(2 * Math.PI * 120 * t) + 0.2 * Math.Sin(i * 1.7)).ToArray();
var butter = Signal.Butter(4, 20.0, FilterType.Lowpass, fs: fs);
var smooth = Signal.Filtfilt(butter.B, butter.A, noisy);
Console.WriteLine($"  4th-order Butterworth at 20 Hz: b = [{string.Join(", ", butter.B.Select(v => v.ToString("E3")))}]");
var spectrum = Fft.Rfft(noisy);
var freqs = Fft.RfftFreq(noisy.Length, 1 / fs);
var peakBin = Enumerable.Range(1, spectrum.Length - 1).OrderByDescending(k => spectrum[k].Magnitude).Take(2).Order().ToArray();
Console.WriteLine($"  rfft finds the two tones at {string.Join(" Hz and ", peakBin.Select(k => freqs[k]))} Hz");

var filterPlot = new ScottPlot.Plot();
var raw = filterPlot.Add.Scatter(time, noisy);
raw.LegendText = "signal + 120 Hz hum";
raw.MarkerSize = 0;
var clean = filterPlot.Add.Scatter(time, smooth);
clean.LegendText = "filtfilt, Butterworth 20 Hz";
clean.MarkerSize = 0;
clean.LineWidth = 3;
filterPlot.Title("SciNet - zero-phase low-pass filtering");
filterPlot.XLabel("time (s)");
filterPlot.ShowLegend();
Save(filterPlot, "scinet_filtfilt.png", 1000, 500);

// Spectral analysis: a long noisy recording with three tones, Welch-averaged.
var noise = new GraviRandom(11);
var recording = Enumerable.Range(0, 200_000).Select(i =>
{
    var t = i / fs;
    return Math.Sin(2 * Math.PI * 50 * t) + 0.5 * Math.Sin(2 * Math.PI * 120 * t) + 0.2 * Math.Sin(2 * Math.PI * 310 * t)
           + noise.Normal(0, 1.0);
}).ToArray();
watch.Restart();
var (welchF, welchP) = Signal.Welch(recording, fs: fs, nperseg: 2048);
watch.Stop();
var welchDb = welchP.Select(v => 10 * Math.Log10(v)).ToArray();
var tones = Signal.FindPeaks(welchDb, prominence: 10.0);
Console.WriteLine($"  Welch PSD of 200,000 samples in {watch.Elapsed.TotalMilliseconds:F1} ms; peaks at " +
                  string.Join(", ", tones.Peaks.Select(k => $"{welchF[k]:F1} Hz")));

var psdPlot = new ScottPlot.Plot();
var psd = psdPlot.Add.Scatter(welchF, welchDb);
psd.MarkerSize = 0;
var marks = psdPlot.Add.Scatter(tones.Peaks.Select(k => welchF[k]).ToArray(), tones.Peaks.Select(k => welchDb[k]).ToArray());
marks.LineWidth = 0;
marks.MarkerSize = 10;
psdPlot.Title("SciNet - Welch power spectral density with find_peaks");
psdPlot.XLabel("frequency (Hz)");
psdPlot.YLabel("PSD (dB/Hz)");
Save(psdPlot, "scinet_welch.png", 1000, 500);
Console.WriteLine();

// ---------------------------------------------------------------- ndimage
Section("9. N-dimensional image processing");

var image = NdArray.Zeros(64, 64);
for (var i = 0; i < 64; i++)
    for (var j = 0; j < 64; j++)
        if ((i - 20) * (i - 20) + (j - 20) * (j - 20) < 100 || (i > 38 && i < 56 && j > 34 && j < 58)) image[i, j] = 1;
var blurred = NdImage.GaussianFilter(image, 2.0);
var (labelled, count) = NdImage.Label(image);
var edt = NdImage.DistanceTransformEdt(image);
Console.WriteLine($"  label found {count} objects; Gaussian blur keeps total mass {Statistics.Sum(blurred):F2} of {Statistics.Sum(image):F0}");
Console.WriteLine($"  deepest pixel is {Statistics.Max(edt):F3} pixels from the background");
Console.WriteLine($"  objects' centres of mass: {string.Join("; ", NdImage.CenterOfMass(image, labelled, [1, 2]).Select(c => $"({c[0]:F1}, {c[1]:F1})"))}");
Console.WriteLine();

// ---------------------------------------------------------------- stats
Section("10. Statistics: distributions, tests, KDE and QMC");

var normal = Stats.Norm(loc: 100, scale: 15);
Console.WriteLine($"  Norm(100, 15): Pdf(115) = {normal.Pdf(115):F6}, Cdf(130) = {normal.Cdf(130):F6}, Ppf(0.975) = {normal.Ppf(0.975):F4}");
Console.WriteLine($"  far tail done right: Norm.Sf(30) = {Stats.Norm().Sf(30):E4} (1 - Cdf would give 0)");
var t = Stats.T(df: 5);
Console.WriteLine($"  Student t(5): 95% interval [{t.Interval(0.95).Lower:F4}, {t.Interval(0.95).Upper:F4}], kurtosis {t.Kurtosis():F1}");
Console.WriteLine($"  Binom(20, 0.3): Pmf(6) = {Stats.Binom(20, 0.3).Pmf(6):F6}, Ppf(0.9) = {Stats.Binom(20, 0.3).Ppf(0.9)}");

var draws = Stats.Gamma(a: 2.5, scale: 1.5).Rvs(50_000, seed: 3);
var fitted = (GammaDistribution)Stats.Gamma(a: 1).FitTo(draws, floc: 0);
Console.WriteLine($"  Gamma(2.5, scale 1.5) fitted from 50,000 draws: a = {fitted.A:F3}, scale = {fitted.Scale:F3}");
var mvn = Stats.MultivariateNormal([0, 0], NdArray.FromRows([[1, 0.6], [0.6, 1]]));
Console.WriteLine($"  bivariate normal, rho 0.6: P(X < 0, Y < 0) = {mvn.Cdf([0, 0], seed: 1):F4} (exact {0.25 + Math.Asin(0.6) / (2 * Math.PI):F4})");
// Hypothesis tests on Iris: setosa (rows 0-49) against versicolor (rows 50-99).
var setosaSepal = iris.Take(50).Select(r => r[0]).ToArray();
var versicolorSepal = iris.Skip(50).Take(50).Select(r => r[0]).ToArray();
var welch = Stats.TtestInd(setosaSepal, versicolorSepal, equalVar: false);
Console.WriteLine($"  Welch t-test, sepal length setosa vs versicolor: t = {welch.Statistic:F3}, p = {welch.Pvalue:E3}");
var mwu = Stats.MannWhitneyU(setosaSepal, versicolorSepal);
Console.WriteLine($"  Mann-Whitney U = {mwu.Statistic}, p = {mwu.Pvalue:E3}");
var shapiro = Stats.Shapiro(setosaSepal);
Console.WriteLine($"  Shapiro-Wilk on setosa sepal length: W = {shapiro.Statistic:F4}, p = {shapiro.Pvalue:F4}");
var petalLength = iris.Select(r => r[2]).ToArray();
var petalWidth = iris.Select(r => r[3]).ToArray();
var regression = Stats.Linregress(petalLength, petalWidth);
Console.WriteLine($"  petal width = {regression.Slope:F4} * length {(regression.Intercept < 0 ? "-" : "+")} {Math.Abs(regression.Intercept):F4}, r = {regression.Rvalue:F4}");
var ks = Stats.Kstest(draws, Stats.Gamma(a: 2.5, scale: 1.5));
Console.WriteLine($"  KS test of the gamma draws against their true law: D = {ks.Statistic:F5}, p = {ks.Pvalue:F3}");

var kde = new GaussianKde(petalLength);
Console.WriteLine($"  KDE of petal length (Scott bandwidth {kde.Factor:F3}): density at 1.5 = {kde.Evaluate([1.5])[0]:F4}, at 4.5 = {kde.Evaluate([4.5])[0]:F4}");
var sobol = new SobolEngine(2, scramble: false).RandomBase2(8);
var pseudo = new GraviRandom(5).Random(256, 2);
Console.WriteLine($"  centred discrepancy of 256 points: Sobol {Qmc.Discrepancy(sobol):E3} vs pseudo-random {Qmc.Discrepancy(pseudo):E3}");
Console.WriteLine();

// ---------------------------------------------------------------- io
Section("11. Scientific file formats");

var scratch = Directory.CreateTempSubdirectory("scinet-sample-").FullName;
var matPath = Path.Combine(scratch, "results.mat");
MatFile.Save(matPath, new Dictionary<string, object?>
{
    ["image"] = image,
    ["label"] = "two shapes",
    ["meta"] = new Dictionary<string, object?> { ["count"] = (double)count, ["sigma"] = 2.0 },
}, doCompression: true);
var loaded = MatFile.Load(matPath);
Console.WriteLine($"  MATLAB v5 (compressed) round trip: {string.Join(", ", MatFile.Whosmat(matPath).Select(v => $"{v.Name} {v.ClassName}"))}");
Console.WriteLine($"    image survives: {((MatArray)loaded["image"]).Real.ToArray().SequenceEqual(image.ToArray())}");

var mtxPath = Path.Combine(scratch, "poisson.mtx");
// SciPy (and SciNet) only auto-detect symmetry below 100 rows, so say it explicitly.
MatrixMarket.MmWrite(mtxPath, poisson, comment: "2-D Poisson", symmetry: "symmetric");
var info = MatrixMarket.MmInfo(mtxPath);
Console.WriteLine($"  Matrix Market: {info.Rows}x{info.Columns}, {info.Entries:N0} stored entries, symmetry '{info.Symmetry}'");

var wavPath = Path.Combine(scratch, "chirp.wav");
var chirpTime = Enumerable.Range(0, 8000).Select(i => i / 8000.0).ToArray();
var chirp = Signal.Chirp(chirpTime, 100, 1.0, 2000);
WavFile.Write(wavPath, 8000, new NdArray(chirp.Select(v => v * 0.8).ToArray(), chirp.Length), WavSampleFormat.Int16, normalized: true);
var wav = WavFile.Read(wavPath, normalize: true);
Console.WriteLine($"  WAV: {wav.SampleRate} Hz, {wav.BitsPerSample}-bit, {wav.Data.Size} samples");
Directory.Delete(scratch, recursive: true);
Console.WriteLine();

Console.WriteLine(SciNetInfo.Attribution);
return;

static string[] Tab10Colors() =>
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

static double MaxAbs(NdArray x) => x.ToArray().Max(Math.Abs);

void Save(ScottPlot.Plot plot, string name, int width, int height)
{
    var path = Path.Combine(screenshots, name);
    plot.SavePng(path, width, height);
    Console.WriteLine($"  saved {path}");
}

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string? Resolve(string relative)
{
    // The build output first, then the working directory, so the sample also runs from a copy.
    foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
    {
        var directory = new DirectoryInfo(start);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, relative);
            if (Directory.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
    }
    return null;
}

static string ResolveScreenshots()
{
    var found = Resolve(Path.Combine("docs", "screenshots"));
    if (found is not null) return found;
    var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
    Directory.CreateDirectory(fallback);
    return fallback;
}
