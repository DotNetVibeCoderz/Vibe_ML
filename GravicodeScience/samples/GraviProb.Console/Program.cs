using System.Diagnostics;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.GraviProb.Models;

Console.WriteLine(GraviInfo.Banner("GraviProb"));

var datasets = Resolve("datasets") ?? throw new DirectoryNotFoundException("datasets directory not found.");
var screenshots = ResolveScreenshots();
var watch = new Stopwatch();

// ---------------------------------------------------------------- data
Section("1. The coin data");

var flips = DataFrame.ReadCsv(Path.Combine(datasets, "bayesian_coin.csv"));
var outcomes = flips.Numeric("outcome");
var trials = flips.RowCount;
var heads = (int)outcomes.Sum();

Console.WriteLine($"  {trials} flips, {heads} heads, {trials - heads} tails");
Console.WriteLine($"  observed proportion: {(double)heads / trials:F4}");
Console.WriteLine();
Console.WriteLine("  Heads per session:");
Console.WriteLine(flips.GroupBy("session").Sum("outcome").ToString());
Console.WriteLine();

// ---------------------------------------------------------------- model
Section("2. The model");

var model = new BayesianModel()
    .AddDistribution("theta", Distribution.Beta(1, 1))
    .AddObservation("data", DistributionSpec.Binomial(trials, "theta"), heads);

Console.Write(model);
Console.WriteLine("  Beta(1,1) is the uniform prior: before seeing data, every bias is equally plausible.");
Console.WriteLine();

// The Beta prior is conjugate to the binomial likelihood, so the posterior is exact.
var exact = Distribution.Beta(1, 1).PosteriorAfter(heads, trials - heads);
Console.WriteLine($"  Conjugate posterior: {exact.Name}");
Console.WriteLine($"    mean {exact.Mean:F6}, sd {exact.StandardDeviation:F6}");
Console.WriteLine();

// ---------------------------------------------------------------- mcmc
Section("3. Metropolis-Hastings");

watch.Restart();
var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, warmup: 10_000, seed: 42);
watch.Stop();

Console.WriteLine($"  sampled in {watch.ElapsedMilliseconds} ms");
Console.Write(posterior.Summary());
Console.WriteLine();
Console.WriteLine($"  exact mean {exact.Mean:F6} vs sampled {posterior.Mean("theta"):F6}  " +
                  $"(error {Math.Abs(exact.Mean - posterior.Mean("theta")):E2})");

var (low, high) = posterior.HighestDensityInterval("theta", 0.95);
Console.WriteLine($"  95% HDI: [{low:F4}, {high:F4}]");
Console.WriteLine($"  P(theta > 0.5) = {posterior["theta"].ToArray().Count(v => v > 0.5) / (double)posterior.TotalDraws:P2}");
Console.WriteLine();

Section("4. Metropolis within Gibbs");

watch.Restart();
var gibbs = model.SampleGibbs(iterations: 20_000, chains: 4, warmup: 10_000, seed: 42);
watch.Stop();
Console.WriteLine($"  sampled in {watch.ElapsedMilliseconds} ms, acceptance {gibbs.AcceptanceRate:P1}");
Console.WriteLine($"  mean {gibbs.Mean("theta"):F6}, r_hat {gibbs.RHat("theta"):F4}, ESS {gibbs.EffectiveSampleSize("theta"):F0}");
Console.WriteLine();

Section("5. Hamiltonian Monte Carlo and NUTS");

Console.WriteLine($"  model is differentiable: {model.IsDifferentiable}");

// One tape pass gives the gradient of the whole log posterior. At the mode of the Beta(126, 76)
// posterior it must vanish, which is a check the sampler itself cannot give you.
var mode = (exact.Alpha - 1) / (exact.Alpha + exact.BetaParameter - 2);
var (_, gradientAtMode) = model.LogPosteriorGradient(
    new Dictionary<string, double>(StringComparer.Ordinal) { ["theta"] = mode });
Console.WriteLine($"  d/dtheta at the posterior mode {mode:F6} = {gradientAtMode[0]:E2}  (exactly 0)");
Console.WriteLine();

watch.Restart();
var nuts = model.SampleNUTS(iterations: 2000, chains: 4, seed: 42);
watch.Stop();

Console.WriteLine($"  NUTS: {watch.ElapsedMilliseconds} ms, mean {nuts.Mean("theta"):F6}, " +
                  $"r_hat {nuts.RHat("theta"):F4}");
Console.WriteLine($"  error against the exact posterior: {Math.Abs(exact.Mean - nuts.Mean("theta")):E2}");
Console.WriteLine($"  effective draws: {nuts.EffectiveSampleSize("theta"):F0} of {nuts.TotalDraws} " +
                  $"({nuts.EffectiveSampleSize("theta") / nuts.TotalDraws:P1})");
Console.WriteLine($"  versus the random walk above: " +
                  $"{posterior.EffectiveSampleSize("theta") / posterior.TotalDraws:P1} of its draws were effective");
Console.WriteLine();
Console.WriteLine("  On one parameter the random walk is the faster choice per effective sample.");
Console.WriteLine("  NUTS earns its cost as the dimension grows, where a random walk stops");
Console.WriteLine("  converging at all rather than merely slowing down.");
Console.WriteLine();

Section("6. Variational inference");

watch.Restart();
var variational = model.FitVariational(iterations: 1200, learningRate: 0.05, seed: 42);
watch.Stop();
Console.WriteLine($"  fitted in {watch.ElapsedMilliseconds} ms");
Console.Write(variational.Summary());
Console.WriteLine("  Mean-field VI is much faster than MCMC but understates the spread; it approximates");
Console.WriteLine("  the posterior rather than sampling from it.");
Console.WriteLine();

// ---------------------------------------------------------------- predictive
Section("7. Posterior predictive check");

var replicated = posterior.PosteriorPredictive(
    (values, rng) => rng.Binomial(trials, values["theta"]), draws: 5000, seed: 42);

Console.WriteLine($"  replicated head counts: mean {Statistics.Mean(replicated):F2}, sd {Statistics.Std(replicated):F2}");
Console.WriteLine($"  observed {heads} sits at percentile " +
                  $"{replicated.ToArray().Count(v => v < heads) / 5000.0:P1} of the replications");
Console.WriteLine("  (a value near the middle means the model can plausibly produce the data it was fitted to)");
Console.WriteLine();

// ---------------------------------------------------------------- multi parameter
Section("8. A two-parameter model");

var rng = new GraviRandom(7);
var measurements = Enumerable.Range(0, 250).Select(_ => rng.Normal(12.5, 3.2)).ToArray();

var normalModel = new BayesianModel()
    .AddDistribution("mu", Distribution.Normal(0, 20))
    .AddDistribution("sigma", Distribution.HalfNormal(10))
    .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), measurements);

watch.Restart();
var normalPosterior = normalModel.SampleMCMC(iterations: 20_000, chains: 4, seed: 7);
watch.Stop();

var sample = NdArray.FromValues(measurements);
Console.WriteLine($"  250 measurements: sample mean {Statistics.Mean(sample):F4}, sample sd {Statistics.Std(sample):F4}");
Console.WriteLine($"  sampled in {watch.ElapsedMilliseconds} ms");
Console.Write(normalPosterior.Summary());
Console.WriteLine();

// ---------------------------------------------------------------- bayesian network
Section("9. A Bayesian network");

var network = new BayesianNetwork()
    .AddVariable("rain", 0.8, 0.2)
    .AddVariable("sprinkler", 2, ["rain"], [[0.6, 0.4], [0.99, 0.01]])
    .AddVariable("wet", 2, ["rain", "sprinkler"],
    [
        [1.00, 0.00],
        [0.10, 0.90],
        [0.20, 0.80],
        [0.01, 0.99],
    ]);

Console.Write(network);
Console.WriteLine($"  P(rain)                    = {network.Infer("rain")[1]:F4}");
Console.WriteLine($"  P(wet)                     = {network.Infer("wet")[1]:F4}");
Console.WriteLine($"  P(rain | wet)              = {network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1 })[1]:F4}");
Console.WriteLine($"  P(rain | wet, sprinkler)   = {network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1, ["sprinkler"] = 1 })[1]:F4}");
Console.WriteLine("  The last two show explaining away: once the sprinkler accounts for the wet");
Console.WriteLine("  grass, rain becomes a less necessary explanation.");
Console.WriteLine();

// ---------------------------------------------------------------- hmm
Section("10. A hidden Markov model");

var weather = new HiddenMarkovModel(
    [0.6, 0.4],
    new[,] { { 0.7, 0.3 }, { 0.4, 0.6 } },
    new[,] { { 0.1, 0.4, 0.5 }, { 0.6, 0.3, 0.1 } });

var (trueStates, observations) = weather.Generate(400, new GraviRandom(11));
Console.WriteLine($"  generated 400 steps; log-likelihood under the true model: {weather.LogLikelihood(observations):F2}");

var decoded = weather.Viterbi(observations);
var agreement = decoded.Where((s, i) => s == trueStates[i]).Count() / 400.0;
Console.WriteLine($"  Viterbi recovered {agreement:P1} of the hidden states");

var sequences = Enumerable.Range(0, 25)
    .Select(_ => (IReadOnlyList<int>)weather.Generate(120, new GraviRandom(100 + _)).Observations)
    .ToList();

var learned = HiddenMarkovModel.Random(2, 3, seed: 13);
var before = sequences.Sum(learned.LogLikelihood);
watch.Restart();
var after = learned.Fit(sequences, iterations: 60);
watch.Stop();

Console.WriteLine($"  Baum-Welch in {watch.ElapsedMilliseconds} ms: log-likelihood {before:F1} -> {after:F1}");
Console.WriteLine($"  (the generating model scores {sequences.Sum(weather.LogLikelihood):F1} on the same data)");
Console.WriteLine();

// ---------------------------------------------------------------- regression
Section("11. Bayesian linear regression");

var x = new GraviRandom(17).StandardNormal(120, 1);
var y = NdArray.Zeros(120);
for (var i = 0; i < 120; i++) y.SetAt(i, 2.0 + 3.5 * x[i, 0] + new GraviRandom(1000 + i).Normal(0, 0.8));

var regression = new BayesianLinearRegression(priorPrecision: 1e-3, noisePrecision: 1.5).Fit(x, y);
Console.WriteLine($"  intercept posterior mean: {regression.CoefficientMeans.At(0):F4} (true 2.0)");
Console.WriteLine($"  slope     posterior mean: {regression.CoefficientMeans.At(1):F4} (true 3.5)");

var query = NdArray.FromArray(new double[,] { { -2.0 }, { 0.0 }, { 2.0 }, { 6.0 } });
var (mean, deviation) = regression.PredictWithUncertainty(query);
Console.WriteLine($"  {"x",8}{"prediction",14}{"uncertainty",14}");
for (var i = 0; i < query.Shape[0]; i++)
    Console.WriteLine($"  {query[i, 0],8:F1}{mean.At(i),14:F4}{deviation.At(i),14:F4}");
Console.WriteLine("  Uncertainty grows outside the observed range, which a point estimate cannot show.");
Console.WriteLine();

// ---------------------------------------------------------------- chart
Section("12. Posterior plot");

var draws = posterior["theta"].ToArray();
var (edges, counts) = Statistics.Histogram(posterior["theta"], bins: 60);
var centres = new double[counts.Length];
var density = new double[counts.Length];
var width = edges[1] - edges[0];
for (var i = 0; i < counts.Length; i++)
{
    centres[i] = (edges[i] + edges[i + 1]) / 2;
    density[i] = counts[i] / (draws.Length * width);
}

var curveX = NdArray.Linspace(0.01, 0.99, 400).ToArray();
var curveY = curveX.Select(exact.Density).ToArray();

var plot = new ScottPlot.Plot();
var bars = plot.Add.Bars(centres, density);
bars.LegendText = "MCMC draws";
foreach (var bar in bars.Bars) bar.Size = width * 0.9;

var curve = plot.Add.Scatter(curveX, curveY);
curve.LegendText = "exact Beta posterior";
curve.MarkerSize = 0;
curve.LineWidth = 3;

var hdi = plot.Add.HorizontalSpan(low, high);
hdi.LegendText = "95% HDI";
hdi.FillColor = ScottPlot.Colors.Orange.WithAlpha(0.15);

plot.Title($"GraviProb - posterior for a coin: {heads} heads in {trials} flips");
plot.XLabel("theta (probability of heads)");
plot.YLabel("density");
plot.ShowLegend();
plot.SavePng(Path.Combine(screenshots, "graviprob_posterior.png"), 1000, 650);
Console.WriteLine($"  saved {Path.Combine(screenshots, "graviprob_posterior.png")}");

// ---------------------------------------------------------------- v0.4: multivariate
Section("10. Multivariate distributions");

var mvnMean = NdArray.FromValues([1.0, -2.0]);
var mvnCov = NdArray.FromArray(new double[,] { { 4.0, 1.5 }, { 1.5, 2.0 } });
var mvn = new MultivariateNormal(mvnMean, mvnCov);

Console.WriteLine($"  MultivariateNormal, correlated: rho = {mvnCov[0, 1] / Math.Sqrt(mvnCov[0, 0] * mvnCov[1, 1]):F3}");
Console.WriteLine("  Everything runs off one Cholesky factor: density, log determinant and sampling.");

var conditioned = mvn.Conditional([0], [1], NdArray.FromValues([1.0]));
Console.WriteLine($"    unconditional: mean {mvnMean.At(0):F4}, variance {mvnCov[0, 0]:F4}");
Console.WriteLine($"    given x1 = 1:  mean {conditioned.Mean.At(0):F4}, variance {conditioned.Covariance[0, 0]:F4}");
Console.WriteLine("    conditioning a normal on part of itself gives another normal, in closed form -");
Console.WriteLine("    which is the entire mechanism behind Gaussian process regression");

var dirichlet = new Dirichlet(2, 3, 5);
Console.WriteLine($"  Dirichlet(2, 3, 5): mean = [{string.Join(", ", dirichlet.Mean.ToArray().Select(v => v.ToString("F3")))}]");
Console.WriteLine($"    every marginal is a Beta: component 0 ~ Beta({dirichlet.Marginal(0).Alpha:F0}, {dirichlet.Marginal(0).BetaParameter:F0})");
Console.WriteLine($"    conjugate, so observing counts [10, 5, 0] is addition:");
Console.WriteLine($"      posterior alpha = [{string.Join(", ", dirichlet.Posterior([10.0, 5.0, 0.0]).Alpha)}]");
Console.WriteLine("    off-diagonal covariance is negative and must be - the components sum to one,");
Console.WriteLine("    so a Dirichlet cannot express positively correlated proportions at all");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: Gaussian process
Section("11. Gaussian processes");

const int observed = 12;
var gpX = NdArray.Zeros(observed, 1);
var gpY = NdArray.Zeros(observed);
for (var i = 0; i < observed; i++)
{
    var value = i * 2 * Math.PI / observed;
    gpX[i, 0] = value;
    gpY.SetAt(i, Math.Sin(value));
}

var gp = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1e-6).Fit(gpX, gpY);

Console.WriteLine($"  Fitted to {observed} points of a sine, with no optimisation - the posterior is");
Console.WriteLine("  a closed-form conditional, and the uncertainty arrives with the prediction.");
Console.WriteLine($"    log marginal likelihood = {gp.LogMarginalLikelihood():F4}");

foreach (var probe in new[] { 1.0, 3.0, 20.0 })
{
    var point = NdArray.Zeros(1, 1);
    point[0, 0] = probe;
    var prediction = gp.Predict(point);
    Console.WriteLine($"    x = {probe,5:F1}  mean {prediction.Mean.At(0),8:F4}  " +
                      $"sd {prediction.StandardDeviation.At(0),7:F4}  (true sin = {Math.Sin(probe):F4})");
}
Console.WriteLine("    far from the data the uncertainty returns to the prior - honest, and the");
Console.WriteLine("    main reason to use a GP over a point-estimate regressor");

var tuned = GaussianProcess.Optimise(gpX, gpY);
Console.WriteLine($"  Grid search over length scale and noise: {tuned.Kernel.Name}");
Console.WriteLine("    a grid, not a gradient method - the marginal likelihood is not concave and has");
Console.WriteLine("    real local optima, one explaining the data as signal and another as noise");

// The band and a few coherent posterior draws.
var gpPath = Path.Combine(screenshots, "graviprob_gaussian_process.png");
var gpPlot = new ScottPlot.Plot();

const int gridSize = 200;
var gridX = NdArray.Zeros(gridSize, 1);
var gridValues = new double[gridSize];
for (var i = 0; i < gridSize; i++)
{
    gridValues[i] = -1 + i * 9.0 / (gridSize - 1);
    gridX[i, 0] = gridValues[i];
}

var band = gp.Predict(gridX);
var (lower, upper) = band.Interval(0.95);

var fill = gpPlot.Add.FillY(gridValues, lower.ToArray(), upper.ToArray());
fill.LegendText = "95% credible interval";
fill.FillColor = ScottPlot.Colors.SteelBlue.WithAlpha(0.25);

var meanLine = gpPlot.Add.Scatter(gridValues, band.Mean.ToArray());
meanLine.LegendText = "posterior mean";
meanLine.MarkerSize = 0;

var gpDraws = gp.SamplePosterior(gridX, count: 3, new GraviRandom(5));
for (var s = 0; s < 3; s++)
{
    var draw = new double[gridSize];
    for (var i = 0; i < gridSize; i++) draw[i] = gpDraws[s, i];

    var line = gpPlot.Add.Scatter(gridValues, draw);
    line.MarkerSize = 0;
    line.LineWidth = 1;
    line.LinePattern = ScottPlot.LinePattern.Dotted;
    if (s == 0) line.LegendText = "posterior draws";
}

var points = gpPlot.Add.Scatter(
    Enumerable.Range(0, observed).Select(i => gpX[i, 0]).ToArray(), gpY.ToArray());
points.LineWidth = 0;
points.MarkerSize = 9;
points.LegendText = "observations";

gpPlot.Title("GraviProb - Gaussian process posterior");
gpPlot.XLabel("x");
gpPlot.YLabel("f(x)");
gpPlot.ShowLegend();
gpPlot.SavePng(gpPath, 900, 550);
Console.WriteLine($"  saved {gpPath}");
Console.WriteLine("    the band widens beyond the data; the dotted draws are coherent FUNCTIONS,");
Console.WriteLine("    which a marginal band cannot express - it says nothing about the shape");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: state space
Section("12. Kalman filter and smoother");

var kalman = KalmanFilter.LocalLevel(processVariance: 0.05, observationVariance: 1.0);
var (hiddenLevel, noisy) = kalman.Simulate(200, new GraviRandom(31));

var filtered = kalman.Filter(noisy);
var smoothed = kalman.Smooth(noisy);

double RootMeanSquare(IReadOnlyList<double> estimate)
{
    var total = 0.0;
    for (var t = 0; t < estimate.Count; t++)
    {
        var error = estimate[t] - hiddenLevel[t, 0];
        total += error * error;
    }
    return Math.Sqrt(total / estimate.Count);
}

var rawError = 0.0;
for (var t = 0; t < 200; t++)
{
    var error = noisy[t, 0] - hiddenLevel[t, 0];
    rawError += error * error;
}
rawError = Math.Sqrt(rawError / 200);

Console.WriteLine("  Simulated from the model, then estimated against a truth the filter never saw:");
Console.WriteLine($"    raw observations  RMSE = {rawError:F4}");
Console.WriteLine($"    filtered          RMSE = {RootMeanSquare(filtered.Filtered.Select(s => s.Mean.At(0)).ToList()):F4}");
Console.WriteLine($"    smoothed          RMSE = {RootMeanSquare(smoothed.Select(s => s.Mean.At(0)).ToList()):F4}");
Console.WriteLine("    Filtering uses only the past, which is what a real-time system can do.");
Console.WriteLine("    Smoothing uses the whole series and is strictly better - using smoothed states");
Console.WriteLine("    to evaluate a forecasting rule is a look-ahead error, and a common one.");
Console.WriteLine($"    log likelihood = {filtered.LogLikelihood:F2}, which is what parameter fitting maximises");

var forecast = kalman.Forecast(noisy, horizon: 20);
Console.WriteLine($"  20-step forecast: uncertainty grows from {forecast[0].StandardDeviation.At(0):F4} " +
                  $"to {forecast[^1].StandardDeviation.At(0):F4}");
Console.WriteLine("    no observations arrive, so only the predict step runs and Q accumulates -");
Console.WriteLine("    a forecast whose uncertainty does not grow is not a forecast");

var kalmanPath = Path.Combine(screenshots, "graviprob_kalman.png");
var kalmanPlot = new ScottPlot.Plot();
var times = Enumerable.Range(0, 200).Select(t => (double)t).ToArray();

var observationSeries = kalmanPlot.Add.Scatter(times,
    Enumerable.Range(0, 200).Select(t => noisy[t, 0]).ToArray());
observationSeries.LineWidth = 0;
observationSeries.MarkerSize = 3;
observationSeries.Color = ScottPlot.Colors.Gray.WithAlpha(0.5);
observationSeries.LegendText = "observations";

var truthSeries = kalmanPlot.Add.Scatter(times,
    Enumerable.Range(0, 200).Select(t => hiddenLevel[t, 0]).ToArray());
truthSeries.MarkerSize = 0;
truthSeries.LineWidth = 2;
truthSeries.LegendText = "hidden state (never observed)";

var smoothSeries = kalmanPlot.Add.Scatter(times, smoothed.Select(s => s.Mean.At(0)).ToArray());
smoothSeries.MarkerSize = 0;
smoothSeries.LineWidth = 2;
smoothSeries.LegendText = "smoothed estimate";

kalmanPlot.Title("GraviProb - Kalman smoother recovering a hidden state");
kalmanPlot.XLabel("time");
kalmanPlot.YLabel("level");
kalmanPlot.ShowLegend();
kalmanPlot.SavePng(kalmanPath, 900, 500);
Console.WriteLine($"  saved {kalmanPath}");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: model comparison
Section("13. Model comparison with WAIC and LOO");

var comparisonRng = new GraviRandom(41);
var comparisonData = NdArray.Zeros(60);
for (var i = 0; i < 60; i++) comparisonData.SetAt(i, comparisonRng.Normal());

// A posterior over the mean, with a flat prior.
var sampleMean = 0.0;
for (var i = 0; i < 60; i++) sampleMean += comparisonData.At(i);
sampleMean /= 60;

var drawRng = new GraviRandom(43);
var drawnMeans = new double[1500];
for (var s = 0; s < 1500; s++) drawnMeans[s] = sampleMean + drawRng.Normal() / Math.Sqrt(60);

NdArray PointwiseLogLikelihood(double sigma)
{
    var matrix = NdArray.Zeros(drawnMeans.Length, comparisonData.Size);
    for (var s = 0; s < drawnMeans.Length; s++)
    {
        var normal = new Normal(drawnMeans[s], sigma);
        for (var i = 0; i < comparisonData.Size; i++) matrix[s, i] = normal.LogDensity(comparisonData.At(i));
    }
    return matrix;
}

Console.WriteLine("  Data came from N(0, 1). Three candidate scales, judged on out-of-sample");
Console.WriteLine("  predictive accuracy rather than in-sample fit:");

var candidates = new Dictionary<string, InformationCriterion>();
foreach (var (label, sigma) in new[] { ("sigma = 0.5", 0.5), ("sigma = 1.0", 1.0), ("sigma = 4.0", 4.0) })
{
    var matrix = PointwiseLogLikelihood(sigma);
    var waic = ModelComparison.Waic(matrix);
    var loo = ModelComparison.Loo(matrix);

    candidates[label] = waic;
    Console.WriteLine($"    {label,-12} WAIC {waic.Estimate,8:F2}  LOO {loo.Criterion.Estimate,8:F2}  " +
                      $"p_eff {waic.EffectiveParameters,5:F2}  reliable {loo.IsReliable}");
}

Console.WriteLine("  Ranked, with the standard error of each DIFFERENCE from the best:");
foreach (var (name, estimate, difference, error) in ModelComparison.Compare(candidates))
    Console.WriteLine($"    {name,-12} {estimate,8:F2}  {(difference == 0 ? "best" : $"+{difference:F2} +/- {error:F2}")}");

Console.WriteLine("  Lower is better, and only differences mean anything - the absolute value is not");
Console.WriteLine("  interpretable. The error of a difference uses the PAIRED pointwise terms, because");
Console.WriteLine("  the models are scored on the same observations and their errors are correlated.");
Console.WriteLine("  LOO's Pareto-k diagnostic says whether its reweighting can be trusted, and here it");
Console.WriteLine("  fired: sigma = 0.5 reports 'reliable False'. That model is badly misspecified, so a");
Console.WriteLine("  few observations dominate the importance weights and its LOO estimate should not be");
Console.WriteLine("  believed. WAIC has no equivalent - it fails silently in exactly the cases LOO flags.");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string? Resolve(string relative)
{
    var directory = new DirectoryInfo(AppContext.BaseDirectory);
    for (var depth = 0; directory is not null && depth < 12; depth++)
    {
        var candidate = Path.Combine(directory.FullName, relative);
        if (Directory.Exists(candidate)) return candidate;
        directory = directory.Parent;
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

