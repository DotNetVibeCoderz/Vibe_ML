using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn;

/// <summary>A labelled dataset ready for modelling.</summary>
/// <param name="Features">Samples by features.</param>
/// <param name="Target">One target per sample.</param>
/// <param name="FeatureNames">Column names, aligned with <paramref name="Features"/>.</param>
/// <param name="TargetNames">Human-readable class names, indexed by class code.</param>
public sealed record Dataset(
    NdArray Features,
    NdArray Target,
    IReadOnlyList<string> FeatureNames,
    IReadOnlyList<string> TargetNames)
{
    /// <summary>Number of samples.</summary>
    public int SampleCount => Features.Shape[0];

    /// <summary>Number of features.</summary>
    public int FeatureCount => Features.Shape[1];

    /// <summary>Class code to name, for readable reports.</summary>
    public IReadOnlyDictionary<double, string> LabelNames =>
        TargetNames.Select((n, i) => ((double)i, n)).ToDictionary(t => t.Item1, t => t.n);

    /// <summary>The dataset as a frame, with the target appended as a column.</summary>
    public DataFrame ToDataFrame(string targetColumn = "target")
    {
        var frame = DataFrame.FromMatrix(Features, FeatureNames);
        return frame.WithColumn(new NumericSeries(targetColumn, Target.ToArray()));
    }
}

/// <summary>
/// Loads the sample datasets shipped in <c>datasets/</c>, and generates synthetic ones.
/// </summary>
/// <remarks>
/// File loading walks up from the running assembly looking for the repository's <c>datasets</c>
/// directory, so samples, notebooks and tests all find the data without hard-coded paths or a
/// copy step. The generators exist so tests and benchmarks can run with no files at all.
/// </remarks>
public static class Datasets
{
    /// <summary>Locates the repository's <c>datasets</c> directory, or <c>null</c> when not found.</summary>
    public static string? FindDatasetDirectory(string? start = null)
    {
        var directory = new DirectoryInfo(start ?? AppContext.BaseDirectory);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, "datasets");
            if (Directory.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
        return null;
    }

    /// <summary>Resolves a file inside the datasets directory.</summary>
    public static string ResolvePath(string fileName)
    {
        var directory = FindDatasetDirectory()
            ?? throw new DirectoryNotFoundException(
                "Could not locate the 'datasets' directory. Run from inside the repository or pass an explicit path.");
        var path = Path.Combine(directory, fileName);
        if (!File.Exists(path)) throw new FileNotFoundException($"Dataset '{fileName}' not found.", path);
        return path;
    }

    /// <summary>The Iris flower dataset: 150 samples, four features, three species.</summary>
    public static Dataset LoadIris(string? path = null)
    {
        var frame = DataFrame.ReadCsv(path ?? ResolvePath("iris.csv"));
        var featureNames = new[] { "sepal_length", "sepal_width", "petal_length", "petal_width" };
        var features = frame.ToNdArray(featureNames);

        var (codes, categories) = frame.Text("species").Factorize();
        return new Dataset(features, codes.ToNdArray(), featureNames, categories);
    }

    /// <summary>The Titanic survival dataset, numerically encoded and imputed.</summary>
    public static Dataset LoadTitanic(string? path = null)
    {
        var frame = DataFrame.ReadCsv(path ?? ResolvePath("titanic.csv"));

        var (sexCodes, _) = frame.Text("sex").Factorize();
        var (embarkedCodes, _) = frame.Text("embarked").Factorize();

        var prepared = frame
            .WithColumn(sexCodes.Rename("sex_code"))
            .WithColumn(embarkedCodes.Rename("embarked_code"))
            .WithColumn(frame.Numeric("age").FillMissingWithMedian().Rename("age_filled"))
            .WithColumn(frame.Numeric("fare").FillMissingWithMedian().Rename("fare_filled"));

        var featureNames = new[] { "pclass", "sex_code", "age_filled", "sibsp", "parch", "fare_filled", "embarked_code" };
        return new Dataset(
            prepared.ToNdArray(featureNames),
            prepared.Numeric("survived").ToNdArray(),
            featureNames,
            ["died", "survived"]);
    }

    /// <summary>The MNIST digit subset: 8x8 grayscale images flattened to 64 features.</summary>
    public static Dataset LoadDigits(string? path = null)
    {
        var directory = path ?? Path.Combine(
            FindDatasetDirectory() ?? throw new DirectoryNotFoundException("datasets directory not found."),
            "mnist_subset");
        var file = Path.Combine(directory, "digits.csv");
        if (!File.Exists(file)) throw new FileNotFoundException("MNIST subset not found.", file);

        var frame = DataFrame.ReadCsv(file);
        var featureNames = Enumerable.Range(0, 64).Select(i => $"pixel{i}").ToArray();
        return new Dataset(
            frame.ToNdArray(featureNames),
            frame.Numeric("label").ToNdArray(),
            featureNames,
            Enumerable.Range(0, 10).Select(i => i.ToString()).ToArray());
    }

    /// <summary>
    /// Generates linearly separable Gaussian blobs, the standard sanity check for a classifier.
    /// </summary>
    public static Dataset MakeBlobs(int samples = 300, int features = 2, int centers = 3,
        double spread = 1.0, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var centroids = rng.Uniform(-10.0, 10.0, centers, features);

        var x = NdArray.Zeros(samples, features);
        var y = NdArray.Zeros(samples);

        for (var i = 0; i < samples; i++)
        {
            var c = i % centers;
            y.SetAt(i, c);
            for (var j = 0; j < features; j++)
                x[i, j] = centroids[c, j] + rng.Normal(0, spread);
        }

        return new Dataset(x, y,
            Enumerable.Range(0, features).Select(j => $"x{j}").ToArray(),
            Enumerable.Range(0, centers).Select(c => $"cluster{c}").ToArray());
    }

    /// <summary>Generates a regression problem with a known linear signal plus noise.</summary>
    public static (NdArray X, NdArray Y, NdArray TrueCoefficients) MakeRegression(
        int samples = 200, int features = 5, double noise = 0.5, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var x = rng.StandardNormal(samples, features);
        var coefficients = rng.Uniform(-3.0, 3.0, features);

        var y = NdArray.Zeros(samples);
        for (var i = 0; i < samples; i++)
        {
            var acc = 0.0;
            for (var j = 0; j < features; j++) acc += x[i, j] * coefficients.At(j);
            y.SetAt(i, acc + rng.Normal(0, noise));
        }
        return (x, y, coefficients);
    }

    /// <summary>
    /// Generates two interleaved half-circles - not linearly separable, which is what makes it
    /// useful for showing where a linear model fails and a tree or kNN does not.
    /// </summary>
    public static Dataset MakeMoons(int samples = 200, double noise = 0.1, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(samples, 2);
        var y = NdArray.Zeros(samples);
        var half = samples / 2;

        for (var i = 0; i < samples; i++)
        {
            var angle = Math.PI * (i % half) / (half - 1);
            if (i < half)
            {
                x[i, 0] = Math.Cos(angle) + rng.Normal(0, noise);
                x[i, 1] = Math.Sin(angle) + rng.Normal(0, noise);
                y.SetAt(i, 0);
            }
            else
            {
                x[i, 0] = 1 - Math.Cos(angle) + rng.Normal(0, noise);
                x[i, 1] = 0.5 - Math.Sin(angle) + rng.Normal(0, noise);
                y.SetAt(i, 1);
            }
        }
        return new Dataset(x, y, ["x0", "x1"], ["moon0", "moon1"]);
    }
}
