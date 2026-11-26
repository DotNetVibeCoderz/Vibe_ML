using System.Collections.Concurrent;
using System.Security.Cryptography;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Gravicode.Science.DaskNet.ML;

/// <summary>
/// Batch scoring of partitioned frames with an ONNX model, one ONNX Runtime session per worker
/// process: a model trained anywhere - scikit-learn through skl2onnx, PyTorch, GraviLearn's
/// <c>OnnxExport</c> - scores a frame larger than memory, on a cluster too.
/// </summary>
/// <remarks>
/// The model's bytes are one value in the graph that every partition's task reads, so each
/// worker fetches them once. Sessions are cached per process by the model's SHA-256 (the last
/// four), so a worker loads a model once however many partitions or calls score with it.
/// </remarks>
public static class OnnxScoring
{
    /// <summary>
    /// Adds the model's output to each partition. The model's first input is fed
    /// <paramref name="features"/> as a rows x features tensor (float or double, whichever the
    /// model declares). The output named <paramref name="output"/> (the first by default) becomes
    /// one column per output column: named <paramref name="columns"/> if given, otherwise the
    /// output's name, or <c>{name}_{i}</c> when it is wider than one.
    /// </summary>
    public static DaskFrame ScoreOnnx(this DaskFrame frame, byte[] model, string[] features, string? output = null, string[]? columns = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(features);
        if (features.Length == 0) throw new ArgumentException("Name at least one feature column.", nameof(features));
        foreach (var f in features)
            if (!frame.Columns.Contains(f)) throw new ArgumentException($"No column '{f}'. Columns: {string.Join(", ", frame.Columns)}.", nameof(features));

        var session = OnnxKernels.Session(model);
        var input = session.InputMetadata.First();
        var width = input.Value.Dimensions.Length == 2 ? input.Value.Dimensions[1] : -1;
        if (width > 0 && width != features.Length)
            throw new ArgumentException($"The model's input '{input.Key}' takes {width} features; {features.Length} were named.", nameof(features));
        var outputName = output ?? session.OutputMetadata.Keys.First();
        if (!session.OutputMetadata.TryGetValue(outputName, out var meta))
            throw new ArgumentException($"The model has no output '{outputName}'. Outputs: {string.Join(", ", session.OutputMetadata.Keys)}.", nameof(output));
        var outWidth = meta.Dimensions.Length >= 2 ? meta.Dimensions[1] : 1;
        string[] names;
        if (columns is not null)
        {
            if (outWidth > 0 && columns.Length != outWidth)
                throw new ArgumentException($"Output '{outputName}' has {outWidth} columns; {columns.Length} names were given.", nameof(columns));
            names = columns;
        }
        else if (outWidth == 1) names = [outputName];
        else if (outWidth > 1) names = [.. Enumerable.Range(0, outWidth).Select(i => $"{outputName}_{i}")];
        else throw new ArgumentException($"Output '{outputName}' has a symbolic width; name its columns.", nameof(columns));

        var bytes = Dask.Value(model, "onnx-model");
        var spec = Dask.Value<object?[]>([features, outputName, names], "onnx-spec");
        var parts = frame.ToDelayed().Select(p => Dask.Delayed(OnnxKernels.Score, p, bytes, spec, "onnx-score")).ToArray();
        return DaskFrame.FromDelayed(parts, [.. frame.Columns.Where(c => !names.Contains(c)), .. names]);
    }

    /// <summary>Reads the model from <paramref name="path"/> and scores with it.</summary>
    public static DaskFrame ScoreOnnx(this DaskFrame frame, string path, string[] features, string? output = null, string[]? columns = null)
        => frame.ScoreOnnx(File.ReadAllBytes(path), features, output, columns);
}

/// <summary>The remotable scoring step.</summary>
public static class OnnxKernels
{
    private const int CacheSize = 4;
    private static readonly ConcurrentDictionary<string, InferenceSession> Sessions = new();
    private static readonly Queue<string> Order = new();

    internal static InferenceSession Session(byte[] model)
    {
        var id = Convert.ToHexString(SHA256.HashData(model));
        if (Sessions.TryGetValue(id, out var cached)) return cached;
        lock (Order)
        {
            if (Sessions.TryGetValue(id, out cached)) return cached;
            var session = new InferenceSession(model);
            Sessions[id] = session;
            Order.Enqueue(id);
            // Evicted sessions are not disposed: another thread may be scoring with one right now,
            // and a session holds no resource that the finaliser does not release.
            while (Order.Count > CacheSize) Sessions.TryRemove(Order.Dequeue(), out _);
            return session;
        }
    }

    /// <summary>Scores one partition.</summary>
    [RemoteTask]
    public static DataFrame Score(DataFrame partition, byte[] model, object?[] spec)
    {
        var features = (string[])spec[0]!;
        var outputName = (string)spec[1]!;
        var names = (string[])spec[2]!;
        var kept = partition.ColumnNames.Where(c => !names.Contains(c)).Select(c => partition[c]).ToList();
        var n = partition.RowCount;
        if (n == 0)
        {
            kept.AddRange(names.Select(c => new NumericSeries(c, [])));
            return new DataFrame(kept);
        }

        var session = Session(model);
        var input = session.InputMetadata.First();
        var d = features.Length;
        var columns = features.Select(f => partition.Numeric(f).Values).ToArray();
        NamedOnnxValue value;
        if (input.Value.ElementType == typeof(double))
        {
            var t = new DenseTensor<double>([n, d]);
            for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) t[i, j] = columns[j][i];
            value = NamedOnnxValue.CreateFromTensor(input.Key, t);
        }
        else
        {
            var t = new DenseTensor<float>([n, d]);
            for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) t[i, j] = (float)columns[j][i];
            value = NamedOnnxValue.CreateFromTensor(input.Key, t);
        }

        using var results = session.Run([value], [outputName]);
        var result = results.First();
        var values = ToDoubles(result);
        var width = names.Length;
        if (values.Length != n * width)
            throw new InvalidOperationException($"Output '{outputName}' gave {values.Length} values for {n} rows x {width} columns.");
        for (var k = 0; k < width; k++)
        {
            var col = new double[n];
            for (var i = 0; i < n; i++) col[i] = values[i * width + k];
            kept.Add(new NumericSeries(names[k], col));
        }
        return new DataFrame(kept);
    }

    private static double[] ToDoubles(DisposableNamedOnnxValue value) => value.Value switch
    {
        Tensor<float> t => [.. t.Select(v => (double)v)],
        Tensor<double> t => [.. t],
        Tensor<long> t => [.. t.Select(v => (double)v)],
        Tensor<int> t => [.. t.Select(v => (double)v)],
        Tensor<bool> t => [.. t.Select(v => v ? 1.0 : 0.0)],
        _ => throw new NotSupportedException($"Output '{value.Name}' is a {value.Value?.GetType().Name}; numeric tensors are supported (not maps or sequences)."),
    };
}
