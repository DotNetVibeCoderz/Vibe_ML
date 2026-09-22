using Gravicode.HFNet.GraviHub.Io;
using Microsoft.ML.OnnxRuntime;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviOptimum;

/// <summary>What an optimisation produced.</summary>
/// <param name="Session">The session to run inference through.</param>
/// <param name="Target">The provider it ended up on.</param>
/// <param name="Source">The file it was built from.</param>
public sealed record OptimizedModel(OnnxSession Session, ExecutionTarget Target, string Source) : IDisposable
{
    /// <summary>Runs the model.</summary>
    public Dictionary<string, NdArray> Run(IReadOnlyDictionary<string, NdArray> inputs) => Session.Run(inputs);

    /// <summary>Times the model on a fixed input.</summary>
    public LatencyReport Measure(IReadOnlyDictionary<string, NdArray> inputs, int iterations = 20)
        => Session.Measure(inputs, iterations);

    /// <inheritdoc />
    public void Dispose() => Session.Dispose();

    /// <inheritdoc />
    public override string ToString() => $"OptimizedModel({Path.GetFileName(Source)} on {Target})";
}

/// <summary>How aggressively to shrink a checkpoint's weights.</summary>
public enum QuantizationLevel
{
    /// <summary>Leave them as they are.</summary>
    None,

    /// <summary>Store as bfloat16: half the bytes, and the same exponent range as float32.</summary>
    BFloat16,

    /// <summary>Store as IEEE half: half the bytes, and a much smaller exponent range.</summary>
    Float16,

    /// <summary>Per-tensor symmetric int8, with the scale recorded alongside.</summary>
    Int8,
}

/// <summary>What quantising a checkpoint cost and saved.</summary>
/// <param name="OriginalBytes">Size of the source file.</param>
/// <param name="QuantizedBytes">Size of the written file.</param>
/// <param name="MaxAbsoluteError">Largest absolute deviation introduced, across every tensor.</param>
/// <param name="MeanAbsoluteError">Mean absolute deviation introduced.</param>
public readonly record struct QuantizationReport(
    long OriginalBytes, long QuantizedBytes, double MaxAbsoluteError, double MeanAbsoluteError)
{
    /// <summary>Fraction of the original size the result occupies.</summary>
    public double SizeRatio => OriginalBytes == 0 ? 1 : (double)QuantizedBytes / OriginalBytes;

    /// <inheritdoc />
    public override string ToString()
        => $"{OriginalBytes / (1024.0 * 1024):F1} MB -> {QuantizedBytes / (1024.0 * 1024):F1} MB "
            + $"({SizeRatio:P0}), max error {MaxAbsoluteError:E2}, mean {MeanAbsoluteError:E2}";
}

/// <summary>
/// The blueprint's entry point: <c>Optimum.Optimize(model, target: "CUDA")</c>.
/// </summary>
/// <remarks>
/// Optimisation here means two separable things, and they are kept separable: choosing the
/// execution provider a graph runs on, and shrinking the weights it carries. The first is free and
/// lossless; the second always costs accuracy, so <see cref="Quantize"/> measures and reports the
/// error it introduced rather than leaving the caller to assume it was negligible.
/// </remarks>
public static class Optimum
{
    /// <summary>Loads a Hub model's ONNX export onto the best available provider.</summary>
    /// <param name="repoId">A model id.</param>
    /// <param name="target">
    /// The provider name: <c>CPU</c>, <c>CUDA</c>, <c>DirectML</c>, <c>oneDNN</c> or <c>auto</c>.
    /// </param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static OptimizedModel Optimize(string repoId, string target = "auto", string revision = "main")
    {
        var parsed = ParseTarget(target);
        var session = OnnxSession.FromPretrained(repoId, fileName: null, parsed, revision);

        return new OptimizedModel(session, session.ActualTarget, session.Path);
    }

    /// <summary>Loads a local ONNX file onto a chosen provider.</summary>
    /// <param name="path">Path to the <c>.onnx</c> file.</param>
    /// <param name="target">The provider name.</param>
    public static OptimizedModel OptimizeFile(string path, string target = "auto")
    {
        var parsed = ParseTarget(target);
        var session = OnnxSession.Open(path, parsed);

        return new OptimizedModel(session, session.ActualTarget, path);
    }

    /// <summary>Turns a provider name into a target, accepting the spellings people actually use.</summary>
    /// <exception cref="ArgumentException">The name matches no provider.</exception>
    public static ExecutionTarget ParseTarget(string target) => target?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" or "best" => ExecutionTarget.Auto,
        "cpu" => ExecutionTarget.Cpu,
        "cuda" or "gpu" or "nvidia" => ExecutionTarget.Cuda,
        "directml" or "dml" => ExecutionTarget.DirectML,
        "onednn" or "dnnl" or "mkl" or "mkldnn" => ExecutionTarget.OneDnn,
        _ => throw new ArgumentException(
            $"'{target}' is not a known execution target. Use CPU, CUDA, DirectML, oneDNN or auto.",
            nameof(target)),
    };

    /// <summary>Lists the providers that can actually be registered in this process.</summary>
    /// <remarks>
    /// Determined by trying each one, because the answer depends on which ONNX Runtime package was
    /// restored and on what the machine has installed - neither of which can be read from managed
    /// metadata.
    /// </remarks>
    public static IReadOnlyList<ExecutionTarget> AvailableTargets(string probeModelPath)
    {
        var available = new List<ExecutionTarget>();

        foreach (var target in (ExecutionTarget[])
            [ExecutionTarget.Cpu, ExecutionTarget.Cuda, ExecutionTarget.DirectML, ExecutionTarget.OneDnn])
        {
            try
            {
                using var session = OnnxSession.Open(probeModelPath, target);
                if (session.ActualTarget == target) available.Add(target);
            }
            catch (Exception exception) when (exception is NotSupportedException or OnnxRuntimeException)
            {
                // Not available here.
            }
        }

        return available;
    }

    /// <summary>
    /// Runs the same input on several providers and reports what each one costs.
    /// </summary>
    /// <param name="modelPath">The ONNX file.</param>
    /// <param name="inputs">The input to run.</param>
    /// <param name="targets">Which providers to try. All of them when null.</param>
    /// <remarks>
    /// Comparing providers is the only way to know whether an accelerator is worth its deployment
    /// complexity. For a small encoder on a short sequence the CPU provider frequently wins, because
    /// the transfer and launch overhead is a larger share of the work than the arithmetic is.
    /// </remarks>
    public static IReadOnlyList<LatencyReport> Compare(
        string modelPath,
        IReadOnlyDictionary<string, NdArray> inputs,
        IReadOnlyList<ExecutionTarget>? targets = null)
    {
        targets ??= [ExecutionTarget.Cpu, ExecutionTarget.Cuda, ExecutionTarget.DirectML];

        var reports = new List<LatencyReport>();

        foreach (var target in targets)
        {
            OnnxSession? session = null;
            try
            {
                session = OnnxSession.Open(modelPath, target);
            }
            catch (NotSupportedException)
            {
                continue;
            }

            using (session)
            {
                reports.Add(session.Measure(inputs));
            }
        }

        return reports;
    }

    // ------------------------------------------------------------------ quantisation

    /// <summary>
    /// Rewrites a safetensors checkpoint at a lower precision and reports the error it introduced.
    /// </summary>
    /// <param name="sourcePath">The checkpoint to read.</param>
    /// <param name="destinationPath">Where to write the result.</param>
    /// <param name="level">How far to shrink.</param>
    /// <param name="keepFullPrecision">
    /// Tensor names to leave at full precision, in addition to any the source already stores as
    /// integers. Pass index buffers here when the source has already been widened to float.
    /// </param>
    /// <remarks>
    /// <para>
    /// bfloat16 is usually the right choice over float16 despite having three fewer mantissa bits.
    /// It keeps float32's exponent range, so a weight near the bottom of the distribution stays
    /// representable; float16 underflows below about 6e-5, and a tensor with a long tail of small
    /// weights quietly loses that tail.
    /// </para>
    /// <para>
    /// The error is measured by reading back what was written rather than by predicting it from the
    /// format. That is the only way the number reflects the rounding actually performed.
    /// </para>
    /// </remarks>
    public static QuantizationReport Quantize(
        string sourcePath,
        string destinationPath,
        QuantizationLevel level = QuantizationLevel.BFloat16,
        IReadOnlySet<string>? keepFullPrecision = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);

        if (level == QuantizationLevel.None)
        {
            File.Copy(sourcePath, destinationPath, overwrite: true);
            var size = new FileInfo(sourcePath).Length;
            return new QuantizationReport(size, size, 0, 0);
        }

        if (level == QuantizationLevel.Int8)
        {
            throw new NotSupportedException(
                "Int8 quantisation needs per-tensor scales stored beside the weights, which the "
                + "safetensors format has no place for without a companion file. Quantise the ONNX "
                + "graph instead, with onnxruntime's quantisation tools, and load the result here.");
        }

        var dtype = level == QuantizationLevel.BFloat16 ? SafeTensorDType.BF16 : SafeTensorDType.F16;

        var original = SafeTensors.ReadAll(sourcePath);

        // Integer buffers keep full precision. A checkpoint's position_ids and token_type_ids are
        // indices, not weights: shrinking them saves almost nothing and corrupts them outright,
        // because the value 511 is not representable in bfloat16's eight mantissa bits.
        var integerTensors = SafeTensors.Inspect(sourcePath)
            .Where(t => t.DType is SafeTensorDType.I64 or SafeTensorDType.I32 or SafeTensorDType.I16
                or SafeTensorDType.I8 or SafeTensorDType.U64 or SafeTensorDType.U32
                or SafeTensorDType.U16 or SafeTensorDType.U8 or SafeTensorDType.Bool)
            .Select(t => t.Name)
            .ToHashSet(StringComparer.Ordinal);

        if (keepFullPrecision is not null) integerTensors.UnionWith(keepFullPrecision);

        SafeTensors.Write(
            destinationPath,
            original,
            name => integerTensors.Contains(name) ? SafeTensorDType.F32 : dtype);

        double maxError = 0;
        double totalError = 0;
        long count = 0;

        using (var written = SafeTensors.Open(destinationPath))
        {
            foreach (var (name, source) in original)
            {
                var roundTripped = written.Read(name);

                for (var i = 0; i < source.Size; i++)
                {
                    var error = Math.Abs(source.At(i) - roundTripped.At(i));
                    if (error > maxError) maxError = error;
                    totalError += error;
                    count++;
                }
            }
        }

        return new QuantizationReport(
            new FileInfo(sourcePath).Length,
            new FileInfo(destinationPath).Length,
            maxError,
            count == 0 ? 0 : totalError / count);
    }

    /// <summary>
    /// Builds the inputs a BERT-style ONNX encoder expects from a text.
    /// </summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="text">The input text.</param>
    /// <param name="session">The session, whose input names decide what is supplied.</param>
    /// <remarks>
    /// Exported graphs differ in whether they take <c>token_type_ids</c>, so the feed is built from
    /// the session's declared inputs rather than from a fixed list. Supplying an input the graph
    /// does not declare is an error in ORT, and omitting one it does declare is too.
    /// </remarks>
    public static Dictionary<string, NdArray> BuildEncoderInputs(
        HfTokenizer tokenizer, string text, OnnxSession session)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(session);

        var encoding = tokenizer.Encode(text);
        var length = encoding.Length;

        var feeds = new Dictionary<string, NdArray>(StringComparer.Ordinal);

        foreach (var input in session.Inputs)
        {
            var values = input.Name switch
            {
                "input_ids" => encoding.Ids,
                "attention_mask" => encoding.AttentionMask,
                "token_type_ids" => encoding.TypeIds,
                _ => null,
            };

            if (values is null) continue;

            var data = new double[length];
            for (var i = 0; i < length; i++) data[i] = values[i];
            feeds[input.Name] = new NdArray(data, 1, length);
        }

        return feeds;
    }
}
