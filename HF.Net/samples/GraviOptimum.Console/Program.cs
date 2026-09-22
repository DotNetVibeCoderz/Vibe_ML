using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.HFNet.GraviTokenizers;

// GraviOptimum sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
var id = args.Length > 0 ? args[0] : "hf-internal-testing/tiny-random-BertModel";
Console.WriteLine($"=== GraviOptimum: {id} ===\n");

// --- 1. run the repository's ONNX export -------------------------------------
using var model = Optimum.Optimize(id, target: "auto");
Console.WriteLine(model);
Console.WriteLine($"  requested auto, running on {model.Session.ActualTarget}");
Console.WriteLine("  inputs :");
foreach (var spec in model.Session.Inputs) Console.WriteLine($"    {spec}");
Console.WriteLine("  outputs:");
foreach (var spec in model.Session.Outputs) Console.WriteLine($"    {spec}");
Console.WriteLine();

var tokenizer = HfTokenizer.FromPretrained(id);
var feeds = Optimum.BuildEncoderInputs(tokenizer, "HF.Net runs ONNX exports on .NET.", model.Session);
Console.WriteLine($"  feeding: {string.Join(", ", feeds.Select(f => $"{f.Key}[{string.Join(",", f.Value.Shape.ToArray())}]"))}");

var outputs = model.Run(feeds);
foreach (var (name, tensor) in outputs)
{
    Console.WriteLine($"  {name}: [{string.Join(" x ", tensor.Shape.ToArray())}]"
        + $" first {string.Join(", ", tensor.ToArray().Take(4).Select(v => v.ToString("F4")))}");
}
Console.WriteLine();

// --- 2. latency ---------------------------------------------------------------
Console.WriteLine("--- latency ---");
Console.WriteLine($"  {model.Measure(feeds, iterations: 30)}");
Console.WriteLine();

// --- 3. quantisation ----------------------------------------------------------
Console.WriteLine("--- quantisation ---");
var info = Hub.ModelInfo("prajjwal1/bert-tiny");
if (info.HasSafeTensors || true)
{
    // bert-tiny publishes only a pickle, so the source for this demo is written first as F32
    // safetensors and then shrunk - which also exercises the writer.
    var bin = Hub.DownloadFile("prajjwal1/bert-tiny", "pytorch_model.bin");
    var tensors = Gravicode.HFNet.GraviHub.Io.PyTorchCheckpoint.ReadAll(bin);

    // Which tensors are index buffers rather than weights. Widening them to F32 below loses that
    // distinction, so it is captured here and handed to Quantize - otherwise position_ids, whose
    // values reach 511, rounds to 512 in bfloat16 and dominates the reported error.
    var indexTensors = Gravicode.HFNet.GraviHub.Io.PyTorchCheckpoint.Inspect(bin)
        .Where(t => t.DType is not (Gravicode.HFNet.GraviHub.Io.SafeTensorDType.F32
            or Gravicode.HFNet.GraviHub.Io.SafeTensorDType.F64
            or Gravicode.HFNet.GraviHub.Io.SafeTensorDType.F16
            or Gravicode.HFNet.GraviHub.Io.SafeTensorDType.BF16))
        .Select(t => t.Name)
        .ToHashSet(StringComparer.Ordinal);

    Console.WriteLine($"  index tensors kept at F32: {string.Join(", ", indexTensors)}");

    var work = Path.Combine(Path.GetTempPath(), "hfnet-quant");
    Directory.CreateDirectory(work);

    var f32 = Path.Combine(work, "model-f32.safetensors");
    Gravicode.HFNet.GraviHub.Io.SafeTensors.Write(f32, tensors, Gravicode.HFNet.GraviHub.Io.SafeTensorDType.F32);

    foreach (var level in (QuantizationLevel[])[QuantizationLevel.BFloat16, QuantizationLevel.Float16])
    {
        var target = Path.Combine(work, $"model-{level}.safetensors");
        Console.WriteLine($"  {level,-10} {Optimum.Quantize(f32, target, level, indexTensors)}");
    }
}
