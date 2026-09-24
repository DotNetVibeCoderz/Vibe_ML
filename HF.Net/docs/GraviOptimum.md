# GraviOptimum

**Hardware-aware inference through ONNX Runtime, and weight quantisation that reports its cost.**

Mirrors `optimum`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviOptimum;
```

## Why this exists

The managed encoder in [GraviTransformers](GraviTransformers.md) computes in `double` and agrees
with torch to about 1e-13. It is there so a model can be loaded, inspected and understood in pure
.NET. A production request should go through an ONNX export running on single-precision kernels
that were written for the hardware. On `bert-base-uncased`, one 12-token sentence takes **23.8 ms**
this way, against 36.4 ms in torch and 111 ms managed. See [benchmarks](benchmarks.md).

```csharp
using var model = Optimum.Optimize("hf-internal-testing/tiny-random-BertModel", target: "auto");
using var tokenizer = HfTokenizer.FromPretrained("hf-internal-testing/tiny-random-BertModel");

var feeds = Optimum.BuildEncoderInputs(tokenizer, "HF.Net runs ONNX on .NET.", model.Session);

foreach (var (name, tensor) in model.Run(feeds))
    Console.WriteLine($"{name}: [{string.Join(" x ", tensor.Shape.ToArray())}]");

Console.WriteLine(model.Measure(feeds, iterations: 50));
// Cpu: best 0.67 ms, median 1.04 ms (50 runs)
```

## Execution providers

```csharp
Optimum.Optimize(id, target: "CPU");       // or CUDA, DirectML, oneDNN, auto
Optimum.OptimizeFile("model.onnx", "cuda");
Optimum.ParseTarget("gpu");                // -> ExecutionTarget.Cuda
```

**A provider that is named and missing throws.** ONNX Runtime's own default is to fall back to the
CPU silently, which is exactly how a "CUDA" deployment turns out to have been running on the CPU for
months. Use `"auto"` when a fallback is genuinely wanted; it tries CUDA, then DirectML, then CPU.

```csharp
session.RequestedTarget;      // what was asked for
session.ActualTarget;         // what was registered
session.RanOnRequestedTarget;
```

The base `Microsoft.ML.OnnxRuntime` package ships the **CPU provider only**. CUDA needs
`Microsoft.ML.OnnxRuntime.Gpu`; DirectML needs `Microsoft.ML.OnnxRuntime.DirectML`.

## Comparing providers

```csharp
foreach (var report in Optimum.Compare("model.onnx", feeds))
    Console.WriteLine(report);
```

Comparing is the only way to know whether an accelerator is worth its deployment complexity. For a
small encoder on a short sequence the CPU provider frequently wins, because transfer and launch
overhead is a larger share of the work than the arithmetic is.

## Measurement

```csharp
session.Measure(feeds, iterations: 20, warmup: 5);   // -> LatencyReport
report.Best; report.Median; report.PeakThroughput;
```

**The warm-up is not optional.** The first call builds the execution plan and allocates the arena,
and on a GPU provider it also compiles kernels. Timing it reports setup cost as if it were inference
cost, which is how ONNX exports get reported as slower than they are.

The best time is reported alongside the median for the same reason a benchmark harness does: on a
throttling laptop the mean measures the thermal state of the room.

## Quantisation

```csharp
var report = Optimum.Quantize("model.safetensors", "model-bf16.safetensors",
                              QuantizationLevel.BFloat16);

Console.WriteLine(report);
// 31.9 MB -> 16.0 MB (50 %), max error 1.56E-002, mean 6.79E-005
```

**The error is measured by reading back what was written**, not predicted from the format. That is
the only way the number reflects the rounding actually performed.

### bfloat16 or float16?

Both are sixteen bits and they spend them differently:

| | Exponent | Mantissa | Consequence |
|---|---|---|---|
| bfloat16 | 8 bits, same as float32 | 7 bits | Full range; coarser steps |
| float16 | 5 bits | 10 bits | Finer steps; underflows below ~6e-5 |

**bfloat16 is usually the right choice** despite the coarser steps: it keeps float32's exponent
range, so a weight near the bottom of the distribution stays representable. float16 quietly loses
the tail of a long-tailed weight distribution.

### Index buffers

A checkpoint is not uniformly weights. A BERT checkpoint carries an integer `position_ids` buffer
holding values up to 511, and bfloat16 has eight mantissa bits - so quantising it along with
everything else **rounds 511 to 512** and reports a maximum error of 1.0 that has nothing to do with
the model.

Tensors the source stores as integers are kept at full precision automatically. For a source already
widened to float, name them:

```csharp
Optimum.Quantize(source, destination, QuantizationLevel.BFloat16,
                 keepFullPrecision: new HashSet<string> { "position_ids" });
```

### Int8

**Refused, not approximated.** Symmetric int8 needs per-tensor scales stored beside the weights, and
the safetensors format has nowhere to put them. Quantise the ONNX graph instead, with onnxruntime's
own quantisation tools, and load the result here.

## Building inputs

```csharp
Optimum.BuildEncoderInputs(tokenizer, text, session);
```

Exported graphs differ in whether they take `token_type_ids`, so the feed is built from the
**session's declared inputs** rather than from a fixed list. Supplying an input the graph does not
declare is an error in ORT, and so is omitting one it does.

## Type conversion

Everything in the Gravicode stack is `double`; essentially every ONNX graph wants `float` for
activations and `int64` for token ids. The conversion happens at the boundary, driven by the graph's
declared element type - feeding a double tensor to a float input throws inside the runtime with a
message that names neither the tensor nor the caller.

## See also

[GraviTransformers](GraviTransformers.md) · [GraviDiffusers](GraviDiffusers.md) · [GraviAccelerate](GraviAccelerate.md)
