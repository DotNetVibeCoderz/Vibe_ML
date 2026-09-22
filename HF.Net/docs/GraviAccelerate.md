# GraviAccelerate

**Device selection, sharding, and measurement that can be trusted.**

Mirrors `accelerate`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviAccelerate;
```

## What hardware is this?

```csharp
foreach (var device in Accelerator.Devices()) Console.WriteLine(device);
// Cpu: 8 logical cores, SIMD width 4
// Gpu: <device name> / unavailable (DllNotFoundException)

Console.WriteLine(Accelerator.Describe());
```

Probing for a GPU initialises ILGPU, which can throw on a machine with a broken driver. A missing
accelerator is a normal condition, not an error worth propagating, so the probe never throws.

## The GPU is not the default

```csharp
var accelerator = new Accelerator(DeviceKind.Auto);      // Cpu | Gpu | Auto
accelerator.Backend(elementCount);
accelerator.Dot(a, b);
```

This is a **measured** decision, not caution. The whole stack is `double`, and consumer and
integrated GPUs run double-precision arithmetic at a small fraction of their single-precision rate -
the foundation measured its ILGPU path **5 to 8 times slower** than the CPU on an integrated device.

`DeviceKind.Auto` therefore only reaches for the GPU above `Accelerator.GpuThreshold` (2²⁰
elements), where the transfer and the arithmetic both pay for themselves. Below it the kernel launch
and the round trip over PCIe dominate, so the GPU loses even when it is genuinely faster per
element.

If you want single-precision throughput, that is [GraviOptimum](GraviOptimum.md)'s job.

## Sharding

```csharp
var shards = accelerator.Partition(itemCount);

var averaged = accelerator.ParallelGradient(itemCount, shard =>
{
    var total = 0.0;
    foreach (var index in shard.Indices) total += Loss(index);
    return new NdArray([total / shard.Count], 1);
});
```

**The averaging is weighted by shard size.** A plain mean of per-worker means equals the global mean
only when the shards are the same size, and `Partition` produces uneven ones whenever the worker
count does not divide the item count. Unweighted, a model trains to something slightly wrong that no
shape or convergence check would catch.

Shards are collected in **worker order, not completion order**. Floating-point addition is not
associative, so arrival order would make the result depend on thread scheduling - and a training run
that cannot be reproduced cannot be debugged.

## Training

```csharp
var report = accelerator.Train(estimator, dataset, labelColumn: "survived", epochs: 5);

Console.WriteLine(report);
// 5 epochs, 4,450 examples in 0.82s (5,427/s) on Auto across 8 worker(s), score 0.8112
```

This prepares the matrices, runs the estimator's own `Fit` once per epoch, and reports what the run
cost. It deliberately does **not** shard the fit: sharding only composes for an estimator whose
parameters can be averaged, and averaging the coefficients of two decision trees produces something
that is not a tree. Where averaging is valid, build the loop from `Partition` and `ParallelGradient`.

## Measurement

```csharp
var elapsed = Accelerator.Measure(() => model.Embed(text), iterations: 20, warmup: 40);
```

**The warm-up count is not padding.** Tiered JIT recompiles a hot method after roughly thirty calls,
so a measurement taken with two warm-up iterations times the interpreter's output. The foundation
recorded a 192-element cube measuring eight times slower than a 224-element one that way.

The **best** observed time is returned rather than the mean, for the same reason a benchmark harness
does: on a throttling laptop the mean measures the thermal state of the room. Never compare timings
taken in separate runs - run the variants alternately in one process.

## Releasing the GPU

```csharp
Accelerator.ReleaseGpu();
```

## See also

[GraviOptimum](GraviOptimum.md) · [GraviDatasets](GraviDatasets.md)
