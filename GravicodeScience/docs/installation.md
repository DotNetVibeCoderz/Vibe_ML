# Installation

*[Bahasa Indonesia](id/installation.md)*

## Requirements

| | |
|---|---|
| **.NET SDK** | 10.0 or later |
| **OS** | Windows, Linux or macOS |
| **RAM** | 4 GB minimum; 8 GB to run the benchmarks |
| **GPU** *(optional)* | Any CUDA or OpenCL device with double-precision support |

Check your SDK:

```bash
dotnet --version   # expects 10.0.x
```

## Building from source

```bash
git clone https://github.com/DotNetVibeCoderz/Vibe_ML.git
cd Vibe_ML/GravicodeScience
dotnet build Gravicode.Science.sln -c Release
dotnet test
```

The whole test suite (1,049 tests) runs in about a minute and a half.

## Adding the libraries to a project

The libraries are independent packages. Take only what you need — every one of them pulls in
`GraviNum`, and nothing else is mandatory.

```bash
dotnet add package Gravicode.Science.GraviNum
dotnet add package Gravicode.Science.GraviFrame
dotnet add package Gravicode.Science.GraviLearn
dotnet add package Gravicode.Science.GraviText
dotnet add package Gravicode.Science.GraviGraph
dotnet add package Gravicode.Science.GraviProb
```

While working from a clone, reference the projects directly instead:

```xml
<ItemGroup>
  <ProjectReference Include="../GravicodeScience/src/GraviNum/GraviNum.csproj" />
  <ProjectReference Include="../GravicodeScience/src/GraviFrame/GraviFrame.csproj" />
</ItemGroup>
```

Each package carries its XML documentation, so the comments in this repository reach you through
IntelliSense, and a `.snupkg` symbol package, so a debugger can step into library code. See
[publishing.md](publishing.md) for how the packages are built and released.

### What depends on what

```
GraviNum ──┬── GraviFrame ── GraviLearn ── GraviText ── GraviGraph
           ├── GraviProb
           └── (everything)
```

`GraviNum` never references anything above it. `GraviText` depends on `GraviLearn` because its
task pipelines train real classifiers rather than reimplementing them, and `GraviGraph` depends
on `GraviText` because node2vec is random walks fed to skip-gram.

## GPU support

The GPU backend is built on [ILGPU](https://ilgpu.net) and is **entirely optional**. Nothing has
to be configured: the first time anything asks for it, the library probes for a device, and if
none is usable it silently keeps running on the CPU.

```csharp
using Gravicode.Science.GraviNum.Compute;

Console.WriteLine(Compute.DescribeDevices());
// CPU: CPU (SIMD x4, 8 threads); GPU: OpenCL - Intel(R) UHD Graphics 620 (6502 MB)

Console.WriteLine(Compute.IsGpuAvailable);
```

To use a specific backend explicitly:

```csharp
var product = Compute.Dot(a, b, Compute.Gpu);   // force GPU
var onCpu   = Compute.Dot(a, b, Compute.Cpu);   // force CPU
var best    = Compute.Dot(a, b);                // let the size decide
```

**A GPU is not automatically faster.** Every call copies both operands across the bus and the
result back, and that cost is fixed while the arithmetic grows with the problem. Below roughly a
quarter of a million elements the CPU wins; `Compute.Best` encodes that threshold so the default
path picks correctly. See [benchmarks.md](benchmarks.md) for the measured crossover.

### Driver requirements

| Backend | Needs |
|---|---|
| CUDA | NVIDIA driver 450 or later |
| OpenCL | The vendor's OpenCL runtime (Intel, AMD or NVIDIA) |
| CPU accelerator | Nothing — ILGPU's software fallback, useful only for testing kernels |

If `Compute.IsGpuAvailable` reports `false`, the most common causes are a missing OpenCL runtime
and a device without float64 support. Neither is an error condition: the CPU path is fully
featured and is what the test suite exercises.

## Verifying the installation

```csharp
using Gravicode.Science.GraviNum;

Console.WriteLine(GraviInfo.Banner("GraviNum"));
Console.WriteLine(GraviInfo.HardwareReport());

var a = NdArray.Arange(6).Reshape(2, 3);
Console.WriteLine(a.Dot(a.T));
```

Or run any of the six samples, which print the same report before doing their work:

```bash
dotnet run --project samples/GraviNum.Console
dotnet run --project samples/GraviFrame.Console
dotnet run --project samples/GraviLearn.Console
dotnet run --project samples/GraviText.Console
dotnet run --project samples/GraviGraph.Console
dotnet run --project samples/GraviProb.Console
```

## Notebooks

The notebooks in [`notebooks/`](../notebooks) need the
[Polyglot Notebooks](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)
extension for VS Code. They reference the built assemblies, so build in Release first:

```bash
dotnet build Gravicode.Science.sln -c Release
```

## Benchmarks

BenchmarkDotNet requires a Release build and will refuse to run otherwise:

```bash
dotnet run --project benchmarks/GraviNum.Benchmark -c Release
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --filter "*MatrixProduct*"
```

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
