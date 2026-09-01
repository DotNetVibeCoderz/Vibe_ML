# Installation

## From an installer

The quickest route, and the only one that needs nothing installed beforehand — the packages carry
the .NET runtime with them:

| Platform | Package |
| --- | --- |
| Windows 10/11 x64 | `LocalGen-<version>-win-x64.msi` |
| macOS 12+ | `LocalGen-<version>-osx-arm64.pkg` or `-x64` |
| Debian, Ubuntu | `localgen_<version>_amd64.deb` |
| Other Linux | `localgen-<version>-linux-x64.tar.gz` |

Take the `-cuda12` variant for an NVIDIA GPU; it is larger by the size of the CUDA runtime.

Everything below is for building from source, which is what you want if you are working on
LocalGen rather than using it. [Installers](installers.md) covers how the packages are built,
signed and released.

## Prerequisites

- **.NET 10 SDK** — [download](https://dotnet.microsoft.com/download)
- **Disk space** — models range from 100 MB to 40 GB. A useful 7B model at Q4_K_M is about 4.5 GB.
- **Memory** — roughly the model's file size plus 1–2 GB for the KV cache. LocalGen estimates this
  per model and shows it on the Models screen.

Optional:

- **NVIDIA GPU with CUDA 12** for GPU inference
- **Python, Node.js** if you want the Code Execution kernel function to run those languages
- **Foundry Local** (`winget install Microsoft.FoundryLocal`) for that backend

## Build from source

```bash
git clone https://github.com/gravicode/LocalGen.git
cd LocalGen
dotnet build
```

The default build uses the CPU backend. It is the smaller download and works everywhere.

### GPU backends

The CUDA package is around 500 MB, so it is opt-in:

```bash
# NVIDIA
dotnet build -p:LlamaBackend=cuda12

# Both CPU and CUDA, selected at runtime
dotnet build -p:LlamaBackend=all
```

After building with a GPU backend, pick the device on the **Engine** screen in Admin Control, or
set it in configuration:

```json
{ "LocalGen": { "Engine": { "Device": "Cuda" } } }
```

Verify what was detected:

```bash
dotnet run --project src/LocalGen.Cli -- engines
```

A working CUDA build reports `Cuda` among its devices and a banner beginning with the device name:

```
devices : Auto · Cpu · Cuda
build   : CUDA0 | CUDA : ARCHS = 500,610,700,750,800,860,890 | USE_GRAPHS = 1 …
```

If it says `Cpu` alone, the CUDA build did not load and inference will be quietly slow rather
than failing. Measured on an RTX 4060 with a 1.5B model at Q4_K_M, the difference is large enough
to notice without a benchmark:

| | CPU | CUDA |
| --- | ---: | ---: |
| Throughput | 21 tokens/s | 163 tokens/s |
| Time to first token | 339 ms | 54 ms |

#### Two things LocalGen handles for you

Both of these cause a CUDA build to fall back to CPU silently, and both are worked around
automatically — they are recorded here because the symptom is so easy to misread.

**The CUDA package has no CPU module.** Its `ggml.dll` imports from `ggml-cpu.dll`, which only the
CPU package ships. Without it the CUDA libraries fail to load with "the specified module could not
be found". LocalGen references the CPU backend alongside CUDA and copies that module into the CUDA
folder after every build.

**A newer CUDA toolkit is judged incompatible.** LLamaSharp compares the machine's CUDA toolkit
version against the backend it ships, so a machine whose primary toolkit is CUDA 13 is treated as
having no usable CUDA at all — even though the driver runs CUDA 12 binaries perfectly well.
LocalGen names the CUDA library outright when one is present, skipping that check.

The NVIDIA driver is what has to be recent; the CUDA toolkit does not need to match the backend.

## Install the CLI as a global tool

```bash
dotnet pack src/LocalGen.Cli -c Release
dotnet tool install --global --add-source src/LocalGen.Cli/bin/Release LocalGen.Cli
localgen --help
```

## First run

```bash
# A small model to check everything works — about 100 MB
localgen pull huggingface:bartowski/SmolLM2-135M-Instruct-GGUF
localgen run smollm2-135m-instruct:q4_k_m "Say hello"

# Something actually useful
localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF
localgen run qwen2.5-7b-instruct:q4_k_m
```

## Where LocalGen keeps its data

| Platform | Location |
| --- | --- |
| Windows | `%LOCALAPPDATA%\LocalGen` |
| macOS, Linux | `~/.localgen` |

Override it with the `LOCALGEN_HOME` environment variable.

```
<data>/
  models/       Downloaded weights
  skills/       Installed skills
  sessions/     Saved playground conversations
  logs/         Log files
  workspace/    Working directory for the file and code tools
  manifest.json Model catalogue
  mcp.json      Configured MCP servers
  rag.localgen.db  Vector index, when using the SQLite backend
```

Models are large and belong outside source control; the repository's `.gitignore` already
excludes `*.gguf`, `*.onnx` and `*.safetensors`.

## Configuration

Settings come from `appsettings.json`, environment variables prefixed `LOCALGEN_`, and
command-line flags, in increasing order of precedence.

```json
{
  "LocalGen": {
    "Server":  { "Host": "127.0.0.1", "Port": 11434, "ApiKey": null,
                 "ModelIdleTimeout": "00:05:00", "MaxLoadedModels": 2 },
    "Engine":  { "Default": "LlamaSharp", "Device": "Auto",
                 "GpuLayers": null, "ContextSize": null, "Threads": null },
    "Runtime": { "OfflineMode": false, "PreloadModel": null, "CollectMetrics": true },
    "Tools":   { "CodeExecution": true, "AllowDependencyInstall": false,
                 "TavilyApiKey": null, "AllowedPaths": [] },
    "Rag":     { "Provider": "sqlite", "EmbeddingModel": "nomic-embed-text",
                 "ChunkSize": 1000, "ChunkOverlap": 200, "TopK": 5 }
  }
}
```

Settings worth knowing about:

- **`Server:ApiKey`** — required on every request when set. LocalGen binds to loopback by default,
  so a key is only needed once you expose it on a network. Once you do, set one: an open inference
  endpoint also exposes the tool functions.
- **`Server:MaxLoadedModels`** — how many models stay resident. Raising it costs memory.
- **`Runtime:OfflineMode`** — blocks every outbound call. Inference keeps working.
- **`Tools:AllowedPaths`** — extra directories the file and code tools may touch. The LocalGen
  workspace is always allowed; nothing else is.
- **`Tools:AllowDependencyInstall`** — lets executed code install packages. Off by default because
  it lets a model change the host machine.

## Troubleshooting

**"The llama.cpp native library is missing."** No backend package was restored. Build with
`-p:LlamaBackend=cpu` or `cuda12`.

**"onnxruntime-genai native binaries are missing."** Publish with a runtime identifier, for
example `dotnet publish -r win-x64`.

**A model fails to load with an out-of-memory error.** Lower the context size on the Engine screen,
or choose a smaller quantization. The Models screen shows an estimate per model.

**Inference is slower than expected.** Check the Engine screen — if it says CPU and you have a GPU,
the CUDA backend was not built in. `localgen benchmark <model>` gives a repeatable number.
