# LocalGen — Development Roadmap

*Gravicode Studios · dipimpin oleh Kang Fadhil*

This is the plan the product is being built against. Phase 1 is complete and running; the phases
after it are the intended order of work, not a schedule.

---

## Guiding constraints

These hold across every phase and are what most decisions get checked against.

| Constraint | Why it binds |
| --- | --- |
| The HTTP API is a drop-in replacement for OpenAI's | Endpoint shapes, field names and SSE framing follow OpenAI, not LocalGen's internal model. Wire types live apart from domain types so the contract cannot drift. |
| Backends stay interchangeable | Nothing above `IInferenceEngine` may reference LlamaSharp, ONNX or Foundry types. A new backend is a new project, not a change to the server. |
| Offline mode must work | No code path required for inference may touch the network. Features that need it fail with a clear message rather than hanging. |
| Tools are a permission boundary | Code execution and file access are gated by configuration and path allow-lists, never by prompt instructions alone. |
| Both UIs ship light and dark | The palette is defined once as tokens per surface and reused, not re-picked per screen. |

---

## Phase 1 — Engine and API ✅ complete

The foundation: a working inference engine with an OpenAI-compatible surface.

- [x] `LocalGen.Core` — engine contracts, chat/embedding types, Modelfile format, OpenAI protocol,
      prompt-based tool-call protocol with a streaming tag filter
- [x] `LocalGen.Runtime` — GGUF header reader, manifest-backed model store with directory scanning,
      Hugging Face catalogue and resumable downloader, engine registry, session manager with idle
      eviction, CPU/GPU sampling
- [x] `LocalGen.Engines.LlamaSharp` — GGUF on CPU or GPU, GBNF grammar, JSON mode, embeddings,
      chat template read from the model file
- [x] `LocalGen.Engines.Onnx` — ONNX Runtime GenAI with execution provider selection
- [x] `LocalGen.Engines.FoundryLocal` — drives the Foundry service and proxies over HTTP
- [x] `LocalGen.Server` — `/v1/chat/completions` with SSE, `/v1/completions`, `/v1/embeddings`,
      `/v1/models`, plus management endpoints; hostable in-process
- [x] `LocalGen.Sdk` — typed client, streaming, `IChatClient` adapter, DI extensions
- [x] `LocalGen.Cli` — ten commands, working against a running server or in-process

## Phase 2 — Orchestration and knowledge ✅ complete

- [x] `LocalGen.Kernel` — Semantic Kernel chat service over local weights, agent loop with visible
      tool calls
- [x] Built-in kernel functions — Math, Internet Search, Download, Web Scrape, Code Execution,
      Time/Date, File System; all on by default, individually toggleable
- [x] Skills — manifest format, assets, runnable scripts, gallery install
- [x] MCP — stdio and HTTP servers, curated gallery, per-conversation attachment
- [x] `LocalGen.Rag` — chunking, PDF/HTML/markdown/code ingestion, SQLite, Qdrant, Chroma,
      Azure AI Search and in-memory backends

## Phase 3 — Interfaces ✅ complete

- [x] Avalonia Admin Control — Service, Playground, Models, Engine, Monitor, About
- [x] Blazor web UI — chat, model browser, analytics
- [x] Shared design language across both, light and dark

## Phase 4 — Ecosystem ✅ complete

The surface a newcomer meets first.

- [x] Sample applications — console, agent with tools, RAG, WPF, and an Avalonia app with the
      runtime embedded; Blazor is covered by `LocalGen.Web` itself
- [x] `dotnet new` templates — `localgen-console`, `localgen-webapi`, `localgen-desktop`, each
      parameterised by endpoint and model
- [x] VS Code extension — model tree with load state, pull with progress, chat panel, and editor
      commands for explaining code and diagnostics
- [x] Packable NuGet artifacts for `LocalGen.Sdk`, `LocalGen.Core`, `LocalGen.Cli` and the
      template pack

Still outstanding in this area, both needing infrastructure rather than code:

- [ ] Publish the packages to nuget.org and the extension to the marketplace
- [ ] Host a community skill gallery index with a submission process

## Phase 5 — Depth

Capabilities the architecture already allows.

- [x] **Vision models** — image parts are spliced into the token sequence through llama.cpp's
      MTMD interface, with the projector loaded beside the weights
- [x] **Multi-GPU tensor parallelism** — the accelerators llama.cpp registered are enumerated and
      surfaced through the API, the CLI and the Engine screen; `TensorSplit`, `SplitMode` and
      `MainGpu` are configurable per server and per model; a split is resolved against the devices
      actually present, with every mismatch reported rather than silently obeyed. **Not yet run on
      a two-GPU host** — see below
- [x] **Sharded GGUF** — split models are listed, downloaded whole, and treated as one model
      throughout the store (loading them has not been run against real multi-part weights, which
      start around 50 GB)
- [x] **Native tool calling** — each model family's own convention is detected from its chat
      template and used for both the prompt and the parser; the Hermes form is the fallback
- [x] **Batched inference** — concurrent requests decode together against one context, opt-in
      through `Engine.BatchedInference`
- [x] **Agent Framework orchestration** — sequential, concurrent and handoff teams over the
      existing agent loop
- [x] **Model quantization** — `localgen quantize` on the CLI and a Quantize button per model in
      the Admin Control, both converting in-process through llama.cpp

**What "not yet run on a two-GPU host" means for multi-GPU.** Everything that can be checked
without a second card has been: device enumeration reports the real GPU on this machine, a split
naming two GPUs is refused on it with the right message, an out-of-range `MainGpu` falls back to
device 0, and the resolver's behaviour on two, three and four devices is under unit test. What is
left is the part only hardware can answer — whether a model split 60/40 across two cards produces
correct tokens at a useful speed. The code path that applies the split to `ModelParams` is three
lines; the reason it is still called unverified is that llama.cpp accepts a wrong split without
complaint, so "it loaded" would prove nothing.

## Phase 6 — Production

- [x] Prometheus/OpenTelemetry export from the existing metrics
- [x] Helm chart — GPU, ingress, quotas, `ServiceMonitor` and the web tier, all templated
- [ ] Kubernetes operator — deferred; see below
- [x] Multi-tenant API keys with per-key quotas
- [x] Response caching keyed on prompt and sampling settings
- [x] Installers for Windows, macOS and Linux — an MSI, a `.pkg` and a `.deb`/tarball, each
      carrying the CLI, the server and the Admin Control self-contained, built by one script per
      platform and one release workflow
- [ ] **Signing them.** The signing and notarisation steps are written and wired to credentials,
      and skipped with a warning when none are present. They cannot be exercised without the
      certificates — see below

**What signing needs that code cannot supply.** A Windows code-signing certificate (~$200–600/year
from a public CA) and an Apple Developer Program membership ($99/year), which issues the Developer
ID Application and Developer ID Installer certificates and the App Store Connect key notarisation
uses. Neither can be generated locally or worked around: macOS refuses an un-notarised package
outright, and an unsigned Windows installer arrives behind a SmartScreen warning most people read
as a malware alert. Linux needs only an OpenPGP key, which costs nothing — that path is complete
apart from someone deciding which key to publish under.
[`docs/installers.md`](docs/installers.md) lists what to obtain and where each credential goes.

**Why the operator is deferred.** An operator earns its keep by reconciling many instances or a
lifecycle a chart cannot express. LocalGen is a single stateful pod pinned to the node holding its
GPU and its model volume, which the chart already describes completely. The one thing an operator
would add is declarative model management — a `LocalGenModel` resource that ensures a model is
pulled — and that is worth building only once someone is running enough instances to feel the
absence. Writing a controller that could not be exercised against a real cluster from here would
also mean shipping it unverified, which is not how anything else in this project was built.

---

## Known limitations

Recorded here so they are decisions rather than surprises.

- **Generation is serialised per model unless batching is on.** A llama.cpp context decodes one
  sequence at a time, so by default requests queue per model. `Engine.BatchedInference` switches to
  a shared context where concurrent requests decode together — measured at 2.9× the aggregate
  throughput on four overlapping requests — at the cost of those requests sharing the context
  window between them.
- **Batching does not apply to vision models.** An image is encoded by the projector before any of
  it reaches the batch, so the part that dominates a vision request is the part batching cannot
  overlap. Vision models stay on the serialised path even when batching is enabled.
- **Handoff orchestration needs a model that can call tools reliably.** A delegation is a tool
  call, so a coordinator is only as good as the model's function calling. A 1.5B model handles a
  single specialist but does not reliably route among three; the sequential and concurrent
  patterns leave nothing to its judgement and work at any size.
- **Embeddings need the LlamaSharp backend.** ONNX Runtime GenAI has no embedding path.
- **Semantic Kernel is held at the 1.74 line, and cannot move until the vector-store connectors
  do.** The connectors call `VectorSearchOptions.OldFilter`, which
  `Microsoft.Extensions.VectorData.Abstractions` removed in 10.5.0, while Semantic Kernel 1.79
  requires 10.5.2 or newer — so on 1.79 every vector search threw at runtime. 1.74 is the release
  the connectors shipped alongside, so the whole set agrees on abstractions 10.1.0 and search
  works on every backend. The pin costs five minor versions of Semantic Kernel; nothing LocalGen
  uses needed them, and the four packages should be raised together once connectors built against
  the current abstractions are published.
- **GPU counters are NVIDIA-only.** They come from `nvidia-smi`; AMD and Intel report nothing.
- **Images are not OCR'd during ingestion.** They are indexed by name and left for a
  vision-capable model to read.
- **A vision request under-reports its prompt tokens.** The count covers the text; the tokens the
  image expands into are added inside llama.cpp, where LocalGen cannot see them. This matters for
  the per-key token quota, which therefore undercharges image requests.
- **Vision requests build a context per request.** LLamaSharp exposes multimodal input only on its
  stateful executors, whose KV cache would otherwise carry one request's conversation into the
  next. The correctness is worth the allocation, but it does make a vision request cost more to
  start than a text one.
- **Foundry Local is Windows and macOS only**, following its own platform support.
- **Layer splitting across GPUs buys capacity, not speed.** Each card computes its own layers and
  hands the activations on, so two cards run in sequence rather than together. What it makes
  possible is running a model that fits in neither card alone. `SplitMode: Row` does make them
  compute together, but exchanges activations at every layer, so on consumer boards without a fast
  link between the cards it is usually slower than layer splitting rather than faster.
- **An installer carries one backend.** The CPU and CUDA packages are separate downloads because
  bundling both would add several hundred megabytes for every user to spare NVIDIA owners a
  choice on the download page.
