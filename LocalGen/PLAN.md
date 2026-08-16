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

## Phase 5 — Depth 🔜 next

Capabilities the architecture already allows but that are not built yet.

- [ ] **Vision models** — the content model already carries image parts; wiring them through
      LlamaSharp's multimodal support is the remaining work
- [ ] **Multi-GPU tensor parallelism** — `TensorSplit` is plumbed through to llama.cpp but has not
      been exercised on real multi-GPU hardware
- [x] **Sharded GGUF** — split models are listed, downloaded whole, and treated as one model
      throughout the store (loading them has not been run against real multi-part weights, which
      start around 50 GB)
- [ ] **Native tool calling** — use each model's own tool template where it has one, keeping the
      prompt protocol as the fallback
- [ ] **Batched inference** — serve concurrent requests against one context rather than serialising
- [ ] **Agent Framework orchestration** — multi-agent delegation beyond the single-agent loop
- [x] **Model quantization** — `localgen quantize` on the CLI and a Quantize button per model in
      the Admin Control, both converting in-process through llama.cpp

## Phase 6 — Production

- [ ] Prometheus/OpenTelemetry export from the existing metrics
- [ ] Kubernetes operator and Helm chart
- [ ] Multi-tenant API keys with per-key quotas
- [ ] Response caching keyed on prompt and sampling settings
- [ ] Signed and notarised installers for Windows, macOS and Linux

---

## Known limitations

Recorded here so they are decisions rather than surprises.

- **Generation is serialised per model.** llama.cpp decodes one sequence at a time against a
  context; running several concurrently would corrupt the KV cache. Requests queue per model.
- **Embeddings need the LlamaSharp backend.** ONNX Runtime GenAI has no embedding path.
- **GPU counters are NVIDIA-only.** They come from `nvidia-smi`; AMD and Intel report nothing.
- **Images are not OCR'd during ingestion.** They are indexed by name and left for a
  vision-capable model to read.
- **Foundry Local is Windows and macOS only**, following its own platform support.
