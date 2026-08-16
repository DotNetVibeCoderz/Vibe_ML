# Samples

Each sample shows one thing well.

| Sample | Shows | Needs a server |
| --- | --- | --- |
| [ConsoleChat](ConsoleChat/) | Streaming chat with the SDK — the smallest useful client | yes |
| [AgentWithTools](AgentWithTools/) | Function calling through `Microsoft.Extensions.AI` | yes |
| [DocumentQA](DocumentQA/) | RAG end to end: ingest a folder, then answer from it | no |
| [WpfChat](WpfChat/) | A WPF client, and the dispatcher marshalling it needs | yes |
| [AvaloniaEmbedded](AvaloniaEmbedded/) | A desktop app with the runtime **inside it** — no server at all | no |

Where a sample needs a server, start one first:

```bash
dotnet run --project src/LocalGen.Cli -- serve
```

## Running one

```bash
dotnet run --project samples/ConsoleChat
dotnet run --project samples/ConsoleChat -- qwen2.5-7b-instruct:q4_k_m   # pick the model
```

`DocumentQA` takes a folder instead:

```bash
dotnet run --project samples/DocumentQA -- ./docs
```

## The two patterns

**Client of a server** — `ConsoleChat`, `AgentWithTools` and `WpfChat` use `LocalGen.Sdk` over
HTTP. Several applications can share one loaded model, and the server can be on another machine.
This is the right default.

**Embedded runtime** — `AvaloniaEmbedded` and `DocumentQA` compose `LocalGen.Runtime` into their
own container and load models in-process. The result is one executable with nothing to install or
start alongside it. The trade-off is that the model's memory is the application's memory, and
nothing else can share it.

`AvaloniaEmbedded` reads the same data directory the CLI and server use, so a model pulled with
`localgen pull` appears in its picker with no extra step.

## Beyond these

The two full applications in the repository are worked examples in their own right:

- **`src/LocalGen.Web`** — a Blazor client with streaming chat, a model browser and an analytics
  page built on the metrics endpoint.
- **`src/LocalGen.Desktop`** — Avalonia Admin Control, which hosts the *server itself* in-process:
  a third pattern, where one window gives you both the API and a playground.

For a new project, start from a [template](../templates/README.md) rather than copying a sample:

```bash
dotnet new localgen-console -n MyAssistant
dotnet new localgen-webapi  -n DocumentService
dotnet new localgen-desktop -n ChatApp
```
