# Architecture

How the projects fit together, and why the seams are where they are.

## Dependency shape

```
                        ┌──────────────┐
                        │ LocalGen.Core│   contracts only — no backend types
                        └──────┬───────┘
             ┌─────────────────┼──────────────────┐
             │                 │                  │
      ┌──────▼──────┐   ┌──────▼──────┐    ┌──────▼──────┐
      │  Engines.*  │   │   Runtime   │    │     Sdk     │
      │ LlamaSharp  │   │ store,      │    │ HTTP client │
      │ Onnx        │   │ sessions,   │    └──────┬──────┘
      │ FoundryLocal│   │ metrics     │           │
      └──────┬──────┘   └──────┬──────┘           │
             │                 │                  │
             └────────┬────────┘                  │
                      │                           │
              ┌───────▼────────┐                  │
              │ Kernel   Rag   │                  │
              │ SK, tools,     │                  │
              │ skills, MCP    │                  │
              └───────┬────────┘                  │
                      │                           │
        ┌─────────────┼─────────────┐             │
   ┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐   ┌────▼────┐
   │ Server  │  │  Desktop  │  │   Cli   │   │   Web   │
   └─────────┘  └───────────┘  └─────────┘   └─────────┘
```

Arrows point from dependency to dependent. Nothing above `Core` may reference a backend's types.

## The seams that matter

### `IInferenceEngine` — backends stay interchangeable

Every backend implements the same three operations: probe the machine, say whether it can serve a
model, and load one. Loading returns an `IModelSession` whose only generation primitive is a
stream of chunks; the non-streaming path is an extension method that drains it.

This is why the server, SDK, CLI and both UIs contain no reference to LlamaSharp, ONNX Runtime or
Foundry Local. Adding a backend is a new project and one DI call.

`EngineRegistry` picks between them: an explicit override, then the model's declared engines, then
the configured default, then anything that can serve the format. When nothing works it explains
which of those two cases it hit — "no backend supports this format" is a different problem from
"the right backend is installed but its native library is missing."

### Wire types apart from domain types

`LocalGen.Core.Protocol` holds the OpenAI request and response shapes, in `snake_case`, including
fields LocalGen ignores. `LocalGen.Core.Inference` holds the domain model. `OpenAiMapper`
translates.

Keeping them apart is what lets the internal model change without touching a contract that
external clients depend on. It is deliberate duplication.

### The server is a library

`LocalGenServerHost` builds and controls a `WebApplication`, with start, stop and state. The
executable is a thin wrapper; Admin Control and `localgen serve` drive the class.

That is what lets the desktop app host the service in its own process, so the Playground and the
HTTP API share one set of loaded weights rather than each loading its own copy.

### Sessions are leased, not owned

`ModelSessionManager` caches loaded models and hands out `SessionLease` values. A lease marks the
model busy for as long as the caller holds it — including the whole streaming enumeration, which
outlives the method that started it.

Eviction is least-recently-used and never touches a model with a lease outstanding. Disposal waits
for in-flight requests to drain, because freeing native memory under a running decode loop would
crash the process.

## How a request flows

```
POST /v1/chat/completions
  → OpenAiEndpoints          validate, map wire → domain
  → InferenceService         acquire lease, start metrics
  → ModelSessionManager      cache hit, or select engine and load
  → IModelSession.StreamAsync
      → render the prompt with the model's chat template, or ChatML
      → decode, filtering tool-call tags out of the visible text
      → yield chunks
  ← record one InferenceSample, whatever the outcome
  ← frame as SSE, terminate with [DONE]
```

## Tool calling

Neither llama.cpp nor ONNX Runtime exposes a structured function-call channel, so LocalGen uses a
prompt protocol: tool schemas go into the system prompt, and the model is asked to answer with a
`<tool_call>{…}</tool_call>` block.

The hard part is streaming. A tag arrives split across tokens, so `ToolCallStreamFilter` holds
back any tail that could still grow into an opening tag. Without it the user would see a stray
`<tool` for one token before the rest arrived. Text that merely looks like a tag —
`<toolbox>` — is released normally.

Malformed JSON is surfaced as visible text rather than swallowed, and a call cut off by the token
limit is released on flush. Nothing the model generated is silently dropped.

## The agent loop

`LocalGenAgent` drives a turn: call the model, invoke any tools it asked for, feed the results
back, repeat until it answers in plain text or the iteration budget runs out.

The loop is LocalGen's rather than Semantic Kernel's auto-invocation because it streams an event
per step — call started, call completed with its result and duration — which is what lets the
Playground show the work rather than just the answer.

## Configuration

One options tree, `LocalGenOptions`, bound from the `LocalGen` configuration section and shared by
every host. The CLI layers its flags in as an in-memory configuration source, so a command-line
override follows the same binding rules as a file or an environment variable.

## Testing

Unit tests cover the parts with real logic and no I/O: the Modelfile parser, the tool-call
protocol and its stream filter at every chunk boundary, text chunking, and model reference
parsing. Inference itself is verified by running it against a real model.
