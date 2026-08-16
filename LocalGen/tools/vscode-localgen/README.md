# LocalGen for VS Code

Manage local models and use them from the editor. Nothing you send leaves the machine.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

## What it does

**Model management** — a Models view listing what is installed, with size, quantization and
context length. A filled dot means the model is in memory, which is the difference between an
instant answer and waiting for a multi-gigabyte load. Load, unload, delete, or pull a new model
with progress in the notification area.

**Chat** — a panel that streams answers from the local model, themed with the editor's own colours.

**Editor commands** — right-click a selection:

| Command | Sends |
| --- | --- |
| Explain this code | The selection, asking what it does and what looks fragile |
| Write a doc comment for this | The selection, asking for a comment in the language's convention |
| Explain the error at the cursor | The diagnostic message plus five lines either side |

The last one is the case where a local model earns its place: an error in proprietary code gets
explained without the code being sent anywhere.

**Status bar** — shows whether LocalGen is reachable and which model is resident. Click to open
chat.

## Requirements

A LocalGen server. If the `localgen` CLI is on your PATH, the extension can start one for you —
**LocalGen: Start the server**, or the link in the empty Models view.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `localgen.endpoint` | `http://127.0.0.1:11434` | Server address. Point it at another machine if you like. |
| `localgen.apiKey` | *(empty)* | Bearer token, when the server requires one |
| `localgen.model` | *(empty)* | Model the editor commands use. Empty prefers one already loaded. |
| `localgen.temperature` | `0.3` | Low, because the editor commands are about code |
| `localgen.maxTokens` | `1024` | Cap on generated tokens |
| `localgen.refreshInterval` | `10` | Seconds between model refreshes; `0` disables polling |

Polling exists because the server may be started or stopped elsewhere — from the desktop Admin
Control, or a terminal — and the view should reflect that without being told.

## Build

```bash
cd tools/vscode-localgen
npm install
npm run compile
```

Press <kbd>F5</kbd> in VS Code to launch an Extension Development Host.

To package:

```bash
npx vsce package
code --install-extension localgen-0.1.0.vsix
```

## Notes

The extension carries no runtime dependencies. It talks to LocalGen with `fetch` and reads
server-sent events directly, so the installed size is the compiled JavaScript and nothing else.
