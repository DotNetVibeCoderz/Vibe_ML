import * as vscode from 'vscode';
import { LocalGenClient } from './client';

/**
 * A chat panel backed by a local model.
 *
 * Rendered as a webview rather than an output channel so answers can stream into a scrolling
 * transcript. The page carries no external assets — it uses VS Code's own theme variables, so it
 * follows the editor's theme without shipping a stylesheet.
 */
export class ChatPanel {
    private static current: ChatPanel | undefined;

    private readonly disposables: vscode.Disposable[] = [];
    private history: Array<{ role: string; content: string }> = [];
    private generation?: AbortController;

    private constructor(
        private readonly panel: vscode.WebviewPanel,
        private client: LocalGenClient,
        private model: string,
    ) {
        this.panel.webview.html = this.render();

        this.panel.webview.onDidReceiveMessage(
            (message) => this.handle(message),
            undefined,
            this.disposables,
        );

        this.panel.onDidDispose(() => this.dispose(), undefined, this.disposables);
    }

    static show(client: LocalGenClient, model: string, seed?: string): void {
        const column = vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.One;

        if (ChatPanel.current) {
            ChatPanel.current.client = client;
            ChatPanel.current.model = model;
            ChatPanel.current.panel.reveal(column);

            if (seed) {
                void ChatPanel.current.ask(seed);
            }

            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'localgen.chat',
            'LocalGen',
            { viewColumn: column, preserveFocus: false },
            { enableScripts: true, retainContextWhenHidden: true },
        );

        ChatPanel.current = new ChatPanel(panel, client, model);

        if (seed) {
            void ChatPanel.current.ask(seed);
        }
    }

    private handle(message: { type: string; text?: string }): void {
        switch (message.type) {
            case 'ask':
                if (message.text) {
                    void this.ask(message.text);
                }
                break;

            case 'stop':
                this.generation?.abort();
                break;

            case 'clear':
                this.history = [];
                break;
        }
    }

    /** Sends a turn and streams the answer back into the webview. */
    async ask(prompt: string): Promise<void> {
        this.generation?.abort();
        this.generation = new AbortController();

        this.history.push({ role: 'user', content: prompt });

        void this.panel.webview.postMessage({ type: 'user', text: prompt });
        void this.panel.webview.postMessage({ type: 'start' });

        let reply = '';

        try {
            for await (const delta of this.client.streamChat(this.model, this.history, {
                temperature: vscode.workspace.getConfiguration('localgen').get('temperature', 0.3),
                maxTokens: vscode.workspace.getConfiguration('localgen').get('maxTokens', 1024),
                signal: this.generation.signal,
            })) {
                reply += delta;
                void this.panel.webview.postMessage({ type: 'delta', text: delta });
            }

            this.history.push({ role: 'assistant', content: reply });
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);

            // The failed turn is dropped so the next request has no dangling user message.
            this.history.pop();
            void this.panel.webview.postMessage({ type: 'error', text: message });
        } finally {
            void this.panel.webview.postMessage({ type: 'end' });
            this.generation = undefined;
        }
    }

    private render(): string {
        const nonce = Array.from({ length: 32 }, () =>
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.charAt(
                Math.floor(Math.random() * 62),
            ),
        ).join('');

        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; padding: 0;
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-editor-background);
    display: flex; flex-direction: column; height: 100vh;
  }
  #log { flex: 1 1 auto; overflow-y: auto; padding: 14px; }
  .turn {
    max-width: 46rem; margin: 0 0 10px; padding: 9px 12px; border-radius: 5px;
    white-space: pre-wrap; overflow-wrap: anywhere;
    border: 1px solid var(--vscode-panel-border);
    background: var(--vscode-editorWidget-background);
  }
  .turn.user { margin-left: auto; background: var(--vscode-input-background); }
  .turn.error { border-color: var(--vscode-inputValidation-errorBorder);
                background: var(--vscode-inputValidation-errorBackground); }
  .speaker { font-size: 0.75em; opacity: 0.6; margin-bottom: 3px; }
  #composer { display: flex; gap: 8px; padding: 12px 14px;
              border-top: 1px solid var(--vscode-panel-border); }
  textarea {
    flex: 1 1 auto; resize: none; min-height: 38px; max-height: 160px; padding: 7px 9px;
    font-family: inherit; font-size: inherit;
    color: var(--vscode-input-foreground); background: var(--vscode-input-background);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 3px;
  }
  textarea:focus { outline: 1px solid var(--vscode-focusBorder); }
  button {
    padding: 6px 14px; border: none; border-radius: 3px; cursor: pointer;
    color: var(--vscode-button-foreground); background: var(--vscode-button-background);
  }
  button:hover { background: var(--vscode-button-hoverBackground); }
  button.secondary { color: var(--vscode-button-secondaryForeground);
                     background: var(--vscode-button-secondaryBackground); }
  #empty { opacity: 0.6; padding: 24px 14px; }
</style>
</head>
<body>
  <div id="log"><div id="empty">Ask about the code you are working on. Nothing leaves this machine.</div></div>

  <div id="composer">
    <textarea id="prompt" rows="1" placeholder="Ask something — Enter sends, Shift+Enter adds a line"></textarea>
    <button id="send">Send</button>
    <button id="stop" class="secondary" style="display:none">Stop</button>
    <button id="clear" class="secondary">Clear</button>
  </div>

<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const log = document.getElementById('log');
  const prompt = document.getElementById('prompt');
  const send = document.getElementById('send');
  const stop = document.getElementById('stop');
  const clear = document.getElementById('clear');

  let current = null;

  function addTurn(kind, speaker, text) {
    document.getElementById('empty')?.remove();

    const turn = document.createElement('div');
    turn.className = 'turn ' + kind;

    const label = document.createElement('div');
    label.className = 'speaker';
    label.textContent = speaker;

    const body = document.createElement('div');
    body.textContent = text;

    turn.append(label, body);
    log.append(turn);
    log.scrollTop = log.scrollHeight;

    return body;
  }

  function ask() {
    const text = prompt.value.trim();
    if (!text) { return; }
    prompt.value = '';
    vscode.postMessage({ type: 'ask', text });
  }

  send.addEventListener('click', ask);
  stop.addEventListener('click', () => vscode.postMessage({ type: 'stop' }));
  clear.addEventListener('click', () => {
    log.innerHTML = '';
    vscode.postMessage({ type: 'clear' });
  });

  prompt.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
  });

  window.addEventListener('message', (event) => {
    const message = event.data;

    switch (message.type) {
      case 'user':
        addTurn('user', 'You', message.text);
        break;
      case 'start':
        current = addTurn('assistant', 'LocalGen', '');
        send.style.display = 'none';
        stop.style.display = '';
        break;
      case 'delta':
        if (current) {
          current.textContent += message.text;
          log.scrollTop = log.scrollHeight;
        }
        break;
      case 'error':
        addTurn('error', 'Error', message.text);
        break;
      case 'end':
        current = null;
        send.style.display = '';
        stop.style.display = 'none';
        break;
    }
  });
</script>
</body>
</html>`;
    }

    dispose(): void {
        ChatPanel.current = undefined;
        this.generation?.abort();
        this.panel.dispose();

        for (const disposable of this.disposables) {
            disposable.dispose();
        }
    }
}
