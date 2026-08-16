import * as vscode from 'vscode';
import { ChatPanel } from './chatPanel';
import { LocalGenClient } from './client';
import { ModelNode, ModelsProvider } from './modelsView';

let client: LocalGenClient;
let models: ModelsProvider;
let statusBar: vscode.StatusBarItem;
let poller: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext): void {
    client = buildClient();
    models = new ModelsProvider(client);

    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'localgen.openChat';
    statusBar.show();

    context.subscriptions.push(
        statusBar,
        models,
        vscode.window.createTreeView('localgen.models', { treeDataProvider: models }),

        vscode.commands.registerCommand('localgen.refreshModels', () => refresh()),
        vscode.commands.registerCommand('localgen.pullModel', () => pullModel()),
        vscode.commands.registerCommand('localgen.startServer', () => startServer()),
        vscode.commands.registerCommand('localgen.openChat', () => openChat()),
        vscode.commands.registerCommand('localgen.openSettings', () =>
            vscode.commands.executeCommand('workbench.action.openSettings', 'localgen'),
        ),

        vscode.commands.registerCommand('localgen.loadModel', (node: ModelNode) =>
            withModel(node, async (id) => {
                await vscode.window.withProgress(
                    { location: vscode.ProgressLocation.Notification, title: `Loading ${id}…` },
                    () => client.loadModel(id),
                );
                await refresh();
            }),
        ),

        vscode.commands.registerCommand('localgen.unloadModel', (node: ModelNode) =>
            withModel(node, async (id) => {
                await client.unloadModel(id);
                await refresh();
            }),
        ),

        vscode.commands.registerCommand('localgen.deleteModel', (node: ModelNode) =>
            withModel(node, async (id) => {
                // Deleting means re-downloading gigabytes, so this always confirms.
                const answer = await vscode.window.showWarningMessage(
                    `Delete ${id}? The weights will have to be downloaded again.`,
                    { modal: true },
                    'Delete',
                );

                if (answer === 'Delete') {
                    await client.deleteModel(id);
                    await refresh();
                }
            }),
        ),

        vscode.commands.registerCommand('localgen.setDefaultModel', (node: ModelNode) =>
            withModel(node, async (id) => {
                await vscode.workspace
                    .getConfiguration('localgen')
                    .update('model', id, vscode.ConfigurationTarget.Workspace);

                vscode.window.showInformationMessage(`LocalGen will use ${id}.`);
                updateStatusBar();
            }),
        ),

        vscode.commands.registerCommand('localgen.explainSelection', () =>
            askAboutSelection(
                'Explain what this code does, then note anything that looks wrong or fragile.',
            ),
        ),

        vscode.commands.registerCommand('localgen.documentSelection', () =>
            askAboutSelection(
                'Write a doc comment for this, in the language’s own convention. ' +
                    'Describe what it does and why, not how. Reply with the comment alone.',
            ),
        ),

        vscode.commands.registerCommand('localgen.explainProblem', () => explainProblem()),

        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('localgen')) {
                client = buildClient();
                models.setClient(client);
                schedulePolling();
            }
        }),
    );

    void refresh();
    schedulePolling();
}

export function deactivate(): void {
    if (poller) {
        clearInterval(poller);
    }
}

function buildClient(): LocalGenClient {
    const settings = vscode.workspace.getConfiguration('localgen');

    return new LocalGenClient(
        settings.get('endpoint', 'http://127.0.0.1:11434'),
        settings.get('apiKey', '') || undefined,
    );
}

/**
 * Polls so the view reflects a server started or stopped elsewhere — from the desktop app, or
 * from a terminal. Zero disables it for anyone who would rather refresh by hand.
 */
function schedulePolling(): void {
    if (poller) {
        clearInterval(poller);
        poller = undefined;
    }

    const seconds = vscode.workspace.getConfiguration('localgen').get('refreshInterval', 10);

    if (seconds > 0) {
        poller = setInterval(() => void refresh(), seconds * 1000);
    }
}

async function refresh(): Promise<void> {
    await models.refresh();
    updateStatusBar();
}

function updateStatusBar(): void {
    if (!models.isReachable) {
        statusBar.text = '$(circle-slash) LocalGen';
        statusBar.tooltip = 'LocalGen is not running. Click to open chat, or run "LocalGen: Start the server".';
        statusBar.backgroundColor = undefined;
        return;
    }

    const resident = models.loadedModels[0];

    statusBar.text = resident ? `$(sparkle) ${resident}` : '$(sparkle) LocalGen';
    statusBar.tooltip = resident
        ? `${resident} is in memory. Click to open chat.`
        : `${models.installedModels.length} model(s) installed, none loaded. Click to open chat.`;
    statusBar.backgroundColor = undefined;
}

/** Resolves the model the editor commands should use. */
async function resolveModel(): Promise<string | undefined> {
    const configured = vscode.workspace.getConfiguration('localgen').get('model', '');

    if (configured) {
        return configured;
    }

    const installed = models.installedModels;

    if (installed.length === 0) {
        const answer = await vscode.window.showWarningMessage(
            'No models are installed.',
            'Pull a model…',
        );

        if (answer) {
            await pullModel();
        }

        return undefined;
    }

    // Prefer one already in memory — otherwise the first command pays a multi-gigabyte load.
    return models.loadedModels[0] ?? installed[0].id;
}

async function ensureReachable(): Promise<boolean> {
    if (models.isReachable) {
        return true;
    }

    const answer = await vscode.window.showWarningMessage(
        'LocalGen is not running.',
        'Start the server',
    );

    if (answer) {
        await startServer();
    }

    return false;
}

async function openChat(seed?: string): Promise<void> {
    if (!(await ensureReachable())) {
        return;
    }

    const model = await resolveModel();

    if (model) {
        ChatPanel.show(client, model, seed);
    }
}

/**
 * Starts the server in a terminal rather than as a hidden child process: model loading prints
 * progress worth seeing, and the user keeps a handle on something that holds gigabytes of memory.
 */
async function startServer(): Promise<void> {
    const terminal =
        vscode.window.terminals.find((t) => t.name === 'LocalGen') ??
        vscode.window.createTerminal('LocalGen');

    terminal.show();
    terminal.sendText('localgen serve');

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'Waiting for LocalGen…' },
        async () => {
            for (let attempt = 0; attempt < 30; attempt++) {
                await new Promise((resolve) => setTimeout(resolve, 1000));

                if (await client.ping()) {
                    await refresh();
                    return;
                }
            }

            vscode.window.showWarningMessage(
                'LocalGen did not start. Check the terminal — the CLI may not be on PATH.',
            );
        },
    );
}

async function pullModel(): Promise<void> {
    if (!(await ensureReachable())) {
        return;
    }

    const reference = await vscode.window.showInputBox({
        title: 'Pull a model',
        prompt: 'Model reference',
        placeHolder: 'huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF',
        validateInput: (value) =>
            value.trim().length === 0 ? 'Enter a model reference.' : undefined,
    });

    if (!reference) {
        return;
    }

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: `Pulling ${reference}`,
            cancellable: true,
        },
        async (progress, token) => {
            const controller = new AbortController();
            token.onCancellationRequested(() => controller.abort());

            let last = 0;

            try {
                for await (const update of client.pullModel(reference, controller.signal)) {
                    if (update.error) {
                        vscode.window.showErrorMessage(`Pull failed: ${update.error}`);
                        return;
                    }

                    // withProgress takes an increment, not an absolute value.
                    progress.report({
                        increment: update.percentage - last,
                        message: `${update.status} — ${update.percentage.toFixed(0)}%`,
                    });

                    last = update.percentage;

                    if (update.done) {
                        vscode.window.showInformationMessage(`Installed ${reference}.`);
                        await refresh();
                        return;
                    }
                }
            } catch (error) {
                if (!controller.signal.aborted) {
                    const message = error instanceof Error ? error.message : String(error);
                    vscode.window.showErrorMessage(`Pull failed: ${message}`);
                }
            }
        },
    );
}

/** Sends the current selection to the model with an instruction. */
async function askAboutSelection(instruction: string): Promise<void> {
    const editor = vscode.window.activeTextEditor;

    if (!editor) {
        return;
    }

    const selection = editor.document.getText(editor.selection);

    if (!selection.trim()) {
        vscode.window.showInformationMessage('Select some code first.');
        return;
    }

    const language = editor.document.languageId;

    await openChat(`${instruction}\n\n\`\`\`${language}\n${selection}\n\`\`\``);
}

/**
 * Explains the diagnostic under the cursor, with the surrounding code as context — the case where
 * a local model is most obviously worth having, since the code never leaves the machine.
 */
async function explainProblem(): Promise<void> {
    const editor = vscode.window.activeTextEditor;

    if (!editor) {
        return;
    }

    const position = editor.selection.active;

    const diagnostics = vscode.languages
        .getDiagnostics(editor.document.uri)
        .filter((diagnostic) => diagnostic.range.contains(position));

    if (diagnostics.length === 0) {
        vscode.window.showInformationMessage('No error or warning at the cursor.');
        return;
    }

    const diagnostic = diagnostics[0];

    // A few lines either side is usually enough to explain the error without sending the file.
    const start = Math.max(0, diagnostic.range.start.line - 5);
    const end = Math.min(editor.document.lineCount - 1, diagnostic.range.end.line + 5);

    const context = editor.document.getText(
        new vscode.Range(start, 0, end, editor.document.lineAt(end).text.length),
    );

    await openChat(
        [
            'Explain this compiler or linter message and how to fix it.',
            '',
            `Message: ${diagnostic.message}`,
            diagnostic.code ? `Code: ${String(diagnostic.code)}` : '',
            '',
            `\`\`\`${editor.document.languageId}`,
            context,
            '```',
        ]
            .filter(Boolean)
            .join('\n'),
    );
}

async function withModel(node: ModelNode, action: (id: string) => Promise<void>): Promise<void> {
    if (!node?.modelId) {
        return;
    }

    try {
        await action(node.modelId);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(message);
    }
}
