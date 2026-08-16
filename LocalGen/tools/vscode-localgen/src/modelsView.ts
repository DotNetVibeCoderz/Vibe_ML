import * as vscode from 'vscode';
import { LocalGenClient, ModelInfo, formatBytes } from './client';

/**
 * The Models tree.
 *
 * A model's most important property in this view is whether it is resident, because that is what
 * decides whether the next request is instant or waits for a multi-gigabyte load. The item's
 * icon and context value both carry that state.
 */
export class ModelsProvider implements vscode.TreeDataProvider<ModelNode> {
    private readonly changed = new vscode.EventEmitter<ModelNode | undefined>();
    readonly onDidChangeTreeData = this.changed.event;

    private models: ModelInfo[] = [];
    private loaded = new Set<string>();
    private reachable = false;

    constructor(private client: LocalGenClient) {}

    /** Swaps the client when the endpoint setting changes. */
    setClient(client: LocalGenClient): void {
        this.client = client;
        void this.refresh();
    }

    get isReachable(): boolean {
        return this.reachable;
    }

    get loadedModels(): string[] {
        return [...this.loaded];
    }

    get installedModels(): ModelInfo[] {
        return this.models;
    }

    async refresh(): Promise<void> {
        this.reachable = await this.client.ping();

        if (!this.reachable) {
            this.models = [];
            this.loaded.clear();
        } else {
            try {
                this.models = await this.client.listModels();
                this.loaded = new Set((await this.client.status()).loadedModels);
            } catch {
                // A transient failure should leave the last known list rather than blanking it.
                this.reachable = false;
            }
        }

        // Drives the view's welcome content, which explains how to start the server.
        await vscode.commands.executeCommand('setContext', 'localgen.reachable', this.reachable);
        this.changed.fire(undefined);
    }

    getTreeItem(node: ModelNode): vscode.TreeItem {
        return node;
    }

    getChildren(): ModelNode[] {
        if (!this.reachable) {
            return [];
        }

        if (this.models.length === 0) {
            const empty = new ModelNode('No models installed', vscode.TreeItemCollapsibleState.None);
            empty.description = 'use the download button above';
            empty.contextValue = 'empty';
            empty.iconPath = new vscode.ThemeIcon('info');
            return [empty];
        }

        return this.models.map((model) => {
            const isLoaded = this.loaded.has(model.id);

            const node = new ModelNode(model.id, vscode.TreeItemCollapsibleState.None);

            node.description = [
                model.quantization,
                formatBytes(model.sizeBytes),
                `${model.contextLength.toLocaleString()} ctx`,
            ]
                .filter(Boolean)
                .join(' · ');

            node.tooltip = new vscode.MarkdownString(
                [
                    `**${model.id}**`,
                    '',
                    `| | |`,
                    `|---|---|`,
                    `| Quantization | \`${model.quantization || 'unknown'}\` |`,
                    `| Size | ${formatBytes(model.sizeBytes)} |`,
                    `| Context | ${model.contextLength.toLocaleString()} tokens |`,
                    `| Capabilities | ${model.capabilities.join(', ')} |`,
                    `| State | ${isLoaded ? 'in memory' : 'on disk'} |`,
                ].join('\n'),
            );

            node.contextValue = isLoaded ? 'model-loaded' : 'model-unloaded';
            node.iconPath = new vscode.ThemeIcon(
                isLoaded ? 'circle-filled' : 'circle-outline',
                isLoaded ? new vscode.ThemeColor('charts.green') : undefined,
            );
            node.modelId = model.id;

            return node;
        });
    }

    dispose(): void {
        this.changed.dispose();
    }
}

export class ModelNode extends vscode.TreeItem {
    modelId?: string;
}
