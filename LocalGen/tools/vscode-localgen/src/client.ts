/**
 * A minimal LocalGen client for the extension.
 *
 * Written against `fetch` rather than pulling in an HTTP library: VS Code ships a modern Node
 * runtime, and an extension should not carry dependencies it does not need.
 */

export interface ModelInfo {
    id: string;
    quantization: string;
    sizeBytes: number;
    contextLength: number;
    capabilities: string[];
}

export interface ServerStatus {
    status: string;
    version: string;
    baseUrl: string;
    loadedModels: string[];
}

export class LocalGenError extends Error {
    constructor(message: string, readonly statusCode?: number) {
        super(message);
        this.name = 'LocalGenError';
    }
}

export class LocalGenClient {
    constructor(
        private readonly endpoint: string,
        private readonly apiKey?: string,
    ) {}

    /** Whether the server answers. Never throws — an unreachable server is an expected state. */
    async ping(): Promise<boolean> {
        try {
            const response = await this.request('/api/health', { timeoutMs: 2000 });
            return response.ok;
        } catch {
            return false;
        }
    }

    async status(): Promise<ServerStatus> {
        const body = await this.json<{
            status: string;
            version: string;
            base_url: string;
            loaded_models: Array<{ id: string }>;
        }>('/api/status');

        return {
            status: body.status,
            version: body.version,
            baseUrl: body.base_url,
            loadedModels: body.loaded_models.map((model) => model.id),
        };
    }

    async listModels(): Promise<ModelInfo[]> {
        // The management API serialises snake_case, matching the OpenAI surface it sits beside.
        const body = await this.json<
            Array<{
                id: string;
                quantization: { name: string };
                size_bytes: number;
                context_length: number;
                capabilities: string[];
            }>
        >('/api/models');

        return body.map((model) => ({
            id: model.id,
            quantization: model.quantization?.name ?? '',
            sizeBytes: model.size_bytes ?? 0,
            contextLength: model.context_length ?? 0,
            capabilities: model.capabilities ?? [],
        }));
    }

    async loadModel(id: string): Promise<void> {
        await this.request(`/api/models/${encodeURIComponent(id)}/load`, { method: 'POST' });
    }

    async unloadModel(id: string): Promise<void> {
        await this.request(`/api/models/${encodeURIComponent(id)}/unload`, { method: 'POST' });
    }

    async deleteModel(id: string): Promise<void> {
        await this.request(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
    }

    /**
     * Pulls a model, yielding progress. The server streams because a download runs for minutes;
     * a single response would time out and leave the user with no feedback.
     */
    async *pullModel(
        reference: string,
        signal?: AbortSignal,
    ): AsyncGenerator<{ percentage: number; status: string; done: boolean; error?: string }> {
        const response = await this.request('/api/models/pull', {
            method: 'POST',
            body: JSON.stringify({ reference }),
            signal,
        });

        for await (const frame of readServerSentEvents(response, signal)) {
            if (frame.status === 'error') {
                yield { percentage: 0, status: 'failed', done: true, error: frame.error };
                return;
            }

            if (frame.status === 'success') {
                yield { percentage: 100, status: 'installed', done: true };
                return;
            }

            const total = Number(frame.total_bytes ?? 0);
            const downloaded = Number(frame.bytes_downloaded ?? 0);

            yield {
                percentage: total > 0 ? (downloaded / total) * 100 : 0,
                status: frame.file_name ?? 'downloading',
                done: false,
            };
        }
    }

    /** Streams a chat completion, yielding text as it arrives. */
    async *streamChat(
        model: string,
        messages: Array<{ role: string; content: string }>,
        options: { temperature?: number; maxTokens?: number; signal?: AbortSignal } = {},
    ): AsyncGenerator<string> {
        const response = await this.request('/v1/chat/completions', {
            method: 'POST',
            signal: options.signal,
            body: JSON.stringify({
                model,
                messages,
                temperature: options.temperature,
                max_tokens: options.maxTokens,
                stream: true,
            }),
        });

        for await (const frame of readServerSentEvents(response, options.signal)) {
            const delta = frame?.choices?.[0]?.delta?.content;

            if (typeof delta === 'string' && delta.length > 0) {
                yield delta;
            }
        }
    }

    private async json<T>(path: string): Promise<T> {
        const response = await this.request(path);
        return (await response.json()) as T;
    }

    private async request(
        path: string,
        init: RequestInit & { timeoutMs?: number } = {},
    ): Promise<Response> {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };

        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        // A caller-supplied signal wins; the timeout is only for the reachability probe, where
        // waiting the default is the wrong behaviour.
        const signal =
            init.signal ??
            (init.timeoutMs ? AbortSignal.timeout(init.timeoutMs) : undefined);

        const response = await fetch(`${this.endpoint.replace(/\/+$/, '')}${path}`, {
            ...init,
            headers,
            signal,
        });

        if (!response.ok) {
            throw new LocalGenError(await describeFailure(response), response.status);
        }

        return response;
    }
}

/** Reads a `text/event-stream` body, yielding each parsed `data:` frame. */
async function* readServerSentEvents(
    response: Response,
    signal?: AbortSignal,
): AsyncGenerator<any> {
    if (!response.body) {
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (!signal?.aborted) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });

            // Frames are newline-delimited; the tail may be a partial line, so it stays buffered.
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.startsWith('data:')) {
                    continue;
                }

                const payload = line.slice(5).trim();

                if (payload === '[DONE]') {
                    return;
                }

                try {
                    yield JSON.parse(payload);
                } catch {
                    // A malformed frame must not abort a long stream.
                }
            }
        }
    } finally {
        await reader.cancel().catch(() => undefined);
    }
}

/** Surfaces the server's own error message rather than a bare status line. */
async function describeFailure(response: Response): Promise<string> {
    try {
        const body: any = await response.json();

        if (body?.error?.message) {
            return body.error.message;
        }
    } catch {
        // Not a JSON error body.
    }

    return `${response.status} ${response.statusText}`;
}

export function formatBytes(bytes: number): string {
    if (bytes <= 0) {
        return '—';
    }

    if (bytes < 1024 * 1024) {
        return `${Math.round(bytes / 1024)} KB`;
    }

    if (bytes < 1024 * 1024 * 1024) {
        return `${Math.round(bytes / (1024 * 1024))} MB`;
    }

    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
