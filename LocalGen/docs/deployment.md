# Deployment

LocalGen runs on a laptop, a server, or in a container. The constant is that models are large and
slow to fetch, so wherever it runs, the model directory should outlive the process.

## Laptop

```bash
localgen serve
```

Binds to `127.0.0.1:11434`. Nothing else on the network can reach it, so no key is needed.

To start with the service already running, add the desktop app to your login items — it hosts the
service in-process, so one window gives you both the API and the Playground.

## Server on a private network

```bash
localgen serve --bind 0.0.0.0 --port 11434 --api-key "$(openssl rand -hex 24)"
```

Once LocalGen is reachable from other machines, set a key. An open inference endpoint gives away
the machine's compute — and, if the tool functions are on, the machine itself.

Turn code execution off for anything shared:

```json
{ "LocalGen": { "Tools": { "CodeExecution": false } } }
```

### systemd

```ini
# /etc/systemd/system/localgen.service
[Unit]
Description=LocalGen inference server
After=network.target

[Service]
Type=simple
User=localgen
Environment=LOCALGEN_HOME=/var/lib/localgen
Environment=LocalGen__Server__Host=0.0.0.0
EnvironmentFile=/etc/localgen/secrets.env
ExecStart=/usr/local/bin/localgen serve
Restart=on-failure
RestartSec=10

# The tool functions run code; keep the rest of the filesystem out of reach.
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
ReadWritePaths=/var/lib/localgen

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now localgen
```

## Docker

```bash
docker build -t localgen:latest .

docker run -d --name localgen \
  -p 11434:11434 \
  -v localgen-data:/data \
  -e LocalGen__Server__ApiKey="$(openssl rand -hex 24)" \
  localgen:latest
```

The named volume matters: without it, every container restart re-downloads the models.

### With a GPU

```bash
docker build --build-arg LLAMA_BACKEND=cuda12 -t localgen:cuda .

docker run -d --gpus all \
  -p 11434:11434 -v localgen-data:/data \
  -e LocalGen__Engine__Device=Cuda \
  localgen:cuda
```

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/).

### Compose

`docker-compose.yml` brings up LocalGen, Qdrant for RAG, and the web UI:

```bash
export LOCALGEN_API_KEY=$(openssl rand -hex 24)
docker compose up -d
docker compose --profile gpu up -d      # GPU variant instead
```

The image binds to `0.0.0.0` because a container's loopback is not reachable from outside it.
Publish the port only where you intend it to be reachable.

## Kubernetes

```bash
kubectl create namespace localgen
kubectl -n localgen create secret generic localgen-secrets \
  --from-literal=api-key="$(openssl rand -hex 24)"
kubectl -n localgen apply -f deploy/kubernetes.yaml
```

The manifest covers the parts that are easy to get wrong:

- **A PersistentVolumeClaim for models.** Re-downloading gigabytes on every pod start is not
  viable.
- **A generous startup probe.** A cold start loads a multi-gigabyte model from disk; a normal
  liveness probe would kill the pod first.
- **A memory limit above the model size plus its KV cache.** Too low and the pod is OOM-killed
  mid-load.
- **One replica.** Generation is serialised per model and the volume is ReadWriteOnce, so LocalGen
  scales by adding nodes with their own volumes, not replicas against one.
- **A NetworkPolicy** so only the web UI can reach the inference service.
- **Code execution off**, non-root, all capabilities dropped.

For GPUs, use the CUDA image, add `nvidia.com/gpu: 1` to the limits, and schedule onto a GPU node
pool.

### Helm

The chart in `deploy/helm/localgen` is the same deployment with the awkward parts parameterised.
The defaults install a single CPU pod with a model volume and no ingress:

```bash
helm install localgen deploy/helm/localgen --namespace localgen --create-namespace
```

A GPU node pool, metrics and two metered keys:

```bash
helm install localgen deploy/helm/localgen -n localgen --create-namespace \
  --set gpu.enabled=true --set gpu.count=1 \
  --set engine.device=Cuda \
  --set telemetry.prometheus.enabled=true \
  --set telemetry.prometheus.serviceMonitor.enabled=true \
  --set auth.keys[0].name=research --set auth.keys[0].requestsPerMinute=120 \
  --set auth.keys[1].name=batch --set auth.keys[1].maxConcurrentRequests=2
```

Worth knowing before you install:

- **`gpu.enabled` also changes the image.** The CPU image cannot use an accelerator even when the
  device plugin has attached one, and the only symptom is that inference is slow. The chart
  switches to the CUDA tag for you.
- **Keys are generated if you do not supply them,** and pinned on upgrade by reading back the
  existing Secret — otherwise every `helm upgrade` would mint new keys and break every client.
  Read one with:

  ```bash
  kubectl -n localgen get secret localgen-auth -o jsonpath='{.data.api-key}' | base64 -d
  ```

- **The model volume outlives the release.** It carries `helm.sh/resource-policy: keep`, so
  `helm uninstall` will not silently delete a hundred gigabytes of weights. Remove the PVC by hand
  when you actually mean to.
- **Updates use `Recreate`, not a rolling update.** The volume is ReadWriteOnce, so a new pod
  cannot mount it until the old one has released it; a rolling update would deadlock.
- **`telemetry.prometheus.serviceMonitor.enabled` needs the Prometheus Operator's CRDs.** Without
  them the install fails on an unknown kind.

## Offline and edge

```bash
# On a connected machine
localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF
localgen pull huggingface:nomic-ai/nomic-embed-text-v1.5-GGUF

# Copy the data directory to the target, then
localgen serve --offline
```

Offline mode blocks every outbound call. Inference, RAG over an existing index, and stdio MCP
servers keep working; model downloads, web search, scraping and the skill gallery fail with a
clear message.

Models can also be moved by hand: drop a `.gguf` file into the models directory and LocalGen picks
it up on the next scan, reading its metadata from the file header.

## Sizing

| Model | Quantization | Disk | Memory in use |
| --- | --- | --- | --- |
| 3B | Q4_K_M | ~2 GB | ~3 GB |
| 7B | Q4_K_M | ~4.5 GB | ~6 GB |
| 7B | Q8_0 | ~8 GB | ~10 GB |
| 14B | Q4_K_M | ~9 GB | ~11 GB |
| 32B | Q4_K_M | ~20 GB | ~24 GB |

Memory includes the KV cache at a moderate context. A larger context costs more; LocalGen shows an
estimate per model on the Models screen.

On a GPU, the model needs to fit in VRAM to be worth offloading. When it does not, lower
`GpuLayers` so only part of it is offloaded — partial offload is still faster than none.

## More than one GPU

Two 12 GB cards will hold a model that neither of them holds alone. Check first that the backend
can see both — the number that matters is not how many cards are in the machine but how many the
loaded build registered:

```bash
localgen engines
```

```
GPUs   CUDA0, CUDA1 (splittable)
```

One entry means there is nothing to split across: a CPU-only build, a driver that bound one card,
or `CUDA_VISIBLE_DEVICES` narrowing the set. Fix that before configuring anything.

With both visible, the default is already right for identical cards — llama.cpp divides the model
in proportion to free VRAM. Configure a split only when the cards differ or when one of them has
another job:

```json
{
  "LocalGen": {
    "Engine": {
      "Device": "Cuda",
      "TensorSplit": [0.7, 0.3],
      "SplitMode": "Layer",
      "MainGpu": 0
    }
  }
}
```

| Setting | What it does |
| --- | --- |
| `TensorSplit` | Relative weight per GPU, in device order. `[0.7, 0.3]` and `[70, 30]` are the same split. A zero keeps a card out of inference entirely — which is how you leave the display GPU alone. |
| `SplitMode` | `Layer` gives each card a contiguous run of layers, and is the right default. `Row` splits every tensor across all the cards so they compute together; it lowers single-request latency but exchanges activations at every layer, so without NVLink or equivalent it is usually slower. `None` keeps the whole model on `MainGpu`. |
| `MainGpu` | The card holding the KV cache and the tensors that are not split. Point it at the card with room to spare when the two differ in size. |

The same settings exist as Modelfile parameters (`tensor_split`, `split_mode`, `main_gpu`) when one
model needs a different split from the server default, and on the Engine screen of the Admin
Control, which shows what the typed split resolves to before a model is loaded with it.

A mismatch between the split and the hardware is reported rather than obeyed, because llama.cpp's
own behaviour is to accept it silently:

- A split naming more GPUs than are present has the extra entries dropped, with a warning naming
  both counts.
- A split shorter than the device list leaves the remaining cards empty, and says which.
- A split on a single-GPU host is ignored, and says so.

Under Helm the same mistakes fail the install instead, since a chart knows how many GPUs the pod
has been asked for:

```bash
helm install localgen deploy/helm/localgen -n localgen \
  --set gpu.enabled=true --set gpu.count=2 \
  --set engine.device=Cuda \
  --set 'engine.tensorSplit={0.6,0.4}' \
  --set engine.splitMode=layer
```

Two cards do not double throughput. Layer splitting runs them in sequence — card 0 computes its
layers, hands the activations to card 1, and waits — so what it buys is capacity, not speed. The
speed-up comes from being able to run a model that would otherwise have been swapping against
system RAM.

## Behind a reverse proxy

Streaming needs buffering turned off, or responses arrive all at once at the end:

```nginx
location / {
    proxy_pass http://127.0.0.1:11434;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
}
```

LocalGen already sends `X-Accel-Buffering: no`, which nginx honours, but the explicit settings
cover proxies that do not.

## Health and metrics

- `GET /api/health` — liveness, never requires a key
- `GET /api/status` — uptime and resident models
- `GET /api/metrics` — throughput, latency, token counts, CPU and GPU, as JSON for the dashboard
- `GET /api/usage` — per-key quota consumption and cache hit rate

### Prometheus and OpenTelemetry

Both exporters are off by default, so a desktop install opens no extra port and dials nothing.

```jsonc
"Telemetry": {
  "PrometheusEnabled": true,       // serves the exposition format at MetricsPath
  "MetricsPath": "/metrics",
  "RequireApiKeyForMetrics": false, // scrapers inside a cluster rarely have one
  "OtlpEndpoint": "http://collector:4317",
  "OtlpProtocol": "grpc",          // or httpprotobuf
  "Traces": true
}
```

The exported series come from the same recording the dashboard reads — there is no second
measurement path to drift out of step:

| Series | What it is |
| --- | --- |
| `localgen_requests_total` | Requests, tagged by model, engine, status and key |
| `localgen_tokens_prompt_total`, `localgen_tokens_completion_total` | Tokens in and out |
| `localgen_ttft_milliseconds` | Time to first token |
| `localgen_throughput` | Generation throughput |
| `localgen_cache_hits_total`, `localgen_cache_misses_total`, `localgen_cache_entries` | Cache behaviour |
| `localgen_tenant_tokens_today`, `localgen_tenant_requests_per_minute`, `localgen_tenant_in_flight` | Per-key consumption |

Every tag is drawn from a bounded set — installed models, compiled-in engines, configured key
names — so the series cannot fan out with traffic. ASP.NET Core and .NET runtime instrumentation
are included alongside. Traces are exported only when an OTLP endpoint is configured, since
Prometheus has nowhere to put a span; scrapes are filtered out of them.

## API keys and quotas

One key, the common case:

```jsonc
"Server": { "ApiKey": "sk-…" }
```

Several callers sharing one instance, each with its own limits. A zero is unmetered, which is the
default — adding a key to share access does not silently start throttling it:

```jsonc
"Server": {
  "ApiKeys": [
    { "Name": "research", "Key": "sk-…", "RequestsPerMinute": 120, "TokensPerDay": 2000000 },
    { "Name": "batch",    "Key": "sk-…", "MaxConcurrentRequests": 2 },
    { "Name": "demo",     "Key": "sk-…", "AllowedModels": ["smollm2-135m-instruct:q4_k_m"] }
  ]
}
```

A refused request returns 429 in the OpenAI error shape with `Retry-After` set, or 403
`model_forbidden` for a model outside the key's allow-list. Two details worth knowing:

- **The token budget only gates endpoints that generate.** A key that has spent its allowance can
  still list models and read `GET /api/usage` — otherwise it could not find out why it is being
  refused.
- **A cache hit is not charged tokens.** The daily budget measures compute actually performed. The
  per-minute request limit still applies, so a cached prompt cannot be replayed without limit.

## Batched inference

By default a model serves one request at a time: a llama.cpp context decodes a single sequence,
so concurrent requests queue. Batching puts them on one shared context and decodes them together.

```jsonc
"Engine": {
  "BatchedInference": true,
  "MaxBatchedSequences": 4   // null follows Server.MaxConcurrentRequests
}
```

Measured on `qwen2.5-1.5b-instruct:q4_k_m`, four concurrent requests of 128 tokens each:

| | Serialised | Batched |
| --- | ---: | ---: |
| Wall clock | 24.3 s | 8.6 s |
| Aggregate throughput | 20.7 tok/s | 59.3 tok/s |

The gain comes from amortising the weight read: most of the time spent generating one token is
spent moving the model through memory, and a batch pays that once for every sequence in it. The
second concurrent request is very nearly free.

What it costs, and when not to turn it on:

- **The context window is shared.** Four sequences against an 8,192-token context have roughly
  2,048 tokens each. If your callers hold long conversations, either raise `Engine.ContextSize` or
  lower `MaxBatchedSequences`. A request that runs out of room is refused with a message naming
  both settings rather than failing obscurely.
- **It does nothing for a single caller.** One request at a time decodes at the same speed either
  way; batching only pays when requests overlap.
- **Vision models ignore it.** An image is encoded by the projector before the batch is formed, so
  the dominant cost is not something batching can overlap. Those models stay serialised.

## Response caching

Off by default, because a cache changes observable behaviour:

```jsonc
"Cache": { "Enabled": true, "MaxEntries": 256, "Ttl": "00:30:00", "CacheNonDeterministic": false }
```

The key covers the model, every message (image bytes included), the tool schemas and every
sampling setting, so nothing that changes the output can be missed. Only reproducible requests are
cached — `temperature: 0` or an explicit seed — unless `CacheNonDeterministic` is set: a caller
who chose a temperature asked for variety, and replaying one answer forever would quietly take
that away. Entries are shared across keys, which is safe because an entry is a pure function of
the prompt the caller supplied.

A hit is checked before a model session is leased, so it neither queues behind the per-model lock
nor keeps a model resident that would otherwise be evicted.
