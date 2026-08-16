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
- `GET /api/metrics` — throughput, latency, token counts, CPU and GPU

Metrics are also published to a `System.Diagnostics.Metrics` meter named `LocalGen.Inference`, so
an OpenTelemetry exporter can pick them up without going through the HTTP endpoint.
