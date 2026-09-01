# LocalGen inference server.
#
# Build (CPU):
#   docker build -t localgen:latest .
# Build (NVIDIA):
#   docker build --build-arg LLAMA_BACKEND=cuda12 -t localgen:cuda .

ARG LLAMA_BACKEND=cpu

# ─────────────────────────────  Build  ─────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG LLAMA_BACKEND

WORKDIR /src

# Restore against the project files alone so the layer is reused whenever only source changed.
COPY Directory.Build.props Directory.Packages.props global.json ./
COPY src/LocalGen.Core/*.csproj                 src/LocalGen.Core/
COPY src/LocalGen.Runtime/*.csproj              src/LocalGen.Runtime/
COPY src/LocalGen.Engines.LlamaSharp/*.csproj   src/LocalGen.Engines.LlamaSharp/
COPY src/LocalGen.Engines.Onnx/*.csproj         src/LocalGen.Engines.Onnx/
COPY src/LocalGen.Server/*.csproj               src/LocalGen.Server/

RUN dotnet restore src/LocalGen.Server/LocalGen.Server.csproj -p:LlamaBackend=$LLAMA_BACKEND

COPY src/LocalGen.Core/                 src/LocalGen.Core/
COPY src/LocalGen.Runtime/              src/LocalGen.Runtime/
COPY src/LocalGen.Engines.LlamaSharp/   src/LocalGen.Engines.LlamaSharp/
COPY src/LocalGen.Engines.Onnx/         src/LocalGen.Engines.Onnx/
COPY src/LocalGen.Server/               src/LocalGen.Server/

RUN dotnet publish src/LocalGen.Server/LocalGen.Server.csproj \
    -c Release -o /app --no-restore -p:LlamaBackend=$LLAMA_BACKEND

# ─────────────────────────────  Runtime  ─────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime

# llama.cpp's native library needs a C++ runtime that the ASP.NET image does not carry.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libstdc++6 curl \
 && rm -rf /var/lib/apt/lists/*

# A non-root user, since code execution should never have more privilege than it needs.
RUN useradd --create-home --shell /usr/sbin/nologin localgen

WORKDIR /app
COPY --from=build /app .

# Models are large and must outlive the container.
ENV LOCALGEN_HOME=/data
RUN mkdir -p /data && chown -R localgen:localgen /data /app

USER localgen
VOLUME ["/data"]
EXPOSE 11434

# 0.0.0.0 because the container's loopback is not reachable from outside it. Set
# LocalGen__Server__ApiKey when publishing the port anywhere but a private network.
ENV LocalGen__Server__Host=0.0.0.0 \
    LocalGen__Server__Port=11434 \
    DOTNET_gcServer=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:11434/api/health || exit 1

ENTRYPOINT ["dotnet", "localgen-server.dll"]
