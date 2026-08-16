# Project templates

Three starting points for applications built on LocalGen.

## Install

```bash
dotnet pack templates/LocalGen.Templates.csproj -o ./artifacts
dotnet new install ./artifacts/LocalGen.Templates.0.1.0.nupkg
```

Once published, this becomes:

```bash
dotnet new install LocalGen.Templates
```

## Templates

| Short name | Creates |
| --- | --- |
| `localgen-console` | A console client with streaming chat, one-shot and interactive |
| `localgen-webapi` | An ASP.NET Core API exposing summarise, classify, extract and streaming write |
| `localgen-desktop` | A cross-platform Avalonia chat window |

## Use

```bash
dotnet new localgen-console -n MyAssistant
dotnet new localgen-webapi  -n DocumentService --Model qwen2.5-7b-instruct:q4_k_m
dotnet new localgen-desktop -n ChatApp --Endpoint http://192.168.1.50:11434
```

### Parameters

Every template accepts the same three:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--Endpoint` | `http://127.0.0.1:11434` | Address of the LocalGen server |
| `--Model` | `qwen2.5-7b-instruct:q4_k_m` | Model the application uses |
| `--SdkVersion` | `0.1.0` | Version of `LocalGen.Sdk` to reference |

Values land in both the code and `appsettings.json`, so they can still be changed at runtime
without editing source.

## Note on the web API template

Its endpoints are domain operations — `/summarise`, `/classify`, `/extract`, `/write` — rather
than a chat proxy. That is the shape worth copying for an application service: the prompt stays
your concern and callers see an API in your own vocabulary, not the model's.

`/extract` asks for JSON mode, which on the LlamaSharp backend constrains decoding with a grammar
so the reply cannot be prose wrapped around an object.

## Uninstall

```bash
dotnet new uninstall LocalGen.Templates
```
