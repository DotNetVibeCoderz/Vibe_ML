# Skills and MCP

Two ways to extend what a model can do: **Skills** package instructions with the assets and
scripts that carry them out, and **MCP** attaches external tool servers.

---

## Built-in kernel functions

Seven functions ship with LocalGen, all enabled by default and individually toggleable per
Playground session. Unchecking one removes it from the kernel — a prompt is not a permission
boundary.

| Function | Provides |
| --- | --- |
| **Math** | `evaluate`, `percentage`, `statistics` — a real expression evaluator, so arithmetic is computed rather than guessed |
| **Time and date** | `now`, `today`, `add_time`, `days_between`, `day_of_week`, `convert_timezone` |
| **Internet search** | `web_search` through Tavily, which returns extracted answers rather than links |
| **Web scrape** | `scrape_url`, `extract_links` — the page is parsed and reduced to readable text |
| **Download** | `download_file` into the workspace, size-capped |
| **File system** | `list_directory`, `read_file`, `write_file`, `append_file`, `search_files`, `delete_file` |
| **Code execution** | `execute_code`, `run_command`, `install_dependency`, `check_runtime` |

### Configuration

```json
{
  "LocalGen": {
    "Tools": {
      "CodeExecution": true,
      "AllowDependencyInstall": false,
      "CodeExecutionTimeout": "00:02:00",
      "TavilyApiKey": "tvly-…",
      "AllowedPaths": ["D:\\projects\\notes"]
    }
  }
}
```

### Safety

The file, download and code functions act on paths the model chooses, so every path is resolved
and checked against an allow-list. The LocalGen workspace is always writable; nothing else is
unless you add it to `AllowedPaths`. A relative path resolves inside the workspace.

Code execution runs with the privileges of the LocalGen process. Three things bound it: the
working directory is the workspace, every run is killed at `CodeExecutionTimeout` along with its
child processes, and installing packages needs `AllowDependencyInstall` turned on separately.

**Turn code execution off before exposing LocalGen beyond loopback.** It is the one function that
can change the host machine.

---

## Skills

A skill is a directory. Instructions alone would only be a prompt snippet; bundling templates and
runnable scripts is what lets a skill carry real behaviour.

```
skills/
  weekly-report/
    SKILL.md                  frontmatter + instructions
    assets/
      report-template.md
    scripts/
      collect-metrics.py
```

### `SKILL.md`

```markdown
---
name: weekly-report
description: Produces a weekly engineering report from the metrics database.
version: 1.0.0
author: Gravicode Studios
tags: [reporting, metrics]
---

# Weekly report

## When to use this

The user asks for a weekly report, a sprint summary, or "how did we do this week".

## Steps

1. Run `scripts/collect-metrics.py` with the week number to gather the figures.
2. Read `assets/report-template.md` and fill in each section from that output.
3. Leave any section without data marked "no data this week" rather than inventing figures.
4. Save the result as `weekly-report-<week>.md` in the workspace.
```

Frontmatter keys: `name`, `description`, `version`, `author`, `tags`, `enabled`.

### How the model uses one

Skills are exposed through kernel functions rather than being pasted into the system prompt, so a
user can install dozens without spending context on skills the conversation never touches — the
model sees one line each, and pulls in the full instructions only on use.

| Function | Purpose |
| --- | --- |
| `list_skills` | Names and one-line descriptions |
| `use_skill(name)` | Full instructions, plus an inventory of assets and scripts |
| `read_skill_asset(skill, path)` | Read a bundled template |
| `run_skill_script(skill, script, arguments)` | Run a bundled `.py`, `.js`, `.ps1` or `.sh` |
| `search_skill_gallery(query)` | Browse installable skills |

Asset and script paths are resolved and confined to the skill's own directory. Scripts run through
the code execution tool, so the same timeout and output limits apply.

### Installing

Install from the gallery in the Playground, or drop a folder into the skills directory by hand —
they are read from disk on demand, so a hand-edited skill shows up without a restart.

Archives are extracted with path checks: an entry that would escape the skill directory is
rejected, because skills are downloaded from the internet.

---

## MCP

The Model Context Protocol lets LocalGen use tool servers it does not implement itself — a
database, an issue tracker, an internal API.

### Configuring servers

Servers live in `mcp.json` in the data directory, and can be added from the Playground:

```json
[
  {
    "name": "filesystem",
    "description": "Read and write files in a directory you nominate.",
    "command": "npx",
    "arguments": ["-y", "@modelcontextprotocol/server-filesystem", "D:\\projects"],
    "enabled": true
  },
  {
    "name": "internal-api",
    "description": "Our service catalogue.",
    "url": "https://mcp.internal.example.com/sse",
    "enabled": true
  }
]
```

A server with a `command` is launched over stdio as a child process; one with a `url` is reached
over HTTP.

### The gallery

The Playground offers a curated set so the panel is useful before you know any server names:
filesystem, git, sqlite, fetch, memory and github.

### How they attach

Enabled servers are connected when a conversation starts and their tools registered as a kernel
plugin — one plugin per server, so two servers exposing a `search` tool do not collide. A server
that fails to start is skipped with a warning rather than blocking the others.

Connections are long-lived, since a stdio server is a child process, and are torn down on explicit
disconnect.

In offline mode, HTTP servers are refused; stdio servers still work, because they are local.
