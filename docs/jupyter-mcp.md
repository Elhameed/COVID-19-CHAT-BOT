# Jupyter MCP — live notebook development

Claude Code drives `notebooks/development.ipynb` against a **live kernel**: it edits
cells, executes them, reads results, tracebacks and plots, and iterates. Without this it
can only rewrite the `.ipynb` JSON and never see whether the code runs.

## Running it

Two processes. Start the first, leave it running:

```bash
python scripts/start_jupyter.py          # JupyterLab on :8888
```

The second is started by Claude Code itself from [`.mcp.json`](../.mcp.json) — you don't
run it. On first use Claude Code asks you to approve the project-scoped server; approve
it once. (`claude mcp reset-project-choices` resets that.)

Verify with `/mcp` — the `jupyter` server should be connected.

## How the pieces fit

```
Claude Code ──stdio──► uvx jupyter-mcp-server@latest   (isolated env, not .venv)
                              │
                              │ HTTP + WebSocket, token auth
                              ▼
                       JupyterLab :8888   (.venv, RTC enabled)
                              │
                              ▼
                       ipykernel  ──►  the project venv: pandas, torch+CUDA,
                                       src/, data/kb.parquet
```

### Why the MCP server is not in `requirements.txt`

`jupyter-mcp-server` depends on **unpinned** `fastapi` and `uvicorn`. Installing it into
`.venv` could silently bump `fastapi==0.141.1` / `uvicorn==0.52.3`, which the Phase 6 API
pins deliberately (PRD NFR6). `uvx` runs it in a throwaway environment instead, so the
project venv never sees those dependencies.

`jupyter-collaboration` *is* in `requirements.txt`, because RTC is server-side — the
JupyterLab instance needs it. It pulls 10 packages and upgrades nothing already pinned
(verified). It requires `jupyterlab>=4.6,<5`; we pin 4.6.3.

## The token

`.mcp.json` is committed and **contains no secret**. It references `${JUPYTER_TOKEN}`,
which Claude Code expands from its environment.

| File | Committed | Holds the token |
|---|---|---|
| `.mcp.json` | yes | no — `${JUPYTER_TOKEN}` |
| `.env.example` | yes | no — documentation |
| `.claude/settings.local.json` | **no** | yes, under `env` |
| `.jupyter_token` | **no** | yes, for shell use |

`scripts/start_jupyter.py` generates a token on first run and writes it to
`.claude/settings.local.json`. **Restart Claude Code after that** so the new value is in
its environment when it spawns the MCP server.

To rotate: delete both gitignored files and re-run the script.

## Available tools

18 tools, covering read / insert / edit / delete / execute / inspect:

| Group | Tools |
|---|---|
| Read | `read_notebook`, `read_cell`, `list_notebooks`, `list_files` |
| Edit | `insert_cell`, `insert_execute_code_cell`, `edit_cell_source`, `overwrite_cell_source`, `delete_cell`, `move_cell` |
| Execute | `execute_cell`, `execute_code`, `clear_cell_output` |
| Session | `use_notebook`, `unuse_notebook`, `restart_notebook`, `list_kernels`, `connect_to_jupyter` |

`execute_code` runs against the kernel **without writing to the notebook** — the right
tool for probing state. `execute_cell` runs a real cell and persists its output.

`ALLOW_IMG_OUTPUT=true` is set, so matplotlib figures come back as PNG. Don't force
`matplotlib.use("Agg")` in a cell — that suppresses the inline backend and returns
nothing. The default backend works.

## Notebook authority

**The MCP server owns `notebooks/development.ipynb`.**

Phase 1 built that notebook by generating it with `nbformat` from a script and executing
it with `jupyter execute --inplace`. That approach is now retired. With RTC, notebook
state lives in a YDoc CRDT as well as the file, so writing the file underneath a live
session can be silently overwritten or diverge.

One authority, not two:

- ✅ edit cells through the MCP tools
- ✅ read/execute through the MCP tools
- ❌ don't regenerate the notebook from a script while the server is running
- ❌ don't use the JSON-level notebook editor on it

`src/` modules are still edited normally as files — the constraint applies only to the
notebook. The PRD §7.6 contract is unchanged: reusable logic lives in `src/` and the
notebook imports it.

## Kernels

Two kernelspecs resolve to this venv:

| Spec | argv | Notes |
|---|---|---|
| `python3` | bare `python` | Jupyter's default; correct because the server runs from `.venv` |
| `covid-bot` | absolute `.venv\Scripts\python.exe` | leftover, immune to PATH order |

`use_notebook`'s `kernel_id` takes a **running kernel's UUID** from `list_kernels`, not a
kernelspec name — passing `"covid-bot"` fails. Omit it and a kernel is started
automatically.

Verified live: the kernel reports `sys.executable` inside `.venv`, `torch.cuda.is_available()`
is `True` on the Quadro T2000, and `data/kb.parquet` loads at its 7,077 rows.

## Troubleshooting

| Symptom | Cause |
|---|---|
| MCP server won't connect | JupyterLab isn't running — `python scripts/start_jupyter.py` |
| 403 on every call | Token mismatch; restart Claude Code after regenerating |
| Server stuck "pending approval" | Approve the project server, or trust the workspace |
| Plots return no image | A cell forced `matplotlib.use("Agg")` |
| Kernels accumulating | Each `use_notebook` may start one; `list_kernels` then `restart_notebook`/shut down |
