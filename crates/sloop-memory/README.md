# sloop-memory

Hybrid local search over private markdown, as a resident daemon with three front
ends: an MCP server, a CLI, and a `UserPromptSubmit` prompt hook for Claude Code.

The retrieval itself lives in [`sloop-memory-core`](../sloop-memory-core). This
crate is everything that touches the outside world -- the daemon and its
filesystem watcher, the MCP stdio server, the clap CLI, and the hook path.

## Installation

    nix profile install github:sloop-ai/sloop#sloop-memory

That gives you a `sloop-memory` with `ORT_DYLIB_PATH` and `SLOOP_MEMORY_MODEL`
already baked in, which is what the "run the nix-wrapped binary" error message
refers to. `nix run github:sloop-ai/sloop -- status` runs it without
installing. Building with plain `cargo build` also works, but the resulting
binary has neither of those two values and will refuse to start until you
supply both yourself.

Then four steps. The order matters: the daemon has to be up before the hook or
the MCP server is worth registering.

**1. Point it at some markdown.** This is the one value nothing can infer:

    export SLOOP_MEMORY_ROOTS="notes=$HOME/notes"

**2. Build the index.** First run embeds everything, so expect it to take a
while -- roughly 40s per 900 chunks on an Apple Silicon Mac:

    sloop-memory index

**3. Run the daemon under a supervisor.** Copy the unit for your platform and
replace the paths marked in it:

| Platform | File | Load with |
|---|---|---|
| macOS | [`nix/launchd/ai.sloop.memory.plist`](../../nix/launchd/ai.sloop.memory.plist) | `launchctl load ~/Library/LaunchAgents/ai.sloop.memory.plist` |
| Linux | [`nix/systemd/sloop-memory.service`](../../nix/systemd/sloop-memory.service) | `systemctl --user enable --now sloop-memory` |

Both spell out `SLOOP_MEMORY_ROOTS` inline, because neither launchd nor a
systemd user unit inherits your shell environment.

**4. Register the front ends.** The MCP server reaches the daemon over the same
socket the hook uses, and Claude Code spawns it with none of your shell
environment, so name the socket explicitly:

    claude mcp add sloop-memory \
      -e SLOOP_MEMORY_SOCKET="$HOME/.local/state/sloop-memory/daemon.sock" \
      -- sloop-memory mcp

It needs no `SLOOP_MEMORY_ROOTS`: the daemon owns the index, and the MCP server
only asks it questions.

For the prompt hook, see [Installing the prompt hook](#installing-the-prompt-hook)
below.

### Check it worked

    sloop-memory status

`status` prints a row count per root and exits non-zero if any root failed to
report one. Do not skip it. The hook fails open by design -- if the daemon is
not running, it exits 0 with no output and your prompts go through unchanged.
That is the right behaviour for a hook and a bad failure mode for an install,
because a completely broken setup looks exactly like a working one from inside
Claude Code. `status` is the only thing that tells them apart.

## Roots

A *root* is one directory of markdown that `sloop-memory` indexes. Configure as
many as you like through `SLOOP_MEMORY_ROOTS`, each with a label. Every root
goes into the same index; the label namespaces each file's identity, so two
roots holding the same relative path never collide.

One label is special. A root labeled exactly `memory` is framed to the model as
recorded facts rather than dated notes, on the grounds that a recorded fact has
nothing live to re-check it against. Every other label gets the notes framing,
which tells the model to treat the pointer as a snapshot and re-verify against
the live source.

## Usage

    sloop-memory index [--full]      # ingest; re-embeds only changed files
    sloop-memory search QUERY [--k N] [--filter SQL] [--json]   # default k=3
    sloop-memory daemon              # foreground; run under a process supervisor
    sloop-memory recall              # hook mode, reads the payload on stdin
    sloop-memory mcp                 # MCP stdio server
    sloop-memory status              # row counts and paths

`--filter` takes a SQL predicate pushed down before the scan over any indexed
column, e.g. `--filter "rel_path LIKE 'notes/%'"` or
`--filter "note_type = 'system'"`.

`status` exits non-zero if any root failed to report a count, so a monitoring
job can tell "this root has no rows" from "this root did not answer".

## Note frontmatter

YAML frontmatter is stripped from the chunk text and indexed as filterable
columns instead. Three keys are read; everything else in the block is ignored.

| Key | Column | Meaning |
|---|---|---|
| `type` | `note_type` | Free-form. There is no taxonomy -- whatever you write is stored verbatim and can be filtered on. `system` in the example below is one author's convention, not a defined value. |
| `captured` | `captured` | The date this note's contents were true, set by whoever wrote the note. Nothing derives or checks it; it is not the file mtime. |
| `tags` | `tags` | A list, inline (`[a, b]`) or block form. |

    ---
    type: system
    captured: 2026-04-10
    tags: [infra, caching]
    ---

`captured` is what makes the pointer design work: the hook injects paths,
headings and `captured` dates rather than note text, so the model can see how
old a claim is before deciding to trust it. Notes without the key still index
normally -- the column is empty, and the hook renders `captured: unknown`.

## Configuration

Six environment variables, all read at startup.

| Variable | What it does | Default |
|---|---|---|
| `SLOOP_MEMORY_ROOTS` | Colon-separated `label=path` pairs naming every root to index. | none -- required; startup fails without it |
| `SLOOP_MEMORY_MODEL` | Directory holding the embedding model and its tokenizer. | baked into the nix package; otherwise none, and startup fails without it |
| `SLOOP_MEMORY_STATE` | State directory: the LanceDB database, the socket, the injection log. | `$XDG_STATE_HOME/sloop-memory`, else `~/.local/state/sloop-memory` |
| `SLOOP_MEMORY_SOCKET` | Daemon's Unix socket path. | `<state>/daemon.sock` |
| `SLOOP_MEMORY_INJECTION_LOG` | Where hook injections are appended as JSON lines. | `<state>/injections.jsonl` |
| `SLOOP_MEMORY_EMBED_THREADS` | ONNX intra-op thread count. | `available_parallelism()` capped at 8 |

Not every command needs every variable. `index`, `search`, `daemon` and
`status` read the index themselves and need `SLOOP_MEMORY_ROOTS` and
`SLOOP_MEMORY_MODEL`. `recall` and `mcp` only talk to the daemon over the
socket, so they need `SLOOP_MEMORY_SOCKET` and nothing else -- which matters,
because Claude Code spawns both with none of your shell environment.

A roots spec looks like `notes=/path/to/notes:memory=/path/to/memory`. Labels
and paths are validated at daemon startup: a label may not contain `:` or `=`,
since both are separators, and every path must already exist. An empty entry --
a leading, trailing or doubled `:` -- is rejected rather than skipped, because
skipping it silently indexes fewer roots than the operator wrote.

## MCP tools

The MCP server exposes two tools:

| Tool | |
|---|---|
| `sloop_search` | Hybrid search over every root. Takes `query`, optional `k` and `filter`. Unlike the hook, it returns full chunks. |
| `sloop_status` | Row counts and paths, broken down by root. Checks whether the index is populated or stale. |

## Installing the prompt hook

`sloop-memory recall` is a Claude Code `UserPromptSubmit` hook: Claude Code
writes the prompt payload to its stdin, and it writes back a JSON
`hookSpecificOutput` block that is prepended to the turn. Register it in
`~/.claude/settings.json`, or in a project's `.claude/settings.json` to scope it
to one repository:

    {
      "hooks": {
        "UserPromptSubmit": [
          {
            "hooks": [
              {
                "type": "command",
                "command": "SLOOP_MEMORY_SOCKET=$HOME/.local/state/sloop-memory/daemon.sock sloop-memory recall"
              }
            ]
          }
        ]
      }
    }

`UserPromptSubmit` takes no matcher; it fires on every prompt. Set the socket
explicitly, and to the same path the daemon is listening on -- a hook does not
inherit an interactive shell's environment, so the two agreeing by accident is
not something to rely on. Use an absolute path to the binary if it is not on the
PATH the hook runs with.

The hook talks to a running `sloop-memory daemon` over that socket; it never
loads the embedding model itself, which is what keeps it inside its 250ms
budget. If the daemon is not running, or anything else fails, the hook exits 0
with no output and the prompt goes through unchanged.

## Design notes

- **Private local memory only.** Nothing is sent anywhere. The roots are
  whatever markdown trees you point it at, and they stay on the machine.
- **Pointers, not bodies.** The hook injects paths, headings and `captured`
  dates, wrapped in a `<sloop-recall>` block. Notes are dated snapshots, and
  injecting their text invites answering from a stale snapshot instead of
  checking the live source.
- **Pointers win.** If the hook already supplied a path for a query, read it
  directly. `sloop_search` returns full chunks and is for a new query or a
  filtered search, not for replaying the hook and duplicating content in
  context.
- **Framing follows the label, not the content.** Which preamble a pointer gets
  is decided by its root label alone. Two pointers with identical text get
  different framing if they came from different roots, and that is the point.
- **User turns only.** Task-notification and local-command envelopes are skipped
  before embedding, so subagent plumbing cannot consume the three-pointer
  budget.
- **Fail open.** Any error on the hook path exits 0 with no output. A dead
  daemon degrades to "no injection", never to a broken prompt.
- **Watched, not polled.** The daemon watches its roots over FSEvents with a 4s
  debounce, sweeps once at startup to catch anything missed while it was down,
  and sweeps hourly as a backstop, because FSEvents legitimately drops events
  across sleep and volume remounts. `.obsidian` is excluded: Obsidian rewrites
  `workspace.json` on every pane change, which would otherwise reindex forever.
- **Every injection is logged** to `injections.jsonl`, so a hook quietly feeding
  stale pointers is auditable rather than invisible.

## Rough edges

- `SLOOP_MEMORY_SOCKET` must stay under 104 bytes (`SUN_LEN`). This surfaces as
  a named error rather than an opaque bind failure, but it is still a
  constraint.
- `:` is a legal filename character on macOS, so a root at `/Users/x/My: Notes`
  cannot be expressed in `SLOOP_MEMORY_ROOTS` at all. Rename the directory or
  point a colon-free symlink at it.

## Build and test

See [the workspace README](../../README.md). `nix develop -c cargo test -p
sloop-memory` runs this crate's tests alone.
