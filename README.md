# sloop

> That is, despite one's sense of departing ever further from one's origin, one
> winds up, to one's shock, exactly where one had started out. In short, a
> strange loop is a paradoxical level-crossing feedback loop.
>
> -- Douglas Hofstadter, *I Am a Strange Loop*

sloop is an experiment in memory for AI coding harnesses.

A coding agent forgets everything between sessions and re-derives the same
conclusions. The usual answers -- a longer context window, a notes file, a
retrieval bolt-on -- all treat memory as storage: put something in, get it back
out. sloop treats it as a loop instead. What the agent works out gets written
down, what is written down shapes the next prompt, and the next prompt is what
the agent works out from. Nothing leaves the machine.

## The loop

`sloop-memory` handles the retrieval half, and works today. Every prompt is
embedded and matched against the indexed roots; anything clearing a cosine
threshold is surfaced as a pointer before the model sees the turn. The trigger
is mechanical rather than a judgment call about whether a question "sounds like"
it needs the notes -- that kind of trigger fails silently on questions that do
not announce themselves.  This is deliberately separate so it can be embedded
into any harness.

`sloop-harness` is meant to handle the other half. Feeding a transcript back
into the index is only safe if you can tell what the transcript *concluded*
from what it *rejected*. A branching harness records which branches were kept
and which were abandoned, so a labeled rejection stays distinguishable from a
conclusion -- and that label is what would make discarded reasoning safe to
index, rather than turning every rejected idea into a retrievable fact.

It holds a conversation tree today: nodes are content blocks, each carries a
kept / abandoned / pending label, and any branch replays to a `messages[]`
array. What it does not yet have is the client that would send one.

## The parts

| Crate | |
|---|---|
| [`sloop-memory-core`](crates/sloop-memory-core) | Library. Chunking, embedding, index building, hybrid retrieval, and the daemon's socket protocol. |
| [`sloop-memory`](crates/sloop-memory) | Binary. A resident daemon with a filesystem watcher, an MCP server, a CLI, and a Claude Code prompt hook. |
| [`sloop-harness`](crates/sloop-harness) | Binary. A branching agent harness over the Anthropic Messages API. Has the conversation tree; no client yet. |

The library/binary split exists so a consumer can link the engine directly and
call it in-process instead of standing up a daemon and talking to it over a Unix
socket. `sloop-harness` does exactly that. The daemon is one way to use the
engine, not the only way.

Each crate's README covers its own surface: `sloop-memory-core` for how
retrieval works and what it costs, `sloop-memory` for configuration, the CLI,
the MCP tools and the hook.

## Install

`sloop-memory` is the part that works today:

    nix profile install github:sloop-ai/sloop#sloop-memory
    export SLOOP_MEMORY_ROOTS="notes=$HOME/notes"
    sloop-memory index
    sloop-memory status

That is enough to search from the CLI. Running it as a daemon, registering the
MCP server, and installing the prompt hook are covered in
[`crates/sloop-memory`](crates/sloop-memory#installation).

## Build and test

    nix develop
    cargo test

The toolchain is pinned, not inherited. `rust-toolchain.toml` names the exact
Rust release; the flake feeds that file to rust-overlay, so cargo, rustc,
clippy and rustfmt all come from one release, and `nix build` compiles with the
same compiler `nix develop` hands you. A contributor without nix gets the same
release, because rustup reads `rust-toolchain.toml` directly.

Every component has to come from a single source. Mixing them -- nixpkgs'
clippy alongside a rustup cargo -- gave clippy-driver a different rustc
identity from the cargo doing the building, and every cached dependency
`.rmeta` was rejected with E0514.

The shell also supplies `protoc` (build-time codegen for lance-encoding's
`.proto` files, not something linked against), `ORT_DYLIB_PATH` pointing at
nixpkgs' onnxruntime (`ort` uses `load-dynamic`, so it is dlopened rather than
linked), and `SLOOP_MEMORY_MODEL` pointing at the pinned model weights in the
nix store. `direnv allow` puts the same environment in an interactive shell.

Supported systems are `aarch64-darwin`, `aarch64-linux` and `x86_64-linux`.
`x86_64-darwin` is absent because nixpkgs-unstable dropped it as of 26.11.

## License

MIT. See [LICENSE](LICENSE).
