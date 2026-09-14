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

`sloop-harness` handles the other half. It holds a conversation as a tree of
content blocks, streams turns into it from the Messages API, forks a branch at
a block boundary and regenerates it -- and writes the whole session as markdown
into a root the daemon already watches. Four seconds later it is retrievable.
Nothing else connects them: the harness writes a file, the watcher indexes it,
and the daemon stays the only writer to the index.

Feeding a transcript back is only safe if a rejected idea cannot come back as a
fact. This project spent a while believing the answer was a kept / abandoned
label on every node, and that turned out to be wrong. The unsafety was never
the missing label; it was stripping a block out of its context and presenting
it as an assertion. A label is metadata, the text is what gets read, and every
consumer has to remember to join them.

What replaces it is structural, and it is cheaper. A branch that was tried and
dropped renders under a `### Not continued` heading, and the chunker turns
headings into the `heading_path` it carries on every chunk. That path goes into
the pointer the prompt hook injects, and into the text that gets embedded and
matched. So a hit inside a discarded branch says so in every retrieval path,
and no code has to remember to make it say so. The heading is the label, and it
cannot be stripped by a renderer that forgets it exists.

The tree keeps its kept / abandoned labels for the harness's own use -- which
branch is live, what to display -- and the indexer does not read them.
[`docs/plans/2026-09-13-transcript-indexing-design.md`](docs/plans/2026-09-13-transcript-indexing-design.md)
records the argument, including what was tried first.

## The parts

| Crate | |
|---|---|
| [`sloop-memory-core`](crates/sloop-memory-core) | Library. Chunking, embedding, index building, hybrid retrieval, and the daemon's socket protocol. |
| [`sloop-memory`](crates/sloop-memory) | Binary. A resident daemon with a filesystem watcher, an MCP server, a CLI, and a Claude Code prompt hook. |
| [`sloop-harness`](crates/sloop-harness) | Binary. A branching agent harness over the Anthropic Messages API. Holds the conversation tree, streams turns into it, and writes each session to the `transcripts` root for the daemon to index. No tools yet. |

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
