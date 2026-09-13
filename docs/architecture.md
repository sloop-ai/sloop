# Architecture

Why `sloop` is three crates rather than one, and what the boundaries between
them are for. This describes the workspace as it stands; the harness's own
internals are not designed yet.

## The split

    crates/sloop-memory-core/   library: the retrieval engine
    crates/sloop-memory/        binary: daemon, MCP server, CLI, hook client
    crates/sloop-harness/       binary: branching agent harness (scaffold)

The engine is a library so a consumer can **link it and call it in-process**
rather than standing up a daemon and talking to it over a Unix socket. That is
not hypothetical: `sloop-harness` links it directly, and a build failure there
is the signal that the library has stopped exporting enough to be usable
outside its own crate.

The daemon is one way to use the engine, not the only way. Everything that
talks to the outside world -- the resident daemon and its filesystem watcher,
the MCP stdio server, the clap CLI, the prompt hook -- lives in `sloop-memory`.

## The rule that keeps the seam honest

**No module in the library may name the binary.** Every dependency points from
`sloop-memory` into `sloop-memory-core`, never back. The seam is not a
convention to remember; the crate graph enforces it, and violating it does not
compile.

## Module layering

Within the library the `use crate::` graph is acyclic:

    chunk    config    proto        (no internal dependencies)
      |        |  |      |
      |     embed store  |
      |        |  |    client
      +--------+--+
               |
             index                  (chunk, config, embed, store)

`index` sits on top because ingestion is the step that needs everything else:
it chunks a file, hashes it against the manifest, embeds what changed, and
writes through `store`.

Nothing here is a cycle, and that is worth stating because it is easy to
believe otherwise. `config` and `store` both mention `crate::index` in **doc
comments** -- cross-references to `index::source_id` and `index::rel_id`, which
explain why a root label may not contain `:`. Those are prose, not edges.
`store` also imports `lancedb::index` and `lance_index`, which are foreign
crates that happen to share a name with a local module. Read the `use crate::`
lines, not every occurrence of `index`.

## The socket protocol is the daemon's, not the engine's

`proto` and `client` are published from the library because the daemon and its
clients both need them, and they must agree on one definition of the wire
format. They are **not** part of the in-process API.

An in-process consumer that reaches for `proto` has reintroduced exactly the
coupling that linking the library was meant to avoid: it is now speaking a wire
protocol to itself. `sloop-harness` therefore uses `config` and `index` only,
and says so in a comment at the point of temptation.

Put another way: `proto` is how two *processes* agree. If there is only one
process, there is nothing to agree with.

## Where tests live

Tests travel with the code they cover, which puts a real boundary in a place
that might look arbitrary. `sloop-memory`'s tests cover pointer rendering,
status printing, exit codes and envelope filtering -- all **presentation**, so
they belong to the binary even though they are testing behaviour that feels
like it belongs to memory. The engine's tests cover chunking, config parsing,
storage and indexing.

If a test needs a running daemon it is in the wrong place. The engine is
callable in-process precisely so its tests do not need one.

## Three build inputs that are not in Cargo.toml

`cargo build` alone is not sufficient, and the reasons are not discoverable
from the manifests:

| Input | Why it is not a normal dependency |
|---|---|
| `protoc` | Build-time codegen for `lance-encoding`'s `.proto` files. A tool the build shells out to, not a library anything links against. |
| `ORT_DYLIB_PATH` | `ort` uses `load-dynamic`, so onnxruntime is `dlopen`ed at runtime. Nothing links it, so nothing declares it. |
| `SLOOP_MEMORY_MODEL` | The embedding model and tokenizer. `config` refuses to start without it, deliberately -- a missing model should fail loudly at startup, not silently return bad vectors. |

The nix dev shell supplies all three, and the `sloop-memory` package bakes the
last two into a wrapper so an installed binary needs none of them set. This is
why the error message for a missing model names the wrapped binary: a raw
`cargo build` artifact genuinely cannot work without help.

## What this does not cover

`sloop-harness` now holds a conversation tree and a client that fills it.
Nodes are content blocks carrying a kept / abandoned / pending label, a branch
replays to a `messages[]` array, and `api` streams a turn back from
`/v1/messages` as blocks the tree appends. Its README covers both. Still
unbuilt: `tool_use` and `tool_result` and the fork-validity rule they need,
retries, cache-hit instrumentation, and indexing a transcript back into
`sloop-memory`.

The client has its own internal seam, and it exists for the same reason the
library/binary split does. Everything except the single socket call is a pure
function over bytes -- framing, event decoding, block accumulation -- so the
whole decoder is exercised by `cargo test` in the nix sandbox, which has no
network and no API key. Only `Api::send` cannot be tested that way, and it is
deliberately thin enough that the demo binary is what proves it.

Three constraints shape that crate and are worth repeating here, because all
three are external facts rather than choices. There is no official Anthropic
SDK for Rust, so the client talks raw HTTP. Assistant prefill was removed from
current models, which is why the tree's nodes have to be content blocks rather
than whole messages: a branch cannot resume a truncated turn, so it must be
able to name the block boundary it regenerates from. And thinking is on by
default on the model the harness targets, so a response carries `thinking`
blocks whether or not the caller asked -- which is why `ContentBlock` has that
variant while `tool_use` can wait. A thinking block's signature binds the
conversation prefix that produced it and has to travel back byte-identical,
which the tree gets for free: it only ever appends, and a fork truncates the
tail rather than rewriting history. What is not free is *where* a fork
truncates -- sharing a signed block across two generations gives one turn
reasoning from two requests -- so the harness forks at the turn boundary
whenever the block it would share carries a signature.

The harness still reaches only for `config` and `index`. The tree itself needs
nothing from the engine at all, which is the expected shape -- the link exists
so that indexing a transcript, when it arrives, is an in-process call rather
than a socket round trip.
