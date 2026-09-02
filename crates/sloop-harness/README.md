# sloop-harness

A branching agent harness over the Anthropic Messages API. Scaffold only -- see
"Not built yet" below.

It links [`sloop-memory-core`](../sloop-memory-core) directly, as a library,
rather than talking to a running `sloop-memory` daemon over a socket. That is
the case the library/binary split exists for.

## Not built yet

Only the scaffold exists. Planned:

- a conversation tree, rather than a flat message list
- forking at content-block boundaries
- replaying a branch to a `messages[]` array for the API call
- kept / abandoned / pending status per branch
- cache-hit instrumentation

The point of the status field is that a labeled rejection is distinguishable
from a conclusion, which is what would make a transcript's discarded reasoning
safe to feed into the memory index. See [the workspace README](../../README.md)
for why that matters.

## Two constraints on the design

There is no official Anthropic SDK for Rust. Python, TypeScript, Java, Go, Ruby,
C# and PHP have one; Rust does not. This talks raw HTTP to `/v1/messages`.

Assistant prefill was removed from current models -- supplying a partial final
assistant turn returns HTTP 400. So a branch cannot continue a truncated turn.
It regenerates the turn from a content-block boundary instead, which is why the
tree's nodes are content blocks rather than whole messages.

## Build and test

See [the workspace README](../../README.md). `nix develop -c cargo test -p
sloop-harness` runs this crate's tests alone.
