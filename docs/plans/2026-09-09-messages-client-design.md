# Messages client design

The second slice of `sloop-harness`: a streaming client for `/v1/messages`,
and the block kinds a real response turns out to contain. The tree can already
produce a `messages[]` array. Nothing in the repo can send one.

This slice sends one. It closes no part of the memory loop either -- indexing
a transcript is still ahead -- but it is the first code here that learns what
the API actually returns, and one of the things it returns forces a change to
the tree's content model.

## What is being built

1. A streaming POST to `/v1/messages`, over raw HTTP.
2. A `Thinking` content block, because a response contains one whether or not
   the caller asked.
3. A demo that forks a *real* transcript and regenerates the forked turn.

Out of scope, deliberately: tools, retries, cost accounting, cache hits, tree
persistence, and indexing into `sloop-memory`.

## Thinking is not an optional block kind

On `claude-opus-5` thinking is on by default. Omitting the `thinking`
parameter runs adaptive thinking; it is not the off switch it was on Opus 4.8
and earlier. So the first real response this crate ever receives contains
`thinking` blocks, and a content model with one `Text` variant cannot hold it.

That settles what would otherwise have been a scheduling question. `ToolUse`
and `ToolResult` can wait for a slice that has tools in it. `Thinking` cannot
wait for anything, because there is no request shape that avoids it and no
version of this slice that does not receive one.

Two consequences follow, and both are about the signature rather than the
text.

A thinking block carries a `signature`, and a continuation of the same
conversation must echo the block back **unchanged**. Since `ContentBlock` is
the same type `replay` serializes into `messages[]`, this is structural: the
block that goes into the tree is the block that comes back out, and no code
has to remember to preserve anything. The test that pins it is a round trip --
decode a block from a fixture, re-serialize it, compare bytes.

The signature also binds the conversation prefix that produced it. Editing an
earlier turn invalidates every later thinking block, which is a constraint a
*branching* harness should have to think about, and does not. Forking
truncates the tail and regenerates from a block boundary; steering appends a
user turn after a partial assistant one. Both are append-only over the prefix
that remains, so the transcripts this tree produces satisfy the check by
construction. That is luck rather than foresight -- the tree was designed
around prefill removal, and this falls out of the same shape -- but it is
worth writing down, because the obvious alternative design does not have it.
A harness that rewrites history to retry a turn would.

Requesting `display: "summarized"` is a choice rather than a constraint. The
default, `"omitted"`, returns thinking blocks whose text is an empty string.
That is cheaper and round-trips identically, and it is useless twice over: the
demo shows a long pause with nothing to print, and the memory index is
eventually offered a block with no content to label. The summary costs
nothing extra -- thinking is billed the same under every display setting.

## The client does not know about branching

    api::Request        build the body; pure, serializable
    api::Event          one decoded SSE event; serde-tagged on "type"
    api::Accumulator    events -> completed ContentBlocks; pure, no I/O
    api::send           the only async fn, the only one to touch a socket

`send` takes a `&[Message]` and yields blocks. The caller appends them. The
tree stays sync, pure and property-testable; the client stays ignorant of
status labels, forks and tips.

The seam needs no adapter, which is the argument for putting it here rather
than anywhere else. `prompt_for(tip)` already returns exactly the value the
request's `messages` field takes. A `Session` type owning both, or a sink
trait writing into `&mut Tree`, would each introduce a conversion across a
boundary that currently has none.

The module is `api`, not `client`. `client` in this workspace is the daemon's
socket client, and this crate exists partly to demonstrate *not* speaking a
wire protocol to itself.

## Blocks land complete

A block is appended to the tree at `content_block_stop`, not at
`content_block_start`.

Appending on start would mean holding a `NodeId` and mutating its text as
deltas arrive, and the tree has no mutation operation -- `append` is the only
way it grows. Waiting for the stop event keeps it that way. The tree is
strictly append-only, which is both simpler and, as above, the property the
signature check wants.

The cost is that the tree is not a live view of a turn in flight. Streaming
output is printed from the event stream on its way past. Nothing needs the
partial state twice.

## The request

    POST https://api.anthropic.com/v1/messages
    x-api-key: $ANTHROPIC_API_KEY
    anthropic-version: 2023-06-01
    content-type: application/json

    {
      "model": "claude-opus-5",
      "max_tokens": 64000,
      "stream": true,
      "thinking": { "type": "adaptive", "display": "summarized" },
      "messages": [ ... ]
    }

Three fields are absent on purpose. `budget_tokens` is a 400 on this model.
`output_config.effort` defaults to `high`, so naming it changes nothing.
`cache_control` belongs to a slice that measures cache hits, and setting it
without measuring proves nothing.

`max_tokens` is 64000 because the request streams. The ceiling that matters
for a non-streaming request is the HTTP timeout, and streaming removes it.

`ANTHROPIC_API_KEY` is the only credential source, and a missing one fails
before any request is built, with a message naming the variable. This follows
`SLOOP_MEMORY_MODEL`: a missing input that makes the program useless should
fail loudly at startup rather than at first use.

## Decoding

The transport is SSE: frames separated by a blank line, each with an `event:`
line and a `data:` line. Only `data:` is parsed. The JSON on it carries its
own `type`, so the `event:` line is redundant, and a decoder that reads both
has two sources of truth for one fact.

Handled: `message_start`, `content_block_start`, `content_block_delta`
(`text_delta`, `thinking_delta`, `signature_delta`), `content_block_stop`,
`message_delta`, `message_stop`.

Unknown event types are **ignored rather than rejected**. The server may add
events, and a parser that treats an unrecognized `type` as a failure turns
every such addition into an outage.

Deltas carry an `index`, and the accumulator keys open blocks by it rather
than assuming arrival order. In practice the indices arrive in order; relying
on that buys nothing and makes the failure silent and content-corrupting if it
ever stops being true.

A thinking block streams its text as `thinking_delta` and its signature as a
separate `signature_delta`. Accumulating only the deltas that look like text
would produce a block that serializes without a signature and is rejected on
the next turn.

## Failure

`anyhow` with `.context()`, matching the two crates that already exist. A
typed error enum would buy the caller a retry decision it does not yet make.

| Failure | Handling |
|---|---|
| Missing API key | Fails before the request; names the variable |
| Non-2xx | Status plus the response body, which carries the API's own message |
| `error` event mid-stream | Surfaced as an error, not a short read |
| `stop_reason: "refusal"` | HTTP 200 -- checked before content is read |
| `stop_reason: "max_tokens"` | Not an error. A truncated turn is a legal interior state for this tree; it is printed |

There are no retries. A 429 or 529 is reported and the run ends. Backoff
belongs to the slice that has run often enough to know what it is backing off
from.

## Testing

Everything except `send` is a pure function over bytes, which is what makes
this testable in a sandbox with no network and no key -- the environment `nix
build` runs `cargo test --workspace` in.

Fixtures are recorded SSE streams in `crates/sloop-harness/tests/fixtures/`.
They double as the only readable record of the wire format in the repo.

1. A text-only turn accumulates to one `Text` block.
2. A thinking turn accumulates to a `Thinking` block whose signature survives.
3. Interleaved block indices accumulate to the right blocks.
4. An unknown event type is ignored.
5. An `error` event mid-stream fails the turn.
6. The serialized request body has the exact expected shape.
7. A `Thinking` block decoded from a fixture and re-serialized into
   `messages[]` is byte-identical.

Seven is the one that matters. It is the preserved-signature guarantee written
as an assertion, and it fails the moment someone gives `ContentBlock` a
convenience representation that does not round-trip.

One property test joins the existing four: appending a `Thinking` block does
not break replay's strict alternation.

## Two packaging consequences

`reqwest`, `hyper`, `rustls` and `ring` are **already in `Cargo.lock`**,
pulled in transitively by the lance tree, and `cargo deny` passes on them
today. Taking `reqwest` as a direct dependency adds no crate and no license
that is not already vendored. `Cargo.lock` still changes, so `cargoHash` in
`flake.nix` is invalidated and updated in the same commit; the `vendor` CI job
exists to catch that.

The flake's fileset is the one that would break quietly:

    (fileset.fileFilter (f: f.hasExt "rs" || f.name == "Cargo.toml") ./crates)

Fixtures are `.sse` files, so they are not in the build source, and the
package build runs the whole workspace's tests. Those tests would pass locally
and fail in `nix build`. `include_str!` does not avoid it -- it reads the same
filtered tree at compile time. The fileset gains `.sse` in the same commit
that adds the first fixture.

## What this slice still does not do

`ToolUse` and `ToolResult`, and with them the `append` validity rule: an
assistant turn ending in `tool_use` needs a matching `tool_result` in the
following message, so a fork that truncates across the pair produces a request
the API rejects. Text and thinking blocks are unaffected, so the rule is still
unwritten and still needed.

Also absent: retries, cost and cache-hit accounting, tree persistence, and the
one that closes the loop -- indexing a labeled transcript back into
`sloop-memory`.
