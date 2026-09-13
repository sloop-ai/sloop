# sloop-harness

A branching agent harness over the Anthropic Messages API.

It links [`sloop-memory-core`](../sloop-memory-core) directly, as a library,
rather than talking to a running `sloop-memory` daemon over a socket. That is
the case the library/binary split exists for.

## The conversation tree

`tree` holds a conversation as a tree of **content blocks**, not of whole
messages. `cargo run -p sloop-harness` forks a transcript, abandons one side,
and prints both branches as `messages[]` arrays.

    Tree::new(block)                   -- start from an opening user block
    tree.append(parent, role, block)   -- grow; twice on one parent is a fork
    tree.set_status(id, status)        -- kept / abandoned / pending
    tree.effective_status(id)          -- the label in force, ancestors included
    tree.replay(tip)                   -- the branch as a messages[] array
    tree.prompt_for(tip)               -- what to POST for the next turn

There is no `fork` operation and no branch type. Calling `append` twice on one
parent *is* the fork, and a branch is named by its leaf. A first-class branch
holding a path looks natural and is worse: branches share prefixes, so two of
them own overlapping paths that nothing keeps in agreement.

**Status is a lattice**, `Pending < Kept < Abandoned`, joined along the path
from the root. An `Abandoned` ancestor poisons everything below it; an
unlabeled node inherits a `Kept` ancestor. Status lives on the node rather
than on a branch for the same reason there is no branch type -- a shared
prefix node belongs to both a kept path and an abandoned one, so a per-path
label has nowhere coherent to live.

That field is the point of the crate. Feeding a transcript into the memory
index is only safe if a rejection stays distinguishable from a conclusion.
Without the label, every abandoned idea becomes a retrievable fact. See
[the workspace README](../../README.md) for why that matters.

**Replay** groups consecutive same-role blocks into one message, so a role
change is a turn boundary. Strict alternation is structural rather than
checked: grouping this way cannot emit two adjacent messages of one role.

`prompt_for` is `replay` with a trailing assistant message dropped, which is
the prefill constraint made executable -- an assistant turn cannot be
continued, so it is regenerated from the last user boundary. Because the root
is a user block, its result always starts and ends with a user message.

## Three constraints on the design

All three are facts about the API rather than choices made here.

There is no official Anthropic SDK for Rust. Python, TypeScript, Java, Go, Ruby,
C# and PHP have one; Rust does not. This talks raw HTTP to `/v1/messages`.

Assistant prefill was removed from current models -- supplying a partial final
assistant turn returns HTTP 400. So a branch cannot continue a truncated turn.
It regenerates the turn from a content-block boundary instead, which is why the
tree's nodes are content blocks rather than whole messages.

Thinking is on by default on the model this targets. Omitting the `thinking`
parameter runs adaptive thinking rather than disabling it, so the first real
response contains `thinking` blocks whether or not the caller asked. That is
why `ContentBlock` has that variant while `tool_use` waits for a slice with
tools in it -- there is no request shape that avoids one.

The consequence worth knowing is about the signature rather than the text. A
thinking block's `signature` binds the conversation prefix that produced it,
and editing an earlier turn invalidates every later one, so it has to travel
back byte-identical. This tree satisfies that by construction rather than by
care: forking truncates the tail and steering appends, so every transcript it
produces is append-only over whatever prefix remains. A harness that rewrote
history to retry a turn would not have that for free.

## Steering and forking are the same primitive

Interrupting a turn mid-flight keeps the partial assistant content and appends
a user turn after it. Forking discards the partial turn and regenerates it.
Both resolve one constraint: a partial assistant turn is a legal interior
state and an illegal terminal one.

The tree needs no second operation for this. Steering is
`append(mid_turn_assistant_node, Role::User, block)` -- a user child hanging
off an interior assistant block. `prompt_for` then leaves the branch alone,
because it already ends on a user message.

## The client

`api` talks raw HTTP to `/v1/messages` and streams a turn back as blocks the
tree appends. It is three layers, and every one of them except the socket call
is a pure function over bytes:

    api::sse::FrameDecoder   a chunked byte stream -> SSE frames
    api::sse::Event          a frame's payload -> a typed event
    api::accumulate          a stream of events -> completed ContentBlocks
    api::Api::send           the only function that touches a socket

That split is the reason the decoder is testable at all. `nix build` runs the
workspace's tests in a sandbox with no network and no API key, so anything
reaching for a socket could not be covered there. Recorded `.sse` fixtures
stand in for the wire, and they double as the only readable record of the
format in the repo.

A block reaches the tree at `content_block_stop`, never part-written. The tree
keeps its single growth operation and stays append-only, which is also the
shape a thinking block's signature requires of a history.

`ANTHROPIC_API_KEY` is the only credential source, and a missing one fails
before any request is built.

    cargo run -p sloop-harness -- "How should the cache expire?"

sends the opening turn, forks at a block boundary, regenerates the forked turn,
labels one branch kept and the other abandoned, and prints both as `messages[]`
arrays. It also writes the session to the `transcripts` root after each turn,
if one is configured, and says so if none is.

## What the label is for now

`Status` marks which branch is live for the harness's own use. The indexer does
not read it, and the slice that closed the memory loop is the reason why.

Feeding a transcript back was supposed to depend on it: a labelled rejection
staying distinguishable from a conclusion. It does not work. A label is
metadata and the text is what gets read, so every consumer has to remember to
join them -- and the lattice makes it worse, since `Abandoned` dominates
`Kept`, marking a fact established on the way to a wrong conclusion as
discarded along with it.

What carries the distinction instead is a heading. A branch that was dropped
renders under `### Not continued`, which the chunker turns into `heading_path`,
which travels into both the injected pointer and the embedded chunk text. No
filter, no column, and nothing to forget.
[The design doc](../../docs/plans/2026-09-13-transcript-indexing-design.md)
records the whole argument.

## Not built yet

- `tool_use` and `tool_result`, and the fork-validity rule they need
- retries and backoff
- cache-hit instrumentation
- usage feedback, and lighting up the wikilink graph in retrieval

Not every block boundary is a legal fork point. Two constraints say so; one is
met and the other is not.

The met one is about signatures. Forking *inside* a turn keeps that turn's
first block as a shared prefix, which is free when the block is text and is not
free when it is `thinking`. The shared signature was produced by the generation
that also produced the rest of the first branch, so a second generation hung
under it replays to an assistant turn whose reasoning came from two different
requests -- and the request that produced the second half never contained the
first. Today's model accepts it; it is the shape preserved thinking rejects,
where a signature binds the conversation prefix before it. So the tree would
have held a branch it cannot send, which is the one thing its append-only shape
exists to prevent. `fork_point` now falls back to the turn boundary when the
shared block would carry a signature, the way a one-block turn already did.

That fallback is the common case rather than a corner. Thinking is on by
default, so the opening block of a real turn is usually `thinking`, and against
the live API the demo now forks at the turn boundary far more often than inside
the turn. Only the *shared* block is consulted: a signature below the
divergence belongs to one branch alone, and refusing the interior fork for it
would give up a legal one.

The unmet one arrives with tools. An assistant turn ending in a tool call
requires a matching `tool_result` in the next message, so a fork that truncates
across the pair produces a request the API rejects. `append` will need a
validity rule then.

Both were invisible to the tests, and the first was invisible to the design
that specified it -- the fork example in
[the tree design](../../docs/plans/2026-09-07-conversation-tree-design.md) was
worked with text blocks, where sharing a prefix costs nothing. It took reading
a real transcript to see that the rule stops holding as soon as a block carries
a signature.

## Design notes

[`docs/plans/2026-09-07-conversation-tree-design.md`](../../docs/plans/2026-09-07-conversation-tree-design.md)
records why node granularity is forced rather than chosen, and what was
rejected on the way.

## Build and test

See [the workspace README](../../README.md). `nix develop -c cargo test -p
sloop-harness` runs this crate's tests alone. The replay properties are
`proptest` cases over arbitrary tree shapes, because fork placement is exactly
the axis where generated cases beat hand-picked ones.
