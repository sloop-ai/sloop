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

## Two constraints on the design

There is no official Anthropic SDK for Rust. Python, TypeScript, Java, Go, Ruby,
C# and PHP have one; Rust does not. This talks raw HTTP to `/v1/messages`.

Assistant prefill was removed from current models -- supplying a partial final
assistant turn returns HTTP 400. So a branch cannot continue a truncated turn.
It regenerates the turn from a content-block boundary instead, which is why the
tree's nodes are content blocks rather than whole messages.

## Steering and forking are the same primitive

Interrupting a turn mid-flight keeps the partial assistant content and appends
a user turn after it. Forking discards the partial turn and regenerates it.
Both resolve one constraint: a partial assistant turn is a legal interior
state and an illegal terminal one.

The tree needs no second operation for this. Steering is
`append(mid_turn_assistant_node, Role::User, block)` -- a user child hanging
off an interior assistant block. `prompt_for` then leaves the branch alone,
because it already ends on a user message.

## Not built yet

- the HTTP client for `/v1/messages`, and with it API keys and cost
- block kinds beyond text: `thinking`, `tool_use`, `tool_result`
- cache-hit instrumentation
- indexing a transcript into `sloop-memory`

One constraint is known and not yet met. Once `tool_use` exists, not every
block boundary is a legal fork point: an assistant turn ending in a tool call
requires a matching `tool_result` in the next message, so a fork that
truncates across the pair produces a request the API rejects. `append` will
need a validity rule then. Text-only blocks are unaffected.

## Design notes

[`docs/plans/2026-09-07-conversation-tree-design.md`](../../docs/plans/2026-09-07-conversation-tree-design.md)
records why node granularity is forced rather than chosen, and what was
rejected on the way.

## Build and test

See [the workspace README](../../README.md). `nix develop -c cargo test -p
sloop-harness` runs this crate's tests alone. The replay properties are
`proptest` cases over arbitrary tree shapes, because fork placement is exactly
the axis where generated cases beat hand-picked ones.
