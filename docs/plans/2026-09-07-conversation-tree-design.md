# Conversation tree design

The first real slice of `sloop-harness`: a branching conversation tree, in
memory, with no network. It closes none of the loop by itself. It builds the
structure the rest of the loop needs, and settles the one decision that would
be expensive to revisit.

## What is being built

1. A tree whose nodes are content blocks, not whole messages.
2. A kept / abandoned / pending label on every node.
3. Replay of any branch to a `messages[]` array shaped for `/v1/messages`.

Out of scope, deliberately: HTTP, API keys, cost accounting, cache-hit
instrumentation, and indexing transcripts into `sloop-memory`. Nothing here
touches `proto` or `client` -- those are the daemon's socket protocol, and an
in-process consumer that speaks them is speaking a wire protocol to itself.
`config` and `index` remain fair game, and `main.rs` keeps calling them so the
library seam stays load-bearing.

## Node granularity is forced, not chosen

Assistant prefill was removed from current Anthropic models. Supplying a
partial final assistant turn returns HTTP 400. A branch therefore cannot
resume a truncated turn; it must regenerate the turn from a block boundary.

That single fact decides the granularity. If nodes were whole messages, the
finest available fork point would be a turn boundary, and "regenerate this
turn differently starting from its second paragraph" would be inexpressible.
Nodes are content blocks because the API leaves no alternative.

The cost is that a node no longer corresponds to anything the API accepts.
Replay has to reassemble blocks into messages, which is the next section.

## The model

    pub struct NodeId(u32);                       // arena index, scoped to one Tree
    pub enum Role { User, Assistant }
    pub enum ContentBlock { Text { text: String } }
    pub enum Status { Pending, Kept, Abandoned }
    pub struct Message { role: Role, content: Vec<ContentBlock> }

    struct Node {
        parent: Option<NodeId>,
        children: Vec<NodeId>,
        role: Role,
        block: ContentBlock,
        status: Status,
    }

    pub struct Tree { nodes: Vec<Node>, root: NodeId }

Identity is an arena index. It is `Copy`, needs no `Rc` or `RefCell`, and
serializes as a flat array when persistence arrives. Content-hash identity was
considered and rejected: two identical text blocks in different positions
collide, so the hash would have to cover the whole path -- expensive, and it
changes the id whenever a status label changes.

`ContentBlock` carries one variant today and serializes as
`{"type": "text", "text": "..."}`, which is the real API shape. Adding
`thinking`, `tool_use` and `tool_result` when the HTTP slice lands is not a
breaking change to `Node`.

## A fork is not a thing

There is no `fork()` and no `Branch` struct. The only growth operation is:

    append(parent, role, block) -> Option<NodeId>

Calling it twice on the same parent *is* the fork. A node with two children is
a fork point; that is the whole representation. A branch is named by its leaf
and recovered by walking `parent` links to the root.

This is worth stating because the alternative -- a first-class `Branch` holding
a path -- looks natural and is worse. Branches share prefixes. Two `Branch`
values owning overlapping paths have to agree about the shared part, and
nothing makes them.

`Tree::new` requires the root to be a `User` block, which pins the role of the
first replayed message.

Lookups return `Option`. One failure mode exists -- a `NodeId` minted by a
different tree -- so a dedicated error type would carry a single variant.

## Status is a lattice, joined along the path

    Pending < Kept < Abandoned

`effective_status(id)` walks root to `id` and takes the strongest label on the
way:

- any `Abandoned` ancestor poisons the entire subtree below it;
- an unlabeled node inherits a `Kept` ancestor;
- otherwise `Pending`.

Status lives on the node rather than on a branch because branches share
prefixes. A prefix node belongs to both a kept path and an abandoned one, so a
per-path label has nowhere coherent to live, while a per-node label with an
inheritance rule is well defined everywhere.

Two operations fall out cheaply. Marking a branch abandoned is one
`set_status` at the divergence point. Asking whether a block is discarded
reasoning -- the question the memory index will ask of every block it is
offered -- is one upward walk.

That question is the point of the whole field. Feeding a transcript into the
index is only safe if a rejection stays distinguishable from a conclusion.
Without the label, every abandoned idea becomes a retrievable fact.

## Replay, and the prefill constraint as a function

`replay(tip)` walks root to `tip` and groups **consecutive same-role** nodes
into one `Message`.

There are no turn identifiers. A role change is a turn boundary, and grouping
consecutive equal roles cannot emit two adjacent same-role messages. Strict
alternation is therefore structural rather than checked -- a property to test,
not an invariant to enforce at runtime.

`prompt_for(tip)` is `replay(tip)` with a trailing assistant message dropped.
One rule, and it is the prefill constraint made executable: an assistant turn
cannot be continued, so it is regenerated from the last user boundary.

    fork mid-assistant-turn, tip = a2'
      replay      -> [user{u1}, assistant{a1, a2'}]
      prompt_for  -> [user{u1}]                     // the whole turn regenerates

    fork at a user block, tip = u2'
      replay      -> [user{u1}, assistant{a1}, user{u2'}]
      prompt_for  -> unchanged                      // already ends on user

Because the root is a `User` block, `prompt_for` always returns a non-empty
array that both starts and ends with a user message. That is a guarantee the
type system does not express, so it is a property test.

## Steering falls out for free

Steering -- interrupting a turn mid-flight and injecting an instruction --
keeps the partial assistant turn and appends a user turn after it. Forking
discards the partial turn and regenerates it. They are two resolutions of one
constraint: a partial assistant turn is a legal interior state and an illegal
terminal one.

The tree expresses both with the same primitive. Steering is
`append(mid_turn_assistant_node, Role::User, block)` -- a user child hanging
off an interior assistant block. Replay yields
`[user, assistant{a1, a2}, user{steer}]`, and `prompt_for` leaves it alone
because it already ends on a user message. No second operation, no special
case.

## A constraint this slice does not yet meet

Once block kinds exist, not every block boundary is a legal fork point. An
assistant turn ending in `tool_use` requires a matching `tool_result` in the
following message, so a fork that truncates across a tool-call pair produces a
request the API rejects. Text-only blocks are unaffected, so nothing is done
about it here. `append` will need a validity rule when the HTTP slice lands.

## Testing

Unit tests cover the milestone directly: build a transcript, fork it, mark one
branch abandoned, and check that replay yields two coherent and *different*
`messages[]` arrays while the shared prefix stays shared.

Property tests cover replay over arbitrary tree shapes with forks at arbitrary
positions, because fork placement is exactly the axis where generated cases
beat hand-picked ones:

1. `replay` alternates strictly and starts with a user message.
2. `prompt_for` is non-empty, alternates, and starts *and* ends with user.
3. `replay` preserves every block on the path, in order, none lost or
   duplicated.
4. An `Abandoned` ancestor forces `Abandoned` on every descendant.

## One packaging consequence

`proptest` is a new dependency, so `Cargo.lock` changes, so the `cargoHash` in
`flake.nix` is invalidated. CI's `vendor` job builds the vendor derivation
precisely to catch that. The hash is updated in the same commit.
