# Transcript indexing design

The slice that closes the loop. `sloop-memory` retrieves and `sloop-harness`
generates, and nothing connects them: a session's reasoning dies with the
process. This slice writes the live conversation somewhere the existing
indexer already looks, so what one session works out is retrievable by the
next.

It needs no new ingestion API, no schema change, and no label. That is the
surprising part, and it took disproving the project's stated thesis to get
there.

## The thesis this slice disproves

The README argues that a branching harness is what makes indexing safe:

> A branching harness records which branches were kept and which were
> abandoned, so a labeled rejection stays distinguishable from a conclusion --
> and that label is what would make discarded reasoning safe to index, rather
> than turning every rejected idea into a retrievable fact.

That is wrong, and the tree's own design doc states the claim more strongly
still, calling the question "the point of the whole field."

The unsafety was never the absence of a label. It was stripping a block out of
its context and presenting it as an assertion. A label cannot repair that,
because the label is metadata and the text is what gets read. Every consumer
has to remember to join them, and there are already two consumers with
separate code paths -- the prompt hook and the `sloop_search` MCP tool. The
hook passes no filter at all today (`daemon.rs:51`). One forgotten join turns
a rejection into a fact.

Three further problems, each fatal on its own:

**The lattice discards true things.** `Abandoned` dominates `Kept`, so an
abandoned ancestor poisons its whole subtree. Reasoning that establishes a
fact on the way to a wrong conclusion -- "the cache API is not thread-safe" --
is marked discarded along with the conclusion. The label sits on a position in
a tree and is asked to carry a judgment about ideas, which do not respect
subtree boundaries.

**Node granularity is an API artifact.** Nodes are content blocks because
prefill was removed from current models, so a branch must name a block
boundary to regenerate from. That is a fact about `/v1/messages`. Using it as
the chunking unit for retrieval is a category error, and the code shows it:
`chunk::parse_note` is built for authored markdown, and a raw block has no
frontmatter, no heading and no capture date, so every field the pointer
renderer depends on comes back empty.

**Nothing applies the labels.** `set_status` is a plain setter. Its only two
calls are in the demo, which forks, regenerates, and arbitrarily declares the
first branch kept and the second abandoned. No judgment happens anywhere. Worse,
`Abandoned` conflates "this was wrong" with "we went another way" -- opposite
retrieval values under one bit.

`Status` stays in the tree. It works, it is tested, and it costs nothing. It
is simply not load-bearing for the index, and the README's argument for the
harness needs rewriting when this ships.

## What replaces it

Keep the context instead of labeling its absence.

The hook's existing contract is not "here is a fact." It is "here is a path;
read it if you care." Both preambles say so. A transcript hit meaning *this
conversation touched this -- go look* is the same contract, not a new one. If
the retrieval unit is a pointer into a conversation, and following it hands
back the conversation, the context carries the epistemic status. Nothing was
stripped, so nothing needs a label to restore it.

The keystone is `append_pointer_line`, which renders a pointer as
`{path} > {heading_path}`. Render forks as headings and the frame arrives in
the pointer for free:

    transcripts/2026-09-13-cache-expiry.md > Turn 1 > Not continued
      (source transcripts, cos 0.79)

The heading *is* the label. No renderer can strip it, because the renderer
already prints heading paths. It cannot drift from its content, because it is
a heading over that content. It costs no column, no filter and no schema
change.

## What is being built

1. A `Tree` to markdown renderer in `sloop-harness`.
2. A `transcripts` root, written at each turn boundary.
3. A per-root cap in the hook, so transcripts cannot crowd out notes.
4. A third preamble framing transcripts as records rather than conclusions.

Out of scope, deliberately: `Status` in the index, usage feedback, the
wikilink graph, and any LLM in the ingestion path. Those are named at the end.

## Rendering

    ---
    note_type: transcript
    captured: 2026-09-13
    ---

    # How should the cache expire?

    ## Turn 1

    **user:** How should the cache expire?

    **assistant:** Expire on write, with a 5s grace window -- the read
    path never blocks on revalidation.

    ### Not continued

    > TTL-only expiry at 60s. Simple, but every read after expiry
    > blocks while it revalidates.

Frontmatter supplies `captured` and `note_type`. The H1 comes from the opening
user block. Turns become H2s, so every chunk gets a populated `heading_path`
rather than an empty one.

**The spine is the leaf with the highest `NodeId`.** Nodes are appended in
creation order and never reparented -- the argument `path()` uses to prove it
terminates -- so the highest-numbered leaf is exactly the branch the session
was on when it ended. Every fork off the spine renders in place as a
`### Not continued` subsection, quoted.

The alternative, rendering every branch as a peer section, was rejected.
Branches share prefixes in the tree and a flat render does not, so it must
either duplicate the shared prefix into every section -- putting near-identical
chunks in the index to compete with each other and distort the embedding
statistics -- or omit it and leave branches unreadable standalone. It also
replaces `Turn 3 > Not continued` with `Branch 2`, discarding the framing that
makes the whole design safe.

**Thinking blocks are omitted.** They run several times the length of a
response, so including them would multiply the transcript corpus for the same
number of conclusions -- worsening crowding on the day it is introduced. They
are also mid-derivation: a thinking block states every idea considered,
including wrong ones asserted confidently a paragraph before being dropped.
Matching a query against that reintroduces exactly the failure this design
avoids. Omitting is the reversible direction; adding later costs a re-render,
while removing later means the index already holds them.

No configuration flag. A flag would change what is *in the index*, so two runs
would produce indexes differing in kind and neither would yield a clean signal.

If transcripts prove too thin, the next thing to try is thinking blocks inside
`### Not continued` sections only: on the spine the response is the conclusion
and the thinking is redundant, while an abandoned branch often has no polished
response at all and the reason it was dropped lives only in the thinking.

## Writing and indexing

One file per session, under the root labelled `transcripts`, located through
`config::roots()`. The file is rewritten **in full** at each turn boundary.

Full rewrite is idempotent, so there is no partial-append state to get wrong,
and the manifest's content hash makes an unchanged rewrite free. The daemon's
watcher sees the write and re-indexes after `WATCH_DEBOUNCE_SECS`, so a turn
becomes retrievable about four seconds after it lands -- incremental indexing,
using machinery that already exists and is already tested.

The harness does not index. There is exactly one writer to the index, the
daemon, which is what keeps the harness writing files and the watcher indexing
them from racing on the same rows.

## Crowding

`recall` sorts hits globally by cosine and truncates to `HOOK_MAX_HITS`
(`daemon.rs:86-88`). Transcripts are voluminous next to notes: one session
generates more text than anyone writes down deliberately, so a curated note
would compete for three slots against conversational prose that happens to
share vocabulary.

Selection becomes round-robin across roots -- best hit from each root, then
second-best, and so on -- so one root cannot take every slot while others have
hits above threshold. Deterministic, and no ranking model.

## Testing

Rendering, in `sloop-harness`: hand-built trees to expected markdown, covering
a fork, a single-branch session, and the spine rule with more than two leaves.

Chunkability, in `chunk.rs` where the code under test lives: a transcript
fixture through `parse_note`, asserting `heading_path`, `captured` and
`note_type` come back populated. Silent empties are the failure mode, and they
would not surface anywhere else.

Round-robin selection, in `sloop-memory`: hits from two roots, asserting
neither monopolises the block.

## What comes after

Crowding is not solved, only deferred. Flat retrieval over chunks has no notion
of level, so every chunk competes as a peer forever; a per-root cap bounds the
damage without addressing it. Three slices follow, each earning the next:

**Usage feedback.** The hook knows what it surfaced and not what was read.
Capturing that gives a relevance prior learned from use -- a note followed nine
times outranking a transcript chunk never opened. This is the only part of the
arc that is genuinely a loop, in the level-crossing sense the project is named
for: what gets retrieved changes what gets retrieved next.

**The wikilink graph.** `chunk::extract_wikilinks` already pulls every
`[[link]]` into an `entities` column, and `store.rs:236` builds a LabelList
index on it. Nothing reads it. That is a hand-authored concept graph, already
extracted and already indexed, wired to nothing. One hop of traversal, no LLM.
Transcripts contain no wikilinks, so they enter that graph as isolated nodes
and rank below well-linked notes structurally rather than by a hardcoded cap.

**An async concept graph.** A graph rather than a hierarchy: a hierarchy forces
each chunk into one parent, and concepts belong to many. A graph also degrades
gracefully under incremental update, where a clustered hierarchy is invalidated
by every insert. Built on its own clock so ingestion stays deterministic,
hash-addressed and fast; if it is stale, retrieval falls back to the slice
before it.
