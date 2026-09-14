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

It reaches further than the pointer. `parse_note` prepends a context header to
the text it embeds -- `"{title} > {heading_path}\n\n{piece}"` -- so "Not
continued" is inside the string that gets embedded and full-text indexed, and
inside the `Hit.text` the MCP tool renders. Both retrieval paths carry the
frame, and neither needs a filter to do it.

The frame also survives chunk splitting, which was not obvious and was
verified against the real chunker rather than reasoned about: `parse_note`
keys `heading_path` per *section*, not per chunk, so an abandoned branch long
enough to be split across several chunks carries "Not continued" on every one
of them. The mechanism does not depend on a branch fitting in one chunk.

`title` there is the **filename stem** rather than the H1 (`index.rs:196`), so
it appears in every chunk header of the file. Transcript filenames must
therefore be descriptive: `2026-09-13-how-should-the-cache-expire.md`, not a
session id.

## What is being built

1. A `Tree` to markdown renderer in `sloop-harness`.
2. A `transcripts` root, written at each turn boundary.
3. A per-root cap in the hook, so transcripts cannot crowd out notes.
4. A third preamble framing transcripts as records rather than conclusions.

Out of scope, deliberately: `Status` in the index, usage feedback, the
wikilink graph, and any LLM in the ingestion path. Those are named at the end.

## Rendering

    ---
    type: transcript
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

Frontmatter supplies `captured` and `note_type` -- the YAML key is `type`,
which `parse_frontmatter` reads into `Frontmatter::note_type`. The H1 comes
from the opening user block, and is a heading rather than the title. Turns
become H2s, so every chunk gets a populated `heading_path` rather than an
empty one.

**The spine is an argument to `render`, not something it infers.** The caller
grafted the branches and is the only party that knows which one is live. A
recency rule -- the leaf with the highest `NodeId` -- was tried and is wrong
for the one caller there is: the harness grafts the kept turn, then forks and
grafts the turn it is abandoning, so the discarded branch always holds the
higher ids and "newest" names exactly the wrong side of every fork. Every fork
off the spine renders in place as a `### Not continued` subsection, quoted.

A tip that is not a leaf ends the spine there and renders what hangs below it
as an aside. A tip the tree never minted renders nothing, rather than falling
back to a branch that would read as a faithful transcript of the wrong
conversation.

**Every block leaves code fences balanced, and a block that opens on a fence
puts it on its own line.** `parse_note` ignores headings while its fence flag
is set, so any transcript that leaves a fence open erases every heading below
it -- the `### Not continued` that carries the whole design included. Two ways
in: a fence swallowed into `**assistant:** ` never toggles the flag while its
closer does, and a turn interrupted mid-fence never closes at all. Both are
handled in the renderer, so nothing downstream has to know. Asides need
neither -- `> ` in front of a fence means it is no longer a fence -- but get
both anyway, because they share the same writer and a blockquote around an
unterminated fence is malformed for every reader.

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

**Run end to end on 2026-09-13.** A scratch daemon over two roots
(`transcripts` and `notes`), started before the transcript existed so that the
startup sweep could not be what indexed it. Dropping a file written by
`record` into the watched root logged `reindexed from watch root=transcripts
changed=1 chunks=3` five seconds later. Searching the abandoned branch's
vocabulary returned it as the top hit at cos 0.82, pointing at
`... > Turn 1 > Not continued`, with the same string inside the chunk text.
`recall` returned two pointers, one per root, under their separate preambles
-- so round-robin holds and the third preamble is live.

The content blocks were canned rather than fetched, since `MESSAGES_URL` is a
constant and a live run spends tokens. Everything downstream of them was the
real path: `main`'s own `graft`, `fork_point` and `record`. What remains
unverified is only whether real API block shapes render differently from
canned ones.

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
a fork built in the order the harness builds it, a single-branch session, an
interior tip, and a tip from another tree.

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
index on it. Nothing reads it: already extracted, already indexed, wired to
nothing. One hop of traversal, no LLM.

The claim that this is a *hand-authored concept graph* was an assumption, and
counting the configured roots on 2026-09-13 disproved it:

| root | files | `[[link]]`s | files linking | distinct targets |
| --- | --- | --- | --- | --- |
| `loomis` (Obsidian) | 89 | 0 | 0 | 0 |
| `oro` (Obsidian) | 9 | 11 | 1 | 7 |
| `memory` (agent-written) | 95 | 84 | 65 | 59 |

The hand-authored half of the corpus is not linked at all: 11 links across 98
files, every one of them in a single file. Nearly every link that exists was
written by an agent into `memory`, and 26 of those 59 targets resolve to no
file -- `[[name]]` is written as a note-to-self before the note exists, which
the memory instructions explicitly encourage.

So the traversal premise fails in both directions. 127 of 193 files are
isolated nodes, which means one hop from a hit usually reaches nothing; and
transcripts entering as isolated nodes would *not* thereby rank below
well-linked notes, because most notes are equally isolated. There is no
structural signal to inherit.

That reorders the two slices. Generating edges is no longer the speculative
follow-on to traversing them -- it is the precondition, and the only one of
the two with a measurable win available. Traversal over 0.49 links per file
is not worth building first. Before either, re-run the count: the number that
matters is links per file in the roots actually configured, and it is cheap
enough that no slice here should rest on a guess about it again.

**An async concept graph.** A graph rather than a hierarchy: a hierarchy forces
each chunk into one parent, and concepts belong to many. A graph also degrades
gracefully under incremental update, where a clustered hierarchy is invalidated
by every insert. Built on its own clock so ingestion stays deterministic,
hash-addressed and fast; if it is stale, retrieval falls back to the slice
before it.
