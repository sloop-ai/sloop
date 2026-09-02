# sloop-memory-core

The retrieval engine: chunking, embedding, index building, storage and hybrid
search, plus the socket protocol and client that [`sloop-memory`](../sloop-memory)
wraps in a daemon.

It is a separate crate so a consumer can link it and call it in-process rather
than standing up a daemon and talking to it over a Unix socket.
[`sloop-harness`](../sloop-harness) does that. Everything that talks to the
outside world -- the daemon, the filesystem watcher, the MCP server, the CLI,
the hook -- lives in `sloop-memory` instead.

Chunks from every configured root share one LanceDB table, `chunks`,
distinguished by `source_type`/`source_id` rather than by a table per root.
Roots differ in what they hold, not in how they are stored, so one schema and
one set of indexes serve all of them; a per-root table would multiply index
builds without buying any query the filter cannot already express.

## Retrieval

**Hybrid.** Vector recall plus BM25, fused with reciprocal rank. Notes are dense
with exact identifiers that embeddings alone handle badly, and full of
paraphrase that BM25 alone misses.

**Gate on cosine, not on RRF.** RRF scores are rank-derived. LanceDB's default
`RRFReranker` sums `1/(rank + k)` across the two ranked lists, over zero-based
ranks and with k = 60, so a score caps at 2/k = 0.033, and a hit that appears in
only one of the lists gets a value fixed by its rank alone -- 1/k = 0.0167 at
the top of that list, whether the match is good or garbage. That orders results
well but says nothing about whether anything relevant was found at all. Cosine
is calibrated: unrelated prose sits near 0.30, on-topic chunks above 0.75.

**Query prefix on queries only.** BGE is trained asymmetrically. Putting the
prefix on documents too measurably hurts retrieval.

**Vector index above 25k rows.** Below that, LanceDB brute-forces faster than
`IVF_PQ` can be trained, and with too few rows training fails outright.
Personal-scale corpora usually sit well below the threshold.

## Ingest

**Chunk length caps.** Markdown tables have no blank lines, so paragraph
splitting alone leaves 9000-character chunks, which the tokenizer truncates at
512 tokens -- the tail of every large table is invisible to vector search while
BM25 can still see it. Capping also makes each sequence much cheaper to embed.

**Chunker version in the content hash.** Changing how notes are split
invalidates every manifest entry, so the next sweep re-embeds. Without this the
index silently keeps chunks the current code would never produce, because the
*files* are unchanged and nothing else notices.

**Content hashes, not mtimes or git.** Roots are not assumed to be under version
control, so change detection relies on hashing contents.

**One merge per batch.** `merge_insert` is the right primitive for ingest -- one
table version instead of two per file, and `when_not_matched_by_source_delete`
correctly drops orphan chunks when a note shrinks.

## Measured costs

Embedding dominates and nothing else is close. A cold run -- nothing in the
manifest, so every file is embedded -- over a small personal corpus of roughly
900 chunks, on an Apple Silicon Mac, with ONNX intra-op threads at their default
cap of 8:

    embed 37736ms   write 19ms   index-build 46ms      (40.7s wall, 744% CPU)

Writes measured at under 0.1% of the run, so `merge_insert` is a correctness win
rather than a speed one.

Per chunk that is 41ms. Extrapolated to a million-chunk corpus it is still ~11
hours of embedding, so the next real lever is the execution provider: `ort`
exposes a `coreml` feature and the Apple Neural Engine is currently unused.

## Rough edges

- `index_paths` accumulates every row in memory before one merge. Fine at this
  size; it would need periodic flushing for anything an order of magnitude
  larger.
- Embedding throughput has headroom left if it is ever wanted. Roughly 7 cores
  are busy during ingest, because files are embedded one at a time through a
  single session behind a mutex, and nixpkgs' onnxruntime is built with CoreML
  so the Apple Neural Engine is available and unused. Neither is worth doing at
  personal-corpus scale.

## Build and test

See [the workspace README](../../README.md). `nix develop -c cargo test -p
sloop-memory-core` runs this crate's tests alone.
