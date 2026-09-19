# Splitting long queries

The indexer bounds every chunk so it fits the embedding model. Nothing bounds a
query. `embed.rs:45` sets `with_truncation(max_length: MAX_SEQ_LEN)`, so a query
over 512 tokens is cut to its first 512 and embedded as though that were the
whole thing. The engine then returns a confident vector for a query it read only
part of.

This slice gives queries the treatment documents already get.

## What the log shows

Measured over `injections.jsonl`, 220 genuine user prompts captured between
2026-08-28 and 2026-09-18:

| p10 | p25 | median | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 19 | 33 | 80 | 173 | 1,088 | 6,263 | 28,341 |

The median query is 80 characters. Ninety percent fall under 1,100. Eighteen of
220 -- eight percent -- exceed roughly 512 tokens and are therefore truncated
today.

Eight percent is a tail, and the tail is the part worth fixing. Those queries
are pasted stack traces, logs and code with a question attached, and truncation
keeps the wrong half: it embeds the boilerplate at the top of the paste and
discards the question at the bottom.

## The asymmetry

`chunk::split_body` cuts document bodies at `TARGET_CHARS` (1200), capped at
`MAX_CHARS` (2000), so that no chunk overflows the model's window. That bound is
deliberate and it is the reason retrieval works at all on long notes.

Queries carry no such bound. One side of the comparison is sized to the model
and the other is not, which is the defect -- truncation is only how it shows up.
Fixing it means applying the same splitter to both sides, not inventing a second
mechanism for queries.

## Merging: max cosine, not RRF

The obvious merge is RRF, and it is wrong here. `store.rs:52` records why:

> `score`: RRF fusion score. Good for ordering, useless as a relevance gate: it
> is derived purely from rank [...] the ceiling is 2/k = 0.033
>
> `cosine`: Unlike `score` this is calibrated and comparable across queries, so
> it is what the hook gates on.

`HOOK_MIN_COSINE` (0.75) gates the hook. Merging sub-queries by RRF would yield
a rank-derived number that gate cannot read, so the truncation fix would break
the threshold on its way past.

Cosine is comparable across queries, which is exactly the property a merge
across sub-queries needs. A chunk therefore scores as the best it matched any
one unit:

    score(chunk) = max over units of cosine(chunk, unit)

This reads naturally. A long query asks several things, and a note answering any
one of them is a hit.

`recall` already performs this merge. Its dedup loop keeps the highest cosine
per `(source_type, rel_path)`:

    if h.cosine > existing.cosine {
        existing.cosine = h.cosine;
        existing.heading_path = h.heading_path;
    }

Hand it the concatenated hits from several units and it consolidates to
max-cosine-per-file unchanged. The threshold and `take_round_robin` then run as
they do now. Pointers address files, so consolidating at file granularity loses
nothing.

## What is being built

`chunk::split_query`, beside the chunker and built on `split_body`. Living there
is the argument rather than a convenience: queries and documents obey one rule
because they call one splitter.

The query path becomes split, embed each unit, search per unit, concatenate.
Everything downstream stays. A short query splits to a single unit and takes a
path identical to today's, which is what 92 percent of traffic does.

Every host inherits this -- the Claude Code hook, the MCP server, and the
harness -- with no host-specific code. The truncation is engine-side, so the fix
belongs engine-side.

BM25 improves as a side effect. Each unit carries its own text into
`full_text_search`, so the lexical side sees every term rather than the first
512 tokens' worth.

## The cap

A 28,341-character paste splits into roughly fourteen units, and fourteen
searches to serve one prompt is too many. Units are capped at 8, about 9,600
characters of query.

Past that point more paste carries less intent, and the bound is stated rather
than silent. That is the difference from today: truncation discards without
saying so, while the cap discards a documented amount.

`top_cosine` becomes the maximum across units, keeping miss logging comparable
to the existing record.

## Testing

A long query whose answer matches only its **tail**. That query fails today and
passes after, so it is the test that carries the slice.

A short query, asserting one unit and behaviour identical to today.

A query above the cap, asserting the bound holds.

A property: `split_query` never emits a unit longer than `MAX_CHARS`. The whole
design rests on that invariant, and a property test covers inputs no example
would think to include.

## Out of scope

**The Claude Code envelope filter.** 184 of 342 non-skipped prompts in the log
are Claude Code's own internals -- `<task-notification>` envelopes and an
"Analyze this conversation and determine:" classifier template carrying
conversation JSON. Each costs an embed and a search, and each pollutes the
injection record. That is real, and it is a Claude Code adapter concern, not an
engine one. Separate slice, different layer.

Worth recording here because it shares a root cause: those 152 classifier
prompts share a template prefix, so truncation embedded them to nearly the same
vector and retrieved the same note every time. `subagents-standing-grant.md` was
surfaced 16 times, all 16 to synthetic prompts and none to a person. Splitting
fixes the vector; only the filter stops the query being asked.

**Usage feedback.** The hook knows what it surfaced and not what was read. That
needs a `Used { path }` primitive the engine joins against injections, and
per-host reporting: Claude Code through a `PostToolUse` hook, MCP through the
tool call itself, the harness directly. It also needs the filter above first,
because a prior learned from today's log would learn that the most important
note on the machine is one no person has read.
