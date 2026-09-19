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
`MAX_CHARS` (2000), so that no chunk greatly overruns the model's window. The
cap is in characters and the window is in tokens, so it approximates the window
rather than guaranteeing it — see Known limitations. That bound is deliberate
and it is the reason retrieval works at all on long notes.

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

BM25 changes, but not in the way this document first claimed. The 512-token
truncation lives in the embedder's tokenizer (`embed.rs:45`) and bounds the
query *vector* alone; `hybrid_search` passes the raw query text to
`FullTextSearchQuery` (`store.rs:329`), so the lexical side always saw every
term. Splitting recovers nothing there. What it changes is that BM25 runs once
per unit rather than once over the whole query, which widens the candidate pool
feeding RRF instead of restoring lost coverage. The gain this slice delivers is
on the vector side.

## The cap

A unit advances about 1,000 characters of distinct text: packing targets
`TARGET_CHARS` (1,200) and carries `OVERLAP_CHARS` (200) forward into the next
one. Every figure here is on that basis — measured, 100,000 characters produce
99 units.

So a 28,341-character paste splits into roughly 28 units, and 28 searches to
serve one prompt is too many. Units are capped at 8, about 8,000 characters of
distinct query.

Past that point more paste carries less intent, and the bound is stated rather
than silent. That is the difference from today: truncation discards without
saying so, while the cap discards a documented amount.

The cap keeps the leading units **and the final one** — the first seven and the
last, not the first eight. Taking units from the front alone would reproduce the
original failure at a higher threshold: a pasted log with the question typed
underneath it would again be read entirely from the boilerplate at the top. The
middle of a paste is what it can spare, because the question usually sits at the
end. Keeping the tail costs nothing, so the cap bounds the work without
reintroducing what the splitter exists to remove.

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

## Known limitations

Both are accepted for this slice, not solved by it.

**Residual truncation.** The bound is characters, the window is tokens, so
splitting narrows the truncation rather than removing it. Estimated from typical
tokenizer behaviour rather than measured here: prose runs about 4 characters per
token and fits comfortably; code sits nearer 2.5-3, and log lines dense with
timestamps, UUIDs and hex run 2-2.5. At those ratios a worst-case 2,000-character
unit reaches roughly 800-1,000 tokens, so something like 60-75% of it survives
the tokenizer. Overlap softens the loss — text cut from one unit's end usually
reappears at the next unit's head — so this degrades recall on dense pastes
rather than dropping content outright. Settling it means lowering `MAX_CHARS`,
which changes document indexing and forces a `CHUNKER_VERSION` bump and a
reindex. That is its own slice.

**Overlap fragments waste searches.** On unbroken text with no line or paragraph
boundaries — minified JSON, base64, a single enormous log line — `split_body`
emits alternating sizes like `[2000, 2000, 202, 2000, 202, …]`, where the short
units are overlap tails carried forward. Each still costs an embed and a search
while repeating text the previous unit already covered. This is pre-existing
behaviour in `split_body`, unchanged here, but it bites harder on queries: the
cap makes each of the 8 searches a scarce slot, and a fragment spends one
re-searching the previous unit's tail.

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
