# Splitting Long Queries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop silently truncating queries over 512 tokens; split them with the same splitter documents get, search each unit, and merge by max cosine.

**Architecture:** `chunk::split_query` wraps the existing `split_body`, so queries and document chunks obey one bound. `recall` and the daemon's `Search` handler loop over units, concatenate the hits, and let the existing max-cosine dedup consolidate them. The relevance gate (`HOOK_MIN_COSINE`) is untouched because cosine is comparable across queries; RRF is deliberately **not** used to merge.

**Tech Stack:** Rust, LanceDB, `tokio`, `proptest` (already a dev-dep of `sloop-harness`; this plan adds it to `sloop-memory-core`).

**Design:** `docs/plans/2026-09-18-splitting-long-queries-design.md`

**Baseline:** branch `split-long-queries` off `main` (`2cbb685`), 149 tests passing, clippy and fmt clean.

**House rules that bite here:**
- **A `cargo test` filter that matches nothing exits 0 and prints `ok. 0 passed`.** It is a green light for a suite that never ran. Filters match the *full test path* (`chunk::tests::a_short_query_is_one_unit_unchanged`), not the function under test — so filtering on `split_query` selects zero tests even when six of them call it. Check the passed count is non-zero before believing any filtered run.
- `unwrap_used` is **denied**, `expect_used` warned. Test modules carry `#[expect(clippy::unwrap_used, reason = "...")]`.
- `print_stdout`/`print_stderr` denied.
- Run everything through `nix develop -c` if the dev shell is not already active (direnv usually has it).
- 100-char lines; `cargo fmt` decides.

---

### Task 1: `split_query`, the pure splitter

**Files:**
- Modify: `crates/sloop-memory-core/src/config.rs` (add `MAX_QUERY_UNITS` near `HOOK_MIN_COSINE`, ~line 40)
- Modify: `crates/sloop-memory-core/src/chunk.rs` (add `split_query` after `split_body`, ~line 211)

**Step 1: Write the failing tests**

Add to the `mod tests` block at the bottom of `crates/sloop-memory-core/src/chunk.rs`:

```rust
// ---- query splitting --------------------------------------------------
//
// A query is one half of the same comparison a chunk is, so it gets the
// same splitter. These pin what the search path depends on: a short query
// stays one unit, a long one keeps its tail, nothing exceeds the chunk-size
// bound, and the cap holds without throwing the tail away.

#[test]
fn a_short_query_is_one_unit_unchanged() {
    assert_eq!(split_query("how should the cache expire?"),
               vec!["how should the cache expire?".to_string()]);
}

/// The regression this whole slice exists for. Truncation keeps the head of
/// a paste and drops the question at the end; splitting must not.
#[test]
fn a_long_query_keeps_its_tail() {
    let paste = "stack frame line\n".repeat(400);
    let query = format!("{paste}\nwhy does the cache never expire?");

    let units = split_query(&query);

    assert!(units.len() > 1, "a {}-char query did not split", query.len());
    assert!(
        units.last().is_some_and(|u| u.contains("why does the cache never expire?")),
        "the question at the tail reached no unit"
    );
}

#[test]
fn units_never_exceed_the_chunk_size_bound() {
    let query = "word ".repeat(20_000);
    for unit in split_query(&query) {
        assert!(
            unit.chars().count() <= MAX_CHARS,
            "unit of {} chars exceeds {MAX_CHARS}",
            unit.chars().count()
        );
    }
}

#[test]
fn a_whitespace_only_query_yields_no_units() {
    assert_eq!(split_query("   \n\n  "), Vec::<String>::new());
}

/// The cap is live code. Without this, deleting the truncation changes no
/// test result -- the per-unit length assertion is indifferent to count.
/// Measured, this fixture splits into 99 units, so the 8 is the cap rather
/// than the fixture's own size.
#[test]
fn a_very_long_query_is_capped() {
    let units = split_query(&"word ".repeat(20_000));
    assert_eq!(units.len(), config::MAX_QUERY_UNITS);
}

/// Above the cap the tail still has to survive, or the cap reintroduces the
/// failure the splitter exists to remove. Measured, this fixture splits into
/// 44 units before capping, so the 8 is the cap rather than a coincidence.
#[test]
fn a_query_over_the_cap_still_keeps_its_tail() {
    let paste = "stack frame line\n".repeat(3000);
    let query = format!("{paste}\nwhy does the cache never expire?");

    let units = split_query(&query);

    assert_eq!(units.len(), config::MAX_QUERY_UNITS);
    assert!(
        units.last().is_some_and(|u| u.contains("why does the cache never expire?")),
        "the cap dropped the question at the tail"
    );
}
```

The paste in the last test must split to **more** than `MAX_QUERY_UNITS` before capping, or it passes vacuously — 3000 lines gives 44 units against a cap of 8. Confirm it by checking that the test fails when the cap is removed (Step 5).

**Step 2: Run them and watch them fail**

```
nix develop -c cargo test -p sloop-memory-core chunk::tests
```

Expected: compile error, `cannot find function 'split_query'`.

**Step 3: Add the cap constant**

In `crates/sloop-memory-core/src/config.rs`, immediately after `HOOK_MIN_COSINE`:

```rust
/// How many units a long query searches before the middle of it is dropped.
///
/// A unit advances `chunk::TARGET_CHARS` less `chunk::OVERLAP_CHARS` of new
/// text, since each split carries its predecessor's tail forward -- so eight
/// is roughly 8,000 characters of distinct query at today's values. Past that
/// a paste carries less intent than the searches cost: measured over 220 real
/// prompts the median is 80 characters and 92% need no splitting at all.
/// Unlike the tokenizer's truncation this bound is stated rather than silent.
pub const MAX_QUERY_UNITS: usize = 8;
```

**Step 4: Implement `split_query`**

In `crates/sloop-memory-core/src/chunk.rs`, directly after `split_body`:

```rust
/// Split a query on the same character bound documents get.
///
/// Deliberately the same splitter. `split_body` holds every piece to
/// `MAX_CHARS`; a query is the other half of that comparison and was bounded
/// by nothing.
///
/// That bound is characters, not tokens, so this narrows the tokenizer's
/// truncation rather than removing it: dense text tokenizes at 2-3 characters
/// per token, so a full-size unit of log or code can still exceed
/// `MAX_SEQ_LEN`. Fixing that means lowering `MAX_CHARS`, which touches
/// document indexing too.
///
/// A query that already fits comes back as one unit, which is the common
/// case by a wide margin. Whitespace-only input yields no units; callers must
/// reject empty queries, as the search paths do before reaching here.
///
/// Above [`config::MAX_QUERY_UNITS`] the cap keeps the first
/// `MAX_QUERY_UNITS - 1` units and the last one.
#[must_use]
pub fn split_query(query: &str) -> Vec<String> {
    let mut units = split_body(query);
    if units.len() > config::MAX_QUERY_UNITS {
        // Keep the last unit. Dropping it would reproduce at a higher
        // threshold the very failure this function removes: a pasted log
        // with the question underneath it, read from the boilerplate at
        // the top. The middle is what a paste can spare.
        let tail = units.remove(units.len() - 1);
        units.truncate(config::MAX_QUERY_UNITS - 1);
        units.push(tail);
    }
    units
}
```

A plain `units.truncate(MAX_QUERY_UNITS)` is the tempting simplification and it is wrong: it drops the tail, reproducing at ~8,000 characters the exact failure this slice removes. `a_query_over_the_cap_still_keeps_its_tail` is the test that catches it.

Add the import at the top of `chunk.rs` if absent:

```rust
use crate::config;
```

**Step 5: Run the tests, and prove the cap tests have teeth**

```
nix develop -c cargo test -p sloop-memory-core chunk::tests
```

Expected: 11 passed — 5 pre-existing chunk tests plus the 6 added here. (Filtering on `split_query` matches nothing; see the house rule above.)

Then confirm the two cap tests are not vacuous. Per `[[mutation-test-in-a-worktree]]`, mutate in a scratch worktree, never the shared checkout:

- Replace the body with a plain `units.truncate(config::MAX_QUERY_UNITS)` → `a_query_over_the_cap_still_keeps_its_tail` must FAIL with "the cap dropped the question at the tail".
- Replace the body with a bare `split_body(query)` → both `a_very_long_query_is_capped` and `a_query_over_the_cap_still_keeps_its_tail` must FAIL.

**Step 6: Commit**

```bash
git add crates/sloop-memory-core/src/chunk.rs crates/sloop-memory-core/src/config.rs
git commit -S -m "Split a query with the splitter documents already use"
```

---

### Task 2: Property test on the invariant

The whole design rests on "no unit exceeds `MAX_CHARS`". An example test covers the inputs someone thought of; this covers the ones they did not.

**Files:**
- Modify: `crates/sloop-memory-core/Cargo.toml` (`[dev-dependencies]`)
- Modify: `crates/sloop-memory-core/src/chunk.rs` (tests)

**Step 1: Add the dev-dependency**

In `crates/sloop-memory-core/Cargo.toml`, under `[dev-dependencies]` beside `tempfile`:

```toml
# The unit-size bound is a property over arbitrary text, and the inputs that
# break splitters -- no blank lines, one enormous line, lone newlines -- are
# exactly the ones an example test omits. sloop-harness already uses proptest.
proptest = "1"
```

**Step 2: Write the failing property**

```rust
proptest::proptest! {
    /// Whatever the input, every unit fits the window and the cap holds.
    #[test]
    fn every_unit_fits_the_window(query in ".{0,10000}") {
        let units = split_query(&query);
        proptest::prop_assert!(units.len() <= config::MAX_QUERY_UNITS);
        for unit in units {
            proptest::prop_assert!(unit.chars().count() <= MAX_CHARS);
        }
    }
}
```

**Step 3: Run it**

```
nix develop -c cargo test -p sloop-memory-core chunk::tests
```

Expected 12 passed (the 11 from Task 1 plus this property). `every_unit_fits` would also match, but `chunk::tests` shows the whole module's count, which is the number to check against.

Expected: PASS (it should already hold; if it fails, `split_body`'s `hard_split` fallback has a gap and that is a real find — stop and report it rather than weakening the assertion).

**Step 4: Commit**

```bash
git add crates/sloop-memory-core/Cargo.toml crates/sloop-memory-core/src/chunk.rs
git commit -S -m "Pin the unit-size bound with a property test"
```

---

### Task 3: Extract the merge so it can be tested

`recall` consolidates hits inline. Extracting it makes the max-cosine merge testable without a table or an embedder, and it is the function that does the real work once several units feed it.

**Files:**
- Modify: `crates/sloop-memory/src/daemon.rs:36-92` (`recall`)

**Step 1: Extract, changing no behaviour**

Lift the loop out of `recall` into a free function above it:

```rust
/// Reduce raw hits to one pointer per file, keeping the best cosine.
///
/// This is where a split query is put back together. Each unit is searched
/// separately, and a chunk scores as the best it matched any one of them --
/// max cosine, not RRF. RRF ranks well but is derived purely from rank
/// (`store::Hit::score`), so `HOOK_MIN_COSINE` could not gate on it; cosine is
/// calibrated and comparable across queries, which is exactly what merging
/// across sub-queries needs.
fn consolidate(hits: Vec<store::Hit>, roots: &config::Roots) -> Vec<Pointer> {
    let mut best: Vec<Pointer> = Vec::new();
    for h in hits
        .into_iter()
        .filter(|h| h.cosine >= config::HOOK_MIN_COSINE)
    {
        if let Some(existing) = best
            .iter_mut()
            .find(|p| p.source_type == h.source_type && p.rel_path == h.rel_path)
        {
            if h.cosine > existing.cosine {
                existing.cosine = h.cosine;
                existing.heading_path = h.heading_path;
            }
        } else {
            let path = roots
                .iter()
                .find(|root| root.label.as_str() == h.source_type)
                .map_or_else(
                    || h.rel_path.clone(),
                    |root| root.dir.as_path().join(&h.rel_path).display().to_string(),
                );
            best.push(Pointer {
                title: h.title,
                source_type: h.source_type,
                path,
                rel_path: h.rel_path,
                heading_path: h.heading_path,
                captured: h.captured,
                cosine: h.cosine,
            });
        }
    }
    best
}
```

And in `recall`, replace the loop with:

```rust
    let best = consolidate(hits, &state.roots);
    Ok(RecallResult {
        pointers: take_round_robin(best, config::HOOK_MAX_HITS),
        top_cosine,
    })
```

**Step 2: Confirm nothing changed**

```
nix develop -c cargo test --workspace
```

Expected: 155 passed (149 baseline + 6 from Task 1; Task 2's property counts as 1 → 156 once Task 2 lands. Confirm the number and carry it forward).

**Step 3: Add the merge test**

In `daemon.rs`'s `mod tests`, build synthetic `store::Hit`s (follow the `take_round_robin` tests just below for the existing construction style):

```rust
/// Two units matched the same file; the better match is the one that counts.
/// This is the merge a split query depends on.
#[test]
fn the_best_matching_unit_wins_for_a_file() {
    let roots = /* reuse whatever the neighbouring tests use; an empty Roots is fine
                   since the path fallback is rel_path */;
    let hits = vec![
        hit("notes", "cache.md", "Turn 1", 0.76),
        hit("notes", "cache.md", "Turn 9", 0.91),
    ];

    let out = consolidate(hits, &roots);

    assert_eq!(out.len(), 1, "the same file produced two pointers");
    assert!((out[0].cosine - 0.91).abs() < f32::EPSILON);
    assert_eq!(out[0].heading_path, "Turn 9", "kept the weaker unit's heading");
}

/// A unit below the gate contributes nothing, even when another unit of the
/// same query cleared it. The threshold is per-chunk, not per-query.
#[test]
fn a_unit_below_the_gate_is_dropped() {
    let hits = vec![hit("notes", "unrelated.md", "H", 0.40)];
    assert!(consolidate(hits, &roots).is_empty());
}
```

Write a small `fn hit(source_type, rel_path, heading, cosine) -> store::Hit` helper alongside; fill remaining fields with empty strings and `score: 0.0`.

**Step 4: Run**

```
nix develop -c cargo test -p sloop-memory daemon::tests
```

Expected: 8 passed — the 6 existing `daemon::tests` plus the 2 added here. Filtering on `consolidate` matches nothing (no test path contains it); `daemon::tests` is the filter that selects the module.

**Step 5: Prove the test has teeth**

Per `[[mutation-test-in-a-worktree]]`, in a scratch worktree only — never the shared checkout:

```bash
SP=<scratchpad>
git worktree add --detach "$SP/mutate" HEAD
/bin/cp -Rc target "$SP/target-mutate"
cd "$SP/mutate"
# mutant: keep the FIRST match rather than the best
#   change `if h.cosine > existing.cosine {` to `if false {`
git diff --stat          # must show the file changed
CARGO_TARGET_DIR="$SP/target-mutate" cargo test -p sloop-memory the_best_matching_unit
# expect FAIL naming 0.76 — the weaker unit — then:
cd - && git worktree remove --force "$SP/mutate" && trash "$SP/target-mutate"
```

**Step 6: Commit**

```bash
git add crates/sloop-memory/src/daemon.rs
git commit -S -m "Extract the pointer merge and test max-cosine directly"
```

---

### Task 4: Search every unit in `recall`

**Files:**
- Modify: `crates/sloop-memory/src/daemon.rs:36-56` (`recall`)

**Step 1: Replace the single search**

Swap the two lines that embed and search:

```rust
    let qvec = { state.embedder.lock().await.encode_query(prompt)? };
    let hits = store::hybrid_search(&table, qvec, prompt, config::HOOK_MAX_HITS * 4, None).await?;
```

for a loop over units:

```rust
    // One search per unit, concatenated. `consolidate` reduces them to the
    // best cosine per file, so a note answering any part of a long query is a
    // hit. A query that fits the window splits to one unit and this is the
    // path it took before.
    let mut hits = Vec::new();
    for unit in chunk::split_query(prompt) {
        let qvec = { state.embedder.lock().await.encode_query(&unit)? };
        hits.extend(
            store::hybrid_search(&table, qvec, &unit, config::HOOK_MAX_HITS * 4, None).await?,
        );
    }
```

`top_cosine` below is already `hits.iter().map(|h| h.cosine).max_by(...)`, so it becomes the max across units with no edit — which is what the design calls for.

Add `chunk` to the `sloop_memory_core` imports at the top of `daemon.rs`.

**Step 2: Verify**

```
nix develop -c cargo test --workspace
nix develop -c cargo clippy --all-targets --all-features -- -D warnings
```

Expected: all pass, no warnings. The embedder lock is taken and released per unit rather than held across the loop — keep it that way; holding a lock across `.await` trips `await_holding_lock`, which is denied.

**Step 3: Commit**

```bash
git add crates/sloop-memory/src/daemon.rs
git commit -S -m "Search every unit of a split query in recall"
```

---

### Task 5: The same for the `Search` handler

MCP (`sloop_search`, used by Cursor) reaches the engine through `Request::Search`, not `Recall`, so it truncates today exactly as the hook did.

**Files:**
- Modify: `crates/sloop-memory/src/daemon.rs:156-185` (`Request::Search`)

**Step 1: Apply the same loop**

Replace the single `encode_query` + `hybrid_search` with the unit loop, honouring the caller's `k` as the per-unit limit and the overall cap. Keep the existing error handling shape (this arm returns `Response::Error` rather than `?`).

Sort the concatenated hits by cosine descending and truncate to `k` before returning, so a split query still returns `k` hits rather than `k × units`.

**Step 2: Verify**

```
nix develop -c cargo test --workspace
nix develop -c cargo clippy --all-targets --all-features -- -D warnings
nix develop -c cargo fmt --all
```

**Step 3: Commit**

```bash
git add crates/sloop-memory/src/daemon.rs
git commit -S -m "Split long queries on the MCP search path too"
```

---

### Task 6: Verify against a live daemon

Unit tests cover the splitter and the merge. They do not cover the claim the slice is for: that a question at the **tail** of a long paste now retrieves the right note. That needs real embeddings.

Per `[[my-own-probe-is-a-fixture-not-an-integration-test]]`, drive the real binaries; do not hand-build the input.

**Steps:**

1. Scratch roots under the scratchpad: `roots/notes/` holding one note whose subject appears **only** in the tail question, never in the paste.
2. Start a scratch daemon — `SLOOP_MEMORY_SOCKET` must be a **short** path (`/tmp/sloop-rt.sock`); the socket limit is 104 bytes and a scratchpad path exceeds it:
   ```
   SLOOP_MEMORY_ROOTS="notes=$SP/roots/notes" SLOOP_MEMORY_STATE="$SP/state" \
   SLOOP_MEMORY_SOCKET=/tmp/sloop-rt.sock ./target/debug/sloop-memory daemon
   ```
3. Search with a long paste plus a tail question:
   ```
   SLOOP_MEMORY_SOCKET=/tmp/sloop-rt.sock ./target/debug/sloop-memory search "<8KB paste>\n<question>"
   ```
4. Confirm the note is returned. Re-run against `main` to confirm it is **not** — a fix that passes both ways proves nothing.
5. Never touch the user's running daemon (`launchctl list | grep sloop`); the scratch one uses its own state, socket and roots.
6. Record the result in the design doc's own section, as the transcript slice did.

**Commit:**

```bash
git add docs/plans/2026-09-18-splitting-long-queries-design.md
git commit -S -m "Record the live verification of split queries"
```

---

### Task 7: Ship

1. Full suite: `nix develop -c cargo test --workspace`, clippy, `cargo fmt --all --check`.
2. `Cargo.lock` changed in Task 2 (proptest added to `sloop-memory-core`). **The `cargoHash` in `flake.nix` is now stale** — see `[[cargohash-needs-the-full-nix-build]]`. Even though proptest is already vendored, `fetchCargoVendor` puts `Cargo.lock` inside the vendor output. Push and take the correct hash from CI's `cargo vendor hash` `got:` line, or set `cargoHash = ""` and build locally.
3. Push with the gh credential route (Secretive refuses SSH auth unpredictably; see the **Commits** section of `~/.claude/CLAUDE.md`):
   ```bash
   git -c url."https://github.com/".insteadOf=git@github.com: \
       -c credential."https://github.com".helper='!gh auth git-credential' \
       push origin split-long-queries
   ```
4. Open the PR describing only what the diff does.
