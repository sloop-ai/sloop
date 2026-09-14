# Transcript Indexing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write each `sloop-harness` session to markdown in a `transcripts` root so the existing indexer makes it retrievable, and stop transcripts from crowding curated notes out of the prompt hook.

**Architecture:** The harness renders its `Tree` to markdown and rewrites one file per session at every turn boundary. The daemon's filesystem watcher indexes that file with the existing chunker after a 4-second debounce, so the harness never touches the index and there is exactly one writer. Forks render as `### Not continued` headings, which puts the epistemic status into both the pointer line and the embedded chunk text without a label, a column, or a filter.

**Tech Stack:** Rust (workspace-pinned toolchain), `chrono` for dates, existing `sloop-memory-core` `config`/`chunk` modules, `proptest` already present for the harness.

**Design:** [`docs/plans/2026-09-13-transcript-indexing-design.md`](2026-09-13-transcript-indexing-design.md)

---

## Before you start

Read the design doc. The non-obvious constraints it records, which the tasks below depend on:

- `parse_frontmatter` reads the YAML key `type`, **not** `note_type`.
- `parse_note`'s `title` is the **filename stem** (`index.rs:196`), and it is prepended to every chunk's embedded text as `"{title} > {heading_path}"`. Filenames must be descriptive.
- `Tree::leaves()` returns leaves in **ascending creation order**, because it enumerates the arena. The last element is therefore the highest `NodeId`.
- A node's id is always greater than its parent's — nodes are appended and never reparented. `Tree::path()` relies on this to prove termination.
- Crate lints deny `unwrap_used`, `panic`, `print_stdout` and `todo`. Tests opt out with `#[expect(clippy::unwrap_used, reason = "...")]`, as the existing test modules do.

Verify the baseline before the first change:

    cargo test --workspace
    cargo clippy --all-targets --all-features -- -D warnings

Expected: 110 passing, zero warnings.

---

## Task 1: Expose a node's children

`render` must find the branches hanging off the spine. `Tree` has no children accessor.

**Files:**
- Modify: `crates/sloop-harness/src/tree.rs` (beside `Tree::block`, ~line 219)

**Step 1: Write the failing test**

Add to `mod tests` in `crates/sloop-harness/src/tree.rs`:

```rust
#[test]
fn children_come_back_in_append_order() {
    let mut tree = Tree::new(ContentBlock::text("q"));
    let root = tree.root();
    let first = tree.append(root, Role::Assistant, ContentBlock::text("a")).unwrap();
    let second = tree.append(root, Role::Assistant, ContentBlock::text("b")).unwrap();

    assert_eq!(tree.children(root), Some(&[first, second][..]));
    assert_eq!(tree.children(first), Some(&[][..]));
}
```

**Step 2: Run it and watch it fail**

    cargo test -p sloop-harness children_come_back

Expected: a compile error, `no method named 'children' found for struct 'Tree'`.

**Step 3: Implement**

In `impl Tree`, directly after `block`:

```rust
/// This node's children, in append order.
///
/// Empty for a leaf. The order is the order the branches were created,
/// which is what lets a renderer put the spine first.
#[must_use]
pub fn children(&self, id: NodeId) -> Option<&[NodeId]> {
    Some(&self.node(id)?.children)
}
```

**Step 4: Verify**

    cargo test -p sloop-harness children_come_back

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/tree.rs
git commit -m "Expose a node's children"
```

---

## Task 2: Render a linear session

The spine only. Forks come in Task 4.

**Files:**
- Create: `crates/sloop-harness/src/transcript.rs`
- Modify: `crates/sloop-harness/src/main.rs` (add `mod transcript;` beside `mod tree;`)

**Step 1: Write the failing test**

Create `crates/sloop-harness/src/transcript.rs` with only the test module:

```rust
//! Rendering a conversation tree as markdown for the memory index.

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "a test reports failure by panicking")]
mod tests {
    use super::render;
    use crate::tree::{ContentBlock, Role, Tree};

    #[test]
    fn a_linear_session_renders_as_turns_under_a_title() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let reply = tree
            .append(root, Role::Assistant, ContentBlock::text("Expire on write."))
            .unwrap();
        tree.append(reply, Role::User, ContentBlock::text("Why not TTL?"))
            .unwrap();

        assert_eq!(
            render(&tree, "2026-09-13"),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.

## Turn 2

**user:** Why not TTL?
"
        );
    }
}
```

Add `mod transcript;` to `main.rs` beside `mod tree;`.

**Step 2: Run it and watch it fail**

    cargo test -p sloop-harness a_linear_session

Expected: a compile error, `cannot find function 'render'`.

**Step 3: Implement**

Above the test module in `transcript.rs`:

```rust
use std::fmt::Write as _;

use crate::tree::{ContentBlock, NodeId, Role, Tree};

/// The branch the session ended on.
///
/// `leaves` enumerates the arena, so it comes back in ascending creation
/// order and the last entry is the highest `NodeId`. A node is always
/// appended after its parent and never reparented, so that leaf is the one
/// most recently extended -- the branch that was live when the run stopped.
fn spine(tree: &Tree) -> Option<NodeId> {
    tree.leaves().last().copied()
}

/// The block's text, or `None` for a kind that is not written down.
///
/// Thinking blocks are deliberately dropped. The design doc records why: they
/// run several times the length of a response and state every idea
/// considered, including ones abandoned a paragraph later.
fn text_of(block: &ContentBlock) -> Option<&str> {
    match block {
        ContentBlock::Text { text } => Some(text),
        ContentBlock::Thinking { .. } => None,
    }
}

/// The title line: the first line of the opening user block.
fn title_of(tree: &Tree) -> String {
    tree.block(tree.root())
        .and_then(text_of)
        .and_then(|t| t.lines().next())
        .unwrap_or("untitled session")
        .to_owned()
}

/// One role-grouped run of blocks on a branch, and the nodes it came from.
struct Group {
    role: Role,
    ids: Vec<NodeId>,
}

/// Group consecutive same-role nodes, the way `Tree::replay` does, but
/// keeping the node ids so a renderer can find what forked off each one.
fn group(tree: &Tree, path: &[NodeId]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for id in path {
        let Some(role) = tree.role(*id) else { continue };
        match groups.last_mut() {
            Some(last) if last.role == role => last.ids.push(*id),
            Some(_) | None => groups.push(Group {
                role,
                ids: vec![*id],
            }),
        }
    }
    groups
}

/// Write one group's blocks as a labelled paragraph.
fn write_group(out: &mut String, tree: &Tree, g: &Group) {
    let label = match g.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    for id in &g.ids {
        let Some(text) = tree.block(*id).and_then(text_of) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let _ = writeln!(out, "**{label}:** {}\n", text.trim());
    }
}

/// Render the whole tree as one markdown document.
///
/// `captured` is an ISO date, and goes into frontmatter so the chunker can
/// put it on every chunk. The H1 is a heading rather than the note title --
/// the title the indexer uses is the filename stem.
#[must_use]
pub fn render(tree: &Tree, captured: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "---\ntype: transcript\ncaptured: {captured}\n---\n");
    let _ = writeln!(out, "# {}\n", title_of(tree));

    let Some(tip) = spine(tree) else {
        return out;
    };
    let Some(path) = tree.path(tip) else {
        return out;
    };

    let mut turn = 0;
    for g in group(tree, &path) {
        if g.role == Role::User {
            turn += 1;
            let _ = writeln!(out, "## Turn {turn}\n");
        }
        write_group(&mut out, tree, &g);
    }
    out
}
```

`Tree::role` does not exist yet. Add it beside `Tree::block` in `tree.rs`, with a test in the same style as Task 1:

```rust
/// This node's role.
#[must_use]
pub fn role(&self, id: NodeId) -> Option<Role> {
    Some(self.node(id)?.role)
}
```

**Step 4: Verify**

    cargo test -p sloop-harness a_linear_session
    cargo clippy --all-targets --all-features -- -D warnings

Expected: PASS, no warnings.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/transcript.rs crates/sloop-harness/src/tree.rs crates/sloop-harness/src/main.rs
git commit -m "Render a linear session as markdown"
```

---

## Task 3: Drop thinking blocks

The behaviour is already implemented by `text_of`. It has no test, so it is not yet real.

**Files:**
- Modify: `crates/sloop-harness/src/transcript.rs`

**Step 1: Write the test**

```rust
// Thinking blocks are the bulk of a real turn and the worst thing in it to
// match a query against: mid-derivation, and full of ideas dropped a
// paragraph later. The design doc records the argument.
#[test]
fn a_thinking_block_is_not_written_down() {
    let mut tree = Tree::new(ContentBlock::text("q"));
    let root = tree.root();
    let thought = tree
        .append(root, Role::Assistant, ContentBlock::thinking("weighing both", "sig"))
        .unwrap();
    tree.append(thought, Role::Assistant, ContentBlock::text("answer"))
        .unwrap();

    let out = render(&tree, "2026-09-13");

    assert!(!out.contains("weighing both"), "{out}");
    assert!(!out.contains("sig"), "{out}");
    assert!(out.contains("**assistant:** answer"), "{out}");
}
```

**Step 2: Run it**

    cargo test -p sloop-harness a_thinking_block_is_not

Expected: PASS immediately — this pins existing behaviour rather than driving new code.

**Step 3: Prove the test has teeth**

Temporarily change `text_of`'s `Thinking` arm to `Some(thinking)` and re-run. Expected: FAIL. Restore it, and re-run to confirm PASS. Do the mutation and the restore in **one** shell invocation so the working tree is never left mutated between turns.

**Step 4: Commit**

```bash
git add crates/sloop-harness/src/transcript.rs
git commit -m "Pin that thinking blocks stay out of the transcript"
```

---

## Task 4: Render forks as "Not continued"

The heart of the design. A branch off the spine becomes an `### Not continued` subsection, quoted, at the point it diverged.

**Files:**
- Modify: `crates/sloop-harness/src/transcript.rs`

**Step 1: Write the failing test**

```rust
// The whole safety story. "Not continued" lands in `heading_path`, which
// `append_pointer_line` prints and `parse_note` prepends to the embedded
// chunk text -- so a hit inside a discarded branch says so in both retrieval
// paths, with no label and no filter.
#[test]
fn a_branch_off_the_spine_renders_as_not_continued() {
    let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
    let root = tree.root();
    tree.append(root, Role::Assistant, ContentBlock::text("Expire on write."))
        .unwrap();
    tree.append(root, Role::Assistant, ContentBlock::text("TTL at 60s."))
        .unwrap();

    let out = render(&tree, "2026-09-13");

    assert!(out.contains("**assistant:** TTL at 60s."), "{out}");
    assert!(out.contains("### Not continued"), "{out}");
    // The spine is the highest-numbered leaf, so the *first* branch is the
    // one that gets set aside.
    let aside = out.split("### Not continued").nth(1).unwrap();
    assert!(aside.contains("> **assistant:** Expire on write."), "{out}");
}
```

**Step 2: Run it and watch it fail**

    cargo test -p sloop-harness a_branch_off_the_spine

Expected: FAIL — no `### Not continued` in the output.

**Step 3: Implement**

Add to `transcript.rs`:

```rust
/// Every leaf under `id`, including `id` when it is itself a leaf.
///
/// Filtering the global leaf list by ancestry rather than walking the subtree:
/// the tree is one session, and this keeps the traversal in `Tree::path`
/// where its termination argument already lives.
fn leaves_under(tree: &Tree, id: NodeId) -> Vec<NodeId> {
    let mut found = Vec::new();
    for leaf in tree.leaves() {
        if tree.path(leaf).is_some_and(|p| p.contains(&id)) {
            found.push(leaf);
        }
    }
    found
}

/// Render one abandoned branch, quoted, under its own heading.
///
/// The branch is followed to its own highest-numbered leaf, so a fork inside
/// a fork still renders as one readable run rather than stopping at the
/// divergence.
fn write_aside(out: &mut String, tree: &Tree, head: NodeId) {
    let Some(tip) = leaves_under(tree, head).last().copied() else {
        return;
    };
    let Some(full) = tree.path(tip) else { return };
    let Some(start) = full.iter().position(|n| *n == head) else {
        return;
    };

    let mut body = String::new();
    for g in group(tree, &full[start..]) {
        write_group(&mut body, tree, &g);
    }
    if body.trim().is_empty() {
        return;
    }

    let _ = writeln!(out, "### Not continued\n");
    for line in body.trim_end().lines() {
        if line.is_empty() {
            let _ = writeln!(out, ">");
        } else {
            let _ = writeln!(out, "> {line}");
        }
    }
    out.push('\n');
}
```

Then, in `render`, replace the group loop body so asides are emitted after each spine node:

```rust
    let on_spine: std::collections::HashSet<NodeId> = path.iter().copied().collect();

    let mut turn = 0;
    for g in group(tree, &path) {
        if g.role == Role::User {
            turn += 1;
            let _ = writeln!(out, "## Turn {turn}\n");
        }
        write_group(&mut out, tree, &g);
        for id in &g.ids {
            let Some(children) = tree.children(*id) else {
                continue;
            };
            for child in children {
                if !on_spine.contains(child) {
                    write_aside(&mut out, tree, *child);
                }
            }
        }
    }
```

`NodeId` must derive `Hash` for the set. Add `Hash` to its derive list in `tree.rs` if absent.

**Step 4: Verify**

    cargo test -p sloop-harness
    cargo clippy --all-targets --all-features -- -D warnings

Expected: all pass, no warnings.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/transcript.rs crates/sloop-harness/src/tree.rs
git commit -m "Render branches off the spine as Not continued"
```

---

## Task 5: Derive the filename

`parse_note` uses the filename stem as the title and prepends it to every chunk's embedded text, so it has to describe the session.

**Files:**
- Modify: `crates/sloop-harness/src/transcript.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn a_filename_is_the_date_and_a_slug_of_the_opening_question() {
    let tree = Tree::new(ContentBlock::text("How should the cache expire?"));

    assert_eq!(
        filename(&tree, "2026-09-13"),
        "2026-09-13-how-should-the-cache-expire.md"
    );
}

// A slug has to survive punctuation, casing and length without producing a
// path that is illegal or unreadable.
#[test]
fn a_filename_slug_drops_punctuation_and_caps_its_length() {
    let tree = Tree::new(ContentBlock::text(&format!("Why/When: {}", "x ".repeat(60))));
    let name = filename(&tree, "2026-09-13");

    assert!(!name.contains('/'), "{name}");
    assert!(!name.contains(':'), "{name}");
    assert!(name.starts_with("2026-09-13-why-when-x-x"), "{name}");
    assert!(name.len() <= 80, "{name} is {} chars", name.len());
}
```

**Step 2: Run and watch it fail**

    cargo test -p sloop-harness a_filename

Expected: compile error, `cannot find function 'filename'`.

**Step 3: Implement**

```rust
/// Longest slug allowed, in characters, before the date and extension.
const SLUG_CHARS: usize = 60;

/// A filesystem-safe, readable stem for this session.
///
/// The stem is not cosmetic: `index.rs` passes it to `parse_note` as the
/// title, and `parse_note` prepends it to the text of every chunk in the
/// file. A session id here would put a session id in every embedding.
#[must_use]
pub fn filename(tree: &Tree, captured: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = true;
    for c in title_of(tree).chars() {
        if c.is_ascii_alphanumeric() {
            slug.extend(c.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
        if slug.chars().count() >= SLUG_CHARS {
            break;
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        return format!("{captured}-session.md");
    }
    format!("{captured}-{slug}.md")
}
```

**Step 4: Verify**

    cargo test -p sloop-harness a_filename

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/transcript.rs
git commit -m "Derive a descriptive transcript filename"
```

---

## Task 6: Write the transcript at each turn boundary

**Files:**
- Modify: `crates/sloop-harness/src/main.rs`
- Modify: `crates/sloop-harness/Cargo.toml` (add `chrono`)

**Step 1: Add the dependency**

In `crates/sloop-harness/Cargo.toml`, beside the others, with a comment in the file's established style:

```toml
# The indexer reads `captured` out of frontmatter, so the transcript needs a
# date in the same format the rest of the workspace writes. Both sibling
# crates already depend on chrono.
chrono = "0.4"
```

**Step 2: Write the failing test**

In `main.rs`'s test module:

```rust
// A missing `transcripts` root is a configuration state, not a failure: the
// harness's job is the turn, and a run without the root configured must still
// complete. It has to say so, though -- silently not recording is the failure
// mode this whole slice exists to remove.
#[test]
fn a_missing_transcripts_root_is_reported_and_not_fatal() {
    let roots = Vec::new();
    assert_eq!(
        transcript_dir(&roots),
        Err("no root labelled 'transcripts' is configured, so this session is not being recorded"
            .to_owned())
    );
}
```

**Step 3: Run and watch it fail**

    cargo test -p sloop-harness a_missing_transcripts_root

Expected: compile error, `cannot find function 'transcript_dir'`.

**Step 4: Implement**

In `main.rs`:

```rust
/// The root label a session is written to.
const TRANSCRIPT_ROOT: &str = "transcripts";

/// Where to write this session, or why it is not being written.
///
/// Returns `Err` with a message rather than an `anyhow::Error` because the
/// caller reports it and carries on: a run without the root configured is a
/// legitimate way to use the harness, and failing the turn over it would be
/// the wrong trade.
fn transcript_dir(roots: &[config::Root]) -> Result<PathBuf, String> {
    roots
        .iter()
        .find(|r| r.label.as_str() == TRANSCRIPT_ROOT)
        .map(|r| r.dir.as_path().to_path_buf())
        .ok_or_else(|| {
            format!(
                "no root labelled '{TRANSCRIPT_ROOT}' is configured, so this session is not \
                 being recorded"
            )
        })
}

/// Write the whole session to its file, replacing what is there.
///
/// A full rewrite rather than an append: it is idempotent, so there is no
/// partial-write state to recover, and the indexer's manifest hash makes an
/// unchanged rewrite free. The daemon's watcher picks it up after
/// `WATCH_DEBOUNCE_SECS` -- the harness never touches the index itself, which
/// is what keeps the two from racing on the same rows.
fn record(dir: &Path, tree: &Tree) -> Result<PathBuf> {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let path = dir.join(transcript::filename(tree, &today));
    std::fs::write(&path, transcript::render(tree, &today))
        .with_context(|| format!("writing transcript {}", path.display()))?;
    Ok(path)
}
```

Add the imports `std::path::PathBuf`, `anyhow::Context`, and `sloop_memory_core::config`.

In `main`, resolve the directory once after `Api::from_env()`:

```rust
    let roots: Vec<config::Root> = match config::roots() {
        Ok(r) => r.iter().cloned().collect(),
        Err(e) => {
            line(&format!("note: {e}"))?;
            Vec::new()
        }
    };
    let record_to = match transcript_dir(&roots) {
        Ok(dir) => Some(dir),
        Err(why) => {
            line(&format!("note: {why}"))?;
            None
        }
    };
```

Then after **each** `graft` call — both of them — write:

```rust
    if let Some(dir) = &record_to {
        let path = record(dir, &tree)?;
        line(&format!("recorded {}", path.display()))?;
    }
```

`config::Root` must be `Clone` for the `cloned()` above. If it is not, hold the `Roots` value and pass `&Roots` instead; do not add a derive to the library for the binary's convenience.

**Step 5: Verify**

    cargo test -p sloop-harness
    cargo clippy --all-targets --all-features -- -D warnings

Expected: all pass, no warnings.

**Step 6: Commit**

```bash
git add crates/sloop-harness/
git commit -m "Write the session transcript at each turn boundary"
```

---

## Task 7: Prove a transcript chunks correctly

The failure mode here is silent: a transcript that parses to chunks with an empty `heading_path` and an empty `captured` still indexes, and every pointer into it renders bare. This test lives in `chunk.rs` because that is the code under test.

**Files:**
- Modify: `crates/sloop-memory-core/src/chunk.rs` (test module)

**Step 1: Write the failing test**

```rust
/// A transcript rendered by `sloop-harness` has to survive this parser with
/// its headings and frontmatter intact, or every pointer into it renders
/// without the framing the harness put there. The fixture mirrors
/// `transcript::render`; update both together.
#[test]
fn a_rendered_transcript_keeps_its_headings_and_frontmatter() {
    let source = "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write, with a 5s grace window.

### Not continued

> **assistant:** TTL at 60s, which blocks reads on revalidation.
";
    let note = parse_note("2026-09-13-how-should-the-cache-expire", source);

    assert_eq!(note.frontmatter.note_type, "transcript");
    assert_eq!(note.frontmatter.captured, "2026-09-13");

    let discarded = note
        .chunks
        .iter()
        .find(|c| c.text.contains("TTL at 60s"))
        .expect("the not-continued branch produced no chunk");

    assert_eq!(
        discarded.heading_path,
        "How should the cache expire? > Turn 1 > Not continued"
    );
    // The header is prepended to the embedded text, so the framing is inside
    // what gets matched and inside what the MCP tool renders back.
    assert!(discarded.text.starts_with(
        "2026-09-13-how-should-the-cache-expire > How should the cache expire? > Turn 1 > Not \
         continued"
    ), "{}", discarded.text);
}
```

**Step 2: Run it**

    cargo test -p sloop-memory-core a_rendered_transcript

If it fails, the renderer is wrong, not the test — the heading nesting or the frontmatter key does not match what `parse_note` accepts. Fix `transcript.rs` and re-run.

**Step 3: Commit**

```bash
git add crates/sloop-memory-core/src/chunk.rs
git commit -m "Prove a rendered transcript chunks with its framing intact"
```

---

## Task 8: Stop one root taking every pointer slot

**Files:**
- Modify: `crates/sloop-memory/src/daemon.rs:86-88`

**Step 1: Write the failing test**

In `sloop-memory`'s test module (the crate that owns pointer presentation):

```rust
// Transcripts are voluminous next to notes -- one session outweighs anything
// written down deliberately -- so a global sort by cosine hands every slot to
// whichever root is wordiest. Round-robin bounds that without a ranking model.
#[test]
fn no_single_root_takes_every_pointer_slot() {
    let hits = vec![
        pointer("transcripts", 0.95),
        pointer("transcripts", 0.94),
        pointer("transcripts", 0.93),
        pointer("notes", 0.80),
    ];

    let picked = take_round_robin(hits, 3);
    let labels: Vec<&str> = picked.iter().map(|p| p.source_type.as_str()).collect();

    assert_eq!(labels, vec!["transcripts", "notes", "transcripts"]);
}

#[test]
fn one_root_still_fills_the_block_when_it_is_the_only_one() {
    let hits = vec![
        pointer("notes", 0.90),
        pointer("notes", 0.85),
        pointer("notes", 0.80),
    ];

    assert_eq!(take_round_robin(hits, 3).len(), 3);
}
```

Write a small `fn pointer(source_type: &str, cosine: f32) -> Pointer` test helper that fills the remaining fields with empty strings.

**Step 2: Run and watch it fail**

    cargo test -p sloop-memory round_robin

Expected: compile error, `cannot find function 'take_round_robin'`.

**Step 3: Implement**

```rust
/// Take up to `limit` pointers, cycling through roots so one cannot take
/// every slot while another has hits above threshold.
///
/// Within a root the order stays by cosine, and a root that is the only one
/// with hits still fills the block. This bounds crowding rather than solving
/// it: flat retrieval has no notion of level, so every chunk competes as a
/// peer. Usage feedback and the wikilink graph are the real answers, and both
/// are later slices.
fn take_round_robin(mut hits: Vec<Pointer>, limit: usize) -> Vec<Pointer> {
    hits.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));

    let mut by_root: Vec<Vec<Pointer>> = Vec::new();
    for hit in hits {
        match by_root
            .iter_mut()
            .find(|group| group.first().is_some_and(|p| p.source_type == hit.source_type))
        {
            Some(group) => group.push(hit),
            None => by_root.push(vec![hit]),
        }
    }

    let mut picked = Vec::new();
    let mut round = 0;
    while picked.len() < limit {
        let mut progressed = false;
        for group in &by_root {
            if let Some(hit) = group.get(round) {
                picked.push(hit.clone());
                progressed = true;
                if picked.len() == limit {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
        round += 1;
    }
    picked
}
```

Replace `daemon.rs:86-88`:

```rust
    best.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
    best.truncate(config::HOOK_MAX_HITS);
```

with:

```rust
    let best = take_round_robin(best, config::HOOK_MAX_HITS);
```

**Step 4: Verify**

    cargo test -p sloop-memory
    cargo clippy --all-targets --all-features -- -D warnings

Expected: all pass, no warnings.

**Step 5: Commit**

```bash
git add crates/sloop-memory/src/daemon.rs
git commit -m "Round-robin pointer slots across roots"
```

---

## Task 9: Frame transcripts as records, not conclusions

**Files:**
- Modify: `crates/sloop-memory/src/main.rs:143-198`

**Step 1: Write the failing test**

```rust
// A transcript pointer must not read like a note pointer. A note is something
// someone decided to write down; a transcript is a record of working it out,
// and it contains approaches that were tried and dropped. The reader has to
// know which one it is looking at before it opens the file.
#[test]
fn a_transcript_pointer_is_framed_as_a_record() {
    let block = render_pointer_block(&[pointer("transcripts", "2026-09-13-cache.md")]);

    assert!(block.contains("record of a conversation"), "{block}");
    assert!(block.contains("read around"), "{block}");
    assert!(!block.contains(NOTES_PREAMBLE), "{block}");
}
```

**Step 2: Run and watch it fail**

    cargo test -p sloop-memory a_transcript_pointer_is_framed

Expected: FAIL — the transcript pointer falls through to `NOTES_PREAMBLE`.

**Step 3: Implement**

Add beside the other two constants:

```rust
/// The root label whose pointers are conversations rather than notes.
const TRANSCRIPT_LABEL: &str = "transcripts";

/// Framing for pointers from the `transcripts` root: a record of working
/// something out, not a conclusion about it. A heading of `Not continued`
/// marks a branch that was tried and dropped, and the surrounding turns are
/// the reason it was -- which is why this says to read around the hit rather
/// than to read the hit.
const TRANSCRIPT_PREAMBLE: &str = "Relevant session transcript pointers. Each is a record \
    of a conversation, not a conclusion: read around the hit for the context that gives it \
    meaning. A `Not continued` heading marks an approach that was tried and dropped -- do \
    not read it as a finding.\n";
```

Extend the grouping in `render_pointer_block` to three buckets, keeping the existing rule that whichever group appeared first in the input renders first.

**Step 4: Verify**

    cargo test -p sloop-memory
    cargo clippy --all-targets --all-features -- -D warnings

Expected: all pass, no warnings.

**Step 5: Commit**

```bash
git add crates/sloop-memory/src/main.rs
git commit -m "Frame transcript pointers as records"
```

---

## Task 10: Correct the documentation

The README still argues the label is what makes indexing safe. Once this ships that describes a design the repo knowingly rejected.

**Files:**
- Modify: `README.md:30-40`
- Modify: `docs/architecture.md` ("What this does not cover")
- Modify: `crates/sloop-harness/README.md` ("Not built yet")

**Step 1: Rewrite the README's claim**

Replace the paragraph beginning "`sloop-harness` is meant to handle the other half" with the argument that survived: a transcript is indexed as a record rather than as facts, forks render as headings so the frame travels in the pointer and the embedded text, and the branching harness is justified by branching being useful rather than by being a precondition for safe indexing. Say plainly that the label was tried and did not carry the weight, and point at the design doc.

**Step 2: Update the two "not built" lists**

`indexing a transcript into sloop-memory` is done. What remains: `tool_use`/`tool_result` and their fork-validity rule, retries and backoff, cache-hit instrumentation, usage feedback, and the wikilink graph.

**Step 3: State what `Status` is now for**

In `crates/sloop-harness/README.md`, record that the label marks which branch is live for the harness's own use and is deliberately not read by the indexer, with the reason.

**Step 4: Verify**

    cargo test --workspace
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --all -- --check

Expected: all clean.

**Step 5: Commit**

```bash
git add README.md docs/architecture.md crates/sloop-harness/README.md
git commit -m "Retire the label thesis from the docs"
```

---

## Finishing

Run the full suite and open the PR:

    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --workspace

Then use @superpowers:finishing-a-development-branch.

**Manual check worth doing once, because no test covers the live path:** set `SLOOP_MEMORY_ROOTS` to include `transcripts=<dir>`, run the daemon, run `cargo run -p sloop-harness -- "How should the cache expire?"`, wait five seconds, and confirm `sloop-memory search "cache expire"` returns a pointer into the transcript with a `Not continued` heading path.
