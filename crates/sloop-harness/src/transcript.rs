//! Rendering a conversation tree as markdown for the memory index.

use std::collections::HashSet;
use std::fmt::Write as _;

use crate::tree::{ContentBlock, NodeId, Role, Tree};

/// The block's text, or `None` for a kind that is not written down.
///
/// Thinking blocks are deliberately dropped. They run several times the
/// length of a response and state every idea considered, including ones
/// abandoned a paragraph later, so indexing them would multiply the corpus
/// while making a query match mid-derivation hedging.
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
/// keeping the node ids so a later step can find what forked off each one.
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

/// The run of fence characters this line opens or closes with, or `None` when
/// the line is not a fence.
///
/// The predicate has to be the chunker's, character for character: three or
/// more backticks or tildes after leading whitespace, which is what
/// `parse_note` toggles its fence flag on. Anything looser or tighter here and
/// the balancing below counts a different set of lines than the reader does.
///
/// The whole run comes back rather than just three characters so that a closer
/// this module emits matches the opener it is closing. The chunker does not
/// care -- three is enough to toggle it -- but a four-backtick block wrapping a
/// three-backtick one is real markdown, and closing it with three would end the
/// wrong block for every other reader.
fn fence_marker(line: &str) -> Option<&str> {
    let line = line.trim_start();
    let first = line.chars().next()?;
    if first != '`' && first != '~' {
        return None;
    }
    let run = line.len() - line.trim_start_matches(first).len();
    if run < 3 {
        return None;
    }
    Some(&line[..run])
}

/// The marker that would close this text's last unterminated fence, or `None`
/// when its fences already balance.
fn unclosed_fence(text: &str) -> Option<&str> {
    let mut open = None;
    for line in text.lines() {
        let Some(marker) = fence_marker(line) else {
            continue;
        };
        open = match open {
            None => Some(marker),
            Some(_) => None,
        };
    }
    open
}

/// Write one group's blocks as labelled paragraphs.
///
/// The label stays inline with the text -- `**assistant:** ...` -- unless the
/// block opens on a code fence, which is given its own line. Always breaking
/// after the label would drop the condition, at the cost of a line and a token
/// on every block in a corpus that is overwhelmingly prose and is read by an
/// agent paying for each one. One string comparison buys the common case back.
///
/// The break is not typography. `parse_note` toggles a fence flag on any line
/// that starts with a fence and ignores headings while it is set, so an opening
/// fence swallowed into `**assistant:** ` never toggles while the closer on its
/// own line does. The flag is then stuck on for the rest of the file and every
/// heading below reads as code -- including the `### Not continued` that is the
/// only thing marking an abandoned branch as abandoned.
///
/// Moving the label is not sufficient on its own, which is why the block is
/// also balanced. A turn interrupted mid-fence already has its opening fence on
/// its own line; what it lacks is the closer, and an unterminated fence leaks
/// into the next heading exactly the same way.
fn write_group(out: &mut String, tree: &Tree, g: &Group) {
    let label = match g.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    for id in &g.ids {
        let Some(text) = tree.block(*id).and_then(text_of) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let opens_on_fence = text
            .lines()
            .next()
            .is_some_and(|l| fence_marker(l).is_some());
        let gap = if opens_on_fence { '\n' } else { ' ' };
        let _ = writeln!(out, "**{label}:**{gap}{text}");
        if let Some(marker) = unclosed_fence(text) {
            let _ = writeln!(out, "{marker}");
        }
        out.push('\n');
    }
}

/// Every leaf under `id`, including `id` when it is itself a leaf.
///
/// Filtering the global leaf list by ancestry rather than walking the
/// subtree: the tree is one session, and this keeps traversal in
/// [`Tree::path`], where the termination argument already lives.
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
/// The heading level is the safety mechanism, not decoration. The chunker
/// turns headings into a `heading_path` that every retrieval path prints
/// verbatim, so a level-3 heading here puts `Turn N > Not continued` on the
/// chunk and no consumer has to remember to say the idea was dropped.
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
        // A bare `>` rather than a bare newline, so the branch stays one
        // blockquote instead of several. Not for the heading's sake: the
        // indexer keys heading_path per *section*, so a branch long enough to
        // split across chunks says "Not continued" on every one of them. This
        // is about the branch reading as one passage when a pointer is
        // followed into it.
        if line.is_empty() {
            let _ = writeln!(out, ">");
        } else {
            let _ = writeln!(out, "> {line}");
        }
    }
    out.push('\n');
}

/// Write the live branch, numbering turns from its user blocks, and every
/// branch that forked off it as an aside.
///
/// `path` is the spine, root first, as the caller's tip resolves it.
fn write_spine(out: &mut String, tree: &Tree, path: &[NodeId]) {
    let on_spine: HashSet<NodeId> = path.iter().copied().collect();

    let mut turn = 0;
    // Asides are held back to the end of the turn they forked off. Emitting
    // one the moment its parent is written would drop a `### Not continued`
    // between the user block and the reply, splitting the turn it belongs to.
    let mut asides = String::new();
    for g in group(tree, path) {
        if g.role == Role::User {
            out.push_str(&asides);
            asides.clear();
            turn += 1;
            let _ = writeln!(out, "## Turn {turn}\n");
        }
        write_group(out, tree, &g);
        for id in &g.ids {
            let Some(children) = tree.children(*id) else {
                continue;
            };
            for child in children {
                if !on_spine.contains(child) {
                    write_aside(&mut asides, tree, *child);
                }
            }
        }
    }
    out.push_str(&asides);
}

/// Longest slug allowed, in characters, before the date and extension.
///
/// 60 keeps the whole name well inside the 255-byte component limit every
/// filesystem this runs on enforces, with room for the date, the extension,
/// and a title whose first line is a paragraph.
const SLUG_CHARS: usize = 60;

/// A filesystem-safe, readable stem for this session.
///
/// The stem is not cosmetic: the indexer passes it to the chunker as the note
/// title, and the chunker prepends it to the text of every chunk in the file.
/// A session id here would put a session id in every embedding.
///
/// ASCII-only, which mangles a title in any other script down to the fallback.
/// The trade is taken for the manifest's sake rather than for portability:
/// macOS normalizes filenames to NFD while the string written here is NFC, so
/// a non-ASCII name read back off the disk is not byte-equal to the one
/// written. The indexer keys its manifest on the relative path, so the same
/// session would look like two files. A dull name indexes correctly; a pretty
/// one indexes twice.
///
/// The fallback is `{captured}-session.md` rather than a bare date: the stem
/// is a chunk prefix, and "2026-09-13" alone reads as a date stamp on the
/// content instead of as what the file is.
///
/// Two sessions whose titles slug the same on the same day land on the same
/// name, and the second overwrites the first. That is inherent to naming by
/// content rather than by id, and it is the same property that makes the
/// caller's rewrite idempotent.
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

/// Render the whole tree as one markdown document.
///
/// `captured` is an ISO date. It goes into frontmatter under the key `type`
/// and `captured`, which is what `sloop-memory-core`'s `parse_frontmatter`
/// reads -- the key is `type`, not `note_type`.
///
/// The H1 is a heading rather than the note's title: the indexer takes the
/// title from the filename stem, and headings become the `heading_path` that
/// every chunk carries.
///
/// `tip` is the live branch, and it is an argument because the renderer cannot
/// work it out. Recency does not answer it: the harness forks and regenerates,
/// so the branch it threw away is the one holding the highest node ids, and any
/// rule reading "newest" off the arena names the wrong side of every fork. The
/// caller grafted the branches and is the only party that knows which one it
/// kept.
///
/// A `tip` that is not a leaf is a legal thing to ask for and means what it
/// says: the spine ends there, and whatever hangs below it renders as an aside
/// like any other branch that was not continued.
///
/// A `tip` that is not a node of this tree returns `None`, matching what
/// [`Tree`] does with a foreign [`NodeId`] everywhere else. Rendering some
/// other branch instead would produce a document that looks right and records
/// the wrong conversation, which is exactly the failure this argument exists to
/// remove.
#[must_use]
pub fn render(tree: &Tree, tip: NodeId, captured: &str) -> Option<String> {
    let path = tree.path(tip)?;
    let mut out = String::new();
    let _ = writeln!(out, "---\ntype: transcript\ncaptured: {captured}\n---\n");
    let _ = writeln!(out, "# {}\n", title_of(tree));
    write_spine(&mut out, tree, &path);

    // Every paragraph is written with a blank line after it, so the last one
    // leaves the document ending in a blank. The indexer splits on blank
    // lines, and a trailing one is an empty chunk.
    out.truncate(out.trim_end().len());
    out.push('\n');
    Some(out)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "a test reports failure by panicking")]
mod tests {
    use sloop_memory_core::chunk::{parse_note, ParsedNote};

    use super::{filename, render};
    use crate::tree::{ContentBlock, Role, Tree};

    #[test]
    fn a_filename_is_the_date_and_a_slug_of_the_opening_question() {
        let tree = Tree::new(ContentBlock::text("How should the cache expire?"));

        assert_eq!(
            filename(&tree, "2026-09-13"),
            "2026-09-13-how-should-the-cache-expire.md"
        );
    }

    /// Casing and punctuation both have to go: the stem is a path component,
    /// and `/` would silently write into a subdirectory -- or fail -- rather
    /// than name a file.
    #[test]
    fn punctuation_and_casing_never_reach_the_name() {
        let tree = Tree::new(ContentBlock::text("Cache/TTL: Why NOT?"));

        assert_eq!(
            filename(&tree, "2026-09-13"),
            "2026-09-13-cache-ttl-why-not.md"
        );
    }

    /// A title is a whole first line, and a first line can be a paragraph.
    /// The cap is what keeps the name inside every filesystem's component
    /// limit; it truncates mid-word rather than at a word boundary, which is
    /// ugly and is the price of the bound being on the name, not the prose.
    #[test]
    fn a_long_title_is_cut_at_the_cap() {
        let tree = Tree::new(ContentBlock::text(
            "How should the cache expire when the process restarts and the disk fills up?",
        ));

        assert_eq!(
            filename(&tree, "2026-09-13"),
            "2026-09-13-how-should-the-cache-expire-when-the-process-restarts-and-th.md"
        );
    }

    /// All-punctuation slugs to nothing, and the fallback has to be a name a
    /// person can still read as a transcript rather than a bare date.
    #[test]
    fn a_title_of_pure_punctuation_falls_back() {
        let tree = Tree::new(ContentBlock::text("!!! ???"));

        assert_eq!(filename(&tree, "2026-09-13"), "2026-09-13-session.md");
    }

    /// The slug is ASCII-only, so a title in another script reaches the same
    /// fallback -- see [`filename`] for why that trade is taken deliberately.
    #[test]
    fn a_non_ascii_title_falls_back_too() {
        let tree = Tree::new(ContentBlock::text("キャッシュはどう失効させるべき?"));

        assert_eq!(filename(&tree, "2026-09-13"), "2026-09-13-session.md");
    }

    /// An empty opening block has no first line at all, so it never reaches
    /// the slug: `title_of` substitutes its own placeholder first, and that is
    /// what gets slugged. Distinct from the fallback above, and worth pinning
    /// so a change to either one cannot quietly take over the other's case.
    #[test]
    fn an_empty_opening_block_is_named_from_the_placeholder_title() {
        let tree = Tree::new(ContentBlock::text(""));

        assert_eq!(
            filename(&tree, "2026-09-13"),
            "2026-09-13-untitled-session.md"
        );
    }

    #[test]
    fn a_linear_session_renders_as_turns_under_a_title() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let reply = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();
        let tip = tree
            .append(reply, Role::User, ContentBlock::text("Why not TTL?"))
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
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

    /// Dropping thinking has to leave nothing behind, not even the blank
    /// paragraph an empty block would render as, so this compares the whole
    /// document rather than asserting the prose is absent.
    #[test]
    fn a_thinking_block_leaves_no_trace() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let thinking = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::thinking("Weighing LRU against TTL.", "ErUBCkYIBRgCIkA="),
            )
            .unwrap();
        let tip = tree
            .append(
                thinking,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.
"
        );
    }

    /// The aside has to follow the abandoned branch to its own end.
    ///
    /// Stopping at the divergence would quote the rejected proposal without
    /// the exchange that rejected it, which is the one shape that makes a
    /// discarded idea read as a standing one.
    #[test]
    fn an_abandoned_branch_is_quoted_to_its_own_tip() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let aside = tree
            .append(root, Role::Assistant, ContentBlock::text("TTL at 60s."))
            .unwrap();
        let question = tree
            .append(
                aside,
                Role::User,
                ContentBlock::text("Does that handle writes?"),
            )
            .unwrap();
        tree.append(
            question,
            Role::Assistant,
            ContentBlock::text("No, stale until expiry."),
        )
        .unwrap();
        let tip = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.

### Not continued

> **assistant:** TTL at 60s.
>
> **user:** Does that handle writes?
>
> **assistant:** No, stale until expiry.
"
        );
    }

    /// The tree `main` actually builds, in the order it builds it.
    ///
    /// The kept turn is grafted first and the fork hangs the regenerated turn
    /// off it afterwards, so the abandoned branch always holds the higher node
    /// ids. Any spine picked by recency therefore picks the branch that was
    /// thrown away, and the whole document comes out inverted: the rejected
    /// answer in the turn body, the kept one disowned under `Not continued`.
    /// The assertion is the whole document because the defect is which side of
    /// that heading each answer lands on.
    ///
    /// This replaces an earlier test that grafted the abandoned branch first,
    /// which passed under a recency rule and under this one alike.
    #[test]
    fn the_kept_branch_is_the_spine_even_though_the_fork_is_newer() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let kept = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();
        tree.append(root, Role::Assistant, ContentBlock::text("TTL at 60s."))
            .unwrap();

        assert_eq!(
            render(&tree, kept, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.

### Not continued

> **assistant:** TTL at 60s.
"
        );
    }

    /// An interior tip ends the spine where the caller says it does, and the
    /// continuation below it is a branch that was not continued like any
    /// other. Nothing is dropped, and nothing is silently promoted back onto
    /// the spine.
    #[test]
    fn a_tip_that_is_not_a_leaf_ends_the_spine_there() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let reply = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();
        tree.append(reply, Role::User, ContentBlock::text("Why not TTL?"))
            .unwrap();

        assert_eq!(
            render(&tree, reply, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.

### Not continued

> **user:** Why not TTL?
"
        );
    }

    /// A tip from another tree renders nothing at all.
    ///
    /// The alternative -- falling back to some branch of this tree -- is the
    /// defect this argument was added to remove: a document that reads as a
    /// faithful transcript while recording a conversation nobody had.
    #[test]
    fn a_tip_this_tree_never_minted_renders_nothing() {
        let mut other = Tree::new(ContentBlock::text("A different session."));
        let root = other.root();
        let foreign = other
            .append(root, Role::Assistant, ContentBlock::text("A reply."))
            .unwrap();
        let tree = Tree::new(ContentBlock::text("How should the cache expire?"));

        assert_eq!(render(&tree, foreign, "2026-09-13"), None);
    }

    /// A branch of nothing but thinking must not leave the heading behind.
    ///
    /// Thinking is dropped, so such an aside has no body -- and a bare
    /// `### Not continued` is worse than nothing: it chunks into a section
    /// that disowns content it does not contain.
    #[test]
    fn a_branch_that_renders_to_nothing_gets_no_heading() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        tree.append(
            root,
            Role::Assistant,
            ContentBlock::thinking("Weighing LRU against TTL.", "ErUBCkYIBRgCIkA="),
        )
        .unwrap();
        let tip = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.
"
        );
    }

    // ---- code fences ----------------------------------------------------
    //
    // The four tests below pin the bytes this renderer emits around a code
    // fence. What those bytes have to be *worth* is asserted separately, under
    // "the renderer/chunker contract" at the end of this module, which puts
    // real rendered output through the real `parse_note`.

    /// A reply that opens on a fence puts the fence on its own line.
    ///
    /// Inline, the opening fence would be swallowed into `**assistant:** ` and
    /// never toggle the chunker's fence flag, while its closer on the next line
    /// would -- leaving the flag stuck on and every heading after it, this
    /// document's own `### Not continued` included, read as code.
    #[test]
    fn a_reply_opening_on_a_fence_starts_its_own_line() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let reply = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text(
                    "```rust\nfn expire(entry: &mut Entry) {\n    entry.stale = true;\n}\n```",
                ),
            )
            .unwrap();
        let tip = tree
            .append(
                reply,
                Role::User,
                ContentBlock::text("Does that handle reads?"),
            )
            .unwrap();
        tree.append(root, Role::Assistant, ContentBlock::text("TTL at 60s."))
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:**
```rust
fn expire(entry: &mut Entry) {
    entry.stale = true;
}
```

### Not continued

> **assistant:** TTL at 60s.

## Turn 2

**user:** Does that handle reads?
"
        );
    }

    /// The same rule on the user side. An opening prompt that is a pasted code
    /// block also makes the H1 the fence line itself -- which is a heading, not
    /// a fence, because `#` comes first -- so the turn body is the only place
    /// the fence can do damage.
    #[test]
    fn a_prompt_opening_on_a_fence_starts_its_own_line() {
        let mut tree = Tree::new(ContentBlock::text(
            "```rust\nfn expire(entry: &mut Entry) {}\n```",
        ));
        let root = tree.root();
        let tip = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("That never marks the entry stale."),
            )
            .unwrap();
        tree.append(
            root,
            Role::Assistant,
            ContentBlock::text("Looks correct to me."),
        )
        .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# ```rust

## Turn 1

**user:**
```rust
fn expire(entry: &mut Entry) {}
```

**assistant:** That never marks the entry stale.

### Not continued

> **assistant:** Looks correct to me.
"
        );
    }

    /// A turn interrupted inside a fence gets the fence closed for it.
    ///
    /// Steering mid-turn is a first-class shape here -- the partial reply is
    /// kept and a user block is appended under it -- so a response that stops
    /// inside a code block is a transcript this renderer has to emit. Moving
    /// the label off the line does nothing for it: the fence already starts its
    /// own line and is simply never closed, which swallows the rest of the
    /// document just as thoroughly.
    #[test]
    fn a_turn_interrupted_inside_a_fence_is_closed() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let partial = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Like this:\n```rust\nfn expire(entry: &mut Entry) {"),
            )
            .unwrap();
        let steer = tree
            .append(partial, Role::User, ContentBlock::text("Stop, wrong file."))
            .unwrap();
        let tip = tree
            .append(
                steer,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();
        tree.append(steer, Role::Assistant, ContentBlock::text("TTL at 60s."))
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Like this:
```rust
fn expire(entry: &mut Entry) {
```

## Turn 2

**user:** Stop, wrong file.

**assistant:** Expire on write.

### Not continued

> **assistant:** TTL at 60s.
"
        );
    }

    /// An aside needs none of this and is left alone.
    ///
    /// Every line of an aside is prefixed with `> `, so a fenced line inside
    /// one does not start with a fence after `trim_start` and never toggles the
    /// chunker's flag whether it balances or not. The quoting is the whole
    /// protection, and this pins that it stays the whole protection.
    ///
    /// The aside is built by the same [`write_group`], so it picks up the line
    /// break and the closer anyway. That is worth having for a different
    /// reason: a blockquote holding an unterminated fence is malformed markdown
    /// to anyone reading the file, chunker or not.
    #[test]
    fn a_fence_inside_an_aside_stays_quoted() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let reply = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
            .unwrap();
        let tip = tree
            .append(
                reply,
                Role::User,
                ContentBlock::text("Does that handle reads?"),
            )
            .unwrap();
        tree.append(
            root,
            Role::Assistant,
            ContentBlock::text("```rust\nfn ttl() {}"),
        )
        .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:** Expire on write.

### Not continued

> **assistant:**
> ```rust
> fn ttl() {}
> ```

## Turn 2

**user:** Does that handle reads?
"
        );
    }

    /// The closer matches the marker it is closing.
    ///
    /// The chunker would be satisfied by backticks either way -- it toggles on
    /// both markers and never pairs them -- so nothing about the abandonment
    /// heading depends on this. Every other reader of the file does: closing a
    /// tilde block with backticks leaves both fences open.
    #[test]
    fn an_unclosed_tilde_fence_is_closed_with_tildes() {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();
        let tip = tree
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("~~~rust\nfn expire() {"),
            )
            .unwrap();

        assert_eq!(
            render(&tree, tip, "2026-09-13").unwrap(),
            "\
---
type: transcript
captured: 2026-09-13
---

# How should the cache expire?

## Turn 1

**user:** How should the cache expire?

**assistant:**
~~~rust
fn expire() {
~~~
"
        );
    }

    // ---- the renderer/chunker contract -----------------------------------
    //
    // Everything above pins bytes. These pin what the chunker makes of them,
    // which is the property the transcript design actually rests on: a
    // `### Not continued` heading has to reach `heading_path`, because that is
    // the only thing marking a dropped branch as dropped, on both retrieval
    // paths and with no consumer having to remember a filter.
    //
    // It is a property of the renderer and the chunker together, so neither
    // crate's own tests can see it. `parse_note` is public for exactly this.
    // Nothing here is a fixture: the markdown under test is what `record`
    // wrote, built by the `graft`/`fork_point` sequence `main` uses.

    /// Replay `main`'s build order over two canned turns, then chunk the file
    /// it wrote.
    ///
    /// The turns are canned because the API is not. Everything downstream of
    /// them is the real path -- `main` grafts the kept turn, asks `fork_point`
    /// where the second hangs, grafts it there, and records with the kept tip
    /// still live -- so a renderer change cannot pass here against a shape the
    /// harness does not emit.
    ///
    /// The title comes back from the filename `record` chose, not from a
    /// constant, because `parse_note` takes the filename stem and prepends it
    /// to every chunk's embedded text.
    fn chunked_session(
        prompt: &str,
        kept: Vec<ContentBlock>,
        abandoned: Vec<ContentBlock>,
    ) -> ParsedNote {
        let dir = tempfile::tempdir().unwrap();
        let mut tree = Tree::new(ContentBlock::text(prompt));
        let root = tree.root();

        let first = crate::graft(&mut tree, root, kept).unwrap();
        let live = *first.last().unwrap();
        let (fork_at, _) = crate::fork_point(&tree, root, &first).unwrap();
        crate::graft(&mut tree, fork_at, abandoned).unwrap();

        let path = crate::record(dir.path(), &tree, live).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        parse_note(&stem, &std::fs::read_to_string(&path).unwrap())
    }

    #[track_caller]
    fn abandonment_survives(note: &ParsedNote) {
        let paths: Vec<&str> = note
            .chunks
            .iter()
            .map(|c| c.heading_path.as_str())
            .collect();
        assert!(
            paths.iter().any(|p| p.ends_with("Not continued")),
            "the abandoned branch lost its heading: {paths:?}"
        );
    }

    /// The plain shape, with no fence anywhere.
    ///
    /// Without it the three below prove only that fences do no *additional*
    /// damage, and a renderer that stopped emitting the heading at all would
    /// pass every one of them.
    #[test]
    fn a_dropped_branch_reaches_the_chunker_as_a_heading() {
        let note = chunked_session(
            "How should the cache expire?",
            vec![ContentBlock::text("Expire on write.")],
            vec![ContentBlock::text("TTL at 60s.")],
        );

        abandonment_survives(&note);
    }

    /// Frontmatter has to survive the round trip too. Empty is the failure
    /// mode that shows up nowhere else: the pointer renderer reads both of
    /// these, and a blank one degrades the pointer silently rather than
    /// failing.
    #[test]
    fn a_recorded_session_carries_its_type_and_date_through_the_chunker() {
        let note = chunked_session(
            "How should the cache expire?",
            vec![ContentBlock::text("Expire on write.")],
            vec![ContentBlock::text("TTL at 60s.")],
        );

        assert_eq!(note.frontmatter.note_type, "transcript");
        assert_eq!(
            note.frontmatter.captured,
            chrono::Local::now().format("%Y-%m-%d").to_string()
        );
    }

    /// A reply that opens on a code fence.
    ///
    /// The regression this exists for: rendered inline, the opening fence was
    /// swallowed into `**assistant:** ` and never toggled the chunker's fence
    /// flag while its closer did, so the flag stayed set and the whole rest of
    /// the document -- `### Not continued` included -- chunked as code.
    #[test]
    fn a_reply_opening_on_a_fence_leaves_the_heading_below_it_intact() {
        let note = chunked_session(
            "How should the cache expire?",
            vec![ContentBlock::text(
                "```rust\nfn expire(entry: &mut Entry) {\n    entry.stale = true;\n}\n```",
            )],
            vec![ContentBlock::text("TTL at 60s.")],
        );

        abandonment_survives(&note);
    }

    /// A turn interrupted inside a fence, which never closes it. Putting the
    /// label on its own line does nothing here -- the fence already starts one
    /// -- so the renderer has to close the block itself.
    #[test]
    fn a_turn_interrupted_inside_a_fence_leaves_the_heading_below_it_intact() {
        let note = chunked_session(
            "How should the cache expire?",
            vec![ContentBlock::text(
                "```rust\nfn expire(entry: &mut Entry) {",
            )],
            vec![ContentBlock::text("TTL at 60s.")],
        );

        abandonment_survives(&note);
    }

    /// An interrupted fence must not swallow the turns that follow it either.
    ///
    /// Distinct from the test above: the aside is written before the rest of
    /// the spine, so a leaked fence can leave `### Not continued` intact and
    /// still erase every heading after it. Nothing downstream would report
    /// that -- the file simply chunks to fewer chunks than it has sections.
    #[test]
    fn an_interrupted_fence_does_not_swallow_the_turn_after_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let root = tree.root();

        let first = crate::graft(
            &mut tree,
            root,
            vec![ContentBlock::text(
                "```rust\nfn expire(entry: &mut Entry) {",
            )],
        )
        .unwrap();
        let (fork_at, _) = crate::fork_point(&tree, root, &first).unwrap();
        crate::graft(&mut tree, fork_at, vec![ContentBlock::text("TTL at 60s.")]).unwrap();

        // `graft` appends assistant blocks, and turns are numbered from user
        // ones, so the follow-up is appended directly in order to open Turn 2.
        let live = tree
            .append(
                *first.last().unwrap(),
                Role::User,
                ContentBlock::text("Does that handle reads?"),
            )
            .unwrap();

        let path = crate::record(dir.path(), &tree, live).unwrap();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let note = parse_note(&stem, &std::fs::read_to_string(&path).unwrap());

        let paths: Vec<&str> = note
            .chunks
            .iter()
            .map(|c| c.heading_path.as_str())
            .collect();
        assert!(
            paths.iter().any(|p| p.ends_with("Turn 2")),
            "the turn after the interrupted one was swallowed: {paths:?}"
        );
    }

    /// The fence in the *opening prompt* rather than in a reply. That block is
    /// rendered twice -- once as the H1, once in the turn body -- so it is the
    /// one input that can leave a fence open in two places.
    #[test]
    fn an_opening_prompt_of_pasted_code_leaves_the_heading_below_it_intact() {
        let note = chunked_session(
            "```rust\nfn expire(entry: &mut Entry) {}\n```",
            vec![ContentBlock::text("That never marks the entry stale.")],
            vec![ContentBlock::text("Looks correct to me.")],
        );

        abandonment_survives(&note);
    }

    /// The abandoned branch alone must not be able to satisfy the assertion.
    ///
    /// A renderer that lost the spine and quoted everything would leave a
    /// `Not continued` heading in place while recording the wrong conversation
    /// as live, which is the defect this slice already shipped once.
    #[test]
    fn the_live_branch_is_not_itself_marked_not_continued() {
        let note = chunked_session(
            "How should the cache expire?",
            vec![ContentBlock::text("Expire on write.")],
            vec![ContentBlock::text("TTL at 60s.")],
        );

        let live: Vec<&str> = note
            .chunks
            .iter()
            .filter(|c| c.text.contains("Expire on write."))
            .map(|c| c.heading_path.as_str())
            .collect();
        assert!(!live.is_empty(), "the live reply reached no chunk at all");
        assert!(
            !live.iter().any(|p| p.ends_with("Not continued")),
            "the live branch was recorded as abandoned: {live:?}"
        );
    }
}
