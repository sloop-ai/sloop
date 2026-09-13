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

/// Write one group's blocks as labelled paragraphs.
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
}
