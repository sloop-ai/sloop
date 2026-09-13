//! Rendering a conversation tree as markdown for the memory index.
//!
//! Nothing in the binary calls [`render`] yet -- the step that names a file
//! and writes it is a later one -- so under `not(test)` the whole module is
//! dead. The expectation covers the module rather than each item so that it
//! comes off in one place once that caller lands.
#![cfg_attr(not(test), expect(dead_code, reason = "see above"))]

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

/// Write the branch the session ended on, numbering turns from its user blocks.
///
/// A tree whose spine cannot be walked contributes nothing rather than
/// aborting the render: the frontmatter and title are already worth indexing.
fn write_spine(out: &mut String, tree: &Tree) {
    let Some(tip) = spine(tree) else { return };
    let Some(path) = tree.path(tip) else { return };

    let mut turn = 0;
    for g in group(tree, &path) {
        if g.role == Role::User {
            turn += 1;
            let _ = writeln!(out, "## Turn {turn}\n");
        }
        write_group(out, tree, &g);
    }
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
#[must_use]
pub fn render(tree: &Tree, captured: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "---\ntype: transcript\ncaptured: {captured}\n---\n");
    let _ = writeln!(out, "# {}\n", title_of(tree));
    write_spine(&mut out, tree);

    // Every paragraph is written with a blank line after it, so the last one
    // leaves the document ending in a blank. The indexer splits on blank
    // lines, and a trailing one is an empty chunk.
    out.truncate(out.trim_end().len());
    out.push('\n');
    out
}

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
            .append(
                root,
                Role::Assistant,
                ContentBlock::text("Expire on write."),
            )
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
        tree.append(
            thinking,
            Role::Assistant,
            ContentBlock::text("Expire on write."),
        )
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
"
        );
    }
}
