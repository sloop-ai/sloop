//! Branching agent harness over the Anthropic Messages API.
//!
//! The conversation tree in [`tree`] is the first real piece of it. The binary
//! around it stays thin, and does two things.
//!
//! It links `sloop-memory-core` directly rather than reaching a daemon over a
//! socket, so the library boundary stays load-bearing: a build failure here
//! means the engine has stopped exporting enough to be usable outside its own
//! crate.
//!
//! And it runs the milestone the tree exists to prove -- fork a transcript at
//! a content-block boundary, abandon one side, and replay both branches to
//! `messages[]` arrays a real request could carry.

use std::io::{Error, Write};
use std::path::Path;

use sloop_memory_core::config::{db_dir, TABLE_CHUNKS};
use sloop_memory_core::index::is_indexable;

mod api;
mod tree;

use tree::{ContentBlock, Role, Status, Tree};

fn rejected_id() -> Error {
    Error::other("a node id was rejected by the tree that minted it")
}

fn main() -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();

    // Both calls are deliberately engine-side -- where the table lives, and
    // what the indexer will accept. Nothing here touches `proto`: that is the
    // daemon's socket protocol, and reaching for it would be exactly the
    // coupling that linking the library directly is meant to avoid.
    let table = db_dir().join(TABLE_CHUNKS);
    let indexable = is_indexable(Path::new("notes/branch-replay.md"));
    writeln!(out, "{} indexable={indexable}", table.display())?;

    let Some(tree) = forked_transcript() else {
        return Err(rejected_id());
    };
    render(&mut out, &tree)
}

/// Build the milestone transcript.
///
///     u1  "How should the cache expire?"                     user
///     a1  "Two options."                                     assistant
///     |-- a2   "Use an LRU."         [Kept]                  assistant
///     |   `-- a3  "Size it by RSS."  (unlabeled)             assistant
///     `-- a2'  "Use a TTL map."      [Abandoned]             assistant
///         `-- a3' "Sweep on read."   (unlabeled)             assistant
///
/// The fork is a *sibling* of `a2` rather than a continuation of it. That is
/// the whole reason nodes are content blocks: prefill was removed, so the
/// second branch cannot resume the first one's half-finished turn and has to
/// regenerate it from this boundary.
///
/// Neither leaf carries a label of its own. Both inherit one, which is what
/// makes a discarded branch identifiable all the way down rather than only at
/// the point where it was rejected.
fn forked_transcript() -> Option<Tree> {
    let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
    let root = tree.root();

    let opening = tree.append(root, Role::Assistant, ContentBlock::text("Two options."))?;

    let kept = tree.append(opening, Role::Assistant, ContentBlock::text("Use an LRU."))?;
    tree.append(kept, Role::Assistant, ContentBlock::text("Size it by RSS."))?;
    tree.set_status(kept, Status::Kept)?;

    let abandoned = tree.append(
        opening,
        Role::Assistant,
        ContentBlock::text("Use a TTL map."),
    )?;
    tree.append(
        abandoned,
        Role::Assistant,
        ContentBlock::text("Sweep on read."),
    )?;
    tree.set_status(abandoned, Status::Abandoned)?;

    Some(tree)
}

fn render(out: &mut impl Write, tree: &Tree) -> std::io::Result<()> {
    for (number, tip) in tree.leaves().into_iter().enumerate() {
        let (Some(own), Some(inherited), Some(messages), Some(prompt)) = (
            tree.status(tip),
            tree.effective_status(tip),
            tree.replay(tip),
            tree.prompt_for(tip),
        ) else {
            return Err(rejected_id());
        };

        writeln!(
            out,
            "\n=== branch {number}: labeled {own:?}, in force {inherited:?} ==="
        )?;
        let transcript = serde_json::to_string_pretty(&messages).map_err(Error::other)?;
        writeln!(out, "{transcript}")?;

        // What a request would actually carry. The trailing assistant turn is
        // gone, because that is the turn being regenerated and a partial final
        // assistant turn is an HTTP 400.
        let ends_on = prompt.last().map_or(Role::User, |message| message.role);
        writeln!(
            out,
            "-- prompt_for: {} message(s), ending on {ends_on:?}",
            prompt.len()
        )?;
    }
    Ok(())
}
