//! Branching agent harness over the Anthropic Messages API.
//!
//! The binary around the tree stays thin, and does two things.
//!
//! It links `sloop-memory-core` directly rather than reaching a daemon over a
//! socket, so the library boundary stays load-bearing: a build failure here
//! means the engine has stopped exporting enough to be usable outside its own
//! crate.
//!
//! And it runs the milestone the tree exists to prove, against the live API:
//! send an opening turn, fork the transcript at a content-block boundary,
//! regenerate the turn down the new branch, label one side kept and the other
//! abandoned, and replay both to `messages[]` arrays a real request could
//! carry. Every run spends tokens; there is no offline mode.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context as _, Error, Result};
use sloop_memory_core::config::{self, db_dir, TABLE_CHUNKS};
use sloop_memory_core::index::is_indexable;

mod api;
mod transcript;
mod tree;

use api::{Api, Turn};
use tree::{ContentBlock, NodeId, Role, Status, Tree};

/// What the demo asks when the command line says nothing. Short on purpose:
/// the question is scenery, and the branching is the subject.
const DEFAULT_PROMPT: &str = "How should the cache expire?";

/// The root label a session is written to.
const TRANSCRIPT_ROOT: &str = "transcripts";

fn rejected_id() -> Error {
    anyhow!("a node id was rejected by the tree that minted it")
}

/// Where to write this session, or why it is not being written.
///
/// Returns `Err` with a message rather than an `anyhow::Error` because the
/// caller reports it and carries on: a run without the root configured is a
/// legitimate way to use the harness, and failing the turn over it would be
/// the wrong trade.
///
/// Takes label/path pairs rather than `config::Roots` because a `config::Root`
/// cannot be built outside `sloop-memory-core` -- `RootDir`'s only constructor
/// is private, deliberately, so that every root on the type has been proved to
/// exist. Pairs are what this actually needs, and they keep the test off the
/// filesystem.
fn transcript_dir(roots: &[(&str, &Path)]) -> Result<PathBuf, String> {
    for (label, dir) in roots {
        if *label == TRANSCRIPT_ROOT {
            return Ok(dir.to_path_buf());
        }
    }
    Err(format!(
        "no root labelled '{TRANSCRIPT_ROOT}' is configured, so this session is not being recorded"
    ))
}

/// Write the whole session to its file, replacing what is there.
///
/// A full rewrite rather than an append: it is idempotent, so there is no
/// partial-write state to recover, and the indexer's manifest hash makes an
/// unchanged rewrite free. The daemon's watcher picks it up after
/// `WATCH_DEBOUNCE_SECS` -- the harness never touches the index itself, which
/// is what keeps the two from racing on the same rows.
fn record(dir: &Path, tree: &Tree) -> Result<PathBuf> {
    let captured = chrono::Local::now().format("%Y-%m-%d").to_string();
    let path = dir.join(transcript::filename(tree, &captured));
    std::fs::write(&path, transcript::render(tree, &captured))
        .with_context(|| format!("writing the session transcript to {}", path.display()))?;
    Ok(path)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Both calls are deliberately engine-side -- where the table lives, and
    // what the indexer will accept. Nothing here touches `proto`: that is the
    // daemon's socket protocol, and reaching for it would be exactly the
    // coupling that linking the library directly is meant to avoid.
    let table = db_dir().join(TABLE_CHUNKS);
    let indexable = is_indexable(Path::new("notes/branch-replay.md"));
    line(&format!("{} indexable={indexable}", table.display()))?;

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PROMPT.to_owned());

    // Before the tree, so a missing credential costs nothing and is reported
    // at startup rather than after the first turn is half built.
    let api = Api::from_env()?;

    // Resolved once, before any tokens are spent, so a misconfigured root is
    // reported at the top of the run rather than after the first turn. Both
    // failures are survivable: a run that cannot be recorded is still a run.
    let dir = match config::roots() {
        Ok(roots) => {
            let pairs: Vec<(&str, &Path)> = roots
                .iter()
                .map(|root| (root.label.as_str(), root.dir.as_path()))
                .collect();
            transcript_dir(&pairs)
        }
        Err(error) => Err(format!(
            "the indexing roots could not be read, so this session is not being recorded: {error}"
        )),
    };
    if let Err(why) = &dir {
        line(&format!("note: {why}"))?;
    }

    let mut tree = Tree::new(ContentBlock::text(&prompt));
    let root = tree.root();

    line("\nturn 1  streaming...")?;
    let first = turn(&api, &tree, root).await?;
    let kept = graft(&mut tree, root, first.blocks)?;
    if let Ok(dir) = &dir {
        line(&format!("recorded {}", record(dir, &tree)?.display()))?;
    }

    // The fork is a *sibling* of the node it hangs off, not a continuation of
    // it. Where it hangs is `fork_point`'s decision: inside the turn when the
    // block that would be shared is text, and at the turn boundary when that
    // block carries a signature. `prompt_for` drops the trailing assistant
    // turn either way, so both branches regenerate from the same user boundary
    // -- the difference is what the tree remembers, not what the request
    // carries.
    let (fork_at, kept_divergence) = fork_point(&tree, root, &kept)?;

    line("\nfork, abandon, regenerate\n\nturn 1' streaming...")?;
    let second = turn(&api, &tree, fork_at).await?;
    let abandoned = graft(&mut tree, fork_at, second.blocks)?;
    if let Ok(dir) = &dir {
        line(&format!("recorded {}", record(dir, &tree)?.display()))?;
    }
    let [abandoned_divergence, ..] = abandoned.as_slice() else {
        return Err(anyhow!(
            "the regenerated turn came back with no content blocks, so there is no second branch"
        ));
    };

    // One `set_status` per branch, placed at the divergence rather than at the
    // leaf. Everything below inherits it, which is what makes a discarded
    // branch identifiable all the way down rather than only where it was
    // rejected.
    tree.set_status(kept_divergence, Status::Kept)
        .ok_or_else(rejected_id)?;
    tree.set_status(*abandoned_divergence, Status::Abandoned)
        .ok_or_else(rejected_id)?;

    for stop in [first.stop_reason, second.stop_reason]
        .into_iter()
        .flatten()
    {
        if stop != "end_turn" {
            // `max_tokens` is not an error here: a truncated assistant turn is
            // a legal interior state for this tree, and forking is how it
            // resumes. Worth saying out loud all the same, because the printed
            // transcript looks the same either way.
            line(&format!("\nnote: a turn stopped on {stop}"))?;
        }
    }

    render(&tree)
}

/// Send the prompt that the branch at `tip` implies, printing blocks as they
/// complete.
async fn turn(api: &Api, tree: &Tree, tip: NodeId) -> Result<Turn> {
    let prompt = tree.prompt_for(tip).ok_or_else(rejected_id)?;
    api.send(&prompt, |block| line(&summarize(block))).await
}

/// Append a whole turn under `parent`, returning the node ids in order.
fn graft(tree: &mut Tree, parent: NodeId, blocks: Vec<ContentBlock>) -> Result<Vec<NodeId>> {
    let mut nodes = Vec::new();
    let mut tip = parent;
    for block in blocks {
        tip = tree
            .append(tip, Role::Assistant, block)
            .ok_or_else(rejected_id)?;
        nodes.push(tip);
    }
    Ok(nodes)
}

/// Where to hang the second branch, and the node of the first branch that
/// diverges there.
///
/// A turn of two or more blocks has an interior boundary, so the fork can go
/// inside it: its first block becomes the shared prefix and its second is
/// where the two branches part. Otherwise the fork falls back to the root and
/// the whole turn is what differs.
///
/// Both come back from one call because they are one decision. The divergence
/// is by definition `fork_at`'s child on the first branch, and deriving them
/// separately is precisely how that goes wrong: an index off by one labels a
/// node the two branches share, which no later step can detect -- the tree is
/// still well formed, it just means something else.
///
/// The interior fork is only legal when the shared block carries no signature.
/// Sharing a prefix block is free when the block is text and is not free when
/// it is `thinking`: the shared signature was produced by the generation that
/// also produced the rest of the first branch, so a second generation hung
/// under it replays to an assistant turn whose reasoning came from two
/// requests, and the request that produced the second half never contained the
/// first. Today's model accepts that; preserved thinking is the shape that
/// rejects it. So a signed shared block takes the same fallback a one-block
/// turn does, and the tree never holds a branch it cannot send.
///
/// Because thinking is on by default on the model this targets, the opening
/// block of a real turn is usually `thinking` -- which makes the fallback the
/// common path against the live API and the interior fork the exception.
fn fork_point(tree: &Tree, root: NodeId, first_branch: &[NodeId]) -> Result<(NodeId, NodeId)> {
    match first_branch {
        [shared, next, ..] if !signed(tree, *shared)? => Ok((*shared, *next)),
        [first, ..] => Ok((root, *first)),
        [] => Err(anyhow!(
            "the opening turn came back with no content blocks, so there is nothing to fork"
        )),
    }
}

/// Whether this node's block carries a signature bound to the prefix above it.
///
/// The question [`fork_point`] asks of a candidate shared block, and the one
/// place the two block kinds are not interchangeable.
fn signed(tree: &Tree, id: NodeId) -> Result<bool> {
    match tree.block(id).ok_or_else(rejected_id)? {
        ContentBlock::Text { .. } => Ok(false),
        ContentBlock::Thinking { .. } => Ok(true),
    }
}

/// The first line of a block, tagged with its kind.
///
/// Only the first line, deliberately: this runs while the turn is still
/// streaming, and the point of it is to show that blocks land one at a time.
/// The whole text of every block is printed below by [`render`], so nothing
/// here is lost -- it is a progress line, not an elision.
fn summarize(block: &ContentBlock) -> String {
    let (kind, text) = match block {
        ContentBlock::Text { text } => ("text", text),
        ContentBlock::Thinking {
            thinking,
            signature: _,
        } => ("thinking", thinking),
    };
    format!("  [{kind}] {}", text.lines().next().unwrap_or_default())
}

/// Print every branch as the `messages[]` array it replays to.
///
/// Branch 0 is the first one generated. `leaves` walks the arena in creation
/// order, and the whole first turn is appended before the second one starts,
/// so the kept branch's leaf always has the lower index of the two.
fn render(tree: &Tree) -> Result<()> {
    for (number, tip) in tree.leaves().into_iter().enumerate() {
        let (Some(own), Some(inherited), Some(messages), Some(prompt)) = (
            tree.status(tip),
            tree.effective_status(tip),
            tree.replay(tip),
            tree.prompt_for(tip),
        ) else {
            return Err(rejected_id());
        };

        line(&format!(
            "\n=== branch {number}: labeled {own:?}, in force {inherited:?} ==="
        ))?;
        line(&serde_json::to_string_pretty(&messages)?)?;

        // What a request would actually carry. The trailing assistant turn is
        // gone, because that is the turn being regenerated and a partial final
        // assistant turn is an HTTP 400.
        let ends_on = prompt.last().map_or(Role::User, |message| message.role);
        line(&format!(
            "-- prompt_for: {} message(s), ending on {ends_on:?}",
            prompt.len()
        ))?;
    }
    Ok(())
}

/// Write one line to stdout.
///
/// A function rather than `println!`, which this crate's lints deny outright.
/// It returns the write's `Result` instead of ignoring it, which is what the
/// streaming callback in [`turn`] hands back to `send`: a closed pipe stops
/// the turn there rather than being swallowed while the rest of it is pulled
/// over the network to be written into a reader that has gone.
fn line(text: &str) -> Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{text}")?;
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        fork_point, graft, record, summarize, transcript, transcript_dir, ContentBlock, Status,
        Tree,
    };

    // A missing `transcripts` root is a configuration state, not a failure: the
    // harness's job is the turn, and a run without the root configured must still
    // complete. It has to say so, though -- silently not recording is the failure
    // mode this whole slice exists to remove.
    #[test]
    fn a_missing_transcripts_root_is_reported_and_not_fatal() {
        let roots = Vec::new();
        assert_eq!(
            transcript_dir(&roots),
            Err(
                "no root labelled 'transcripts' is configured, so this session is not being \
                 recorded"
                    .to_owned()
            )
        );
    }

    // The label selects, not the position. A `transcripts` root that is neither
    // first nor last is the case that tells those two apart.
    #[test]
    fn the_transcripts_root_is_found_among_others_by_its_label() {
        let roots = vec![
            ("notes", Path::new("/roots/notes")),
            ("transcripts", Path::new("/roots/sessions")),
            ("memory", Path::new("/roots/memory")),
        ];

        assert_eq!(transcript_dir(&roots), Ok(PathBuf::from("/roots/sessions")));
    }

    /// `record` derives the name rather than using a fixed one, and writes the
    /// rendered document under it.
    ///
    /// Both halves are needed. A fixed name still produces a file the daemon
    /// indexes, so nothing downstream would notice that every session on the
    /// machine had been overwriting one note.
    #[test]
    fn a_session_is_written_under_its_derived_name() {
        let dir = tempfile::tempdir().unwrap();
        let tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let captured = chrono::Local::now().format("%Y-%m-%d").to_string();

        let path = record(dir.path(), &tree).unwrap();

        assert_eq!(
            path,
            dir.path()
                .join(format!("{captured}-how-should-the-cache-expire.md"))
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            transcript::render(&tree, &captured)
        );
    }

    #[test]
    fn a_summary_names_the_kind_and_keeps_only_the_first_line() {
        assert_eq!(
            summarize(&ContentBlock::text("one\ntwo")),
            "  [text] one".to_owned()
        );
        assert_eq!(
            summarize(&ContentBlock::thinking("weighing\nboth", "sig")),
            "  [thinking] weighing".to_owned()
        );
    }

    // An empty block has no first line, and `lines()` on an empty string
    // yields nothing rather than one empty item -- so this is the one input
    // where the fallback is what prints.
    #[test]
    fn a_summary_of_an_empty_block_is_the_tag_alone() {
        assert_eq!(summarize(&ContentBlock::text("")), "  [text] ".to_owned());
    }

    // The whole point of `fork_point` returning a pair: these two indices are
    // read off the same slice, and swapping either one for the other's answer
    // still type-checks.
    #[test]
    fn a_multi_block_turn_forks_inside_itself() {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();
        let turn = graft(
            &mut tree,
            root,
            vec![
                ContentBlock::text("shared"),
                ContentBlock::text("diverges"),
                ContentBlock::text("below"),
            ],
        )
        .unwrap();

        assert_eq!(fork_point(&tree, root, &turn).unwrap(), (turn[0], turn[1]));
    }

    // The interior fork shares the turn's first block between both branches.
    // That is free when the block is text and not free when it is `thinking`:
    // the shared signature was produced by the generation that also produced
    // the rest of the first branch, so hanging a second generation under it
    // replays to an assistant turn whose reasoning came from two requests.
    // The turn boundary is the only fork point that keeps a signature with
    // the prefix that produced it.
    #[test]
    fn a_turn_opening_on_a_thinking_block_forks_at_the_root() {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();
        let turn = graft(
            &mut tree,
            root,
            vec![
                ContentBlock::thinking("weighing both", "sig"),
                ContentBlock::text("expire on write"),
            ],
        )
        .unwrap();

        assert_eq!(fork_point(&tree, root, &turn).unwrap(), (root, turn[0]));
    }

    // The complement, and the reason the test above is not just "any thinking
    // block anywhere". Only the *shared* block travels into the second
    // branch's replay; a signature below the divergence belongs to one branch
    // alone, and refusing the interior fork for it would give up a legal one.
    #[test]
    fn a_thinking_block_below_the_divergence_still_forks_inside_the_turn() {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();
        let turn = graft(
            &mut tree,
            root,
            vec![
                ContentBlock::text("shared"),
                ContentBlock::thinking("weighing both", "sig"),
            ],
        )
        .unwrap();

        assert_eq!(fork_point(&tree, root, &turn).unwrap(), (turn[0], turn[1]));
    }

    #[test]
    fn a_single_block_turn_forks_at_the_root() {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();
        let turn = graft(&mut tree, root, vec![ContentBlock::text("only")]).unwrap();

        assert_eq!(fork_point(&tree, root, &turn).unwrap(), (root, turn[0]));
    }

    #[test]
    fn a_turn_with_no_blocks_has_nothing_to_fork() {
        let tree = Tree::new(ContentBlock::text("q"));
        let Err(error) = fork_point(&tree, tree.root(), &[]) else {
            panic!("an empty turn was accepted as a fork point");
        };

        assert!(error.to_string().contains("no content blocks"), "{error}");
    }

    /// What `main` does after both turns land, on a tree built by hand.
    ///
    /// This is the assertion that a misplaced divergence would fail: labeling
    /// the shared block instead of the one below it puts `Kept` on an ancestor
    /// of *both* leaves, and the abandoned branch would come back `Abandoned`
    /// regardless -- so it is the kept branch that has to be checked, and both
    /// orderings of the pair have to be checked together.
    fn labeled(turn: Vec<ContentBlock>) -> (Tree, Vec<Status>) {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();

        let kept = graft(&mut tree, root, turn).unwrap();
        let (fork_at, kept_divergence) = fork_point(&tree, root, &kept).unwrap();
        let abandoned = graft(&mut tree, fork_at, vec![ContentBlock::text("other")]).unwrap();

        tree.set_status(kept_divergence, Status::Kept).unwrap();
        tree.set_status(abandoned[0], Status::Abandoned).unwrap();

        let mut effective = Vec::new();
        for leaf in tree.leaves() {
            effective.push(tree.effective_status(leaf).unwrap());
        }
        (tree, effective)
    }

    // Branch 0 is the kept one, and `render` prints them in this order.
    #[test]
    fn each_branch_inherits_its_own_label_when_the_fork_is_interior() {
        let (_, effective) = labeled(vec![ContentBlock::text("a"), ContentBlock::text("b")]);

        assert_eq!(effective, vec![Status::Kept, Status::Abandoned]);
    }

    // The fallback path, where the fork sits at the root instead. The labels
    // have to land the same way, and the index that produces them differs.
    #[test]
    fn each_branch_inherits_its_own_label_when_the_fork_is_at_the_root() {
        let (_, effective) = labeled(vec![ContentBlock::text("a")]);

        assert_eq!(effective, vec![Status::Kept, Status::Abandoned]);
    }
}
