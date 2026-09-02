//! Filesystem watching for the configured roots.
//!
//! Knows nothing about indexing: it emits batches of changed paths and the daemon
//! decides what to do with them. Debouncing is the substance here, not watching --
//! Editors save constantly, and a bulk write -- an import, a sync, a bulk rename
//! -- should arrive as one batch rather than forty.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use notify_debouncer_full::new_debouncer;
use notify_debouncer_full::notify::RecursiveMode;
use tokio::sync::mpsc;

use sloop_memory_core::{config, index};

/// Start watching every root; returns a receiver of de-duplicated changed paths.
///
/// One debouncer, watching every root's path -- not one thread per root. The
/// debouncer speaks over a std channel and must stay alive for the duration, so
/// it lives on its own thread and forwards into a tokio channel. Emitted paths
/// are root-agnostic; the caller uses `config::root_for_path` to find out which
/// root a given batch of paths belongs to.
pub fn spawn(roots: config::Roots) -> Result<mpsc::Receiver<Vec<PathBuf>>> {
    let (out_tx, out_rx) = mpsc::channel::<Vec<PathBuf>>(64);

    std::thread::Builder::new()
        .name("sloop-memory-watch".into())
        .spawn(move || {
            if let Err(e) = run(roots, out_tx) {
                tracing::error!(error = %e, "root watcher stopped");
            }
        })
        .context("spawning watcher thread")?;

    Ok(out_rx)
}

// The debouncer takes ownership for the life of the watcher thread, so
// passing by value is what the call site actually wants.
#[expect(clippy::needless_pass_by_value, reason = "owned for the thread's life")]
fn run(roots: config::Roots, out_tx: mpsc::Sender<Vec<PathBuf>>) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut debouncer = new_debouncer(Duration::from_secs(config::WATCH_DEBOUNCE_SECS), None, tx)
        .context("creating debouncer")?;
    for root in &roots {
        debouncer
            .watch(root.dir.as_path(), RecursiveMode::Recursive)
            .with_context(|| format!("watching {}", root.dir.as_path().display()))?;
        tracing::info!(label = %root.label, path = %root.dir.as_path().display(), "watching root");
    }

    for result in rx {
        let events = match result {
            Ok(events) => events,
            Err(errors) => {
                for e in errors {
                    tracing::warn!(error = %e, "watch error");
                }
                continue;
            }
        };

        // A rename reports both sides, so flatten every path and let the indexer
        // sort out which still exist. BTreeSet dedupes an editor's write+touch.
        let paths: BTreeSet<PathBuf> = events
            .iter()
            .flat_map(|e| e.paths.iter().cloned())
            .filter(|p| !index::is_excluded(p) && index::is_indexable(p))
            .collect();

        if paths.is_empty() {
            continue;
        }
        if out_tx.blocking_send(paths.into_iter().collect()).is_err() {
            break; // daemon is shutting down
        }
    }
    Ok(())
}
