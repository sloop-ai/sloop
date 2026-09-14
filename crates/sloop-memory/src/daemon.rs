//! Resident daemon.
//!
//! Exists for one reason: loading the ONNX model costs ~1s, and a query costs
//! ~3ms. Anything on the prompt path has to pay the former once, not per call.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use crate::watch;
use sloop_memory_core::proto::{Pointer, Request, Response};
use sloop_memory_core::{config, embed::Embedder, index, store};

struct State {
    embedder: Mutex<Embedder>,
    conn: lancedb::Connection,
    roots: config::Roots,
}

/// Pointers for the prompt hook.
///
/// Gated on cosine rather than the RRF score, and deduplicated to one pointer per
/// note: three chunks from the same file is one piece of information to the reader,
/// and spending the whole budget on a single note would crowd out everything else.
struct RecallResult {
    pointers: Vec<Pointer>,
    top_cosine: Option<f32>,
}

async fn recall(state: &State, prompt: &str) -> Result<RecallResult> {
    if prompt.trim().is_empty() {
        return Ok(RecallResult {
            pointers: vec![],
            top_cosine: None,
        });
    }
    let Some(table) = store::open_table(&state.conn, config::TABLE_CHUNKS).await? else {
        return Ok(RecallResult {
            pointers: vec![],
            top_cosine: None,
        });
    };

    let qvec = { state.embedder.lock().await.encode_query(prompt)? };
    let hits = store::hybrid_search(&table, qvec, prompt, config::HOOK_MAX_HITS * 4, None).await?;
    let top_cosine = hits.iter().map(|h| h.cosine).max_by(f32::total_cmp);

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
            let path = state
                .roots
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
    Ok(RecallResult {
        pointers: take_round_robin(best, config::HOOK_MAX_HITS),
        top_cosine,
    })
}

/// Take up to `limit` pointers, cycling through roots so one cannot take every
/// slot while another has hits above threshold.
///
/// Within a root the order stays by cosine, and a root that is the only one
/// with hits still fills the block. This bounds crowding rather than solving
/// it: flat retrieval has no notion of level, so every chunk competes as a
/// peer. Usage feedback and a concept graph are the real answers, and both are
/// later slices.
///
/// Deterministic for a given input: `sort_by` is stable, roots are held in a
/// `Vec` ordered by first appearance -- so best-first -- and never in a
/// `HashMap`, whose iteration order is reseeded per process and would make the
/// same prompt render a different block on each run.
fn take_round_robin(mut hits: Vec<Pointer>, limit: usize) -> Vec<Pointer> {
    hits.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));

    let mut by_root: Vec<Vec<Pointer>> = Vec::new();
    for hit in hits {
        match by_root.iter_mut().find(|group| {
            group
                .first()
                .is_some_and(|p| p.source_type == hit.source_type)
        }) {
            Some(group) => group.push(hit),
            None => by_root.push(vec![hit]),
        }
    }

    // Bounded by the deepest root rather than by "did this round pick
    // anything", so the loop cannot spin when every root is exhausted.
    let deepest = by_root.iter().map(Vec::len).max().unwrap_or(0);
    let mut picked = Vec::with_capacity(limit);
    for round in 0..deepest {
        for group in &by_root {
            if picked.len() == limit {
                return picked;
            }
            if let Some(hit) = group.get(round) {
                picked.push(hit.clone());
            }
        }
    }
    picked
}

async fn dispatch(state: &State, req: Request) -> Response {
    let started = std::time::Instant::now();
    // u64::MAX ms is ~584 million years; a single request can't realistically
    // run long enough to overflow it.
    #[expect(clippy::cast_possible_truncation, reason = "see comment above")]
    let ms = |t: std::time::Instant| t.elapsed().as_millis() as u64;

    match req {
        Request::Ping => Response::Pong {
            pid: std::process::id(),
        },

        Request::Status => {
            let roots = store::root_row_counts(&state.conn, &state.roots).await;
            let db = config::db_dir().display().to_string();
            Response::Status { roots, db }
        }

        Request::Search { query, k, filter } => {
            let table = match store::open_table(&state.conn, config::TABLE_CHUNKS).await {
                Ok(Some(t)) => t,
                Ok(None) => {
                    return Response::Error {
                        message: "no index yet; run `sloop-memory index`".into(),
                    }
                }
                Err(e) => {
                    return Response::Error {
                        message: e.to_string(),
                    }
                }
            };
            let qvec = match state.embedder.lock().await.encode_query(&query) {
                Ok(v) => v,
                Err(e) => {
                    return Response::Error {
                        message: e.to_string(),
                    }
                }
            };
            match store::hybrid_search(&table, qvec, &query, k, filter.as_deref()).await {
                Ok(hits) => Response::Hits {
                    hits,
                    elapsed_ms: ms(started),
                },
                Err(e) => Response::Error {
                    message: e.to_string(),
                },
            }
        }

        Request::Recall { prompt } => match recall(state, &prompt).await {
            Ok(result) => Response::Pointers {
                pointers: result.pointers,
                elapsed_ms: ms(started),
                top_cosine: result.top_cosine,
            },
            Err(e) => Response::Error {
                message: e.to_string(),
            },
        },

        Request::Reindex { full } => {
            match index::reindex_all(&state.conn, &state.embedder, &state.roots, full).await {
                Ok(stats) => Response::Reindexed { stats },
                Err(e) => Response::Error {
                    message: e.to_string(),
                },
            }
        }
    }
}

async fn handle(state: Arc<State>, stream: UnixStream) {
    let (read_half, mut write_half) = stream.into_split();
    let mut line = String::new();
    if BufReader::new(read_half)
        .read_line(&mut line)
        .await
        .is_err()
    {
        return;
    }
    let response = match serde_json::from_str::<Request>(&line) {
        Ok(req) => dispatch(&state, req).await,
        Err(e) => Response::Error {
            message: format!("bad request: {e}"),
        },
    };
    if let Ok(mut body) = serde_json::to_vec(&response) {
        body.push(b'\n');
        let _ = write_half.write_all(&body).await;
    }
}

/// Full sweep, with the outcome logged either way -- a reindex that silently does
/// nothing is indistinguishable from a watcher that has stopped firing.
async fn sweep(state: &State, reason: &'static str) {
    match index::reindex_all(&state.conn, &state.embedder, &state.roots, false).await {
        Ok(s) if s.changed > 0 || s.removed > 0 => tracing::info!(
            reason,
            changed = s.changed,
            removed = s.removed,
            chunks = s.chunks_written,
            embed_ms = s.embed_ms,
            write_ms = s.write_ms,
            post_index_ms = s.post_index_ms,
            "swept roots"
        ),
        Ok(_) => tracing::info!(reason, "swept roots, nothing changed"),
        Err(e) => tracing::warn!(reason, error = %e, "sweep failed"),
    }
}

// serve() is a startup sequence, not a computation: socket setup, state,
// and three spawned loops. Splitting it would scatter one linear story
// across five functions.
#[expect(clippy::too_many_lines, reason = "linear startup sequence")]
pub async fn serve() -> Result<()> {
    // macOS caps sockaddr_un.sun_path at 104 bytes and surfaces an overflow as an
    // opaque "path must be shorter than SUN_LEN". Fail with the reason instead.
    const SUN_LEN: usize = 104;

    let socket = config::socket_path();
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let socket_len = socket.as_os_str().as_encoded_bytes().len();
    anyhow::ensure!(
        socket_len < SUN_LEN,
        "socket path is {socket_len} bytes, must be under {SUN_LEN}: {}\n\
         Set SLOOP_MEMORY_SOCKET (or SLOOP_MEMORY_STATE) to something shorter.",
        socket.display()
    );

    // A socket file left behind by a killed daemon would block bind(); it carries
    // no state, so removing it is always safe.
    if socket.exists() {
        std::fs::remove_file(&socket)
            .with_context(|| format!("removing stale socket {}", socket.display()))?;
    }

    let roots = config::roots()?;
    let state = Arc::new(State {
        embedder: Mutex::new(Embedder::load(&config::model_dir()?)?),
        conn: store::connect(&config::db_dir()).await?,
        roots: roots.clone(),
    });

    let listener =
        UnixListener::bind(&socket).with_context(|| format!("binding {}", socket.display()))?;
    tracing::info!(socket = %socket.display(), "sloop-memory daemon ready");

    // Catch up on anything that changed while the daemon was down. Backgrounded so
    // the socket is answerable immediately -- queries during the sweep just see
    // slightly stale data, which is strictly better than refusing to answer.
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move { sweep(&state, "startup").await });
    }

    match watch::spawn(roots.clone()) {
        Ok(mut events) => {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                while let Some(paths) = events.recv().await {
                    // One debouncer watches every root, so a single batch of
                    // paths can span more than one of them; group by root
                    // before indexing so each group hits the right identity
                    // namespace.
                    let mut by_root: HashMap<config::RootLabel, Vec<PathBuf>> = HashMap::new();
                    for path in paths {
                        if let Some(root) = config::root_for_path(&state.roots, &path) {
                            by_root.entry(root.label.clone()).or_default().push(path);
                        }
                    }
                    for (label, paths) in by_root {
                        let Some(root) = state.roots.iter().find(|r| r.label == label) else {
                            continue; // unreachable: label was read from state.roots above
                        };
                        let n = paths.len();
                        let result =
                            index::index_paths(&state.conn, &state.embedder, root, &paths, false)
                                .await;
                        match result {
                            Ok(s) if s.changed > 0 || s.removed > 0 => tracing::info!(
                                root = %root.label,
                                changed = s.changed,
                                removed = s.removed,
                                chunks = s.chunks_written,
                                embed_ms = s.embed_ms,
                                write_ms = s.write_ms,
                                post_index_ms = s.post_index_ms,
                                "reindexed from watch"
                            ),
                            // Obsidian rewrites files without changing content
                            // often enough that this is the common case, not
                            // an anomaly.
                            Ok(_) => tracing::debug!(
                                root = %root.label,
                                paths = n,
                                "watch event, no content change"
                            ),
                            Err(e) => tracing::warn!(
                                root = %root.label,
                                error = %e,
                                "watch reindex failed"
                            ),
                        }
                    }
                }
            });
        }
        Err(e) => {
            tracing::warn!(error = %e, "root watcher unavailable; falling back to periodic sweep only");
        }
    }

    {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(Duration::from_secs(config::SWEEP_INTERVAL_SECS));
            ticker.tick().await; // fires immediately; the startup sweep covers it
            loop {
                ticker.tick().await;
                sweep(&state, "periodic").await;
            }
        });
    }

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let state = Arc::clone(&state);
                    tokio::spawn(handle(state, stream));
                }
                Err(e) => tracing::warn!(error = %e, "accept failed"),
            },
            _ = tokio::signal::ctrl_c() => break,
            _ = sigterm.recv() => break,
        }
    }

    tracing::info!("shutting down");
    let _ = std::fs::remove_file(&socket);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::take_round_robin;
    use sloop_memory_core::proto::Pointer;

    fn pointer(source_type: &str, cosine: f32) -> Pointer {
        Pointer {
            title: String::new(),
            source_type: source_type.to_string(),
            path: String::new(),
            rel_path: String::new(),
            heading_path: String::new(),
            captured: String::new(),
            cosine,
        }
    }

    /// Labels alone are ambiguous once a root appears twice, so pin the cosine
    /// with it: the pair fixes both the interleaving and the within-root order.
    fn labelled(picked: &[Pointer]) -> Vec<(&str, f32)> {
        picked
            .iter()
            .map(|p| (p.source_type.as_str(), p.cosine))
            .collect()
    }

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

    /// Three roots must each get a slot before any root gets a second one, and
    /// the roots themselves must be visited best-first -- otherwise the block's
    /// leading pointer stops being the strongest match.
    #[test]
    fn every_root_gets_a_slot_before_any_root_gets_a_second() {
        let hits = vec![
            pointer("transcripts", 0.95),
            pointer("transcripts", 0.90),
            pointer("transcripts", 0.85),
            pointer("notes", 0.80),
            pointer("notes", 0.70),
            pointer("memory", 0.60),
        ];

        let picked = take_round_robin(hits, 5);

        assert_eq!(
            labelled(&picked),
            vec![
                ("transcripts", 0.95),
                ("notes", 0.80),
                ("memory", 0.60),
                ("transcripts", 0.90),
                ("notes", 0.70),
            ]
        );
    }

    /// Interleaving reorders across roots but must not reorder within one: a
    /// root's own hits stay descending, and a root that runs dry drops out
    /// while the others keep filling.
    #[test]
    fn a_roots_own_hits_stay_in_descending_cosine_order() {
        let hits = vec![
            pointer("notes", 0.40),
            pointer("transcripts", 0.99),
            pointer("notes", 0.90),
            pointer("transcripts", 0.10),
            pointer("notes", 0.65),
        ];

        let picked = take_round_robin(hits, 5);

        assert_eq!(
            labelled(&picked),
            vec![
                ("transcripts", 0.99),
                ("notes", 0.90),
                ("transcripts", 0.10),
                ("notes", 0.65),
                ("notes", 0.40),
            ]
        );
    }

    /// The limit cuts mid-round, not at a round boundary -- rounding it up to
    /// the next whole round would overrun the block budget.
    #[test]
    fn the_limit_cuts_part_way_through_a_round() {
        let hits = vec![
            pointer("transcripts", 0.95),
            pointer("notes", 0.80),
            pointer("memory", 0.60),
            pointer("transcripts", 0.90),
        ];

        let picked = take_round_robin(hits, 2);

        assert_eq!(
            labelled(&picked),
            vec![("transcripts", 0.95), ("notes", 0.80)]
        );
    }

    /// Nothing above threshold is the ordinary case for an unrelated prompt;
    /// the round loop has to terminate on it rather than spin.
    #[test]
    fn no_hits_yields_no_pointers() {
        assert_eq!(labelled(&take_round_robin(vec![], 3)), vec![]);
    }
}
