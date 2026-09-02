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
    best.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
    best.truncate(config::HOOK_MAX_HITS);
    Ok(RecallResult {
        pointers: best,
        top_cosine,
    })
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
