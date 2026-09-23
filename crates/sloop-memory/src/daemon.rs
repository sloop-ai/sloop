//! Resident daemon.
//!
//! Exists for one reason: loading the ONNX model costs ~1s, and a query costs
//! ~3ms. Anything on the prompt path has to pay the former once, not per call.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use crate::watch;
use sloop_memory_core::proto::{Pointer, Request, Response};
use sloop_memory_core::{chunk, config, embed::Embedder, index, store};

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

/// Reduce raw hits to one pointer per file, keeping the best cosine.
///
/// This is where a split query is put back together. Each unit is searched
/// separately, and a chunk scores as the best it matched any one of them --
/// max cosine, not RRF. RRF ranks well but is derived purely from rank
/// (`store::Hit::score`), so `HOOK_MIN_COSINE` could not gate on it; cosine is
/// calibrated and comparable across queries, which is exactly what merging
/// across sub-queries needs.
///
/// Takes label/path pairs rather than `config::Roots` because a `config::Root`
/// cannot be built outside `sloop-memory-core`, so a test would otherwise need
/// the filesystem and the process environment to construct one. See
/// `sloop-harness`'s `transcript_dir` for the same trade.
fn consolidate(hits: Vec<store::Hit>, roots: &[(&str, &Path)]) -> Vec<Pointer> {
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
                .find(|(label, _)| *label == h.source_type)
                .map_or_else(
                    || h.rel_path.clone(),
                    |(_, dir)| dir.join(&h.rel_path).display().to_string(),
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

/// Reduce the concatenated per-unit hits to the `k` the MCP caller asked for.
///
/// Deliberately not `consolidate`: this path hands back chunk bodies, so two
/// chunks of one file are two distinct answers and both may stand. What cannot
/// stand is the *same* chunk repeated, which a split query produces the moment
/// one chunk matches more than one unit -- without this, `k` = 3 could be one
/// chunk three times. There is no cosine gate here either: `HOOK_MIN_COSINE`
/// is the hook's policy, and a caller asking for `k` wants `k`.
fn best_k(mut hits: Vec<store::Hit>, k: usize) -> Vec<store::Hit> {
    hits.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
    // Sorted descending, a chunk's first appearance carries its best cosine.
    // `text` is in the key because a section longer than one chunk yields
    // several chunks sharing a `rel_path` and a `heading_path`.
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    hits.retain(|h| seen.insert((h.source_type.clone(), h.rel_path.clone(), h.text.clone())));
    hits.truncate(k);
    hits
}

/// Split `query` and search every unit, concatenating the hits.
///
/// The truncating tokenizer keeps the first 512 tokens of whatever it is
/// handed, so a single `encode_query` over a pasted log reads the boilerplate
/// and never reaches the question underneath it. Splitting first means every
/// part of the query gets its own search. A query that already fits the
/// chunk-size bound splits to one unit and takes exactly the path it did
/// before.
///
/// `max_units` is the caller's own budget: each unit costs an embed and a
/// hybrid search, and the hook has 250 ms against the MCP path's 30 seconds.
/// It goes into `split_query` rather than trimming the units here, because
/// the split keeps the head *and the tail* and a trim afterwards would throw
/// away the question this whole path exists to reach.
///
/// `search_one` is a parameter so the loop can be tested without a model or a
/// table: the caller supplies the embed-and-search step, a test supplies a
/// stub. It takes the unit by value because the future it returns outlives the
/// call that made it, and returns a future rather than being an `async fn` in a
/// trait so that the embedder guard stays inside the caller's own statement --
/// `clippy::await_holding_lock` is denied, and a guard passed across this
/// boundary would be held across the search.
async fn search_units<F, Fut>(
    query: &str,
    max_units: usize,
    mut search_one: F,
) -> Result<Vec<store::Hit>>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Vec<store::Hit>>>,
{
    let mut hits = Vec::new();
    for unit in chunk::split_query(query, max_units) {
        hits.extend(search_one(unit).await?);
    }
    Ok(hits)
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

    // `consolidate` reduces the concatenated hits to the best cosine per file,
    // so a note answering any part of a long query is a hit.
    let table = &table;
    let hits = search_units(prompt, config::HOOK_MAX_QUERY_UNITS, |unit| async move {
        let qvec = { state.embedder.lock().await.encode_query(&unit)? };
        store::hybrid_search(table, qvec, &unit, config::HOOK_MAX_HITS * 4, None).await
    })
    .await?;
    let top_cosine = hits.iter().map(|h| h.cosine).max_by(f32::total_cmp);

    let pairs: Vec<(&str, &Path)> = state
        .roots
        .iter()
        .map(|root| (root.label.as_str(), root.dir.as_path()))
        .collect();
    let best = consolidate(hits, &pairs);
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
            // Same split as the hook path: MCP callers paste logs too. They
            // get the larger unit cap, though -- `TOOL_TIMEOUT` is 30 seconds
            // against the hook's 250 ms, so there is room to search more of a
            // paste before the middle of it is dropped. A unit that fails to
            // embed or search fails the whole request rather than quietly
            // returning the other units' hits -- a short answer is
            // indistinguishable from a complete one to the caller.
            //
            // An empty or whitespace-only query splits to no units at all and
            // so returns an empty hit list, not an error. That is the intended
            // contract of this arm and not an accident of the split: it
            // replaced embedding the empty string and searching on it, which
            // answered a question nobody asked. Both callers today reject an
            // empty query before it gets here (`mcp.rs`, and the CLI's
            // positional argument), but the next host to speak this protocol
            // will not have read either of them.
            let table = &table;
            let filter = filter.as_deref();
            let searched = search_units(&query, config::MAX_QUERY_UNITS, |unit| async move {
                let qvec = { state.embedder.lock().await.encode_query(&unit)? };
                store::hybrid_search(table, qvec, &unit, k, filter).await
            })
            .await;
            let hits = match searched {
                Ok(hits) => best_k(hits, k),
                Err(e) => {
                    return Response::Error {
                        message: e.to_string(),
                    }
                }
            };
            Response::Hits {
                hits,
                elapsed_ms: ms(started),
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
#[expect(
    clippy::unwrap_used,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use std::cell::RefCell;

    use super::{best_k, consolidate, search_units, take_round_robin};
    use sloop_memory_core::proto::Pointer;
    use sloop_memory_core::{chunk, config, store};

    fn hit(source_type: &str, rel_path: &str, heading: &str, cosine: f32) -> store::Hit {
        store::Hit {
            score: 0.0,
            cosine,
            title: String::new(),
            rel_path: rel_path.to_string(),
            heading_path: heading.to_string(),
            captured: String::new(),
            source_type: source_type.to_string(),
            text: String::new(),
        }
    }

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

    /// Two units matched the same file; the better match is the one that counts.
    /// This is the merge a split query depends on.
    #[test]
    fn the_best_matching_unit_wins_for_a_file() {
        let hits = vec![
            hit("notes", "cache.md", "Turn 1", 0.76),
            hit("notes", "cache.md", "Turn 9", 0.91),
        ];

        let out = consolidate(hits, &[]);

        assert_eq!(out.len(), 1, "the same file produced two pointers");
        assert!(
            (out[0].cosine - 0.91).abs() < f32::EPSILON,
            "kept {}",
            out[0].cosine
        );
        assert_eq!(
            out[0].heading_path, "Turn 9",
            "kept the weaker unit's heading"
        );
    }

    /// The max has to win on arrival order too. Units are searched in query
    /// order, not score order, so the strongest match is as likely to land
    /// first as last.
    #[test]
    fn the_best_matching_unit_wins_whichever_arrives_first() {
        let hits = vec![
            hit("notes", "cache.md", "Turn 9", 0.91),
            hit("notes", "cache.md", "Turn 1", 0.76),
        ];

        let out = consolidate(hits, &[]);

        assert_eq!(out.len(), 1, "the same file produced two pointers");
        assert!(
            (out[0].cosine - 0.91).abs() < f32::EPSILON,
            "kept {}",
            out[0].cosine
        );
        assert_eq!(
            out[0].heading_path, "Turn 9",
            "kept the weaker unit's heading"
        );
    }

    /// A hit below the gate contributes nothing, even when another unit of the
    /// same query cleared it. The threshold is per-chunk, not per-query.
    #[test]
    fn a_hit_below_the_gate_is_dropped() {
        let hits = vec![hit("notes", "unrelated.md", "H", 0.40)];

        assert!(consolidate(hits, &[]).is_empty());
    }

    /// The claim the whole slice rests on, checked at the wiring: a pasted log
    /// with the question underneath it must search the question too, and the
    /// question's match must reach the caller. The stub stands in for
    /// embed-and-search, so this needs neither a model nor a table.
    ///
    /// Searching only the first unit, or dropping the tail, are the wiring
    /// mistakes with the quietest symptoms: short queries keep working either
    /// way. Both fail here on the unit count.
    #[tokio::test]
    async fn a_match_in_only_the_last_unit_still_reaches_the_hits() {
        const QUESTION: &str = "why does the cache never expire?";
        let paste = "stack frame line\n".repeat(400);
        let query = format!("{paste}\n{QUESTION}");
        let units = chunk::split_query(&query, config::MAX_QUERY_UNITS);
        assert!(units.len() > 1, "the fixture query did not split");

        let searched = RefCell::new(Vec::new());
        let log = &searched;
        let hits = search_units(&query, config::MAX_QUERY_UNITS, |unit| async move {
            log.borrow_mut().push(unit.clone());
            // Only the tail carries the question, so only the tail matches.
            let matched = unit.contains(QUESTION);
            anyhow::Ok(if matched {
                vec![hit("notes", "cache.md", "Expiry", 0.88)]
            } else {
                vec![]
            })
        })
        .await
        .unwrap();

        // Counted before compared: a unit is a kilobyte of pasted log, and the
        // common failure -- searching a prefix of them -- is legible as a count
        // and unreadable as two dumped lists.
        let searched = searched.into_inner();
        assert_eq!(
            searched.len(),
            units.len(),
            "searched {} of {} units",
            searched.len(),
            units.len()
        );
        assert_eq!(searched, units, "the units reached the search mangled");
        assert_eq!(hits.len(), 1, "the tail unit's only match did not survive");
        assert_eq!(hits[0].rel_path, "cache.md");
    }

    /// The contract `Request::Search` now offers a host that has not read its
    /// callers: an empty query is no units, so no search runs and the answer
    /// is an empty hit list rather than an error. Pinned rather than merely
    /// commented, because it is the behaviour of `split_query` and nothing in
    /// this file would notice if that changed.
    #[tokio::test]
    async fn an_empty_query_searches_nothing_and_is_not_an_error() {
        for query in ["", "   ", "\n\t\n"] {
            let calls = RefCell::new(0_usize);
            let seen = &calls;
            let hits = search_units(query, config::MAX_QUERY_UNITS, |_unit| async move {
                *seen.borrow_mut() += 1;
                anyhow::Ok(vec![hit("notes", "anything.md", "H", 0.99)])
            })
            .await
            .unwrap();

            assert_eq!(calls.into_inner(), 0, "{query:?} reached the embedder");
            assert!(hits.is_empty(), "{query:?} produced hits");
        }
    }

    /// A failing unit fails the request. A short answer is indistinguishable
    /// from a complete one to the caller, so the other units' hits must not be
    /// handed back as if nothing went wrong.
    #[tokio::test]
    async fn one_unit_failing_fails_the_whole_search() {
        let paste = "stack frame line\n".repeat(400);
        let query = format!("{paste}\nwhy does the cache never expire?");

        let calls = RefCell::new(0_usize);
        let seen = &calls;
        let result = search_units(&query, config::MAX_QUERY_UNITS, |_unit| async move {
            *seen.borrow_mut() += 1;
            Err::<Vec<store::Hit>, _>(anyhow::anyhow!("embedder is wedged"))
        })
        .await;

        assert_eq!(
            result.unwrap_err().to_string(),
            "embedder is wedged",
            "the unit's own error has to survive to the caller"
        );
        assert_eq!(calls.into_inner(), 1, "kept searching after a failure");
    }

    fn chunk_hit(rel_path: &str, text: &str, cosine: f32) -> store::Hit {
        store::Hit {
            text: text.to_string(),
            ..hit("notes", rel_path, "H", cosine)
        }
    }

    /// Splitting makes one chunk matchable by several units. Without the
    /// dedup, `k` = 3 could be the same chunk three times.
    #[test]
    fn a_chunk_matched_by_two_units_is_returned_once_at_its_best_cosine() {
        let hits = vec![
            chunk_hit("cache.md", "the cache entry never expires", 0.71),
            chunk_hit("cache.md", "the cache entry never expires", 0.93),
        ];

        let out = best_k(hits, 3);

        assert_eq!(out.len(), 1, "the same chunk came back twice");
        assert!(
            (out[0].cosine - 0.93).abs() < f32::EPSILON,
            "{}",
            out[0].cosine
        );
    }

    /// Unlike the hook's `consolidate`, this path returns bodies, so two
    /// distinct chunks of one file are two distinct answers.
    #[test]
    fn two_chunks_of_one_file_both_stand() {
        let hits = vec![
            chunk_hit("cache.md", "eviction runs hourly", 0.90),
            chunk_hit("cache.md", "the TTL is never read", 0.80),
        ];

        assert_eq!(best_k(hits, 3).len(), 2);
    }

    /// `k` is what the caller asked for, not `k` per unit -- and the `k` kept
    /// are the best, whichever unit found them.
    #[test]
    fn the_hits_are_cut_to_k_best_first() {
        let hits = vec![
            chunk_hit("a.md", "first unit, weak", 0.40),
            chunk_hit("b.md", "first unit, strong", 0.95),
            chunk_hit("c.md", "last unit, middling", 0.70),
        ];

        let out = best_k(hits, 2);

        let texts: Vec<&str> = out.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, vec!["first unit, strong", "last unit, middling"]);
    }
}
