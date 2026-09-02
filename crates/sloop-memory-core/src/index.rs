//! Ingestion.
//!
//! Re-embedding is the expensive step, so files are content-hashed and only
//! changed ones are touched. Change detection reads the manifest table (one row
//! per file, BTree-indexed) rather than projecting the chunks table, so the
//! watcher can ask about three paths without scanning everything.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use lancedb::Connection;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use walkdir::WalkDir;

use crate::{chunk, config, embed::Embedder, store};

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct IndexStats {
    pub scanned: usize,
    pub changed: usize,
    pub removed: usize,
    pub chunks_written: usize,
    pub indexes: Vec<String>,
    /// Where the wall clock actually went. Reported rather than guessed at: the
    /// split between embedding and writing is not what you would assume from the
    /// code, and only a measurement settles which one to optimize.
    pub embed_ms: u64,
    pub write_ms: u64,
    pub post_index_ms: u64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// File content plus the chunker version, so that changing how notes are split
/// invalidates the manifest without anyone having to remember `--full`.
fn content_hash(raw: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(chunk::CHUNKER_VERSION.to_le_bytes());
    h.update(raw);
    format!("{:x}", h.finalize())
}

#[must_use]
pub fn is_indexable(path: &Path) -> bool {
    path.extension().is_some_and(|x| x == "md")
}

/// Paths that must never reach the indexer or the watcher.
///
/// `.obsidian` is the important one: Obsidian rewrites `workspace.json` on every
/// pane change, so without this the watcher would reindex continuously.
#[must_use]
pub fn is_excluded(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == ".obsidian" || s == ".git" || s == ".trash" || s == ".DS_Store"
    })
}

pub(crate) fn markdown_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_excluded(e.path()))
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .filter(|p| is_indexable(p))
        .collect()
}

fn rel_id(root: &Path, path: &Path) -> String {
    if let Ok(rel) = path.strip_prefix(root) {
        return rel.to_string_lossy().to_string();
    }
    // Second chance for a path that is real but not yet resolved.
    if let Ok(canonical) = std::fs::canonicalize(path) {
        if let Ok(rel) = canonical.strip_prefix(root) {
            return rel.to_string_lossy().to_string();
        }
    }
    path.to_string_lossy().to_string()
}

/// Namespace a root-relative path by the root's label.
///
/// Two roots can legitimately contain the same relative path (both have a
/// `notes/foo.md`), so the relative path alone cannot be an identity: indexing
/// one root would overwrite the other's chunks. The label prefix is what keeps
/// the chunk id, the `source_id` and the manifest key distinct per root -- see
/// `config::RootLabel` for why the label may not contain `:`.
fn source_id(root: &config::Root, rel: &str) -> String {
    format!("{}:{rel}", root.label)
}

fn chunk_id(source_id: &str, chunk_ix: i32) -> String {
    sha256_hex(format!("{source_id}:{chunk_ix}").as_bytes())[..16].to_string()
}

/// Index (or re-index, or drop) exactly these paths, all belonging to `root`.
///
/// Paths that no longer exist on disk are pruned. Everything else is hashed
/// against the manifest and skipped when unchanged, unless `force`.
///
/// # Errors
///
/// Returns an error if a file cannot be read, hashed content cannot be
/// chunked or embedded, or the chunks/manifest tables cannot be written.
// Splitting this is real behavioral risk for no readability gain: the
// phases (prune, hash, embed, write) share too much local state to
// separate cheaply.
#[expect(clippy::too_many_lines, reason = "phases share local state")]
pub async fn index_paths(
    conn: &Connection,
    embedder: &Mutex<Embedder>,
    root: &config::Root,
    paths: &[PathBuf],
    force: bool,
) -> Result<IndexStats> {
    let mut stats = IndexStats::default();
    if paths.is_empty() {
        return Ok(stats);
    }

    let chunks_table = store::ensure_table(conn, config::TABLE_CHUNKS).await?;
    let manifest = store::ensure_manifest_table(conn).await?;

    let mut present: Vec<PathBuf> = Vec::new();
    let mut gone: Vec<String> = Vec::new();
    for path in paths {
        if is_excluded(path) || !is_indexable(path) {
            continue;
        }
        if path.is_file() {
            present.push(path.clone());
        } else {
            gone.push(source_id(root, &rel_id(root.dir.as_path(), path)));
        }
    }
    stats.scanned = present.len();

    if !gone.is_empty() {
        // An empty new-data set plus a by-source delete filter removes every
        // chunk belonging to these sources. Scoped to `gone`'s own (already
        // namespaced) source_ids, so this can never touch another root's rows.
        store::upsert_chunks(&chunks_table, &[], &gone).await?;
        store::manifest_remove(&manifest, &gone).await?;
        stats.removed = gone.len();
    }

    let ids: Vec<String> = present
        .iter()
        .map(|p| source_id(root, &rel_id(root.dir.as_path(), p)))
        .collect();
    let known: HashMap<String, String> = if force {
        HashMap::default()
    } else {
        store::manifest_lookup(&manifest, &ids).await?
    };

    let now = chrono::Local::now().to_rfc3339();
    let mut rows: Vec<store::Row> = Vec::new();
    let mut entries: Vec<store::ManifestEntry> = Vec::new();
    let mut changed_ids: Vec<String> = Vec::new();

    for path in &present {
        let rel = rel_id(root.dir.as_path(), path);
        let sid = source_id(root, &rel);
        let raw = std::fs::read(path).with_context(|| format!("reading {rel}"))?;
        let hash = content_hash(&raw);
        if !force && known.get(&sid) == Some(&hash) {
            continue;
        }
        stats.changed += 1;

        let source = String::from_utf8_lossy(&raw);
        let title = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let note = chunk::parse_note(&title, &source);

        let texts: Vec<String> = note.chunks.iter().map(|c| c.text.clone()).collect();
        // The lock is taken per file rather than for the whole run: a concurrent
        // hook query then waits for one file's embeddings (tens of ms), not for
        // the entire reindex.
        let vectors = {
            let t = std::time::Instant::now();
            let mut e = embedder.lock().await;
            let v = e
                .encode(&texts)
                .with_context(|| format!("embedding {rel}"))?;
            // u64::MAX ms is ~584 million years; a single embed call can't
            // realistically run long enough to overflow it.
            #[expect(clippy::cast_possible_truncation, reason = "see comment above")]
            {
                stats.embed_ms += t.elapsed().as_millis() as u64;
            }
            v
        };

        for (c, vector) in note.chunks.iter().zip(vectors) {
            rows.push(store::Row {
                id: chunk_id(&sid, c.ix),
                text: c.text.clone(),
                vector,
                source_type: root.label.to_string(),
                source_id: sid.clone(),
                title: note.title.clone(),
                heading_path: c.heading_path.clone(),
                rel_path: rel.clone(),
                note_type: note.frontmatter.note_type.clone(),
                captured: note.frontmatter.captured.clone(),
                tags: note.frontmatter.tags.clone(),
                entities: c.entities.clone(),
                content_hash: hash.clone(),
                chunk_ix: c.ix,
            });
        }
        // A note would need multiple GB of body text to produce i32::MAX chunks
        // at the smallest realistic chunk size (see chunk::TARGET_CHARS).
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "bounded by note size; see comment above"
        )]
        let n_chunks = note.chunks.len() as i32;
        entries.push(store::ManifestEntry {
            source_id: sid.clone(),
            source_type: root.label.to_string(),
            content_hash: hash,
            n_chunks,
            indexed_at: now.clone(),
        });
        changed_ids.push(sid);
    }

    let write_started = std::time::Instant::now();
    if !changed_ids.is_empty() {
        // One MERGE for the whole batch. At personal-corpus scale that is a single table
        // version for the run; a corpus-sized run would want to flush this
        // periodically rather than accumulate every row in memory.
        store::upsert_chunks(&chunks_table, &rows, &changed_ids).await?;
        store::manifest_upsert(&manifest, &entries).await?;
        stats.chunks_written = rows.len();
    }
    // u64::MAX ms is ~584 million years; a single indexing run can't
    // realistically run long enough to overflow it.
    #[expect(clippy::cast_possible_truncation, reason = "see comment above")]
    {
        stats.write_ms = write_started.elapsed().as_millis() as u64;
    }

    if stats.changed > 0 || stats.removed > 0 {
        let t = std::time::Instant::now();
        stats.indexes = store::ensure_indexes(&chunks_table).await?;
        store::ensure_manifest_index(&manifest).await?;
        #[expect(clippy::cast_possible_truncation, reason = "see comment above")]
        {
            stats.post_index_ms = t.elapsed().as_millis() as u64;
        }
    }
    Ok(stats)
}

/// Full sweep of one root: everything on disk, plus pruning anything the
/// manifest still lists for this root that has since been deleted.
pub(crate) async fn reindex_root(
    conn: &Connection,
    embedder: &Mutex<Embedder>,
    root: &config::Root,
    full: bool,
) -> Result<IndexStats> {
    let files = markdown_files(root.dir.as_path());
    let on_disk: HashSet<String> = files
        .iter()
        .map(|p| source_id(root, &rel_id(root.dir.as_path(), p)))
        .collect();

    let manifest = store::ensure_manifest_table(conn).await?;
    // The manifest is shared across every root, so orphan detection must only
    // consider this root's own (namespaced) entries -- otherwise another
    // root's source_id would be misread as a stale path under this one and
    // queued for deletion.
    let prefix = format!("{}:", root.label);
    let orphans: Vec<PathBuf> = store::manifest_all(&manifest)
        .await?
        .into_keys()
        .filter(|id| id.starts_with(&prefix) && !on_disk.contains(id))
        .map(|id| root.dir.as_path().join(&id[prefix.len()..]))
        .collect();

    let mut targets = files;
    targets.extend(orphans);
    index_paths(conn, embedder, root, &targets, full).await
}

/// Full sweep of every configured root, with stats summed across all of them.
///
/// # Errors
///
/// Returns an error if `reindex_root` fails for any configured root.
pub async fn reindex_all(
    conn: &Connection,
    embedder: &Mutex<Embedder>,
    roots: &config::Roots,
    full: bool,
) -> Result<IndexStats> {
    let mut total = IndexStats::default();
    for root in roots {
        let stats = reindex_root(conn, embedder, root, full).await?;
        total.scanned += stats.scanned;
        total.changed += stats.changed;
        total.removed += stats.removed;
        total.chunks_written += stats.chunks_written;
        total.embed_ms += stats.embed_ms;
        total.write_ms += stats.write_ms;
        total.post_index_ms += stats.post_index_ms;
        let labeled = stats
            .indexes
            .into_iter()
            .map(|line| format!("{}: {line}", root.label));
        total.indexes.extend(labeled);
    }
    Ok(total)
}

#[cfg(test)]
// Tests are allowed to panic; a failing unwrap/expect *is* the assertion.
#[expect(clippy::unwrap_used, reason = "see comment above")]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase, Select};

    /// A fixed, cheap-to-construct vector standing in for a real embedding.
    /// The bug under test is in id/key derivation, not in inference, so no
    /// ONNX model is needed to exercise it honestly.
    fn dummy_vector(seed: f32) -> Vec<f32> {
        vec![seed; config::EMBED_DIM as usize]
    }

    fn row_for(root: &config::Root, rel: &str, text: &str, seed: f32) -> store::Row {
        let sid = source_id(root, rel);
        store::Row {
            id: chunk_id(&sid, 0),
            text: text.to_string(),
            vector: dummy_vector(seed),
            source_type: root.label.to_string(),
            source_id: sid,
            title: rel.to_string(),
            heading_path: String::new(),
            rel_path: rel.to_string(),
            note_type: String::new(),
            captured: String::new(),
            tags: vec![],
            entities: vec![],
            content_hash: format!("hash-{seed}"),
            chunk_ix: 0,
        }
    }

    fn manifest_entry_for(row: &store::Row) -> store::ManifestEntry {
        store::ManifestEntry {
            source_id: row.source_id.clone(),
            source_type: row.source_type.clone(),
            content_hash: row.content_hash.clone(),
            n_chunks: 1,
            indexed_at: "2026-08-23T00:00:00+00:00".into(),
        }
    }

    /// All (`source_type`, text) pairs currently in the chunks table.
    async fn all_rows(table: &lancedb::Table) -> Vec<(String, String)> {
        let batches: Vec<arrow::array::RecordBatch> = table
            .query()
            .select(Select::columns(&["source_type", "text"]))
            .execute()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let types = batch
                .column_by_name("source_type")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let texts = batch
                .column_by_name("text")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                out.push((types.value(i).to_string(), texts.value(i).to_string()));
            }
        }
        out
    }

    /// Two roots holding the same relative path must not collide on identity.
    ///
    /// Without the label prefix, `notes/foo.md` in root A and `notes/foo.md` in
    /// root B would share one chunk id, one `source_id` and one manifest key --
    /// indexing B would silently overwrite or delete A's rows, with no error
    /// anywhere.
    #[tokio::test]
    async fn same_relative_path_in_two_roots_does_not_collide() {
        let db_dir = tempfile::tempdir().unwrap();
        let conn = store::connect(db_dir.path()).await.unwrap();
        let chunks_table = store::ensure_table(&conn, config::TABLE_CHUNKS)
            .await
            .unwrap();
        let manifest = store::ensure_manifest_table(&conn).await.unwrap();

        // RootDir::parse requires an existing directory, so the two roots need
        // real temp dirs -- going through config::Roots::parse rather than
        // hand-building Root is also the point: it's the same construction
        // path production code uses, no test-only bypass of the label rules.
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let spec = format!(
            "notes={}:memory={}",
            dir_a.path().display(),
            dir_b.path().display()
        );
        let roots = config::Roots::parse(&spec).unwrap();
        let mut it = roots.iter();
        let root_a = it.next().unwrap();
        let root_b = it.next().unwrap();
        let rel = "notes/foo.md";

        let row_a = row_for(root_a, rel, "alpha content", 0.1);
        let row_b = row_for(root_b, rel, "beta content", 0.2);

        store::upsert_chunks(
            &chunks_table,
            std::slice::from_ref(&row_a),
            std::slice::from_ref(&row_a.source_id),
        )
        .await
        .unwrap();
        store::manifest_upsert(&manifest, &[manifest_entry_for(&row_a)])
            .await
            .unwrap();

        store::upsert_chunks(
            &chunks_table,
            std::slice::from_ref(&row_b),
            std::slice::from_ref(&row_b.source_id),
        )
        .await
        .unwrap();
        store::manifest_upsert(&manifest, &[manifest_entry_for(&row_b)])
            .await
            .unwrap();

        let rows = all_rows(&chunks_table).await;
        assert!(
            rows.contains(&("notes".to_string(), "alpha content".to_string())),
            "root A's row is missing: {rows:?}"
        );
        assert!(
            rows.contains(&("memory".to_string(), "beta content".to_string())),
            "root B's row is missing: {rows:?}"
        );
        assert_ne!(
            row_a.id, row_b.id,
            "same relative path in two roots hashed to the same chunk id"
        );
        assert_ne!(row_a.source_id, row_b.source_id);

        // Re-index root A only: a content change, upserted with a delete scoped
        // to root A's own source_ids (store::upsert_chunks' existing behavior).
        let row_a2 = row_for(root_a, rel, "alpha content v2", 0.3);
        store::upsert_chunks(
            &chunks_table,
            std::slice::from_ref(&row_a2),
            std::slice::from_ref(&row_a2.source_id),
        )
        .await
        .unwrap();

        let rows = all_rows(&chunks_table).await;
        assert!(
            rows.contains(&("notes".to_string(), "alpha content v2".to_string())),
            "root A's updated row is missing: {rows:?}"
        );
        assert!(
            rows.contains(&("memory".to_string(), "beta content".to_string())),
            "root B's row was deleted by re-indexing root A: {rows:?}"
        );
    }
}
