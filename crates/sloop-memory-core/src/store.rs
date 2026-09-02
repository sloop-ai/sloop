//! `LanceDB` schema, table lifecycle and hybrid retrieval.
//!
//! One table, `chunks`, shared by every configured root and distinguished by
//! `source_type`/`source_id` (see `crate::config::Root`) rather than by separate
//! tables per root. Roots differ in what they hold, not in how they are stored,
//! so one schema and one set of indexes serve all of them; a per-root table
//! would multiply index builds without buying any query the filter cannot
//! already express.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    Array, FixedSizeListBuilder, Float32Builder, Int32Array, ListBuilder, RecordBatch,
    RecordBatchIterator, StringArray, StringBuilder,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::scalar::{
    BTreeIndexBuilder, BitmapIndexBuilder, FtsIndexBuilder, LabelListIndexBuilder,
};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::rerankers::rrf::RRFReranker;
use lancedb::{Connection, Table};

use crate::config;

#[derive(Debug, Clone, Default)]
pub(crate) struct Row {
    pub(crate) id: String,
    pub(crate) text: String,
    pub(crate) vector: Vec<f32>,
    pub(crate) source_type: String,
    pub(crate) source_id: String,
    pub(crate) title: String,
    pub(crate) heading_path: String,
    pub(crate) rel_path: String,
    pub(crate) note_type: String,
    pub(crate) captured: String,
    pub(crate) tags: Vec<String>,
    pub(crate) entities: Vec<String>,
    pub(crate) content_hash: String,
    pub(crate) chunk_ix: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hit {
    /// RRF fusion score. Good for ordering, useless as a relevance gate: it is
    /// derived purely from rank. `LanceDB`'s default `RRFReranker` sums
    /// `1/(rank + k)` across the vector and BM25 lists, over zero-based ranks and
    /// with k = 60, so the ceiling is 2/k = 0.033, and a hit present in only one
    /// list gets a value fixed by its rank alone -- 1/k = 0.0167 at that list's
    /// top -- regardless of how well the chunk actually matches.
    pub score: f32,
    /// Cosine similarity against the query embedding. Unlike `score` this is
    /// calibrated and comparable across queries, so it is what the hook gates on.
    pub cosine: f32,
    pub title: String,
    pub rel_path: String,
    pub heading_path: String,
    pub captured: String,
    pub source_type: String,
    pub text: String,
}

/// One root's outcome from `root_row_counts`. Deliberately not just `usize`:
/// a query that fails must render differently from a root that legitimately
/// has zero rows, or the one distinction `status` exists to preserve --
/// "this root is missing files" vs "this root answered and has none" --
/// disappears back into a silent zero.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RootCount {
    Rows(usize),
    Error(String),
}

fn utf8_list_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    )
}

pub(crate) fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                config::EMBED_DIM,
            ),
            true,
        ),
        // Everything below is a filterable dimension. These are the columns that
        // make predicate pushdown worth having -- scoping a query to one note
        // type before the ANN scan, rather than filtering after.
        Field::new("source_type", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("heading_path", DataType::Utf8, false),
        Field::new("rel_path", DataType::Utf8, false),
        Field::new("note_type", DataType::Utf8, false),
        Field::new("captured", DataType::Utf8, false),
        utf8_list_field("tags"),
        utf8_list_field("entities"),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("chunk_ix", DataType::Int32, false),
    ]))
}

fn string_col(rows: &[Row], f: impl Fn(&Row) -> &str) -> Arc<dyn Array> {
    Arc::new(rows.iter().map(|r| Some(f(r))).collect::<StringArray>())
}

fn list_col(rows: &[Row], f: impl Fn(&Row) -> &Vec<String>) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(StringBuilder::new());
    for r in rows {
        for v in f(r) {
            b.values().append_value(v);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

pub(crate) fn to_batch(rows: &[Row]) -> Result<RecordBatch> {
    let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), config::EMBED_DIM)
        .with_field(Arc::new(Field::new("item", DataType::Float32, true)));
    for r in rows {
        anyhow::ensure!(
            r.vector.len() == config::EMBED_DIM as usize,
            "row {} has vector of length {}, expected {}",
            r.id,
            r.vector.len(),
            config::EMBED_DIM
        );
        vec_builder.values().append_slice(&r.vector);
        vec_builder.append(true);
    }

    RecordBatch::try_new(
        schema(),
        vec![
            string_col(rows, |r| &r.id),
            string_col(rows, |r| &r.text),
            Arc::new(vec_builder.finish()),
            string_col(rows, |r| &r.source_type),
            string_col(rows, |r| &r.source_id),
            string_col(rows, |r| &r.title),
            string_col(rows, |r| &r.heading_path),
            string_col(rows, |r| &r.rel_path),
            string_col(rows, |r| &r.note_type),
            string_col(rows, |r| &r.captured),
            list_col(rows, |r| &r.tags),
            list_col(rows, |r| &r.entities),
            string_col(rows, |r| &r.content_hash),
            Arc::new(rows.iter().map(|r| r.chunk_ix).collect::<Int32Array>()),
        ],
    )
    .context("building record batch")
}

/// # Errors
///
/// Returns an error if `dir` cannot be created, or `LanceDB` cannot open it.
pub async fn connect(dir: &std::path::Path) -> Result<Connection> {
    std::fs::create_dir_all(dir)?;
    lancedb::connect(dir.to_str().context("db path is not valid UTF-8")?)
        .execute()
        .await
        .context("opening lancedb")
}

/// # Errors
///
/// Returns an error if listing or opening the table fails.
pub async fn open_table(conn: &Connection, name: &str) -> Result<Option<Table>> {
    if !conn
        .table_names()
        .execute()
        .await?
        .iter()
        .any(|n| n == name)
    {
        return Ok(None);
    }
    Ok(Some(conn.open_table(name).execute().await?))
}

pub(crate) async fn ensure_table(conn: &Connection, name: &str) -> Result<Table> {
    if let Some(t) = open_table(conn, name).await? {
        return Ok(t);
    }
    conn.create_empty_table(name, schema())
        .execute()
        .await
        .with_context(|| format!("creating table {name}"))
}

/// Scalar indexes are cheap and always worth having. The vector index is not:
/// below `VECTOR_INDEX_MIN_ROWS` a flat scan beats `IVF_PQ` and training would fail
/// for want of rows, so we skip it and say so rather than failing opaquely.
pub(crate) async fn ensure_indexes(table: &Table) -> Result<Vec<String>> {
    let mut built = Vec::new();

    if table
        .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
        .is_ok()
    {
        built.push("fts(text)".into());
    }
    if table
        .create_index(
            &["source_type"],
            Index::Bitmap(BitmapIndexBuilder::default()),
        )
        .execute()
        .await
        .is_ok()
    {
        built.push("bitmap(source_type)".into());
    }
    // entities is a list column, so LabelList is the index that can answer
    // "which chunks mention [[Payments API]]" without a full scan.
    if table
        .create_index(
            &["entities"],
            Index::LabelList(LabelListIndexBuilder::default()),
        )
        .execute()
        .await
        .is_ok()
    {
        built.push("labellist(entities)".into());
    }

    let rows = table.count_rows(None).await?;
    if rows >= config::VECTOR_INDEX_MIN_ROWS {
        table
            .create_index(&["vector"], Index::IvfPq(IvfPqIndexBuilder::default()))
            .execute()
            .await?;
        built.push(format!("ivf_pq(vector) @ {rows} rows"));
    } else {
        built.push(format!(
            "vector: flat scan ({rows} rows < {} threshold)",
            config::VECTOR_INDEX_MIN_ROWS
        ));
    }
    Ok(built)
}

/// Cosine of row `i`'s stored embedding against the query. Both sides are already
/// L2-normalized, so a plain dot product is the cosine.
fn col_cosine(batch: &RecordBatch, i: usize, query: &[f32]) -> f32 {
    let Some(col) = batch.column_by_name("vector") else {
        return 0.0;
    };
    let Some(list) = col
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
    else {
        return 0.0;
    };
    if i >= list.len() || !list.is_valid(i) {
        return 0.0;
    }
    let values = list.value(i);
    let Some(floats) = values.as_any().downcast_ref::<arrow::array::Float32Array>() else {
        return 0.0;
    };
    floats.values().iter().zip(query).map(|(a, b)| a * b).sum()
}

fn col_str(batch: &RecordBatch, name: &str, i: usize) -> String {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .filter(|a| i < a.len() && a.is_valid(i))
        .map(|a| a.value(i).to_string())
        .unwrap_or_default()
}

/// Reranked hybrid results are scored under a different column than plain vector
/// results, and `LanceDB` has moved this name before. Probe rather than assume.
fn col_score(batch: &RecordBatch, i: usize) -> f32 {
    for name in ["_relevance_score", "_score", "_distance"] {
        if let Some(c) = batch.column_by_name(name) {
            if let Some(a) = c.as_any().downcast_ref::<arrow::array::Float32Array>() {
                if i < a.len() && a.is_valid(i) {
                    return a.value(i);
                }
            }
        }
    }
    0.0
}

/// Vector recall plus BM25, fused with reciprocal rank.
///
/// Pure embeddings are unreliable on exact identifiers -- short proper nouns,
/// versioned ids, config keys -- and BM25 alone misses paraphrase. The fusion is
/// the point.
///
/// # Errors
///
/// Returns an error if the query cannot be built or executed, or if a result
/// batch cannot be read.
pub async fn hybrid_search(
    table: &Table,
    query_vector: Vec<f32>,
    query_text: &str,
    limit: usize,
    filter: Option<&str>,
) -> Result<Vec<Hit>> {
    let query_for_cosine = query_vector.clone();
    let mut q = table
        .query()
        .nearest_to(query_vector)?
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .rerank(Arc::new(RRFReranker::default()))
        .limit(limit);
    if let Some(f) = filter {
        q = q.only_if(f);
    }

    let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
    let mut hits = Vec::new();
    for batch in &batches {
        for i in 0..batch.num_rows() {
            hits.push(Hit {
                score: col_score(batch, i),
                cosine: col_cosine(batch, i, &query_for_cosine),
                title: col_str(batch, "title", i),
                rel_path: col_str(batch, "rel_path", i),
                heading_path: col_str(batch, "heading_path", i),
                captured: col_str(batch, "captured", i),
                source_type: col_str(batch, "source_type", i),
                text: col_str(batch, "text", i),
            });
        }
    }
    Ok(hits)
}

// ---------------------------------------------------------------------------
// Manifest
//
// One row per source file, rather than per chunk.
//
// Change detection could read (source_id, content_hash) straight off the chunks
// table, but that means scanning every chunk of every file to learn about the
// files -- an order of magnitude more rows than the question needs, and the
// watcher asks it once per event. A separate manifest keyed on source_id turns
// the same question into a BTree point lookup.
// ---------------------------------------------------------------------------

pub(crate) const TABLE_MANIFEST: &str = "manifest";

#[derive(Debug, Clone)]
pub(crate) struct ManifestEntry {
    pub(crate) source_id: String,
    pub(crate) source_type: String,
    pub(crate) content_hash: String,
    pub(crate) n_chunks: i32,
    pub(crate) indexed_at: String,
}

pub(crate) fn manifest_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source_id", DataType::Utf8, false),
        Field::new("source_type", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("n_chunks", DataType::Int32, false),
        Field::new("indexed_at", DataType::Utf8, false),
    ]))
}

pub(crate) async fn ensure_manifest_table(conn: &Connection) -> Result<Table> {
    if let Some(t) = open_table(conn, TABLE_MANIFEST).await? {
        return Ok(t);
    }
    let table = conn
        .create_empty_table(TABLE_MANIFEST, manifest_schema())
        .execute()
        .await
        .context("creating manifest table")?;
    Ok(table)
}

/// `BTree` on `source_id` is what turns per-event change detection into a point
/// lookup instead of a scan.
///
/// # Errors
///
/// Never returns an error: index-creation failure is intentionally
/// swallowed, since a missing index only costs query speed, not correctness.
pub(crate) async fn ensure_manifest_index(table: &Table) -> Result<()> {
    let _ = table
        .create_index(&["source_id"], Index::BTree(BTreeIndexBuilder::default()))
        .execute()
        .await;
    Ok(())
}

/// A single-quoted SQL string literal, with embedded quotes doubled per the
/// standard SQL escape -- the only defense between a label like `o'reilly`
/// and a predicate that fails to parse (or worse, means something else).
fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn sql_in_list(values: &[String]) -> String {
    values
        .iter()
        .map(|v| sql_quote(v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Row count per configured root, so a root silently losing files is visible
/// in `Status` instead of being folded into one total that only ever drifts.
///
/// Counted with a `source_type = '<label>'` predicate per root rather than one
/// `count_rows(None)` -- the whole point is per-root numbers, not a shared sum.
/// A query failure becomes `RootCount::Error` rather than being swallowed to
/// zero, which would be indistinguishable from the root legitimately holding
/// no rows.
///
/// Called both by the daemon (answering `Request::Status`) and by `main`'s
/// `status` command when the daemon isn't running -- one implementation for
/// both, so they can't silently diverge.
pub async fn root_row_counts(
    conn: &Connection,
    roots: &config::Roots,
) -> BTreeMap<String, RootCount> {
    let mut counts = BTreeMap::new();
    let table = open_table(conn, config::TABLE_CHUNKS).await;
    for root in roots {
        let outcome = match &table {
            Ok(Some(t)) => {
                let filter = format!("source_type = {}", sql_quote(root.label.as_str()));
                match t.count_rows(Some(filter)).await {
                    Ok(n) => RootCount::Rows(n),
                    Err(e) => RootCount::Error(e.to_string()),
                }
            }
            Ok(None) => RootCount::Rows(0),
            Err(e) => RootCount::Error(e.to_string()),
        };
        counts.insert(root.label.to_string(), outcome);
    }
    counts
}

async fn manifest_rows(table: &Table, filter: Option<String>) -> Result<HashMap<String, String>> {
    let mut q = table
        .query()
        .select(Select::columns(&["source_id", "content_hash"]));
    if let Some(f) = filter {
        q = q.only_if(f);
    }
    let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
    let mut out = HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("source_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .context("manifest source_id column missing")?;
        let hashes = batch
            .column_by_name("content_hash")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .context("manifest content_hash column missing")?;
        for i in 0..batch.num_rows() {
            out.insert(ids.value(i).to_string(), hashes.value(i).to_string());
        }
    }
    Ok(out)
}

pub(crate) async fn manifest_all(table: &Table) -> Result<HashMap<String, String>> {
    manifest_rows(table, None).await
}

/// Hashes for just these sources -- the watcher path, served by the `BTree` index.
pub(crate) async fn manifest_lookup(
    table: &Table,
    source_ids: &[String],
) -> Result<HashMap<String, String>> {
    if source_ids.is_empty() {
        return Ok(HashMap::new());
    }
    manifest_rows(
        table,
        Some(format!("source_id IN ({})", sql_in_list(source_ids))),
    )
    .await
}

fn manifest_batch(entries: &[ManifestEntry]) -> Result<RecordBatch> {
    RecordBatch::try_new(
        manifest_schema(),
        vec![
            Arc::new(
                entries
                    .iter()
                    .map(|e| Some(e.source_id.as_str()))
                    .collect::<StringArray>(),
            ),
            Arc::new(
                entries
                    .iter()
                    .map(|e| Some(e.source_type.as_str()))
                    .collect::<StringArray>(),
            ),
            Arc::new(
                entries
                    .iter()
                    .map(|e| Some(e.content_hash.as_str()))
                    .collect::<StringArray>(),
            ),
            Arc::new(entries.iter().map(|e| e.n_chunks).collect::<Int32Array>()),
            Arc::new(
                entries
                    .iter()
                    .map(|e| Some(e.indexed_at.as_str()))
                    .collect::<StringArray>(),
            ),
        ],
    )
    .context("building manifest batch")
}

pub(crate) async fn manifest_upsert(table: &Table, entries: &[ManifestEntry]) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let batch = manifest_batch(entries)?;
    let reader: Box<dyn arrow::array::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(batch)], manifest_schema()));
    let mut op = table.merge_insert(&["source_id"]);
    op.when_matched_update_all(None)
        .when_not_matched_insert_all();
    op.execute(reader).await?;
    Ok(())
}

pub(crate) async fn manifest_remove(table: &Table, source_ids: &[String]) -> Result<()> {
    if source_ids.is_empty() {
        return Ok(());
    }
    table
        .delete(&format!("source_id IN ({})", sql_in_list(source_ids)))
        .await?;
    Ok(())
}

/// Replace every chunk belonging to `source_ids` in one transaction.
///
/// This is a MERGE rather than delete-then-append. Two reasons: it produces one
/// table version per run instead of two per file, and
/// `when_not_matched_by_source_delete` scoped to these sources is what correctly
/// drops orphan chunks when a note shrinks from ten sections to six.
///
/// The delete predicate is scoped to `source_ids`, which the caller has already
/// namespaced by root label (see `crate::index::source_id`) -- it can never reach
/// another root's rows.
pub(crate) async fn upsert_chunks(
    table: &Table,
    rows: &[Row],
    source_ids: &[String],
) -> Result<()> {
    if source_ids.is_empty() {
        return Ok(());
    }
    let batch = if rows.is_empty() {
        RecordBatch::new_empty(schema())
    } else {
        to_batch(rows)?
    };
    let reader: Box<dyn arrow::array::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema()));

    let mut op = table.merge_insert(&["id"]);
    op.when_matched_update_all(None)
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete(Some(format!(
            "source_id IN ({})",
            sql_in_list(source_ids)
        )));
    op.execute(reader).await?;
    Ok(())
}

#[cfg(test)]
// Tests are allowed to panic; a failing unwrap/expect (or a deliberate
// panic!() asserting on an error variant) *is* the assertion.
#[expect(clippy::unwrap_used, clippy::panic, reason = "see comment above")]
mod tests {
    use super::*;

    /// A fixed, cheap-to-construct vector standing in for a real embedding --
    /// these tests are about counting and escaping, not inference.
    fn row_for(root: &config::Root, rel: &str, seed: f32) -> Row {
        let source_id = format!("{}:{rel}", root.label);
        Row {
            id: source_id.clone(),
            text: "content".into(),
            vector: vec![seed; config::EMBED_DIM as usize],
            source_type: root.label.to_string(),
            source_id,
            rel_path: rel.to_string(),
            title: rel.to_string(),
            ..Default::default()
        }
    }

    async fn insert(table: &Table, rows: &[Row]) {
        let ids: Vec<String> = rows.iter().map(|r| r.source_id.clone()).collect();
        upsert_chunks(table, rows, &ids).await.unwrap();
    }

    /// Different counts per root -- if the implementation returned one shared
    /// total for every label instead of a per-root count, both labels would
    /// report the same number and this test would still catch it.
    #[tokio::test]
    async fn root_row_counts_reports_distinct_totals_per_root() {
        let db_dir = tempfile::tempdir().unwrap();
        let conn = connect(db_dir.path()).await.unwrap();
        let table = ensure_table(&conn, config::TABLE_CHUNKS).await.unwrap();

        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let spec = format!(
            "notes={}:memory={}",
            dir_a.path().display(),
            dir_b.path().display()
        );
        let roots = config::Roots::parse(&spec).unwrap();
        let mut it = roots.iter();
        let root_notes = it.next().unwrap();
        let root_memory = it.next().unwrap();

        insert(
            &table,
            &[
                row_for(root_notes, "a.md", 0.1),
                row_for(root_notes, "b.md", 0.2),
                row_for(root_memory, "c.md", 0.3),
            ],
        )
        .await;

        let counts = root_row_counts(&conn, &roots).await;
        assert_eq!(counts.get("notes"), Some(&RootCount::Rows(2)));
        assert_eq!(counts.get("memory"), Some(&RootCount::Rows(1)));
    }

    /// `RootLabel` forbids `:` and `=` (they separate `SLOOP_MEMORY_ROOTS`
    /// entries) but not `'`, so a label like `o'reilly` is legal -- and is
    /// exactly what `sql_quote`'s escaping exists to defend against. Without
    /// it the generated filter is `source_type = 'o'reilly'`, which is not
    /// valid SQL and fails the query rather than counting correctly.
    #[tokio::test]
    async fn root_row_counts_escapes_apostrophes_in_labels() {
        let db_dir = tempfile::tempdir().unwrap();
        let conn = connect(db_dir.path()).await.unwrap();
        let table = ensure_table(&conn, config::TABLE_CHUNKS).await.unwrap();

        let dir = tempfile::tempdir().unwrap();
        let spec = format!("o'reilly={}", dir.path().display());
        let roots = config::Roots::parse(&spec).unwrap();
        let root = roots.iter().next().unwrap();

        insert(
            &table,
            &[row_for(root, "a.md", 0.1), row_for(root, "b.md", 0.2)],
        )
        .await;

        let counts = root_row_counts(&conn, &roots).await;
        assert_eq!(counts.get("o'reilly"), Some(&RootCount::Rows(2)));
    }

    /// A query failure (I/O error, corrupt fragment, disk pressure) must be
    /// visible as `RootCount::Error`, never folded into `Rows(0)` -- that
    /// fold is indistinguishable from the root legitimately having no files,
    /// which defeats the entire point of per-root counts.
    #[tokio::test]
    async fn root_row_counts_reports_query_failures_instead_of_zero() {
        let db_dir = tempfile::tempdir().unwrap();
        let conn = connect(db_dir.path()).await.unwrap();
        let table = ensure_table(&conn, config::TABLE_CHUNKS).await.unwrap();

        let dir = tempfile::tempdir().unwrap();
        let spec = format!("notes={}", dir.path().display());
        let roots = config::Roots::parse(&spec).unwrap();
        let root = roots.iter().next().unwrap();
        insert(&table, &[row_for(root, "a.md", 0.1)]).await;
        drop(table);

        // Truncate the data fragment on disk. A filtered count (which is what
        // production always runs) has to scan it, so this reliably forces a
        // real query failure rather than one contrived at the API surface.
        let data_dir = db_dir
            .path()
            .join(format!("{}.lance", config::TABLE_CHUNKS))
            .join("data");
        for entry in std::fs::read_dir(&data_dir).unwrap() {
            std::fs::write(entry.unwrap().path(), b"not a lance file").unwrap();
        }

        let counts = root_row_counts(&conn, &roots).await;
        match counts.get("notes") {
            Some(RootCount::Error(_)) => {}
            other => panic!("expected a visible error for a corrupted root, got {other:?}"),
        }
    }
}
