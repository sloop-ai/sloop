//! Wire protocol between the CLI/hook and the daemon.
//!
//! Newline-delimited JSON over a unix socket: one request, one response, close.
//! Deliberately dumb -- the hook path needs to be cheap and easy to debug with
//! `nc`, not clever.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn default_k() -> usize {
    3
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Ping,
    Status,
    Search {
        query: String,
        #[serde(default = "default_k")]
        k: usize,
        #[serde(default)]
        filter: Option<String>,
    },
    /// Hook path: returns pointers only, never chunk bodies.
    Recall {
        prompt: String,
    },
    Reindex {
        #[serde(default)]
        full: bool,
    },
}

/// A pointer at a note. Carries `captured` so staleness is visible at the point of
/// use rather than something the reader has to go and check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pointer {
    pub title: String,
    #[serde(default)]
    pub source_type: String,
    #[serde(default)]
    pub path: String,
    pub rel_path: String,
    pub heading_path: String,
    pub captured: String,
    pub cosine: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    Pong {
        pid: u32,
    },
    Status {
        /// Row count (or query failure) per root label. `BTreeMap` rather than
        /// `HashMap` so CLI output ordering is deterministic.
        roots: BTreeMap<String, crate::store::RootCount>,
        db: String,
    },
    Hits {
        hits: Vec<crate::store::Hit>,
        elapsed_ms: u64,
    },
    Pointers {
        pointers: Vec<Pointer>,
        elapsed_ms: u64,
        /// Best candidate before the hook threshold is applied. Logging this
        /// for misses makes threshold tuning evidence-based rather than guesswork.
        #[serde(default)]
        top_cosine: Option<f32>,
    },
    Reindexed {
        stats: crate::index::IndexStats,
    },
    Error {
        message: String,
    },
}
