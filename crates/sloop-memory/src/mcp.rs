//! Minimal MCP stdio server.
//!
//! This is the on-demand half of the design. The prompt hook fires mechanically
//! but only ever hands over pointers; this exposes a tool the model can choose to
//! call when it actually wants the contents, with filters and a depth the hook
//! deliberately will not provide.
//!
//! Line-delimited JSON-RPC on stdin/stdout. Nothing but protocol may go to stdout
//! -- diagnostics go to stderr or they corrupt the stream.

use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};

use sloop_memory_core::proto::{Request, Response};

const FALLBACK_PROTOCOL_VERSION: &str = "2025-06-18";
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);

fn tool_definitions() -> Value {
    json!([
        {
            "name": "sloop_search",
            "description": "Hybrid search (vector + BM25) over this machine's private local \
                            memory roots: labeled directories of markdown, indexed together. \
                            Returns full chunks with source label, path, heading, and captured \
                            date. If the prompt already contains a <sloop-recall> pointer for \
                            this query, read that exact path instead of calling this tool and \
                            duplicating its content. Notes are snapshots; re-verify claims.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language or exact identifier, e.g. 'BGE tokenizer' or 'how does the debouncer coalesce writes'" },
                    "k": { "type": "integer", "minimum": 1, "maximum": 10, "description": "Max results (default 3; increase only when broader recall is needed)" },
                    "filter": { "type": "string", "description": "Optional SQL predicate pushed down before the scan, e.g. \"note_type = 'system'\" or \"rel_path LIKE 'notes/%'\"" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "sloop_status",
            "description": "Row counts and paths for the sloop-memory index, broken down by root. Useful to check whether the index is populated or stale.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

fn call_tool(name: &str, args: &Value) -> Value {
    let text = match name {
        "sloop_search" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if query.trim().is_empty() {
                return error_content("query is required");
            }
            let k = args
                .get("k")
                .and_then(Value::as_u64)
                .unwrap_or(3)
                .clamp(1, 10) as usize;
            let filter = args
                .get("filter")
                .and_then(Value::as_str)
                .map(str::to_string);

            match sloop_memory_core::client::request(
                &Request::Search {
                    query: query.to_string(),
                    k,
                    filter,
                },
                TOOL_TIMEOUT,
            ) {
                Ok(Response::Hits { hits, .. }) if hits.is_empty() => {
                    "No matching notes in the index.".to_string()
                }
                Ok(Response::Hits { hits, .. }) => hits
                    .iter()
                    .map(|h| {
                        let captured = if h.captured.is_empty() {
                            "unknown".into()
                        } else {
                            h.captured.clone()
                        };
                        format!(
                            "## {}:{} > {}\ncosine {:.3} | captured {}\n\n{}",
                            h.source_type, h.rel_path, h.heading_path, h.cosine, captured, h.text
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n"),
                Ok(Response::Error { message }) => return error_content(&message),
                Ok(_) => return error_content("unexpected response from daemon"),
                Err(e) => {
                    return error_content(&format!(
                        "daemon unreachable ({e}). Start it with `sloop-memory daemon`."
                    ))
                }
            }
        }
        "sloop_status" => {
            match sloop_memory_core::client::request(&Request::Status, TOOL_TIMEOUT) {
                Ok(Response::Status { roots, db }) => {
                    let counts = roots
                        .iter()
                        .map(|(label, count)| match count {
                            sloop_memory_core::store::RootCount::Rows(n) => {
                                format!("{label}: {n} rows")
                            }
                            sloop_memory_core::store::RootCount::Error(message) => {
                                format!("{label}: ERROR -- {message}")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("{counts}\ndb: {db}")
                }
                Ok(Response::Error { message }) => return error_content(&message),
                Ok(_) => return error_content("unexpected response from daemon"),
                Err(e) => return error_content(&format!("daemon unreachable: {e}")),
            }
        }
        other => return error_content(&format!("unknown tool: {other}")),
    };

    json!({ "content": [{ "type": "text", "text": text }] })
}

fn error_content(message: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": message }], "isError": true })
}

pub fn serve() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let method = req
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = req.get("id").cloned();

        // Notifications carry no id and must not be answered.
        if id.is_none() {
            continue;
        }

        let result = match method {
            "initialize" => {
                let version = req
                    .get("params")
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(Value::as_str)
                    .unwrap_or(FALLBACK_PROTOCOL_VERSION)
                    .to_string();
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "sloop-memory", "version": env!("CARGO_PKG_VERSION") }
                })
            }
            "tools/list" => json!({ "tools": tool_definitions() }),
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
                let name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                call_tool(name, &args)
            }
            "ping" => json!({}),
            _ => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("method not found: {method}") }
                });
                writeln!(stdout, "{response}")?;
                stdout.flush()?;
                continue;
            }
        };

        writeln!(
            stdout,
            "{}",
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        )?;
        stdout.flush()?;
    }
    Ok(())
}
