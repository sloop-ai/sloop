mod daemon;
mod mcp;
mod watch;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Duration;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use sloop_memory_core::{
    client, config, embed, index,
    proto::{self, Request, Response},
    store,
};

#[derive(Parser)]
#[command(
    name = "sloop-memory",
    about = "Hybrid local search over private markdown roots"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ingest every configured root, re-embedding only files whose contents changed.
    Index {
        #[arg(long)]
        full: bool,
    },
    /// Hybrid (vector + BM25, RRF-fused) search.
    Search {
        query: String,
        #[arg(long, default_value = "3")]
        k: usize,
        /// SQL predicate pushed down before the scan, e.g. "`note_type` = 'system'".
        #[arg(long)]
        filter: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Run the resident daemon in the foreground, under a process supervisor.
    Daemon,
    /// `UserPromptSubmit` hook: reads the hook payload on stdin, emits pointers.
    Recall,
    /// MCP stdio server exposing `sloop_search` as an on-demand tool.
    Mcp,
    /// Row counts and paths.
    Status,
}

/// Anything on the prompt path must never block or fail loudly.
const HOOK_TIMEOUT: Duration = Duration::from_millis(250);
const CLI_TIMEOUT: Duration = Duration::from_mins(2);

// This is the hook's actual output contract: stdout is read by the calling
// Claude Code process, not a debugging artifact.
#[expect(clippy::print_stdout, reason = "see comment above")]
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        // Fail-open by construction: the hook path returns nothing and exits 0 on
        // any error, so a dead daemon degrades to "no injection", never to a
        // broken prompt.
        Cmd::Recall => {
            if let Some(context) = hook_context() {
                let payload = serde_json::json!({
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": context,
                    }
                });
                println!("{payload}");
            }
            Ok(())
        }
        Cmd::Mcp => mcp::serve(),
        Cmd::Daemon => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .init();
            tokio::runtime::Runtime::new()?.block_on(daemon::serve())
        }
        other => tokio::runtime::Runtime::new()?.block_on(run(other)),
    }
}

fn hook_context() -> Option<String> {
    let mut raw = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut raw).ok()?;
    let prompt = serde_json::from_str::<serde_json::Value>(&raw)
        .ok()?
        .get("prompt")?
        .as_str()?
        .to_string();

    if is_non_user_envelope(&prompt) {
        let _ = log_recall(&prompt, "skipped", &[], None, 0);
        return None;
    }

    let response = client::request(
        &Request::Recall {
            prompt: prompt.clone(),
        },
        HOOK_TIMEOUT,
    )
    .ok()?;
    let Response::Pointers {
        pointers,
        elapsed_ms,
        top_cosine,
    } = response
    else {
        return None;
    };
    let outcome = if pointers.is_empty() {
        "miss"
    } else {
        "injected"
    };
    let _ = log_recall(&prompt, outcome, &pointers, top_cosine, elapsed_ms);
    if pointers.is_empty() {
        return None;
    }

    Some(render_pointer_block(&pointers))
}

/// A root labeled exactly `memory` gets different framing from every other
/// root: it is taken to hold recorded facts -- about the user, the machine, or
/// past decisions -- as opposed to dated notes about something that exists
/// independently and may have moved on. The label is the whole signal; if you
/// name your facts root something else, change this constant rather than the
/// framing logic below.
const MEMORY_LABEL: &str = "memory";

/// Framing for pointers from a notes root: dated notes about something
/// that exists independently of the note, so the thing described may have
/// moved on since it was captured.
const NOTES_PREAMBLE: &str = "Relevant local note pointers. Read the exact file path; do \
    not call sloop_search for the same query. Dated notes are pointers, not evidence -- \
    the thing they describe may have changed since capture; re-verify against the live \
    source.\n";

/// Framing for pointers from the `memory` root: recorded facts with nothing
/// live to re-check them against, so the note itself is the source.
const MEMORY_PREAMBLE: &str = "Relevant recorded-fact pointers -- about this machine, past \
    decisions, or your stated preferences. Read the exact file path; do not call \
    sloop_search for the same query. The note is the source; there is nothing else to \
    check it against.\n";

/// Renders the block injected before the user's prompt. Pointers are grouped by
/// root so each group gets the framing suited to what that root actually is --
/// see `MEMORY_LABEL`, `NOTES_PREAMBLE`, and `MEMORY_PREAMBLE`.
fn render_pointer_block(pointers: &[proto::Pointer]) -> String {
    let mut memory = Vec::new();
    let mut notes = Vec::new();
    for p in pointers {
        if p.source_type == MEMORY_LABEL {
            memory.push(p);
        } else {
            notes.push(p);
        }
    }

    // Render whichever group appeared first in the input first, so a
    // single-root call has one obvious rendering rather than a fixed but
    // arbitrary group order.
    let memory_first = pointers
        .first()
        .is_some_and(|p| p.source_type == MEMORY_LABEL);
    let groups: [(&str, &[&proto::Pointer]); 2] = if memory_first {
        [(MEMORY_PREAMBLE, &memory), (NOTES_PREAMBLE, &notes)]
    } else {
        [(NOTES_PREAMBLE, &notes), (MEMORY_PREAMBLE, &memory)]
    };

    let mut out = String::from("<sloop-recall>\n");
    for (preamble, group) in groups {
        if group.is_empty() {
            continue;
        }
        out.push_str(preamble);
        for p in group {
            append_pointer_line(&mut out, p);
        }
    }
    out.push_str("</sloop-recall>");
    out
}

fn append_pointer_line(out: &mut String, p: &proto::Pointer) {
    let path = if p.path.is_empty() {
        &p.rel_path
    } else {
        &p.path
    };
    let where_ = if p.heading_path.is_empty() {
        path.clone()
    } else {
        format!("{path} > {}", p.heading_path)
    };
    let captured = if p.captured.is_empty() {
        "captured: unknown".into()
    } else {
        format!("captured {}", p.captured)
    };
    let source_type = if p.source_type.is_empty() {
        "notes"
    } else {
        &p.source_type
    };
    let _ = writeln!(
        out,
        "  {where_}  (source {source_type}, cos {:.2}, {captured})",
        p.cosine
    );
}

fn is_non_user_envelope(prompt: &str) -> bool {
    let prompt = prompt.trim_start();
    [
        "<task-notification>",
        "<task_notification>",
        "<local-command-caveat>",
        "<local-command-stdout>",
    ]
    .iter()
    .any(|prefix| prompt.starts_with(prefix))
}

fn log_recall(
    prompt: &str,
    outcome: &str,
    pointers: &[proto::Pointer],
    top_cosine: Option<f32>,
    elapsed_ms: u64,
) -> Result<()> {
    use std::io::Write;
    let path = config::injection_log_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let line = serde_json::json!({
        "at": chrono::Local::now().to_rfc3339(),
        "prompt": prompt,
        "outcome": outcome,
        "elapsed_ms": elapsed_ms,
        "top_cosine": top_cosine,
        "injected": pointers,
    });
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// Prefer the warm daemon; fall back to doing the work in-process so the CLI still
/// works when no daemon is running.
// This is the CLI's actual user-facing output, not a debugging artifact.
#[expect(clippy::print_stdout, reason = "see comment above")]
async fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Index { full } => {
            match client::request(&Request::Reindex { full }, CLI_TIMEOUT) {
                Ok(Response::Reindexed { stats }) => print_index_stats(&stats, "daemon"),
                Ok(Response::Error { message }) => anyhow::bail!("daemon: {message}"),
                _ => {
                    let conn = store::connect(&config::db_dir()).await?;
                    let embedder =
                        tokio::sync::Mutex::new(embed::Embedder::load(&config::model_dir()?)?);
                    let stats =
                        index::reindex_all(&conn, &embedder, &config::roots()?, full).await?;
                    print_index_stats(&stats, "in-process");
                }
            }
            Ok(())
        }

        Cmd::Search {
            query,
            k,
            filter,
            json,
        } => {
            let hits = match client::request(
                &Request::Search {
                    query: query.clone(),
                    k,
                    filter: filter.clone(),
                },
                CLI_TIMEOUT,
            ) {
                Ok(Response::Hits { hits, .. }) => hits,
                Ok(Response::Error { message }) => anyhow::bail!("daemon: {message}"),
                _ => {
                    let conn = store::connect(&config::db_dir()).await?;
                    let Some(table) = store::open_table(&conn, config::TABLE_CHUNKS).await? else {
                        anyhow::bail!("no index yet; run `sloop-memory index`");
                    };
                    let mut embedder = embed::Embedder::load(&config::model_dir()?)?;
                    let qvec = embedder.encode_query(&query)?;
                    store::hybrid_search(&table, qvec, &query, k, filter.as_deref()).await?
                }
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for h in &hits {
                    let where_ = if h.heading_path.is_empty() {
                        h.rel_path.clone()
                    } else {
                        format!("{} > {}", h.rel_path, h.heading_path)
                    };
                    println!("rrf={:.4} cos={:.4}  {where_}", h.score, h.cosine);
                    let snippet: String = h.text.chars().take(120).collect();
                    println!("        {}", snippet.replace('\n', " "));
                }
            }
            Ok(())
        }

        Cmd::Status => {
            if let Ok(Response::Status { roots, db }) =
                client::request(&Request::Status, CLI_TIMEOUT)
            {
                println!("daemon:  running");
                return status_exit(print_status(&roots, &db));
            }
            println!("daemon:  not running");
            let conn = store::connect(&config::db_dir()).await?;
            let roots = store::root_row_counts(&conn, &config::roots()?).await;
            let ok = print_status(&roots, &config::db_dir().display().to_string());
            status_exit(ok)
        }

        Cmd::Daemon | Cmd::Recall | Cmd::Mcp => {
            unreachable!("handled before the runtime starts")
        }
    }
}

/// Shared by both the daemon-up and daemon-down `status` branches so the two
/// can never render differently formatted output. Returns `false` if any root
/// failed to report a count, so the caller can fail the command visibly
/// rather than exiting 0 over a partial answer.
// This is the CLI's actual user-facing output, not a debugging artifact.
#[expect(clippy::print_stdout, reason = "see comment above")]
fn print_status(roots: &BTreeMap<String, store::RootCount>, db: &str) -> bool {
    let mut all_ok = true;
    for (label, count) in roots {
        match count {
            store::RootCount::Rows(n) => println!("{label}:   {n} rows"),
            store::RootCount::Error(message) => {
                println!("{label}:   ERROR -- {message}");
                all_ok = false;
            }
        }
    }
    println!("db:      {db}");
    all_ok
}

/// `status` is meant to be scriptable -- a monitoring job checking for a root
/// silently losing files needs a non-zero exit when any root failed to
/// answer, not just a line of text a human might not be watching.
fn status_exit(all_ok: bool) -> Result<()> {
    if all_ok {
        Ok(())
    } else {
        bail!("one or more roots failed to report a row count")
    }
}

// This is the CLI's actual user-facing output, not a debugging artifact.
#[expect(clippy::print_stdout, reason = "see comment above")]
fn print_index_stats(stats: &index::IndexStats, via: &str) {
    println!(
        "scanned {} notes, {} changed, {} removed, {} chunks written ({via})",
        stats.scanned, stats.changed, stats.removed, stats.chunks_written
    );
    println!(
        "  time: embed {}ms, write {}ms, index-build {}ms",
        stats.embed_ms, stats.write_ms, stats.post_index_ms
    );
    for line in &stats.indexes {
        println!("  index: {line}");
    }
}

#[cfg(test)]
// Tests are allowed to panic; a failing unwrap (or a deliberate assert!) *is*
// the assertion.
#[expect(clippy::unwrap_used, reason = "see comment above")]
mod tests {
    use super::{
        is_non_user_envelope, print_status, render_pointer_block, status_exit, MEMORY_PREAMBLE,
        NOTES_PREAMBLE,
    };
    use sloop_memory_core::proto::Pointer;
    use sloop_memory_core::store::RootCount;
    use std::collections::BTreeMap;

    fn pointer(source_type: &str, title: &str) -> Pointer {
        Pointer {
            title: title.to_string(),
            source_type: source_type.to_string(),
            path: String::new(),
            rel_path: format!("{title}.md"),
            heading_path: "Some Heading".to_string(),
            captured: "2026-08-01".to_string(),
            cosine: 0.42,
        }
    }

    /// Framing must key off the root label, not what's in the note -- two
    /// pointers with identical content prove the label alone drives it.
    #[test]
    fn framing_follows_the_label_not_the_content() {
        let memory_ptr = pointer("memory", "same-title");
        let notes_ptr = pointer("notes", "same-title");
        let block = render_pointer_block(&[memory_ptr, notes_ptr]);

        let memory_start = block.find(MEMORY_PREAMBLE).unwrap();
        let notes_start = block.find(NOTES_PREAMBLE).unwrap();
        let (memory_section, notes_section) = if memory_start < notes_start {
            (&block[memory_start..notes_start], &block[notes_start..])
        } else {
            (&block[memory_start..], &block[notes_start..memory_start])
        };

        assert!(!memory_section.contains("re-verify"));
        assert!(!memory_section.contains("live source"));
        assert!(notes_section.contains("re-verify"));
        assert!(notes_section.contains("live source"));
    }

    /// A single root must render exactly one group -- no empty second
    /// preamble, no stray separator left over from the other branch.
    #[test]
    fn single_root_renders_one_group() {
        let block = render_pointer_block(&[pointer("notes", "only-one")]);

        assert!(block.contains(NOTES_PREAMBLE));
        assert!(!block.contains(MEMORY_PREAMBLE));
    }

    /// An unrecognized label is not "memory", so it must fall back to notes
    /// framing -- the rule is "memory is special", not "first root wins".
    #[test]
    fn unknown_label_gets_notes_framing() {
        let block = render_pointer_block(&[pointer("scratch", "mystery-root")]);

        assert!(block.contains(NOTES_PREAMBLE));
        assert!(!block.contains(MEMORY_PREAMBLE));
    }

    /// The notes preamble must not claim the corpus is memory -- a notes root is
    /// whatever markdown tree the operator pointed at, so calling it "memory" is
    /// a false claim about what the reader is being handed.
    #[test]
    fn notes_preamble_does_not_claim_to_be_memory() {
        assert!(!NOTES_PREAMBLE.to_lowercase().contains("memory"));
    }

    /// Symmetric check: the memory preamble must not carry the notes framing
    /// -- a memory root is recorded facts, not a dated snapshot of something
    /// that exists independently and may have moved on since capture.
    #[test]
    fn memory_preamble_does_not_frame_its_corpus_as_dated_notes() {
        let preamble = MEMORY_PREAMBLE.to_lowercase();
        assert!(!preamble.contains("dated"));
        assert!(!preamble.contains("re-verify"));
        assert!(!preamble.contains("live source"));
    }

    #[test]
    fn skips_task_and_local_command_envelopes() {
        assert!(is_non_user_envelope("<task-notification>\nfinished"));
        assert!(is_non_user_envelope("  <task_notification>finished"));
        assert!(is_non_user_envelope("<local-command-stdout>done"));
    }

    #[test]
    fn keeps_real_user_prompts_including_xml_discussion() {
        assert!(!is_non_user_envelope(
            "Explain <task-notification> handling"
        ));
        assert!(!is_non_user_envelope("How does the debouncer work?"));
    }

    /// A root with zero rows is a legitimate answer; a root that failed to
    /// report a count is not the same thing and must not be misreadable as
    /// one -- that's the entire reason `RootCount` isn't a bare `usize`.
    #[test]
    fn print_status_flags_a_root_that_failed_to_report_a_count() {
        let mut roots = BTreeMap::new();
        roots.insert("notes".to_string(), RootCount::Rows(0));
        roots.insert(
            "memory".to_string(),
            RootCount::Error("disk read failed".into()),
        );
        assert!(!print_status(&roots, "/tmp/db"));
    }

    #[test]
    fn print_status_succeeds_when_every_root_answers() {
        let mut roots = BTreeMap::new();
        roots.insert("notes".to_string(), RootCount::Rows(3));
        assert!(print_status(&roots, "/tmp/db"));
    }

    /// `status` is meant to be scriptable, e.g. a monitoring job
    /// checking for a root silently losing files -- it needs a real exit
    /// code, not just text a human might not be watching.
    #[test]
    fn status_exit_fails_the_command_when_any_root_errored() {
        assert!(status_exit(true).is_ok());
        assert!(status_exit(false).is_err());
    }
}
