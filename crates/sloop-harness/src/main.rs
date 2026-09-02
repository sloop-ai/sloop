//! Branching agent harness over the Anthropic Messages API.
//!
//! Nothing of the harness itself exists yet. This binary is here so the
//! library boundary is load-bearing from the start: `sloop-memory-core` is
//! linked directly rather than reached over a socket, and a build failure
//! here means it does not export enough to be usable outside its own
//! crate.

use std::io::Write as _;
use std::path::Path;

use sloop_memory_core::config::{db_dir, TABLE_CHUNKS};
use sloop_memory_core::index::is_indexable;

fn main() -> std::io::Result<()> {
    // Both calls are deliberately engine-side -- where the table lives, and
    // what the indexer will accept. Nothing here touches `proto`: that is the
    // daemon's socket protocol, and reaching for it would be exactly the
    // coupling that linking the library directly is meant to avoid.
    let table = db_dir().join(TABLE_CHUNKS);
    let indexable = is_indexable(Path::new("notes/branch-replay.md"));
    // `print_stdout` is denied workspace-wide. Writing to the stdout handle
    // through `io::Write` is not the macro path, so it needs no suppression.
    writeln!(
        std::io::stdout(),
        "{} indexable={indexable}",
        table.display()
    )
}
