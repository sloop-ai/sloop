//! Runtime configuration. Every path is env-overridable so the daemon, the CLI
//! and the hook can all be pointed at a scratch copy during testing.
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub(crate) const EMBED_DIM: i32 = 384;
pub(crate) const MAX_SEQ_LEN: usize = 512;

/// BGE is trained asymmetrically: the prefix goes on queries only. Putting it on
/// documents too measurably hurts retrieval.
pub(crate) const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// The only chunk table. Scope is deliberately local: this indexes markdown that
/// lives on this machine, in one or more labeled roots (see `Root`),
/// distinguished from each other by `source_type`/`source_id` rather than by
/// separate tables.
pub const TABLE_CHUNKS: &str = "chunks";

/// Below this, `LanceDB` brute-forces faster than `IVF_PQ` can be trained -- and with
/// too few rows training fails outright. Personal-scale corpora usually sit well
/// below it.
pub(crate) const VECTOR_INDEX_MIN_ROWS: usize = 25_000;

/// `FSEvents` legitimately coalesces and drops events across sleep and volume
/// remounts, so the watcher alone will drift eventually. A slow full sweep is the
/// backstop, and it is cheap because unchanged files are skipped on a hash.
pub const SWEEP_INTERVAL_SECS: u64 = 3600;

/// Editors autosave every few seconds while typing, and any bulk import writes
/// many files at once. Both want coalescing.
pub const WATCH_DEBOUNCE_SECS: u64 = 4;

/// The hook sits in front of the user's turn, so these are ceilings, not targets.
pub const HOOK_MAX_HITS: usize = 3;
/// Gate on cosine, not on the RRF score. Measured over technical markdown notes,
/// ~0.75 is where "the note is actually about this" starts; unrelated prose sits
/// near 0.3.
pub const HOOK_MIN_COSINE: f32 = 0.75;

/// How many units a long query searches before the middle of it is dropped.
///
/// A unit advances `chunk::TARGET_CHARS` less `chunk::OVERLAP_CHARS` of new
/// text, since each split carries its predecessor's tail forward -- so eight
/// is roughly 8,000 characters of distinct query at today's values. Past that
/// a paste carries less intent than the searches cost: measured over 220 real
/// prompts the median is 80 characters and 92% need no splitting at all.
/// Unlike the tokenizer's truncation this bound is stated rather than silent.
pub const MAX_QUERY_UNITS: usize = 8;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

// An unset HOME means the process environment itself is broken; there is no
// sane fallback, so fail immediately rather than build paths off an empty one.
#[expect(clippy::expect_used, reason = "see comment above")]
fn home() -> PathBuf {
    env_path("HOME").expect("HOME must be set")
}

/// Model weights are pinned into the nix store; there is deliberately no fallback,
/// because the failure mode we are avoiding is a silent network fetch.
///
/// # Errors
///
/// Returns an error if `SLOOP_MEMORY_MODEL` is unset.
pub fn model_dir() -> Result<PathBuf> {
    env_path("SLOOP_MEMORY_MODEL")
        .context("SLOOP_MEMORY_MODEL is unset -- run the nix-wrapped binary, not the raw cargo one")
}

/// A root's label. Cannot contain `:` or `=` -- both are `SLOOP_MEMORY_ROOTS`
/// delimiters, and either would corrupt parsing or, worse, corrupt the
/// injectivity of the namespaced `source_id` ("{label}:{rel}") without any parse
/// error to catch it: label "a" plus rel "b:c" would collide with label "a:b"
/// plus rel "c". Cannot be empty either. Guaranteed by construction -- the
/// only way to get one is `FromStr` -- so anything holding a `RootLabel` can
/// build an unambiguous `source_id` without re-checking. See
/// `crate::index::source_id`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootLabel(String);

impl RootLabel {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for RootLabel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        anyhow::ensure!(
            !s.is_empty(),
            "root label is empty -- SLOOP_MEMORY_ROOTS entries are `label=path`, and the label \
             cannot be blank"
        );
        anyhow::ensure!(
            !s.contains(':'),
            "root label {s:?} contains ':', which SLOOP_MEMORY_ROOTS uses to separate entries -- \
             rename the root"
        );
        anyhow::ensure!(
            !s.contains('='),
            "root label {s:?} contains '=', which SLOOP_MEMORY_ROOTS uses to separate a label \
             from its path -- rename the root"
        );
        Ok(RootLabel(s.to_string()))
    }
}

impl std::fmt::Display for RootLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An existing, canonicalized directory.
///
/// Canonicalization is not cosmetic: `FSEvents` reports resolved paths, and on
/// macOS `/tmp` is a symlink to `/private/tmp`, so an unresolved prefix fails
/// to strip in `index::rel_id` and every relative id silently becomes an
/// absolute path -- which then becomes the manifest key, breaking dedup and
/// pruning.
///
/// This proves the directory existed AT PARSE TIME, nothing more. The daemon
/// is long-lived, and a root can be deleted or unmounted later -- downstream
/// code still has to degrade gracefully if that happens (an empty
/// `markdown_files` scan, a failed watch) rather than assume the path is
/// forever valid. What this type buys is turning a config typo from
/// "indexes nothing, reports success" into "fails at startup naming the
/// path" -- a narrower, cheaper guarantee than "the path always exists".
#[derive(Debug, Clone)]
pub struct RootDir(PathBuf);

impl RootDir {
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    fn parse(path: &Path) -> Result<Self> {
        anyhow::ensure!(
            path.exists(),
            "root path {} does not exist", // Point at it before the config user
            path.display()                 // wonders why nothing got indexed.
        );
        anyhow::ensure!(
            path.is_dir(),
            "root path {} is not a directory",
            path.display()
        );
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("canonicalizing root path {}", path.display()))?;
        Ok(RootDir(canonical))
    }
}

/// A labeled indexing root. The label namespaces every id derived from files
/// under it -- `source_id`, the chunk id, and `source_type` -- so that two roots
/// holding the same relative path (e.g. both have `notes/foo.md`) never collide
/// on identity. See `crate::index::source_id`.
#[derive(Debug, Clone)]
pub struct Root {
    pub label: RootLabel,
    pub dir: RootDir,
}

/// Every configured indexing root. Non-empty, and every label unique -- both
/// checked here rather than on `Root`, because both are properties of the
/// whole set, not of any single root.
#[derive(Debug, Clone)]
pub struct Roots(Vec<Root>);

impl Roots {
    /// `spec` is colon-separated `label=path` pairs, e.g.
    /// `notes=/path/one:memory=/path/two`. The only constructor, so there is
    /// exactly one place that enforces the label and path rules.
    ///
    /// An empty entry -- a leading, trailing or doubled `:` -- is rejected
    /// rather than skipped. Skipping it means `notes=/x:` and a genuinely
    /// mistyped second entry both start clean and index fewer roots than the
    /// operator wrote, which is the silent-misconfiguration failure the rest
    /// of this file exists to prevent.
    ///
    /// The separator constrains the *path* as well as the label, which is the
    /// easier limitation to miss: `:` is a legal filename character on macOS,
    /// so a real root at `/Users/x/My: Notes` cannot be expressed here at all.
    /// The outer split cuts it into `notes=/Users/x/My` and ` Notes`; the tail
    /// carries no `=`, which is the detectable signature, so the message for a
    /// missing `=` names the colon rather than only blaming a missing label.
    /// Rename the directory or point a colon-free symlink at it.
    ///
    /// That is why every entry is split before any path is resolved. Checking
    /// as we go would resolve the truncated head first, and `/Users/x/My`
    /// either does not exist -- reporting a path the operator never wrote --
    /// or, worse, does, and the daemon starts having silently indexed the
    /// wrong directory. Both diagnoses point away from the actual colon.
    pub(crate) fn parse(spec: &str) -> Result<Self> {
        anyhow::ensure!(
            !spec.is_empty(),
            "SLOOP_MEMORY_ROOTS is empty -- provide at least one `label=path` entry"
        );
        let mut split = Vec::new();
        for entry in spec.split(':') {
            anyhow::ensure!(
                !entry.is_empty(),
                "SLOOP_MEMORY_ROOTS has an empty entry in {spec:?} -- entries are \
                 colon-separated `label=path` pairs, so a leading, trailing or doubled ':' is \
                 a typo, not an empty root"
            );
            let pair = entry.split_once('=').with_context(|| {
                format!(
                    "SLOOP_MEMORY_ROOTS entry {entry:?} is not `label=path` -- either the label \
                     is missing, or a root path contains ':', which SLOOP_MEMORY_ROOTS uses to \
                     separate entries. A path containing ':' cannot be expressed here: rename \
                     the directory or point a colon-free symlink at it"
                )
            })?;
            split.push(pair);
        }
        let mut parsed = Vec::new();
        for (label, path) in split {
            parsed.push(Root {
                label: label.parse()?,
                dir: RootDir::parse(Path::new(path))?,
            });
        }
        let mut seen = HashSet::new();
        for root in &parsed {
            anyhow::ensure!(
                seen.insert(root.label.clone()),
                "duplicate root label {:?} -- every root needs a unique label",
                root.label.as_str()
            );
        }
        Ok(Roots(parsed))
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Root> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a Roots {
    type Item = &'a Root;
    type IntoIter = std::slice::Iter<'a, Root>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Every configured indexing root, from `SLOOP_MEMORY_ROOTS`.
///
/// The variable is required: there is no default root, because guessing one
/// indexes the wrong directory silently, which is worse than not starting.
/// Its value is colon-separated `label=path` pairs -- see `Roots::parse`.
///
/// # Errors
///
/// Returns an error if `SLOOP_MEMORY_ROOTS` is unset, or if its value fails to
/// parse -- see `Roots::parse`.
pub fn roots() -> Result<Roots> {
    roots_from(std::env::var_os("SLOOP_MEMORY_ROOTS"))
}

/// The env-reading logic of `roots`, with the input passed in rather than read
/// directly, so tests can exercise the unset-`SLOOP_MEMORY_ROOTS` path without
/// mutating process-global env (which would need `unsafe` and would race other
/// tests running in parallel).
fn roots_from(spec: Option<std::ffi::OsString>) -> Result<Roots> {
    let spec = spec.context(
        "SLOOP_MEMORY_ROOTS is not set. Expected colon-separated label=path pairs, e.g. \
         \"notes=/path/to/notes:memory=/path/to/memory\".",
    )?;
    Roots::parse(&spec.to_string_lossy())
}

/// Which root (if any) a filesystem event path belongs to. Needed because the
/// watcher shares one debouncer across every root and emits paths without
/// saying which root they came from.
#[must_use]
pub fn root_for_path<'a>(roots: &'a Roots, path: &Path) -> Option<&'a Root> {
    if let Some(root) = roots.iter().find(|r| path.starts_with(r.dir.as_path())) {
        return Some(root);
    }
    let canonical = std::fs::canonicalize(path).ok()?;
    roots
        .iter()
        .find(|r| canonical.starts_with(r.dir.as_path()))
}

pub(crate) fn state_dir() -> PathBuf {
    env_path("SLOOP_MEMORY_STATE").unwrap_or_else(|| {
        env_path("XDG_STATE_HOME")
            .unwrap_or_else(|| home().join(".local/state"))
            .join("sloop-memory")
    })
}

#[must_use]
pub fn db_dir() -> PathBuf {
    state_dir().join("lancedb")
}

#[must_use]
pub fn socket_path() -> PathBuf {
    env_path("SLOOP_MEMORY_SOCKET").unwrap_or_else(|| state_dir().join("daemon.sock"))
}

/// Every hook injection is appended here as JSON. Without this, a hook that
/// silently feeds stale pointers into the prompt is undetectable; with it, the
/// failure mode is at least auditable after the fact.
#[must_use]
pub fn injection_log_path() -> PathBuf {
    env_path("SLOOP_MEMORY_INJECTION_LOG").unwrap_or_else(|| state_dir().join("injections.jsonl"))
}

#[cfg(test)]
// Tests are allowed to panic; a failing unwrap/expect *is* the assertion.
#[expect(clippy::unwrap_used, reason = "see comment above")]
mod tests {
    use super::*;

    fn existing_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    // These two are RootLabel::from_str tests, not Roots::parse tests: a colon
    // in a label position is always caught by the OUTER `:` split first (it
    // never survives to reach RootLabel with the colon still embedded), and
    // split_once('=') always cuts at the first '=', so a label can never
    // arrive at RootLabel containing one either. The rejection still belongs
    // on RootLabel itself -- it's the type's own invariant, not a property of
    // this one caller -- so it's tested at the level where it actually fires.

    #[test]
    fn label_containing_colon_is_rejected() {
        let err = "va:ult".parse::<RootLabel>().unwrap_err();
        assert!(
            err.to_string().contains(':'),
            "error should name the bad character: {err}"
        );
    }

    #[test]
    fn label_containing_equals_is_rejected() {
        let err = "va=ult".parse::<RootLabel>().unwrap_err();
        assert!(
            err.to_string().contains('='),
            "error should name the bad character: {err}"
        );
    }

    #[test]
    fn empty_label_is_rejected() {
        let dir = existing_dir();
        let spec = format!("={}", dir.path().display());
        assert!(Roots::parse(&spec).is_err());
    }

    /// A trailing ':' used to parse clean by skipping the empty entry, so a
    /// half-written second root started the daemon successfully and indexed
    /// only the first. The message has to name the variable and the format,
    /// because a startup error is all the operator gets.
    #[test]
    fn empty_entry_is_rejected_naming_the_variable_and_format() {
        let dir = existing_dir();
        let spec = format!("notes={}:", dir.path().display());
        let err = Roots::parse(&spec).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("SLOOP_MEMORY_ROOTS"), "got: {msg}");
        assert!(msg.contains("label=path"), "got: {msg}");
    }

    /// `:` is legal in a macOS filename, and the separator makes such a root
    /// inexpressible. This pins both halves of that. The head the split leaves
    /// behind (`<tmp>/My`) deliberately does not exist, so resolving entries as
    /// they come would fail on it first and report a path the operator never
    /// wrote -- which is why the structural check runs over the whole spec
    /// before any path is touched. And the message it raises has to name the
    /// colon, not just report a missing label, or it sends the operator looking
    /// for a typo that is not there.
    #[test]
    fn colon_in_a_path_is_rejected_naming_the_separator() {
        let dir = existing_dir();
        let with_colon = dir.path().join("My: Notes");
        std::fs::create_dir(&with_colon).unwrap();
        let spec = format!("notes={}", with_colon.display());
        let err = Roots::parse(&spec).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(':'), "error should name the separator: {msg}");
        assert!(
            msg.contains("cannot be expressed"),
            "error should say the path is inexpressible, not blame the label: {msg}"
        );
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let a = existing_dir();
        let b = existing_dir();
        let spec = format!("notes={}:notes={}", a.path().display(), b.path().display());
        let err = Roots::parse(&spec).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    #[test]
    fn nonexistent_path_is_rejected_naming_the_path() {
        let spec = "notes=/definitely/does/not/exist/sloop-memory-test";
        let err = Roots::parse(spec).unwrap_err();
        assert!(
            err.to_string()
                .contains("/definitely/does/not/exist/sloop-memory-test"),
            "error should name the missing path: {err}"
        );
    }

    #[test]
    fn file_instead_of_directory_is_rejected() {
        let dir = existing_dir();
        let file_path = dir.path().join("not-a-dir");
        std::fs::write(&file_path, b"x").unwrap();
        let spec = format!("notes={}", file_path.display());
        let err = Roots::parse(&spec).unwrap_err();
        assert!(err.to_string().contains("not a directory"), "got: {err}");
    }

    /// There is no default root: unset `SLOOP_MEMORY_ROOTS` is a startup error,
    /// and the message has to name the variable and show the format, because it
    /// is the only thing telling the operator what to set. Goes through
    /// `roots_from` directly rather than `roots()` itself, so this needs no
    /// process-global env mutation (which would need `unsafe` and would race
    /// any other test touching the same variables).
    #[test]
    fn unset_roots_env_is_rejected_naming_the_variable_and_format() {
        let err = roots_from(None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("SLOOP_MEMORY_ROOTS"), "got: {msg}");
        assert!(msg.contains("label=path"), "got: {msg}");
    }

    /// The set case still parses, so the error path above is a genuine branch
    /// rather than `roots_from` failing unconditionally.
    #[test]
    fn set_roots_env_is_parsed() {
        let dir = existing_dir();
        let spec = format!("notes={}", dir.path().display());
        let parsed = roots_from(Some(spec.into())).unwrap();
        let labeled: Vec<&Root> = parsed.iter().collect();
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].label.as_str(), "notes");
        assert_eq!(labeled[0].dir.as_path(), dir.path().canonicalize().unwrap());
    }
}
