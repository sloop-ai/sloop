//! Markdown chunking for notes.
//!
//! Sections are cut on heading boundaries because that is where these notes
//! actually change subject. Each chunk carries its heading path in the embedded
//! text, which gives both the vector and the BM25 side something to bite on when
//! a chunk's body is mostly prose that never repeats the term.

use std::collections::BTreeSet;

/// Bumped whenever chunking changes. It is folded into each file's content hash,
/// so a chunker change invalidates every entry and the next sweep re-embeds --
/// otherwise the index keeps chunks that the current code would never produce and
/// nothing detects it, because the *files* did not change.
pub(crate) const CHUNKER_VERSION: u32 = 1;

/// Bodies longer than this get split on paragraph boundaries.
const TARGET_CHARS: usize = 1200;
const MAX_CHARS: usize = 2000;
/// Carried between splits so a fact spanning the cut is still retrievable.
const OVERLAP_CHARS: usize = 200;

/// One indexed unit: a section, or a slice of one too long to embed whole.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// What actually gets embedded and full-text indexed: context header + body.
    pub text: String,
    /// The `>`-joined headings above this chunk, keyed per section rather than
    /// per chunk -- so a section split across several chunks repeats its path
    /// on every one of them.
    pub heading_path: String,
    /// Position within the file, in document order.
    pub ix: i32,
    /// Every `[[wikilink]]` target found in the body.
    pub entities: Vec<String>,
}

/// The three YAML keys the indexer reads. Absent keys come back empty rather
/// than missing, so a note without frontmatter parses like one with blank
/// values.
#[derive(Debug, Clone, Default)]
pub struct Frontmatter {
    pub note_type: String,
    pub captured: String,
    pub tags: Vec<String>,
}

/// A whole markdown file, chunked.
#[derive(Debug, Clone)]
pub struct ParsedNote {
    pub title: String,
    pub frontmatter: Frontmatter,
    pub chunks: Vec<Chunk>,
}

/// Minimal YAML: enough for `type`, `captured` and `tags`, in either inline
/// (`[a, b]`) or block (`- a`) form. A real YAML parser would be a dependency
/// bought for three keys of a format we control.
fn parse_frontmatter(lines: &[&str]) -> (Frontmatter, usize) {
    let mut fm = Frontmatter::default();
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        return (fm, 0);
    }
    let Some(end) = lines[1..].iter().position(|l| l.trim_end() == "---") else {
        return (fm, 0);
    };
    let end = end + 1;

    let mut current_list_key: Option<String> = None;
    for raw in &lines[1..end] {
        let line = raw.trim_end();
        if let Some(item) = line.trim_start().strip_prefix("- ") {
            if current_list_key.as_deref() == Some("tags") {
                fm.tags.push(item.trim().trim_matches('"').to_string());
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        current_list_key = Some(key.to_string());
        match key {
            "type" => fm.note_type = value.to_string(),
            "captured" => fm.captured = value.to_string(),
            "tags" => {
                if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                    fm.tags = inner
                        .split(',')
                        .map(|t| t.trim().trim_matches('"').to_string())
                        .filter(|t| !t.is_empty())
                        .collect();
                }
            }
            _ => {}
        }
    }
    (fm, end + 1)
}

/// Pulls `[[Target]]`, `[[Target|alias]]` and `[[Target#heading]]` down to the
/// target. These become a LabelList-indexed column, giving exact entity lookup
/// that neither BM25 nor embeddings do reliably for short proper nouns.
fn extract_wikilinks(text: &str) -> Vec<String> {
    let mut found = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            if let Some(close) = text[i + 2..].find("]]") {
                let inner = &text[i + 2..i + 2 + close];
                let target = inner
                    .split(['|', '#'])
                    .next()
                    .unwrap_or(inner)
                    .trim()
                    .to_string();
                if !target.is_empty() {
                    found.insert(target);
                }
                i = i + 2 + close + 2;
                continue;
            }
        }
        i += 1;
    }
    found.into_iter().collect()
}

fn tail_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars[chars.len().saturating_sub(n)..].iter().collect()
}

fn hard_split(s: &str) -> Vec<String> {
    s.chars()
        .collect::<Vec<char>>()
        .chunks(MAX_CHARS)
        .map(|c| c.iter().collect())
        .collect()
}

/// Break a section into units that each fit `MAX_CHARS`.
///
/// Splitting on blank lines alone is not enough: a markdown table or a long list
/// has no blank lines at all, so it arrives as one paragraph. Left unsplit, one
/// becomes a 9000-character chunk that the tokenizer truncates at 512 tokens, so
/// the tail of a large table is invisible to vector search while BM25 can still
/// see it. Fall back to lines, then to a hard character split.
fn units(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in body.split("\n\n") {
        if para.chars().count() <= MAX_CHARS {
            out.push(para.to_string());
            continue;
        }
        let mut buf = String::new();
        for line in para.lines() {
            if line.chars().count() > MAX_CHARS {
                if !buf.trim().is_empty() {
                    out.push(std::mem::take(&mut buf));
                }
                buf.clear();
                out.extend(hard_split(line));
                continue;
            }
            if buf.chars().count() + line.chars().count() + 1 > TARGET_CHARS {
                out.push(std::mem::take(&mut buf));
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
        if !buf.trim().is_empty() {
            out.push(buf);
        }
    }
    out
}

/// Pack units into chunks of roughly `TARGET_CHARS` with a little overlap, and
/// guarantee nothing exceeds `MAX_CHARS` -- overlap and packing can both overshoot.
fn split_body(body: &str) -> Vec<String> {
    let mut packed = Vec::new();
    let mut current = String::new();
    for unit in units(body) {
        if !current.is_empty() && current.chars().count() + unit.chars().count() > TARGET_CHARS {
            let tail = tail_chars(&current, OVERLAP_CHARS);
            packed.push(std::mem::take(&mut current));
            current = tail;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&unit);
    }
    if !current.trim().is_empty() {
        packed.push(current);
    }
    packed
        .into_iter()
        .flat_map(|p| {
            if p.chars().count() <= MAX_CHARS {
                vec![p]
            } else {
                hard_split(&p)
            }
        })
        .collect()
}

/// Chunk one markdown file.
///
/// `title` is the filename stem rather than the H1 (see `index::note_title`),
/// and it is prepended to every chunk's embedded text as
/// `"{title} > {heading_path}"` -- which is how a heading travels into the
/// vector and BM25 sides as well as into the rendered pointer.
///
/// Public so that producers of markdown can test what the chunker makes of
/// their output. `sloop-harness`'s transcript renderer is the case that needs
/// it: the safety of the whole transcript design rests on a `### Not
/// continued` heading reaching `heading_path`, and that is a property of the
/// two together, not of either alone.
#[must_use]
pub fn parse_note(title: &str, source: &str) -> ParsedNote {
    let lines: Vec<&str> = source.lines().collect();
    let (frontmatter, start) = parse_frontmatter(&lines);

    // (level, text) stack describing where we are in the heading tree.
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut body = String::new();
    let mut in_fence = false;

    let heading_path = |stack: &[(usize, String)]| -> String {
        stack
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" > ")
    };

    for line in &lines[start.min(lines.len())..] {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        // A `#` inside a fence is code, not a heading.
        if !in_fence && trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            if (1..=6).contains(&level) && trimmed.chars().nth(level) == Some(' ') {
                if !body.trim().is_empty() {
                    sections.push((heading_path(&stack), std::mem::take(&mut body)));
                }
                body.clear();
                stack.truncate(level.saturating_sub(1));
                stack.push((level, trimmed[level + 1..].trim().to_string()));
                continue;
            }
        }
        body.push_str(line);
        body.push('\n');
    }
    if !body.trim().is_empty() {
        sections.push((heading_path(&stack), body));
    }

    let mut chunks = Vec::new();
    for (path, section_body) in sections {
        for piece in split_body(section_body.trim()) {
            if piece.trim().is_empty() {
                continue;
            }
            // The context header is why a chunk buried deep under a heading is still
            // findable by the terms its parent sections establish but its own prose
            // never repeats.
            let header = if path.is_empty() {
                title.to_string()
            } else {
                format!("{title} > {path}")
            };
            let text = format!("{header}\n\n{piece}");
            let entities = extract_wikilinks(&piece);
            // A note would need multiple GB of body text to produce i32::MAX chunks
            // at the smallest realistic chunk size (TARGET_CHARS/MAX_CHARS above).
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "bounded by note size; see comment above"
            )]
            let ix = chunks.len() as i32;
            chunks.push(Chunk {
                text,
                heading_path: path.clone(),
                ix,
                entities,
            });
        }
    }

    ParsedNote {
        title: title.to_string(),
        frontmatter,
        chunks,
    }
}

#[cfg(test)]
// Tests are allowed to panic; a failing unwrap/expect *is* the assertion.
#[expect(clippy::unwrap_used, reason = "see comment above")]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    /// A markdown table has no blank lines, so blank-line splitting alone would
    /// leave 9000-character chunks that the tokenizer silently truncates at 512
    /// tokens. This is the case `units` exists for.
    #[test]
    fn table_without_blank_lines_is_split_under_cap() {
        let mut table = String::new();
        for i in 0..400 {
            writeln!(
                table,
                "| endpoint-{i} | POST /v1/thing/{i} | stores a record |"
            )
            .unwrap();
        }
        let note = parse_note("Big Table", &format!("# Big Table\n\n{table}"));
        assert!(note.chunks.len() > 1, "table should have been split");
        for c in &note.chunks {
            assert!(
                c.text.chars().count() <= MAX_CHARS + 200,
                "chunk of {} chars exceeds the cap",
                c.text.chars().count()
            );
        }
    }

    #[test]
    fn a_single_unsplittable_line_is_hard_split() {
        let line = "x".repeat(MAX_CHARS * 3);
        let note = parse_note("Wall", &format!("# Wall\n\n{line}"));
        assert!(note.chunks.len() >= 3);
    }

    #[test]
    fn heading_path_tracks_nesting_and_ignores_fences() {
        let src = "# Top\n\nintro\n\n## Middle\n\nbody\n\n```\n# not a heading\n```\n";
        let note = parse_note("Doc", src);
        let paths: Vec<&str> = note
            .chunks
            .iter()
            .map(|c| c.heading_path.as_str())
            .collect();
        assert!(paths.contains(&"Top"), "got {paths:?}");
        assert!(paths.iter().any(|p| p.contains("Middle")), "got {paths:?}");
        assert!(
            !paths.iter().any(|p| p.contains("not a heading")),
            "fence leaked: {paths:?}"
        );
    }

    #[test]
    fn wikilinks_normalise_to_targets() {
        let note = parse_note(
            "Links",
            "# Links\n\nSee [[Payments API]], [[Rate Limiting|limits]] and [[Glossary#SLO]].\n",
        );
        let mut found: Vec<String> = note
            .chunks
            .iter()
            .flat_map(|c| c.entities.clone())
            .collect();
        found.sort();
        assert_eq!(found, vec!["Glossary", "Payments API", "Rate Limiting"]);
    }

    #[test]
    fn frontmatter_is_parsed_and_excluded_from_body() {
        let note = parse_note(
            "FM",
            "---\ntype: system\ncaptured: 2026-04-10\ntags: [infra, caching]\n---\n\n# FM\n\nbody here\n",
        );
        assert_eq!(note.frontmatter.note_type, "system");
        assert_eq!(note.frontmatter.captured, "2026-04-10");
        assert_eq!(note.frontmatter.tags, vec!["infra", "caching"]);
        assert!(!note.chunks.iter().any(|c| c.text.contains("captured:")));
    }
}
