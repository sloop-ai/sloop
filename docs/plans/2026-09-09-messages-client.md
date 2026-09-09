# Messages Client Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give `sloop-harness` a streaming client for `/v1/messages`, so a branch of the conversation tree can be sent, and a real assistant turn — thinking blocks included — comes back into the tree.

**Architecture:** One new module, `api`, split so that everything except the socket call is a pure function over bytes. `api::send` takes the `Vec<Message>` that `Tree::prompt_for` already returns and yields completed `ContentBlock`s; the caller appends them. The tree stays sync, pure, and ignorant of HTTP; the client stays ignorant of forks and status labels. Blocks are appended at `content_block_stop`, so the tree needs no mutation operation and stays strictly append-only.

**Tech Stack:** Rust (pinned via `rust-toolchain.toml`), `reqwest` 0.12 over rustls, `tokio`, `futures`, `serde`/`serde_json`, `anyhow`, `proptest`. Everything runs inside `nix develop`.

**Design doc:** `docs/plans/2026-09-09-messages-client-design.md`. Read it first — it explains *why* thinking blocks are forced into this slice and why the tree is append-only.

---

## Conventions in this codebase

Read these before Task 1. They are not negotiable and the linter enforces most of them.

- **Every command runs inside the nix shell:** `nix develop -c <cmd>`. A bare `cargo` may pick up a different toolchain and fail with E0514.
- **Lints are deny-by-default** (`Cargo.toml`, `[workspace.lints.clippy]`): no `unwrap`, no `panic`, no `todo`, no `dbg!`, no `println!`/`eprintln!` (`print_stdout` and `print_stderr` are denied — write to a locked `stdout` handle with `writeln!` instead), and `allow_attributes` is denied, so silence a lint with `#[expect(..., reason = "...")]`, never `#[allow(...)]`.
- **Tests are inline**, in a `#[cfg(test)] mod tests` at the bottom of the file they cover. See `crates/sloop-harness/src/tree.rs:264`. The test module carries `#[expect(clippy::unwrap_used, reason = "...")]` so tests may use `unwrap`.
- **Comments explain why, never what.** Match the surrounding density — this codebase comments decisions and constraints, not mechanics.
- **Commit after every task**, with an imperative subject ≤72 chars.

---

## Task 1: Dependencies, flake fileset, cargoHash

No behavior change. This task exists on its own because the `cargoHash` update has to land in the same commit as the `Cargo.lock` change, and because the flake fileset change is easy to forget until CI fails.

**Files:**
- Modify: `crates/sloop-harness/Cargo.toml`
- Modify: `flake.nix:66-73` (the `src` fileset)
- Modify: `flake.nix:81` (`cargoHash`)
- Modify: `Cargo.lock` (generated)

**Step 1: Add the dependencies**

In `crates/sloop-harness/Cargo.toml`, under `[dependencies]`, after the existing `serde` / `serde_json` block:

```toml
# There is no official Anthropic SDK for Rust, so this talks raw HTTP. reqwest,
# hyper, rustls and ring are already in the lock file by way of the lance tree,
# so taking reqwest directly adds no crate and no license that is not already
# vendored. rustls rather than native-tls keeps the build free of a system
# OpenSSL on linux.
reqwest = { version = "0.12", default-features = false, features = [
  "charset",
  "http2",
  "json",
  "rustls-tls-native-roots",
  "stream",
] }
# bytes_stream() yields a futures Stream; StreamExt::next is how it is read.
futures = "0.3"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1"
```

**Step 2: Widen the flake fileset to carry SSE fixtures**

The package build runs `cargoTestFlags = ["--workspace"]`, so the harness's tests run inside `nix build`. Its fixtures are `.sse` files and the current filter admits only `.rs` and `Cargo.toml`, so those tests would pass locally and fail in the sandbox. In `flake.nix`, change the filter:

```nix
                (nixpkgs.lib.fileset.fileFilter
                  (f: f.hasExt "rs" || f.hasExt "sse" || f.name == "Cargo.toml")
                  ./crates)
```

And extend the comment above it with a sentence saying why `.sse` is there:

```
            # rust-toolchain.toml is excluded too: the compiler comes
            # from makeRustPlatform, and cargo would only read that file
            # through a rustup shim that does not exist in the sandbox.
            # .sse files are the harness's recorded SSE fixtures. The tests
            # that read them run in this sandbox, so they have to be here.
```

**Step 3: Update the lock file**

Run: `nix develop -c cargo check -p sloop-harness`
Expected: compiles, and `Cargo.lock` is modified. If the diff pulls in a large set of new crates rather than a handful, a feature is wrong — check with `nix develop -c cargo tree -p reqwest -e features` and prefer the features already enabled by the transitive copy.

**Step 4: Get the new cargoHash**

Run: `nix build .#sloop-memory.cargoDeps --no-link`
Expected: FAILS with a hash mismatch that prints `specified:` and `got:`. Copy the `got:` value into `cargoHash` in `flake.nix`.

**Step 5: Verify the hash**

Run: `nix build .#sloop-memory.cargoDeps --no-link`
Expected: succeeds.

**Step 6: Commit**

```bash
git add crates/sloop-harness/Cargo.toml Cargo.lock flake.nix
git commit -m "Add the harness's HTTP dependencies"
```

---

## Task 2: The Thinking content block

Thinking is on by default on `claude-opus-5`, so the first real response contains a block the current `ContentBlock` cannot hold. The signature is the part that matters: it must round-trip byte-identically or the next turn is rejected.

**Files:**
- Modify: `crates/sloop-harness/src/tree.rs:36-48` (the `ContentBlock` enum and its impl)
- Modify: `crates/sloop-harness/src/tree.rs` (test module)

**Step 1: Write the failing tests**

In the `mod tests` block, in the replay section:

```rust
#[test]
fn a_thinking_block_round_trips_byte_identically() {
    // The signature binds the conversation prefix that produced the block.
    // A representation that does not survive a round trip is rejected on the
    // next turn, so this is the guarantee, not a serialization detail.
    let wire = r#"{"type":"thinking","thinking":"Weighing LRU against TTL.","signature":"ErUBCkYIBRgCIkA="}"#;

    let block: ContentBlock = serde_json::from_str(wire).unwrap();
    assert_eq!(serde_json::to_string(&block).unwrap(), wire);
}

#[test]
fn a_turn_may_open_with_thinking_and_continue_in_text() {
    let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
    let thinking = tree
        .append(
            tree.root(),
            Role::Assistant,
            ContentBlock::thinking("Weighing LRU against TTL.", "ErUBCkYIBRgCIkA="),
        )
        .unwrap();
    let text = tree
        .append(thinking, Role::Assistant, ContentBlock::text("Two options."))
        .unwrap();

    // One assistant turn, two blocks: the grouping rule does not care which
    // kinds they are.
    let messages = tree.replay(text).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].content.len(), 2);
}
```

**Step 2: Run them to verify they fail**

Run: `nix develop -c cargo test -p sloop-harness thinking`
Expected: FAIL — `no variant of enum ContentBlock named Thinking` / `no function or associated item named thinking`.

**Step 3: Add the variant**

In `tree.rs`, replace the `ContentBlock` enum and impl:

```rust
/// One content block, serialized in the shape `/v1/messages` expects.
///
/// `tool_use` and `tool_result` are still absent; they arrive with the slice
/// that has tools to call. `thinking` is here because there is no request
/// shape that avoids it: thinking is on by default on the model this crate
/// talks to, so a response contains one whether or not the caller asked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// The `signature` binds the conversation prefix that produced this block.
    /// It travels back unchanged or the turn after it is rejected, which is
    /// why nothing here reformats or normalizes it.
    Thinking {
        thinking: String,
        signature: String,
    },
}

impl ContentBlock {
    /// A text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// A thinking block, with the signature that authenticates it.
    pub fn thinking(thinking: impl Into<String>, signature: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: thinking.into(),
            signature: signature.into(),
        }
    }
}
```

**Step 4: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness`
Expected: PASS, all of them. The existing property tests still hold — they generate text blocks only.

**Step 5: Extend one property test to generate both kinds**

Find the proptest strategy that produces blocks (near `tree.rs:561`) and make it produce thinking blocks too, so the replay properties are checked over mixed turns:

```rust
fn any_block() -> impl Strategy<Value = ContentBlock> {
    prop_oneof![
        "[a-z ]{1,20}".prop_map(ContentBlock::text),
        ("[a-z ]{1,20}", "[A-Za-z0-9+/=]{8,16}")
            .prop_map(|(thinking, signature)| ContentBlock::thinking(thinking, signature)),
    ]
}
```

Wire it into the existing tree-shape strategy wherever `ContentBlock::text` is currently generated.

**Step 6: Run the property tests**

Run: `nix develop -c cargo test -p sloop-harness`
Expected: PASS. If a property fails, read the shrunk counterexample before changing anything — the properties are the specification.

**Step 7: Commit**

```bash
git add crates/sloop-harness/src/tree.rs
git commit -m "Add the thinking content block"
```

---

## Task 3: SSE framing

The transport is a byte stream, and the frames in it do not align with the chunks reqwest hands over. This is the first pure piece.

**Files:**
- Create: `crates/sloop-harness/src/api/sse.rs`
- Create: `crates/sloop-harness/src/api.rs`
- Modify: `crates/sloop-harness/src/main.rs` (add `mod api;`)

**Step 1: Create the module skeleton**

`crates/sloop-harness/src/api.rs`:

```rust
//! The `/v1/messages` client.
//!
//! Named `api` rather than `client` on purpose. `client` in this workspace is
//! the memory daemon's socket client, and this crate exists partly to show
//! what linking the engine directly looks like instead.
//!
//! Everything here except [`Api::send`] is a pure function over bytes, which
//! is what lets the whole decoder be tested in a sandbox with no network and
//! no API key -- the environment `nix build` runs the test suite in.

mod sse;
```

Add `mod api;` to `main.rs` beside `mod tree;`.

**Step 2: Write the failing tests**

`crates/sloop-harness/src/api/sse.rs`:

```rust
//! Server-sent events, decoded without touching a socket.

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "inputs are literals defined in the test")]
mod tests {
    use super::{data_of, FrameDecoder};

    #[test]
    fn a_frame_split_across_chunks_is_reassembled() {
        let mut decoder = FrameDecoder::default();

        assert!(decoder.push(b"event: ping\ndata: {\"type\"").unwrap().is_empty());
        let frames = decoder.push(b":\"ping\"}\n\n").unwrap();

        assert_eq!(frames.len(), 1);
        assert_eq!(data_of(&frames[0]).as_deref(), Some(r#"{"type":"ping"}"#));
    }

    #[test]
    fn several_frames_in_one_chunk_come_back_in_order() {
        let mut decoder = FrameDecoder::default();
        let frames = decoder.push(b"data: one\n\ndata: two\n\ndata: thr").unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(data_of(&frames[0]).as_deref(), Some("one"));
        assert_eq!(data_of(&frames[1]).as_deref(), Some("two"));
    }

    #[test]
    fn a_frame_with_no_data_line_has_no_payload() {
        assert_eq!(data_of("event: ping\n\n"), None);
    }

    #[test]
    fn multiple_data_lines_join_with_a_newline() {
        assert_eq!(data_of("data: one\ndata: two\n\n").as_deref(), Some("one\ntwo"));
    }
}
```

**Step 3: Run to verify they fail**

Run: `nix develop -c cargo test -p sloop-harness sse`
Expected: FAIL to compile — `FrameDecoder` and `data_of` do not exist.

**Step 4: Implement**

Above the test module in `sse.rs`:

```rust
use anyhow::{Context, Result};

/// Splits a byte stream into SSE frames.
///
/// Frames are separated by a blank line, and the chunk boundaries a HTTP
/// stream hands over have nothing to do with them: one chunk may carry three
/// frames, or half of one. The buffer is what bridges that.
#[derive(Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl FrameDecoder {
    /// Feed a chunk, and take whatever frames it completed.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buffer.extend_from_slice(chunk);

        let mut frames = Vec::new();
        while let Some(end) = separator(&self.buffer) {
            let frame: Vec<u8> = self.buffer.drain(..end + SEPARATOR.len()).collect();
            frames.push(String::from_utf8(frame).context("an SSE frame was not UTF-8")?);
        }
        Ok(frames)
    }
}

const SEPARATOR: &[u8] = b"\n\n";

fn separator(buffer: &[u8]) -> Option<usize> {
    buffer.windows(SEPARATOR.len()).position(|pair| pair == SEPARATOR)
}

/// The payload of a frame: its `data:` lines, joined.
///
/// The `event:` line is deliberately ignored. Every payload carries its own
/// `type`, so reading both would give one fact two sources of truth.
pub fn data_of(frame: &str) -> Option<String> {
    let mut payload: Option<String> = None;

    for line in frame.lines() {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.strip_prefix(' ').unwrap_or(rest);

        match &mut payload {
            Some(joined) => {
                joined.push('\n');
                joined.push_str(rest);
            }
            None => payload = Some(rest.to_owned()),
        }
    }

    payload
}
```

**Step 5: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness sse`
Expected: PASS, 4 tests.

**Step 6: Commit**

```bash
git add crates/sloop-harness/src/api.rs crates/sloop-harness/src/api/sse.rs crates/sloop-harness/src/main.rs
git commit -m "Decode SSE frames from a chunked byte stream"
```

---

## Task 4: Event decoding

**Files:**
- Modify: `crates/sloop-harness/src/api/sse.rs`

**Step 1: Write the failing tests**

Add to `mod tests`:

```rust
use super::{Delta, Event};

#[test]
fn a_text_delta_decodes() {
    let event: Event =
        serde_json::from_str(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Two"}}"#)
            .unwrap();

    match event {
        Event::ContentBlockDelta { index, delta: Delta::Text { text } } => {
            assert_eq!(index, 0);
            assert_eq!(text, "Two");
        }
        other => panic!("wrong event: {other:?}"),
    }
}

#[test]
fn a_signature_delta_is_its_own_kind() {
    let event: Event =
        serde_json::from_str(r#"{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"ErUB"}}"#)
            .unwrap();

    assert!(matches!(
        event,
        Event::ContentBlockDelta { delta: Delta::Signature { .. }, .. }
    ));
}

#[test]
fn an_unknown_event_type_decodes_rather_than_failing() {
    // The server may add events. A decoder that rejects an unrecognized type
    // turns every such addition into an outage.
    let event: Event = serde_json::from_str(r#"{"type":"some_future_event","payload":9}"#).unwrap();

    assert!(matches!(event, Event::Unknown));
}

#[test]
fn an_error_event_carries_its_kind_and_message() {
    let event: Event = serde_json::from_str(
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    )
    .unwrap();

    match event {
        Event::Error { error } => {
            assert_eq!(error.kind, "overloaded_error");
            assert_eq!(error.message, "Overloaded");
        }
        other => panic!("wrong event: {other:?}"),
    }
}
```

`panic!` is denied by the workspace lints, so add `clippy::panic` to the test module's `#[expect]` list:

```rust
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test that decodes the wrong variant should fail loudly"
)]
```

**Step 2: Run to verify they fail**

Run: `nix develop -c cargo test -p sloop-harness sse`
Expected: FAIL to compile — `Event` and `Delta` do not exist.

**Step 3: Implement**

Add to `sse.rs`, above the tests:

```rust
use serde::Deserialize;

/// One decoded SSE event.
///
/// [`Event::Unknown`] is the important variant: it is what keeps a server-side
/// addition from becoming a client-side failure.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    MessageStart,
    ContentBlockStart { index: usize, content_block: BlockStart },
    ContentBlockDelta { index: usize, delta: Delta },
    ContentBlockStop { index: usize },
    MessageDelta { delta: MessageDelta },
    MessageStop,
    Error { error: ApiError },
    #[serde(other)]
    Unknown,
}

/// The opening shape of a block, which is what says how to accumulate it.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockStart {
    Text,
    Thinking,
    /// A kind this slice does not model -- `tool_use`, and whatever comes
    /// later. Accumulated as nothing rather than refused.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    /// A thinking block's signature arrives separately from its text. A
    /// decoder that accumulated only the text would produce a block that
    /// serializes without one and is rejected on the next turn.
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}
```

Note `MessageStart` takes no fields: `#[serde(tag = "type")]` ignores unknown keys, and nothing in this slice reads the opening message envelope.

**Step 4: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness sse`
Expected: PASS, 8 tests.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/api/sse.rs
git commit -m "Decode the message stream's event types"
```

---

## Task 5: The accumulator, against recorded fixtures

Events in, completed blocks out. This is where a fixture becomes worth more than a hand-built event list: it is the only readable record of the wire format in the repo.

**Files:**
- Create: `crates/sloop-harness/src/api/fixtures/text-only.sse`
- Create: `crates/sloop-harness/src/api/fixtures/thinking-then-text.sse`
- Create: `crates/sloop-harness/src/api/fixtures/overloaded-mid-stream.sse`
- Create: `crates/sloop-harness/src/api/accumulate.rs`
- Modify: `crates/sloop-harness/src/api.rs` (add `mod accumulate;`)

**Step 1: Write the fixtures**

`text-only.sse`:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":14,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Two "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"options."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}

event: message_stop
data: {"type":"message_stop"}

```

`thinking-then-text.sse` — same shape, but block 0 is thinking and block 1 is text:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":14,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Weighing LRU "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"against TTL."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"ErUBCkYIBRgCIkA="}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Two options."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":31}}

event: message_stop
data: {"type":"message_stop"}

```

`overloaded-mid-stream.sse` — a turn that starts and then fails:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_03","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":14,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Two "}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}

```

**Step 2: Write the failing tests**

`crates/sloop-harness/src/api/accumulate.rs`:

```rust
//! Events in, completed content blocks out.

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixtures are checked in beside this file; a mis-decode should fail loudly"
)]
mod tests {
    use super::Accumulator;
    use crate::api::sse::{data_of, Event, FrameDecoder};
    use crate::tree::ContentBlock;

    /// Replay a recorded stream the way `send` does, one byte at a time.
    ///
    /// Byte-at-a-time is deliberate: it is the harshest possible chunking, and
    /// it is the case a decoder that assumes frame-aligned chunks fails on.
    fn replay(fixture: &str) -> anyhow::Result<(Vec<ContentBlock>, Option<String>)> {
        let mut frames = FrameDecoder::default();
        let mut accumulator = Accumulator::default();
        let mut blocks = Vec::new();

        for byte in fixture.as_bytes() {
            for frame in frames.push(&[*byte])? {
                let Some(data) = data_of(&frame) else { continue };
                if let Some(block) = accumulator.apply(serde_json::from_str(&data)?)? {
                    blocks.push(block);
                }
            }
        }

        Ok((blocks, accumulator.stop_reason().map(str::to_owned)))
    }

    #[test]
    fn text_deltas_concatenate_into_one_block() {
        let (blocks, stop) = replay(include_str!("fixtures/text-only.sse")).unwrap();

        assert_eq!(blocks, vec![ContentBlock::text("Two options.")]);
        assert_eq!(stop.as_deref(), Some("end_turn"));
    }

    #[test]
    fn a_thinking_block_keeps_its_signature() {
        let (blocks, _) = replay(include_str!("fixtures/thinking-then-text.sse")).unwrap();

        assert_eq!(
            blocks,
            vec![
                ContentBlock::thinking("Weighing LRU against TTL.", "ErUBCkYIBRgCIkA="),
                ContentBlock::text("Two options."),
            ]
        );
    }

    #[test]
    fn a_recorded_thinking_block_re_serializes_unchanged() {
        // The guarantee this whole slice turns on. A ContentBlock that does
        // not survive the round trip is rejected on the turn after it.
        let (blocks, _) = replay(include_str!("fixtures/thinking-then-text.sse")).unwrap();

        assert_eq!(
            serde_json::to_string(&blocks[0]).unwrap(),
            r#"{"type":"thinking","thinking":"Weighing LRU against TTL.","signature":"ErUBCkYIBRgCIkA="}"#
        );
    }

    #[test]
    fn an_error_event_fails_the_turn() {
        let failure = replay(include_str!("fixtures/overloaded-mid-stream.sse")).unwrap_err();

        assert!(failure.to_string().contains("overloaded_error"));
    }

    #[test]
    fn interleaved_block_indices_land_in_the_right_blocks() {
        // Indices arrive in order in practice. Relying on that buys nothing,
        // and makes the failure silent and content-corrupting if it stops
        // being true.
        let mut accumulator = Accumulator::default();
        let events = [
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"second"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_stop","index":0}"#,
        ];

        let blocks: Vec<ContentBlock> = events
            .iter()
            .filter_map(|event| accumulator.apply(serde_json::from_str(event).unwrap()).unwrap())
            .collect();

        assert_eq!(
            blocks,
            vec![ContentBlock::text("second"), ContentBlock::text("first")]
        );
    }

    #[test]
    fn a_block_kind_this_slice_does_not_model_is_dropped() {
        let mut accumulator = Accumulator::default();
        let start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"x","input":{}}}"#;
        let stop = r#"{"type":"content_block_stop","index":0}"#;

        assert!(accumulator.apply(serde_json::from_str(start).unwrap()).unwrap().is_none());
        assert!(accumulator.apply(serde_json::from_str(stop).unwrap()).unwrap().is_none());
    }
}
```

Add `mod accumulate;` to `api.rs`, and make `sse`'s items reachable from it (`pub(crate)` or keep the modules `pub(super)` — the simplest is `mod sse;` with `pub` items, since `api` is private to the binary).

**Step 3: Run to verify they fail**

Run: `nix develop -c cargo test -p sloop-harness accumulate`
Expected: FAIL to compile — `Accumulator` does not exist.

**Step 4: Implement**

Above the tests in `accumulate.rs`:

```rust
use std::collections::BTreeMap;

use anyhow::{bail, Result};

use crate::api::sse::{BlockStart, Delta, Event};
use crate::tree::ContentBlock;

/// Assembles streamed events into completed content blocks.
///
/// A block is emitted at `content_block_stop` and never before, so what
/// reaches the tree is always whole. That is what lets the tree stay
/// append-only: there is no half-written node to go back and mutate.
#[derive(Debug, Default)]
pub struct Accumulator {
    open: BTreeMap<usize, Partial>,
    stop_reason: Option<String>,
}

#[derive(Debug)]
enum Partial {
    Text(String),
    Thinking { thinking: String, signature: String },
    /// A kind this slice does not model. Held open so its deltas have
    /// somewhere to go, and dropped when it closes.
    Unmodelled,
}

impl Accumulator {
    /// Apply one event, and take the block it completed, if it completed one.
    pub fn apply(&mut self, event: Event) -> Result<Option<ContentBlock>> {
        match event {
            Event::ContentBlockStart { index, content_block } => {
                self.open.insert(index, Partial::new(&content_block));
                Ok(None)
            }
            Event::ContentBlockDelta { index, delta } => {
                let Some(partial) = self.open.get_mut(&index) else {
                    bail!("a delta arrived for block {index}, which was never opened");
                };
                partial.extend(delta)?;
                Ok(None)
            }
            Event::ContentBlockStop { index } => {
                let Some(partial) = self.open.remove(&index) else {
                    bail!("block {index} closed without opening");
                };
                Ok(partial.finish())
            }
            Event::MessageDelta { delta } => {
                self.stop_reason = delta.stop_reason;
                Ok(None)
            }
            Event::Error { error } => bail!("{}: {}", error.kind, error.message),
            Event::MessageStart | Event::MessageStop | Event::Unknown => Ok(None),
        }
    }

    /// Why the turn ended, once `message_delta` has said.
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }
}

impl Partial {
    fn new(start: &BlockStart) -> Self {
        match start {
            BlockStart::Text => Self::Text(String::new()),
            BlockStart::Thinking => Self::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            BlockStart::Other => Self::Unmodelled,
        }
    }

    fn extend(&mut self, delta: Delta) -> Result<()> {
        match (self, delta) {
            (Self::Text(text), Delta::Text { text: more }) => text.push_str(&more),
            (Self::Thinking { thinking, .. }, Delta::Thinking { thinking: more }) => {
                thinking.push_str(&more);
            }
            (Self::Thinking { signature, .. }, Delta::Signature { signature: more }) => {
                signature.push_str(&more);
            }
            (Self::Unmodelled, _) | (_, Delta::Other) => {}
            (partial, delta) => bail!("a {delta:?} does not belong to a {partial:?}"),
        }
        Ok(())
    }

    fn finish(self) -> Option<ContentBlock> {
        match self {
            Self::Text(text) => Some(ContentBlock::text(text)),
            Self::Thinking { thinking, signature } => {
                Some(ContentBlock::thinking(thinking, signature))
            }
            Self::Unmodelled => None,
        }
    }
}
```

**Step 5: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness`
Expected: PASS, everything.

**Step 6: Commit**

```bash
git add crates/sloop-harness/src/api.rs crates/sloop-harness/src/api/accumulate.rs crates/sloop-harness/src/api/fixtures
git commit -m "Accumulate streamed events into content blocks"
```

---

## Task 6: The request body

**Files:**
- Modify: `crates/sloop-harness/src/api.rs`

**Step 1: Write the failing test**

In `api.rs`:

```rust
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "the value under test is built in the test")]
mod tests {
    use super::Request;
    use crate::tree::{ContentBlock, Role, Tree};

    #[test]
    fn the_request_body_has_the_shape_the_api_expects() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();

        assert_eq!(
            serde_json::to_value(Request::new(&messages)).unwrap(),
            serde_json::json!({
                "model": "claude-opus-5",
                "max_tokens": 64000,
                "stream": true,
                "thinking": { "type": "adaptive", "display": "summarized" },
                "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }]
            })
        );
    }

    #[test]
    fn the_request_names_no_budget_and_no_effort() {
        // budget_tokens is a 400 on this model, and effort defaults to the
        // value we would name. Both absences are deliberate.
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let body = serde_json::to_string(&Request::new(&messages)).unwrap();

        assert!(!body.contains("budget_tokens"));
        assert!(!body.contains("output_config"));
    }
}
```

The second test needs `Role` unused — drop it from the import if the compiler warns.

**Step 2: Run to verify it fails**

Run: `nix develop -c cargo test -p sloop-harness request`
Expected: FAIL to compile — `Request` does not exist.

**Step 3: Implement**

In `api.rs`, above the tests:

```rust
use serde::Serialize;

use crate::tree::Message;

/// The model this harness talks to. Thinking is on by default here, which is
/// why the tree has a thinking block at all.
const MODEL: &str = "claude-opus-5";

/// Large because the request streams. The ceiling that matters for a
/// non-streaming request is the HTTP timeout, and streaming removes it.
const MAX_TOKENS: u32 = 64_000;

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'static str,
    max_tokens: u32,
    stream: bool,
    thinking: Thinking,
    messages: &'a [Message],
}

/// `budget_tokens` is absent because it is a 400 on this model, and
/// `output_config.effort` because its default is the value we would set.
/// `display: "summarized"` is the one real choice here: the default returns
/// thinking blocks whose text is empty, which is nothing to print and nothing
/// for the memory index to ever label.
#[derive(Debug, Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    kind: &'static str,
    display: &'static str,
}

impl<'a> Request<'a> {
    fn new(messages: &'a [Message]) -> Self {
        Self {
            model: MODEL,
            max_tokens: MAX_TOKENS,
            stream: true,
            thinking: Thinking { kind: "adaptive", display: "summarized" },
            messages,
        }
    }
}
```

**Step 4: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/api.rs
git commit -m "Build the messages request body"
```

---

## Task 7: Credentials

**Files:**
- Modify: `crates/sloop-harness/src/api.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn a_missing_key_names_the_variable() {
    // Deliberately does not set the variable: this asserts the message a
    // first-time user actually sees.
    let message = Api::from_key(None).unwrap_err().to_string();

    assert!(message.contains("ANTHROPIC_API_KEY"));
}
```

Splitting `from_env` into a testable `from_key(Option<String>)` keeps the test off the process environment, which is global mutable state that other tests share.

**Step 2: Run to verify it fails**

Run: `nix develop -c cargo test -p sloop-harness missing_key`
Expected: FAIL to compile.

**Step 3: Implement**

```rust
use anyhow::{anyhow, Result};

const KEY_VAR: &str = "ANTHROPIC_API_KEY";

/// A client for `/v1/messages`.
pub struct Api {
    http: reqwest::Client,
    key: String,
}

impl Api {
    /// Read the credential from the environment.
    ///
    /// This is the only credential source. A missing key fails here rather
    /// than at the first request, following `SLOOP_MEMORY_MODEL`: an input
    /// the program cannot work without should fail loudly at startup.
    pub fn from_env() -> Result<Self> {
        Self::from_key(std::env::var(KEY_VAR).ok())
    }

    fn from_key(key: Option<String>) -> Result<Self> {
        let key = key.ok_or_else(|| {
            anyhow!(
                "{KEY_VAR} is not set.\n\n    export {KEY_VAR}=sk-ant-...\n\n\
                 sloop-harness talks to /v1/messages directly and has no other \
                 credential source."
            )
        })?;

        Ok(Self { http: reqwest::Client::new(), key })
    }
}
```

**Step 4: Run the tests**

Run: `nix develop -c cargo test -p sloop-harness`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/sloop-harness/src/api.rs
git commit -m "Read the API key from the environment"
```

---

## Task 8: Send

The only function that touches a socket, and the only one with no unit test. Everything it composes is already covered; what is left is the HTTP call itself, and the demo in Task 9 is what exercises it.

**Files:**
- Modify: `crates/sloop-harness/src/api.rs`

**Step 1: Implement**

```rust
use anyhow::{bail, Context};
use futures::StreamExt;

use crate::api::accumulate::Accumulator;
use crate::api::sse::{data_of, FrameDecoder};
use crate::tree::ContentBlock;

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// One completed assistant turn.
pub struct Turn {
    pub blocks: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
}

impl Api {
    /// Send a prompt and stream the turn back.
    ///
    /// `messages` is whatever `Tree::prompt_for` returned -- the two types
    /// meet without a conversion, which is the reason this function knows
    /// nothing about trees, tips or status labels.
    ///
    /// `on_block` sees each block as it completes. Nothing here reports
    /// partial blocks: the tree only ever accepts whole ones.
    pub async fn send(
        &self,
        messages: &[Message],
        mut on_block: impl FnMut(&ContentBlock) -> Result<()>,
    ) -> Result<Turn> {
        let response = self
            .http
            .post(MESSAGES_URL)
            .header("x-api-key", &self.key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&Request::new(messages))
            .send()
            .await
            .context("POST /v1/messages")?;

        let status = response.status();
        if !status.is_success() {
            // The body carries the API's own error message, which is more
            // useful than anything this side could say about a 400.
            let body = response.text().await.unwrap_or_default();
            bail!("{status} from /v1/messages: {body}");
        }

        let mut chunks = response.bytes_stream();
        let mut frames = FrameDecoder::default();
        let mut accumulator = Accumulator::default();
        let mut blocks = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.context("reading the response stream")?;

            for frame in frames.push(&chunk)? {
                let Some(data) = data_of(&frame) else { continue };
                let event = serde_json::from_str(&data)
                    .with_context(|| format!("decoding an SSE event: {data}"))?;

                if let Some(block) = accumulator.apply(event)? {
                    on_block(&block)?;
                    blocks.push(block);
                }
            }
        }

        Ok(Turn {
            stop_reason: accumulator.stop_reason().map(str::to_owned),
            blocks,
        })
    }
}
```

**Step 2: Check it compiles clean**

Run: `nix develop -c cargo clippy --all-targets --all-features -- -D warnings`
Expected: no output, exit 0. If `clippy::large_futures` fires, box the future at the call site in `main.rs` rather than restructuring `send`.

**Step 3: Commit**

```bash
git add crates/sloop-harness/src/api.rs
git commit -m "Stream a turn from /v1/messages"
```

---

## Task 9: The demo

Replaces the hand-built transcript in `main.rs` with a real one, forked.

**Files:**
- Modify: `crates/sloop-harness/src/main.rs` (rewrite `main`, `forked_transcript`, keep `render`)

**Step 1: Rewrite main**

```rust
use anyhow::{anyhow, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Both calls are deliberately engine-side -- where the table lives, and
    // what the indexer will accept. Nothing here touches `proto`: that is the
    // daemon's socket protocol, and reaching for it would be exactly the
    // coupling that linking the library directly is meant to avoid.
    let table = db_dir().join(TABLE_CHUNKS);
    line(&format!(
        "{} indexable={}",
        table.display(),
        is_indexable(Path::new("notes/branch-replay.md"))
    ))?;

    let prompt = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let api = Api::from_env()?;

    let mut tree = Tree::new(ContentBlock::text(&prompt));
    let root = tree.root();

    line("\nturn 1  streaming...")?;
    let first = turn(&api, &tree, root).await?;
    let kept = graft(&mut tree, root, first.blocks)?;

    // Fork as a *sibling* of the turn's second block, so the first block stays
    // a shared prefix. `prompt_for` drops the trailing assistant turn either
    // way, so both branches regenerate from the same user boundary -- the
    // difference is what the tree remembers, not what the request carries.
    let fork_at = kept.first().copied().filter(|_| kept.len() >= 2).unwrap_or(root);

    line("\nfork, abandon, regenerate\n\nturn 1' streaming...")?;
    let second = turn(&api, &tree, fork_at).await?;
    let abandoned = graft(&mut tree, fork_at, second.blocks)?;

    // One set_status per branch, at the divergence point. Everything below
    // inherits it, which is what makes a discarded branch identifiable all
    // the way down rather than only where it was rejected.
    let divergence = if fork_at == root { 0 } else { 1 };
    if let Some(node) = kept.get(divergence) {
        tree.set_status(*node, Status::Kept).ok_or_else(rejected_id)?;
    }
    if let Some(node) = abandoned.first() {
        tree.set_status(*node, Status::Abandoned).ok_or_else(rejected_id)?;
    }

    for stop in [first.stop_reason, second.stop_reason].into_iter().flatten() {
        if stop != "end_turn" {
            // max_tokens is not an error here: a truncated assistant turn is a
            // legal interior state for this tree, and forking is how it
            // resumes.
            line(&format!("\nnote: a turn stopped on {stop}"))?;
        }
    }

    render(&tree)
}

const DEFAULT_PROMPT: &str = "How should the cache expire?";

/// Send the branch ending at `tip`, printing each block as it lands.
async fn turn(api: &Api, tree: &Tree, tip: NodeId) -> Result<Turn> {
    let prompt = tree.prompt_for(tip).ok_or_else(rejected_id)?;
    api.send(&prompt, |block| line(&summarize(block))).await
}

/// Append a whole turn under `parent`, returning the node ids in order.
fn graft(tree: &mut Tree, parent: NodeId, blocks: Vec<ContentBlock>) -> Result<Vec<NodeId>> {
    let mut nodes = Vec::new();
    let mut tip = parent;

    for block in blocks {
        tip = tree.append(tip, Role::Assistant, block).ok_or_else(rejected_id)?;
        nodes.push(tip);
    }

    Ok(nodes)
}

fn summarize(block: &ContentBlock) -> String {
    let (kind, text) = match block {
        ContentBlock::Text { text } => ("text", text),
        ContentBlock::Thinking { thinking, .. } => ("thinking", thinking),
    };
    let first_line = text.lines().next().unwrap_or_default();

    format!("  [{kind}] {first_line}")
}

fn line(text: &str) -> Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{text}")?;
    Ok(())
}
```

`render` keeps its current body; change its signature to take only `&Tree` and use `line`, and change `rejected_id` to return `anyhow::Error` via `anyhow!`. Delete `forked_transcript` — it is replaced.

**Step 2: Compile and lint**

Run: `nix develop -c cargo clippy --all-targets --all-features -- -D warnings`
Expected: exit 0.

**Step 3: Verify the key check fires before the network**

Run: `nix develop -c env -u ANTHROPIC_API_KEY cargo run -p sloop-harness`
Expected: exits non-zero with the message naming `ANTHROPIC_API_KEY`, and no HTTP attempt.

**Step 4: Commit**

```bash
git add crates/sloop-harness/src/main.rs
git commit -m "Fork a real transcript in the harness demo"
```

---

## Task 10: End-to-end run

The only step in this plan that spends money and needs the network. It is also the only thing that proves Task 8.

**Step 1: Run it**

```bash
nix develop -c cargo run -p sloop-harness -- "How should the cache expire?"
```

**Step 2: Check each of these**

- Two turns stream, and blocks print as they complete.
- At least one `[thinking]` line appears with real text in it — if thinking blocks are empty, `display: "summarized"` did not take effect.
- The two branches render with different content.
- Branch labels read `Kept` and `Abandoned`, and the abandoned branch's leaf shows `Abandoned` as its *inherited* status while carrying none of its own.
- `prompt_for` reports 1 message for both branches, ending on `User`.

**Step 3: Record what the run produced**

Paste the output into the PR description. This is the evidence for the slice; there is no automated test that can produce it.

---

## Task 11: Documentation

**Files:**
- Modify: `crates/sloop-harness/README.md`
- Modify: `README.md` (the crate table and "The loop" section)
- Modify: `docs/architecture.md` ("What this does not cover")

**Step 1: Update the harness README**

Add a section covering `api` — the module split, why blocks land at `content_block_stop`, the request shape, and `ANTHROPIC_API_KEY`. Update the "Two constraints on the design" section: the no-SDK constraint stands, and thinking-on-by-default joins it. Update the "One constraint is known and not yet met" paragraph — it is still true and still about tools.

**Step 2: Update the workspace README**

The crate table says "Has the conversation tree; no client yet." That is now false. The "The loop" section says the harness "does not yet have the client that would send one" — also now false. Both become: it sends a branch and receives a labeled turn; what remains is tools, and indexing a transcript back into the index.

**Step 3: Update architecture.md**

"What this does not cover" lists the HTTP client and block kinds beyond text as unbuilt. Move the client and thinking to built; leave tool kinds, cache-hit instrumentation, and transcript indexing.

**Step 4: Commit**

```bash
git add README.md docs/architecture.md crates/sloop-harness/README.md
git commit -m "Document the messages client"
```

---

## Task 12: Full verification and PR

**Step 1: The whole suite, as CI runs it**

```bash
nix develop -c cargo fmt --check
nix develop -c cargo clippy --all-targets --all-features -- -D warnings
nix develop -c cargo test --workspace
nix develop -c cargo deny check
```

All four must be clean. `cargo deny` matters here specifically: `reqwest` moving from transitive to direct is exactly the kind of change that surfaces a license that was previously not evaluated.

**Step 2: The nix build, which is where the fixture-fileset mistake would show**

```bash
nix build .#sloop-memory --no-link
```

Expected: succeeds. If the harness's fixture tests fail here while passing under `cargo test`, Task 1 Step 2 was not applied.

**Step 3: Flake evaluation**

```bash
nix flake check --all-systems --no-build
```

**Step 4: Open the PR**

Describe what the code does now: a streaming client, a thinking block, a demo that forks a real transcript. Include the Task 10 output. Do not describe the alternatives that were considered — the design doc is in the repo for that.

---

## Notes for whoever executes this

**Do not add retries.** A 429 or 529 will happen during Task 10 and the temptation will be immediate. It is out of scope on purpose: backoff belongs to the slice that has run often enough to know what it is backing off from. Re-run the command.

**Do not add `tool_use` while you are in here.** The fork-validity rule that a `tool_use`/`tool_result` pair needs is real, unwritten, and larger than it looks.

**If a property test in Task 2 fails, read the counterexample first.** The properties are the specification for replay; a failure there means the thinking variant broke an invariant, not that the property is too strict.
