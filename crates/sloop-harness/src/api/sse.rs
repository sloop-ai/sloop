//! Server-sent events, decoded without touching a socket.
//!
//! Line endings are assumed to be LF: the API sends them, and the spec's
//! `\r\n\r\n` separator is not matched here. A CRLF stream would decode to
//! nothing at all while the buffer grew without bound, so if that ever needs
//! supporting it needs supporting deliberately rather than by accident. The
//! assumption reaches one level further in as well: `data_of` splits on
//! `str::lines`, which counts `\r\n` as one ending and strips the `\r`, so a
//! `data:` value ending in a carriage return does not come back out intact.

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

const SEPARATOR: &[u8] = b"\n\n";

/// How much unterminated frame to tolerate before calling the stream broken.
///
/// A `content_block_delta` is orders of magnitude under a mebibyte, so no
/// healthy stream approaches this and nothing is lost by refusing to buffer
/// past it.
const MAX_BUFFERED: usize = 1 << 20;

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
    pub fn decode(&mut self, chunk: &[u8]) -> Result<Vec<String>> {
        self.buffer.extend_from_slice(chunk);

        // Splitting raw bytes before validating them is safe because every
        // byte of a multi-byte UTF-8 sequence has its high bit set. A `\n` is
        // 0x0A, so it cannot occur inside a character, and a split on it
        // cannot land mid-character and manufacture invalid input.
        //
        // `separator_at` rescans from zero each time it is called, so the cost
        // is quadratic in chunks per frame -- not in frame size, which is the
        // reading that makes it look alarming. The `drain` is what bounds it:
        // the buffer never holds more than one incomplete frame, and
        // `bytes_stream()` hands over TLS-record-sized chunks, so even a 64
        // KiB frame arrives in a handful of calls and is scanned a handful of
        // times. Small chunks are the tripwire here, not large frames.
        let mut frames = Vec::new();
        while let Some(end) = separator_at(&self.buffer) {
            let frame: Vec<u8> = self.buffer.drain(..end + SEPARATOR.len()).collect();
            frames.push(String::from_utf8(frame).context("an SSE frame was not UTF-8")?);
        }

        // Whatever survives the loop is a frame still waiting for its
        // separator, so a stream that never sends one -- a CRLF server, a
        // truncated body, a response that was never SSE -- accumulates here
        // forever. Bounding it turns an OOM with nothing to say into an
        // error that names the problem.
        ensure!(
            self.buffer.len() <= MAX_BUFFERED,
            "no SSE frame separator in {} buffered bytes",
            self.buffer.len()
        );

        Ok(frames)
    }
}

fn separator_at(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(SEPARATOR.len())
        .position(|pair| pair == SEPARATOR)
}

/// The data of a frame: its `data:` lines, joined.
///
/// The `event:` line is deliberately ignored. The data carries its own `type`,
/// so reading both would give one fact two sources of truth.
pub fn data_of(frame: &str) -> Option<String> {
    let mut data: Option<String> = None;

    for line in frame.lines() {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.strip_prefix(' ').unwrap_or(rest);

        match &mut data {
            Some(joined) => {
                joined.push('\n');
                joined.push_str(rest);
            }
            None => data = Some(rest.to_owned()),
        }
    }

    data
}

/// One decoded SSE event.
///
/// [`Event::Unknown`] is the important variant: it is what keeps a server-side
/// addition from becoming a client-side failure.
///
/// `MessageStart` takes no fields because nothing in this slice reads the
/// opening message envelope, and an internally tagged unit variant consumes
/// and discards whatever keys sit beside the tag.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    MessageStart,
    ContentBlockStart {
        // Blocks are identified by position, and `usize` is what the
        // accumulator will index a `Vec` with. A newtype would be unwrapped at
        // its only use site, and there is no second index-shaped quantity here
        // for it to be confused with.
        index: usize,
        content_block: BlockStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: Delta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
    },
    MessageStop,
    Error {
        error: ApiError,
    },
    #[serde(other)]
    Unknown,
}

/// The opening shape of a block, which is what says how to accumulate it.
///
/// Only the kind survives, and whatever content the opening carried is
/// dropped. That is safe because the API opens every block empty and sends
/// the contents as deltas, so there is nothing there to lose -- an invariant
/// of the wire rather than of this type, which is why it is written down.
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

// No `rename_all` beside the tag, unlike the two enums above: a delta's wire
// name is its kind plus `_delta`, which no case convention produces, so every
// variant has to name itself and there is nothing left to rename.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
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

/// The tail of a message, which is where the stop reason arrives.
///
/// `stop_reason` stays a `String` beside four hand-rolled tagged enums, and
/// `end_turn`/`max_tokens`/`refusal` is exactly the closed set this file
/// models as an enum everywhere else. It is a string because its only reader
/// compares it against a single literal; the moment a second reader branches
/// on it, it should become an enum first. [`ApiError::kind`] is the same
/// choice for the same reason.
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use super::{data_of, BlockStart, Delta, Event, FrameDecoder, MAX_BUFFERED};
    use proptest::prelude::*;

    #[test]
    fn a_frame_split_across_chunks_is_reassembled() {
        let mut decoder = FrameDecoder::default();

        assert!(decoder
            .decode(b"event: ping\ndata: {\"type\"")
            .unwrap()
            .is_empty());
        let frames = decoder.decode(b":\"ping\"}\n\n").unwrap();

        assert_eq!(frames.len(), 1);
        assert_eq!(data_of(&frames[0]).as_deref(), Some(r#"{"type":"ping"}"#));
    }

    #[test]
    fn a_frame_carries_its_terminating_blank_line() {
        let mut decoder = FrameDecoder::default();
        let frames = decoder.decode(b"data: one\n\ndata: two\n\n").unwrap();
        assert_eq!(frames, vec!["data: one\n\n", "data: two\n\n"]);
    }

    #[test]
    fn several_frames_in_one_chunk_come_back_in_order() {
        let mut decoder = FrameDecoder::default();
        let frames = decoder
            .decode(b"data: one\n\ndata: two\n\ndata: thr")
            .unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(data_of(&frames[0]).as_deref(), Some("one"));
        assert_eq!(data_of(&frames[1]).as_deref(), Some("two"));
    }

    // The property below subsumes this -- its chunk sizes start at one -- but
    // a named case that fails on its own is a sharper signal than a shrunk
    // counterexample, and byte-at-a-time is the shape most worth naming.
    #[test]
    fn a_stream_delivered_one_byte_at_a_time_still_frames() {
        let mut decoder = FrameDecoder::default();

        let mut frames = Vec::new();
        for byte in b"data: one\n\ndata: two\n\ndata: three\n\n" {
            frames.extend(decoder.decode(&[*byte]).unwrap());
        }

        assert_eq!(
            frames,
            vec!["data: one\n\n", "data: two\n\n", "data: three\n\n"]
        );
    }

    #[test]
    fn a_frame_that_is_not_utf8_is_rejected() {
        let mut decoder = FrameDecoder::default();

        // 0x80 is a continuation byte with nothing in front of it to
        // continue, so no prefix of this frame is a character.
        let error = decoder.decode(b"data: \x80\n\n").unwrap_err();

        assert_eq!(error.to_string(), "an SSE frame was not UTF-8");
    }

    #[test]
    fn a_stream_that_never_yields_a_frame_is_capped() {
        let mut decoder = FrameDecoder::default();

        // Sitting exactly on the cap is still a frame that might yet be
        // terminated by the next chunk, so only the byte past it is a fault.
        let chunk = vec![b'x'; MAX_BUFFERED];
        assert!(decoder.decode(&chunk).unwrap().is_empty());

        let error = decoder.decode(b"x").unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "no SSE frame separator in {} buffered bytes",
                MAX_BUFFERED + 1
            )
        );
    }

    // A keep-alive comment, which is what a frame carrying no payload
    // actually looks like on the wire -- the API's `ping` is not one, it
    // sends `data: {"type": "ping"}` like everything else.
    #[test]
    fn a_frame_with_no_data_line_has_no_payload() {
        assert_eq!(data_of(": keep-alive\n\n"), None);
    }

    #[test]
    fn multiple_data_lines_join_with_a_newline() {
        assert_eq!(
            data_of("data: one\ndata: two\n\n").as_deref(),
            Some("one\ntwo")
        );
    }

    // This guards a simplification rather than an input. The `match &mut
    // payload` in `data_of` reads like an obvious candidate for
    // `get_or_insert_with(String::new)`, but that rewrite cannot tell "no
    // data line yet" from "a data line that was empty", and swallows the
    // leading newline. Deleting this test licenses that change.
    #[test]
    fn an_empty_data_line_is_still_a_line() {
        assert_eq!(data_of("data:\ndata: x\n\n").as_deref(), Some("\nx"));
    }

    /// One line of a generated frame, in the shapes a real stream mixes.
    /// Every rendering is non-empty, so no frame can contain the separator
    /// and the frame count is always exactly the payload count.
    #[derive(Debug, Clone)]
    enum Line {
        Comment(String),
        Event(String),
        Data(String),
    }

    impl Line {
        fn data(&self) -> Option<&str> {
            match self {
                Self::Data(payload) => Some(payload),
                Self::Comment(_) | Self::Event(_) => None,
            }
        }

        fn render(&self) -> String {
            match self {
                Self::Comment(text) => format!(": {text}"),
                Self::Event(name) => format!("event: {name}"),
                Self::Data(payload) => format!("data: {payload}"),
            }
        }
    }

    // `\r` is excluded deliberately, not for tidiness: `str::lines` treats
    // `\r\n` as one ending and eats the `\r`, so a payload ending in one
    // cannot round-trip. That is a second consequence of assuming LF.
    fn line() -> impl Strategy<Value = Line> {
        prop_oneof![
            "[^\r\n]{0,20}".prop_map(Line::Comment),
            "[a-z_]{1,16}".prop_map(Line::Event),
            "[^\r\n]{0,40}".prop_map(Line::Data),
        ]
    }

    /// How the stream reaches the decoder. The degenerate splits are named
    /// rather than left to chance: byte-at-a-time is the harshest case a
    /// real stream can produce, and a single chunk is the case where the
    /// buffer never has to bridge anything. Random sizes alone would sample
    /// both too rarely to count as covered.
    #[derive(Debug, Clone)]
    enum Split {
        EveryByte,
        Whole,
        Sizes(Vec<usize>),
    }

    impl Split {
        fn chunks<'a>(&self, stream: &'a [u8]) -> Vec<&'a [u8]> {
            match self {
                Self::EveryByte => stream.chunks(1).collect(),
                Self::Whole => vec![stream],
                Self::Sizes(sizes) => {
                    let mut chunks = Vec::new();
                    let mut rest = stream;
                    let mut sizes = sizes.iter().cycle();
                    while !rest.is_empty() {
                        let take = (*sizes.next().unwrap()).min(rest.len());
                        let (chunk, tail) = rest.split_at(take);
                        chunks.push(chunk);
                        rest = tail;
                    }
                    chunks
                }
            }
        }
    }

    fn split() -> impl Strategy<Value = Split> {
        prop_oneof![
            1 => Just(Split::EveryByte),
            1 => Just(Split::Whole),
            4 => prop::collection::vec(1usize..48, 1..16).prop_map(Split::Sizes),
        ]
    }

    proptest! {
        /// Chunk boundaries are the one thing this module exists to hide, so
        /// the property is that they cannot be observed: however the same
        /// stream is sliced, the frames that come back are the same. This is
        /// also what covers a multi-byte character split across two chunks,
        /// which no hand-written case is likely to place deliberately.
        #[test]
        fn arbitrary_chunking_yields_the_same_frames(
            frames in prop::collection::vec(prop::collection::vec(line(), 1..5), 1..8),
            split in split(),
        ) {
            let rendered: Vec<String> = frames
                .iter()
                .map(|lines| {
                    let mut frame = String::new();
                    for line in lines {
                        frame.push_str(&line.render());
                        frame.push('\n');
                    }
                    frame.push('\n');
                    frame
                })
                .collect();
            let stream = rendered.concat().into_bytes();

            let mut decoder = FrameDecoder::default();
            let mut decoded = Vec::new();
            for chunk in split.chunks(&stream) {
                decoded.extend(decoder.decode(chunk).unwrap());
            }

            // The frames themselves, not a projection of them: comparing only
            // payloads would let a decoder that mislays a separator byte pass,
            // because `lines` skips the blank line that mistake leaves behind.
            prop_assert_eq!(&decoded, &rendered);

            // Built from the generated data, never by parsing: the test states
            // what the payload should be, rather than restating how `data_of`
            // computes it.
            for (frame, lines) in decoded.iter().zip(&frames) {
                let texts: Vec<&str> = lines.iter().filter_map(Line::data).collect();
                let expected = if texts.is_empty() {
                    None
                } else {
                    Some(texts.join("\n"))
                };
                let payload = data_of(frame);
                prop_assert_eq!(payload, expected);
            }
        }
    }

    #[test]
    fn a_text_delta_carries_its_index_and_text() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Two"}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockDelta {
                index,
                delta: Delta::Text { text },
            } => {
                assert_eq!(index, 0);
                assert_eq!(text, "Two");
            }
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_thinking_delta_is_not_a_text_delta() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Weighing it"}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockDelta {
                delta: Delta::Thinking { thinking },
                ..
            } => assert_eq!(thinking, "Weighing it"),
            other => panic!("wrong event: {other:?}"),
        }
    }

    // Destructured rather than asserted with `matches!` because the variant is
    // only half the claim: a decoder that read the signature out of the wrong
    // key would produce the right variant carrying an empty string, and this
    // has to fail when it does.
    #[test]
    fn a_signature_delta_is_its_own_kind() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"ErUB"}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockDelta {
                index,
                delta: Delta::Signature { signature },
            } => {
                assert_eq!(index, 1);
                assert_eq!(signature, "ErUB");
            }
            other => panic!("wrong event: {other:?}"),
        }
    }

    // `input_json_delta` is a real delta this slice does not model. It has to
    // arrive as something the accumulator can drop, not as an error.
    #[test]
    fn a_delta_kind_this_slice_ignores_still_decodes() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockDelta {
                delta: Delta::Other,
                ..
            } => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    // The `text` is non-empty on purpose. The API never sends it that way, so
    // this pins the discard rather than the wire: if prefilled content ever
    // had to survive, this is the test that fails and says where to look.
    #[test]
    fn a_block_start_carries_its_kind_and_index() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_start","index":3,"content_block":{"type":"text","text":"PREFILL"}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockStart {
                index,
                content_block: BlockStart::Text,
            } => assert_eq!(index, 3),
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_thinking_block_start_is_its_own_kind() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockStart {
                content_block: BlockStart::Thinking,
                ..
            } => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_block_kind_this_slice_ignores_still_decodes() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"grep","input":{}}}"#,
        )
        .unwrap();

        match event {
            Event::ContentBlockStart {
                content_block: BlockStart::Other,
                ..
            } => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_block_stop_carries_the_index_it_closes() {
        let event: Event =
            serde_json::from_str(r#"{"type":"content_block_stop","index":2}"#).unwrap();

        match event {
            Event::ContentBlockStop { index } => assert_eq!(index, 2),
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_message_delta_carries_the_stop_reason() {
        let event: Event = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}"#,
        )
        .unwrap();

        match event {
            Event::MessageDelta { delta } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            }
            other => panic!("wrong event: {other:?}"),
        }
    }

    // The envelope is large and none of it is read here. An internally tagged
    // unit variant is what makes ignoring it the decoder's behaviour rather
    // than a lucky property of a small payload, so the payload is a real one.
    #[test]
    fn a_message_start_ignores_the_envelope_it_carries() {
        let event: Event = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"usage":{"input_tokens":9,"output_tokens":1}}}"#,
        )
        .unwrap();

        match event {
            Event::MessageStart => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    // The terminal event, and `#[serde(other)]` sits right beside it: a
    // `MessageStop` misspelled in the enum would decode as `Unknown` instead
    // of failing, and a reader waiting for the end would wait forever.
    #[test]
    fn a_message_stop_is_not_swallowed_as_unknown() {
        let event: Event = serde_json::from_str(r#"{"type":"message_stop"}"#).unwrap();

        match event {
            Event::MessageStop => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    // Both payloads reach `#[serde(other)]` by the same route, so they are one
    // test. They are both here because the reasons differ: `ping` is a type
    // the server already sends on every stream, and `some_future_event` stands
    // for one it has not invented yet. Rejecting either turns a keep-alive, or
    // a server-side addition, into an outage.
    #[test]
    fn an_unrecognized_event_type_decodes_rather_than_failing() {
        for payload in [
            r#"{"type":"ping"}"#,
            r#"{"type":"some_future_event","payload":9}"#,
        ] {
            let event: Event = serde_json::from_str(payload).unwrap();

            match event {
                Event::Unknown => {}
                other => panic!("{payload} decoded as {other:?}"),
            }
        }
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

    // Documentation of the seam rather than coverage of it: this is where a
    // reader sees that what `data_of` hands back is what `Event` parses. The
    // coverage is already elsewhere, because any mutation here has to change
    // the payload string -- which
    // `a_frame_split_across_chunks_is_reassembled` pins byte for byte, and
    // `a_message_stop_is_not_swallowed_as_unknown` decodes.
    #[test]
    fn a_decoded_frames_payload_is_an_event() {
        let mut decoder = FrameDecoder::default();
        let frames = decoder
            .decode(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .unwrap();

        let payload = data_of(&frames[0]).unwrap();
        let event: Event = serde_json::from_str(&payload).unwrap();

        match event {
            Event::MessageStop => {}
            other => panic!("wrong event: {other:?}"),
        }
    }
}
