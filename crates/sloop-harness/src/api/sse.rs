//! Server-sent events, decoded without touching a socket.
//!
//! Line endings are assumed to be LF: the API sends them, and the spec's
//! `\r\n\r\n` separator is not matched here. A CRLF stream would decode to
//! nothing at all while the buffer grew without bound, so if that ever needs
//! supporting it needs supporting deliberately rather than by accident.

// Nothing outside the tests calls any of this until `Api::send` has a socket
// to read from. `pub` exempts nothing in a binary crate, so the whole module
// needs the label, and it goes once that caller exists. not(test) because
// under cfg(test) every item here is live and an expectation that held there
// would itself go unfulfilled.
#![cfg_attr(not(test), expect(dead_code, reason = "see above"))]

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

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
        let mut frames = Vec::new();
        while let Some(end) = separator(&self.buffer) {
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

const SEPARATOR: &[u8] = b"\n\n";

/// How much unterminated frame to tolerate before calling the stream broken.
///
/// A `content_block_delta` is orders of magnitude under a mebibyte, so no
/// healthy stream approaches this and nothing is lost by refusing to buffer
/// past it.
const MAX_BUFFERED: usize = 1 << 20;

fn separator(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(SEPARATOR.len())
        .position(|pair| pair == SEPARATOR)
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "inputs are literals defined in the test"
)]
mod tests {
    use super::{data_of, BlockStart, Delta, Event, FrameDecoder, MAX_BUFFERED};

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

    #[test]
    fn a_text_delta_decodes() {
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

    #[test]
    fn a_block_start_carries_its_kind_and_index() {
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}"#,
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

    #[test]
    fn a_message_stop_decodes() {
        let event: Event = serde_json::from_str(r#"{"type":"message_stop"}"#).unwrap();

        match event {
            Event::MessageStop => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    // The server may add events, and `ping` is one it already sends. A decoder
    // that rejects an unrecognized type turns every such addition into an
    // outage.
    #[test]
    fn an_unknown_event_type_decodes_rather_than_failing() {
        let event: Event =
            serde_json::from_str(r#"{"type":"some_future_event","payload":9}"#).unwrap();

        match event {
            Event::Unknown => {}
            other => panic!("wrong event: {other:?}"),
        }
    }

    #[test]
    fn a_ping_is_not_an_error() {
        let event: Event = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();

        match event {
            Event::Unknown => {}
            other => panic!("wrong event: {other:?}"),
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

    // The two halves of this module meet exactly here, and nowhere else does
    // a test show that what `data_of` hands back is what `Event` expects.
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
