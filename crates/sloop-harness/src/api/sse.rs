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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "inputs are literals defined in the test"
)]
mod tests {
    use super::{data_of, FrameDecoder, MAX_BUFFERED};

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
}
