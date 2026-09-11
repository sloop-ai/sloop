//! Events in, finished content blocks out.
//!
//! A block arrives as an opening, a run of deltas and a close, and nothing in
//! the stream carries the whole of it. This is the only place that state
//! lives, so `send` stays a loop that hands events over and forwards whatever
//! comes back.

// Only the tests drive this until `Api::send` has a socket to read from; the
// attribute and this note go with that commit. Guarded on not(test) for the
// same reason as `sse.rs`: under cfg(test) every item here is reached, so an
// unconditional expectation would be about a build where it does not hold.
#![cfg_attr(not(test), expect(dead_code, reason = "see above"))]

use std::collections::BTreeMap;

use anyhow::{bail, Result};

use crate::api::sse::{BlockStart, Delta, Event};
use crate::tree::ContentBlock;

/// The blocks of one response, assembled as their events arrive.
///
/// Keyed by index rather than pushed onto a `Vec` because the index is what
/// the wire uses to address a block, and a stream that opened block 1 before
/// block 0 closed would otherwise be silently misfiled. `BTreeMap` over
/// `HashMap` so a debug print of a half-built response reads in block order.
#[derive(Debug, Default)]
pub struct Accumulator {
    open: BTreeMap<usize, Partial>,
    stop_reason: Option<String>,
}

/// A block that has opened and not yet closed.
///
/// Distinct from [`ContentBlock`] because a block under construction is not a
/// block: a thinking block whose `signature_delta` has not landed yet would
/// be a [`ContentBlock::Thinking`] that is rejected on the next turn, and
/// there would be nothing in the type to say so.
#[derive(Debug)]
enum Partial {
    Text(String),
    Thinking {
        thinking: String,
        signature: String,
    },
    /// A kind this slice does not model. Held open so its deltas have
    /// somewhere to go, and dropped when it closes.
    Unmodelled,
}

impl Accumulator {
    /// Apply one event, and take the block it completed, if it completed one.
    ///
    /// Errors are the turn's errors, not this function's: an `error` event
    /// and a stream that contradicts itself both mean the response cannot be
    /// trusted, and a caller that kept reading past either would append a
    /// half-response to the tree as though it were whole.
    pub fn apply(&mut self, event: Event) -> Result<Option<ContentBlock>> {
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => {
                let partial = match content_block {
                    BlockStart::Text => Partial::Text(String::new()),
                    BlockStart::Thinking => Partial::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                    BlockStart::Other => Partial::Unmodelled,
                };
                self.open.insert(index, partial);
                Ok(None)
            }
            Event::ContentBlockDelta { index, delta } => {
                let Some(partial) = self.open.get_mut(&index) else {
                    bail!("a delta arrived for block {index}, which is not open");
                };
                partial.extend(index, delta)?;
                Ok(None)
            }
            Event::ContentBlockStop { index } => {
                let Some(partial) = self.open.remove(&index) else {
                    bail!("block {index} closed without ever opening");
                };
                Ok(partial.finish())
            }
            // Overwritten rather than merged: the API sends one `message_delta`
            // per message and it is where the stop reason lives, so if a second
            // ever arrives the later one is the more recent answer.
            Event::MessageDelta { delta } => {
                self.stop_reason = delta.stop_reason;
                Ok(None)
            }
            Event::Error { error } => bail!("{}: {}", error.kind, error.message),
            // The envelope events carry nothing this slice reads. `Unknown` is
            // the one that deserves a word: it is dropped in silence, and that
            // is a decision rather than an oversight. `#[serde(other)]` can
            // only target a unit variant, so by the time an event arrives here
            // its `type` string is already gone and there is nothing left for
            // a log line to name -- a `debug!` here could only say "something
            // unrecognized happened", which would not shorten anyone's hunt.
            // Recovering the tag means giving `Event` a fallback variant that
            // captures it, and that is the change to make the day a
            // server-side addition turns out to have been swallowed here.
            Event::MessageStart | Event::MessageStop | Event::Unknown => Ok(None),
        }
    }

    /// Why the model stopped, once `message_delta` has said.
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }
}

impl Partial {
    fn extend(&mut self, index: usize, delta: Delta) -> Result<()> {
        match (self, delta) {
            (Self::Text(text), Delta::Text { text: chunk }) => text.push_str(&chunk),
            (Self::Thinking { thinking, .. }, Delta::Thinking { thinking: chunk }) => {
                thinking.push_str(&chunk);
            }
            // Appended, not assigned. The signature arrives whole today, but a
            // value that is only ever correct when it fits in one delta is a
            // truncation waiting to happen, and a truncated signature is a 400
            // on the next turn rather than anything visible here.
            (Self::Thinking { signature, .. }, Delta::Signature { signature: chunk }) => {
                signature.push_str(&chunk);
            }
            // One arm because there is one behaviour -- the delta contributes
            // nothing -- but two unrelated reasons to reach it, and both are
            // load-bearing. A `Delta::Other` in a block this slice does model
            // is the API gaining a feature, a `citations_delta` in a text
            // block; failing the turn over it would turn that into an outage,
            // which is what the variant exists to prevent. Any delta in a
            // block this slice does not model describes content that
            // `content_block_stop` throws away regardless.
            (Self::Text(_) | Self::Thinking { .. }, Delta::Other)
            | (
                Self::Unmodelled,
                Delta::Text { .. }
                | Delta::Thinking { .. }
                | Delta::Signature { .. }
                | Delta::Other,
            ) => {}
            // The remaining pairs are the stream contradicting its own
            // opening, which no amount of forward compatibility explains.
            (Self::Text(_), Delta::Thinking { .. }) => {
                bail!("a thinking_delta arrived for block {index}, which opened as text");
            }
            (Self::Text(_), Delta::Signature { .. }) => {
                bail!("a signature_delta arrived for block {index}, which opened as text");
            }
            (Self::Thinking { .. }, Delta::Text { .. }) => {
                bail!("a text_delta arrived for block {index}, which opened as thinking");
            }
        }

        Ok(())
    }

    fn finish(self) -> Option<ContentBlock> {
        match self {
            Self::Text(text) => Some(ContentBlock::Text { text }),
            Self::Thinking {
                thinking,
                signature,
            } => Some(ContentBlock::Thinking {
                thinking,
                signature,
            }),
            Self::Unmodelled => None,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use super::Accumulator;
    use crate::api::sse::{data_of, Event, FrameDecoder};
    use crate::tree::{ContentBlock, Message, Role};
    use anyhow::Result;

    const TEXT_ONLY: &str = include_str!("fixtures/text-only.sse");
    const THINKING_THEN_TEXT: &str = include_str!("fixtures/thinking-then-text.sse");
    const OVERLOADED_MID_STREAM: &str = include_str!("fixtures/overloaded-mid-stream.sse");

    /// What a replayed stream produced: the blocks, and the accumulator that
    /// built them so its stop reason can be read.
    #[derive(Debug)]
    struct Replayed {
        blocks: Vec<ContentBlock>,
        accumulator: Accumulator,
    }

    /// Drive a recorded stream through the whole path `Api::send` will.
    ///
    /// One byte per `decode` call, which is the harshest chunking a stream can
    /// produce and the one a decoder assuming frame-aligned chunks fails on.
    /// No test here currently distinguishes it from feeding the whole stream
    /// at once, and that is the point: `sse.rs` proves chunk boundaries are
    /// unobservable, so this is a standing check on that guarantee rather than
    /// coverage of a gap. If it ever starts failing, `FrameDecoder` broke.
    fn replay(stream: &str) -> Result<Replayed> {
        let mut decoder = FrameDecoder::default();
        let mut accumulator = Accumulator::default();
        let mut blocks = Vec::new();

        for byte in stream.as_bytes() {
            for frame in decoder.decode(&[*byte])? {
                // `None` here means two things at once: a keep-alive comment
                // frame, which is routine, and a frame with no `data:` line at
                // all, which is not. Skipping both is a decision, not a
                // fallthrough. The keep-alive is the case that actually
                // occurs, and the anomalous frame carries nothing to apply
                // either way, so distinguishing them would buy a diagnostic
                // and no behaviour. Reaching past `data_of` to the raw frame
                // to tell them apart is deliberately not done: a frame does
                // not record which it meant to be, so the guess would be one.
                let Some(payload) = data_of(&frame) else {
                    continue;
                };
                let event: Event = serde_json::from_str(&payload)?;
                if let Some(block) = accumulator.apply(event)? {
                    blocks.push(block);
                }
            }
        }

        Ok(Replayed {
            blocks,
            accumulator,
        })
    }

    #[test]
    fn text_deltas_concatenate_into_one_block() {
        let replayed = replay(TEXT_ONLY).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![ContentBlock::Text {
                text: "The capital of France is Paris.".to_owned()
            }]
        );
        assert_eq!(replayed.accumulator.stop_reason(), Some("end_turn"));
    }

    // The whole blocks, not their prose. A comparison that reached past the
    // signature would pass against an accumulator that never stored one, and
    // the failure it missed is invisible until the next request returns 400.
    #[test]
    fn a_thinking_block_keeps_its_signature() {
        let replayed = replay(THINKING_THEN_TEXT).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![
                ContentBlock::Thinking {
                    thinking: "The question is arithmetic, so 27 times 4 is 108.".to_owned(),
                    signature: "ErUBCkYIBRgCIkA=".to_owned(),
                },
                ContentBlock::Text {
                    text: "108".to_owned()
                },
            ]
        );
    }

    // The guarantee stated where it is actually needed: what matters is not
    // that the signature is in the struct but that it leaves again unchanged,
    // and the JSON below is the request body the next turn sends.
    #[test]
    fn a_recorded_thinking_block_goes_back_out_unchanged() {
        let replayed = replay(THINKING_THEN_TEXT).unwrap();

        let message = Message {
            role: Role::Assistant,
            content: replayed.blocks,
        };

        assert_eq!(
            serde_json::to_string(&message).unwrap(),
            r#"{"role":"assistant","content":[{"type":"thinking","thinking":"The question is arithmetic, so 27 times 4 is 108.","signature":"ErUBCkYIBRgCIkA="},{"type":"text","text":"108"}]}"#
        );
    }

    #[test]
    fn an_error_event_fails_the_turn() {
        let error = replay(OVERLOADED_MID_STREAM).unwrap_err();

        assert_eq!(error.to_string(), "overloaded_error: Overloaded");
    }

    // The other half of forward compatibility, and the half a `tool_use`
    // fixture does not reach: a delta kind this slice does not model arriving
    // in a block it does -- tomorrow's `citations_delta` inside today's text.
    // Refusing it would turn a server-side addition into an outage on every
    // request. Both modelled kinds are here because the arm covers both, and
    // a split that refused it in only one of them would otherwise survive.
    #[test]
    fn an_unmodelled_delta_in_a_modelled_block_is_ignored() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"weighing it\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"some_future_delta\",\"payload\":9}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"ErUBCkYIBRgCIkA=\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"cited\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"citations_delta\",\"citation\":{\"type\":\"page_location\"}}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![
                ContentBlock::Thinking {
                    thinking: "weighing it".to_owned(),
                    signature: "ErUBCkYIBRgCIkA=".to_owned(),
                },
                ContentBlock::Text {
                    text: "cited".to_owned()
                },
            ]
        );
    }

    // Constructed rather than recorded, because the API does not interleave:
    // it closes one block before opening the next. That is exactly why the
    // case is worth pinning -- an accumulator that tracked "the current block"
    // instead of the index passes every recording in this directory, and the
    // index is what the wire says addresses a block.
    #[test]
    fn interleaved_indices_land_in_the_blocks_they_name() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"one\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"zero\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"-one\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );

        let replayed = replay(stream).unwrap();

        // Emitted in the order the blocks closed, which is the order the
        // caller will append them in.
        assert_eq!(
            replayed.blocks,
            vec![
                ContentBlock::Text {
                    text: "one-one".to_owned()
                },
                ContentBlock::Text {
                    text: "zero".to_owned()
                },
            ]
        );
    }

    #[test]
    fn a_block_kind_this_slice_does_not_model_is_dropped() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"grep\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"after\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        );

        let replayed = replay(stream).unwrap();

        // The unmodelled block yields nothing at all -- not an empty text
        // block, which would reach `messages[]` on the next turn as a block
        // the API never sent.
        assert_eq!(
            replayed.blocks,
            vec![ContentBlock::Text {
                text: "after".to_owned()
            }]
        );
    }

    #[test]
    fn a_delta_for_a_block_that_never_opened_is_refused() {
        let stream = "data: {\"type\":\"content_block_delta\",\"index\":4,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n";

        let error = replay(stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "a delta arrived for block 4, which is not open"
        );
    }

    #[test]
    fn a_stop_for_a_block_that_never_opened_is_refused() {
        let stream = "data: {\"type\":\"content_block_stop\",\"index\":2}\n\n";

        let error = replay(stream).unwrap_err();

        assert_eq!(error.to_string(), "block 2 closed without ever opening");
    }

    // A block closes once. A second stop for the same index is the stream
    // contradicting itself, and it has to be caught by the same check rather
    // than quietly emitting the block twice.
    #[test]
    fn a_block_cannot_close_twice() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );

        let error = replay(stream).unwrap_err();

        assert_eq!(error.to_string(), "block 0 closed without ever opening");
    }

    #[test]
    fn a_delta_that_contradicts_its_block_is_refused() {
        for (delta, expected) in [
            (
                r#"{"type":"thinking_delta","thinking":"no"}"#,
                "a thinking_delta arrived for block 0, which opened as text",
            ),
            (
                r#"{"type":"signature_delta","signature":"no"}"#,
                "a signature_delta arrived for block 0, which opened as text",
            ),
        ] {
            let stream = [
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":",
                delta,
                "}\n\n",
            ]
            .concat();

            let error = replay(&stream).unwrap_err();

            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn a_text_delta_in_a_thinking_block_is_refused() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"no\"}}\n\n",
        );

        let error = replay(stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "a text_delta arrived for block 0, which opened as thinking"
        );
    }

    // Two `signature_delta`s for one block. The API sends one, so this pins
    // the append in `extend` rather than the wire: assignment passes every
    // recording and loses everything but the last fragment the day the value
    // outgrows a single delta.
    #[test]
    fn a_signature_split_across_deltas_is_rejoined() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"ErUBCkYI\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"BRgCIkA=\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![ContentBlock::Thinking {
                thinking: String::new(),
                signature: "ErUBCkYIBRgCIkA=".to_owned(),
            }]
        );
    }

    // A stream that ends mid-block. The blocks that did close are still
    // returned, and the one that did not is simply absent -- the caller's
    // check for that is the stop reason, which never arrives.
    #[test]
    fn a_block_left_open_yields_nothing_and_no_stop_reason() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"cut off\"}}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert!(replayed.blocks.is_empty());
        assert_eq!(replayed.accumulator.stop_reason(), None);
    }

    // `ping` decodes to `Event::Unknown`, and every fixture here carries one
    // between real events. This is the assertion that it passes through
    // without disturbing the block it sits inside.
    #[test]
    fn an_unknown_event_between_deltas_changes_nothing() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"one\"}}\n\n",
            "data: {\"type\":\"ping\"}\n\n",
            "data: {\"type\":\"some_future_event\",\"payload\":9}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" two\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![ContentBlock::Text {
                text: "one two".to_owned()
            }]
        );
    }

    // A keep-alive comment frame, which is the case the replay loop skips
    // `None` payloads for. It has to survive the middle of a block.
    #[test]
    fn a_keep_alive_comment_frame_is_skipped() {
        let stream = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            ": keep-alive\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"still here\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert_eq!(
            replayed.blocks,
            vec![ContentBlock::Text {
                text: "still here".to_owned()
            }]
        );
    }

    #[test]
    fn a_stop_reason_that_is_not_end_turn_is_recorded_as_it_arrived() {
        let stream = concat!(
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4096}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let replayed = replay(stream).unwrap();

        assert_eq!(replayed.accumulator.stop_reason(), Some("max_tokens"));
    }
}
