//! The `/v1/messages` client.
//!
//! Named `api` rather than `client` on purpose. `client` in this workspace is
//! the memory daemon's socket client, and this crate exists partly to show
//! what linking the engine directly looks like instead.
//!
//! Everything here except `Api::send` is a pure function over bytes, which is
//! what lets the whole decoder be tested in a sandbox with no network and no
//! API key -- the environment `nix build` runs the test suite in.

mod accumulate;
mod sse;

use serde::Serialize;

use crate::tree::Message;

/// The model this harness talks to. Thinking is on by default here, which is
/// why the tree has a thinking block at all.
const MODEL: &str = "claude-opus-5";

/// Large because the request streams. The ceiling that matters for a
/// non-streaming request is the HTTP timeout, and streaming removes it.
const MAX_TOKENS: u32 = 64_000;

// Dead until `Api::send` builds one. Guarded on not(test) for the reason
// `sse.rs` spells out: under cfg(test) the tests below construct and serialize
// it, so the label would be about a build where it does not hold.
#[cfg_attr(not(test), expect(dead_code, reason = "no caller until Api::send"))]
#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'static str,
    max_tokens: u32,
    stream: bool,
    thinking: Thinking,
    // Borrowed rather than owned because the tree already holds the only copy
    // and a request outlives nothing: it is serialized and dropped.
    messages: &'a [Message],
}

/// `budget_tokens` is absent because it is a 400 on this model, and
/// `output_config.effort` because its default is the value we would set.
/// `display: "summarized"` is the one real choice here: the default returns
/// thinking blocks whose text is empty, which is nothing to print and nothing
/// for the memory index to ever label.
#[cfg_attr(not(test), expect(dead_code, reason = "no caller until Api::send"))]
#[derive(Debug, Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    kind: &'static str,
    display: &'static str,
}

#[cfg_attr(not(test), expect(dead_code, reason = "no caller until Api::send"))]
impl<'a> Request<'a> {
    fn new(messages: &'a [Message]) -> Self {
        Self {
            model: MODEL,
            max_tokens: MAX_TOKENS,
            stream: true,
            thinking: Thinking {
                kind: "adaptive",
                display: "summarized",
            },
            messages,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "the value under test is built in the test"
)]
mod tests {
    use super::Request;
    use crate::tree::{ContentBlock, Tree};

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
