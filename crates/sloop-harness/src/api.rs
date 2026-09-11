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

use anyhow::{anyhow, Context, Result};
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

const KEY_VAR: &str = "ANTHROPIC_API_KEY";

/// A client for `/v1/messages`.
///
/// No `Debug`, unlike everything else in this module. `key` is a credential,
/// and a derived `Debug` would put it into any error or trace that formats the
/// client -- a leak that is invisible at the site that causes it, because the
/// site only asks to print a struct.
// Unconditional rather than guarded on not(test) like `Request`, because it is
// about two different things in the two builds and holds in both: the type is
// never constructed without `Api::send`, and under cfg(test) the tests
// construct it but nothing reads `http` until there is a request to send.
#[expect(dead_code, reason = "no caller until Api::send")]
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
    // Dead in both builds, not just under not(test): the test below calls
    // `from_key` precisely so that no test reads the process environment.
    #[expect(dead_code, reason = "no caller until Api::send")]
    pub fn from_env() -> Result<Self> {
        Self::from_key(std::env::var(KEY_VAR).ok())
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "no caller until Api::send"))]
    fn from_key(key: Option<String>) -> Result<Self> {
        let key = key.ok_or_else(|| {
            anyhow!(
                "{KEY_VAR} is not set.\n\n    export {KEY_VAR}=sk-ant-...\n\n\
                 sloop-harness talks to /v1/messages directly and has no other \
                 credential source."
            )
        })?;

        // `Client::new` is the same call with an `expect` around it, and the
        // failure it hides -- no usable TLS backend -- is exactly the kind
        // this constructor already reports in words. Taking the `Result` keeps
        // the panic out of a binary whose lints deny writing one by hand.
        let http = reqwest::Client::builder()
            .build()
            .context("could not build an HTTPS client")?;

        Ok(Self { http, key })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use super::{Api, Request};
    use crate::tree::{ContentBlock, Tree};

    // Both of these deliberately avoid the process environment: that is global
    // mutable state the whole test binary shares.
    //
    // Destructured rather than `unwrap_err`, which would need `Api: Debug` --
    // and the reason that derive is absent is that it would print the key.
    #[test]
    fn a_missing_key_says_which_variable_to_set() {
        let Err(error) = Api::from_key(None) else {
            panic!("a missing key was accepted");
        };
        let message = error.to_string();

        // Two claims rather than one, because the message has two jobs and
        // only the first survives a careless reword. Naming the variable is
        // what makes the failure diagnosable; the command is what makes it
        // fixable without a trip to the source. Substrings rather than the
        // whole message on purpose -- the prose around them is meant to be
        // rewritten, and a test that pinned it would only ever be repasted.
        assert!(message.contains("ANTHROPIC_API_KEY"), "{message}");
        assert!(message.contains("export ANTHROPIC_API_KEY="), "{message}");
    }

    // The other half of the pair. Without it, a `from_key` that rejected every
    // key -- or stored an empty one -- would pass the test above.
    #[test]
    fn a_key_that_is_present_is_the_one_kept() {
        let Ok(api) = Api::from_key(Some("sk-ant-test".to_owned())) else {
            panic!("a key that was present was rejected");
        };

        assert_eq!(api.key, "sk-ant-test");
    }

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
