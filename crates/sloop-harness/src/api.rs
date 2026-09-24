//! The `/v1/messages` client.
//!
//! Named `api` rather than `client` on purpose. `client` in this workspace is
//! the memory daemon's socket client, and this crate exists partly to show
//! what linking the engine directly looks like instead.
//!
//! Everything here except [`Api::send`] is a pure function over bytes, which
//! is what lets the whole decoder be tested in a sandbox with no network and
//! no API key -- the environment `nix build` runs the test suite in.

mod accumulate;
mod sse;

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use reqwest::header::HeaderValue;
use serde::Serialize;

use crate::api::accumulate::Accumulator;
use crate::api::sse::{data_of, FrameDecoder};
use crate::tree::{serialize_raw_json, ContentBlock, Message};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Pinned rather than tracking whatever the API defaults to. The header is
/// the only thing keeping a server-side breaking change from arriving here as
/// a decoder bug with no commit to blame it on.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The model this harness talks to. Thinking is on by default here, which is
/// why the tree has a thinking block at all.
const MODEL: &str = "claude-opus-5";

/// Large because the request streams. The ceiling that matters for a
/// non-streaming request is the HTTP timeout, and streaming removes it.
const MAX_TOKENS: u32 = 64_000;

/// The body of a `/v1/messages` request.
///
/// `output_config` is absent: its only field this harness would set is
/// `effort`, whose default is already the value we would name.
#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'static str,
    max_tokens: u32,
    stream: bool,
    thinking: Thinking,
    // Borrowed rather than owned because the tree already holds the only copy
    // and a request outlives nothing: it is serialized and dropped.
    messages: &'a [Message],
    // Both skipped when empty, and only ever empty together -- see
    // `Request::new`. `tool_choice` alone is a 400, and `tools` alone would be
    // the caller silently getting a choice it did not name.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [ToolDefinition],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

/// One tool, as `/v1/messages` wants it described.
///
/// A `const` rather than anything built at runtime, which is what fixes the
/// field types: `&'static str` throughout, and `Copy` so a slice of these can
/// be handed around without a lifetime of its own.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ToolDefinition {
    pub name: &'static str,
    /// Prescriptive about *when* to call it, not only what it does -- the
    /// description is the only thing steering the model's should-call
    /// decision, and recent models reach for tools conservatively.
    pub description: &'static str,
    /// The schema's JSON text, written through unchanged.
    ///
    /// Text rather than a `serde_json::Value` built at each call, and text
    /// rather than the `fn() -> Value` this field was first drawn as: a
    /// function pointer has no `Serialize`, so that shape does not compile,
    /// and routing it through `serialize_with` would only be a `const` schema
    /// pretending to be computed. An owned `Value` would cost the `const` and
    /// the `Copy` with it, for a schema that is a literal in the source
    /// either way.
    ///
    /// The tradeoff is the one [`ContentBlock::ToolUse`]'s `input` already
    /// takes: nothing checks the text until it is serialized. `from_string`
    /// re-parses there, so a malformed schema fails the request rather than
    /// corrupting it -- but a well-formed non-object still reaches the API
    /// and comes back as a 400.
    ///
    /// [`ContentBlock::ToolUse`]: crate::tree::ContentBlock::ToolUse
    #[serde(serialize_with = "serialize_raw_json")]
    pub input_schema: &'static str,
}

/// Whether the model may call a tool, must, or must not.
///
/// Three of the wire's four: `any` -- call something, no matter what -- is
/// left out because nothing here wants it, not because it was missed. This is
/// the set of choices this harness makes, not a picture of the API.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    // Nothing outside the tests names either of the two below yet: the demo
    // sends `Auto` and goes on doing so until it starts calling tools. Hence
    // not(test), matching the tree's unbuilt variants -- under cfg(test) they
    // are constructed and an expectation that held there would itself go
    // unfulfilled.
    //
    // Both go live together. The scripted turn that forces a call with `Tool`
    // is followed by one that forbids a call with `None`, which is what makes
    // the second turn terminal text rather than a third request. Neither
    // attribute has to be remembered: constructing a variant expires its
    // expectation, so the build fails here with
    // `unfulfilled_lint_expectations` naming the line to delete. That error
    // is the instruction, not something to suppress.
    //
    // Construction is what does it, and only construction. A `match` arm
    // reads a variant without building one, so `dead_code` goes on firing and
    // the expectation stays fulfilled -- adding one here buys no warning.
    #[cfg_attr(not(test), expect(dead_code, reason = "see above"))]
    None,
    #[cfg_attr(not(test), expect(dead_code, reason = "see above"))]
    Tool {
        name: &'static str,
    },
}

/// `budget_tokens` is absent because it is a 400 on this model.
///
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
    fn new(messages: &'a [Message], tools: &'a [ToolDefinition], choice: ToolChoice) -> Self {
        Self {
            model: MODEL,
            max_tokens: MAX_TOKENS,
            stream: true,
            thinking: Thinking {
                kind: "adaptive",
                display: "summarized",
            },
            messages,
            tools,
            // The choice is dropped rather than refused when there are no
            // tools, because "send nothing about tools" is the only thing a
            // caller can mean by an empty slice, whatever it passed here.
            tool_choice: (!tools.is_empty()).then_some(choice),
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
pub struct Api {
    http: reqwest::Client,
    key: String,
}

/// One completed assistant turn.
///
/// `blocks` is what the tree appends and `stop_reason` is what the caller
/// branches on -- `max_tokens` means the turn is a fragment even though every
/// block in it is whole, which is a distinction no `Vec<ContentBlock>` can
/// carry on its own.
pub struct Turn {
    pub blocks: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
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

        // `Client::new` is the same call with an `expect` around it, and the
        // failure it hides -- no usable TLS backend -- is exactly the kind
        // this constructor already reports in words. Taking the `Result` keeps
        // the panic out of a binary whose lints deny writing one by hand.
        let http = reqwest::Client::builder()
            .build()
            .context("could not build an HTTPS client")?;

        Ok(Self { http, key })
    }

    /// The credential, as a header value that will not print itself.
    ///
    /// `Api` not deriving `Debug` is only half of keeping the key out of a
    /// log, and this is the other half. reqwest's `HeaderMap` and `Request`
    /// print header *values* in their own `Debug`, so anything that formats a
    /// request -- an error path, a retry log, a `tracing` span added later --
    /// would print the key in full and the absent derive would have bought
    /// nothing. `set_sensitive` makes it render as `Sensitive` instead.
    /// `anthropic-version` is deliberately left as a plain `&str` at the call
    /// site: it is a pinned constant rather than a secret, and the asymmetry
    /// is the point rather than an omission.
    ///
    /// A method rather than four lines inside `send` because `send` is the one
    /// thing here that cannot be tested without a socket, and this is the one
    /// thing in `send` that can. The tests below are what keep it from reading
    /// as ceremony and being deleted as such.
    fn key_header(&self) -> Result<HeaderValue> {
        // `from_str` rejects anything outside visible ASCII, which moves one
        // failure earlier: a key carrying a trailing newline or a copy-paste
        // space fails here, naming the problem, rather than arriving as an
        // unexplained 401 from the far end.
        let mut key = HeaderValue::from_str(&self.key).context(
            "the API key is not usable as a header value: \
             it must be visible ASCII with no surrounding whitespace",
        )?;
        key.set_sensitive(true);

        Ok(key)
    }

    /// Send a prompt and stream the turn back.
    ///
    /// `messages` is whatever [`Tree::prompt_for`] returned -- the two types
    /// meet without a conversion, which is the reason this function knows
    /// nothing about trees, tips or status labels.
    ///
    /// `tools` and `choice` are per-request rather than per-client: what the
    /// model may reach for is a property of the turn being asked for, not of
    /// the credential. An empty `tools` sends neither field and `choice` is
    /// then ignored.
    ///
    /// `on_block` sees each block as it completes. Nothing here reports
    /// partial blocks: the tree only ever accepts whole ones, so a caller that
    /// wants a token at a time would be asking for something the tree cannot
    /// store. Its `Result` is load-bearing rather than incidental -- a printing
    /// failure is usually a closed pipe, and a `send` that swallowed one would
    /// go on pulling a whole turn's tokens over the network to write them into
    /// a reader that has gone. It doubles as the caller's only way to stop the
    /// stream early, which is why it is not `FnMut(&ContentBlock)`.
    ///
    /// [`Tree::prompt_for`]: crate::tree::Tree::prompt_for
    // The one item here with no test of its own: this is where the crate
    // stops being pure over bytes, so exercising it needs a socket and a
    // credential, neither of which the sandbox `nix build` runs the suite in
    // has.
    pub async fn send(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        choice: ToolChoice,
        mut on_block: impl FnMut(&ContentBlock) -> Result<()>,
    ) -> Result<Turn> {
        let response = self
            .http
            .post(MESSAGES_URL)
            .header("x-api-key", self.key_header()?)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&Request::new(messages, tools, choice))
            .send()
            .await
            .context("POST /v1/messages")?;

        let status = response.status();
        if !status.is_success() {
            // The body carries the API's own error message, which is more
            // useful than anything this side could say about a 400. Matched
            // rather than `unwrap_or_default()`, because "the API sent no
            // body" and "the connection broke while reading the body" are
            // different failures with different next steps -- the first is a
            // question for the API, the second is a question for the network
            // -- and rendering both as an empty string picks the misleading
            // one of the two every time the connection is at fault.
            let body = match response.text().await {
                Ok(body) => body,
                Err(error) => format!("(the error body could not be read: {error})"),
            };
            bail!("{status} from /v1/messages: {body}");
        }

        let mut chunks = response.bytes_stream();
        let mut frames = FrameDecoder::default();
        let mut accumulator = Accumulator::default();
        let mut blocks = Vec::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.context("reading the response stream")?;

            for frame in frames.decode(&chunk)? {
                let Some(data) = data_of(&frame) else {
                    continue;
                };

                // This `?` ends the turn and discards every block already in
                // `blocks`, and that is the intended reading. `Event::Unknown`
                // already absorbs the forward-compatible case -- an event type
                // this slice has never heard of -- so what reaches here is a
                // tag we do model whose body is not the shape we think it is.
                // That is this client being wrong about the wire, not the
                // server adding something.
                //
                // The alternative, skipping the frame and carrying on, returns
                // a `Turn` that is structurally indistinguishable from a whole
                // one: blocks with a hole in them and a `stop_reason` of
                // `end_turn`. It gets appended to the tree and replayed into
                // every later request, so one dropped delta becomes a
                // permanent silent corruption of the transcript. Failing loses
                // a turn that can be asked for again; tolerating it loses the
                // ability to tell which turns are real.
                let event = serde_json::from_str(&data)
                    .with_context(|| format!("decoding an SSE event: {data}"))?;

                if let Some(block) = accumulator.apply(event)? {
                    on_block(&block).context("reporting a completed block")?;
                    blocks.push(block);
                }
            }
        }

        // A stream that ends without its `message_delta` leaves this `None`,
        // and that is the one truncation the decision above does not catch: a
        // body cut at a frame boundary reaches here as a clean end of stream
        // rather than an error. `Option` is what carries it -- `None` means
        // the turn never said why it stopped, which is the caller's cue that
        // it may be a fragment. A `stop_reason` defaulted to `end_turn` here
        // would erase exactly that.
        Ok(Turn {
            blocks,
            stop_reason: accumulator.stop_reason().map(str::to_owned),
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test reports failure by panicking, and an unwrap is one way"
)]
mod tests {
    use super::{Api, Request, ToolChoice, ToolDefinition};
    use crate::tree::{ContentBlock, Tree};

    // Not the real tool. `search_memory` arrives with the thing that executes
    // it, and a test that named it here would pass whether or not that tool
    // ever existed, while reading as though it had checked.
    const A_FIXTURE_TOOL: ToolDefinition = ToolDefinition {
        name: "a_fixture_tool",
        description: "Stands in for a tool. Never call it; nothing answers.",
        input_schema: r#"{"type": "object", "properties": {"q": {"type": "string"}}}"#,
    };

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

        // Two claims rather than one, because the message has two jobs.
        // Naming the variable is what makes the failure diagnosable; the
        // command is what makes it fixable without a trip to the source.
        //
        // The first is anchored to the start rather than merely present.
        // `contains` passed against a message that had lost its opening
        // mention and read " is not set." -- naming nothing at the one point
        // a reader looks first, while the later `export` line kept the
        // substring alive. One sentence is pinned, not the prose around it,
        // which is still meant to be rewritable.
        assert!(
            message.starts_with("ANTHROPIC_API_KEY is not set"),
            "{message}"
        );
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

    // The half of the credential's protection that `Api` not deriving `Debug`
    // does not cover. This asserts the property -- the key does not appear --
    // rather than the literal `Sensitive` that http happens to print today,
    // because it is the property that matters and the spelling that may move.
    #[test]
    fn the_api_key_header_does_not_print_its_value() {
        let Ok(api) = Api::from_key(Some("sk-ant-secret".to_owned())) else {
            panic!("a key that was present was rejected");
        };
        let header = api.key_header().unwrap();

        assert!(!format!("{header:?}").contains("sk-ant-secret"));
    }

    // A key is pasted by a person, so the newline comes along often enough to
    // be worth a named error. Without this the same input reaches the API and
    // comes back as a 401, which points at the key being wrong rather than at
    // it being punctuated wrong.
    #[test]
    fn a_key_with_a_stray_newline_is_refused_before_the_request() {
        let Ok(api) = Api::from_key(Some("sk-ant-secret\n".to_owned())) else {
            panic!("a key that was present was rejected");
        };
        let Err(error) = api.key_header() else {
            panic!("a key with a trailing newline was accepted as a header");
        };

        assert!(error.to_string().contains("header value"), "{error}");
    }

    #[test]
    fn the_request_body_has_the_shape_the_api_expects() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();

        assert_eq!(
            serde_json::to_value(Request::new(&messages, &[], ToolChoice::Auto)).unwrap(),
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
        let body = serde_json::to_string(&Request::new(&messages, &[], ToolChoice::Auto)).unwrap();

        assert!(!body.contains("budget_tokens"));
        assert!(!body.contains("output_config"));
    }

    #[test]
    fn a_request_with_no_tools_names_neither_field() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let body = serde_json::to_string(&Request::new(&messages, &[], ToolChoice::Auto)).unwrap();

        // `tool_choice` without `tools` is a 400, so neither may appear alone.
        assert!(!body.contains("tools"), "{body}");
        assert!(!body.contains("tool_choice"), "{body}");
    }

    #[test]
    fn a_forced_tool_choice_names_the_tool() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let tools = [A_FIXTURE_TOOL];
        let body = serde_json::to_value(Request::new(
            &messages,
            &tools,
            ToolChoice::Tool {
                name: "a_fixture_tool",
            },
        ))
        .unwrap();

        assert_eq!(
            body["tool_choice"],
            serde_json::json!({ "type": "tool", "name": "a_fixture_tool" })
        );
        // The whole object rather than its `name`. A projection pins only the
        // field it reads: with `assert_eq!(body["tools"][0]["name"], ..)` here,
        // renaming the `description` field to `desc` left all 118 tests green,
        // and the first thing to notice would have been a 400.
        assert_eq!(
            body["tools"][0],
            serde_json::json!({
                "name": "a_fixture_tool",
                "description": "Stands in for a tool. Never call it; nothing answers.",
                "input_schema": { "type": "object", "properties": { "q": { "type": "string" } } }
            })
        );
    }

    // `Auto` was the last arm nothing spelled out, and it is the one the demo
    // actually sends: giving it `#[serde(rename = "always")]` left the suite
    // green. The other two are pinned by the tests around this one.
    #[test]
    fn an_auto_choice_spells_itself_auto() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let tools = [A_FIXTURE_TOOL];
        let body = serde_json::to_value(Request::new(&messages, &tools, ToolChoice::Auto)).unwrap();

        assert_eq!(body["tool_choice"], serde_json::json!({ "type": "auto" }));
    }

    // The claim that the schema travels as an object rather than as the string
    // it is stored in -- the whole risk of holding it as JSON text. A field
    // serialized plainly would read `"{\"type\": \"object\"...}"` here, which
    // the API rejects, and the assertion above on `name` would not notice.
    #[test]
    fn a_tools_schema_travels_as_json_rather_than_as_a_string() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let tools = [A_FIXTURE_TOOL];
        let body = serde_json::to_value(Request::new(&messages, &tools, ToolChoice::Auto)).unwrap();

        assert_eq!(
            body["tools"][0]["input_schema"],
            serde_json::json!({ "type": "object", "properties": { "q": { "type": "string" } } })
        );
    }

    // Do not delete this as redundant. The pairing it appears to check is
    // already covered -- a `new` that dropped the choice fails the forced-tool
    // test above as well as this one -- but this is the only place that
    // constructs `ToolChoice::None`, and the suppression on that variant is
    // `not(test)`. Under cfg(test) `dead_code` applies normally, so removing
    // this turns the variant into a clippy failure at its declaration, which
    // is an obscure thing to meet while tidying a test module.
    #[test]
    fn a_forbidden_choice_spells_itself_none() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let messages = tree.prompt_for(tree.root()).unwrap();
        let tools = [A_FIXTURE_TOOL];
        let body = serde_json::to_value(Request::new(&messages, &tools, ToolChoice::None)).unwrap();

        assert_eq!(body["tool_choice"], serde_json::json!({ "type": "none" }));
    }
}
