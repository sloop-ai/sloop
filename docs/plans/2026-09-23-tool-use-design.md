# Tool use

`sloop-harness` can hold a conversation and cannot do anything. `tree.rs:33`
says so in a comment -- `tool_use` and `tool_result` "arrive with the slice
that has tools to call" -- and `api/sse.rs:148` already absorbs a tool call
into `BlockStart::Other`, accumulates it as `Partial::Unmodelled`, and throws
it away when the block closes. The request has no `tools` field, so today none
of that can happen anyway.

This slice adds the two block kinds, the rule the tree needs to hold them
safely, and one tool worth calling.

## The constraint

Three rules govern the pair, and all three are the API's rather than this
crate's.

An assistant message containing `tool_use` blocks must be followed by a user
message answering every one of them. One turn can contain several calls, and
their results go back in a *single* user message -- splitting them across
several messages teaches the model to stop calling tools in parallel. A tool
that failed comes back as a `tool_result` with `is_error: true`, never as a
dropped block, because a dropped block is an unanswered call and an unanswered
call is a 400.

The third rule is the one that shapes this design: none of it can be fixed
after the fact. A transcript that violates the pairing is not a transcript the
API will accept in any later request, and the tree is append-only, so nothing
downstream can repair it.

## Where the tree can break it

Four cases, and the first is not the one the README predicted.

**Steering under an unanswered call.** Steering is
`append(mid_turn_assistant_node, Role::User, block)` -- a user child hanging
off an interior assistant block. When that assistant block is a `tool_use`,
the branch replays to an assistant turn ending in a call followed by a user
turn that does not answer it. This is the main way the tree can produce an
unsendable branch, and it arrives through an existing primitive rather than a
new one.

**A `tool_result` with nothing to answer.** Appended where no call is
outstanding, or answering a call already answered.

**A partially answered turn.** Two calls, one result appended so far. Legal to
hold, illegal to send, and it is a state the tree *must* be able to hold: the
results arrive one node at a time.

**A fork that truncates the pair.** The case the README named, and it cannot
happen. `prompt_for` drops a trailing assistant message whole, so an
unanswered call leaves with the turn that made it.

## Two states, two mechanisms

The four cases split cleanly by whether the violation is permanent, and that
split is the design.

Cases one and two can never become legal. A node's ancestors never change, so
a branch that is unsendable at `append` time is unsendable forever. **`append`
refuses them**, at the only moment a caller can still act:

    pub enum AppendError {
        UnknownParent,
        InterjectedUnderToolUse { unanswered: Vec<String> },
        UnmatchedToolResult { tool_use_id: String },
    }

`UnknownParent` preserves what `None` used to mean, which makes `tree.rs:18`
true again rather than stale. Both checks are answered by one walk up from the
proposed parent: collect the `tool_use` ids in the assistant run at the tip,
subtract the `tool_result` ids already appended below them, and the remainder
is what is outstanding.

Case three is not a violation, only an unfinished job. **`prompt_for` reports
it**:

    pub enum Prompt {
        Ready(Vec<Message>),
        Pending(Vec<ToolUse>),
    }

`Ready` is today's behaviour, trailing-assistant drop included. `Pending`
hands back the calls to execute, which is exactly what the caller needs to
make progress.

Two alternatives were rejected. Putting the whole rule in `append` cannot
work: case three must be appendable, so `append` would enforce a rule with a
hole in it. Putting the whole rule in `prompt_for` works but reports cases one
and two as "pending tool calls" forever, telling the caller to execute
something that will not help -- an error that recommends a useless action is
worse than none.

The split also makes the tool constraint rhyme with the prefill constraint
rather than fight it. A partial assistant turn is a legal interior state
resolved by **dropping**: an assistant turn cannot be continued, so it is
regenerated from the last user boundary. A partial tool answer is a legal
interior state resolved by **filling**: the calls get executed and the results
appended. Opposite resolutions, and neither one rewrites history, which is
what keeps every transcript this tree produces append-only over its prefix.

## The blocks

    ToolUse {
        id: String,
        name: String,
        /// The input's JSON text, exactly as the deltas delivered it.
        input: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },

`input` stays raw JSON text for the reason `signature` is never reformatted.
The block travels back in every later request, and parsing then reserializing
rewrites bytes the model produced -- key order, escaping, number formatting.
Tool-call inputs are documented to vary in escaping between models and must be
parsed rather than string-matched; that cuts both ways, and the safe direction
keeps the bytes and parses a copy at execution time. It also keeps `Eq` on
`ContentBlock`, which `serde_json::Value` would take away.

The cost is two helpers: a `serialize_with` wrapping the text in `RawValue` so
it emits as an object rather than a quoted string, and a `deserialize_with`
taking it back. `content` stays a plain `String`; the API also accepts an
array of blocks there and nothing here needs one.

Note what the types leave out. A `tool_use` is an assistant block and a
`tool_result` is a user block, but role lives on the node, not in the variant.
Nothing in the type stops a `tool_result` under an assistant node -- that is
the validity rule's job, the same division of labour as everywhere else here.

## Forking a turn that called a tool

`fork_point` gains one case: a `tool_use` cannot be the shared block, and
takes the turn-boundary fallback a signed block takes.

The reason differs from the signature one. Sharing a `tool_use` means two
generations answering one call id in two contexts, and because no ordinary
block follows a call inside a turn, an interior fork under one describes a
turn the API never produces. So `signed()` stops being the right name. The
predicate asks whether a block can belong to two generations at once, and
signatures and tool calls are its two negative cases.

## The tool

One module, `tools.rs`, one tool, no registry and no trait:

    search_memory(query: string) -> hits

Execution is the engine call chain in process -- `Embedder::load`,
`encode_query`, `store::connect(db_dir())`, `open_table(TABLE_CHUNKS)`,
`hybrid_search` -- which is the case the library/binary split exists for. The
embedder loads on first call rather than at startup, and every failure along
that chain returns a `tool_result` with `is_error: true` carrying the message:
no model weights, no database, no table, a query that fails. That is the API's
designed channel for a failed tool, and it means the demo runs end to end on a
machine with no index built. It shows the model being told the index is
missing and recovering, which demonstrates something real rather than
crashing.

The result content is per hit: `rel_path`, `heading_path`, cosine, and the
chunk text. Fuller than a pointer, because a search tool returning only paths
gives the model nothing to reason with.

## What reaches the index

`text_of` at `transcript.rs:13` gains two arms, and they go opposite ways. It
returns `Option<Cow<'_, str>>` to allow a constructed line.

A `tool_use` renders as one line naming the tool and its input. JSON escapes
its newlines, so the input is genuinely one line and flows through the
existing fence balancing like any other text. That the memory was consulted,
and what was asked of it, is worth retrieving later.

A `tool_result` renders as nothing, the way `thinking` does. The reason is
general rather than a property of this tool: a tool result is derived content.
It came from somewhere already indexed -- the duplication case, where the copy
competes with the note it came from and beats it on nothing, having no
frontmatter, no links, a `heading_path` from the wrong document and a
`captured` date from the wrong day -- or from somewhere that is not the
harness's to index at all. Deciding this per tool couples the renderer to
every tool's result schema; deciding it once for the category does not.

The payload reaches the model and never the corpus. What survives into the
index is whatever the assistant concluded from it, which is a conclusion
rather than a copy.

## The demo

Three live calls and one local refusal.

1. Turn 1 with `tools` declared and `tool_choice` naming `search_memory`, so
   the round trip happens because the demo said so rather than because the
   model felt like searching. Ends on `stop_reason: tool_use`.
2. `append(tool_use_node, Role::User, text("actually, never mind"))`, which
   prints the `InterjectedUnderToolUse` refusal. It costs nothing and it is
   the clearest evidence the rule works.
3. The real `tool_result`, one node per call, so a parallel-call turn is
   answered by several nodes that `replay` groups back into one message.
4. Turn 2 with `tool_choice: {"type": "none"}`, which guarantees a terminal
   text turn rather than another call.
5. Fork turn 2, abandon, regenerate, render, as today -- except the shared
   prefix now holds a whole tool exchange, so both branches replay a completed
   pair.

## Testing

The fixture is **captured, not written**. A real `.sse` stream carrying a tool
call, saved beside the existing three. Hand-writing one from documentation
encodes assumptions about `input_json_delta` framing and then tests them
against themselves. The existing fixtures are the only readable record of the
wire, and that is only true of fixtures that came off it.

On top of that:

- `accumulate`: a call assembling from partial JSON chunks, and the
  zero-delta case finishing as `{}` rather than an empty string.
- `tree`: each `AppendError` variant; `Pending` against `Ready`; several
  results replaying into one user message.
- `transcript`: a result payload absent from the markdown, the call line
  present.
- `proptest`, extending the existing replay properties: for any generated
  tree, a `Ready` prompt answers every `tool_use` in its final assistant
  message in the message after it. That is the property the rule exists to
  produce, and fork placement is where generated cases have already earned
  their keep.

Then a mutation check on the rule: relax `append` to accept the interjection
and confirm a test goes red. In a worktree, not the shared checkout.

## What a second tool changes

`tools.rs` is explicit because there is one tool, not because explicit is the
end state. Harnesses generally hand-write their own tools -- the description
is a prompt-engineering surface rather than something derivable from a type --
and reach for a generic path, usually MCP, for tools they do not own.
`sloop-memory` already serves that side of the wire.

Two things stop being cosmetic when tool two arrives.

**Dispatch failure becomes reachable.** With one tool and a forced
`tool_choice`, the model cannot call something that does not exist. With a
registry it can, and the answer is a `tool_result` with `is_error: true`
naming the unknown tool -- never a crash and never a dropped block, because a
dropped block is an unanswered call.

**The tool list enters the cache prefix.** Caching renders `tools`, then
`system`, then `messages`, so a tool set that varies per turn invalidates the
whole conversation every turn. A fixed set of one cannot show this. A stable,
sorted list is the answer when it can, and cache-hit instrumentation is
already on the crate's remaining-work list.

## Out of scope

- The agent loop. This slice is scripted: one round trip, fixed turn count.
  Tools do not imply a loop, and bolting one on here would obscure whether the
  validity rule is right.
- Retries and backoff, still on the crate's list.
- `cache_control` on any block.
- Structured outputs, `strict: true`, and server-side tools.
