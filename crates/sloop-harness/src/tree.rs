//! A conversation as a tree of content blocks.
//!
//! Nodes are content blocks rather than whole messages, and that is forced
//! rather than chosen. Assistant prefill was removed from current models:
//! supplying a partial final assistant turn returns HTTP 400. A branch cannot
//! resume a truncated turn, so it regenerates the turn from a block boundary
//! instead -- and a boundary the tree cannot name is a boundary no branch can
//! start from.
//!
//! The cost is that a node no longer corresponds to anything the API accepts.
//! [`Tree::replay`] reassembles blocks into messages, and [`Tree::prompt_for`]
//! applies the prefill constraint on top.

use serde::{Deserialize, Serialize};

/// A handle to one node of one [`Tree`].
///
/// This is an arena index, so it is meaningful only for the tree that minted
/// it. Passing an id from another tree is the single way every lookup here
/// can fail, which is why they return [`Option`] rather than a richer error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

/// Who authored the turn a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

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
    // Only the tests construct one *through this*. Responses are parsed into
    // the variant now, but `api::accumulate` builds it as a struct literal:
    // it already holds the two strings separately and has nothing to convert,
    // so routing through a constructor taking `impl Into<String>` would buy
    // it nothing. That leaves this a test convenience, and the attribute
    // stays until some caller needs the conversion. Hence not(test) -- under
    // cfg(test) the constructor is live and an expectation that held there
    // would itself go unfulfilled.
    #[cfg_attr(not(test), expect(dead_code, reason = "see above"))]
    pub fn thinking(thinking: impl Into<String>, signature: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: thinking.into(),
            signature: signature.into(),
        }
    }
}

/// What a branch through a node turned out to be worth.
///
/// This is the field the whole crate exists for. Feeding a transcript into
/// the memory index is only safe if a rejection stays distinguishable from a
/// conclusion; without the label every abandoned idea becomes a retrievable
/// fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Status {
    /// Undecided. The default, and the honest answer for most of a live tree.
    #[default]
    Pending,
    /// This block, and the reasoning under it, is a conclusion.
    Kept,
    /// This block, and everything below it, was discarded.
    Abandoned,
}

impl Status {
    /// The stronger of two labels: `Pending < Kept < Abandoned`.
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Pending, Self::Pending) => Self::Pending,
            (Self::Pending, Self::Kept) | (Self::Kept, Self::Pending | Self::Kept) => Self::Kept,
            (Self::Abandoned, Self::Pending | Self::Kept | Self::Abandoned)
            | (Self::Pending | Self::Kept, Self::Abandoned) => Self::Abandoned,
        }
    }
}

/// One message of a `messages[]` array: a role and the blocks under it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug)]
struct Node {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    role: Role,
    block: ContentBlock,
    status: Status,
}

/// A branching conversation.
///
/// There is no `fork` operation and no branch type. [`Tree::append`] is the
/// only way the tree grows, and calling it twice on one parent *is* the fork.
/// A branch is named by its leaf and recovered by walking parent links.
#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    root: NodeId,
}

impl Tree {
    /// Start a conversation from its opening user block.
    ///
    /// The root is a user block by construction, which is what pins the role
    /// of the first message every replay produces.
    #[must_use]
    pub fn new(first_user_block: ContentBlock) -> Self {
        let root = Node {
            parent: None,
            children: Vec::new(),
            role: Role::User,
            block: first_user_block,
            status: Status::Pending,
        };
        Self {
            nodes: vec![root],
            root: NodeId(0),
        }
    }

    /// The opening block.
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(usize::try_from(id.0).ok()?)
    }

    fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(usize::try_from(id.0).ok()?)
    }

    /// This node's children, in append order.
    ///
    /// Empty for a leaf. The order is the order the branches were created,
    /// which is what lets a renderer put the spine first.
    #[must_use]
    pub fn children(&self, id: NodeId) -> Option<&[NodeId]> {
        Some(&self.node(id)?.children)
    }

    /// Hang a block under `parent`, returning the new node.
    ///
    /// Called twice on the same parent, this is a fork. Called on a leaf, it
    /// extends the branch. Nothing distinguishes the two cases in the data.
    pub fn append(&mut self, parent: NodeId, role: Role, block: ContentBlock) -> Option<NodeId> {
        let id = NodeId(u32::try_from(self.nodes.len()).ok()?);
        // Registering the child first means an unknown parent short-circuits
        // here, before the arena has grown a node that nothing points at.
        self.node_mut(parent)?.children.push(id);
        self.nodes.push(Node {
            parent: Some(parent),
            children: Vec::new(),
            role,
            block,
            status: Status::Pending,
        });
        Some(id)
    }

    /// The nodes from the root down to `tip`, inclusive.
    ///
    /// This terminates because a node is pushed after its parent and never
    /// reparented, so a parent index is always lower than its child's. The
    /// arena cannot contain a cycle to walk into.
    #[must_use]
    pub fn path(&self, tip: NodeId) -> Option<Vec<NodeId>> {
        let mut path = Vec::new();
        let mut cursor = Some(tip);
        while let Some(id) = cursor {
            path.push(id);
            cursor = self.node(id)?.parent;
        }
        path.reverse();
        Some(path)
    }

    /// Every leaf, in creation order. One per branch.
    #[must_use]
    pub fn leaves(&self) -> Vec<NodeId> {
        let mut leaves = Vec::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if !node.children.is_empty() {
                continue;
            }
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            leaves.push(NodeId(index));
        }
        leaves
    }

    /// The block this node carries.
    #[must_use]
    pub fn block(&self, id: NodeId) -> Option<&ContentBlock> {
        Some(&self.node(id)?.block)
    }

    /// This node's role.
    #[must_use]
    pub fn role(&self, id: NodeId) -> Option<Role> {
        Some(self.node(id)?.role)
    }

    /// The label placed on this node, ignoring its ancestors.
    #[must_use]
    pub fn status(&self, id: NodeId) -> Option<Status> {
        Some(self.node(id)?.status)
    }

    /// Label a node.
    pub fn set_status(&mut self, id: NodeId, status: Status) -> Option<()> {
        self.node_mut(id)?.status = status;
        Some(())
    }

    /// The label in force at `id`, accounting for its ancestors.
    ///
    /// The strongest label on the path from the root wins, so an `Abandoned`
    /// ancestor poisons everything below it and an unlabeled node inherits a
    /// `Kept` ancestor.
    #[must_use]
    pub fn effective_status(&self, id: NodeId) -> Option<Status> {
        let mut status = Status::Pending;
        for ancestor in self.path(id)? {
            status = status.join(self.node(ancestor)?.status);
        }
        Some(status)
    }

    /// The branch ending at `tip`, as a `messages[]` array.
    ///
    /// Consecutive blocks of the same role are one message, so a role change
    /// is a turn boundary. Strict alternation is therefore structural: this
    /// cannot emit two adjacent messages of the same role.
    #[must_use]
    pub fn replay(&self, tip: NodeId) -> Option<Vec<Message>> {
        let mut messages: Vec<Message> = Vec::new();
        for id in self.path(tip)? {
            let node = self.node(id)?;
            match messages.last_mut() {
                Some(last) if last.role == node.role => last.content.push(node.block.clone()),
                Some(_) | None => messages.push(Message {
                    role: node.role,
                    content: vec![node.block.clone()],
                }),
            }
        }
        Some(messages)
    }

    /// What to POST to get the next assistant turn on the branch at `tip`.
    ///
    /// This is [`Tree::replay`] with a trailing assistant message dropped,
    /// which is the prefill constraint made executable: an assistant turn
    /// cannot be continued, so it is regenerated from the last user boundary.
    ///
    /// Because the root is a user block, the result always starts and ends
    /// with a user message, and is never empty.
    #[must_use]
    pub fn prompt_for(&self, tip: NodeId) -> Option<Vec<Message>> {
        let mut messages = self.replay(tip)?;
        let trailing_assistant = match messages.last() {
            Some(last) => last.role == Role::Assistant,
            None => false,
        };
        if trailing_assistant {
            messages.pop();
        }
        Some(messages)
    }
}

// A NodeId is only ever minted by this module, so every `unwrap` below is on
// an id the test itself just created. A test that mishandles one should fail
// loudly rather than quietly assert on a `None`.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "see comment above")]
mod tests {
    use proptest::prelude::*;

    use super::{ContentBlock, Message, NodeId, Role, Status, Tree};

    /// The transcript every fork test starts from:
    ///
    ///     u1  "How should the cache expire?"        user
    ///     a1  "Two options."                        assistant
    ///     a2  "Use an LRU."                         assistant
    fn transcript() -> (Tree, NodeId, NodeId, NodeId) {
        let mut tree = Tree::new(ContentBlock::text("How should the cache expire?"));
        let u1 = tree.root();
        let a1 = tree
            .append(u1, Role::Assistant, ContentBlock::text("Two options."))
            .unwrap();
        let a2 = tree
            .append(a1, Role::Assistant, ContentBlock::text("Use an LRU."))
            .unwrap();
        (tree, u1, a1, a2)
    }

    fn roles(messages: &[Message]) -> Vec<Role> {
        let mut out = Vec::new();
        for message in messages {
            out.push(message.role);
        }
        out
    }

    /// The prose of every block, whichever kind it is.
    ///
    /// Thinking counts as prose here: the tests that read this one compare
    /// against string literals, so what they are checking is which blocks
    /// landed on a branch, not what kind they are. Anything that turns on a
    /// signature has to compare the blocks themselves.
    fn prose(messages: &[Message]) -> Vec<String> {
        let mut out = Vec::new();
        for message in messages {
            for block in &message.content {
                let text = match block {
                    ContentBlock::Text { text } => text,
                    ContentBlock::Thinking {
                        thinking,
                        signature: _,
                    } => thinking,
                };
                out.push(text.clone());
            }
        }
        out
    }

    // ---- the milestone -------------------------------------------------

    #[test]
    fn forking_and_abandoning_yields_two_coherent_transcripts() {
        let (mut tree, _u1, a1, a2) = transcript();

        // Fork mid-assistant-turn: a sibling of a2, not a continuation of it.
        let a2_alt = tree
            .append(a1, Role::Assistant, ContentBlock::text("Use a TTL map."))
            .unwrap();

        tree.set_status(a2, Status::Kept).unwrap();
        tree.set_status(a2_alt, Status::Abandoned).unwrap();

        let kept = tree.replay(a2).unwrap();
        let abandoned = tree.replay(a2_alt).unwrap();

        // Both are valid messages arrays, and both alternate.
        assert_eq!(roles(&kept), vec![Role::User, Role::Assistant]);
        assert_eq!(roles(&abandoned), vec![Role::User, Role::Assistant]);

        // They share the prefix and differ only in the forked block.
        assert_eq!(kept[0], abandoned[0]);
        assert_eq!(
            prose(&kept),
            vec![
                "How should the cache expire?",
                "Two options.",
                "Use an LRU.",
            ]
        );
        assert_eq!(
            prose(&abandoned),
            vec![
                "How should the cache expire?",
                "Two options.",
                "Use a TTL map.",
            ]
        );

        // And the rejection stays labeled as one.
        assert_eq!(tree.effective_status(a2).unwrap(), Status::Kept);
        assert_eq!(tree.effective_status(a2_alt).unwrap(), Status::Abandoned);
    }

    // ---- growth --------------------------------------------------------

    #[test]
    fn a_fresh_tree_replays_to_a_single_user_message() {
        let tree = Tree::new(ContentBlock::text("hello"));
        let messages = tree.replay(tree.root()).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(prose(&messages), vec!["hello"]);
    }

    #[test]
    fn appending_twice_to_one_parent_is_the_fork() {
        let (mut tree, _u1, a1, a2) = transcript();
        let a2_alt = tree
            .append(a1, Role::Assistant, ContentBlock::text("Use a TTL map."))
            .unwrap();

        assert_ne!(a2, a2_alt);
        assert_eq!(tree.leaves(), vec![a2, a2_alt]);
    }

    #[test]
    fn children_come_back_in_append_order() {
        let mut tree = Tree::new(ContentBlock::text("q"));
        let root = tree.root();
        let first = tree
            .append(root, Role::Assistant, ContentBlock::text("a"))
            .unwrap();
        let second = tree
            .append(root, Role::Assistant, ContentBlock::text("b"))
            .unwrap();

        assert_eq!(tree.children(root), Some(&[first, second][..]));
        assert_eq!(tree.children(first), Some(&[][..]));
    }

    #[test]
    fn a_node_id_from_another_tree_is_rejected() {
        let (mut tree, ..) = transcript();
        let other = Tree::new(ContentBlock::text("a different conversation"));
        let stranger = tree
            .append(tree.root(), Role::Assistant, ContentBlock::text("x"))
            .unwrap();

        assert!(other.replay(stranger).is_none());
        assert!(other.path(stranger).is_none());
        assert!(other.effective_status(stranger).is_none());
    }

    // ---- replay --------------------------------------------------------

    #[test]
    fn consecutive_same_role_blocks_become_one_message() {
        let (tree, _u1, _a1, a2) = transcript();
        let messages = tree.replay(a2).unwrap();

        // Three blocks, but only two messages: a1 and a2 are one turn.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.len(), 2);
    }

    #[test]
    fn a_role_change_starts_a_new_message() {
        let (mut tree, _u1, _a1, a2) = transcript();
        let u2 = tree
            .append(a2, Role::User, ContentBlock::text("Why not a TTL?"))
            .unwrap();

        assert_eq!(
            roles(&tree.replay(u2).unwrap()),
            vec![Role::User, Role::Assistant, Role::User]
        );
    }

    #[test]
    fn replay_serializes_to_the_messages_wire_shape() {
        let tree = Tree::new(ContentBlock::text("hi"));
        let json = serde_json::to_value(tree.replay(tree.root()).unwrap()).unwrap();

        assert_eq!(
            json,
            serde_json::json!([{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }])
        );
    }

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
            .append(
                thinking,
                Role::Assistant,
                ContentBlock::text("Two options."),
            )
            .unwrap();

        // One assistant turn, two blocks: the grouping rule does not care which
        // kinds they are. Asserting the blocks themselves rather than a count
        // is what pins the order -- thinking leads, and the signature it came
        // with is still attached to it.
        let messages = tree.replay(text).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[1].content,
            vec![
                ContentBlock::thinking("Weighing LRU against TTL.", "ErUBCkYIBRgCIkA="),
                ContentBlock::text("Two options."),
            ]
        );
    }

    // ---- prompt_for ----------------------------------------------------

    #[test]
    fn prompt_for_drops_a_trailing_assistant_turn() {
        let (tree, _u1, _a1, a2) = transcript();

        // The whole assistant turn regenerates, so neither of its blocks is sent.
        assert_eq!(roles(&tree.prompt_for(a2).unwrap()), vec![Role::User]);
        assert_eq!(
            prose(&tree.prompt_for(a2).unwrap()),
            vec!["How should the cache expire?"]
        );
    }

    #[test]
    fn prompt_for_leaves_a_user_tip_alone() {
        let (mut tree, _u1, _a1, a2) = transcript();
        let u2 = tree
            .append(a2, Role::User, ContentBlock::text("Why not a TTL?"))
            .unwrap();

        assert_eq!(tree.prompt_for(u2).unwrap(), tree.replay(u2).unwrap());
    }

    #[test]
    fn steering_is_a_user_block_under_a_mid_turn_assistant_block() {
        // Interrupting a turn keeps the partial assistant content and appends
        // a user turn; forking discards it. Both are the same primitive.
        let (mut tree, _u1, a1, _a2) = transcript();
        let steer = tree
            .append(a1, Role::User, ContentBlock::text("Stop, wrong file."))
            .unwrap();

        let messages = tree.replay(steer).unwrap();
        assert_eq!(
            roles(&messages),
            vec![Role::User, Role::Assistant, Role::User]
        );
        // a2 is not on this branch, so the partial turn is a1 alone.
        assert_eq!(messages[1].content.len(), 1);
        // Already ends on a user message, so nothing is dropped.
        assert_eq!(tree.prompt_for(steer).unwrap(), messages);
    }

    // ---- status --------------------------------------------------------

    #[test]
    fn an_unlabeled_node_is_pending() {
        let (tree, _u1, _a1, a2) = transcript();

        assert_eq!(tree.status(a2).unwrap(), Status::Pending);
        assert_eq!(tree.effective_status(a2).unwrap(), Status::Pending);
    }

    #[test]
    fn an_abandoned_ancestor_poisons_its_descendants() {
        let (mut tree, _u1, a1, a2) = transcript();
        tree.set_status(a1, Status::Abandoned).unwrap();

        assert_eq!(tree.status(a2).unwrap(), Status::Pending);
        assert_eq!(tree.effective_status(a2).unwrap(), Status::Abandoned);
    }

    #[test]
    fn a_kept_ancestor_is_inherited_by_an_unlabeled_descendant() {
        let (mut tree, _u1, a1, a2) = transcript();
        tree.set_status(a1, Status::Kept).unwrap();

        assert_eq!(tree.effective_status(a2).unwrap(), Status::Kept);
    }

    #[test]
    fn abandoning_a_later_step_overrides_a_kept_ancestor() {
        let (mut tree, _u1, a1, a2) = transcript();
        tree.set_status(a1, Status::Kept).unwrap();
        tree.set_status(a2, Status::Abandoned).unwrap();

        assert_eq!(tree.effective_status(a2).unwrap(), Status::Abandoned);
    }

    #[test]
    fn a_kept_descendant_does_not_rescue_an_abandoned_ancestor() {
        let (mut tree, _u1, a1, a2) = transcript();
        tree.set_status(a1, Status::Abandoned).unwrap();
        tree.set_status(a2, Status::Kept).unwrap();

        assert_eq!(tree.effective_status(a2).unwrap(), Status::Abandoned);
    }

    #[test]
    fn a_sibling_label_does_not_leak_across_the_fork() {
        let (mut tree, _u1, a1, a2) = transcript();
        let a2_alt = tree
            .append(a1, Role::Assistant, ContentBlock::text("Use a TTL map."))
            .unwrap();
        tree.set_status(a2_alt, Status::Abandoned).unwrap();

        assert_eq!(tree.effective_status(a2).unwrap(), Status::Pending);
    }

    // ---- properties ----------------------------------------------------

    /// Build a tree from a script of `(parent selector, is assistant,
    /// is thinking)` steps.
    ///
    /// Selecting the parent modulo the node count puts forks wherever the
    /// generator likes, which is the axis these properties have to hold on.
    /// The kind is the second axis: replay groups by role alone, so mixing
    /// thinking into a turn must not change where any boundary falls.
    ///
    /// Content is derived from the index rather than generated. No property
    /// asserts a particular value, but it must be distinct per node so
    /// [`replay_preserves_every_block_on_the_path`] can compare sequences
    /// instead of lengths, and so a shrunk counterexample stays readable.
    fn build(script: &[(usize, bool, bool)]) -> Tree {
        let mut tree = Tree::new(ContentBlock::text("root"));
        let mut ids = vec![tree.root()];
        for (index, (selector, is_assistant, is_thinking)) in script.iter().enumerate() {
            let parent = ids[selector % ids.len()];
            let role = if *is_assistant {
                Role::Assistant
            } else {
                Role::User
            };
            // Gated on the role: only a model produces thinking, so a user
            // thinking block is a transcript the API would reject. Nothing
            // here would catch it, which is exactly why it must not be
            // generated -- the next reader would take it for a legal shape.
            let block = if *is_assistant && *is_thinking {
                ContentBlock::thinking(format!("block {index}"), format!("signature {index}"))
            } else {
                ContentBlock::text(format!("block {index}"))
            };
            if let Some(id) = tree.append(parent, role, block) {
                ids.push(id);
            }
        }
        tree
    }

    fn script() -> impl Strategy<Value = Vec<(usize, bool, bool)>> {
        proptest::collection::vec((0_usize..64, any::<bool>(), any::<bool>()), 0..40)
    }

    fn alternates(messages: &[Message]) -> bool {
        for pair in messages.windows(2) {
            if pair[0].role == pair[1].role {
                return false;
            }
        }
        true
    }

    proptest! {
        /// Grouping consecutive same-role blocks cannot produce two adjacent
        /// messages of one role, so this holds wherever the forks sit.
        #[test]
        fn replay_always_alternates_and_starts_with_user(script in script()) {
            let tree = build(&script);
            let leaves = tree.leaves();
            prop_assert!(!leaves.is_empty(), "every tree has at least the root as a leaf");
            for leaf in leaves {
                let messages = tree.replay(leaf).unwrap();
                prop_assert!(!messages.is_empty());
                prop_assert_eq!(messages[0].role, Role::User);
                prop_assert!(alternates(&messages));
                prop_assert!(messages.iter().all(|m| !m.content.is_empty()));
            }
        }

        /// The array `prompt_for` returns must be POSTable: non-empty, and
        /// ending on a user turn, because prefill is a 400.
        #[test]
        fn prompt_for_always_ends_on_a_user_turn(script in script()) {
            let tree = build(&script);
            let leaves = tree.leaves();
            prop_assert!(!leaves.is_empty(), "every tree has at least the root as a leaf");
            for leaf in leaves {
                let messages = tree.prompt_for(leaf).unwrap();
                prop_assert!(!messages.is_empty());
                prop_assert_eq!(messages[0].role, Role::User);
                prop_assert_eq!(messages[messages.len() - 1].role, Role::User);
                prop_assert!(alternates(&messages));
            }
        }

        /// Regrouping blocks into messages must not lose, duplicate or
        /// reorder any of them.
        #[test]
        fn replay_preserves_every_block_on_the_path(script in script()) {
            let tree = build(&script);
            let leaves = tree.leaves();
            prop_assert!(!leaves.is_empty(), "every tree has at least the root as a leaf");
            for leaf in leaves {
                let path = tree.path(leaf).unwrap();
                let messages = tree.replay(leaf).unwrap();

                // The blocks themselves, not a projection of them. Lengths
                // alone would miss a transposition, and prose alone would miss
                // a mangled signature -- which is a 400 on the next turn, so
                // it has to fail here.
                let mut walked = Vec::new();
                for id in &path {
                    walked.push(tree.node(*id).unwrap().block.clone());
                }
                let mut emitted = Vec::new();
                for message in &messages {
                    emitted.extend(message.content.iter().cloned());
                }
                prop_assert_eq!(emitted, walked);
            }
        }

        /// Abandonment is inherited, so no descendant of an abandoned node
        /// reports anything else.
        #[test]
        fn abandonment_reaches_every_descendant(script in script()) {
            let mut tree = build(&script);
            let leaves = tree.leaves();
            prop_assert!(!leaves.is_empty(), "every tree has at least the root as a leaf");
            let first = leaves[0];
            let path = tree.path(first).unwrap();
            // Abandon the midpoint of one branch, then check its whole subtree.
            let pivot = path[path.len() / 2];
            tree.set_status(pivot, Status::Abandoned).unwrap();

            for leaf in tree.leaves() {
                let contains_pivot = tree.path(leaf).unwrap().contains(&pivot);
                let status = tree.effective_status(leaf).unwrap();
                prop_assert_eq!(contains_pivot, status == Status::Abandoned);
            }
        }
    }
}
