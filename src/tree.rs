//! Safe, contiguous, two-generation arena tree (replaces C++ `allocator.h` + `node.h`).
//!
//! ### Architectural Evolution from C++ `beat-it`
//! - **Zero `unsafe`, Zero Raw Pointers**: Replaces 64-bit pointers and `boost::intrusive::list`
//!   with 32-bit [`NodeId`] handles and contiguous child slices `(first_child: NodeId, child_count: u16)`.
//! - **No `PreNode` / `FixNode` / `NodeVisitor` / `shared_mutex`**: Because each tree is strictly
//!   single-threaded and reconstructs `Game` along the root-to-leaf selection path, nodes do not
//!   store `State`, `vptr`, `parent` pointers, or OS locks—shrinking each MCTS node from
//!   ~144–288 bytes in C++ down to **20 bytes** (over 3 nodes per 64-byte CPU cache line!).
//! - **Generational Ping-Pong Preserved**: [`GenerationalTree`] retains two pre-allocated
//!   [`TreeArena`] buffers (`active` and `standby`). Advancing the root copies only the chosen
//!   child's surviving subtree into `standby` (preserving contiguous child layout) and resets
//!   the old generation in `O(1)` without calling `free()` or `malloc()`.

/// Compact 32-bit index into a [`TreeArena`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Sentinel value representing "no node" (analogous to `nullptr`).
    pub const NONE: Self = Self(u32::MAX);

    /// Index of the root node in any initialized [`TreeArena`].
    pub const ROOT: Self = Self(0);

    #[inline]
    pub const fn from_index(index: usize) -> Self {
        assert!(index < (u32::MAX as usize), "TreeArena exceeded u32::MAX - 1 nodes");
        Self(index as u32)
    }

    #[inline]
    pub const fn index(self) -> usize {
        debug_assert!(self.0 != u32::MAX, "attempted to index NodeId::NONE");
        self.0 as usize
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Compact search-tree node generic over move `M` and algorithm payload `P`.
///
/// All children of a node are stored contiguously in [`TreeArena`] at
/// `first_child.index() .. first_child.index() + child_count as usize`.
#[derive(Clone, Copy, Debug)]
pub struct Node<M, P> {
    pub payload: P,
    first_child: NodeId,
    child_count: u16,
    /// The move that transitioned from the parent state into this node.
    pub mv: M,
}

impl<M, P> Node<M, P> {
    #[inline]
    pub const fn new(mv: M, payload: P) -> Self {
        Self {
            payload,
            first_child: NodeId::NONE,
            child_count: 0,
            mv,
        }
    }

    #[inline]
    pub const fn has_children(&self) -> bool {
        self.child_count > 0
    }

    #[inline]
    pub const fn child_count(&self) -> usize {
        self.child_count as usize
    }

    #[inline]
    pub const fn first_child(&self) -> NodeId {
        self.first_child
    }
}

/// Contiguous bump-allocated node buffer representing a single generation of a search tree.
#[derive(Clone, Debug)]
pub struct TreeArena<M, P> {
    nodes: Vec<Node<M, P>>,
}

impl<M, P> TreeArena<M, P> {
    /// Pre-allocates capacity for `capacity` nodes upfront.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Resets the arena in `O(1)` while retaining full allocated heap capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.nodes.capacity()
    }

    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node<M, P>>()
    }

    /// Initializes the arena with a single root node at [`NodeId::ROOT`].
    #[inline]
    pub fn init_root(&mut self, mv: M, payload: P) -> NodeId {
        self.nodes.clear();
        self.nodes.push(Node::new(mv, payload));
        NodeId::ROOT
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> &Node<M, P> {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node<M, P> {
        &mut self.nodes[id.index()]
    }

    /// Returns an immutable contiguous slice of `id`'s children.
    #[inline]
    pub fn children(&self, id: NodeId) -> &[Node<M, P>] {
        let node = &self.nodes[id.index()];
        if node.child_count == 0 {
            &[]
        } else {
            let start = node.first_child.index();
            &self.nodes[start..start + (node.child_count as usize)]
        }
    }

    /// Returns a mutable contiguous slice of `id`'s children (e.g., for writing NN priors
    /// or sorting evaluated moves in iterative-deepening Tree Minimax).
    #[inline]
    pub fn children_mut(&mut self, id: NodeId) -> &mut [Node<M, P>] {
        let node = &self.nodes[id.index()];
        if node.child_count == 0 {
            &mut []
        } else {
            let start = node.first_child.index();
            let count = node.child_count as usize;
            &mut self.nodes[start..start + count]
        }
    }

    /// Returns `(NodeId, &Node<M, P>)` pairs for all children of `id`.
    #[inline]
    pub fn children_with_ids(&self, id: NodeId) -> impl ExactSizeIterator<Item = (NodeId, &Node<M, P>)> {
        let children = self.children(id);
        let base = if children.is_empty() {
            0
        } else {
            self.nodes[id.index()].first_child.index()
        };
        children
            .iter()
            .enumerate()
            .map(move |(offset, node)| (NodeId::from_index(base + offset), node))
    }

    /// Expands `parent` by appending all `(mv, payload)` items contiguously into the arena
    /// and returns a mutable slice over the newly created child nodes.
    #[inline]
    pub fn expand_from_iter<I>(&mut self, parent: NodeId, iter: I) -> &mut [Node<M, P>]
    where
        I: IntoIterator<Item = (M, P)>,
    {
        debug_assert!(
            !self.nodes[parent.index()].has_children(),
            "cannot expand an already-expanded node"
        );
        let start = self.nodes.len();
        for (mv, payload) in iter {
            self.nodes.push(Node::new(mv, payload));
        }
        let count = self.nodes.len() - start;
        assert!(
            count <= (u16::MAX as usize),
            "node child count ({count}) exceeded u16::MAX"
        );
        if count > 0 {
            let parent_node = &mut self.nodes[parent.index()];
            parent_node.first_child = NodeId::from_index(start);
            parent_node.child_count = count as u16;
        }
        &mut self.nodes[start..start + count]
    }

    /// Counts all nodes in the subtree rooted at `id` (port of C++ `node_count`).
    pub fn subtree_count(&self, id: NodeId) -> usize {
        let mut total = 1;
        for (child_id, _) in self.children_with_ids(id) {
            total += self.subtree_count(child_id);
        }
        total
    }
}

/// Two-generation ping-pong tree arena (port of C++ `GenerationalArenaAllocator` + `copy_tree`).
///
/// Retains reusable subtrees across turns without per-node deallocation or heap fragmentation.
#[derive(Clone, Debug)]
pub struct GenerationalTree<M, P> {
    active: TreeArena<M, P>,
    standby: TreeArena<M, P>,
}

impl<M: Copy, P: Copy> GenerationalTree<M, P> {
    /// Creates a new two-generation tree with initial `capacity` nodes per arena
    /// and initializes [`NodeId::ROOT`].
    pub fn new(capacity: usize, root_move: M, root_payload: P) -> Self {
        let mut active = TreeArena::with_capacity(capacity);
        active.init_root(root_move, root_payload);
        let standby = TreeArena::with_capacity(capacity);
        Self { active, standby }
    }

    #[inline]
    pub const fn root_id(&self) -> NodeId {
        NodeId::ROOT
    }

    #[inline]
    pub fn root(&self) -> &Node<M, P> {
        self.active.get(NodeId::ROOT)
    }

    #[inline]
    pub fn root_mut(&mut self) -> &mut Node<M, P> {
        self.active.get_mut(NodeId::ROOT)
    }

    #[inline]
    pub const fn active(&self) -> &TreeArena<M, P> {
        &self.active
    }

    #[inline]
    pub fn active_mut(&mut self) -> &mut TreeArena<M, P> {
        &mut self.active
    }

    /// Total heap memory reserved across both `active` and `standby` arenas in bytes.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.active.allocated_bytes() + self.standby.allocated_bytes()
    }

    /// Resets the tree to a fresh single root node in `O(1)`.
    pub fn reset_root(&mut self, root_move: M, root_payload: P) {
        self.active.init_root(root_move, root_payload);
    }

    /// Promotes `new_root_id` (typically a child of the current root) to become the new
    /// [`NodeId::ROOT`], compacting its surviving subtree into the standby arena and
    /// swapping generations in `O(|surviving_subtree|)`.
    pub fn advance_to(&mut self, new_root_id: NodeId) {
        self.standby.clear();
        let src_root = *self.active.get(new_root_id);
        self.standby.init_root(src_root.mv, src_root.payload);
        Self::copy_children_recursive(&self.active, new_root_id, &mut self.standby, NodeId::ROOT);
        std::mem::swap(&mut self.active, &mut self.standby);
    }

    /// Advances the root to the child matching `mv` if already expanded; otherwise resets
    /// to a fresh unexpanded root `(mv, fallback_payload)`.
    ///
    /// Returns `true` if an existing child subtree was reused.
    pub fn advance_by_move(&mut self, mv: M, fallback_payload: P) -> bool
    where
        M: Eq,
    {
        let found = self
            .active
            .children_with_ids(NodeId::ROOT)
            .find(|(_, node)| node.mv == mv)
            .map(|(id, _)| id);

        if let Some(child_id) = found {
            self.advance_to(child_id);
            true
        } else {
            self.reset_root(mv, fallback_payload);
            false
        }
    }

    fn copy_children_recursive(
        src: &TreeArena<M, P>,
        src_parent: NodeId,
        dst: &mut TreeArena<M, P>,
        dst_parent: NodeId,
    ) {
        let src_children = src.children(src_parent);
        if src_children.is_empty() {
            return;
        }

        let dst_start = dst.nodes.len();
        for child in src_children {
            dst.nodes.push(Node::new(child.mv, child.payload));
        }
        let count = src_children.len();
        let dst_parent_node = &mut dst.nodes[dst_parent.index()];
        dst_parent_node.first_child = NodeId::from_index(dst_start);
        dst_parent_node.child_count = count as u16;

        let src_first = src.get(src_parent).first_child.index();
        for offset in 0..count {
            let src_child_id = NodeId::from_index(src_first + offset);
            if src.get(src_child_id).has_children() {
                let dst_child_id = NodeId::from_index(dst_start + offset);
                Self::copy_children_recursive(src, src_child_id, dst, dst_child_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generational_advance_preserves_surviving_subtree_and_discards_siblings() {
        // Root (0) -> children [mv=10 (payload 1), mv=20 (payload 2), mv=30 (payload 3)]
        // Child mv=20 -> grandchildren [mv=21 (payload 21), mv=22 (payload 22)]
        let mut tree = GenerationalTree::<u8, u32>::new(32, 0, 100);
        let root = tree.root_id();
        tree.active_mut()
            .expand_from_iter(root, [(10, 1), (20, 2), (30, 3)]);
        assert_eq!(tree.active().subtree_count(root), 4);

        let middle_child_id = tree
            .active()
            .children_with_ids(root)
            .find(|(_, n)| n.mv == 20)
            .unwrap()
            .0;
        tree.active_mut()
            .expand_from_iter(middle_child_id, [(21, 21), (22, 22)]);
        assert_eq!(tree.active().subtree_count(root), 6);

        // Advance to mv=20: only mv=20 and its two children [21, 22] survive (3 nodes total).
        let reused = tree.advance_by_move(20, 0);
        assert!(reused);
        assert_eq!(tree.active().len(), 3);
        assert_eq!(tree.root().mv, 20);
        assert_eq!(tree.root().payload, 2);

        let surviving_children = tree.active().children(tree.root_id());
        assert_eq!(surviving_children.len(), 2);
        assert_eq!(surviving_children[0].mv, 21);
        assert_eq!(surviving_children[0].payload, 21);
        assert_eq!(surviving_children[1].mv, 22);
        assert_eq!(surviving_children[1].payload, 22);
    }
}
