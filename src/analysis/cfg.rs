//! Control flow graph.

use std::collections::HashMap;

use koopa::ir::{BasicBlock, FunctionData, ValueKind};
#[allow(unused_imports)]
use nolog::*;
use petgraph::graph::{DiGraph, NodeIndex};

/// Control flow graph.
pub struct ControlFlowGraph {
    pub(super) graph: DiGraph<BasicBlock, Edge>,
    pub(super) bb_map: HashMap<BasicBlock, NodeIndex>,
    pub(super) entry: NodeIndex,
    pub(super) exits: Vec<NodeIndex>,
}

/// Edge type in the control flow graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// If-true edge.
    True,
    /// If-false edge.
    False,
    /// Unconditional edge.
    Unconditional,
}

impl ControlFlowGraph {
    /// Analyze the control flow graph of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze(func: &FunctionData) -> Self {
        let nodes = func.layout().bbs().len();
        let entry = func
            .layout()
            .entry_bb()
            .expect("Function has no basic blocks");
        let mut graph = DiGraph::with_capacity(nodes, nodes);
        let mut bb_map = HashMap::new();
        let mut exits = Vec::new();

        // Add the nodes.
        for &bb in func.layout().bbs().keys() {
            let node = graph.add_node(bb);
            bb_map.insert(bb, node);
        }

        // Get the entry node.
        let entry = bb_map[&entry];

        // Add the edges.
        for (&bb, bb_node) in func.layout().bbs() {
            let node = bb_map[&bb];
            if let Some(&inst) = bb_node.insts().back_key() {
                match func.dfg().value(inst).kind() {
                    ValueKind::Branch(branch) => {
                        let true_node = bb_map[&branch.true_bb()];
                        let false_node = bb_map[&branch.false_bb()];
                        graph.add_edge(node, true_node, Edge::True);
                        graph.add_edge(node, false_node, Edge::False);
                        trace!(->[0] "CFG " => "add edge {} -T> {}", node.index(), true_node.index());
                        trace!(->[0] "CFG " => "add edge {} -F> {}", node.index(), false_node.index());
                    }
                    ValueKind::Jump(jump) => {
                        let jump_node = bb_map[&jump.target()];
                        graph.add_edge(node, jump_node, Edge::Unconditional);
                        trace!(->[0] "CFG " => "add edge {} --> {}", node.index(), jump_node.index());
                    }
                    ValueKind::Return(_) => exits.push(node),
                    _ => {}
                }
            }
        }

        trace!("CFG " => "entry: {}", entry.index());
        trace!("CFG " => "exits: {:?}", exits.iter().map(|&n| n.index()).collect::<Vec<_>>());
        Self {
            graph,
            bb_map,
            entry,
            exits,
        }
    }

    /// Get the entry node.
    pub fn entry(&self) -> BasicBlock { self.graph[self.entry] }

    /// Get the exit nodes.
    pub fn exits(&self) -> impl Iterator<Item = BasicBlock> + '_ {
        self.exits.iter().map(move |&node| self.graph[node])
    }

    /// Get the sinks of the graph, i.e. nodes with no outgoing edges.
    pub fn sinks(&self) -> impl Iterator<Item = BasicBlock> + '_ {
        self.graph
            .externals(petgraph::Direction::Outgoing)
            .map(move |node| self.graph[node])
    }

    /// Get the predecessors of a basic block.
    pub fn preds(&self, bb: BasicBlock) -> impl Iterator<Item = BasicBlock> + '_ {
        self.graph
            .neighbors_directed(self.bb_map[&bb], petgraph::Direction::Incoming)
            .map(move |node| self.graph[node])
    }

    /// Get the successors of a basic block.
    pub fn succs(&self, bb: BasicBlock) -> impl Iterator<Item = BasicBlock> + '_ {
        self.graph
            .neighbors_directed(self.bb_map[&bb], petgraph::Direction::Outgoing)
            .map(move |node| self.graph[node])
    }
}
