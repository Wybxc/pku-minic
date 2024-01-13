//! Control flow graph and dominator tree.

use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

#[allow(unused_imports)]
use nolog::*;
use petgraph::{
    algo::dominators,
    graph::{DiGraph, NodeIndex},
};

/// Control flow graph.
pub struct ControlFlowGraph<BB> {
    graph: DiGraph<BB, Edge>,
    bb_map: HashMap<BB, NodeIndex>,
    entry: NodeIndex,
    exits: Vec<NodeIndex>,
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

pub enum ControlFlowInst<BB> {
    Branch(BB, BB),
    Jump(BB),
    Return,
}

impl<BB: Eq + Copy + Hash> ControlFlowGraph<BB> {
    /// Analyze the control flow graph of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze(bbs: impl IntoIterator<Item = (BB, ControlFlowInst<BB>)>, entry: BB) -> Self {
        let bbs = bbs.into_iter().collect::<Vec<_>>();
        let nodes = bbs.len();
        let mut graph = DiGraph::with_capacity(nodes, nodes);
        let mut bb_map = HashMap::new();
        let mut exits = Vec::new();

        // Add the nodes.
        for (bb, _) in bbs.iter() {
            let node = graph.add_node(*bb);
            bb_map.insert(*bb, node);
        }

        // Get the entry node.
        let entry = bb_map[&entry];

        // Add the edges.
        for (bb, inst) in bbs {
            let node = bb_map[&bb];
            match inst {
                ControlFlowInst::Branch(true_bb, false_bb) => {
                    let true_node = bb_map[&true_bb];
                    let false_node = bb_map[&false_bb];
                    graph.add_edge(node, true_node, Edge::True);
                    graph.add_edge(node, false_node, Edge::False);
                    trace!(->[0] "CFG " => "add edge {} -T> {}", node.index(), true_node.index());
                    trace!(->[0] "CFG " => "add edge {} -F> {}", node.index(), false_node.index());
                }
                ControlFlowInst::Jump(target) => {
                    let jump_node = bb_map[&target];
                    graph.add_edge(node, jump_node, Edge::Unconditional);
                    trace!(->[0] "CFG " => "add edge {} --> {}", node.index(), jump_node.index());
                }
                ControlFlowInst::Return => exits.push(node),
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
    pub fn entry(&self) -> BB { self.graph[self.entry] }

    /// Get the exit nodes.
    pub fn exits(&self) -> impl Iterator<Item = BB> + '_ {
        self.exits.iter().map(move |&node| self.graph[node])
    }

    /// Get the sinks of the graph, i.e. nodes with no outgoing edges.
    pub fn sinks(&self) -> impl Iterator<Item = BB> + '_ {
        self.graph
            .externals(petgraph::Direction::Outgoing)
            .map(move |node| self.graph[node])
    }

    /// Get the predecessors of a basic block.
    pub fn preds(&self, bb: BB) -> impl Iterator<Item = BB> + '_ {
        self.graph
            .neighbors_directed(self.bb_map[&bb], petgraph::Direction::Incoming)
            .map(move |node| self.graph[node])
    }

    /// Get the successors of a basic block.
    pub fn succs(&self, bb: BB) -> impl Iterator<Item = BB> + '_ {
        self.graph
            .neighbors_directed(self.bb_map[&bb], petgraph::Direction::Outgoing)
            .map(move |node| self.graph[node])
    }
}

/// Dominator tree of control flow graph.
pub struct Dominators<BB> {
    doms: HashMap<BB, Vec<BB>>,
    entry: BB,
}

impl<BB: Eq + Copy + Hash> Dominators<BB> {
    /// Analyze the dominator tree of a control flow graph.
    pub fn analyze(cfg: &ControlFlowGraph<BB>) -> Self {
        let mut doms = HashMap::new();

        let graph_dom = dominators::simple_fast(&cfg.graph, cfg.entry);
        for node in cfg.graph.node_indices() {
            if let Some(dom) = graph_dom.immediate_dominator(node) {
                trace!(->[0] "DOM " => "Domination: {} -> {}", dom.index(), node.index());
                let bb = cfg.graph[node];
                let dom = cfg.graph[dom];
                doms.entry(dom).or_insert_with(Vec::new).push(bb);
            }
        }
        let entry = cfg.entry();

        Self { doms, entry }
    }

    /// DFS traversal of the dominator tree.
    pub fn iter(&self) -> impl Iterator<Item = BB> + '_ {
        let mut visited = HashSet::new();
        let mut stack = vec![self.entry];
        std::iter::from_fn(move || {
            while let Some(bb) = stack.pop() {
                if visited.insert(bb) {
                    if let Some(doms) = self.doms.get(&bb) {
                        stack.extend(doms.iter().rev().copied());
                    }
                    return Some(bb);
                }
            }
            None
        })
    }
}
