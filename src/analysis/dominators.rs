//! Dominator tree of control flow graph.

use std::collections::{HashMap, HashSet};

use koopa::ir::BasicBlock;
use petgraph::algo::dominators;

use crate::analysis::cfg::ControlFlowGraph;

/// Dominator tree of control flow graph.
pub struct Dominators {
    doms: HashMap<BasicBlock, HashSet<BasicBlock>>,
    entry: BasicBlock,
}

impl Dominators {
    /// Analyze the dominator tree of a control flow graph.
    pub fn analyze(cfg: &ControlFlowGraph) -> Self {
        let mut doms = HashMap::new();

        let graph_dom = dominators::simple_fast(&cfg.graph, cfg.entry);
        for node in cfg.graph.node_indices() {
            if let Some(dom) = graph_dom.immediate_dominator(node) {
                let bb = cfg.graph[node];
                let dom = cfg.graph[dom];
                doms.entry(bb).or_insert_with(HashSet::new).insert(dom);
            }
        }
        let entry = cfg.entry();

        Self { doms, entry }
    }

    /// DFS traversal of the dominator tree.
    pub fn iter(&self) -> impl Iterator<Item = BasicBlock> + '_ {
        let mut visited = HashSet::new();
        let mut stack = vec![self.entry];
        std::iter::from_fn(move || {
            while let Some(bb) = stack.pop() {
                if visited.insert(bb) {
                    if let Some(doms) = self.doms.get(&bb) {
                        stack.extend(doms.iter().copied());
                    }
                    return Some(bb);
                }
            }
            None
        })
    }
}
