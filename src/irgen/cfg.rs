//! Control flow graph analysis.

use koopa::ir::{BasicBlock, FunctionData, ValueKind};

use crate::common::cfg::ControlFlowInst;

/// Control flow graph.
pub type ControlFlowGraph = crate::common::cfg::ControlFlowGraph<BasicBlock>;

/// Dominator tree.
pub type Dominators = crate::common::cfg::Dominators<BasicBlock>;

/// Analyze the control flow graph of a function.
pub fn analyze(function: &FunctionData) -> (ControlFlowGraph, Dominators) {
    let cfg = ControlFlowGraph::analyze(
        function.layout().bbs().iter().filter_map(|(&bb, node)| {
            let inst = node.insts().back_key()?;
            let inst = match function.dfg().value(*inst).kind() {
                ValueKind::Branch(branch) => {
                    ControlFlowInst::Branch(branch.true_bb(), branch.false_bb())
                }
                ValueKind::Jump(jump) => ControlFlowInst::Jump(jump.target()),
                ValueKind::Return(_) => ControlFlowInst::Return,
                _ => unreachable!(),
            };
            Some((bb, inst))
        }),
        function
            .layout()
            .entry_bb()
            .expect("function has no basic blocks"),
    );
    let dominators = Dominators::analyze(&cfg);
    (cfg, dominators)
}
