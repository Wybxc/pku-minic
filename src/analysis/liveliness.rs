//! Variable liveliness analysis.
//!
//! This module implements a variable liveliness analysis algorithm.
//!
//! # Rules
//! The rules of variable liveliness analysis are:
//!
//! ```text
//! (B -> D) in CFG |- Before[D] subset After[B]
//! |- Before[B] = (After[B] - Defined[B]) | Used[B]
//! ```

use std::collections::{HashMap, VecDeque};

use imbl::HashSet;
use koopa::ir::{dfg::DataFlowGraph, BasicBlock, FunctionData, Value, ValueKind};
#[allow(unused_imports)]
use nolog::*;

use crate::{analysis::cfg::ControlFlowGraph, utils};

/// Variable liveliness analysis result.
pub struct Liveliness {
    /// Live variables at the beginning of each block.
    pub live_in: HashMap<BasicBlock, HashSet<Value>>,
    /// Dead variables after each instruction.
    pub live_out: HashMap<Value, Vec<Value>>,
}

impl Liveliness {
    /// Analyze the liveliness of a function.
    pub fn analysis(cfg: &ControlFlowGraph, func: &FunctionData) -> Self {
        LivelinessAnalyzer::new(cfg, func).analyze()
    }
}

struct LivelinessAnalyzer<'a> {
    func: &'a FunctionData,
    cfg: &'a ControlFlowGraph,
    work_list: VecDeque<BasicBlock>,
    /// Variables used & defined in each block.
    use_defined: HashMap<BasicBlock, (HashSet<Value>, HashSet<Value>)>,
    /// Variables live at the end of each block.
    after: HashMap<BasicBlock, HashSet<Value>>,
    /// Variables live at the beginning of each block.
    live_in: HashMap<BasicBlock, HashSet<Value>>,
    /// Variables live out of each instruction.
    live_out: HashMap<Value, Vec<Value>>,
}

impl<'a> LivelinessAnalyzer<'a> {
    pub fn new(cfg: &'a ControlFlowGraph, func: &'a FunctionData) -> Self {
        let work_list = cfg.sinks().collect();
        Self {
            func,
            cfg,
            work_list,
            use_defined: HashMap::new(),
            after: HashMap::new(),
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        }
    }
    pub fn analyze(mut self) -> Liveliness {
        while let Some(bb) = self.work_list.pop_front() {
            // before = (after - defined) | used
            let after = self.after_mut(bb).clone();
            let (used, defined) = self.analyze_local(bb);
            let before = after.relative_complement(defined).union(used);

            // If changed, add predecessors to work list.
            if self.live_in.get(&bb) != Some(&before) {
                self.live_in.insert(bb, before.clone());
                for pred in self.cfg.preds(bb) {
                    // after = union(before[succ] for succ in successors)
                    self.after_mut(pred).extend(before.clone());
                    // Add to work list.
                    self.work_list.push_back(pred);
                }
            }
        }

        // fix live_out: live_out = live_out - after
        for (&bb, node) in self.func.layout().bbs() {
            let after = self.after_mut(bb).clone();
            for &inst in node.insts().keys() {
                let live_out = self.live_out.get_mut(&inst).unwrap();
                live_out.retain(|val| !after.contains(val));

                trace!("VLA " => "live_out of `{}`: {}", utils::dbg_inst(inst, self.func.dfg()), {
                    live_out
                        .iter()
                        .map(|v| utils::ident_inst(*v, self.func.dfg()))
                        .collect::<Vec<_>>()
                        .join(", ")
                });
            }
        }

        Liveliness {
            live_out: self.live_out,
            live_in: self.live_in,
        }
    }

    /// Local analysis of a basic block.
    ///
    /// This function will calculate the used/defined variables in the block,
    /// and the variables may live out of each instruction.
    ///
    /// Notice that live out information is not correct (it contains false
    /// positives) until the global analysis is done.
    fn analyze_local(&mut self, bb: BasicBlock) -> (HashSet<Value>, HashSet<Value>) {
        // Already analyzed.
        if let Some((used, defined)) = self.use_defined.get(&bb).cloned() {
            return (used, defined);
        }

        // Variables used but not defined in the block.
        let mut used = HashSet::new();
        // Variables defined in the block.
        let mut defined = HashSet::new();

        // In reverse order scan the instructions in a basic block,
        // the last appearing variable is live out.
        let bbs = self.func.layout().bbs();
        let insts = bbs.node(&bb).unwrap().insts();
        let mut cursor = insts.cursor_back();
        while !cursor.is_null() {
            let &inst = cursor.key().unwrap();

            trace!(->[0] "VLA " => "analyzing `{}`", utils::dbg_inst(inst, self.func.dfg()));
            let mut ops = operand_vars(inst, self.func.dfg());
            ops.retain(|val| {
                // Not last use. The variable will not die here.
                used.insert(*val).is_none()
            });
            let value = self.func.dfg().value(inst);
            if !value.ty().is_unit() && !matches!(value.kind(), koopa::ir::ValueKind::Alloc(_)) {
                defined.insert(inst);
                used.remove(&inst);
            }

            self.live_out.insert(inst, ops);

            cursor.move_prev();
        }

        self.use_defined.insert(bb, (used.clone(), defined.clone()));
        (used, defined)
    }

    fn after_mut(&mut self, bb: BasicBlock) -> &mut HashSet<Value> {
        self.after.entry(bb).or_default()
    }
}

/// Get operands of an instruction, only return variables.
fn operand_vars(value: Value, dfg: &DataFlowGraph) -> Vec<Value> {
    let is_var = |v: &Value| dfg.values().get(v).is_some_and(|v| !utils::is_const(v));
    match dfg.value(value).kind() {
        ValueKind::Return(ret) => {
            let val = ret.value().filter(is_var);
            val.iter().copied().collect()
        }
        ValueKind::Binary(bin) => {
            let lhs = bin.lhs();
            let rhs = bin.rhs();
            [Some(lhs).filter(is_var), Some(rhs).filter(is_var)]
                .iter()
                .flatten()
                .copied()
                .collect()
        }
        ValueKind::Store(store) => {
            let val = Some(store.value()).filter(is_var);
            // let ptr = Some(store.dest()).filter(is_var);
            val.iter().copied().collect()
        }
        ValueKind::Load(load) => {
            let addr = Some(load.src()).filter(is_var);
            addr.iter().copied().collect()
        }
        ValueKind::Branch(branch) => {
            let cond = Some(branch.cond()).filter(is_var);
            cond.iter().copied().collect()
        }
        ValueKind::Call(call) => {
            let args = call.args().iter().copied().filter(is_var);
            args.collect()
        }
        _ => vec![],
    }
}
