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
#[allow(unused_imports)]
use nolog::*;

use crate::codegen::riscv::{BlockId, Function, InstId, RegId};

pub type ControlFlowGraph = crate::common::cfg::ControlFlowGraph<BlockId>;
pub type Dominators = crate::common::cfg::Dominators<BlockId>;

/// Variable liveliness analysis result.
pub struct Liveliness {
    /// Live variables at the beginning of each block.
    pub live_in: HashMap<BlockId, HashSet<RegId>>,
    /// Variables live out of each instruction.
    pub live_out: HashMap<InstId, [Option<RegId>; 2]>,
}

impl Liveliness {
    /// Analyze the liveliness of a function.
    pub fn analysis(cfg: &ControlFlowGraph, func: &Function) -> Self {
        LivelinessAnalyzer::new(cfg, func).analyze()
    }
}

struct LivelinessAnalyzer<'a> {
    func: &'a Function,
    cfg: &'a ControlFlowGraph,
    work_list: VecDeque<BlockId>,
    /// Variables used & defined in each block.
    use_defined: HashMap<BlockId, (HashSet<RegId>, HashSet<RegId>)>,
    /// Variables live at the end of each block.
    after: HashMap<BlockId, HashSet<RegId>>,
    /// Variables live at the beginning of each block.
    live_in: HashMap<BlockId, HashSet<RegId>>,
    /// Variables live out of each instruction.
    live_out: HashMap<InstId, [Option<RegId>; 2]>,
}

impl<'a> LivelinessAnalyzer<'a> {
    pub fn new(cfg: &'a ControlFlowGraph, func: &'a Function) -> Self {
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

        // fix liveout
        for (bb, _block) in self.func.iter() {
            let after = self.after.get(&bb).unwrap();

            for (_inst, set) in self.live_out.iter_mut() {
                for op in &mut *set {
                    if let Some(reg) = op {
                        if after.contains(reg) {
                            *op = None;
                        }
                    }
                }

                if let Some(_inst) = _block.get(*_inst) {
                    trace!("VLA " => "live_out of `{}`: {:?}", _inst, set);
                }
            }
        }

        Liveliness {
            live_in: self.live_in,
            live_out: self.live_out,
        }
    }

    /// Local analysis of a basic block.
    ///
    /// This function will calculate the used/defined variables in the block,
    /// and the variables may live out of each instruction.
    ///
    /// Notice that live out information is not correct (it contains false
    /// positives) until the global analysis is done.
    fn analyze_local(&mut self, bb: BlockId) -> (HashSet<RegId>, HashSet<RegId>) {
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

        let block = self.func.get(bb).unwrap();
        let mut cursor = block.cursor(block.back().unwrap());
        while !cursor.is_null() {
            let inst = cursor.inst().unwrap();

            trace!(->[0] "VLA " => "analyzing `{}`", inst);

            if let Some(def) = inst.dest() {
                defined.insert(def);
                used.remove(&def);
            }

            let mut ops = inst.source();
            for op in ops.iter_mut() {
                if let Some(val) = op {
                    // Not last use. The variable will not die here.
                    if used.insert(*val).is_some() {
                        *op = None;
                    }
                }
            }
            // Update liveout
            self.live_out.insert(cursor.id().unwrap(), ops);
            cursor.prev();
        }

        self.use_defined.insert(bb, (used.clone(), defined.clone()));
        (used, defined)
    }

    fn after_mut(&mut self, bb: BlockId) -> &mut HashSet<RegId> {
        self.after.entry(bb).or_default()
    }
}
