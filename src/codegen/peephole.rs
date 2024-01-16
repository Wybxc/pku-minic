//! Peephole optimizations.

use std::collections::VecDeque;

use imbl::HashMap;
use key_node_list::Node;
#[allow(unused_imports)]
use nolog::*;

use crate::codegen::{
    imm::i12,
    riscv::{make_reg_set, Block, BlockId, Function, Inst, InstId, RegId, RegSet},
};

pub fn optimize(function: &mut Function, opt_level: u8) {
    if opt_level >= 1 {
        function.for_each_mut(|_, block| {
            copy_propagation(block);
            constant_fold(block);
        });
        dead_code_elimination(function);
    }
    if opt_level >= 2 {
        for _ in 0..3 {
            function.for_each_mut(|_, block| {
                copy_propagation(block);
                constant_fold(block);
            });
            dead_code_elimination(function);
        }
    }
}

/// Source of the value in a register, if known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Source {
    /// From another register.
    ///
    /// Invariant: the source register does not have a source.
    Reg(RegId),
    /// From an immediate.
    Imm(i32),
    /// Stack pointer with offset.
    Sp(i12),
}

#[derive(Default)]
struct SourceMap {
    map: HashMap<RegId, Source>,
}

impl SourceMap {
    pub fn insert_imm(&mut self, r: RegId, i: i32) {
        self.map.insert(r, Source::Imm(i));
    }

    pub fn insert_reg(&mut self, r: RegId, s: RegId) {
        if let Some(src) = self.map.get(&s) {
            self.map.insert(r, *src);
        } else {
            self.map.insert(r, Source::Reg(s));
        }
    }

    pub fn insert_sp(&mut self, r: RegId, i: i12) {
        if r == RegId::SP {
            return;
        }
        self.map.insert(r, Source::Sp(i));
    }

    pub fn get(&self, r: RegId) -> Option<Source> {
        if r == RegId::X0 {
            return Some(Source::Imm(0));
        }
        self.map.get(&r).copied()
    }

    pub fn make_dirty(&mut self, r: RegId) {
        self.map.remove(&r);
        self.map.retain(|_, &s| s != Source::Reg(r));
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

pub fn copy_propagation(block: &mut Block) {
    let Some(front) = block.front() else { return };
    let mut cursor = block.cursor_mut(front);
    let mut source = SourceMap::default();

    while !cursor.is_null() {
        // remove redundant inst
        let inst = cursor.inst().unwrap();
        match inst {
            Inst::Li(r, i) => {
                if source.get(r) == Some(Source::Imm(i)) {
                    cursor.remove_and_next();
                    continue;
                }
            }
            Inst::Mv(r, s) => {
                if r == s || source.get(r) == Some(Source::Reg(s)) {
                    cursor.remove_and_next();
                    continue;
                }
            }
            _ => {}
        }

        // do propagation
        match inst {
            Inst::Mv(r, s) => match source.get(s) {
                Some(Source::Imm(i)) => cursor.set_inst(Inst::Li(r, i)),
                Some(Source::Reg(s)) => cursor.set_inst(Inst::Mv(r, s)),
                Some(Source::Sp(i)) => cursor.set_inst(Inst::Addi(r, RegId::SP, i)),
                _ => {}
            },
            Inst::Add(r, s1, s2) => match (source.get(s1), source.get(s2)) {
                (Some(Source::Imm(i)), _) => {
                    if let Ok(i) = i12::try_from(i) {
                        cursor.set_inst(Inst::Addi(r, s2, i))
                    }
                }
                (_, Some(Source::Imm(i))) => {
                    if let Ok(i) = i12::try_from(i) {
                        cursor.set_inst(Inst::Addi(r, s1, i))
                    }
                }
                (Some(Source::Reg(s)), _) => cursor.set_inst(Inst::Add(r, s, s2)),
                (_, Some(Source::Reg(s))) => cursor.set_inst(Inst::Add(r, s1, s)),
                _ => {}
            },
            Inst::Sw(r, i, s) if i.value() == 0 => {
                if let Some(Source::Sp(i)) = source.get(s) {
                    cursor.set_inst(Inst::Sw(r, i, RegId::SP))
                }
            }
            Inst::Lw(r, i, s) if i.value() == 0 => {
                if let Some(Source::Sp(i)) = source.get(s) {
                    cursor.set_inst(Inst::Lw(r, i, RegId::SP))
                }
            }
            _ => {}
        }
        let mut inst = cursor.inst().unwrap();
        for src in inst.source_mut().into_iter().flatten() {
            if let Some(Source::Reg(s)) = source.get(*src) {
                *src = s;
            }
        }
        cursor.set_inst(inst);
        let inst = cursor.inst().unwrap();

        // make dirty if dest is overwritten
        if let Some(dest) = inst.dest() {
            source.make_dirty(dest);
        }
        if matches!(inst, Inst::Call(_)) {
            source.clear();
        }

        // record new inst
        match inst {
            Inst::Li(r, i) => {
                source.insert_imm(r, i);
            }
            Inst::Mv(r, s) => {
                source.insert_reg(r, s);
            }
            Inst::Addi(r, s, i) if i.value() == 0 => {
                source.insert_reg(r, s);
            }
            Inst::Addi(r, RegId::SP, i) => {
                source.insert_sp(r, i);
            }
            Inst::Addi(r, s, i) => {
                if let Some(Source::Sp(k)) = source.get(s) {
                    let i = i.value() + k.value();
                    if let Ok(i) = i12::try_from(i) {
                        source.insert_sp(r, i);
                    }
                }
            }
            _ => {}
        }
        cursor.next();
    }
}

#[derive(Default)]
struct ConstantMap {
    map: HashMap<RegId, i32>,
}

impl ConstantMap {
    pub fn insert(&mut self, r: RegId, i: i32) {
        self.map.insert(r, i);
    }

    pub fn get(&self, r: RegId) -> Option<i32> {
        if r == RegId::X0 {
            return Some(0);
        }
        self.map.get(&r).copied()
    }

    pub fn make_dirty(&mut self, r: RegId) {
        self.map.remove(&r);
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[allow(clippy::single_match)]
pub fn constant_fold(block: &mut Block) {
    let Some(front) = block.front() else { return };
    let mut cursor = block.cursor_mut(front);
    let mut constant = ConstantMap::default();

    while !cursor.is_null() {
        // do constant fold
        let inst = cursor.inst().unwrap();
        match inst {
            Inst::Mv(r, s) => {
                if let Some(i) = constant.get(s) {
                    cursor.set_inst(Inst::Li(r, i));
                }
            }
            Inst::Addi(r, s, i) => {
                if let Some(i1) = constant.get(s) {
                    cursor.set_inst(Inst::Li(r, i1.wrapping_add(i.value())));
                }
            }
            Inst::Add(r, s1, s2) => match (constant.get(s1), constant.get(s2)) {
                (Some(i1), Some(i2)) => cursor.set_inst(Inst::Li(r, i1.wrapping_add(i2))),
                _ => {}
            },
            Inst::Sub(r, s1, s2) => match (constant.get(s1), constant.get(s2)) {
                (Some(i1), Some(i2)) => cursor.set_inst(Inst::Li(r, i1.wrapping_sub(i2))),
                _ => {}
            },
            Inst::Mul(r, s1, s2) => match (constant.get(s1), constant.get(s2)) {
                (Some(0), _) | (_, Some(0)) => cursor.set_inst(Inst::Li(r, 0)),
                (Some(i1), Some(i2)) => cursor.set_inst(Inst::Li(r, i1.wrapping_mul(i2))),
                _ => {}
            },
            Inst::Div(r, s1, s2) => match (constant.get(s1), constant.get(s2)) {
                (_, Some(0)) => {}
                (Some(i1), Some(i2)) => cursor.set_inst(Inst::Li(r, i1.wrapping_div(i2))),
                _ => {}
            },
            _ => {}
        }
        let inst = cursor.inst().unwrap();

        // make dirty if dest is overwritten
        if let Some(dest) = inst.dest() {
            constant.make_dirty(dest);
        }
        if matches!(inst, Inst::Call(_)) {
            constant.clear();
        }

        // record new inst
        match inst {
            Inst::Li(r, i) => {
                constant.insert(r, i);
            }
            _ => {}
        }
        cursor.next();
    }
}

pub struct Liveliness<'a> {
    function: &'a Function,
    cfg_pred: HashMap<BlockId, Vec<(BlockId, InstId)>>,
    /// Work list of instructions to process.
    work_list: VecDeque<(BlockId, InstId)>,
    /// Variables live after the instruction.
    after: HashMap<InstId, RegSet>,
    /// Variables live before the instruction.
    before: HashMap<InstId, RegSet>,
}

impl<'a> Liveliness<'a> {
    pub fn new(function: &'a Function) -> Self {
        let mut cfg_pred = HashMap::new();
        let mut sinks = vec![];

        for (bb, block) in function.iter() {
            for (id, inst) in block.insts() {
                match inst {
                    Inst::J(tgt)
                    | Inst::Beq(_, _, tgt)
                    | Inst::Beqz(_, tgt)
                    | Inst::Bge(_, _, tgt)
                    | Inst::Bgeu(_, _, tgt)
                    | Inst::Bgez(_, tgt)
                    | Inst::Bgt(_, _, tgt)
                    | Inst::Bgtu(_, _, tgt)
                    | Inst::Bgtz(_, tgt)
                    | Inst::Ble(_, _, tgt)
                    | Inst::Bleu(_, _, tgt)
                    | Inst::Blez(_, tgt)
                    | Inst::Blt(_, _, tgt)
                    | Inst::Bltu(_, _, tgt)
                    | Inst::Bltz(_, tgt)
                    | Inst::Bne(_, _, tgt)
                    | Inst::Bnez(_, tgt) => {
                        cfg_pred.entry(tgt).or_insert_with(Vec::new).push((bb, id));
                    }
                    Inst::Ret => {
                        sinks.push((bb, id));
                    }
                    _ => {}
                }
            }
        }

        let mut after = HashMap::new();
        for &(_, id) in sinks.iter() {
            after.insert(id, RegSet::from_bitset(make_reg_set!(A0, SP, RA)));
        }

        Self {
            function,
            cfg_pred,
            work_list: sinks.into_iter().collect(),
            after,
            before: HashMap::new(),
        }
    }

    pub fn analyze(mut self) -> HashMap<InstId, RegSet> {
        while let Some((bb, id)) = self.work_list.pop_front() {
            // before = (after - defined) | used
            let after = *self.after_mut(id);
            let (used, defined) = self.analyze_local(bb, id);
            let before = (after - defined) | used;

            // If changed, add predecessors to work list.
            if self.before.get(&id) != Some(&before) {
                self.before.insert(id, before);
                for (bb, pred) in self.preds(bb, id).unwrap() {
                    // after = union(before[succ] for succ in successors)
                    *self.after_mut(pred) |= before;
                    // Add to work list.
                    self.work_list.push_back((bb, pred));
                }
            }
        }

        self.after
    }

    fn analyze_local(&self, bb: BlockId, id: InstId) -> (RegSet, RegSet) {
        let inst = self.function.get(bb).unwrap().get(id).unwrap();
        if let Inst::Call(_) = inst {
            const USED: u32 = make_reg_set!(A0, A1, A2, A3, A4, A5, A6, A7);
            const DEFINED: u32 = make_reg_set!(A0);
            let used = RegSet::from_bitset(USED);
            let defined = RegSet::from_bitset(DEFINED);
            return (used, defined);
        }
        let mut used = RegSet::new();
        let mut defined = RegSet::new();
        for src in inst.source().iter().flatten() {
            used.insert(*src);
        }
        if let Some(dest) = inst.dest() {
            defined.insert(dest);
        }
        (used, defined)
    }

    fn preds(&self, bb: BlockId, id: InstId) -> Option<Vec<(BlockId, InstId)>> {
        let mut preds = vec![];
        // previous instr in the same block
        let mut prev = self
            .function
            .get(bb)?
            .get_node(id)?
            .prev()
            .map(|&id| (bb, id));
        if prev.is_none() {
            // if this is the first instr in the block, add predecessors
            // from cfg
            if let Some(cfg_pred) = self.cfg_pred.get(&bb) {
                preds.extend(cfg_pred.iter().copied());
            }
            // search for the last instr in the previous block
            prev = self
                .function
                .get_node(bb)?
                .prev()
                .and_then(|&bb| Some((bb, self.function.get(bb)?.back()?)));
        }
        // if the previous instr is a jump, ignore it
        if let Some((bb, id)) = prev {
            if matches!(self.function.get(bb)?.get(id)?, Inst::J(_)) {
                prev = None;
            }
        }
        preds.extend(prev);
        Some(preds)
    }

    fn after_mut(&mut self, id: InstId) -> &mut RegSet {
        self.after.entry(id).or_insert(RegSet::new())
    }
}

pub fn dead_code_elimination(function: &mut Function) {
    trace!(->[0] "DCE " => "function:");
    let liveliness = Liveliness::new(function);
    let after = liveliness.analyze();

    function.for_each_mut(|_, block| {
        let Some(front) = block.front() else { return };
        let mut cursor = block.cursor_mut(front);
        while !cursor.is_null() {
            let id = cursor.id().unwrap();
            let inst = cursor.inst().unwrap();
            if let Some(after) = after.get(&id) {
                if let Some(dest) = inst.dest() {
                    trace!("DCE " => "live_out of `{}`: {}", inst, {
                        after
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    });
                    if !after.contains(dest) {
                        cursor.remove_and_next();
                        continue;
                    }
                }
            }
            cursor.next();
        }
    });
}
