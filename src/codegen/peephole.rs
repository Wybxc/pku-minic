//! Peephole optimizations.

use imbl::HashMap;

use crate::codegen::{
    imm::i12,
    riscv::{Block, Inst, RegId},
};

pub fn optimize(block: &mut Block, opt_level: u8) {
    if opt_level >= 1 {
        copy_propagation(block);
        constant_fold(block);
    }
    if opt_level >= 2 {
        copy_propagation(block);
        constant_fold(block);
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
        // do propagation
        match cursor.inst().unwrap() {
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
        match cursor.inst().unwrap() {
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
