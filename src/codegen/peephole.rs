//! Peephole optimizations.

use std::collections::HashMap;

#[allow(unused_imports)]
use nolog::*;

use super::imm::i12;
use crate::codegen::riscv::{make_reg_set, Block, Inst, RegId, RegSet};

/// Source of the value in a register, if known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    /// From another register.
    ///
    /// Invariant: the source register does not have a source.
    Reg(RegId),
    /// From an immediate.
    Imm(i32),
}

/// Register cache.
pub struct RegCache {
    cache: HashMap<RegId, Source>,
}

impl RegCache {
    /// Create a new register cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get the source of the given register, if known.
    pub fn source(&self, reg: RegId) -> Option<Source> { self.cache.get(&reg).copied() }

    /// Insert a source for the given register.
    pub fn insert(&mut self, reg: RegId, source: Source) { self.cache.insert(reg, source); }

    /// Remove the source of the given register.
    pub fn remove(&mut self, reg: &RegId) { self.cache.remove(reg); }

    /// Retain only the elements specified by the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&RegId, &mut Source) -> bool,
    {
        self.cache.retain(f);
    }
}

/// Push an instruction to the block, with optimizations.
///
/// Optimizations:
/// - Reuse known values.
/// - Propagate known values.
///
/// Returns `None` if the instruction is redundant.
/// Returns `Some(instr)` when the given instruction is optimized to `instr`.
pub fn optimize_inst_push(mut instr: Inst, opt_level: u8, cache: &mut RegCache) -> Option<Inst> {
    if opt_level >= 1 {
        match instr {
            Inst::Mv(dst, rs) => match cache.source(rs) {
                Some(Source::Reg(src)) => {
                    trace!("PKV " => "propagating {} from {}", src, rs);
                    instr = Inst::Mv(dst, src)
                }
                Some(Source::Imm(imm)) => {
                    trace!("PKV " => "propagating imm {} from {}", imm, rs);
                    instr = Inst::Li(dst, imm)
                }
                None => {}
            },
            Inst::Add(dst, mut rs1, rs2) => {
                if let Some(Source::Reg(src)) = cache.source(rs1) {
                    trace!("PKV " => "propagating {} from {}", src, rs1);
                    rs1 = src;
                }
                match cache.source(rs2) {
                    Some(Source::Reg(src)) => {
                        trace!("PKV " => "propagating {} from {}", src, rs2);
                        instr = Inst::Add(dst, rs1, src)
                    }
                    Some(Source::Imm(imm)) => {
                        if let Ok(imm) = i12::try_from(imm) {
                            trace!("PKV " => "propagating imm {} from {}", imm, rs2);
                            instr = Inst::Addi(dst, rs1, imm);
                        } else {
                            instr = Inst::Add(dst, rs1, rs2);
                        }
                    }
                    None => {}
                }
            }
            Inst::Sub(dst, mut rs1, rs2) => {
                if let Some(Source::Reg(src)) = cache.source(rs1) {
                    trace!("PKV " => "propagating {} from {}", src, rs1);
                    rs1 = src;
                }
                match cache.source(rs2) {
                    Some(Source::Reg(src)) => {
                        trace!("PKV " => "propagating {} from {}", src, rs2);
                        instr = Inst::Sub(dst, rs1, src)
                    }
                    Some(Source::Imm(imm)) => {
                        if let Ok(imm) = i12::try_from(-imm) {
                            trace!("PKV " => "propagating imm {} from {}", -imm.value(), rs2);
                            instr = Inst::Addi(dst, rs1, imm);
                        } else {
                            instr = Inst::Sub(dst, rs1, rs2);
                        }
                    }
                    None => {}
                }
            }
            _ => {
                for rs in instr.source_mut().into_iter().flatten() {
                    if let Some(Source::Reg(src)) = cache.source(*rs) {
                        trace!("PKV " => "propagating {} from {}", src, rs);
                        *rs = src;
                    }
                }
            }
        }
    }

    // Reuse known values.
    match instr {
        Inst::Li(dst, imm) => {
            if cache.source(dst).unwrap_or(Source::Reg(dst)) == Source::Imm(imm) {
                trace!("RKV " => "reusing imm in {}", dst);
                return None;
            }
            cache.insert(dst, Source::Imm(imm));
        }
        Inst::Mv(dst, src) => {
            let src = cache.source(src).unwrap_or(Source::Reg(src));
            if cache.source(dst).unwrap_or(Source::Reg(dst)) == src {
                trace!("RKV " => "reusing {:?} in {}", src, dst);
                return None;
            }
            cache.insert(dst, src);
        }
        _ => {
            if let Some(dest) = instr.dest() {
                cache.remove(&dest);
            }
        }
    }

    // Clean dirty cache.
    if let Some(dest) = instr.dest() {
        cache.retain(|_, source| *source != Source::Reg(dest));
    }

    Some(instr)
}

/// Do peephole optimizations on a block.
pub fn optimize(block: &mut Block, opt_level: u8) {
    if opt_level >= 1 {
        info!(->[0,1] "OPT " => "Peephole optimization, pass 1");

        const LIVE: RegSet = RegSet::from_bitset(make_reg_set!(
            RA, SP, GP, TP, A0, S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11
        ));
        reduce_unused_values(block, LIVE);
    }
}

/// Reduce unused values.
///
/// # Example
///
/// ```text
/// mv a4, a1
/// mv a5, a2
/// sub a0, a1, a2
/// ret
/// ```
///
/// optimizes to
///
/// ```text
/// sub a0, a1, a2
/// ret
/// ```
fn reduce_unused_values(block: &mut Block, mut live: RegSet) -> bool {
    let mut dirty = false;
    if let Some(back) = block.back() {
        let mut cursor = block.cursor_mut(back);
        while !cursor.is_null() {
            let inst = cursor.inst().unwrap();
            trace!(->[0] "RUV " => "analyzing `{}`", inst);
            if let Some(dest) = inst.dest() {
                if !live.contains(dest) {
                    // The value is not used later.
                    trace!("RUV " => "remove unused value in {}", dest);
                    cursor.remove_and_prev();
                    dirty = true;
                    continue;
                } else {
                    // The value is used later, but assigns killed it
                    live.remove(dest);
                }
            }
            for rs in inst.source().into_iter().flatten() {
                // The value is used here, make it live.
                live.insert(rs);
            }
            cursor.prev();
        }
    }
    dirty
}
