//! Peephole optimizations.

use std::collections::HashMap;

use crate::codegen::riscv::{make_reg_set, RegSet};

use super::imm::i12;
use super::riscv::{Block, Inst, RegId};

/// Basic block builder, with optimizations.
pub struct BlockBuilder {
    pub block: Block,
    opt_level: u8,
    cache: HashMap<RegId, Source>,
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
}

impl BlockBuilder {
    /// Create a new block builder.
    pub fn new(block: Block, opt_level: u8) -> Self {
        Self {
            block,
            opt_level,
            cache: HashMap::new(),
        }
    }

    /// Get a register's source, if known.
    fn source(&self, reg: RegId) -> Option<Source> {
        self.cache.get(&reg).copied()
    }

    /// Push an instruction to the block, with optimizations.
    ///
    /// Optimizations:
    /// - Reuse known values.
    /// - Propagate known values.
    pub fn push(&mut self, mut instr: Inst) {
        // Propagate known values.
        if self.opt_level >= 1 {
            match instr {
                Inst::Mv(dst, rs) => match self.source(rs) {
                    Some(Source::Reg(src)) => {
                        log::debug!("PKV: propagating {} from {}", src, rs);
                        instr = Inst::Mv(dst, src)
                    }
                    Some(Source::Imm(imm)) => {
                        log::debug!("PKV: propagating imm {} from {}", imm, rs);
                        instr = Inst::Li(dst, imm)
                    }
                    None => {}
                },
                Inst::Add(dst, mut rs1, rs2) => {
                    if let Some(Source::Reg(src)) = self.source(rs1) {
                        log::debug!("PKV: propagating {} from {}", src, rs1);
                        rs1 = src;
                    }
                    match self.source(rs2) {
                        Some(Source::Reg(src)) => {
                            log::debug!("PKV: propagating {} from {}", src, rs2);
                            instr = Inst::Add(dst, rs1, src)
                        }
                        Some(Source::Imm(imm)) => {
                            if let Ok(imm) = i12::try_from(imm) {
                                log::debug!("PKV: propagating imm {} from {}", imm, rs2);
                                instr = Inst::Addi(dst, rs1, imm);
                            } else {
                                instr = Inst::Add(dst, rs1, rs2);
                            }
                        }
                        None => {}
                    }
                }
                Inst::Sub(dst, mut rs1, rs2) => {
                    if let Some(Source::Reg(src)) = self.source(rs1) {
                        log::debug!("PKV: propagating {} from {}", src, rs1);
                        rs1 = src;
                    }
                    match self.source(rs2) {
                        Some(Source::Reg(src)) => {
                            log::debug!("PKV: propagating {} from {}", src, rs2);
                            instr = Inst::Sub(dst, rs1, src)
                        }
                        Some(Source::Imm(imm)) => {
                            if let Ok(imm) = i12::try_from(-imm) {
                                log::debug!("PKV: propagating imm {} from {}", -imm.value(), rs2);
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
                        if let Some(Source::Reg(src)) = self.source(*rs) {
                            log::debug!("PKV: propagating {} from {}", src, rs);
                            *rs = src;
                        }
                    }
                }
            }
        }

        // Reuse known values.
        match instr {
            Inst::Li(dst, imm) => {
                if self.source(dst).unwrap_or(Source::Reg(dst)) == Source::Imm(imm) {
                    log::debug!("RKV: reusing imm in {}", dst);
                    return;
                }
                self.cache.insert(dst, Source::Imm(imm));
            }
            Inst::Mv(dst, src) => {
                let src = self.source(src).unwrap_or(Source::Reg(src));
                if self.source(dst).unwrap_or(Source::Reg(dst)) == src {
                    log::debug!("RKV: reusing {:?} in {}", src, dst);
                    return;
                }
                self.cache.insert(dst, src);
            }
            _ => {
                if let Some(dest) = instr.dest() {
                    self.cache.remove(&dest);
                }
            }
        }

        // Clean dirty cache.
        if let Some(dest) = instr.dest() {
            self.cache.retain(|_, source| *source != Source::Reg(dest));
        }

        log::debug!("BLD: push instr: {}", instr);
        self.block.push(instr);
    }

    /// Build the block.
    pub fn build(self) -> Block {
        self.block
    }
}

/// Do peephole optimizations on a block.
pub fn optimize(block: &mut Block, opt_level: u8) {
    if opt_level >= 1 {
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
        let mut cursor = block.cursor(back);
        while !cursor.is_null() {
            let inst = cursor.inst().unwrap();
            if let Some(dest) = inst.dest() {
                if !live.contains(dest) {
                    // The value is not used later.
                    log::debug!("RUV: remove unused value in {}", dest);
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
