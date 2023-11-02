//! Peephole optimizations.

use std::collections::HashMap;

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
        reduce_unused_values(block);
    }

    if opt_level >= 2 {
        while reduce_unused_values(block) {}
    }
}

/// Reduce unused values.
///
/// # Example
///
/// ```text
/// mv a4, a1
/// mv a5, a2
/// sub a4, a1, a2
/// ```
///
/// optimizes to
///
/// ```text
/// mv a5, a2
/// sub a4, a1, a2
/// ```
fn reduce_unused_values(block: &mut Block) -> bool {
    let mut dirty = false;
    if let Some(front) = block.front() {
        let mut cursor = block.cursor(front);
        while !cursor.is_null() {
            let inst = cursor.inst().unwrap();
            let mut delete = false; // whether to delete the instruction

            if let Some(dest) = inst.dest() {
                log::debug!("RUV: checking {}", inst);
                for (_, inst) in cursor.followings() {
                    if inst.source().into_iter().flatten().any(|src| src == dest) {
                        // the value is used before re-assigned, so we cannot delete it
                        log::debug!("RUV: {} is used in {}, abort", dest, inst);
                        delete = false;
                        break;
                    }
                    if inst.dest() == Some(dest) {
                        // the value is re-assigned, so we can delete it
                        log::debug!("RUV: {} is re-assigned in {}, delete", dest, inst);
                        delete = true;
                        break;
                    }
                }
            }
            if delete {
                log::debug!("RUV: deleting {}", inst);
                dirty = true;
                cursor.remove_and_next();
            } else {
                cursor.next();
            }
        }
    }
    dirty
}
