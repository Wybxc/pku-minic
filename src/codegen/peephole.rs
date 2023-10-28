//! Peephole optimizations.

use std::collections::HashMap;

use super::imm::i12;
use super::riscv::{Block, Inst, RegId};

/// Basic block builder, with optimizations.
pub struct BlockBuilder {
    pub block: Block,
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
    pub fn new(block: Block) -> Self {
        Self {
            block,
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
        // Reuse known values.
        match instr {
            Inst::Li(dst, imm) => {
                if self.source(dst).unwrap_or(Source::Reg(dst)) == Source::Imm(imm) {
                    return;
                }
                self.cache.insert(dst, Source::Imm(imm));
            }
            Inst::Mv(dst, src) => {
                let src = self.source(src).unwrap_or(Source::Reg(src));
                if self.source(dst).unwrap_or(Source::Reg(dst)) == src {
                    return;
                }
                self.cache.insert(dst, src);

                // Propagate the source.
                match src {
                    Source::Reg(src) => instr = Inst::Mv(dst, src),
                    Source::Imm(imm) => instr = Inst::Li(dst, imm),
                }
            }
            Inst::Add(dst, mut rs1, rs2) => {
                // Propagate the source.
                if let Some(Source::Reg(src)) = self.source(rs1) {
                    rs1 = src;
                }
                match self.source(rs2) {
                    Some(Source::Reg(src)) => instr = Inst::Add(dst, rs1, src),
                    Some(Source::Imm(imm)) => {
                        if let Ok(imm) = i12::try_from(imm) {
                            instr = Inst::Addi(dst, rs1, imm);
                        } else {
                            instr = Inst::Add(dst, rs1, rs2);
                        }
                    }
                    None => {}
                }
                self.cache.remove(&dst);
            }
            Inst::Sub(dst, mut rs1, rs2) => {
                // Propagate the source.
                if let Some(Source::Reg(src)) = self.source(rs1) {
                    rs1 = src;
                }
                match self.source(rs2) {
                    Some(Source::Reg(src)) => instr = Inst::Sub(dst, rs1, src),
                    Some(Source::Imm(imm)) => {
                        if let Ok(imm) = i12::try_from(-imm) {
                            instr = Inst::Addi(dst, rs1, imm);
                        } else {
                            instr = Inst::Sub(dst, rs1, rs2);
                        }
                    }
                    None => {}
                }
                self.cache.remove(&dst);
            }
            _ => {
                for rs in instr.source_mut().into_iter().flatten() {
                    if let Some(Source::Reg(src)) = self.source(*rs) {
                        *rs = src;
                    }
                }
                if let Some(dest) = instr.dest() {
                    self.cache.remove(&dest);
                }
            }
        }

        // Clean dirty cache.
        if let Some(dest) = instr.dest() {
            self.cache.retain(|_, source| *source != Source::Reg(dest));
        }

        self.block.push(instr);
    }

    /// Build the block.
    pub fn build(self) -> Block {
        self.block
    }
}
