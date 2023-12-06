use crate::codegen::{
    peephole::{self, RegCache},
    riscv::{Block, Inst},
};

/// Basic block builder.
pub struct BlockBuilder {
    block: Block,
    opt_level: u8,
    cache: RegCache,
}

impl BlockBuilder {
    /// Create a new block builder.
    pub fn new(block: Block, opt_level: u8) -> Self {
        Self {
            block,
            opt_level,
            cache: RegCache::new(),
        }
    }

    /// Push an instruction to the block.
    pub fn push(&mut self, instr: Inst) {
        if let Some(instr) = peephole::optimize_inst_push(instr, self.opt_level, &mut self.cache) {
            self.block.push(instr);
        }
    }

    /// Build the block.
    pub fn build(self) -> Block { self.block }
}
