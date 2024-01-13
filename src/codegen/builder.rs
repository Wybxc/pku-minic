use crate::codegen::riscv::{Block, BlockId, Inst};

/// Basic block builder.
pub struct BlockBuilder {
    id: BlockId,
    block: Block,
}

impl BlockBuilder {
    /// Create a new block builder.
    pub fn new(id: BlockId, block: Block) -> Self { Self { id, block } }

    /// Push an instruction to the block.
    pub fn push(&mut self, instr: Inst) {
        // if let Some(instr) = peephole::optimize_inst_push(instr, self.opt_level, &mut
        // self.cache) {     self.block.push(instr);
        // }
        self.block.push(instr);
    }

    /// Build the block.
    pub fn build(self) -> (BlockId, Block) { (self.id, self.block) }
}
