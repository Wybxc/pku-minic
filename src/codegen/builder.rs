use crate::codegen::riscv::{Block, Inst};

/// Basic block builder.
pub struct BlockBuilder {
    block: Block,
}

impl BlockBuilder {
    /// Create a new block builder.
    pub fn new(block: Block) -> Self {
        Self { block }
    }

    /// Push an instruction to the block.
    pub fn push(&mut self, instr: Inst) {
        // if let Some(instr) = peephole::optimize_inst_push(instr, self.opt_level, &mut
        // self.cache) {     self.block.push(instr);
        // }
        self.block.push(instr);
    }

    /// Push a sequence of instructions to the block.
    pub fn push_batch(&mut self, instrs: impl IntoIterator<Item = Inst>) {
        for instr in instrs {
            self.push(instr);
        }
    }

    /// Build the block.
    pub fn build(self) -> Block {
        self.block
    }
}
