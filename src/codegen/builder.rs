use std::collections::HashMap;

use koopa::ir::BasicBlock;

use crate::codegen::{
    peephole::{self, RegCache},
    riscv::{Block, BlockId, Function, Inst},
};

/// Function builder.
pub struct FunctionBuilder {
    function: Function,
    blocks_map: HashMap<BasicBlock, BlockId>,
}

impl FunctionBuilder {
    /// Create a new function builder.
    pub fn new(function: Function) -> Self {
        Self {
            function,
            blocks_map: HashMap::new(),
        }
    }

    /// Push a block to the function.
    pub fn push(&mut self, label: Option<String>, bb: BasicBlock, block: Block) {
        let id = BlockId::next_id();
        self.function.push(id, block);
        self.blocks_map.insert(bb, id);
        if let Some(label) = label {
            id.set_label(label);
        }
    }

    /// Build the function.
    pub fn build(self) -> Function {
        self.function
    }
}

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
    pub fn build(self) -> Block {
        self.block
    }
}
