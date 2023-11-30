use koopa::ir::{
    builder_traits::{BasicBlockBuilder, LocalInstBuilder},
    dfg::DataFlowGraph,
    layout::BasicBlockNode,
    BasicBlock, FunctionData, Value,
};
use miette::Result;

use crate::ast;

/// Basic block layout builder.
pub struct LayoutBuilder<'a> {
    func: &'a mut FunctionData,
    current: BasicBlock,
    rtype: ast::FuncType,
}

impl<'a> LayoutBuilder<'a> {
    /// Create a new layout builder.
    pub fn new(func: &'a mut FunctionData, rtype: ast::FuncType) -> Self {
        let entry = func.dfg_mut().new_bb().basic_block(Some("%entry".into()));
        func.layout_mut()
            .bbs_mut()
            .push_key_back(entry)
            .expect("basic block id conflict");
        LayoutBuilder {
            func,
            current: entry,
            rtype,
        }
    }

    /// Function data.
    pub fn func_mut(&mut self) -> &mut FunctionData {
        self.func
    }

    /// Basic block node.
    pub fn bb_mut(&mut self, bb: BasicBlock) -> &mut BasicBlockNode {
        self.func.layout_mut().bb_mut(bb)
    }

    /// Current basic block id.
    pub fn current(&self) -> BasicBlock {
        self.current
    }

    /// Current basic block node.
    pub fn current_bb_mut(&mut self) -> &mut BasicBlockNode {
        self.bb_mut(self.current)
    }

    /// Dataflow graph.
    pub fn dfg(&self) -> &DataFlowGraph {
        self.func.dfg()
    }

    /// Dataflow graph.
    pub fn dfg_mut(&mut self) -> &mut DataFlowGraph {
        self.func.dfg_mut()
    }

    /// Push an instruction to the current basic block.
    ///
    /// Return the instruction value.
    pub fn push_inst(&mut self, inst: Value) -> Value {
        self.current_bb_mut()
            .insts_mut()
            .push_key_back(inst)
            .expect("instruction id conflict");
        inst
    }

    /// Push instructions to the current basic block.
    ///
    /// Return the last instruction value.
    pub fn push_insts(&mut self, insts: impl IntoIterator<Item = Value>) -> Value {
        let inst_list = self.current_bb_mut().insts_mut();
        for inst in insts.into_iter() {
            inst_list
                .push_key_back(inst)
                .expect("instruction id conflict");
        }
        inst_list.back_key().copied().unwrap()
    }

    /// Terminate the current basic block.
    ///
    /// If the current basic block is not terminated, insert a default terminator.
    pub fn terminate_current_bb(&mut self) {
        let insts = self.current_bb_mut().insts_mut();
        let last_inst = insts.back_key().copied();
        if last_inst.is_none()
            || last_inst.is_some_and(|inst| !crate::utils::is_terminator(self.dfg().value(inst)))
        {
            let dfg = self.func.dfg_mut();
            let value = self.rtype.default_value(dfg);
            let inst = dfg.new_value().ret(Some(value));
            self.push_inst(inst);
        }
    }

    /// Create a new empty basic block.
    pub fn new_bb(&mut self, name: Option<String>) -> BasicBlock {
        let bb = self.dfg_mut().new_bb().basic_block(name);
        self.func
            .layout_mut()
            .bbs_mut()
            .push_key_back(bb)
            .expect("basic block id conflict");
        bb
    }

    /// Temporarily change the current basic block.
    pub fn with_bb(
        &mut self,
        bb: BasicBlock,
        f: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let old = self.current;
        self.current = bb;
        f(self)?;
        self.current = old;
        Ok(())
    }

    /// Switch current basic block.
    ///
    /// # Safety
    /// The new basic block must not be terminated, and current basic block must be terminated.
    pub fn switch_bb(&mut self, bb: BasicBlock) {
        self.current = bb;
    }
}
