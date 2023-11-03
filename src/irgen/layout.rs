use koopa::ir::{
    builder_traits::{BasicBlockBuilder, LocalInstBuilder},
    dfg::DataFlowGraph,
    layout::BasicBlockNode,
    BasicBlock, FunctionData, Value,
};

use crate::ast;

/// Basic block layout builder.
pub struct LayoutBuilder<'a> {
    func: &'a mut FunctionData,
    current: BasicBlock,
    rtype: ast::FuncType,
    new_bb: Option<Option<String>>,
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
            new_bb: None,
        }
    }

    /// Function data.
    pub fn func(&mut self) -> &mut FunctionData {
        self.func
    }

    /// Current basic block layout.
    pub fn current(&mut self) -> &mut BasicBlockNode {
        self.func.layout_mut().bb_mut(self.current)
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
        // Open a new basic block if needed.
        self.try_open_bb();

        self.current()
            .insts_mut()
            .push_key_back(inst)
            .expect("instruction id conflict");
        inst
    }

    /// Push instructions to the current basic block.
    ///
    /// Return the last instruction value.
    pub fn push_insts(&mut self, insts: impl IntoIterator<Item = Value>) -> Value {
        // Open a new basic block if needed.
        self.try_open_bb();

        let inst_list = self.current().insts_mut();
        for inst in insts.into_iter() {
            inst_list
                .push_key_back(inst)
                .expect("instruction id conflict");
        }
        inst_list.back_key().copied().unwrap()
    }

    /// Open a new basic block.
    fn try_open_bb(&mut self) {
        if let Some(name) = self.new_bb.take() {
            let bb = self.dfg_mut().new_bb().basic_block(name);
            self.func
                .layout_mut()
                .bbs_mut()
                .push_key_back(bb)
                .expect("basic block id conflict");
            self.current = bb;
        }
    }

    /// Close the current basic block.
    fn try_close_bb(&mut self) {
        if self.new_bb.is_none() {
            // Check if the current basic block is terminated.
            // If not, add a return instruction.
            let insts = self.current().insts_mut();
            let last_inst = insts.back_key().copied();
            if last_inst.is_none()
                || last_inst
                    .is_some_and(|inst| !crate::irutils::is_terminator(self.dfg().value(inst)))
            {
                let dfg = self.func.dfg_mut();
                let value = self.rtype.default_value(dfg);
                let inst = dfg.new_value().ret(Some(value));
                self.push_inst(inst);
            }
        }
    }

    /// Open a new basic block, return the old one.
    pub fn new_bb(&mut self, name: Option<String>) -> BasicBlock {
        // Close the current basic block.
        self.try_close_bb();

        // Open a new basic block.
        self.new_bb = Some(name);
        self.current
    }
}

impl Drop for LayoutBuilder<'_> {
    fn drop(&mut self) {
        // Close the current basic block.
        self.try_close_bb();
    }
}
