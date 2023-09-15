//! Build IR from AST.

use crate::ast;
use koopa::ir::builder_traits::*;
use koopa::ir::*;

impl ast::CompUnit {
    /// Build IR from AST.
    pub fn build_ir(self) -> Program {
        let mut program = Program::new();
        self.build_ir_in(&mut program);
        program
    }

    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, program: &mut Program) {
        self.func_def.build_ir_in(program);
    }
}

impl ast::FuncDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, program: &mut Program) {
        let func = program.new_func(FunctionData::with_param_names(
            format!("@{}", self.ident),
            vec![],
            self.func_type.build_ir(),
        ));
        let func_data = program.func_mut(func);

        self.block.build_ir_in(func_data);
    }
}

impl ast::FuncType {
    /// Build IR from AST.
    pub fn build_ir(self) -> Type {
        match self {
            ast::FuncType::Int => Type::get_i32(),
        }
    }
}

impl ast::Block {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, func: &mut FunctionData) {
        let entry = func.dfg_mut().new_bb().basic_block(Some("%entry".into()));
        func.layout_mut().bbs_mut().extend([entry]);

        self.stmt.build_ir_in(func, entry);
    }
}

impl ast::Stmt {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, func: &mut FunctionData, block: BasicBlock) {
        match self {
            ast::Stmt::Return { num } => {
                let dfg = func.dfg_mut();
                let num = dfg.new_value().integer(num);
                let ret = dfg.new_value().ret(Some(num));

                func.layout_mut().bb_mut(block).insts_mut().extend([ret]);
            }
        }
    }
}
