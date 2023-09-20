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
            ast::Stmt::Return { expr } => {
                let expr = expr.build_ir_in(func, block);
                let dfg = func.dfg_mut();
                let ret = dfg.new_value().ret(Some(expr));

                push_inst(func, block, ret);
            }
        }
    }
}

impl ast::Expr {
    /// Build IR from AST in a BasicBlock, return the handle of the result value.
    pub fn build_ir_in(self, func: &mut FunctionData, block: BasicBlock) -> Value {
        match self {
            ast::Expr::Unary { op, expr } => {
                let expr = expr.build_ir_in(func, block);
                match op {
                    ast::UnaryOp::Pos => expr,
                    ast::UnaryOp::Neg => {
                        let dfg = func.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let neg = dfg.new_value().binary(BinaryOp::Sub, zero, expr);
                        push_inst(func, block, neg);
                        neg
                    }
                    ast::UnaryOp::Not => {
                        let dfg = func.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let not = dfg.new_value().binary(BinaryOp::Eq, zero, expr);
                        push_inst(func, block, not);
                        not
                    }
                }
            }
            ast::Expr::Number(num) => {
                let dfg = func.dfg_mut();
                let num = dfg.new_value().integer(num);
                num
            }
        }
    }
}

/// Push an instruction to the end of a BasicBlock.
fn push_inst(func: &mut FunctionData, block: BasicBlock, inst: Value) {
    func.layout_mut()
        .bb_mut(block)
        .insts_mut()
        .push_key_back(inst)
        .unwrap();
}
