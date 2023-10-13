//! Build IR from AST.
//!
//! TODO: Sanity check.
//!   - [ ] Check if all variables are declared before use.
//!   - [ ] Check if all instructions have unique value id.

use crate::ast;
use koopa::ir::builder_traits::*;
use koopa::ir::*;

mod error;
mod symtable;

use error::{CompileError, Result};
use symtable::{Symbol, SymbolTable};

impl ast::CompUnit {
    /// Build IR from AST.
    pub fn build_ir(self) -> Program {
        let mut symtable = SymbolTable::new();
        let mut program = Program::new();
        self.build_ir_in(&mut symtable, &mut program);
        program
    }

    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, program: &mut Program) {
        self.func_def.build_ir_in(symtable, program);
    }
}

impl ast::FuncDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, program: &mut Program) {
        let func = program.new_func(FunctionData::with_param_names(
            format!("@{}", self.ident.node),
            vec![],
            self.func_type.node.build_ir(),
        ));
        let func_data = program.func_mut(func);

        self.block.node.build_ir_in(symtable, func_data);
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
    pub fn build_ir_in(self, symtable: &mut SymbolTable, func: &mut FunctionData) {
        let entry = func.dfg_mut().new_bb().basic_block(Some("%entry".into()));
        func.layout_mut().bbs_mut().extend([entry]);

        for item in self.items {
            item.build_ir_in(symtable, func, entry);
        }
    }
}

impl ast::BlockItem {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        func: &mut FunctionData,
        block: BasicBlock,
    ) {
        match self {
            ast::BlockItem::Decl { decl } => decl.build_ir_in(symtable, func, block),
            ast::BlockItem::Stmt { stmt } => stmt.node.build_ir_in(symtable, func, block),
        }
    }
}

impl ast::Decl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        _func: &mut FunctionData,
        _block: BasicBlock,
    ) {
        match self {
            ast::Decl::Const(const_decl) => const_decl.node.build_ir_in(symtable),
        }
    }
}

impl ast::ConstDecl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable) {
        for def in self.defs {
            def.build_ir_in(symtable);
        }
    }
}

impl ast::ConstDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable) {
        let expr = self.expr.const_eval(symtable).unwrap(); // todo: error handling
        symtable.insert_var(self.ident.node, Symbol::Const(expr));
    }
}

impl ast::ConstExpr {
    /// Const evaluation.
    pub fn const_eval(self, symtable: &SymbolTable) -> Result<i32> {
        self.expr.const_eval(symtable)
    }
}

impl ast::Stmt {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &SymbolTable, func: &mut FunctionData, block: BasicBlock) {
        match self {
            ast::Stmt::Return { expr } => {
                let expr = expr.build_ir_in(symtable, func, block);
                let dfg = func.dfg_mut();
                let ret = dfg.new_value().ret(Some(expr));

                push_inst(func, block, ret);
            }
        }
    }
}

impl ast::Expr {
    /// Build IR from AST in a BasicBlock, return the handle of the result value.
    pub fn build_ir_in(
        self,
        symtable: &SymbolTable,
        func: &mut FunctionData,
        block: BasicBlock,
    ) -> Value {
        match self {
            ast::Expr::Unary { op, expr } => {
                let expr = expr.build_ir_in(symtable, func, block);
                match op.node {
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
                        let not = dfg.new_value().binary(BinaryOp::Eq, expr, zero);
                        push_inst(func, block, not);
                        not
                    }
                }
            }
            ast::Expr::Binary { op, lhs, rhs } => {
                let lhs = lhs.build_ir_in(symtable, func, block);
                let rhs = rhs.build_ir_in(symtable, func, block);
                let dfg = func.dfg_mut();
                match op.node {
                    ast::BinaryOp::Add => {
                        let add = dfg.new_value().binary(BinaryOp::Add, lhs, rhs);
                        push_inst(func, block, add);
                        add
                    }
                    ast::BinaryOp::Sub => {
                        let sub = dfg.new_value().binary(BinaryOp::Sub, lhs, rhs);
                        push_inst(func, block, sub);
                        sub
                    }
                    ast::BinaryOp::Mul => {
                        let mul = dfg.new_value().binary(BinaryOp::Mul, lhs, rhs);
                        push_inst(func, block, mul);
                        mul
                    }
                    ast::BinaryOp::Div => {
                        let div = dfg.new_value().binary(BinaryOp::Div, lhs, rhs);
                        push_inst(func, block, div);
                        div
                    }
                    ast::BinaryOp::Mod => {
                        let rem = dfg.new_value().binary(BinaryOp::Mod, lhs, rhs);
                        push_inst(func, block, rem);
                        rem
                    }
                    ast::BinaryOp::Eq => {
                        let eq = dfg.new_value().binary(BinaryOp::Eq, lhs, rhs);
                        push_inst(func, block, eq);
                        eq
                    }
                    ast::BinaryOp::Ne => {
                        let not_eq = dfg.new_value().binary(BinaryOp::NotEq, lhs, rhs);
                        push_inst(func, block, not_eq);
                        not_eq
                    }
                    ast::BinaryOp::Lt => {
                        let lt = dfg.new_value().binary(BinaryOp::Lt, lhs, rhs);
                        push_inst(func, block, lt);
                        lt
                    }
                    ast::BinaryOp::Le => {
                        let lt_eq = dfg.new_value().binary(BinaryOp::Le, lhs, rhs);
                        push_inst(func, block, lt_eq);
                        lt_eq
                    }
                    ast::BinaryOp::Gt => {
                        let gt = dfg.new_value().binary(BinaryOp::Gt, lhs, rhs);
                        push_inst(func, block, gt);
                        gt
                    }
                    ast::BinaryOp::Ge => {
                        let gt_eq = dfg.new_value().binary(BinaryOp::Ge, lhs, rhs);
                        push_inst(func, block, gt_eq);
                        gt_eq
                    }
                    ast::BinaryOp::LAnd => {
                        let dfg = func.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);
                        let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                        let and = dfg.new_value().binary(BinaryOp::And, lhs, rhs);
                        push_inst(func, block, lhs);
                        push_inst(func, block, rhs);
                        push_inst(func, block, and);
                        and
                    }
                    ast::BinaryOp::LOr => {
                        let dfg = func.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);
                        let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                        let or = dfg.new_value().binary(BinaryOp::Or, lhs, rhs);
                        push_inst(func, block, lhs);
                        push_inst(func, block, rhs);
                        push_inst(func, block, or);
                        or
                    }
                }
            }
            ast::Expr::Number(num) => {
                let dfg = func.dfg_mut();
                let num = dfg.new_value().integer(num.node);
                num
            }
            ast::Expr::LVar(name) => {
                let var = symtable
                    .get_var(&name.node)
                    .ok_or(CompileError::VariableNotFound(name.node))
                    .unwrap(); // todo: error handling
                match var {
                    Symbol::Const(num) => {
                        let dfg = func.dfg_mut();
                        let num = dfg.new_value().integer(*num);
                        num
                    }
                }
            }
        }
    }

    /// Const evaluation.
    pub fn const_eval(self, symtable: &SymbolTable) -> Result<i32> {
        Ok(match self {
            ast::Expr::Unary { op, expr } => {
                let expr = expr.const_eval(symtable)?;
                match op.node {
                    ast::UnaryOp::Pos => expr,
                    ast::UnaryOp::Neg => -expr,
                    ast::UnaryOp::Not => (expr == 0) as i32,
                }
            }
            ast::Expr::Binary { op, lhs, rhs } => {
                let lhs = lhs.const_eval(symtable)?;
                let rhs = rhs.const_eval(symtable)?;
                match op.node {
                    ast::BinaryOp::Add => lhs + rhs,
                    ast::BinaryOp::Sub => lhs - rhs,
                    ast::BinaryOp::Mul => lhs * rhs,
                    ast::BinaryOp::Div => lhs / rhs,
                    ast::BinaryOp::Mod => lhs % rhs,
                    ast::BinaryOp::Eq => (lhs == rhs) as i32,
                    ast::BinaryOp::Ne => (lhs != rhs) as i32,
                    ast::BinaryOp::Lt => (lhs < rhs) as i32,
                    ast::BinaryOp::Le => (lhs <= rhs) as i32,
                    ast::BinaryOp::Gt => (lhs > rhs) as i32,
                    ast::BinaryOp::Ge => (lhs >= rhs) as i32,
                    ast::BinaryOp::LAnd => ((lhs != 0) && (rhs != 0)) as i32,
                    ast::BinaryOp::LOr => ((lhs != 0) || (rhs != 0)) as i32,
                }
            }
            ast::Expr::Number(num) => num.node,
            ast::Expr::LVar(name) => {
                let var = symtable
                    .get_var(&name.node)
                    .ok_or(CompileError::VariableNotFound(name.node))?;
                match var {
                    Symbol::Const(num) => *num,
                }
            }
        })
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
