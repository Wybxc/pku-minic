//! Build IR from AST.

use crate::ast::Spanned;
use crate::irgen::layout::LayoutBuilder;
use crate::irgen::metadata::FunctionMetadata;
use crate::{ast, irgen::metadata::ProgramMetadata};
use koopa::ir::builder_traits::*;
use koopa::ir::dfg::DataFlowGraph;
use koopa::ir::*;
use miette::Result;

mod error;
mod layout;
pub mod metadata;
mod symtable;

use error::CompileError;

use symtable::{Symbol, SymbolTable};

#[must_use]
/// Whether the current basic block is terminated.
pub struct Terminated(bool);

impl From<bool> for Terminated {
    fn from(terminated: bool) -> Self {
        Terminated(terminated)
    }
}

impl Terminated {
    /// Create a new `Terminated` with the given value.
    pub fn new(b: bool) -> Self {
        Self(b)
    }

    /// Whether the current basic block is terminated.
    pub fn is_terminated(&self) -> bool {
        self.0
    }
}

impl ast::CompUnit {
    /// Build IR from AST.
    pub fn build_ir(self) -> Result<(Program, ProgramMetadata)> {
        let mut symtable = SymbolTable::new();
        let mut program = Program::new();
        let mut metadata = ProgramMetadata::new();
        self.build_ir_in(&mut symtable, &mut program, &mut metadata)?;
        Ok((program, metadata))
    }

    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        metadata: &mut ProgramMetadata,
    ) -> Result<()> {
        self.func_def.build_ir_in(symtable, program, metadata)
    }
}

impl ast::FuncDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        metadata: &mut ProgramMetadata,
    ) -> Result<()> {
        let func = program.new_func(FunctionData::with_param_names(
            format!("@{}", self.ident.node),
            vec![],
            self.func_type.node.build_ir(),
        ));

        let func_metadata = FunctionMetadata::new(self.ident);
        metadata.functions.insert(func, func_metadata);

        let mut layout = LayoutBuilder::new(program.func_mut(func), self.func_type.node);
        let terminated = self.block.node.build_ir_in(symtable, &mut layout)?;
        if !terminated.is_terminated() {
            layout.terminate_current_bb();
        }

        Ok(())
    }
}

impl ast::FuncType {
    /// Build IR from AST.
    pub fn build_ir(self) -> Type {
        match self {
            ast::FuncType::Int => Type::get_i32(),
        }
    }

    /// Default value of the type.
    pub fn default_value(self, dfg: &mut DataFlowGraph) -> Value {
        match self {
            ast::FuncType::Int => dfg.new_value().integer(0),
        }
    }
}

impl ast::Block {
    /// Build IR from AST in an existing IR node.
    #[must_use = "this returns whether the current basic block is terminated"]
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        let mut terminated = Terminated::new(false);
        symtable.push();
        for item in self.items {
            if terminated.is_terminated() {
                break;
            }
            terminated = item.build_ir_in(symtable, layout)?;
        }
        symtable.pop();
        Ok(terminated)
    }
}

impl ast::BlockItem {
    /// Build IR from AST in an existing IR node.
    #[must_use = "this returns whether the current basic block is terminated"]
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        match self {
            ast::BlockItem::Decl { decl } => {
                decl.build_ir_in(symtable, layout)?;
                Ok(false.into())
            }
            ast::BlockItem::Stmt { stmt } => stmt.node.build_ir_in(symtable, layout),
        }
    }
}

impl ast::Decl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, layout: &mut LayoutBuilder) -> Result<()> {
        match self {
            ast::Decl::Const(const_decl) => const_decl.node.build_ir_in(symtable),
            ast::Decl::Var(var_decl) => var_decl.node.build_ir_in(symtable, layout),
        }
    }
}

impl ast::BType {
    /// Build IR from AST.
    pub fn build_ir(&self) -> Type {
        match self {
            ast::BType::Int => Type::get_i32(),
        }
    }
}

impl ast::ConstDecl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable) -> Result<()> {
        for def in self.defs {
            def.build_ir_in(symtable)?;
        }
        Ok(())
    }
}

impl ast::ConstDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable) -> Result<()> {
        let expr = self.expr.const_eval(symtable)?;
        symtable.insert_var(self.ident.node, Symbol::Const(expr));
        Ok(())
    }
}

impl ast::ConstExpr {
    /// Const evaluation.
    pub fn const_eval(self, symtable: &SymbolTable) -> Result<i32> {
        self.expr.const_eval(symtable)
    }
}

impl ast::VarDecl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, layout: &mut LayoutBuilder) -> Result<()> {
        for def in self.defs {
            def.build_ir_in(symtable, &self.ty.node, layout)?;
        }

        Ok(())
    }
}

impl ast::VarDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        ty: &ast::BType,
        layout: &mut LayoutBuilder,
    ) -> Result<()> {
        // Allocate a new variable.
        let dfg = layout.dfg_mut();
        let var = dfg.new_value().alloc(ty.build_ir());
        dfg.set_value_name(var, Some(format!("@{}", self.ident.node)));
        layout.push_inst(var);

        // Initialize the variable.
        if let Some(init) = self.init {
            let init = init.build_ir_in(symtable, layout)?;
            let store = layout.dfg_mut().new_value().store(init, var);
            layout.push_inst(store);
        }

        // Insert the variable into the symbol table.
        symtable.insert_var(self.ident.node, Symbol::Var(var));

        Ok(())
    }
}

impl ast::InitVal {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &SymbolTable, layout: &mut LayoutBuilder) -> Result<Value> {
        self.expr.build_ir_in(symtable, layout)
    }
}

impl ast::Stmt {
    /// Build IR from AST in an existing IR node.
    #[must_use = "this returns whether the current basic block is terminated"]
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        match self {
            ast::Stmt::Assign { ident, expr } => {
                // Get the variable.
                let span = ident.span().into();
                let var = symtable
                    .get_var(&ident.node)
                    .ok_or(CompileError::VariableNotFound { span })?;
                let var = match var {
                    Symbol::Const(_) => Err(CompileError::AssignToConst { span })?,
                    Symbol::Var(var) => *var,
                };

                // Compute the expression.
                let expr = expr.build_ir_in(symtable, layout)?;

                // Store the value.
                let store = layout.dfg_mut().new_value().store(expr, var);
                layout.push_inst(store);
                Ok(false.into())
            }
            ast::Stmt::Return { expr } => {
                let expr = expr.build_ir_in(symtable, layout)?;
                let dfg = layout.dfg_mut();
                let ret = dfg.new_value().ret(Some(expr));
                layout.push_inst(ret);
                Ok(true.into())
            }
            ast::Stmt::Expr { expr } => {
                if let Some(expr) = expr {
                    expr.build_ir_in(symtable, layout)?;
                }
                Ok(false.into())
            }
            ast::Stmt::Block { block } => block.node.build_ir_in(symtable, layout),
            ast::Stmt::If { cond, then, els } => {
                let cond = cond.build_ir_in(symtable, layout)?;

                let then_bb = layout.new_bb(None);
                let els_bb = layout.new_bb(None);
                let next_bb = if els.is_some() {
                    layout.new_bb(None)
                } else {
                    els_bb
                };

                layout.with_bb(then_bb, |layout| {
                    let terminated = then.node.build_ir_in(symtable, layout)?;
                    if !terminated.is_terminated() {
                        let jump_to_next = layout.dfg_mut().new_value().jump(next_bb);
                        layout.push_inst(jump_to_next);
                    }
                    Ok(())
                })?;

                if let Some(els) = els {
                    layout.with_bb(els_bb, |layout| {
                        let terminated = els.node.build_ir_in(symtable, layout)?;
                        if !terminated.is_terminated() {
                            let jump_to_next = layout.dfg_mut().new_value().jump(next_bb);
                            layout.push_inst(jump_to_next);
                        }
                        Ok(())
                    })?;
                }

                let branch = layout.dfg_mut().new_value().branch(cond, then_bb, els_bb);
                layout.push_inst(branch);
                // Safety: `next_bb` is empty, and `branch` is the last instruction.
                layout.switch_bb(next_bb);
                Ok(false.into())
            }
        }
    }
}

impl ast::Expr {
    /// Build IR from AST in a BasicBlock, return the handle of the result value.
    pub fn build_ir_in(self, symtable: &SymbolTable, layout: &mut LayoutBuilder) -> Result<Value> {
        Ok(match self {
            ast::Expr::Unary { op, expr } => {
                let expr = expr.build_ir_in(symtable, layout)?;
                match op.node {
                    ast::UnaryOp::Pos => expr,
                    ast::UnaryOp::Neg => {
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let neg = dfg.new_value().binary(BinaryOp::Sub, zero, expr);
                        layout.push_inst(neg)
                    }
                    ast::UnaryOp::Not => {
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let not = dfg.new_value().binary(BinaryOp::Eq, expr, zero);
                        layout.push_inst(not)
                    }
                }
            }
            ast::Expr::Binary { op, lhs, rhs } => {
                let lhs = lhs.build_ir_in(symtable, layout)?;
                let rhs = rhs.build_ir_in(symtable, layout)?;
                let dfg = layout.dfg_mut();
                match op.node {
                    ast::BinaryOp::Add => {
                        let add = dfg.new_value().binary(BinaryOp::Add, lhs, rhs);
                        layout.push_inst(add)
                    }
                    ast::BinaryOp::Sub => {
                        let sub = dfg.new_value().binary(BinaryOp::Sub, lhs, rhs);
                        layout.push_inst(sub)
                    }
                    ast::BinaryOp::Mul => {
                        let mul = dfg.new_value().binary(BinaryOp::Mul, lhs, rhs);
                        layout.push_inst(mul)
                    }
                    ast::BinaryOp::Div => {
                        let div = dfg.new_value().binary(BinaryOp::Div, lhs, rhs);
                        layout.push_inst(div)
                    }
                    ast::BinaryOp::Mod => {
                        let rem = dfg.new_value().binary(BinaryOp::Mod, lhs, rhs);
                        layout.push_inst(rem)
                    }
                    ast::BinaryOp::Eq => {
                        let eq = dfg.new_value().binary(BinaryOp::Eq, lhs, rhs);
                        layout.push_inst(eq)
                    }
                    ast::BinaryOp::Ne => {
                        let not_eq = dfg.new_value().binary(BinaryOp::NotEq, lhs, rhs);
                        layout.push_inst(not_eq)
                    }
                    ast::BinaryOp::Lt => {
                        let lt = dfg.new_value().binary(BinaryOp::Lt, lhs, rhs);
                        layout.push_inst(lt)
                    }
                    ast::BinaryOp::Le => {
                        let lt_eq = dfg.new_value().binary(BinaryOp::Le, lhs, rhs);
                        layout.push_inst(lt_eq)
                    }
                    ast::BinaryOp::Gt => {
                        let gt = dfg.new_value().binary(BinaryOp::Gt, lhs, rhs);
                        layout.push_inst(gt)
                    }
                    ast::BinaryOp::Ge => {
                        let gt_eq = dfg.new_value().binary(BinaryOp::Ge, lhs, rhs);
                        layout.push_inst(gt_eq)
                    }
                    ast::BinaryOp::LAnd => {
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);
                        let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                        let and = dfg.new_value().binary(BinaryOp::And, lhs, rhs);
                        layout.push_insts([lhs, rhs, and])
                    }
                    ast::BinaryOp::LOr => {
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);
                        let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                        let or = dfg.new_value().binary(BinaryOp::Or, lhs, rhs);
                        layout.push_insts([lhs, rhs, or])
                    }
                }
            }
            ast::Expr::Number(num) => {
                let dfg = layout.dfg_mut();
                let num = dfg.new_value().integer(num.node);
                num
            }
            ast::Expr::LVar(name) => {
                let var = symtable
                    .get_var(&name.node)
                    .ok_or(CompileError::VariableNotFound {
                        span: name.span().into(),
                    })?;
                match var {
                    Symbol::Const(num) => {
                        let dfg = layout.dfg_mut();
                        let num = dfg.new_value().integer(*num);
                        num
                    }
                    Symbol::Var(var) => {
                        let dfg = layout.dfg_mut();
                        let load = dfg.new_value().load(*var);
                        layout.push_inst(load)
                    }
                }
            }
        })
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
                    ast::BinaryOp::Add => lhs.saturating_add(rhs),
                    ast::BinaryOp::Sub => lhs.saturating_sub(rhs),
                    ast::BinaryOp::Mul => lhs.saturating_mul(rhs),
                    ast::BinaryOp::Div => lhs.checked_div(rhs).unwrap_or(-1), // Maybe a compile error?
                    ast::BinaryOp::Mod => lhs.checked_rem(rhs).unwrap_or(-1),
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
                let span = name.span().into();
                let var = symtable
                    .get_var(&name.node)
                    .ok_or(CompileError::VariableNotFound { span })?;
                match var {
                    Symbol::Const(num) => *num,
                    Symbol::Var(_) => Err(CompileError::NonConstantExpression { span })?,
                }
            }
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        #[test]
        fn test_variables(
            program in ast::arbitrary::arb_comp_unit()
                .prop_map(ast::display::Displayable)
        ) {
            let (program, _) = program.0.build_ir().unwrap();
            for func in program.funcs().values() {
                let mut defined = HashSet::new();
                for (_, block) in func.layout().bbs() {
                    for &inst in block.insts().keys() {
                        for value in func.dfg().value(inst).kind().value_uses() {
                            if !is_const(func.dfg().value(value)) {
                                assert!(
                                    defined.contains(&value),
                                    "variable {:?} is used before defined",
                                    value
                                );
                            }
                        }
                        if !defined.insert(inst) {
                            panic!("variable {:?} is defined twice", inst);
                        }
                    }
                }
            }
        }

        #[test]
        fn test_basic_block_end(
            program in ast::arbitrary::arb_comp_unit()
                .prop_map(ast::display::Displayable)
        ) {
            let (program, _) = program.0.build_ir().unwrap();
            for func in program.funcs().values() {
                for (&bb, block) in func.layout().bbs() {
                    let last_inst = block.insts().back_key().unwrap();
                    let value = func.dfg().value(*last_inst);
                    assert!(
                        is_terminator(value),
                        "block {} does not end with a terminator",
                        ident_block(bb, func.dfg())
                    );
                }
            }
        }

        #[test]
        fn test_terminator_in_middle(
            program in ast::arbitrary::arb_comp_unit()
                .prop_map(ast::display::Displayable)
        ) {
            let (program, _) = program.0.build_ir().unwrap();
            for func in program.funcs().values() {
                for (&bb, block) in func.layout().bbs() {
                    let mut cursor = block.insts().cursor(*block.insts().back_key().unwrap());
                    cursor.move_prev();
                    while !cursor.is_null() {
                        let inst = *cursor.key().unwrap();
                        let value = func.dfg().value(inst);
                        assert!(
                            !is_terminator(value),
                            "terminator {:?} is not at the end of block {}",
                            inst,
                            ident_block(bb, func.dfg())
                        );
                        cursor.move_prev();
                    }
                }
            }
        }
    }
}
