//! Build IR from AST.

use koopa::ir::{builder_traits::*, dfg::DataFlowGraph, *};
use miette::Result;

use crate::{
    ast,
    ast::Spanned,
    irgen::{
        layout::LayoutBuilder,
        metadata::{FunctionMetadata, ProgramMetadata},
    },
};

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
                    if !then.node.build_ir_in(symtable, layout)?.is_terminated() {
                        let jump_to_next = layout.dfg_mut().new_value().jump(next_bb);
                        layout.push_inst(jump_to_next);
                    }
                    Ok(())
                })?;

                if let Some(els) = els {
                    layout.with_bb(els_bb, |layout| {
                        if !els.node.build_ir_in(symtable, layout)?.is_terminated() {
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
            ast::Stmt::While { cond, body } => {
                let cond_bb = layout.new_bb(None);
                let body_bb = layout.new_bb(None);
                let next_bb = layout.new_bb(None);

                let jump_to_cond = layout.dfg_mut().new_value().jump(cond_bb);
                layout.push_inst(jump_to_cond);
                // Safety: `cond_bb` is empty, and `into_cond` is the last instruction.
                layout.switch_bb(cond_bb);

                let cond = cond.build_ir_in(symtable, layout)?;
                let branch = layout.dfg_mut().new_value().branch(cond, body_bb, next_bb);
                layout.push_inst(branch);

                layout.with_bb(body_bb, |layout| {
                    if !body.node.build_ir_in(symtable, layout)?.is_terminated() {
                        let jump_to_cond = layout.dfg_mut().new_value().jump(cond_bb);
                        layout.push_inst(jump_to_cond);
                    }
                    Ok(())
                })?;

                // Safety: `next_bb` is empty, and `branch` is the last instruction.
                layout.switch_bb(next_bb);
                Ok(false.into())
            }
        }
    }
}

impl ast::Expr {
    /// Build IR from AST in a BasicBlock, return the handle of the result
    /// value.
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
                match op.node {
                    ast::BinaryOp::LAnd => {
                        let lhs = lhs.build_ir_in(symtable, layout)?;
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);

                        // short-circuit
                        let result = dfg.new_value().alloc(Type::get_i32());
                        let true_bb = layout.new_bb(None);
                        let false_bb = layout.new_bb(None);
                        let next_bb = layout.new_bb(None);

                        let branch = layout.dfg_mut().new_value().branch(lhs, true_bb, false_bb);
                        layout.push_insts([lhs, result, branch]);

                        layout.with_bb(true_bb, |layout| {
                            let rhs = rhs.build_ir_in(symtable, layout)?;
                            let dfg = layout.dfg_mut();
                            let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                            let store = dfg.new_value().store(rhs, result);
                            let jump = dfg.new_value().jump(next_bb);
                            layout.push_insts([rhs, store, jump]);
                            Ok(())
                        })?;

                        layout.with_bb(false_bb, |layout| {
                            let dfg = layout.dfg_mut();
                            let store = dfg.new_value().store(zero, result);
                            let jump = dfg.new_value().jump(next_bb);
                            layout.push_insts([store, jump]);
                            Ok(())
                        })?;

                        // Safety: `next_bb` is empty, and `branch` is the last instruction.
                        layout.switch_bb(next_bb);

                        let result = layout.dfg_mut().new_value().load(result);
                        layout.push_inst(result)
                    }
                    ast::BinaryOp::LOr => {
                        let lhs = lhs.build_ir_in(symtable, layout)?;
                        let dfg = layout.dfg_mut();
                        let zero = dfg.new_value().integer(0);
                        let lhs = dfg.new_value().binary(BinaryOp::NotEq, lhs, zero);

                        // short-circuit
                        let result = dfg.new_value().alloc(Type::get_i32());
                        let true_bb = layout.new_bb(None);
                        let false_bb = layout.new_bb(None);
                        let next_bb = layout.new_bb(None);

                        let branch = layout.dfg_mut().new_value().branch(lhs, true_bb, false_bb);
                        layout.push_insts([lhs, result, branch]);

                        layout.with_bb(true_bb, |layout| {
                            let dfg = layout.dfg_mut();
                            let store = dfg.new_value().store(lhs, result);
                            let jump = dfg.new_value().jump(next_bb);
                            layout.push_insts([store, jump]);
                            Ok(())
                        })?;

                        layout.with_bb(false_bb, |layout| {
                            let rhs = rhs.build_ir_in(symtable, layout)?;
                            let dfg = layout.dfg_mut();
                            let rhs = dfg.new_value().binary(BinaryOp::NotEq, rhs, zero);
                            let store = dfg.new_value().store(rhs, result);
                            let jump = dfg.new_value().jump(next_bb);
                            layout.push_insts([rhs, store, jump]);
                            Ok(())
                        })?;

                        // Safety: `next_bb` is empty, and `branch` is the last instruction.
                        layout.switch_bb(next_bb);

                        let result = layout.dfg_mut().new_value().load(result);
                        layout.push_inst(result)
                    }
                    _ => {
                        let lhs = lhs.build_ir_in(symtable, layout)?;
                        let rhs = rhs.build_ir_in(symtable, layout)?;
                        let dfg = layout.dfg_mut();
                        let op = match op.node {
                            ast::BinaryOp::Add => BinaryOp::Add,
                            ast::BinaryOp::Sub => BinaryOp::Sub,
                            ast::BinaryOp::Mul => BinaryOp::Mul,
                            ast::BinaryOp::Div => BinaryOp::Div,
                            ast::BinaryOp::Mod => BinaryOp::Mod,
                            ast::BinaryOp::Eq => BinaryOp::Eq,
                            ast::BinaryOp::Ne => BinaryOp::NotEq,
                            ast::BinaryOp::Lt => BinaryOp::Lt,
                            ast::BinaryOp::Le => BinaryOp::Le,
                            ast::BinaryOp::Gt => BinaryOp::Gt,
                            ast::BinaryOp::Ge => BinaryOp::Ge,
                            ast::BinaryOp::LAnd | ast::BinaryOp::LOr => unreachable!(),
                        };
                        let inst = dfg.new_value().binary(op, lhs, rhs);
                        layout.push_inst(inst)
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
                    ast::BinaryOp::Div => lhs.checked_div(rhs).unwrap_or(-1), // A compile error?
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
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::*;
    use crate::{analysis::Analyzer, utils::*};

    proptest! {
        #[test]
        fn test_variables(
            program in ast::arbitrary::arb_comp_unit()
                .prop_map(ast::display::Displayable)
        ) {
            let (program, metadata) = program.0.build_ir().unwrap();
            let mut analyzer = Analyzer::new(&program, &metadata);
            for (&func, func_data) in program.funcs() {
                let dominators = analyzer.analyze_dominators(func);
                let mut defined = HashSet::new();
                for bb in dominators.iter() {
                    let block = func_data.layout().bbs().node(&bb).expect("bb not found");
                    for &inst in block.insts().keys() {
                        for value in func_data.dfg().value(inst).kind().value_uses() {
                            if !is_const(func_data.dfg().value(value)) {
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
