//! Build IR from AST.

use koopa::ir::{builder_traits::*, dfg::DataFlowGraph, *};
use miette::Result;

use crate::{
    ast::{self, Span, Spanned},
    irgen::{
        context::Context,
        layout::LayoutBuilder,
        metadata::{FunctionMetadata, ProgramMetadata},
    },
};

mod context;
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
        symtable.push();
        self.build_ir_in(&mut symtable, &mut program, &mut metadata)?;
        symtable.pop();
        Ok((program, metadata))
    }

    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        metadata: &mut ProgramMetadata,
    ) -> Result<()> {
        // decl @getint(): i32
        self.decl_function(symtable, program, "@getint", vec![], Type::get_i32());
        // decl @getch(): i32
        self.decl_function(symtable, program, "@getch", vec![], Type::get_i32());
        // decl @getarray(*i32): i32
        // self.decl_function(symtable, program, "@getarray", vec![Type::get_ptr(Type::get_i32())], Type::get_i32());
        // decl @putint(i32)
        self.decl_function(
            symtable,
            program,
            "@putint",
            vec![Type::get_i32()],
            Type::get_unit(),
        );
        // decl @putch(i32)
        self.decl_function(
            symtable,
            program,
            "@putch",
            vec![Type::get_i32()],
            Type::get_unit(),
        );
        // decl @putarray(i32, *i32)
        // self.decl_function(symtable, program, "@putarray", vec![Type::get_i32(), Type::get_ptr(Type::get_i32())], Type::get_unit());
        // decl @starttime()
        self.decl_function(symtable, program, "@starttime", vec![], Type::get_unit());
        // decl @stoptime()
        self.decl_function(symtable, program, "@stoptime", vec![], Type::get_unit());

        for item in self.top_levels {
            match item {
                ast::TopLevelItem::FuncDef(func_def) => {
                    func_def.build_ir_in(symtable, program, metadata)?;
                }
                ast::TopLevelItem::Decl(decl) => {
                    decl.build_ir_global(symtable, program)?;
                }
            }
        }
        Ok(())
    }

    fn decl_function(
        &self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        name: &'static str,
        params_ty: Vec<Type>,
        ret_ty: Type,
    ) {
        let func = program.new_func(FunctionData::new_decl(name.into(), params_ty, ret_ty));
        if symtable
            .insert_var(name[1..].into(), Symbol::Func(func))
            .is_some()
        {
            unreachable!();
        }
    }
}

impl ast::FuncDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        prog_metadata: &mut ProgramMetadata,
    ) -> Result<()> {
        let span = self.ident.span().into();
        let metadata = FunctionMetadata::new(self.ident.clone());
        let kp_name = format!("@{}", self.ident.node);
        let params = self.params.iter().map(|p| p.build_ir()).collect();
        let ret_type = self.func_type.node.build_ir();
        let func = program.new_func(FunctionData::with_param_names(kp_name, params, ret_type));
        prog_metadata.functions.insert(func, metadata);

        // add function to global symbol table
        if symtable
            .insert_var(self.ident.node, Symbol::Func(func))
            .is_some()
        {
            Err(CompileError::FunctionDefinedTwice { span })?;
        }

        let mut context = Context::new();
        let mut layout = LayoutBuilder::new(program.func_mut(func), self.func_type.node);

        // alloc var for parameters
        symtable.push();
        let params = layout.params().to_vec();
        for (value, param) in params.into_iter().zip(self.params) {
            let span = param.ident.span().into();
            let dfg = layout.dfg_mut();
            let name = param.ident.node;
            let ty = dfg.value(value).ty().clone();
            let var = dfg.new_value().alloc(ty.clone());
            let store = dfg.new_value().store(value, var);
            dfg.set_value_name(var, Some(format!("%{}", name)));
            layout.push_insts([var, store]);
            if symtable
                .insert_var(name, Symbol::Var(var, ty, false))
                .is_some()
            {
                Err(CompileError::DuplicateParameter { span })?;
            }
        }

        let terminated = self
            .block
            .node
            .build_ir_in(symtable, &mut context, &mut layout)?;
        if !terminated.is_terminated() {
            layout.terminate_current_bb();
        }
        symtable.pop();

        Ok(())
    }
}

impl ast::FuncParam {
    /// Build IR from AST.
    pub fn build_ir(&self) -> (Option<String>, Type) {
        (
            Some(format!("@{}", self.ident.node)),
            self.ty.node.build_ir(),
        )
    }
}

impl ast::Block {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        context: &mut Context,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        let mut terminated = Terminated::new(false);
        symtable.push();
        for item in self.items {
            if terminated.is_terminated() {
                break;
            }
            terminated = item.build_ir_in(symtable, context, layout)?;
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
        context: &mut Context,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        match self {
            ast::BlockItem::Decl { decl } => {
                decl.build_ir_in(symtable, layout)?;
                Ok(false.into())
            }
            ast::BlockItem::Stmt { stmt } => stmt.node.build_ir_in(symtable, context, layout),
        }
    }
}

impl ast::Decl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, layout: &mut LayoutBuilder) -> Result<()> {
        match self {
            ast::Decl::Const(const_decl) => const_decl.node.build_ir_in(symtable, layout),
            ast::Decl::Var(var_decl) => var_decl.node.build_ir_in(symtable, layout),
        }
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(self, symtable: &mut SymbolTable, program: &mut Program) -> Result<()> {
        match self {
            ast::Decl::Const(const_decl) => const_decl.node.build_ir_global(symtable, program),
            ast::Decl::Var(var_decl) => var_decl.node.build_ir_global(symtable, program),
        }
    }
}

impl ast::BType {
    /// Build IR from AST.
    pub fn build_ir(&self) -> Type {
        match self {
            ast::BType::Int => Type::get_i32(),
            ast::BType::Void => Type::get_unit(),
        }
    }

    /// Default value of the type.
    pub fn default_value(&self, dfg: &mut DataFlowGraph) -> Option<Value> {
        match self {
            ast::BType::Int => Some(dfg.new_value().integer(0)),
            ast::BType::Void => None,
        }
    }
}

impl ast::ConstDecl {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(self, symtable: &mut SymbolTable, layout: &mut LayoutBuilder) -> Result<()> {
        for def in self.defs {
            def.build_ir_in(symtable, &self.ty.node, layout)?;
        }
        Ok(())
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(self, symtable: &mut SymbolTable, program: &mut Program) -> Result<()> {
        for def in self.defs {
            def.build_ir_global(symtable, program, &self.ty.node)?;
        }
        Ok(())
    }
}

impl ast::ConstDef {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        ty: &ast::BType,
        layout: &mut LayoutBuilder,
    ) -> Result<()> {
        let span = self.ident.span().into();
        if let Some(index) = self.index {
            // Constant array.
            let values = self.init.init_array_const(index, symtable, ty)?;

            // Allocate a new array.
            let dfg = layout.dfg_mut();
            let ty = Type::get_array(ty.build_ir(), values.len());
            let array = dfg.new_value().alloc(ty.clone());
            dfg.set_value_name(array, Some(format!("@{}", self.ident.node)));
            layout.push_inst(array);

            // Initialize the array.
            for (index, value) in values.into_iter().enumerate() {
                let dfg = layout.dfg_mut();
                let index = dfg.new_value().integer(index as i32);
                let ptr = dfg.new_value().get_elem_ptr(array, index);
                let value = dfg.new_value().integer(value);
                let store = dfg.new_value().store(value, ptr);
                layout.push_insts([ptr, store]);
            }

            // Insert the array into the symbol table.
            if symtable
                .insert_var(self.ident.node, Symbol::Var(array, ty, true))
                .is_some()
            {
                Err(CompileError::VariableDefinedTwice { span })?;
            }

            Ok(())
        } else {
            // Constant variable.
            self.build_const(symtable, ty)
        }
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        ty: &ast::BType,
    ) -> Result<()> {
        let span = self.ident.span().into();
        if let Some(index) = self.index {
            // Constant array.
            let values = self.init.init_array_const(index, symtable, ty)?;
            let ty = Type::get_array(ty.build_ir(), values.len());
            let values = values
                .into_iter()
                .map(|v| program.new_value().integer(v))
                .collect();
            let init = program.new_value().aggregate(values);

            // Allocate a new array.
            let array = program.new_value().global_alloc(init);
            program.set_value_name(array, Some(format!("@{}", self.ident.node)));

            // Insert the array into the symbol table.
            if symtable
                .insert_var(self.ident.node, Symbol::Var(array, ty, true))
                .is_some()
            {
                Err(CompileError::VariableDefinedTwice { span })?;
            }

            Ok(())
        } else {
            // Constant variable.
            self.build_const(symtable, ty)
        }
    }

    /// Build IR from AST for constant variables (not arrays).
    fn build_const(self, symtable: &mut SymbolTable, ty: &ast::BType) -> Result<()> {
        let span = self.ident.span().into();
        let expr = self.init.try_into_expr(ty)?.const_eval(symtable)?;
        if symtable
            .insert_var(self.ident.node, Symbol::Const(expr))
            .is_some()
        {
            Err(CompileError::VariableDefinedTwice { span })?;
        }
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
            def.node.build_ir_in(symtable, &self.ty.node, layout)?;
        }
        Ok(())
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(self, symtable: &mut SymbolTable, program: &mut Program) -> Result<()> {
        for def in self.defs {
            def.node.build_ir_global(symtable, program, &self.ty.node)?;
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
        // Check the type.
        let span = self.ident.span().into();
        if matches!(ty, ast::BType::Void) {
            Err(CompileError::VariableTypeVoid { span })?;
        }

        if let Some(index) = self.index {
            // Array.
            let index_span = index.span().into();
            let index = index.const_eval(symtable)?;
            if index < 0 {
                Err(CompileError::InvalidArraySize {
                    span: index_span,
                    size: index,
                })?;
            }

            // Allocate a new array.
            let dfg = layout.dfg_mut();
            let ty_kp = Type::get_array(ty.build_ir(), index as usize);
            let array = dfg.new_value().alloc(ty_kp.clone());
            dfg.set_value_name(array, Some(format!("@{}", self.ident.node)));
            layout.push_inst(array);

            // Initialize the array.
            if let Some(init) = self.init {
                let values = init.init_array(index, symtable, ty, layout)?;
                for (index, value) in values.into_iter().enumerate() {
                    let dfg = layout.dfg_mut();
                    let index = dfg.new_value().integer(index as i32);
                    let ptr = dfg.new_value().get_elem_ptr(array, index);
                    let store = dfg.new_value().store(value, ptr);
                    layout.push_insts([ptr, store]);
                }
            }

            // Insert the array into the symbol table.
            if symtable
                .insert_var(self.ident.node, Symbol::Var(array, ty_kp, false))
                .is_some()
            {
                Err(CompileError::VariableDefinedTwice { span })?;
            }

            Ok(())
        } else {
            // Allocate a new variable.
            let dfg = layout.dfg_mut();
            let ty_kp = ty.build_ir();
            let var = dfg.new_value().alloc(ty_kp.clone());
            dfg.set_value_name(var, Some(format!("@{}", self.ident.node)));
            layout.push_inst(var);

            // Initialize the variable.
            if let Some(init) = self.init {
                let init = init.try_into_expr(ty)?.build_ir_in(symtable, layout)?;
                let store = layout.dfg_mut().new_value().store(init, var);
                layout.push_inst(store);
            }

            // Insert the variable into the symbol table.
            if symtable
                .insert_var(self.ident.node, Symbol::Var(var, ty_kp, false))
                .is_some()
            {
                Err(CompileError::VariableDefinedTwice { span })?;
            }
            Ok(())
        }
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(
        self,
        symtable: &mut SymbolTable,
        program: &mut Program,
        ty: &ast::BType,
    ) -> Result<()> {
        // Check the type.
        let span = self.ident.span().into();
        if matches!(ty, ast::BType::Void) {
            Err(CompileError::VariableTypeVoid { span })?;
        }

        // Initialize the variable.
        let init = if let Some(init) = self.init {
            if let Some(index) = self.index {
                // Array.
                let values = init.init_array_const(index, symtable, ty)?;
                let values = values
                    .into_iter()
                    .map(|v| program.new_value().integer(v))
                    .collect();
                program.new_value().aggregate(values)
            } else {
                // Variable.
                let value = init.try_into_expr(ty)?.const_eval(symtable)?;
                program.new_value().integer(value)
            }
        } else {
            program.new_value().zero_init(ty.build_ir())
        };

        // Allocate a new variable.
        let var = program.new_value().global_alloc(init);
        program.set_value_name(var, Some(format!("@{}", self.ident.node)));

        // Insert the variable into the symbol table.
        if symtable
            .insert_var(self.ident.node, Symbol::Var(var, ty.build_ir(), false))
            .is_some()
        {
            Err(CompileError::VariableDefinedTwice { span })?;
        }

        Ok(())
    }
}

impl Span<ast::InitVal> {
    /// Try to convert the initializer into an expression.
    pub fn try_into_expr(self, ty: &ast::BType) -> Result<ast::Expr> {
        let span = self.span().into();
        match self.node {
            ast::InitVal::Expr(expr) => Ok(expr),
            ast::InitVal::InitList(exprs) => {
                if exprs.len() != 1 {
                    Err(CompileError::TypeError {
                        span,
                        expected: format!("{}", ty),
                        found: format!("{}[]", ty),
                    })?;
                }
                Ok(exprs.into_iter().next().unwrap())
            }
        }
    }

    /// Try to convert the initializer into a list of expressions.
    pub fn try_into_exprs(self, ty: &ast::BType) -> Result<Vec<ast::Expr>> {
        let span = self.span().into();
        match self.node {
            ast::InitVal::Expr(_) => Err(CompileError::TypeError {
                span,
                expected: format!("{}[]", ty),
                found: format!("{}", ty),
            })?,
            ast::InitVal::InitList(exprs) => Ok(exprs),
        }
    }

    /// Initialize an array.
    pub fn init_array(
        self,
        index: i32,
        symtable: &mut SymbolTable,
        ty: &ast::BType,
        layout: &mut LayoutBuilder,
    ) -> Result<Vec<Value>> {
        let span = self.span().into();

        // Check the type.
        let exprs = self.try_into_exprs(ty)?;
        if exprs.len() > index as usize {
            Err(CompileError::TypeError {
                span,
                expected: format!("{}[{}]", ty, index),
                found: format!("{}[{}]", ty, exprs.len()),
            })?;
        }

        // Compute the values.
        let mut values = exprs
            .into_iter()
            .map(|expr| expr.build_ir_in(symtable, layout))
            .collect::<Result<Vec<_>>>()?;
        if values.len() < index as usize {
            values.resize(index as usize, layout.dfg_mut().new_value().integer(0));
        }

        Ok(values)
    }

    /// Initialize an array with constant values.
    pub fn init_array_const(
        self,
        index: ast::ConstExpr,
        symtable: &mut SymbolTable,
        ty: &ast::BType,
    ) -> Result<Vec<i32>> {
        let span = self.span().into();
        let index_span = index.span().into();
        let index = index.const_eval(symtable)?;
        if index < 0 {
            Err(CompileError::InvalidArraySize {
                span: index_span,
                size: index,
            })?;
        }

        // Check the type.
        let exprs = self.try_into_exprs(ty)?;
        if exprs.len() > index as usize {
            Err(CompileError::TypeError {
                span,
                expected: format!("{}[{}]", ty, index),
                found: format!("{}[{}]", ty, exprs.len()),
            })?;
        }

        // Compute the values.
        let mut values = exprs
            .into_iter()
            .map(|expr| expr.const_eval(symtable))
            .collect::<Result<Vec<_>>>()?;
        if values.len() < index as usize {
            values.resize(index as usize, 0);
        }

        Ok(values)
    }
}

impl ast::Stmt {
    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        symtable: &mut SymbolTable,
        context: &mut Context,
        layout: &mut LayoutBuilder,
    ) -> Result<Terminated> {
        match self {
            ast::Stmt::Assign { lval, expr } => {
                // Get the variable.
                let var = lval.build_lval(symtable, layout)?;

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
            ast::Stmt::Block { block } => block.node.build_ir_in(symtable, context, layout),
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
                    if !then
                        .node
                        .build_ir_in(symtable, context, layout)?
                        .is_terminated()
                    {
                        let jump_to_next = layout.dfg_mut().new_value().jump(next_bb);
                        layout.push_inst(jump_to_next);
                    }
                    Ok(())
                })?;

                if let Some(els) = els {
                    layout.with_bb(els_bb, |layout| {
                        if !els
                            .node
                            .build_ir_in(symtable, context, layout)?
                            .is_terminated()
                        {
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
                context.push_loop_boundary(cond_bb, next_bb);

                let jump_to_cond = layout.dfg_mut().new_value().jump(cond_bb);
                layout.push_inst(jump_to_cond);
                // Safety: `cond_bb` is empty, and `into_cond` is the last instruction.
                layout.switch_bb(cond_bb);

                let cond = cond.build_ir_in(symtable, layout)?;
                let branch = layout.dfg_mut().new_value().branch(cond, body_bb, next_bb);
                layout.push_inst(branch);

                layout.with_bb(body_bb, |layout| {
                    if !body
                        .node
                        .build_ir_in(symtable, context, layout)?
                        .is_terminated()
                    {
                        let jump_to_cond = layout.dfg_mut().new_value().jump(cond_bb);
                        layout.push_inst(jump_to_cond);
                    }
                    Ok(())
                })?;

                // Safety: `next_bb` is empty, and `branch` is the last instruction.
                layout.switch_bb(next_bb);
                context.pop_loop_boundary();
                Ok(false.into())
            }
            ast::Stmt::Break(span) => {
                let loop_boundary =
                    context
                        .current_loop_boundary()
                        .ok_or(CompileError::BreakNotInLoop {
                            span: span.span().into(),
                        })?;
                let jump = layout.dfg_mut().new_value().jump(loop_boundary.exit);
                layout.push_inst(jump);
                Ok(true.into())
            }
            ast::Stmt::Continue(span) => {
                let loop_boundary =
                    context
                        .current_loop_boundary()
                        .ok_or(CompileError::ContinueNotInLoop {
                            span: span.span().into(),
                        })?;
                let jump = layout.dfg_mut().new_value().jump(loop_boundary.entry);
                layout.push_inst(jump);
                Ok(true.into())
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
            ast::Expr::LVal(name) => {
                let var = symtable.get_var(&name.node.ident.node).ok_or(
                    CompileError::VariableNotFound {
                        span: name.span().into(),
                    },
                )?;
                match var {
                    Symbol::Const(num) => {
                        let dfg = layout.dfg_mut();
                        let num = dfg.new_value().integer(*num);
                        num
                    }
                    Symbol::Var(var, _, _) => {
                        let dfg = layout.dfg_mut();
                        let load = dfg.new_value().load(*var);
                        layout.push_inst(load)
                    }
                    Symbol::Func(_) => Err(CompileError::FuncAsVar {
                        span: name.span().into(),
                    })?,
                }
            }
            ast::Expr::Call(call) => {
                let span = call.node.ident.span().into();
                let func = symtable
                    .get_var(&call.node.ident.node)
                    .ok_or(CompileError::VariableNotFound { span })?;
                let func = match func {
                    Symbol::Const(_) => Err(CompileError::ConstNotAFunction { span })?,
                    Symbol::Var(_, _, _) => Err(CompileError::VarNotAFunction { span })?,
                    Symbol::Func(func) => *func,
                };

                let args = call
                    .node
                    .args
                    .into_iter()
                    .map(|arg| arg.build_ir_in(symtable, layout))
                    .collect::<Result<Vec<_>>>()?;

                let dfg = layout.dfg_mut();
                let call = dfg.new_value().call(func, args);
                layout.push_inst(call);

                call
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
            ast::Expr::LVal(name) => {
                let span = name.span().into();
                let var = symtable
                    .get_var(&name.node.ident.node)
                    .ok_or(CompileError::VariableNotFound { span })?;
                match var {
                    Symbol::Const(num) => *num,
                    Symbol::Var(_, _, _) => Err(CompileError::NonConstantExpression { span })?,
                    Symbol::Func(_) => Err(CompileError::FuncAsVar { span })?,
                }
            }
            ast::Expr::Call(_) => Err(CompileError::NonConstantExpression {
                span: self.span().into(),
            })?,
        })
    }
}

impl Span<ast::LVal> {
    /// Build IR for a left value.
    pub fn build_lval(self, symtable: &SymbolTable, layout: &mut LayoutBuilder) -> Result<Value> {
        let span = self.span().into();
        let var = symtable
            .get_var(&self.node.ident.node)
            .ok_or(CompileError::VariableNotFound { span })?;
        let var = match var {
            Symbol::Const(_) | Symbol::Var(_, _, true) => {
                Err(CompileError::AssignToConst { span })?
            }
            Symbol::Var(var, _, false) => *var,
            Symbol::Func(_) => Err(CompileError::FuncAsVar { span })?,
        };
        if let Some(index) = self.node.index {
            let index = index.build_ir_in(symtable, layout)?;
            let ptr = layout.dfg_mut().new_value().get_elem_ptr(var, index);
            Ok(layout.push_inst(ptr))
        } else {
            Ok(var)
        }
    }

    /// Build IR for a right value.
    pub fn build_rval(self, symtable: &SymbolTable, layout: &mut LayoutBuilder) -> Result<Value> {
        let name_span = self.node.ident.span().into();
        let var = symtable
            .get_var(&self.node.ident.node)
            .ok_or(CompileError::VariableNotFound { span: name_span })?;
        match var {
            Symbol::Const(num) => {
                let dfg = layout.dfg_mut();
                let num = dfg.new_value().integer(*num);
                if self.node.index.is_some() {
                    Err(CompileError::TypeError {
                        span: self.span().into(),
                        expected: "int".to_string(),
                        found: "int[]".to_string(),
                    })?;
                }
                Ok(num)
            }
            Symbol::Var(var, _, _) => {
                let var = if let Some(index) = self.node.index {
                    let index = index.build_ir_in(symtable, layout)?;
                    let ptr = layout.dfg_mut().new_value().get_elem_ptr(*var, index);
                    layout.push_inst(ptr)
                } else {
                    *var
                };
                let dfg = layout.dfg_mut();
                let load = dfg.new_value().load(var);
                Ok(layout.push_inst(load))
            }
            Symbol::Func(_) => Err(CompileError::FuncAsVar { span: name_span })?,
        }
    }
}
