//! Build IR from AST.

use imbl::Vector;
use koopa::ir::{builder_traits::*, dfg::DataFlowGraph, *};
use miette::Result;

use crate::{
    ast::{self, ConstExpr, NonSpanned, Span, Spanned},
    irgen::{
        context::Context,
        layout::LayoutBuilder,
        metadata::{FunctionMetadata, ProgramMetadata},
        types::{IndexCast, VType},
    },
};

mod context;
mod error;
mod layout;
pub mod metadata;
mod symtable;
mod types;

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
        self.decl_function(symtable, program, "@getint", [], VType::Int);
        // decl @getch(): i32
        self.decl_function(symtable, program, "@getch", [], VType::Int);
        // decl @getarray(*i32): i32
        self.decl_function(
            symtable,
            program,
            "@getarray",
            [VType::Int.into_ptr()],
            VType::Int,
        );
        // decl @putint(i32)
        self.decl_function(symtable, program, "@putint", [VType::Int], VType::Void);
        // decl @putch(i32)
        self.decl_function(symtable, program, "@putch", [VType::Int], VType::Void);
        // decl @putarray(i32, *i32)
        self.decl_function(
            symtable,
            program,
            "@putarray",
            [VType::Int, VType::Int.into_ptr()],
            VType::Void,
        );
        // decl @starttime()
        self.decl_function(symtable, program, "@starttime", [], VType::Void);
        // decl @stoptime()
        self.decl_function(symtable, program, "@stoptime", [], VType::Void);

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
        params_ty: impl Into<Vector<VType>>,
        ret_ty: VType,
    ) {
        let params_ty = params_ty.into();
        let func = program.new_func(FunctionData::new_decl(
            name.into(),
            params_ty.iter().map(VType::to_koopa).collect(),
            ret_ty.to_koopa(),
        ));
        if symtable
            .insert_var(name[1..].into(), Symbol::Func(func, params_ty, ret_ty))
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

        let param_idents = self
            .params
            .iter()
            .map(|p| p.node.ident.clone())
            .collect::<Vec<_>>();
        let params_ty = self
            .params
            .into_iter()
            .map(|p| p.node.get_type(symtable))
            .collect::<Result<Vector<_>>>()?;
        let params = param_idents
            .iter()
            .zip(params_ty.iter())
            .map(|(ident, ty)| (Some(format!("@{}", ident.node)), ty.to_koopa()))
            .collect();
        let ret_type = self.func_type.node.build_primitive();
        let func = program.new_func(FunctionData::with_param_names(
            kp_name,
            params,
            ret_type.to_koopa(),
        ));
        prog_metadata.functions.insert(func, metadata);

        // add function to global symbol table
        if symtable
            .insert_var(
                self.ident.node,
                Symbol::Func(func, params_ty.clone(), ret_type),
            )
            .is_some()
        {
            Err(CompileError::FunctionDefinedTwice { span })?;
        }

        let mut context = Context::new();
        let mut layout = LayoutBuilder::new(program.func_mut(func), self.func_type.node);

        // alloc var for parameters
        symtable.push();
        let params = layout.params().to_vec();
        for ((value, ident), ty) in params.into_iter().zip(param_idents).zip(params_ty) {
            let span = ident.span().into();
            let dfg = layout.dfg_mut();
            let var = dfg.new_value().alloc(ty.to_koopa());
            let store = dfg.new_value().store(value, var);
            dfg.set_value_name(var, Some(format!("%{}", ident.node)));
            layout.push_insts([var, store]);
            if symtable
                .insert_var(ident.node, Symbol::Var(var, ty, false))
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
    pub fn get_type(self, symtable: &SymbolTable) -> Result<VType> {
        Ok(if let Some(indices) = self.indices {
            self.ty.node.build_ir(indices, symtable)?.0.into_ptr()
        } else {
            self.ty.node.build_primitive()
        })
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
    pub fn build_primitive(&self) -> VType {
        match self {
            ast::BType::Int => VType::Int,
            ast::BType::Void => VType::Void,
        }
    }

    /// Build IR from AST for array types.
    pub fn build_ir(
        &self,
        indices: Vec<ConstExpr>,
        symtable: &SymbolTable,
    ) -> Result<(VType, Vec<i32>)> {
        let indices = indices
            .into_iter()
            .map(|expr| expr.const_eval(symtable))
            .collect::<Result<Vec<_>>>()?;
        let mut ty = self.build_primitive();
        for index in indices.iter().copied().rev() {
            ty = ty.into_array(index);
        }
        Ok((ty, indices))
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
        if self.indices.is_empty() {
            // Constant variable.
            self.build_const(symtable)
        } else {
            // Get the type of the array.
            let (ty, indices) = ty.build_ir(self.indices, symtable)?;

            // Allocate a new array.
            let array = layout.dfg_mut().new_value().alloc(ty.to_koopa());
            layout.push_inst(array);

            // Initialize the array.
            self.init
                .canonicalize(&indices)?
                .build_ir_in(array, symtable, layout, true)?;

            // Insert the array into the symbol table.
            if symtable
                .insert_var(self.ident.node, Symbol::Var(array, ty, true))
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
        let span = self.ident.span().into();
        if self.indices.is_empty() {
            // Constant variable.
            self.build_const(symtable)
        } else {
            // Get the type of the array.
            let (ty, indices) = ty.build_ir(self.indices, symtable)?;

            // Initialize the array.
            let init = self
                .init
                .canonicalize(&indices)?
                .build_ir_global(symtable, program)?;

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
        }
    }

    /// Build IR from AST for constant variables (not arrays).
    fn build_const(self, symtable: &mut SymbolTable) -> Result<()> {
        let span = self.ident.span().into();
        let expr = self.init.canonicalize(&[])?.const_eval(symtable)?;
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

        // Get the type of the array or variable.
        let (ty, indices) = ty.build_ir(self.indices, symtable)?;

        // Allocate a new variable.
        let dfg = layout.dfg_mut();
        let var = dfg.new_value().alloc(ty.to_koopa());
        dfg.set_value_name(var, Some(format!("@{}", self.ident.node)));
        layout.push_inst(var);

        // Initialize the variable.
        if let Some(init) = self.init {
            init.canonicalize(&indices)?
                .build_ir_in(var, symtable, layout, false)?;
        }

        // Insert the array into the symbol table.
        if symtable
            .insert_var(self.ident.node, Symbol::Var(var, ty, false))
            .is_some()
        {
            Err(CompileError::VariableDefinedTwice { span })?;
        }

        Ok(())
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

        // Get the type of the array or variable.
        let (ty, indices) = ty.build_ir(self.indices, symtable)?;

        // Initialize the variable.
        let init = if let Some(init) = self.init {
            init.canonicalize(&indices)?
                .build_ir_global(symtable, program)?
        } else {
            program.new_value().zero_init(ty.to_koopa())
        };

        // Allocate a new variable.
        let var = program.new_value().global_alloc(init);
        program.set_value_name(var, Some(format!("@{}", self.ident.node)));

        // Insert the variable into the symbol table.
        if symtable
            .insert_var(self.ident.node, Symbol::Var(var, ty, false))
            .is_some()
        {
            Err(CompileError::VariableDefinedTwice { span })?;
        }

        Ok(())
    }
}

impl Span<ast::InitVal> {
    /// Canonicalize the initializer.
    pub fn canonicalize(self, indices: &[i32]) -> Result<Span<ast::InitVal>> {
        let span = self.span().into();
        let new_ce = || CompileError::InvalidInitializer { span };

        if indices.is_empty() {
            // not an array
            return match self.node {
                ast::InitVal::Expr(_) => Ok(self),
                ast::InitVal::InitList(inits) => {
                    if inits.len() != 1 {
                        Err(new_ce())?;
                    }
                    let init = inits.into_iter().next().unwrap();
                    match init.node {
                        ast::InitVal::Expr(_) => Ok(init),
                        ast::InitVal::InitList(_) => Err(new_ce())?,
                    }
                }
            };
        }

        let inits = match self.node {
            ast::InitVal::Expr(_) => Err(new_ce())?,
            ast::InitVal::InitList(inits) => inits,
        };

        let mut digits = vec![vec![]; indices.len()];

        for init in inits {
            let (base, init) = match init.node {
                ast::InitVal::Expr(_) => (indices.len() - 1, init),
                ast::InitVal::InitList(_) => {
                    let base = digits.iter().rposition(|d| !d.is_empty()).unwrap_or(0);
                    let sub_indices = &indices[base + 1..];
                    let init = init.canonicalize(sub_indices)?;
                    (base, init)
                }
            };
            let mut carry = Some(init);
            for i in (0..=base).rev() {
                if let Some(carry) = carry.take() {
                    digits[i].push(carry);
                } else {
                    break;
                }
                if digits[i].len() >= indices[i] as usize {
                    carry = Some(
                        ast::InitVal::InitList(std::mem::take(&mut digits[i])).into_span(0, 0),
                    );
                }
            }
            if let Some(carry) = carry {
                return Ok(carry);
            }
        }

        // pad zeros
        let mut carry = None;
        for i in (0..indices.len()).rev() {
            debug_assert!(digits[i].len() < indices[i] as usize);
            if let Some(carry) = carry.take() {
                digits[i].push(carry);
            }
            digits[i].resize(indices[i] as usize, Self::zeroed(&indices[i + 1..]));
            carry = Some(ast::InitVal::InitList(std::mem::take(&mut digits[i])).into_span(0, 0));
        }
        let carry = carry.unwrap();
        Ok(carry)
    }

    /// Return a zeroed initializer list.
    fn zeroed(indices: &[i32]) -> Span<ast::InitVal> {
        match indices {
            [] => ast::InitVal::Expr(ast::Expr::Number(0.into_span(0, 0))).into_span(0, 0),
            [index, indices @ ..] => ast::InitVal::InitList(
                (0..*index)
                    .map(|_| Self::zeroed(indices))
                    .collect::<Vec<_>>(),
            )
            .into_span(0, 0),
        }
    }

    /// Build IR from AST in an existing IR node.
    pub fn build_ir_in(
        self,
        ptr: Value,
        symtable: &SymbolTable,
        layout: &mut LayoutBuilder,
        is_const: bool,
    ) -> Result<()> {
        match self.node {
            ast::InitVal::Expr(expr) => {
                let expr = if is_const {
                    let val = expr.const_eval(symtable)?;
                    layout.dfg_mut().new_value().integer(val)
                } else {
                    expr.build_ir_in(symtable, layout, Some(VType::Int))?
                };
                let store = layout.dfg_mut().new_value().store(expr, ptr);
                layout.push_inst(store);
            }
            ast::InitVal::InitList(inits) => {
                for (i, init) in inits.into_iter().enumerate() {
                    let dfg = layout.dfg_mut();
                    let index = dfg.new_value().integer(i as i32);
                    let ptr = dfg.new_value().get_elem_ptr(ptr, index);
                    layout.push_inst(ptr);
                    init.build_ir_in(ptr, symtable, layout, is_const)?;
                }
            }
        }
        Ok(())
    }

    /// Build IR from AST for global variables.
    pub fn build_ir_global(self, symtable: &SymbolTable, program: &mut Program) -> Result<Value> {
        match self.node {
            ast::InitVal::Expr(expr) => {
                let val = expr.const_eval(symtable)?;
                Ok(program.new_value().integer(val))
            }
            ast::InitVal::InitList(inits) => {
                let values = inits
                    .into_iter()
                    .map(|init| init.build_ir_global(symtable, program))
                    .collect::<Result<Vec<_>>>()?;
                Ok(program.new_value().aggregate(values))
            }
        }
    }

    /// Const evaluation.
    pub fn const_eval(self, symtable: &SymbolTable) -> Result<i32> {
        match self.node {
            ast::InitVal::Expr(expr) => expr.const_eval(symtable),
            ast::InitVal::InitList(_) => unreachable!(),
        }
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
                let (var, ty) = lval.build_ir_in(symtable, layout)?;

                // Compute the expression.
                let expr = expr.build_ir_in(symtable, layout, Some(ty))?;

                // Store the value.
                let store = layout.dfg_mut().new_value().store(expr, var);
                layout.push_inst(store);
                Ok(false.into())
            }
            ast::Stmt::Return { expr } => {
                let expr = expr.build_ir_in(symtable, layout, Some(VType::Int))?;
                let dfg = layout.dfg_mut();
                let ret = dfg.new_value().ret(Some(expr));
                layout.push_inst(ret);
                Ok(true.into())
            }
            ast::Stmt::Expr { expr } => {
                if let Some(expr) = expr {
                    expr.build_ir_in(symtable, layout, None)?;
                }
                Ok(false.into())
            }
            ast::Stmt::Block { block } => block.node.build_ir_in(symtable, context, layout),
            ast::Stmt::If { cond, then, els } => {
                let cond = cond.build_ir_in(symtable, layout, Some(VType::Int))?;

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

                let cond = cond.build_ir_in(symtable, layout, Some(VType::Int))?;
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
    pub fn build_ir_in(
        self,
        symtable: &SymbolTable,
        layout: &mut LayoutBuilder,
        expected_type: Option<VType>,
    ) -> Result<Value> {
        let span = self.span().into();
        match self {
            ast::Expr::Unary { .. } | ast::Expr::Binary { .. } | ast::Expr::Number(_) => {
                if let Some(expected_type) = &expected_type {
                    if !expected_type.is_int() {
                        Err(CompileError::TypeError {
                            span,
                            expected: expected_type.to_string(),
                            found: "int".to_string(),
                        })?;
                    }
                }
            }
            _ => {}
        }
        Ok(match self {
            ast::Expr::Unary { op, expr } => {
                let expr = expr.build_ir_in(symtable, layout, Some(VType::Int))?;
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
                        let lhs = lhs.build_ir_in(symtable, layout, Some(VType::Int))?;
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
                            let rhs = rhs.build_ir_in(symtable, layout, Some(VType::Int))?;
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
                        let lhs = lhs.build_ir_in(symtable, layout, Some(VType::Int))?;
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
                            let rhs = rhs.build_ir_in(symtable, layout, Some(VType::Int))?;
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
                        let lhs = lhs.build_ir_in(symtable, layout, Some(VType::Int))?;
                        let rhs = rhs.build_ir_in(symtable, layout, Some(VType::Int))?;
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
            ast::Expr::LVal(lval) => {
                let (var, ty) = lval.build_ir_in(symtable, layout)?;
                if let Some(expected_type) = &expected_type {
                    match ty.cast(expected_type) {
                        types::TypeCast::Noop => {}
                        types::TypeCast::ArrayToPtr => {
                            // T[N][..]* -> T[..]*
                            let dfg = layout.dfg_mut();
                            let index = dfg.new_value().integer(0);
                            let ptr = dfg.new_value().get_elem_ptr(var, index);
                            return Ok(layout.push_inst(ptr));
                        }
                        types::TypeCast::Fail => Err(CompileError::TypeError {
                            span,
                            expected: expected_type.to_string(),
                            found: ty.to_string(),
                        })?,
                    }
                }
                // T* -> T
                let load = layout.dfg_mut().new_value().load(var);
                layout.push_inst(load)
            }
            ast::Expr::Call(call) => {
                let span = call.node.ident.span().into();
                let func = symtable
                    .get_var(&call.node.ident.node)
                    .ok_or(CompileError::VariableNotFound { span })?;
                let (func, params_ty, ret_ty) = match func {
                    Symbol::Const(_) => Err(CompileError::ConstNotAFunction { span })?,
                    Symbol::Var(_, _, _) => Err(CompileError::VarNotAFunction { span })?,
                    Symbol::Func(func, params_ty, ret_ty) => {
                        (*func, params_ty.clone(), ret_ty.clone())
                    }
                };

                if let Some(expected_type) = &expected_type {
                    if &ret_ty != expected_type {
                        Err(CompileError::TypeError {
                            span,
                            expected: expected_type.to_string(),
                            found: ret_ty.to_string(),
                        })?;
                    }
                }

                let args = call
                    .node
                    .args
                    .into_iter()
                    .zip(params_ty.into_iter())
                    .map(|(arg, ty)| arg.build_ir_in(symtable, layout, Some(ty)))
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
                    Symbol::Func(_, _, _) => Err(CompileError::FuncAsVar { span })?,
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
    pub fn build_ir_in(
        self,
        symtable: &SymbolTable,
        layout: &mut LayoutBuilder,
    ) -> Result<(Value, VType)> {
        let span = self.span().into();
        let var = symtable
            .get_var(&self.node.ident.node)
            .ok_or(CompileError::VariableNotFound { span })?;
        let (var, ty) = match var {
            Symbol::Const(_) | Symbol::Var(_, _, true) => {
                Err(CompileError::AssignToConst { span })?
            }
            Symbol::Var(var, ty, false) => (*var, ty.clone()),
            Symbol::Func(_, _, _) => Err(CompileError::FuncAsVar { span })?,
        };
        let (var, ty) = self.indices(var, ty, symtable, layout)?;
        Ok((var, ty))
    }

    fn indices(
        self,
        mut var: Value,
        mut ty: VType,
        symtable: &SymbolTable,
        layout: &mut LayoutBuilder,
    ) -> Result<(Value, VType)> {
        let span = self.span().into();
        for index in self.node.indices {
            let index = index.build_ir_in(symtable, layout, Some(VType::Int))?;
            let (ptr, new_ty) = match ty.to_indexed() {
                IndexCast::Array(new_ty) => {
                    // T[N][..]* -> T[..]*
                    let ptr = layout.dfg_mut().new_value().get_elem_ptr(var, index);
                    (ptr, new_ty)
                }
                IndexCast::Ptr(new_ty) => {
                    // T** -> T*
                    let ptr = layout.dfg_mut().new_value().load(var);
                    layout.push_inst(ptr);
                    let ptr = layout.dfg_mut().new_value().get_ptr(ptr, index);
                    (ptr, new_ty)
                }
                IndexCast::Fail => Err(CompileError::TypeError {
                    span,
                    expected: format!("{}[]", ty),
                    found: ty.to_string(),
                })?,
            };
            var = layout.push_inst(ptr);
            ty = new_ty;
        }
        Ok((var, ty))
    }
}
