use proptest::prelude::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct LocalEnv {
    consts: im::HashSet<String>,
    vars: im::HashSet<String>,
}

impl LocalEnv {
    /// Declare a constant.
    pub fn decl_const(&mut self, ident: String) {
        self.consts.insert(ident);
    }

    /// Declare a variable.
    pub fn decl_var(&mut self, ident: String) {
        self.vars.insert(ident);
    }

    /// Check if there are not enough declarations.
    pub fn enough_decls(&self) -> bool {
        !(self.consts.is_empty() || self.vars.is_empty())
    }

    /// Generate an arbitrary free identifier.
    pub fn arb_free_lvar(self) -> impl Strategy<Value = Span<String>> {
        "[a-zA-Z_][a-zA-Z0-9_]*"
            .prop_map(String::from)
            .prop_filter("new lvar must not be in consts or vars", move |ident| {
                !self.consts.contains(ident) && !self.vars.contains(ident)
            })
            .prop_map(|ident| ident.into_span(0, 0))
    }

    /// Generate an arbitrary constant identifier.
    #[allow(dead_code)]
    pub fn arb_const_lvar(self) -> impl Strategy<Value = Span<String>> {
        let consts: Vec<_> = self.consts.into_iter().collect();
        Just(consts)
            .prop_filter("consts must not be empty", |consts| !consts.is_empty())
            .prop_flat_map(|consts| {
                prop::sample::select(consts).prop_map(|ident| ident.into_span(0, 0))
            })
    }

    /// Generate an arbitrary variable identifier.
    pub fn arb_var_lvar(self) -> impl Strategy<Value = Span<String>> {
        let vars: Vec<_> = self.vars.into_iter().collect();
        Just(vars)
            .prop_filter("vars must not be empty", |vars| !vars.is_empty())
            .prop_flat_map(|vars| {
                prop::sample::select(vars).prop_map(|ident| ident.into_span(0, 0))
            })
    }
}

/// Generate an arbitrary program.
pub fn arb_comp_unit() -> impl Strategy<Value = CompUnit> {
    arb_func_def().prop_map(|func_def| CompUnit { func_def })
}

/// Generate an arbitrary function definition.
pub fn arb_func_def() -> impl Strategy<Value = FuncDef> {
    let strg_ident = "[a-zA-Z_][a-zA-Z0-9_]*".prop_map(String::from);
    (arb_func_type(), strg_ident, arb_block(LocalEnv::default())).prop_map(
        |(func_type, ident, (block, _))| FuncDef {
            func_type: func_type.into_span(0, 0),
            ident: ident.into_span(0, 0),
            block: block.into_span(0, 0),
        },
    )
}

/// Generate an arbitrary function type.
pub fn arb_func_type() -> impl Strategy<Value = FuncType> {
    Just(FuncType::Int)
}

/// Generate an arbitrary block.
pub fn arb_block(local: LocalEnv) -> impl Strategy<Value = (Block, LocalEnv)> {
    arb_block_item(local)
        .prop_map(|(item, local)| (im::vector![item], local))
        .prop_recursive(
            32,
            64,
            10,
            |inner: BoxedStrategy<(im::Vector<BlockItem>, LocalEnv)>| {
                inner.prop_flat_map(|(block, local)| {
                    arb_block_item(local).prop_map(move |(item, local)| {
                        let mut block = block.clone();
                        block.push_back(item);
                        (block, local)
                    })
                })
            },
        )
        .prop_map(|(items, local)| {
            let block = Block { items };
            (block, local)
        })
}

/// Generate an arbitrary block item.
pub fn arb_block_item(local: LocalEnv) -> impl Strategy<Value = (BlockItem, LocalEnv)> {
    let enough_decls = local.enough_decls();
    let decl = arb_decl(local.clone()).prop_map(|(decl, local)| (BlockItem::Decl { decl }, local));
    let stmt = arb_stmt(local.clone()).prop_map(move |stmt| {
        let stmt = stmt.into_span(0, 0);
        (BlockItem::Stmt { stmt }, local.clone())
    });
    if !enough_decls {
        decl.boxed()
    } else {
        prop_oneof![decl, stmt].boxed()
    }
}

/// Generate an arbitrary declaration.
pub fn arb_decl(local: LocalEnv) -> impl Strategy<Value = (Decl, LocalEnv)> {
    let const_decl = arb_const_decl(local.clone())
        .prop_map(|(decl, local)| (Decl::Const(decl.into_span(0, 0)), local));
    let var_decl = arb_var_decl(local.clone())
        .prop_map(|(decl, local)| (Decl::Var(decl.into_span(0, 0)), local));
    prop_oneof![1 => const_decl, 2 => var_decl]
}

/// Generate an constant declaration.
pub fn arb_const_decl(local: LocalEnv) -> impl Strategy<Value = (ConstDecl, LocalEnv)> {
    (arb_btype(), arb_const_def(local.clone())).prop_map(move |(ty, def)| {
        let decl = ConstDecl {
            ty: ty.into_span(0, 0),
            defs: im::vector![def.clone()],
        };
        let mut local = local.clone();
        local.decl_const(def.ident.node.clone());
        (decl, local)
    })
}

/// Generate an arbitrary constant definition.
pub fn arb_const_def(local: LocalEnv) -> impl Strategy<Value = ConstDef> {
    let lvar = local.clone().arb_free_lvar();
    (lvar, arb_const_expr(local)).prop_map(|(ident, expr)| ConstDef { ident, expr })
}

/// Generate an arbitrary variable declaration.
pub fn arb_var_decl(local: LocalEnv) -> impl Strategy<Value = (VarDecl, LocalEnv)> {
    (arb_btype(), arb_var_def(local.clone())).prop_map(move |(ty, def)| {
        let decl = VarDecl {
            ty: ty.into_span(0, 0),
            defs: im::vector![def.clone()],
        };
        let mut local = local.clone();
        local.decl_var(def.ident.node.clone());
        (decl, local)
    })
}

pub fn arb_var_def(local: LocalEnv) -> impl Strategy<Value = VarDef> {
    let lvar = local.clone().arb_free_lvar();
    (lvar, prop::option::of(arb_init_val(local))).prop_map(|(ident, init)| VarDef { ident, init })
}

pub fn arb_init_val(local: LocalEnv) -> impl Strategy<Value = InitVal> {
    arb_expr(local).prop_map(|expr| InitVal { expr })
}

pub fn arb_btype() -> impl Strategy<Value = BType> {
    Just(BType::Int)
}

pub fn arb_const_expr(_local: LocalEnv) -> impl Strategy<Value = ConstExpr> {
    // There might be div-by-zero errors, so cannot generate arbitrary expressions.
    let expr = (1..100i32).prop_map(|n| Expr::Number(n.into_span(0, 0)));
    expr.prop_map(|expr| ConstExpr { expr })
}

pub fn arb_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    prop_oneof![arb_assign_stmt(local.clone()), arb_return_stmt(local),]
}

pub fn arb_assign_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    (local.clone().arb_var_lvar(), arb_expr(local))
        .prop_map(|(ident, expr)| Stmt::Assign { ident, expr })
}

pub fn arb_return_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    arb_expr(local).prop_map(|expr| Stmt::Return { expr })
}

pub fn arb_expr(local: LocalEnv) -> impl Strategy<Value = Expr> {
    let leaf = if local.enough_decls() {
        prop_oneof![
            arb_number_expr(),
            local.clone().arb_var_lvar().prop_map(Expr::LVar)
        ]
        .boxed()
    } else {
        arb_number_expr().boxed()
    };
    leaf.prop_recursive(3, 64, 10, |inner| {
        prop_oneof![arb_unary_expr(inner.clone()), arb_binary_expr(inner),]
    })
}

pub fn arb_unary_expr(strg_expr: impl Strategy<Value = Expr>) -> impl Strategy<Value = Expr> {
    (arb_unary_op(), strg_expr).prop_map(|(op, expr)| Expr::Unary {
        op: op.into_span(0, 0),
        expr: Box::new(expr),
    })
}

pub fn arb_binary_expr(
    strg_expr: impl Strategy<Value = Expr> + Clone,
) -> impl Strategy<Value = Expr> {
    (strg_expr.clone(), arb_binary_op(), strg_expr).prop_map(|(lhs, op, rhs)| Expr::Binary {
        op: op.into_span(0, 0),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    })
}

pub fn arb_number_expr() -> impl Strategy<Value = Expr> {
    (0..100i32).prop_map(|n| Expr::Number(n.into_span(0, 0)))
}

pub fn arb_unary_op() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![Just(UnaryOp::Pos), Just(UnaryOp::Neg), Just(UnaryOp::Not),]
}

pub fn arb_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![
        Just(BinaryOp::Add),
        Just(BinaryOp::Sub),
        Just(BinaryOp::Mul),
        Just(BinaryOp::Div),
        Just(BinaryOp::Mod),
        Just(BinaryOp::Eq),
        Just(BinaryOp::Ne),
        Just(BinaryOp::Lt),
        Just(BinaryOp::Le),
        Just(BinaryOp::Gt),
        Just(BinaryOp::Ge),
        Just(BinaryOp::LAnd),
        Just(BinaryOp::LOr),
    ]
}
