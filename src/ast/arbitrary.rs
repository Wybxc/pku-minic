//! Arbitrary AST generator.

use proptest::prelude::*;

use super::*;

/// Local environment.
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
    pub fn enough_const_decls(&self) -> bool {
        !(self.consts.is_empty())
    }

    /// Check if there are not enough declarations.
    pub fn enough_var_decls(&self) -> bool {
        !(self.vars.is_empty())
    }

    /// Generate an arbitrary free identifier.
    pub fn arb_free_lvar(self) -> impl Strategy<Value = Span<String>> {
        r"[a-z][0-9]{0,2}"
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

    /// Generate an arbitrary variable identifier.
    pub fn arb_var_rvar(self) -> impl Strategy<Value = Span<String>> {
        let vars: Vec<_> = self.vars.into_iter().chain(self.consts).collect();
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
    let strg_ident = "main".prop_map(String::from);
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
    arb_block_item(local) // The first block item
        .prop_map(|(item, local)| (im::vector![item], local))
        .prop_recursive(
            32,
            64,
            1,
            |inner: BoxedStrategy<(im::Vector<BlockItem>, LocalEnv)>| {
                // Append a new block item to the end of the block
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
    let enough_const_decls = local.enough_const_decls();
    let enough_var_decls = local.enough_var_decls();

    let const_decl = arb_const_decl(local.clone()).prop_map(|(decl, local)| {
        let decl = Decl::Const(decl.into_span(0, 0));
        (BlockItem::Decl { decl }, local)
    });

    if !enough_const_decls {
        return const_decl.boxed();
    }

    let var_decl = arb_var_decl(local.clone()).prop_map(|(decl, local)| {
        let decl = Decl::Var(decl.into_span(0, 0));
        (BlockItem::Decl { decl }, local)
    });

    if !enough_var_decls {
        return var_decl.boxed();
    }

    let stmt = arb_stmt(local.clone()).prop_map(move |stmt| {
        let stmt = stmt.into_span(0, 0);
        (BlockItem::Stmt { stmt }, local.clone())
    });

    prop_oneof![1 => const_decl, 2 => var_decl, 12 => stmt].boxed()
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
    prop_oneof![
        5 => arb_assign_stmt(local.clone()),
        1 => arb_return_stmt(local),
    ]
}

pub fn arb_assign_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    (local.clone().arb_var_lvar(), arb_expr(local))
        .prop_map(|(ident, expr)| Stmt::Assign { ident, expr })
}

pub fn arb_return_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    arb_expr(local).prop_map(|expr| Stmt::Return { expr })
}

pub fn arb_expr(local: LocalEnv) -> impl Strategy<Value = Expr> {
    let leaf = if local.enough_var_decls() {
        prop_oneof![
            arb_number_expr(),
            local.clone().arb_var_rvar().prop_map(Expr::LVar)
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

#[cfg(all(test, feature = "arb-coverage"))]
mod test {
    use proptest::{strategy::ValueTree, test_runner::TestRunner};

    use super::*;

    struct Samples<'a, T> {
        total: usize,
        current: usize,
        failed: usize,
        gen: BoxedStrategy<T>,
        runner: &'a mut TestRunner,
    }

    impl<'a, T> Samples<'a, T> {
        fn new(gen: BoxedStrategy<T>, runner: &'a mut TestRunner, total: usize) -> Self {
            Samples {
                total,
                current: 0,
                failed: 0,
                gen,
                runner,
            }
        }
    }

    impl<'a, T> Iterator for Samples<'a, T>
    where
        T: std::fmt::Debug,
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.total {
                return None;
            }
            loop {
                match self.gen.new_tree(self.runner) {
                    Ok(tree) => {
                        self.current += 1;
                        return Some(tree.current());
                    }
                    Err(_) => {
                        self.failed += 1;
                        if self.failed > self.total / 2 {
                            panic!("too many failed samples");
                        }
                    }
                }
            }
        }
    }

    fn samples(runner: &mut TestRunner, n: usize) -> Samples<CompUnit> {
        let gen = arb_comp_unit().boxed();
        Samples::new(gen, runner, n)
    }

    #[test]
    fn test_stmt_coverage() {
        const N: usize = 10000;

        let mut runner = TestRunner::default();
        let samples = samples(&mut runner, N);

        let mut decls = 0;
        let mut stmts = 0;

        for (i, ast) in samples.enumerate() {
            for item in ast.func_def.block.node.items {
                match item {
                    BlockItem::Decl { .. } => decls += 1,
                    BlockItem::Stmt { .. } => stmts += 1,
                }
            }
            if i % 1000 == 0 {
                println!("{} samples generated", i);
            }
        }

        let total = decls + stmts;
        let decl_ratio = (decls as f64) / (total as f64);
        let stmt_ratio = (stmts as f64) / (total as f64);

        println!(
            "decls: {:.2}% stmts: {:.2}%",
            decl_ratio * 100.0,
            stmt_ratio * 100.0
        );

        assert!(decl_ratio > 0.4, "too few decls, {decls} out of {total}",);
        assert!(stmt_ratio > 0.5, "too few stmts, {stmts} out of {total}",);
    }

    #[test]
    fn test_return_coverage() {
        const N: usize = 10000;

        let mut runner = TestRunner::default();
        let samples = samples(&mut runner, N);

        let mut have_returns = 0;
        let mut multiple_returns = 0;

        for (i, ast) in samples.enumerate() {
            let returns = ast
                .func_def
                .block
                .node
                .items
                .iter()
                .filter(|item| match item {
                    BlockItem::Stmt { stmt } => matches!(stmt.node, Stmt::Return { .. }),
                    _ => false,
                })
                .count();
            if returns > 0 {
                have_returns += 1;
            }
            if returns > 1 {
                multiple_returns += 1;
            }

            if i % 1000 == 0 {
                println!("{} samples generated", i);
            }
        }

        let total = N;
        let have_return_ratio = (have_returns as f64) / (total as f64);
        let multiple_return_ratio = (multiple_returns as f64) / (total as f64);

        println!(
            "have_returns: {:.2}% multiple_returns: {:.2}%",
            have_return_ratio * 100.0,
            multiple_return_ratio * 100.0
        );

        assert!(
            have_return_ratio > 0.5,
            "too few have_returns, {have_returns} out of {total}",
        );

        assert!(
            multiple_return_ratio < 0.2,
            "too many multiple_returns, {multiple_returns} out of {total}",
        );
    }
}
