//! Arbitrary AST generator.

use proptest::{
    prelude::*,
    strategy::{NewTree, ValueTree},
    test_runner::TestRunner,
};
use std::fmt::Debug;

use super::*;

/// Local environment.
#[derive(Debug, Clone, Default)]
struct LocalEnv {
    pub prev: Option<Box<LocalEnv>>,
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

    /// Generate an arbitrary free identifier.
    pub fn arb_free_lvar(self) -> impl Strategy<Value = String> {
        r"[a-z]"
            .prop_map(String::from)
            .prop_filter("new lvar must not be in consts or vars", move |ident| {
                !self.consts.contains(ident) && !self.vars.contains(ident)
            })
    }

    /// Consts in the environment.
    pub fn consts(self) -> Vec<String> {
        let mut consts = if let Some(prev) = self.prev {
            prev.consts()
        } else {
            Vec::new()
        };
        consts.extend(self.consts);
        consts.retain(|ident| !self.vars.contains(ident)); // remove shadowed vars
        consts
    }

    /// Lvars in the environment.
    pub fn lvars(self) -> Vec<String> {
        let mut lvars = if let Some(prev) = self.prev {
            prev.lvars()
        } else {
            Vec::new()
        };
        lvars.extend(self.vars);
        lvars.retain(|ident| !self.consts.contains(ident)); // remove shadowed consts
        lvars
    }

    /// Rvars in the environment.
    pub fn rvars(self) -> Vec<String> {
        let mut vars = Vec::new();
        let mut env = self;
        vars.extend(env.vars);
        vars.extend(env.consts);
        while let Some(prev) = env.prev {
            env = prev.as_ref().clone();
            vars.extend(prev.vars.into_iter());
            vars.extend(prev.consts.into_iter());
        }
        vars
    }

    /// Generate an arbitrary constant identifier.
    pub fn arb_const(self) -> Option<impl Strategy<Value = Span<String>>> {
        let consts = self.consts();
        if consts.is_empty() {
            return None;
        }
        let strat = Just(consts)
            .prop_filter("consts must not be empty", |consts| !consts.is_empty())
            .prop_flat_map(|consts| {
                prop::sample::select(consts).prop_map(|ident| ident.into_span(0, 0))
            });
        Some(strat)
    }

    /// Generate an arbitrary variable identifier.
    pub fn arb_lvar(self) -> Option<impl Strategy<Value = Span<String>>> {
        let lvars = self.lvars();
        if lvars.is_empty() {
            return None;
        }
        let strat = Just(lvars)
            .prop_filter("lvars must not be empty", |vars| !vars.is_empty())
            .prop_flat_map(|vars| {
                prop::sample::select(vars).prop_map(|ident| ident.into_span(0, 0))
            });
        Some(strat)
    }

    /// Generate an arbitrary variable identifier.
    pub fn arb_rvar(self) -> Option<impl Strategy<Value = Span<String>>> {
        let rvars = self.rvars();
        if rvars.is_empty() {
            return None;
        }
        let strat = Just(rvars)
            .prop_filter("rvars must not be empty", |vars| !vars.is_empty())
            .prop_flat_map(|vars| {
                prop::sample::select(vars).prop_map(|ident| ident.into_span(0, 0))
            });
        Some(strat)
    }
}

/// Generate an arbitrary program.
pub fn arb_comp_unit() -> impl Strategy<Value = CompUnit> {
    arb_func_def().prop_map(|func_def| CompUnit { func_def })
}

/// Generate an arbitrary function definition.
fn arb_func_def() -> impl Strategy<Value = FuncDef> {
    let s_ident = "main".prop_map(String::from);
    let s_block = BlockStrategy::new(LocalEnv::default(), 20);
    (arb_func_type(), s_ident, s_block).prop_map(|(func_type, ident, block)| FuncDef {
        func_type: func_type.into_span(0, 0),
        ident: ident.into_span(0, 0),
        block: block.into_span(0, 0),
    })
}

/// Generate an arbitrary function type.
fn arb_func_type() -> impl Strategy<Value = FuncType> {
    Just(FuncType::Int)
}

#[derive(Debug)]
struct BlockStrategy {
    local: LocalEnv,
    max_insts: usize,
}

impl BlockStrategy {
    pub fn new(local: LocalEnv, max_insts: usize) -> Self {
        assert!(max_insts > 0, "max_insts must be positive");
        BlockStrategy { local, max_insts }
    }
}

impl Strategy for BlockStrategy {
    type Tree = BlockValueTree;
    type Value = Block;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let mut local = LocalEnv {
            prev: Some(Box::new(self.local.clone())),
            ..Default::default()
        };
        let mut items = Vec::new();
        let mut insts = 0;

        while insts < self.max_insts {
            let mut weights = [
                1, // const decl
                2, // var decl
                5, // stmt
                2, // block
            ];
            if insts + 1 >= self.max_insts {
                weights[3] = 0;
            }
            let selected = runner
                .rng()
                .sample(rand::distributions::WeightedIndex::new(weights).unwrap());
            match selected {
                0 => {
                    // const decl
                    let name = local.clone().arb_free_lvar().new_tree(runner)?.current();
                    let decl = arb_const_decl(name.clone(), local.clone())
                        .prop_map(|decl| BlockItem::Decl {
                            decl: Decl::Const(decl.into_span(0, 0)),
                        })
                        .new_tree(runner)?;
                    items.push(BlockItemValueTree::Item(Box::new(decl)));
                    local.decl_const(name);
                    insts += 1;
                }
                1 => {
                    // var decl
                    let name = local.clone().arb_free_lvar().new_tree(runner)?.current();
                    let decl = arb_var_decl(name.clone(), local.clone())
                        .prop_map(|decl| BlockItem::Decl {
                            decl: Decl::Var(decl.into_span(0, 0)),
                        })
                        .new_tree(runner)?;
                    items.push(BlockItemValueTree::Item(Box::new(decl)));
                    local.decl_var(name);
                    insts += 1;
                }
                2 => {
                    // stmt
                    let stmt = arb_stmt(local.clone())
                        .prop_map(|stmt| BlockItem::Stmt {
                            stmt: stmt.into_span(0, 0),
                        })
                        .new_tree(runner)?;
                    items.push(BlockItemValueTree::Item(Box::new(stmt)));
                    insts += 1;
                }
                3 => {
                    // block
                    let block = BlockStrategy::new(local.clone(), (self.max_insts + 1) / 2)
                        .new_tree(runner)?;
                    insts += block.items_len();
                    items.push(BlockItemValueTree::Block(Box::new(block)));
                }
                _ => unreachable!(),
            }
        }

        Ok(BlockValueTree::new(items))
    }
}

/// Value tree for block.
///
/// # Partial Ordering
/// If `b1` contains `b2`, `b1 >= b2`.
struct BlockValueTree {
    items: Vec<BlockItemValueTree>,
    len: proptest::num::usize::BinarySearch,
    len_simplified: bool,
}

impl BlockValueTree {
    /// Generate an arbitrary block.
    pub fn new(items: Vec<BlockItemValueTree>) -> Self {
        let len = proptest::num::usize::BinarySearch::new(items.len());
        BlockValueTree {
            items,
            len,
            len_simplified: false,
        }
    }

    /// Get the number of items in the block.
    pub fn items_len(&self) -> usize {
        self.items
            .iter()
            .map(|item| match item {
                BlockItemValueTree::Item(_) => 1,
                BlockItemValueTree::Block(tree) => tree.items_len(),
            })
            .sum()
    }
}

impl ValueTree for BlockValueTree {
    type Value = Block;

    fn current(&self) -> Self::Value {
        let items = self
            .items
            .iter()
            .take(self.len.current())
            .map(|tree| tree.current())
            .collect();
        Block { items }
    }

    fn simplify(&mut self) -> bool {
        if self.len.simplify() {
            self.len_simplified = true;
            return true;
        }

        for items in self.items.iter_mut().rev() {
            if items.simplify() {
                return true;
            }
        }
        false
    }

    fn complicate(&mut self) -> bool {
        for items in self.items.iter_mut() {
            if items.complicate() {
                return true;
            }
        }

        if self.len_simplified {
            self.len.complicate()
        } else {
            false
        }
    }
}

/// Value tree for block item.
///
/// # Partial Ordering
/// For `i1 i2: Item`, `b1 b2: Block`,
/// `i1 < i2 -> b1 < b2 -> i1 < i2 < b1 < b2`.
///
/// If the block contains only one statement, it is seen equivalent
/// to the statement.
enum BlockItemValueTree {
    Item(Box<dyn ValueTree<Value = BlockItem>>),
    Block(Box<BlockValueTree>),
}

impl ValueTree for BlockItemValueTree {
    type Value = BlockItem;

    fn current(&self) -> Self::Value {
        match self {
            BlockItemValueTree::Item(tree) => tree.current(),
            BlockItemValueTree::Block(tree) => BlockItem::Stmt {
                stmt: Stmt::Block {
                    block: tree.current().into_span(0, 0),
                }
                .into_span(0, 0),
            },
        }
    }

    fn simplify(&mut self) -> bool {
        // try downgrade if the block contains only one statement
        if let BlockItemValueTree::Block(tree) = self {
            if tree.items.len() == 1 && matches!(tree.items[0].current(), BlockItem::Stmt { .. }) {
                *self = tree.items.pop().unwrap();
            }
        }

        match self {
            BlockItemValueTree::Item(tree) => tree.simplify(),
            BlockItemValueTree::Block(tree) => tree.simplify(),
        }
    }

    fn complicate(&mut self) -> bool {
        match self {
            BlockItemValueTree::Item(tree) => tree.complicate(),
            BlockItemValueTree::Block(tree) => tree.complicate(),
        }
    }
}

/// Generate an constant declaration.
fn arb_const_decl(name: String, local: LocalEnv) -> impl Strategy<Value = ConstDecl> {
    (arb_btype(), arb_const_def(name, local.clone())).prop_map(move |(ty, def)| ConstDecl {
        ty: ty.into_span(0, 0),
        defs: im::vector![def],
    })
}

/// Generate an arbitrary constant definition.
fn arb_const_def(name: String, local: LocalEnv) -> impl Strategy<Value = ConstDef> {
    let lvar = name.into_span(0, 0);
    arb_const_expr(local).prop_map(move |expr| ConstDef {
        ident: lvar.clone(),
        expr,
    })
}

/// Generate an arbitrary variable declaration.
fn arb_var_decl(name: String, local: LocalEnv) -> impl Strategy<Value = VarDecl> {
    (arb_btype(), arb_var_def(name, local.clone())).prop_map(move |(ty, def)| VarDecl {
        ty: ty.into_span(0, 0),
        defs: im::vector![def],
    })
}

fn arb_var_def(name: String, local: LocalEnv) -> impl Strategy<Value = VarDef> {
    let lvar = name.into_span(0, 0);
    prop::option::of(arb_init_val(local)).prop_map(move |init| VarDef {
        ident: lvar.clone(),
        init,
    })
}

fn arb_init_val(local: LocalEnv) -> impl Strategy<Value = InitVal> {
    arb_expr(local).prop_map(|expr| InitVal { expr })
}

fn arb_btype() -> impl Strategy<Value = BType> {
    Just(BType::Int)
}

fn arb_const_expr(local: LocalEnv) -> impl Strategy<Value = ConstExpr> {
    let s_number = arb_number_expr();
    let s_leaf = if let Some(s_const) = local.clone().arb_const() {
        let s_const = s_const.prop_map(Expr::LVar);
        prop_oneof![s_number, s_const].boxed()
    } else {
        s_number.boxed()
    };
    s_leaf
        .prop_recursive(3, 64, 10, |inner| {
            prop_oneof![arb_unary_expr(inner.clone()), arb_binary_expr(inner)]
        })
        .prop_map(|expr| ConstExpr { expr })
}

fn arb_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    let s_expr =
        prop::option::weighted(0.8, arb_expr(local.clone())).prop_map(|expr| Stmt::Expr { expr });
    let s_return = arb_return_stmt(local.clone());
    if let Some(s_assign) = arb_assign_stmt(local.clone()) {
        prop_oneof![
            10 => s_assign,
            10 => s_expr,
            1 => s_return,
        ]
        .boxed()
    } else {
        prop_oneof![
            10 => s_expr,
            1 => s_return,
        ]
        .boxed()
    }
}

fn arb_assign_stmt(local: LocalEnv) -> Option<impl Strategy<Value = Stmt>> {
    Some(
        (local.clone().arb_lvar()?, arb_expr(local))
            .prop_map(|(ident, expr)| Stmt::Assign { ident, expr }),
    )
}

fn arb_return_stmt(local: LocalEnv) -> impl Strategy<Value = Stmt> {
    arb_expr(local).prop_map(|expr| Stmt::Return { expr })
}

fn arb_expr(local: LocalEnv) -> impl Strategy<Value = Expr> {
    let s_number = arb_number_expr();
    let s_leaf = if let Some(s_rvar) = local.clone().arb_rvar() {
        let s_rvar = s_rvar.prop_map(Expr::LVar);
        prop_oneof![s_number, s_rvar].boxed()
    } else {
        s_number.boxed()
    };
    s_leaf.prop_recursive(3, 64, 10, |inner| {
        prop_oneof![arb_unary_expr(inner.clone()), arb_binary_expr(inner)]
    })
}

fn arb_unary_expr(s_expr: impl Strategy<Value = Expr>) -> impl Strategy<Value = Expr> {
    (arb_unary_op(), s_expr).prop_map(|(op, expr)| Expr::Unary {
        op: op.into_span(0, 0),
        expr: Box::new(expr),
    })
}

fn arb_binary_expr(s_expr: impl Strategy<Value = Expr> + Clone) -> impl Strategy<Value = Expr> {
    (s_expr.clone(), arb_binary_op(), s_expr).prop_map(|(lhs, op, rhs)| Expr::Binary {
        op: op.into_span(0, 0),
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    })
}

fn arb_number_expr() -> impl Strategy<Value = Expr> {
    (0..100i32).prop_map(|n| Expr::Number(n.into_span(0, 0)))
}

fn arb_unary_op() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![Just(UnaryOp::Pos), Just(UnaryOp::Neg), Just(UnaryOp::Not),]
}

fn arb_binary_op() -> impl Strategy<Value = BinaryOp> {
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

    fn count_decl<F>(block: &Block, criteria: &F) -> usize
    where
        F: Fn(&Decl) -> bool,
    {
        block
            .items
            .iter()
            .map(|item| match item {
                BlockItem::Decl { decl } => criteria(decl) as usize,
                BlockItem::Stmt { stmt } => match &stmt.node {
                    Stmt::Block { block } => count_decl(&block.node, criteria),
                    _ => 0,
                },
            })
            .sum()
    }

    fn count_stmt<F>(block: &Block, criteria: &F) -> usize
    where
        F: Fn(&Stmt) -> bool,
    {
        block
            .items
            .iter()
            .map(|item| match item {
                BlockItem::Stmt { stmt } => match &stmt.node {
                    Stmt::Block { block } => count_stmt(&block.node, criteria),
                    Stmt::Expr { expr } => criteria(&Stmt::Expr { expr: expr.clone() }) as usize,
                    Stmt::Assign { .. } => criteria(&stmt.node) as usize,
                    Stmt::Return { .. } => criteria(&stmt.node) as usize,
                },
                _ => 0,
            })
            .sum()
    }

    #[test]
    fn test_stmt_coverage() {
        const N: usize = 10000;

        let mut runner = TestRunner::default();
        let samples = samples(&mut runner, N);

        let mut decls = 0;
        let mut stmts = 0;

        for (i, ast) in samples.enumerate() {
            decls += count_decl(&ast.func_def.block.node, &|_| true);
            stmts += count_stmt(&ast.func_def.block.node, &|_| true);
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

        assert!(decl_ratio > 0.3, "too few decls, {decls} out of {total}",);
        assert!(stmt_ratio > 0.6, "too few stmts, {stmts} out of {total}",);
    }

    #[test]
    fn test_return_coverage() {
        const N: usize = 10000;

        let mut runner = TestRunner::default();
        let samples = samples(&mut runner, N);

        let mut have_returns = 0;
        let mut multiple_returns = 0;

        for (i, ast) in samples.enumerate() {
            let returns = count_stmt(&ast.func_def.block.node, &|stmt| {
                matches!(stmt, Stmt::Return { .. })
            });
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
            multiple_return_ratio < 0.3,
            "too many multiple_returns, {multiple_returns} out of {total}",
        );
    }
}
