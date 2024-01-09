//! Arbitrary AST generator.

use std::fmt::Debug;

use proptest::{
    prelude::*,
    strategy::{NewTree, ValueTree},
    test_runner::TestRunner,
};

use super::*;

/// Local environment.
#[derive(Debug, Clone, Default)]
struct LocalEnv {
    pub prev: Option<Box<LocalEnv>>,
    consts: imbl::HashSet<String>,
    vars: imbl::HashSet<String>,
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
                3, // var decl
                5, // stmt
                2, // block
                2, // if
                1, // while
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
                4 => {
                    // if
                    let cond = arb_expr(local.clone()).new_tree(runner)?;
                    let then = BlockStrategy::new(local.clone(), (self.max_insts + 1) / 2)
                        .new_tree(runner)?;
                    let els = BlockStrategy::new(local.clone(), (self.max_insts + 1) / 2)
                        .new_tree(runner)?;
                    insts += then.items_len() + els.items_len();
                    items.push(BlockItemValueTree::If {
                        cond: Box::new(cond),
                        then: Box::new(then),
                        els: Some(Box::new(els)),
                    });
                }
                5 => {
                    // while
                    let cond = arb_expr(local.clone()).new_tree(runner)?;
                    let body = BlockStrategy::new(local.clone(), (self.max_insts + 1) / 2)
                        .new_tree(runner)?;
                    insts += body.items_len();
                    items.push(BlockItemValueTree::While {
                        cond: Box::new(cond),
                        body: Box::new(body),
                    });
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
                BlockItemValueTree::If { then, els, .. } => {
                    then.items_len() + els.as_ref().map(|els| els.items_len()).unwrap_or(0)
                }
                BlockItemValueTree::While { body, .. } => body.items_len(),
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
    If {
        cond: Box<dyn ValueTree<Value = Expr>>,
        then: Box<BlockValueTree>,
        els: Option<Box<BlockValueTree>>,
    },
    While {
        cond: Box<dyn ValueTree<Value = Expr>>,
        body: Box<BlockValueTree>,
    },
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
            BlockItemValueTree::If { cond, then, els } => BlockItem::Stmt {
                stmt: Stmt::If {
                    cond: cond.current(),
                    then: Box::new(
                        Stmt::Block {
                            block: then.current().into_span(0, 0),
                        }
                        .into_span(0, 0),
                    ),
                    els: els.as_ref().map(|els| {
                        Box::new(
                            Stmt::Block {
                                block: els.current().into_span(0, 0),
                            }
                            .into_span(0, 0),
                        )
                    }),
                }
                .into_span(0, 0),
            },
            BlockItemValueTree::While { cond, body } => BlockItem::Stmt {
                stmt: Stmt::While {
                    cond: cond.current(),
                    body: Box::new(
                        Stmt::Block {
                            block: body.current().into_span(0, 0),
                        }
                        .into_span(0, 0),
                    ),
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
            BlockItemValueTree::If { cond, then, els } => {
                if cond.simplify() {
                    return true;
                }
                if then.simplify() {
                    return true;
                }
                if let Some(els) = els {
                    els.simplify()
                } else {
                    false
                }
            }
            BlockItemValueTree::While { cond, body } => {
                if cond.simplify() {
                    return true;
                }
                body.simplify()
            }
        }
    }

    fn complicate(&mut self) -> bool {
        match self {
            BlockItemValueTree::Item(tree) => tree.complicate(),
            BlockItemValueTree::Block(tree) => tree.complicate(),
            BlockItemValueTree::If { cond, then, els } => {
                if let Some(els) = els {
                    if els.complicate() {
                        return true;
                    }
                }
                if then.complicate() {
                    return true;
                }
                cond.complicate()
            }
            BlockItemValueTree::While { cond, body } => {
                if body.complicate() {
                    return true;
                }
                cond.complicate()
            }
        }
    }
}

/// Generate an constant declaration.
fn arb_const_decl(name: String, local: LocalEnv) -> impl Strategy<Value = ConstDecl> {
    (arb_btype(), arb_const_def(name, local)).prop_map(move |(ty, def)| ConstDecl {
        ty: ty.into_span(0, 0),
        defs: imbl::vector![def],
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
    (arb_btype(), arb_var_def(name, local)).prop_map(move |(ty, def)| VarDecl {
        ty: ty.into_span(0, 0),
        defs: imbl::vector![def],
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
    let s_leaf = if let Some(s_const) = local.arb_const() {
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
    if let Some(s_assign) = arb_assign_stmt(local) {
        prop_oneof![
            10 => s_assign,
            10 => s_expr,
            1 => s_return,
        ]
        .boxed()
    } else {
        prop_oneof![
            20 => s_expr,
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
    let s_leaf = if let Some(s_rvar) = local.arb_rvar() {
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
        T: Debug,
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
        fn count<F>(stmt: &Stmt, criteria: &F) -> usize
        where
            F: Fn(&Decl) -> bool,
        {
            match stmt {
                Stmt::Block { block } => count_decl(&block.node, criteria),
                Stmt::If { then, els, .. } => {
                    let count_then = count(&then.node, criteria);
                    let count_els = els
                        .as_ref()
                        .map(|els| count(&els.node, criteria))
                        .unwrap_or(0);
                    count_then + count_els
                }
                Stmt::While { body, .. } => count(&body.node, criteria),
                _ => 0,
            }
        }

        block
            .items
            .iter()
            .map(|item| match item {
                BlockItem::Decl { decl } => criteria(decl) as usize,
                BlockItem::Stmt { stmt } => count(&stmt.node, criteria),
            })
            .sum()
    }

    fn count_stmt<F>(block: &Block, criteria: &F) -> usize
    where
        F: Fn(&Stmt) -> bool,
    {
        fn count<F>(stmt: &Stmt, criteria: &F) -> usize
        where
            F: Fn(&Stmt) -> bool,
        {
            match stmt {
                Stmt::Block { block } => count_stmt(&block.node, criteria),
                Stmt::If { then, els, .. } => {
                    let count_self = criteria(stmt) as usize;
                    let count_then = count(&then.node, criteria);
                    let count_els = els
                        .as_ref()
                        .map(|els| count(&els.node, criteria))
                        .unwrap_or(0);
                    count_self + count_then + count_els
                }
                Stmt::While { body, .. } => {
                    let count_self = criteria(stmt) as usize;
                    let count_body = count(&body.node, criteria);
                    count_self + count_body
                }
                _ => criteria(stmt) as usize,
            }
        }

        block
            .items
            .iter()
            .map(|item| match item {
                BlockItem::Decl { .. } => 0,
                BlockItem::Stmt { stmt } => count(&stmt.node, criteria),
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
