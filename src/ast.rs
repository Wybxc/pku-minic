//! Abstract Syntax Tree

/// Compilation Unit
#[derive(Debug, Clone)]
pub struct CompUnit {
    pub func_def: FuncDef,
}

/// Function Definition
#[derive(Debug, Clone)]
pub struct FuncDef {
    pub func_type: FuncType,
    pub ident: String,
    pub block: Block,
}

/// Function Type
#[derive(Debug, Clone)]
pub enum FuncType {
    Int,
}

/// Block
#[derive(Debug, Clone)]
pub struct Block {
    pub items: Vec<BlockItem>,
}

/// Block Item
#[derive(Debug, Clone)]
pub enum BlockItem {
    Decl { decl: Decl },
    Stmt { stmt: Stmt },
}

/// Declaration
#[derive(Debug, Clone)]
pub enum Decl {
    Const { ty: BType, defs: Vec<ConstDef> },
}

/// Constant Declaration
#[derive(Debug, Clone)]
pub struct ConstDef {
    pub ident: String,
    pub expr: ConstExpr,
}

/// Basic Type
#[derive(Debug, Clone)]
pub enum BType {
    Int,
}

/// Constant Expression
#[derive(Debug, Clone)]
pub struct ConstExpr {
    pub expr: Expr,
}

/// Statement
#[derive(Debug, Clone)]
pub enum Stmt {
    Return { expr: Expr },
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expr {
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Number(i32),
    LVar(String),
}

/// Unary Operator
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
}

/// Binary Operator
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    LAnd,
    LOr,
}
