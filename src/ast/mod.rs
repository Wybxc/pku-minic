//! Abstract Syntax Tree.
#![allow(missing_docs)]

use std::ops::Range;

#[cfg(any(test, feature = "proptest"))]
pub mod arbitrary;
pub(crate) mod display;

/// Compilation Unit
///
/// ```text
/// CompUnit ::= FuncDef
/// ```
#[derive(Debug, Clone)]
pub struct CompUnit {
    pub func_def: FuncDef,
}

impl Spanned for CompUnit {
    fn start_pos(&self) -> usize {
        self.func_def.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.func_def.end_pos()
    }
}

/// Function Definition
///
/// ```text
/// FuncDef ::= FuncType IDENT "(" ")" Block
/// ```
#[derive(Debug, Clone)]
pub struct FuncDef {
    pub func_type: Span<FuncType>,
    pub ident: Span<String>,
    pub block: Span<Block>,
}

impl Spanned for FuncDef {
    fn start_pos(&self) -> usize {
        self.func_type.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.block.end_pos()
    }
}

/// Function Type
///
/// ```text
/// FuncType ::= "int"
/// ```
#[derive(Debug, Clone, Copy)]
pub enum FuncType {
    Int,
}

impl NonSpanned for FuncType {}

/// Block
///
/// ```text
/// Block ::= "{" {BlockItem} "}"
/// ```
#[derive(Debug, Clone)]
pub struct Block {
    pub items: imbl::Vector<BlockItem>,
}

impl NonSpanned for Block {}

/// Block Item
///
/// ```text
/// BlockItem ::= Decl | Stmt
/// ```
#[derive(Debug, Clone)]
pub enum BlockItem {
    Decl { decl: Decl },
    Stmt { stmt: Span<Stmt> },
}

impl Spanned for BlockItem {
    fn start_pos(&self) -> usize {
        match self {
            BlockItem::Decl { decl } => decl.start_pos(),
            BlockItem::Stmt { stmt } => stmt.start_pos(),
        }
    }

    fn end_pos(&self) -> usize {
        match self {
            BlockItem::Decl { decl } => decl.end_pos(),
            BlockItem::Stmt { stmt } => stmt.end_pos(),
        }
    }
}

/// Declaration
///
/// ```text
/// Decl ::= ConstDecl | VarDecl
/// ```
#[derive(Debug, Clone)]
pub enum Decl {
    Const(Span<ConstDecl>),
    Var(Span<VarDecl>),
}

impl Spanned for Decl {
    fn start_pos(&self) -> usize {
        match self {
            Decl::Const(decl) => decl.start_pos(),
            Decl::Var(decl) => decl.start_pos(),
        }
    }

    fn end_pos(&self) -> usize {
        match self {
            Decl::Const(decl) => decl.end_pos(),
            Decl::Var(decl) => decl.end_pos(),
        }
    }
}

/// Constant Declaration
///
/// ```text
/// ConstDecl ::= "const" BType ConstDef {"," ConstDef} ";"
/// ```
#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub ty: Span<BType>,
    pub defs: imbl::Vector<ConstDef>,
}

impl NonSpanned for ConstDecl {}

/// Constant Definition
///
/// ```text
/// ConstDef ::= IDENT "=" ConstExpr
/// ```
#[derive(Debug, Clone)]
pub struct ConstDef {
    pub ident: Span<String>,
    pub expr: ConstExpr,
}

impl Spanned for ConstDef {
    fn start_pos(&self) -> usize {
        self.ident.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.expr.end_pos()
    }
}

/// Variable Declaration
///
/// ```text
/// VarDecl ::= BType VarDef {"," VarDef} ";"
/// ```
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub ty: Span<BType>,
    pub defs: imbl::Vector<VarDef>,
}

impl NonSpanned for VarDecl {}

/// Variable Definition
///
/// ```text
/// VarDef ::= IDENT | IDENT "=" InitVal
/// ```
#[derive(Debug, Clone)]
pub struct VarDef {
    pub ident: Span<String>,
    pub init: Option<InitVal>,
}

impl Spanned for VarDef {
    fn start_pos(&self) -> usize {
        self.ident.start_pos()
    }

    fn end_pos(&self) -> usize {
        match &self.init {
            Some(init) => init.end_pos(),
            None => self.ident.end_pos(),
        }
    }
}

/// Initial Value
///
/// ```text
/// InitVal ::= Expr
/// ```
#[derive(Debug, Clone)]
pub struct InitVal {
    pub expr: Expr,
}

impl Spanned for InitVal {
    fn start_pos(&self) -> usize {
        self.expr.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.expr.end_pos()
    }
}

/// Basic Type
///
/// ```text
/// BType ::= "int"
/// ```
#[derive(Debug, Clone)]
pub enum BType {
    Int,
}

impl NonSpanned for BType {}

/// Constant Expression
///
/// ```text
/// ConstExpr ::= Expr
/// ```
#[derive(Debug, Clone)]
pub struct ConstExpr {
    pub expr: Expr,
}

impl Spanned for ConstExpr {
    fn start_pos(&self) -> usize {
        self.expr.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.expr.end_pos()
    }
}

/// Statement
///
/// ```text
///            Stmt ::= MatchedIfStmt
///                   | UnmatchedIfStmt
///   MatchedIfStmt ::= LVal "=" Exp ";"
///                   | [Exp] ";"
///                   | Block
///                   | "if" "(" Exp ")" MatchedIfStmt "else" MatchedIfStmt
///                   | "return" Exp ";"
/// UnmatchedIfStmt ::= "if" "(" Exp ")" Stmt
///                   | "if" "(" Exp ")" MatchedIfStmt "else" UnmatchedIfStmt
/// ```
#[derive(Debug, Clone)]
pub enum Stmt {
    Assign {
        ident: Span<String>,
        expr: Expr,
    },
    Expr {
        expr: Option<Expr>,
    },
    Block {
        block: Span<Block>,
    },
    If {
        cond: Expr,
        then: Box<Span<Stmt>>,
        els: Option<Box<Span<Stmt>>>,
    },
    While {
        cond: Expr,
        body: Box<Span<Stmt>>,
    },
    Return {
        expr: Expr,
    },
}

impl NonSpanned for Stmt {}

/// Expression
///
/// ```text
/// Expr          ::= LOrExp;
/// LVal          ::= IDENT;
/// PrimaryExp    ::= "(" Exp ")" | LVal | Number;
/// Number        ::= INT_CONST;
/// UnaryExp      ::= PrimaryExp | UnaryOp UnaryExp;
/// MulExp        ::= UnaryExp | MulExp ("*" | "/" | "%") UnaryExp;
/// AddExp        ::= MulExp | AddExp ("+" | "-") MulExp;
/// RelExp        ::= AddExp | RelExp ("<" | ">" | "<=" | ">=") AddExp;
/// EqExp         ::= RelExp | EqExp ("==" | "!=") RelExp;
/// LAndExp       ::= EqExp | LAndExp "&&" EqExp;
/// LOrExp        ::= LAndExp | LOrExp "||" LAndExp;
/// ```
#[derive(Debug, Clone)]
pub enum Expr {
    Unary {
        op: Span<UnaryOp>,
        expr: Box<Expr>,
    },
    Binary {
        op: Span<BinaryOp>,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Number(Span<i32>),
    LVar(Span<String>),
}

impl Spanned for Expr {
    fn start_pos(&self) -> usize {
        match self {
            Expr::Unary { op, .. } => op.start_pos(),
            Expr::Binary { lhs, .. } => lhs.start_pos(),
            Expr::Number(n) => n.start_pos(),
            Expr::LVar(ident) => ident.start_pos(),
        }
    }

    fn end_pos(&self) -> usize {
        match self {
            Expr::Unary { expr, .. } => expr.end_pos(),
            Expr::Binary { rhs, .. } => rhs.end_pos(),
            Expr::Number(n) => n.end_pos(),
            Expr::LVar(ident) => ident.end_pos(),
        }
    }
}

/// Unary Operator
///
/// ```text
/// UnaryOp ::= "+" | "-" | "!"
/// ```
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
}

impl NonSpanned for UnaryOp {}

/// Binary Operator
///
/// ```text
/// BinaryOp ::= "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||"
/// ```
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LAnd,
    LOr,
}

impl NonSpanned for BinaryOp {}

impl NonSpanned for i32 {}

impl NonSpanned for String {}

/// Syntax unit with span information.
pub trait Spanned {
    /// Start position of the syntax unit.
    fn start_pos(&self) -> usize;

    /// End position of the syntax unit.
    fn end_pos(&self) -> usize;

    /// Range of the syntax unit.
    fn span(&self) -> Range<usize> {
        self.start_pos()..self.end_pos()
    }
}

/// AST nodes that do not themselves contain span information
/// and thus should be enclosed within the [`Span<T>`] struct.
pub trait NonSpanned {
    /// Convert the syntax unit into a spanned syntax unit.
    fn into_span(self, start: usize, end: usize) -> Span<Self>
    where
        Self: Sized,
    {
        Span {
            start,
            end,
            node: self,
        }
    }
}

/// Attach span information to a syntax unit.
#[derive(Debug, Clone)]
pub struct Span<T> {
    /// Start position of the syntax unit.
    pub start: usize,
    /// End position of the syntax unit.
    pub end: usize,
    /// Inner syntax unit.
    pub node: T,
}

impl<T> Spanned for Span<T> {
    fn start_pos(&self) -> usize {
        self.start
    }

    fn end_pos(&self) -> usize {
        self.end
    }
}

impl<T> std::fmt::Display for Span<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.node)
    }
}
