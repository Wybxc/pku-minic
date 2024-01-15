//! Abstract Syntax Tree.
#![allow(missing_docs)]

use std::ops::Range;

pub(crate) mod display;

/// Compilation Unit
///
/// ```text
/// CompUnit ::= {TopLevelItem}
/// ```
#[derive(Debug, Clone)]
pub struct CompUnit {
    pub top_levels: Vec<TopLevelItem>,
}

impl Spanned for CompUnit {
    fn start_pos(&self) -> usize {
        self.top_levels[0].start_pos()
    }

    fn end_pos(&self) -> usize {
        self.top_levels[self.top_levels.len() - 1].end_pos()
    }
}

/// Top Level Item
///
/// ```text
/// TopLevelItem ::= FuncDef | Decl
/// ```
#[derive(Debug, Clone)]
pub enum TopLevelItem {
    FuncDef(FuncDef),
    Decl(Decl),
}

impl Spanned for TopLevelItem {
    fn start_pos(&self) -> usize {
        match self {
            TopLevelItem::FuncDef(func_def) => func_def.start_pos(),
            TopLevelItem::Decl(decl) => decl.start_pos(),
        }
    }

    fn end_pos(&self) -> usize {
        match self {
            TopLevelItem::FuncDef(func_def) => func_def.end_pos(),
            TopLevelItem::Decl(decl) => decl.end_pos(),
        }
    }
}

/// Function Definition
///
/// ```text
/// FuncDef ::= FuncType IDENT "(" [FuncParams] ")" Block
/// FuncParams ::= FuncParam {"," FuncParam}
/// ```
#[derive(Debug, Clone)]
pub struct FuncDef {
    pub func_type: Span<BType>,
    pub ident: Span<String>,
    pub params: Vec<FuncParam>,
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

/// Function Parameter
///
/// ```text
/// FuncParam ::= BType IDENT
/// ```
#[derive(Debug, Clone)]
pub struct FuncParam {
    pub ty: Span<BType>,
    pub ident: Span<String>,
}

impl Spanned for FuncParam {
    fn start_pos(&self) -> usize {
        self.ty.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.ident.end_pos()
    }
}

/// Block
///
/// ```text
/// Block ::= "{" {BlockItem} "}"
/// ```
#[derive(Debug, Clone)]
pub struct Block {
    pub items: Vec<BlockItem>,
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
    pub defs: Vec<ConstDef>,
}

impl NonSpanned for ConstDecl {}

/// Constant Definition
///
/// ```text
/// ConstDef ::= IDENT ["[" ConstExp "]"] "=" InitVal
/// ```
#[derive(Debug, Clone)]
pub struct ConstDef {
    pub ident: Span<String>,
    pub indices: Vec<ConstExpr>,
    pub init: Span<InitVal>,
}

impl Spanned for ConstDef {
    fn start_pos(&self) -> usize {
        self.ident.start_pos()
    }

    fn end_pos(&self) -> usize {
        self.init.end_pos()
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
    pub defs: Vec<Span<VarDef>>,
}

impl NonSpanned for VarDecl {}

/// Variable Definition
///
/// ```text
/// VarDef ::= IDENT ["[" ConstExp "]"]
///          | IDENT ["[" ConstExp "]"] "=" InitVal
/// ```
#[derive(Debug, Clone)]
pub struct VarDef {
    pub ident: Span<String>,
    pub indices: Vec<ConstExpr>,
    pub init: Option<Span<InitVal>>,
}

impl NonSpanned for VarDef {}

/// Initial Value
///
/// ```text
/// InitVal ::= Expr
///           | "{" [Expr {"," Expr}] "}"
/// ```
#[derive(Debug, Clone)]
pub enum InitVal {
    Expr(Expr),
    InitList(Vec<Span<InitVal>>),
}

impl NonSpanned for InitVal {}

/// Basic Type
///
/// ```text
/// BType ::= "int"
/// ```
#[derive(Debug, Clone)]
pub enum BType {
    Int,
    Void,
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
///                   | "while" "(" Exp ")" MatchedIfStmt
///                   | "break" ";"
///                   | "continue" ";"
///                   | "return" Exp ";"
/// UnmatchedIfStmt ::= "if" "(" Exp ")" Stmt
///                   | "if" "(" Exp ")" MatchedIfStmt "else" UnmatchedIfStmt
/// ```
#[derive(Debug, Clone)]
pub enum Stmt {
    Assign {
        lval: Span<LVal>,
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
    Break(Span<()>),
    Continue(Span<()>),
    Return {
        expr: Expr,
    },
}

impl NonSpanned for Stmt {}

/// Expression
///
/// ```text
/// Expr          ::= LOrExp;
/// LVal          ::= IDENT ["[" Expr "]"];
/// PrimaryExp    ::= "(" Expr ")" | LVal | Number;
/// Number        ::= INT_CONST;
/// UnaryExp      ::= PrimaryExp | UnaryOp UnaryExp | CallExp;
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
    LVal(Span<LVal>),
    Call(Span<CallExpr>),
}

impl Spanned for Expr {
    fn start_pos(&self) -> usize {
        match self {
            Expr::Unary { op, .. } => op.start_pos(),
            Expr::Binary { lhs, .. } => lhs.start_pos(),
            Expr::Number(n) => n.start_pos(),
            Expr::LVal(ident) => ident.start_pos(),
            Expr::Call(call) => call.start_pos(),
        }
    }

    fn end_pos(&self) -> usize {
        match self {
            Expr::Unary { expr, .. } => expr.end_pos(),
            Expr::Binary { rhs, .. } => rhs.end_pos(),
            Expr::Number(n) => n.end_pos(),
            Expr::LVal(ident) => ident.end_pos(),
            Expr::Call(call) => call.end_pos(),
        }
    }
}

/// LValue
///
/// ```text
/// LVal ::= IDENT ["[" Expr "]"]
/// ```
#[derive(Debug, Clone)]
pub struct LVal {
    pub ident: Span<String>,
    pub indices: Vec<Expr>,
}

impl NonSpanned for LVal {}

/// Function Call Expression
///
/// ```text
/// CallExp ::= IDENT "(" [FuncArgs] ")";
/// FuncArgs ::= Expr {"," Expr};
/// ```
#[derive(Debug, Clone)]
pub struct CallExpr {
    pub ident: Span<String>,
    pub args: Vec<Expr>,
}

impl NonSpanned for CallExpr {}

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

impl NonSpanned for () {}

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
