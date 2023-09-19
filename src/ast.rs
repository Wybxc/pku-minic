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
    pub stmt: Stmt,
}

/// Statement
#[derive(Debug, Clone)]
pub enum Stmt {
    Return { num: i32 },
}
