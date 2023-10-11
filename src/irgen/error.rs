use thiserror::Error;

/// Error type for the IR generator.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum CompileError {
    #[error("non-constant expression in constant context")]
    NonConstantExpression,
    #[error("variable not found: {0}")]
    VariableNotFound(String),
}

pub type Result<T> = std::result::Result<T, CompileError>;
