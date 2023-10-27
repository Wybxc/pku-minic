use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// Error type for the IR generator.
#[derive(Error, Debug, Diagnostic)]
#[allow(dead_code)]
pub enum CompileError {
    #[error("non-constant expression in constant context")]
    #[diagnostic(code(minic::non_const_expr))]
    NonConstantExpression {
        #[label("non-constant expression")]
        span: SourceSpan,
    },

    #[error("variable not found")]
    #[diagnostic(code(minic::var_not_found))]
    VariableNotFound {
        ident: String,
        #[label("not found")]
        span: SourceSpan,
    },
}
