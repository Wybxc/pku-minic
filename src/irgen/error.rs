//! Error type for the IR generator.

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// Error type for the IR generator.
#[derive(Error, Debug, Diagnostic)]
#[allow(dead_code)]
pub enum CompileError {
    /// non-constant expression in constant context
    #[error("non-constant expression in constant context")]
    #[diagnostic(code(minic::non_const_expr))]
    NonConstantExpression {
        #[label("non-constant expression")]
        span: SourceSpan,
    },

    /// variable not found
    #[error("variable not found")]
    #[diagnostic(code(minic::var_not_found))]
    VariableNotFound {
        #[label("not found")]
        span: SourceSpan,
    },

    /// cannot assign to constant
    #[error("cannot assign to constant")]
    #[diagnostic(code(minic::assign_to_const))]
    AssignToConst {
        #[label("cannot assign to constant")]
        span: SourceSpan,
    },

    /// break statement not within a loop
    #[error("break statement not within a loop")]
    #[diagnostic(code(minic::break_not_in_loop))]
    BreakNotInLoop {
        #[label("break statement not within a loop")]
        span: SourceSpan,
    },

    /// continue statement not within a loop
    #[error("continue statement not within a loop")]
    #[diagnostic(code(minic::continue_not_in_loop))]
    ContinueNotInLoop {
        #[label("continue statement not within a loop")]
        span: SourceSpan,
    },

    /// cannot use function as variable
    #[error("cannot use function as variable")]
    #[diagnostic(code(minic::func_as_var))]
    FuncAsVar {
        #[label("expected variable, found function")]
        span: SourceSpan,
    },

    /// undefined reference to function
    #[error("undefined reference to function")]
    #[diagnostic(code(minic::undefined_func))]
    UndefinedFunc {
        #[label("undefined reference to function")]
        span: SourceSpan,
    },

    /// not a function
    #[error("not a function")]
    #[diagnostic(code(minic::not_a_func))]
    ConstNotAFunction {
        #[label("is a constant")]
        span: SourceSpan,
    },

    /// not a function
    #[error("not a function")]
    #[diagnostic(code(minic::not_a_func))]
    VarNotAFunction {
        #[label("is a variable")]
        span: SourceSpan,
    },
}
