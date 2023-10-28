//! Error type for the code generator.

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// Error type for the code generator.
#[derive(Error, Debug, Diagnostic)]
#[allow(dead_code)]
pub enum CodegenError {
    #[error("too many local variables")]
    #[diagnostic(code(minic::too_many_locals))]
    TooManyLocals {
        #[label("in function")]
        span: SourceSpan,
    },
}
