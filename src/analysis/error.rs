//! Error type for analysis.

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// Error type for analysis.
#[derive(Error, Debug, Diagnostic)]
#[allow(dead_code)]
pub enum AnalysisError {
    /// Too many local variables.
    #[error("too many local variables")]
    #[diagnostic(code(minic::too_many_locals))]
    TooManyLocals {
        /// Span of the declaration.
        #[label("in function")]
        span: SourceSpan,
    },
}
