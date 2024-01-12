//! Error type for the code generator.

use miette::Diagnostic;
use thiserror::Error;

/// Error type for the code generator.
#[derive(Error, Debug, Diagnostic)]
#[allow(dead_code)]
pub enum CodegenError {
    // /// analysis error
    // #[error(transparent)]
    // AnalysisError(#[from] AnalysisError),
}
