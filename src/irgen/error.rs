use std::ops::Range;

use thiserror::Error;

/// Error type for the IR generator.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum CompileError {
    #[error("non-constant expression in constant context")]
    NonConstantExpression { span: (usize, usize) },
    #[error("variable not found")]
    VariableNotFound { ident: String, span: (usize, usize) },
}

impl CompileError {
    pub fn span(&self) -> (usize, usize) {
        match self {
            CompileError::NonConstantExpression { span } => *span,
            CompileError::VariableNotFound { span, .. } => *span,
        }
    }

    pub fn report<'a>(&self, filename: &'a str) -> ariadne::Report<'a, (&'a str, Range<usize>)> {
        use ariadne::*;
        let mut colors = ColorGenerator::new();
        let span = self.span();

        let report_builder =
            Report::build(ReportKind::Error, filename, span.0).with_message(self.to_string());

        let report_builder = match self {
            CompileError::NonConstantExpression { .. } => report_builder.with_label(
                Label::new((filename, span.0..span.1))
                    .with_message("non-constant expression")
                    .with_color(colors.next()),
            ),
            CompileError::VariableNotFound { ident, .. } => report_builder.with_label(
                Label::new((filename, span.0..span.1))
                    .with_message(format!("variable `{}` not found", ident))
                    .with_color(colors.next()),
            ),
        };

        report_builder.finish()
    }
}

pub type Result<T> = std::result::Result<T, CompileError>;
