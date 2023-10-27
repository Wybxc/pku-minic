//! A compiler for the SysY language.
#![deny(missing_docs)]

use lalrpop_util::lalrpop_mod;

pub(crate) mod ast;

mod codegen;
mod irgen;
pub(crate) mod irutils;

lalrpop_mod!(sysy);

use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

/// Parse error
#[derive(Debug, Error, Diagnostic)]
pub enum MinicParseError {
    /// Invalid token
    #[error("invalid token")]
    #[diagnostic(code(minic::invalid_token))]
    InvalidToken {
        /// Span of the invalid token
        #[label("invalid token")]
        span: SourceSpan,
    },

    /// Unexpected end of file
    #[error("unexpected end of file")]
    #[diagnostic(code(minic::unexpected_eof))]
    UnexpectedEof {
        /// Expected tokens
        #[help]
        expected: String,
        /// Span of the unexpected end of file
        #[label("unexpected end of file")]
        span: SourceSpan,
    },

    /// Unexpected token
    #[error("unexpected token")]
    #[diagnostic(code(minic::unexpected_token))]
    UnexpectedToken {
        /// Expected tokens
        #[help]
        expected: String,
        /// Span of the unexpected token
        #[label("unexpected token")]
        span: SourceSpan,
    },

    /// Extra token
    #[error("extra token")]
    #[diagnostic(code(minic::extra_token))]
    ExtraToken {
        /// Span of the extra token
        #[label("extra token")]
        span: SourceSpan,
    },

    /// Unknown error
    #[error("unknown error")]
    #[diagnostic(code(minic::unknown))]
    Unknown,
}

/// Compile SysY source code to Koopa IR.
pub fn compile(input: &str) -> Result<koopa::ir::Program> {
    let ast = sysy::CompUnitParser::new().parse(input).map_err(|e| {
        use lalrpop_util::ParseError;

        match e {
            ParseError::InvalidToken { location } => MinicParseError::InvalidToken {
                span: SourceSpan::new(location.into(), 1.into()),
            },
            ParseError::UnrecognizedEof { location, expected } => MinicParseError::UnexpectedEof {
                expected: format!("expect one of: {}", expected.join(", ")),
                span: SourceSpan::new(location.into(), 1.into()),
            },
            ParseError::UnrecognizedToken { token, expected } => MinicParseError::UnexpectedToken {
                expected: if !expected.is_empty() {
                    format!("expect one of: {}", expected.join(", "))
                } else {
                    "expect nothing".to_string()
                },
                span: (token.0..token.2).into(),
            },
            ParseError::ExtraToken { token } => MinicParseError::ExtraToken {
                span: (token.0..token.2).into(),
            },
            _ => MinicParseError::Unknown,
        }
    })?;
    let ir = ast.build_ir()?;
    Ok(ir)
}

/// Generate code from Koopa IR.
pub fn codegen(ir: koopa::ir::Program, mut w: impl std::io::Write) -> std::io::Result<()> {
    use codegen::Codegen;

    Codegen(&ir).generate(&mut w)
}
