//! A compiler for the SysY language.
#![deny(missing_docs)]

use lalrpop_util::lalrpop_mod;

#[macro_use]
#[doc(hidden)]
pub mod logger_setup {
    #[macro_export]
    #[cfg(feature = "trace")]
    macro_rules! color {
        ( [trace] ) => {
            "\x1B[34m"
        };
        ( [debug] ) => {
            "\x1B[36m"
        };
        ( [info]  ) => {
            "\x1B[32m"
        };
        ( [warn]  ) => {
            "\x1B[33m"
        };
        ( [error] ) => {
            "\x1B[31m"
        };
        ( [crit]  ) => {
            "\x1B[35m"
        };
        ( [sep]   ) => {
            "\x1B[1m\x1B[2m"
        }; // +bold +dim
        ( [msg]   ) => {
            ""
        }; // default term font color
        ( [from]  ) => {
            "\x1B[90m\x1B[3m"
        }; // `[src/main.rs 101:5]` in `location_style_classic`
        ( [sep2]  ) => {
            "\x1B[90m\x1B[2m"
        }; // sep2 in default style
        ( [sep3]  ) => {
            "\x1B[90m\x1B[2m"
        }; // sep3 in default style
        ( [line]  ) => {
            "\x1B[38;5;67m\x1B[1m\x1B[2m"
        }; // line number in default style
        ( [key]   ) => {
            "\x1B[3m\x1B[1m"
        }; // +italic +bold
        ( [value] ) => {
            ""
        }; // default term font color
        ( [rm]    ) => {
            "\x1B[0m"
        }; // remove previous colors
    }
}

pub mod ast;
pub mod codegen;
pub mod irgen;
pub mod irutils;

lalrpop_mod!(sysy);

use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

use crate::irgen::metadata::ProgramMetadata;

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
pub fn compile(input: &str, _opt_level: u8) -> Result<(koopa::ir::Program, ProgramMetadata)> {
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
pub fn codegen(
    ir: koopa::ir::Program,
    metadata: &ProgramMetadata,
    opt_level: u8,
) -> Result<codegen::riscv::Program> {
    use codegen::Codegen;

    let program = Codegen(&ir).generate(metadata, opt_level)?;
    Ok(program)
}
