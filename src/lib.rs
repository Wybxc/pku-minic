//! A compiler for the SysY language.
#![deny(missing_docs)]

use std::ops::Range;

use lalrpop_util::lalrpop_mod;

pub(crate) mod ast;

mod codegen;
mod irgen;
pub(crate) mod irutils;

lalrpop_mod!(sysy);

type Report<'a> = Box<ariadne::Report<'a, (&'a str, Range<usize>)>>;

/// Compile SysY source code to Koopa IR.
pub fn compile<'a>(input: &'a str, filename: &'a str) -> Result<koopa::ir::Program, Report<'a>> {
    let ast = sysy::CompUnitParser::new().parse(input).map_err(|e| {
        use ariadne::*;
        use lalrpop_util::ParseError;

        let location = match &e {
            ParseError::InvalidToken { location } => *location,
            ParseError::UnrecognizedEof { location, .. } => *location,
            ParseError::UnrecognizedToken { token, .. } => token.0,
            ParseError::ExtraToken { token } => token.0,
            ParseError::User { .. } => 0,
        };
        let message = match &e {
            ParseError::InvalidToken { .. } => "invalid token",
            ParseError::UnrecognizedEof { .. } => "unexpected end of file",
            ParseError::UnrecognizedToken { .. } => "unexpected token",
            ParseError::ExtraToken { .. } => "extra token",
            ParseError::User { error } => error,
        };
        let label_message = match e {
            ParseError::InvalidToken { .. } => "invalid token".to_string(),
            ParseError::UnrecognizedEof { expected, .. } => {
                format!("expected one of {}", expected.join(", "))
            }
            ParseError::UnrecognizedToken { expected, .. } => {
                format!("expected one of {}", expected.join(", "))
            }
            ParseError::ExtraToken { token } => format!("extra token `{}`", token.1),
            ParseError::User { error } => error.to_string(),
        };

        let mut colors = ColorGenerator::new();
        let report = Report::build(ReportKind::Error, filename, location)
            .with_message(message)
            .with_label(
                Label::new((filename, location..location))
                    .with_message(label_message)
                    .with_color(colors.next()),
            )
            .finish();

        Box::new(report)
    })?;
    let ir = ast.build_ir().map_err(|e| Box::new(e.report(filename)))?;
    Ok(ir)
}

/// Generate code from Koopa IR.
pub fn codegen(ir: koopa::ir::Program, mut w: impl std::io::Write) -> std::io::Result<()> {
    use codegen::Codegen;

    Codegen(&ir).generate(&mut w)
}
