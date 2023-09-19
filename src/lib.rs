//! A compiler for the SysY language.
#![deny(missing_docs)]

use lalrpop_util::lalrpop_mod;

pub(crate) mod ast;

mod code_gen;
mod ir_gen;

lalrpop_mod!(sysy);

/// Compile SysY source code to Koopa IR.
pub fn compile(input: &str) -> koopa::ir::Program {
    let ast = sysy::CompUnitParser::new().parse(input).unwrap();
    ast.build_ir()
}

/// Generate code from Koopa IR.
pub fn codegen(ir: koopa::ir::Program, mut w: impl std::io::Write) -> std::io::Result<()> {
    use code_gen::Codegen;

    ir.generate(&mut w)
}
