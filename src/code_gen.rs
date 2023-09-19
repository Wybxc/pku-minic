use std::io::Write;

use koopa::ir::{dfg::DataFlowGraph, ValueKind};

/// Generate code from IR.
pub trait Codegen {
    /// Generate code from IR.
    fn generate<W: Write>(&self, w: &mut W) -> std::io::Result<()>;
}

/// Generate code from IR with DFG.
trait CodegenWithDfg {
    fn generate<W: Write>(&self, w: &mut W, dfg: &DataFlowGraph) -> std::io::Result<()>;
}

impl Codegen for koopa::ir::Program {
    fn generate<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        for &func in self.func_layout() {
            let func_data = self.func(func);
            func_data.generate(w)?;
        }
        Ok(())
    }
}

impl Codegen for koopa::ir::FunctionData {
    fn generate<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "    .text")?;
        writeln!(w, "    .globl {}", &self.name()[1..])?;
        writeln!(w, "{}:", &self.name()[1..])?;

        let dfg = self.dfg();
        for (&_bb, node) in self.layout().bbs() {
            for &inst in node.insts().keys() {
                inst.generate(w, dfg)?;
            }
        }

        Ok(())
    }
}

impl CodegenWithDfg for koopa::ir::entities::Value {
    fn generate<W: Write>(&self, w: &mut W, dfg: &DataFlowGraph) -> std::io::Result<()> {
        match dfg.value(*self).kind() {
            ValueKind::Integer(i) => writeln!(w, "    li a0, {}", i.value()),
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    val.generate(w, dfg)?;
                }
                writeln!(w, "    ret")
            }
            _ => unimplemented!(),
        }
    }
}
