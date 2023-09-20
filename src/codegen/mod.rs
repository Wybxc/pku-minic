use std::io::{Result, Write};

use koopa::ir::{dfg::DataFlowGraph, BinaryOp, ValueKind};

use crate::codegen::register::{RegAlloc, RegId, Storage};

mod register;

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate<W: Write>(self, w: &mut W) -> Result<()> {
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            Codegen(func_data).generate(w)?;
        }
        Ok(())
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Generate code from Koopa IR.
    pub fn generate<W: Write>(self, w: &mut W) -> Result<()> {
        let name = &self.0.name()[1..];

        writeln!(w, "    .text")?;
        writeln!(w, "    .globl {}", name)?;
        writeln!(w, "{}:", name)?;

        let regs = RegAlloc::new(self.0);
        let dfg = self.0.dfg();
        for (&_bb, node) in self.0.layout().bbs() {
            for &inst in node.insts().keys() {
                let mut frame_size = regs.frame_size();
                if frame_size > 0 {
                    frame_size = (frame_size + 15) & !15; // 16-byte alignment.
                    writeln!(w, "    add $sp, $sp, -{}", frame_size)?;
                }
                Codegen(inst).generate_inst(w, dfg, &regs)?;
                if frame_size > 0 {
                    writeln!(w, "    add $sp, $sp, {}", frame_size)?;
                }
            }
        }

        Ok(())
    }
}

impl Codegen<koopa::ir::entities::Value> {
    /// Generate code from Koopa IR for a single instruction.
    pub fn generate_inst<W: Write>(
        self,
        w: &mut W,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
    ) -> Result<()> {
        match dfg.value(self.0).kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    let val = Codegen(val);
                    let data = val.data(dfg);
                    if data.is_const() {
                        data.generate_const(w, RegId::A0)?;
                    } else {
                        // non-const value has been calculated before
                        match regs.get(val.0) {
                            Storage::Reg(reg) => writeln!(w, "    mv {}, {}", RegId::A0, reg)?,
                            Storage::Slot(slot) => {
                                writeln!(w, "    lw {}, {}($sp)", RegId::A0, slot)?
                            }
                        }
                    }
                }
                writeln!(w, "    ret")
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_to_reg(w, dfg, regs, RegId::T0)?;
                let rhs = Codegen(bin.rhs()).load_to_reg(w, dfg, regs, RegId::T1)?;

                let storage = regs.get(self.0);
                let reg = match storage {
                    Storage::Reg(reg) => reg,
                    Storage::Slot(_) => RegId::A0,
                };

                match bin.op() {
                    BinaryOp::Add => {
                        writeln!(w, "    add {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Sub => {
                        writeln!(w, "    sub {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Lt => {
                        writeln!(w, "    slt {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Gt => {
                        writeln!(w, "    sgt {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Le => {
                        writeln!(w, "    sle {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Ge => {
                        writeln!(w, "    sge {}, {}, {}", reg, lhs, rhs)?;
                    }
                    BinaryOp::Eq => {
                        if lhs != RegId::T0 {
                            writeln!(w, "    mv {}, {}", RegId::T0, lhs)?;
                        }
                        writeln!(w, "    xor {}, {}, {}", RegId::T0, RegId::T0, rhs)?;
                        writeln!(w, "    seqz {}, {}", reg, RegId::T0)?;
                    }
                    BinaryOp::NotEq => {
                        if lhs != RegId::T0 {
                            writeln!(w, "    mv {}, {}", RegId::T0, lhs)?;
                        }
                        writeln!(w, "    xor {}, {}, {}", RegId::T0, RegId::T0, rhs)?;
                        writeln!(w, "    snez {}, {}", reg, RegId::T0)?;
                    }
                    _ => panic!("unexpected binary op: {:?}", bin.op()),
                }

                if let Storage::Slot(slot) = storage {
                    writeln!(w, "    sw {}, {}($sp)", reg, slot)?;
                }
                Ok(())
            }
            _ => panic!("unexpected value kind: {:?}", dfg.value(self.0).kind()),
        }
    }

    fn data<'a>(&'a self, dfg: &'a DataFlowGraph) -> Codegen<&'a koopa::ir::entities::ValueData> {
        Codegen(dfg.value(self.0))
    }

    /// Load value to a register.
    /// If the value is already in a register, return the register.
    /// If the value is a constant or stored in stack, load it to a temporary register.
    fn load_to_reg<W: Write>(
        &self,
        w: &mut W,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        temp: RegId,
    ) -> Result<RegId> {
        let data = self.data(dfg);
        if data.is_const() {
            data.generate_const(w, temp)?;
            Ok(temp)
        } else {
            match regs.get(self.0) {
                Storage::Reg(reg) => Ok(reg),
                Storage::Slot(slot) => {
                    writeln!(w, "    lw {}, {}($sp)", temp, slot)?;
                    Ok(temp)
                }
            }
        }
    }
}

impl Codegen<&koopa::ir::entities::ValueData> {
    /// Generate code from Koopa IR for a constant.
    pub fn generate_const<W: Write>(self, w: &mut W, reg: RegId) -> Result<()> {
        match self.0.kind() {
            ValueKind::Integer(i) => {
                writeln!(w, "    li {}, {}", reg, i.value())
            }
            ValueKind::ZeroInit(_) => {
                writeln!(w, "    mv {}, x0", reg)
            }
            _ => panic!("unexpected value kind: {:?}", self.0),
        }
    }

    fn is_const(&self) -> bool {
        matches!(
            self.0.kind(),
            ValueKind::Integer(_) | ValueKind::ZeroInit(_)
        )
    }
}
