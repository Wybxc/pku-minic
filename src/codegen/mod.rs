//! Code generator for Koopa IR.
//!
//! This module generates RISC-V assembly from Koopa IR.
//!
//! - `generate` generates code for a function / instruction.
//! - `load` loads a value.
//!   + `load_xxx` loads a value and returns the register that contains the value.
//!      If the value is not in a register, load it to a given temporary register.
//!   + `load_xxx_to_reg` loads a value to a given register. This ensures that the
//!      value will be stored in the given register.

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
                Codegen(inst).generate(w, dfg, &regs)?;
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
    pub fn generate<W: Write>(self, w: &mut W, dfg: &DataFlowGraph, regs: &RegAlloc) -> Result<()> {
        match dfg.value(self.0).kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    Codegen(val).load_value_to_reg(w, dfg, regs, RegId::A0)?;
                }
                writeln!(w, "    ret")
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_value(w, dfg, regs, RegId::T0)?;
                let rhs = Codegen(bin.rhs()).load_value(w, dfg, regs, RegId::T1)?;

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

    /// Load value.
    /// If the value is already in a register, return the register.
    /// If the value is a constant or stored in stack, load it to a temporary register.
    fn load_value<W: Write>(
        &self,
        w: &mut W,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        temp: RegId,
    ) -> Result<RegId> {
        let data = self.data(dfg);
        if data.is_const() {
            data.load_const(w, temp)
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

    /// Load value to a register.
    fn load_value_to_reg<W: Write>(
        &self,
        w: &mut W,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        reg: RegId,
    ) -> Result<()> {
        let res = self.load_value(w, dfg, regs, reg)?;
        if res != reg {
            writeln!(w, "    mv {}, {}", reg, res)?;
        }
        Ok(())
    }
}

impl Codegen<&koopa::ir::entities::ValueData> {
    /// Load a constant.
    /// Return the register that contains the constant.
    pub fn load_const<W: Write>(self, w: &mut W, temp: RegId) -> Result<RegId> {
        match self.0.kind() {
            ValueKind::Integer(i) => {
                if i.value() == 0 {
                    Ok(RegId::X0)
                } else {
                    writeln!(w, "    li {}, {}", temp, i.value())?;
                    Ok(temp)
                }
            }
            ValueKind::ZeroInit(_) => Ok(RegId::X0),
            _ => panic!("unexpected value kind: {:?}", self.0),
        }
    }

    /// Load a constant to a register.
    #[allow(dead_code)]
    pub fn load_const_to_reg<W: Write>(self, w: &mut W, reg: RegId) -> Result<()> {
        let res = self.load_const(w, reg)?;
        if res != reg {
            writeln!(w, "    mv {}, {}", reg, res)?;
        }
        Ok(())
    }

    fn is_const(&self) -> bool {
        matches!(
            self.0.kind(),
            ValueKind::Integer(_) | ValueKind::ZeroInit(_)
        )
    }
}
