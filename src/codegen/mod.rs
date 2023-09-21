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

use std::{
    collections::HashMap,
    io::{Result, Write},
};

use koopa::ir::{dfg::DataFlowGraph, BinaryOp, ValueKind};

use crate::{
    codegen::register::{RegAlloc, RegId, Storage},
    irutils,
};

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

        // Prologue.
        let mut frame_size = regs.frame_size();
        if frame_size > 0 {
            frame_size = (frame_size + 15) & !15; // 16-byte alignment.
            writeln!(w, "    add sp, sp, -{}", frame_size)?;
        }

        // Epilogue.
        let epologue = if frame_size > 0 {
            Some(format!("add sp, sp, {}", frame_size))
        } else {
            None
        };

        // Generate code for the instruction.
        for (&_bb, node) in self.0.layout().bbs() {
            // Remember the value of each register, so that we will not load the same value
            // multiple times. Currently we adopt per-bb register cache. Future work may
            // consider a global register cache by analyzing the data flow.
            let mut reg_cache = HashMap::new();
            for &inst in node.insts().keys() {
                Codegen(inst).generate(w, dfg, &regs, &mut reg_cache, epologue.as_deref())?;
            }
        }

        Ok(())
    }
}

/// Simple binary operations that can be directly translated to RISC-V assembly.
fn simple_bop_to_asm(op: BinaryOp) -> Option<&'static str> {
    match op {
        BinaryOp::Add => Some("add"),
        BinaryOp::Sub => Some("sub"),
        BinaryOp::Mul => Some("mul"),
        BinaryOp::Div => Some("div"),
        BinaryOp::Mod => Some("rem"),
        BinaryOp::Lt => Some("slt"),
        BinaryOp::Gt => Some("sgt"),
        BinaryOp::And => Some("and"),
        BinaryOp::Or => Some("or"),
        BinaryOp::Xor => Some("xor"),
        BinaryOp::Shl => Some("sll"),
        BinaryOp::Shr => Some("srl"),
        BinaryOp::Sar => Some("sra"),
        _ => None,
    }
}

impl Codegen<koopa::ir::entities::Value> {
    /// Generate code from Koopa IR for a single instruction.
    pub fn generate<W: Write>(
        self,
        w: &mut W,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        reg_cache: &mut HashMap<RegId, i32>,
        epilogue: Option<&str>,
    ) -> Result<()> {
        match dfg.value(self.0).kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    Codegen(val).load_value_to_reg(w, dfg, regs, RegId::A0, reg_cache)?;
                }
                if let Some(epilogue) = epilogue {
                    writeln!(w, "    {}", epilogue)?;
                }
                writeln!(w, "    ret")
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_value(w, dfg, regs, RegId::T0, reg_cache)?;
                let rhs = Codegen(bin.rhs()).load_value(w, dfg, regs, RegId::T1, reg_cache)?;

                // Get the register that stores the result.
                // If the result will be stored in stack, use a0 as the temporary register.
                let storage = regs.get(self.0);
                let reg = match storage {
                    Storage::Reg(reg) => reg,
                    Storage::Slot(_) => RegId::A0,
                };

                if let Some(asm) = simple_bop_to_asm(bin.op()) {
                    // Simple binary operations, such as add, sub, and, etc.
                    writeln!(w, "    {} {}, {}, {}", asm, reg, lhs, rhs)?;
                } else {
                    // Specially deal with eq and ne.
                    match bin.op() {
                        BinaryOp::Eq => {
                            writeln!(w, "    xor {}, {}, {}", RegId::T0, lhs, rhs)?;
                            writeln!(w, "    seqz {}, {}", reg, RegId::T0)?;
                        }
                        BinaryOp::NotEq => {
                            writeln!(w, "    xor {}, {}, {}", RegId::T0, lhs, rhs)?;
                            writeln!(w, "    snez {}, {}", reg, RegId::T0)?;
                        }
                        _ => panic!("unexpected binary op: {:?}", bin.op()),
                    }
                }

                // Write the result to stack if necessary.
                if let Storage::Slot(slot) = storage {
                    writeln!(w, "    sw {}, {}(sp)", reg, slot)?;
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
        reg_cache: &mut HashMap<RegId, i32>,
    ) -> Result<RegId> {
        let data = self.data(dfg);
        if data.is_const() {
            data.load_const(w, temp, reg_cache)
        } else {
            match regs.get(self.0) {
                Storage::Reg(reg) => Ok(reg),
                Storage::Slot(slot) => {
                    writeln!(w, "    lw {}, {}(sp)", temp, slot)?;
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
        reg_cache: &mut HashMap<RegId, i32>,
    ) -> Result<()> {
        let res = self.load_value(w, dfg, regs, reg, reg_cache)?;
        if res != reg {
            writeln!(w, "    mv {}, {}", reg, res)?;
        }
        Ok(())
    }
}

impl Codegen<&koopa::ir::entities::ValueData> {
    /// Load a constant.
    /// Return the register that contains the constant.
    pub fn load_const<W: Write>(
        self,
        w: &mut W,
        temp: RegId,
        reg_cache: &mut HashMap<RegId, i32>,
    ) -> Result<RegId> {
        match self.0.kind() {
            ValueKind::Integer(i) => {
                if i.value() == 0 {
                    Ok(RegId::X0)
                } else {
                    if reg_cache.get(&temp) != Some(&i.value()) {
                        writeln!(w, "    li {}, {}", temp, i.value())?;
                        reg_cache.insert(temp, i.value());
                    }
                    Ok(temp)
                }
            }
            ValueKind::ZeroInit(_) => Ok(RegId::X0),
            _ => panic!("unexpected value kind: {:?}", self.0),
        }
    }

    /// Load a constant to a register.
    #[allow(dead_code)]
    pub fn load_const_to_reg<W: Write>(
        self,
        w: &mut W,
        reg: RegId,
        reg_cache: &mut HashMap<RegId, i32>,
    ) -> Result<()> {
        let res = self.load_const(w, reg, reg_cache)?;
        if res != reg {
            writeln!(w, "    mv {}, {}", reg, res)?;
        }
        Ok(())
    }

    fn is_const(&self) -> bool {
        irutils::is_const(self.0)
    }
}
