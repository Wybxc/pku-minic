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

use koopa::ir::{dfg::DataFlowGraph, BinaryOp, ValueKind};
use miette::Result;

mod error;
mod imm;
mod peephole;
mod register;
pub mod riscv;

use crate::{
    irgen::metadata::{FunctionMetadata, ProgramMetadata},
    irutils,
};
use peephole::BlockBuilder;
use register::{RegAlloc, Storage};
use riscv::{Inst, RegId};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, metadata: &ProgramMetadata) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            let func = Codegen(func_data).generate(metadata.functions.get(&func).unwrap())?;
            program.push(func);
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Generate code from Koopa IR.
    pub fn generate(self, metadata: &FunctionMetadata) -> Result<riscv::Function> {
        let name = &self.0.name()[1..];

        let mut func = riscv::Function::new(name.to_string());

        let regs = RegAlloc::new(self.0, metadata)?;
        let dfg = self.0.dfg();

        // Prologue.
        let frame_size = regs.frame_size();
        if frame_size > 0 {
            let mut block = func.new_block();
            block.push(Inst::Addi(RegId::SP, RegId::SP, -frame_size));
            func.push(block);
        }

        // Epilogue.
        let epilogue = if frame_size > 0 {
            Some(Inst::Addi(RegId::SP, RegId::SP, frame_size))
        } else {
            None
        };

        // Generate code for the instruction.
        for (&_bb, node) in self.0.layout().bbs() {
            let mut block = BlockBuilder::new(func.new_block());
            for &inst in node.insts().keys() {
                Codegen(inst).generate(&mut block, dfg, &regs, epilogue)?;
            }
            func.push(block.build());
        }

        Ok(func)
    }
}

/// Simple binary operations that can be directly translated to RISC-V assembly.
fn simple_bop_to_asm(op: BinaryOp, dst: RegId, lhs: RegId, rhs: RegId) -> Option<riscv::Inst> {
    match op {
        BinaryOp::Add => Some(Inst::Add(dst, lhs, rhs)),
        BinaryOp::Sub => Some(Inst::Sub(dst, lhs, rhs)),
        BinaryOp::Mul => Some(Inst::Mul(dst, lhs, rhs)),
        BinaryOp::Div => Some(Inst::Div(dst, lhs, rhs)),
        BinaryOp::Mod => Some(Inst::Rem(dst, lhs, rhs)),
        BinaryOp::Lt => Some(Inst::Slt(dst, lhs, rhs)),
        BinaryOp::Gt => Some(Inst::Slt(dst, rhs, lhs)),
        BinaryOp::And => Some(Inst::And(dst, lhs, rhs)),
        BinaryOp::Or => Some(Inst::Or(dst, lhs, rhs)),
        BinaryOp::Xor => Some(Inst::Xor(dst, lhs, rhs)),
        BinaryOp::Shl => Some(Inst::Sll(dst, lhs, rhs)),
        BinaryOp::Shr => Some(Inst::Srl(dst, lhs, rhs)),
        BinaryOp::Sar => Some(Inst::Sra(dst, lhs, rhs)),
        _ => None,
    }
}

impl Codegen<koopa::ir::entities::Value> {
    /// Generate code from Koopa IR for a single instruction.
    pub fn generate(
        self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        epilogue: Option<riscv::Inst>,
    ) -> Result<()> {
        match dfg.value(self.0).kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    Codegen(val).load_value_to_reg(block, dfg, regs, RegId::A0)?;
                }
                if let Some(epilogue) = epilogue {
                    block.push(epilogue);
                }
                block.push(Inst::Ret);
                Ok(())
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_value(block, dfg, regs, RegId::T0)?;
                let rhs = Codegen(bin.rhs()).load_value(block, dfg, regs, RegId::T1)?;

                // Get the register that stores the result.
                // If the result will be stored in stack, use a0 as the temporary register.
                let storage = regs.get(self.0);
                let reg = match storage {
                    Storage::Reg(reg) => reg,
                    Storage::Slot(_) => RegId::A0,
                };

                if let Some(asm) = simple_bop_to_asm(bin.op(), reg, lhs, rhs) {
                    // Simple binary operations, such as add, sub, and, etc.
                    block.push(asm);
                } else {
                    // Specially deal with eq and ne.
                    match bin.op() {
                        BinaryOp::Eq => {
                            block.push(Inst::Xor(RegId::T0, lhs, rhs));
                            block.push(Inst::Seqz(reg, RegId::T0));
                        }
                        BinaryOp::NotEq => {
                            block.push(Inst::Xor(RegId::T0, lhs, rhs));
                            block.push(Inst::Snez(reg, RegId::T0));
                        }
                        BinaryOp::Le => {
                            block.push(Inst::Slt(reg, rhs, lhs));
                            block.push(Inst::Seqz(reg, reg));
                        }
                        BinaryOp::Ge => {
                            block.push(Inst::Slt(reg, lhs, rhs));
                            block.push(Inst::Seqz(reg, reg));
                        }
                        _ => panic!("unexpected binary op: {:?}", bin.op()),
                    }
                }

                // Write the result to stack if necessary.
                if let Storage::Slot(slot) = storage {
                    // writeln!(w, "    sw {}, {}(sp)", reg, slot)?;
                    block.push(Inst::Sw(reg, slot, RegId::SP));
                }
                Ok(())
            }
            ValueKind::Alloc(_) => {
                // Nothing to do here. Arrays will be considered in the future.
                Ok(())
            }
            ValueKind::Store(store) => {
                let val = Codegen(store.value());
                let dest_storage = regs.get(store.dest());

                // Get the register that stores the value.
                // If the value is stored in stack, use a0 as the temporary register.
                let dest_reg = match dest_storage {
                    Storage::Reg(reg) => reg,
                    Storage::Slot(_) => RegId::A0,
                };

                // Load the value to the register.
                val.load_value_to_reg(block, dfg, regs, dest_reg)?;

                // Write the value to stack if necessary.
                if let Storage::Slot(slot) = dest_storage {
                    block.push(Inst::Sw(dest_reg, slot, RegId::SP));
                }

                Ok(())
            }
            ValueKind::Load(load) => {
                let storage = regs.get(self.0);
                let reg = match storage {
                    Storage::Reg(reg) => reg,
                    Storage::Slot(_) => RegId::A0,
                };

                Codegen(load.src()).load_value_to_reg(block, dfg, regs, reg)?;

                // Write the result to stack if necessary.
                if let Storage::Slot(slot) = storage {
                    block.push(Inst::Sw(reg, slot, RegId::SP));
                }

                Ok(())
            }
            _ => panic!("unexpected value kind: {:?}", dfg.value(self.0).kind()),
        }
    }

    fn data<'a>(&self, dfg: &'a DataFlowGraph) -> Codegen<&'a koopa::ir::entities::ValueData> {
        Codegen(dfg.value(self.0))
    }

    /// Load value.
    /// If the value is already in a register, return the register.
    /// If the value is a constant or stored in stack, load it to a temporary register.
    fn load_value(
        &self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        temp: RegId,
    ) -> Result<RegId> {
        let data = self.data(dfg);
        if data.is_const() {
            data.load_const(block, temp)
        } else {
            match regs.get(self.0) {
                Storage::Reg(reg) => Ok(reg),
                Storage::Slot(slot) => {
                    // writeln!(w, "    lw {}, {}(sp)", temp, slot)?;
                    block.push(Inst::Lw(temp, slot, RegId::SP));
                    Ok(temp)
                }
            }
        }
    }

    /// Load value to a register.
    fn load_value_to_reg(
        &self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        reg: RegId,
    ) -> Result<()> {
        let res = self.load_value(block, dfg, regs, reg)?;
        if res != reg {
            block.push(Inst::Mv(reg, res));
        }
        Ok(())
    }
}

impl Codegen<&koopa::ir::entities::ValueData> {
    /// Load a constant.
    /// Return the register that contains the constant.
    pub fn load_const(self, block: &mut BlockBuilder, temp: RegId) -> Result<RegId> {
        match self.0.kind() {
            ValueKind::Integer(i) => {
                if i.value() == 0 {
                    Ok(RegId::X0)
                } else {
                    block.push(Inst::Li(temp, i.value()));
                    Ok(temp)
                }
            }
            ValueKind::ZeroInit(_) => Ok(RegId::X0),
            _ => panic!("unexpected value kind: {:?}", self.0),
        }
    }

    /// Load a constant to a register.
    #[allow(dead_code)]
    pub fn load_const_to_reg(self, block: &mut BlockBuilder, reg: RegId) -> Result<()> {
        let res = self.load_const(block, reg)?;
        if res != reg {
            block.push(Inst::Mv(reg, res));
        }
        Ok(())
    }

    fn is_const(&self) -> bool {
        irutils::is_const(self.0)
    }
}
