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

mod builder;
mod error;
mod imm;
mod peephole;
mod register;
pub mod riscv;
#[cfg(any(test, feature = "proptest"))]
pub mod simulate;

use crate::{
    codegen::riscv::Block,
    irgen::metadata::{FunctionMetadata, ProgramMetadata},
    utils,
};
use builder::{BlockBuilder, FunctionBuilder};
use register::{RegAlloc, Storage};
use riscv::{Inst, RegId};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, metadata: &ProgramMetadata, opt_level: u8) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            let func =
                Codegen(func_data).generate(metadata.functions.get(&func).unwrap(), opt_level)?;
            program.push(func);
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Generate code from Koopa IR.
    pub fn generate(self, metadata: &FunctionMetadata, opt_level: u8) -> Result<riscv::Function> {
        let name = &self.0.name()[1..];

        let func = riscv::Function::new(name.to_string());
        let mut func = FunctionBuilder::new(func);

        let regs = RegAlloc::new(self.0, metadata)?;
        let dfg = self.0.dfg();

        // Prologue.
        let frame_size = regs.frame_size();
        let prologue = if frame_size > 0 {
            Some(Inst::Addi(RegId::SP, RegId::SP, -frame_size))
        } else {
            None
        };

        // Epilogue.
        let epilogue = if frame_size > 0 {
            Some(Inst::Addi(RegId::SP, RegId::SP, frame_size))
        } else {
            None
        };

        // Generate code for the instruction.
        for (&bb, node) in self.0.layout().bbs() {
            let label = self.0.dfg().bb(bb).name().clone();
            let mut block = BlockBuilder::new(Block::new(), opt_level);
            if self.0.layout().entry_bb() == Some(bb) {
                if let Some(prologue) = prologue {
                    block.push(prologue);
                }
            }
            for &inst in node.insts().keys() {
                Codegen(inst).generate(&mut block, dfg, &regs, epilogue)?;
            }
            let mut block = block.build();

            // peephole optimization.
            peephole::optimize(&mut block, opt_level);

            func.push(label, bb, block);
        }

        Ok(func.build())
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
                            block.push(Inst::Xor(reg, lhs, rhs));
                            block.push(Inst::Seqz(reg, reg));
                        }
                        BinaryOp::NotEq => {
                            block.push(Inst::Xor(reg, lhs, rhs));
                            block.push(Inst::Snez(reg, reg));
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
                match regs.get(store.dest()) {
                    Storage::Reg(reg) => {
                        // Load the value to the register.
                        val.load_value_to_reg(block, dfg, regs, reg)?
                    }
                    Storage::Slot(slot) => {
                        // Load the value to a0.
                        val.load_value_to_reg(block, dfg, regs, RegId::A0)?;

                        // Write the value to stack.
                        block.push(Inst::Sw(RegId::A0, slot, RegId::SP));
                    }
                }
                Ok(())
            }
            ValueKind::Load(load) => {
                let src = Codegen(load.src());
                match regs.get(self.0) {
                    Storage::Reg(reg) => {
                        // Load the value to the register.
                        src.load_value_to_reg(block, dfg, regs, reg)?;
                    }
                    Storage::Slot(slot) => {
                        // Load the value to a0.
                        src.load_value_to_reg(block, dfg, regs, RegId::A0)?;

                        // Write the value to stack.
                        block.push(Inst::Sw(RegId::A0, slot, RegId::SP));
                    }
                }
                Ok(())
            }
            ValueKind::Branch(branch) => {
                let _cond = Codegen(branch.cond()).load_value(block, dfg, regs, RegId::A0)?;
                let _then = branch.true_bb();
                let _els = branch.false_bb();
                todo!()
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
        utils::is_const(self.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ast;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn opt_correct(
            regs in simulate::Regs::arbitrary(),
            program in ast::arbitrary::arb_comp_unit().prop_map(ast::display::Displayable),
        ) {
            let (program, meta) = program.0.build_ir().unwrap();
            let program_o0 = Codegen(&program).generate(&meta, 0).unwrap();
            let program_o1 = Codegen(&program).generate(&meta, 1).unwrap();

            let mut regs1 = regs.clone();
            let mut mem1 = simulate::Memory::new();
            simulate::simulate(&program_o0, &mut regs1, &mut mem1);
            simulate::erase_temp_regs(&mut regs1);

            let mut regs2 = regs.clone();
            let mut mem2 = simulate::Memory::new();
            simulate::simulate(&program_o1, &mut regs2, &mut mem2);
            simulate::erase_temp_regs(&mut regs2);

            assert_eq!(regs1, regs2);
            assert_eq!(mem1, mem2);
        }
    }
}
