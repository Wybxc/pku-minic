//! Code generator for Koopa IR.
//!
//! This module generates RISC-V assembly from Koopa IR.
//!
//! - `generate` generates code for a function / instruction.
//! - `load` loads a value.
//!   + `load_xxx` loads a value and returns the register that contains the
//!     value. If the value is not in a register, load it to a given temporary
//!     register.
//!   + `load_xxx_to_reg` loads a value to a given register. This ensures that
//!     the value will be stored in the given register.

use std::{cell::RefCell, collections::HashMap};

use koopa::ir::{dfg::DataFlowGraph, BasicBlock, BinaryOp, Function, ValueKind};
use miette::Result;

mod builder;
pub mod error;
pub mod imm;
pub mod riscv;

use builder::BlockBuilder;
use riscv::{Inst, RegId};

use crate::{
    analysis::Analyzer,
    codegen::riscv::{Block, BlockId, PseudoInst, PseudoReg},
    utils,
};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, analyzer: &mut Analyzer) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            let func = Codegen(func_data).generate(analyzer, func)?;
            program.push(func);
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Generate code from Koopa IR.
    pub fn generate(self, analyzer: &mut Analyzer, func: Function) -> Result<riscv::Function> {
        // Register allocation.
        // let regs = analyzer.analyze_register_alloc(func)?;
        // Dominators tree.
        let dominators = analyzer.analyze_dominators(func);

        // Function builder.
        let name = &self.0.name()[1..];
        let mut func = riscv::Function::new(name.to_string());
        let dfg = self.0.dfg();
        let bbs = self.0.layout().bbs();

        // Basic blocks map.
        let bb_map: HashMap<_, _> = bbs
            .keys()
            .map(|&bb| {
                let id = BlockId::next_id();
                (bb, id)
            })
            .collect();

        // Generate code for the instruction.
        for bb in dominators.iter() {
            let node = bbs.node(&bb).unwrap();
            let id = bb_map[&bb];
            // if let Some(label) = dfg.bb(bb).name() {
            //     id.set_label(label.clone());
            // }

            let mut block = BlockBuilder::new(Block::new());
            if self.0.layout().entry_bb() == Some(bb) {
                block.push(Inst::Pseudo(PseudoInst::AllocFrame));
            }
            for &inst in node.insts().keys() {
                Codegen(inst).generate(&mut block, dfg, &bb_map)?;
            }
            let block = block.build();

            // // peephole optimization.
            // peephole::optimize(&mut block, opt_level);

            func.push(id, block);
        }

        Ok(func)
    }
}

/// Simple binary operations that can be directly translated to RISC-V assembly.
fn simple_bop_to_asm(op: BinaryOp, dst: RegId, lhs: RegId, rhs: RegId) -> Option<Inst> {
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
        bb_map: &HashMap<BasicBlock, BlockId>,
    ) -> Result<()> {
        match dfg.value(self.0).kind() {
            ValueKind::Integer(i) => {
                let storage = self.storage();
                if i.value() == 0 {
                    block.push(Inst::Mv(storage, RegId::X0));
                } else {
                    block.push(Inst::Li(storage, i.value()));
                }
                Ok(())
            }
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    let storage = Codegen(val).load(block, dfg);
                    block.push(Inst::Mv(RegId::A0, storage));
                }
                block.push(Inst::Pseudo(PseudoInst::DisposeFrame));
                block.push(Inst::Ret);
                Ok(())
            }
            ValueKind::Binary(bin) => {
                // Get the register that stores the result and the operands.
                let res = self.storage();
                let lhs = Codegen(bin.lhs()).load(block, dfg);
                let rhs = Codegen(bin.rhs()).load(block, dfg);

                if let Some(asm) = simple_bop_to_asm(bin.op(), res, lhs, rhs) {
                    // Simple binary operations, such as add, sub, and, etc.
                    block.push(asm);
                } else {
                    // Specially deal with eq and ne.
                    match bin.op() {
                        BinaryOp::Eq => {
                            block.push(Inst::Xor(res, lhs, rhs));
                            block.push(Inst::Seqz(res, res));
                        }
                        BinaryOp::NotEq => {
                            block.push(Inst::Xor(res, lhs, rhs));
                            block.push(Inst::Snez(res, res));
                        }
                        BinaryOp::Le => {
                            block.push(Inst::Slt(res, rhs, lhs));
                            block.push(Inst::Seqz(res, res));
                        }
                        BinaryOp::Ge => {
                            block.push(Inst::Slt(res, lhs, rhs));
                            block.push(Inst::Seqz(res, res));
                        }
                        _ => panic!("unexpected binary op: {:?}", bin.op()),
                    }
                }
                Ok(())
            }
            ValueKind::Alloc(_) => {
                // Nothing to do here. Arrays will be considered in the future.
                Ok(())
            }
            ValueKind::Store(store) => {
                let src = Codegen(store.value()).load(block, dfg);
                let dest = Codegen(store.dest()).storage();
                block.push(Inst::Mv(dest, src));
                Ok(())
            }
            ValueKind::Load(load) => {
                let src = Codegen(load.src()).load(block, dfg);
                let dest = self.storage();
                block.push(Inst::Mv(dest, src));
                Ok(())
            }
            ValueKind::Branch(branch) => {
                let cond = Codegen(branch.cond()).load(block, dfg);
                let then = bb_map[&branch.true_bb()];
                let els = bb_map[&branch.false_bb()];
                block.push(Inst::Beqz(cond, els));
                block.push(Inst::J(then));
                Ok(())
            }
            ValueKind::Jump(jump) => {
                let target = bb_map[&jump.target()];
                block.push(Inst::J(target));
                Ok(())
            }
            _ => panic!("unexpected value kind: {:?}", dfg.value(self.0).kind()),
        }
    }

    /// Get the storage of a value.
    fn storage(&self) -> RegId {
        thread_local! {
            static VALUE_MAP: RefCell<HashMap<koopa::ir::entities::Value, RegId>> = RefCell::new(HashMap::new());
        }
        VALUE_MAP.with(|map| {
            *map.borrow_mut()
                .entry(self.0)
                .or_insert_with(|| RegId::Pseudo(PseudoReg::next()))
        })
    }

    fn load(&self, block: &mut BlockBuilder, dfg: &DataFlowGraph) -> RegId {
        let value = dfg.value(self.0);
        if utils::is_const(value) {
            match value.kind() {
                ValueKind::Integer(i) => {
                    if i.value() == 0 {
                        RegId::X0
                    } else {
                        let reg = self.storage();
                        block.push(Inst::Li(reg, i.value()));
                        reg
                    }
                }
                _ => panic!("unexpected const value kind: {:?}", value.kind()),
            }
        } else {
            self.storage()
        }
    }
}

// #[cfg(test)]
// mod test {
//     use proptest::prelude::*;

//     use super::*;
//     use crate::ast;

//     proptest! {
//         #[test]
//         fn opt_correct(
//             regs in simulate::Regs::arbitrary(),
//             program in
// ast::arbitrary::arb_comp_unit().prop_map(ast::display::Displayable),
//         ) {
//             let (program, meta) = program.0.build_ir().unwrap();
//             let mut analyzer = Analyzer::new(&program, &meta);
//             let program_o0 = Codegen(&program).generate(&mut analyzer,
// 0).unwrap();             let program_o1 = Codegen(&program).generate(&mut
// analyzer, 1).unwrap();

//             let mut regs1 = regs.clone();
//             let mut mem1 = simulate::Memory::new();
//             simulate::simulate(&program_o0, &mut regs1, &mut mem1);
//             simulate::erase_temp_regs(&mut regs1);

//             let mut regs2 = regs.clone();
//             let mut mem2 = simulate::Memory::new();
//             simulate::simulate(&program_o1, &mut regs2, &mut mem2);
//             simulate::erase_temp_regs(&mut regs2);

//             assert_eq!(regs1, regs2);
//             assert_eq!(mem1, mem2);
//         }
//     }
// }
