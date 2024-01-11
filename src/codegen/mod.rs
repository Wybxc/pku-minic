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

use std::collections::HashMap;

use koopa::ir::{dfg::DataFlowGraph, BasicBlock, BinaryOp, Function, Value, ValueKind};
use miette::Result;

mod builder;
pub mod error;
pub mod imm;
pub mod riscv;

use builder::BlockBuilder;
use riscv::{Inst, RegId};

use crate::{
    analysis::Analyzer,
    codegen::riscv::{Block, BlockId, FunctionId, PseudoInst, PseudoReg, Storage},
    utils,
};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, analyzer: &mut Analyzer) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();
        let mut func_map = HashMap::new();
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            let func = Codegen(func_data).generate(analyzer, func, &mut func_map)?;
            program.push(func);
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Generate code from Koopa IR.
    pub fn generate(
        self,
        analyzer: &mut Analyzer,
        func: Function,
        func_map: &mut HashMap<Function, FunctionId>,
    ) -> Result<riscv::Function> {
        // Register allocation.
        // let regs = analyzer.analyze_register_alloc(func)?;
        // Dominators tree.
        let dominators = analyzer.analyze_dominators(func);

        // Function name.
        let name = self.0.name()[1..].to_string();
        let id = FunctionId::next_id();
        id.set_name(name.clone());
        func_map.insert(func, id);

        // Function builder.
        let mut func = riscv::Function::new(name);
        let dfg = self.0.dfg();
        let bbs = self.0.layout().bbs();

        // Value map.
        let mut value_map: HashMap<_, _> = dfg
            .values()
            .keys()
            .map(|&value| {
                let storage = Storage::next_reg();
                (value, storage)
            })
            .collect();
        for (i, &value) in self.0.params().iter().enumerate() {
            let storage = Storage::param(i);
            value_map.insert(value, storage);
        }

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
                Codegen(inst).generate(&mut block, dfg, &value_map, &bb_map, func_map);
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
        value_map: &HashMap<Value, Storage>,
        bb_map: &HashMap<BasicBlock, BlockId>,
        func_map: &HashMap<Function, FunctionId>,
    ) {
        let value = dfg.value(self.0);
        match value.kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    let storage = Codegen(val).load(block, dfg, value_map);
                    block.push(Inst::Mv(RegId::A0, storage));
                }
                block.push(Inst::Pseudo(PseudoInst::DisposeFrame));
                block.push(Inst::Ret);
            }
            ValueKind::Binary(bin) => {
                // Get the register that stores the result and the operands.
                let res = &value_map[&self.0];
                let lhs = Codegen(bin.lhs()).load(block, dfg, value_map);
                let rhs = Codegen(bin.rhs()).load(block, dfg, value_map);

                res.store_with(block, |block, res| {
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
                });
            }
            ValueKind::Alloc(_) => {
                // Nothing to do here. Arrays will be considered in the future.
            }
            ValueKind::Store(store) => {
                let src = Codegen(store.value()).load(block, dfg, value_map);
                let dest = &value_map[&store.dest()];
                dest.store(block, src);
            }
            ValueKind::Load(load) => {
                let src = Codegen(load.src()).load(block, dfg, value_map);
                let dest = &value_map[&self.0];
                dest.store(block, src);
            }
            ValueKind::Branch(branch) => {
                let cond = Codegen(branch.cond()).load(block, dfg, value_map);
                let then = bb_map[&branch.true_bb()];
                let els = bb_map[&branch.false_bb()];
                block.push(Inst::Beqz(cond, els));
                block.push(Inst::J(then));
            }
            ValueKind::Jump(jump) => {
                let target = bb_map[&jump.target()];
                block.push(Inst::J(target));
            }
            ValueKind::Call(call) => {
                let func = func_map[&call.callee()];
                block.push(Inst::Pseudo(PseudoInst::SaveRegs));

                let args = call.args();
                for (i, &arg) in args.iter().enumerate() {
                    let arg = Codegen(arg).load(block, dfg, value_map);
                    let storage = Storage::arg(i);
                    storage.store(block, arg);
                }

                block.push(Inst::Call(func));

                if !value.ty().is_unit() {
                    let res = &value_map[&self.0];
                    res.store(block, RegId::A0);
                }
                block.push(Inst::Pseudo(PseudoInst::RestoreRegs));
            }
            _ => panic!("unexpected value kind: {:?}", dfg.value(self.0).kind()),
        }
    }

    /// Load a value.
    fn load(
        &self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        value_map: &HashMap<Value, Storage>,
    ) -> RegId {
        let value = dfg.value(self.0);
        if utils::is_const(value) {
            match value.kind() {
                ValueKind::Integer(i) => {
                    if i.value() == 0 {
                        RegId::X0
                    } else {
                        let reg = RegId::Pseudo(PseudoReg::next());
                        block.push(Inst::Li(reg, i.value()));
                        reg
                    }
                }
                _ => panic!("unexpected const value kind: {:?}", value.kind()),
            }
        } else {
            let storage = &value_map[&self.0];
            storage.load(block)
        }
    }
}

impl Storage {
    /// Load a value from the storage.
    fn load(&self, block: &mut BlockBuilder) -> RegId {
        match self {
            Storage::Reg(reg) => *reg,
            Storage::Local(offset) => {
                let reg = RegId::Pseudo(PseudoReg::next());
                block.push(Inst::Pseudo(PseudoInst::LoadFrameBottom(reg, *offset)));
                reg
            }
            Storage::Arg(_) => panic!("cannot load from arg"),
            Storage::Param(offset) => {
                let reg = RegId::Pseudo(PseudoReg::next());
                block.push(Inst::Pseudo(PseudoInst::LoadFrameBelow(reg, *offset)));
                reg
            }
        }
    }

    /// Store a value to the storage.
    fn store(&self, block: &mut BlockBuilder, value: RegId) {
        match self {
            Storage::Reg(reg) => {
                block.push(Inst::Mv(*reg, value));
            }
            Storage::Local(offset) => {
                block.push(Inst::Pseudo(PseudoInst::StoreFrameBottom(value, *offset)));
            }
            Storage::Arg(offset) => {
                block.push(Inst::Pseudo(PseudoInst::StoreFrameTop(value, *offset)));
            }
            Storage::Param(_) => panic!("cannot store to param"),
        }
    }

    /// Store a value to the storage.
    fn store_with(
        &self,
        block: &mut BlockBuilder,
        store_builder: impl FnOnce(&mut BlockBuilder, RegId),
    ) {
        match self {
            Storage::Reg(reg) => {
                store_builder(block, *reg);
            }
            _ => {
                let reg = RegId::Pseudo(PseudoReg::next());
                store_builder(block, reg);
                self.store(block, reg);
            }
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
