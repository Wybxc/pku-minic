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

use koopa::ir::{dfg::DataFlowGraph, BasicBlock, BinaryOp, Function, ValueKind};
use miette::Result;

mod builder;
pub mod error;
pub mod imm;
pub mod riscv;

use builder::BlockBuilder;
use riscv::{Inst, RegId};

use crate::{
    analysis::{
        call::FunctionCalls,
        frame::Frame,
        global::GlobalValues,
        localvar::LocalVars,
        register::{RegAlloc, Storage},
        Analyzer,
    },
    codegen::{
        imm::i12,
        riscv::{Block, BlockId, FrameSlot, FunctionId, Global},
    },
    utils,
};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, analyzer: &mut Analyzer) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();

        // Global variables.
        let global = analyzer.analyze_global_values();
        for data in global.values.values() {
            let id = data.id;
            let size = data.size;
            let init = data.init.clone();
            let global = Global::new(id, size, init);
            program.push_global(global);
        }

        // Generate code for each function.
        let mut func_map = HashMap::new();
        for &func in self.0.func_layout() {
            let func_data = self.0.func(func);
            let id = Codegen(func_data).declare(func, &mut func_map);
            // Skip functions declarations.
            if func_data.layout().entry_bb().is_some() {
                let func = Codegen(func_data).generate(analyzer, func, id, &mut func_map)?;
                program.push(func);
            }
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Declare a function.
    pub fn declare(
        &self,
        func: Function,
        func_map: &mut HashMap<Function, FunctionId>,
    ) -> FunctionId {
        // Add the function to the function map.
        let name = &self.0.name()[1..];
        let func_id = FunctionId::next_id();
        func_id.set_name(name.to_string());
        func_map.insert(func, func_id);
        func_id
    }

    /// Generate code from Koopa IR.
    pub fn generate(
        self,
        analyzer: &mut Analyzer,
        func: Function,
        id: FunctionId,
        func_map: &mut HashMap<Function, FunctionId>,
    ) -> Result<riscv::Function> {
        // Global variables.
        let global = analyzer.analyze_global_values();
        // Local variables.
        let local_vars = analyzer.analyze_local_vars(func);
        // Register allocation.
        let regs = analyzer.analyze_register_alloc(func);
        // Dominators tree.
        let dominators = analyzer.analyze_dominators(func);
        // Function calls.
        let calls = analyzer.analyze_function_calls(func);
        // Frame size.
        let frame = analyzer.analyze_frame(func)?;

        // Generate code for the function.
        let mut func = riscv::Function::new(id);
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

        // Prologue and epilogue.
        let frame_size = frame.total;
        let mut prologue = vec![];
        let mut epilogue = vec![];
        if frame_size > 0 {
            prologue.push(Inst::Addi(RegId::SP, RegId::SP, -frame_size));
            epilogue.push(Inst::Addi(RegId::SP, RegId::SP, frame_size));
        }
        if !calls.is_leaf {
            let slot = FrameSlot::RetAddr;
            let slot = slot.offset(&frame.size);
            let slot = i12::try_from(slot).unwrap();
            prologue.push(Inst::Sw(RegId::RA, slot, RegId::SP));
            epilogue.push(Inst::Lw(RegId::RA, slot, RegId::SP));
        }
        let prologue = prologue;
        epilogue.reverse();
        let epilogue = epilogue;

        // Generate code for the instruction.
        for bb in dominators.iter() {
            let node = bbs.node(&bb).unwrap();
            let id = bb_map[&bb];
            // if let Some(label) = dfg.bb(bb).name() {
            //     id.set_label(label.clone());
            // }

            let mut block = BlockBuilder::new(Block::new());
            if self.0.layout().entry_bb() == Some(bb) {
                for &prologue in prologue.iter() {
                    block.push(prologue);
                }
            }
            for &inst in node.insts().keys() {
                Codegen(inst).generate(
                    &mut block,
                    dfg,
                    &global,
                    &bb_map,
                    func_map,
                    &regs,
                    &local_vars,
                    &calls,
                    &frame,
                    epilogue.iter(),
                )?;
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
    #[allow(clippy::too_many_arguments)]
    pub fn generate<'a>(
        self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        global: &GlobalValues,
        bb_map: &HashMap<BasicBlock, BlockId>,
        func_map: &HashMap<Function, FunctionId>,
        regs: &RegAlloc,
        local_vars: &LocalVars,
        calls: &FunctionCalls,
        frame: &Frame,
        epilogue: impl Iterator<Item = &'a Inst> + 'a,
    ) -> Result<()> {
        let value = dfg.value(self.0);
        match value.kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    Codegen(val).load_value_to_reg(block, dfg, regs, frame, RegId::A0)?;
                }
                for &epilogue in epilogue.into_iter() {
                    block.push(epilogue);
                }
                block.push(Inst::Ret);
                Ok(())
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_value(block, dfg, regs, frame, RegId::T0)?;
                let rhs = Codegen(bin.rhs()).load_value(block, dfg, regs, frame, RegId::T1)?;

                // Get the register that stores the result.
                // If the result will be stored in stack, use a0 as the temporary register.
                let storage = regs.map[&self.0];
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
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
                    block.push(Inst::Sw(reg, slot, RegId::SP));
                }
                Ok(())
            }
            ValueKind::Alloc(_) => {
                // Nothing to do here.
                Ok(())
            }
            ValueKind::Store(store) => {
                let val = Codegen(store.value());
                if global.values.contains_key(&store.dest()) {
                    // global variable
                    block.push(Inst::La(RegId::A0, global.values[&store.dest()].id));
                    val.load_value_to_reg(block, dfg, regs, frame, RegId::T0)?;
                    block.push(Inst::Sw(RegId::T0, 0.try_into().unwrap(), RegId::A0));
                } else {
                    // local variable
                    let slot = local_vars.map[&store.dest()];
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
                    val.load_value_to_reg(block, dfg, regs, frame, RegId::A0)?;
                    block.push(Inst::Sw(RegId::A0, slot, RegId::SP));
                }
                Ok(())
            }
            ValueKind::Load(load) => {
                if global.values.contains_key(&load.src()) {
                    // global variable
                    block.push(Inst::La(RegId::A0, global.values[&load.src()].id));
                    block.push(Inst::Lw(RegId::A0, 0.try_into().unwrap(), RegId::A0));
                } else {
                    // local variable
                    let src = local_vars.map[&load.src()];
                    let src = src.offset(&frame.size);
                    let src = i12::try_from(src).unwrap();
                    block.push(Inst::Lw(RegId::A0, src, RegId::SP));
                }
                match regs.map[&self.0] {
                    Storage::Reg(reg) => {
                        // Load the value to the register.
                        block.push(Inst::Mv(reg, RegId::A0));
                    }
                    Storage::Slot(slot) => {
                        // Write the value to stack.
                        let slot = slot.offset(&frame.size);
                        let slot = i12::try_from(slot).unwrap();
                        block.push(Inst::Sw(RegId::A0, slot, RegId::SP));
                    }
                }
                Ok(())
            }
            ValueKind::Branch(branch) => {
                let cond = Codegen(branch.cond()).load_value(block, dfg, regs, frame, RegId::A0)?;
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
            ValueKind::Call(call) => {
                let func = func_map[&call.callee()];

                // Save registers.
                let save_regs = calls.save_regs[&self.0];
                for (i, reg) in save_regs.iter().enumerate() {
                    let slot = FrameSlot::Saved(i as i32 * 4);
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
                    block.push(Inst::Sw(reg, slot, RegId::SP));
                }

                // Set up arguments.

                // Save 9th and later arguments to stack.
                for (i, &arg) in call.args().iter().skip(8).enumerate() {
                    let slot = FrameSlot::Arg(i as i32 * 4);
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
                    let temp = RegId::A0; // load in reverse order, so a0 is available here
                    Codegen(arg).load_value_to_reg(block, dfg, regs, frame, temp)?;
                    block.push(Inst::Sw(temp, slot, RegId::SP));
                }

                // Save first 8 arguments to registers.
                // Schedule moves to avoid overwriting arguments.
                // `mv ai, aj` is scheduled before `mv aj, ak`.
                let mut moves = HashMap::new();
                let mut after_moves = vec![];
                for (i, &arg) in call.args().iter().enumerate().take(8) {
                    let target = match i {
                        0 => RegId::A0,
                        1 => RegId::A1,
                        2 => RegId::A2,
                        3 => RegId::A3,
                        4 => RegId::A4,
                        5 => RegId::A5,
                        6 => RegId::A6,
                        7 => RegId::A7,
                        _ => unreachable!(),
                    };
                    if let Some(reg) = regs.map.get(&arg).and_then(|storage| match storage {
                        Storage::Reg(reg) => Some(*reg),
                        _ => None,
                    }) {
                        if RegId::A0 <= reg && reg <= RegId::A7 {
                            moves.insert(target, reg);
                            continue;
                        }
                    }
                    after_moves.push((target, arg));
                }

                // Schedule moves.
                let nodes = moves.len();
                {
                    // block.push(Inst::Nop); // for debugging
                    use petgraph::prelude::*;
                    let mut graph = DiGraph::with_capacity(nodes, nodes);
                    for (&target, &arg) in moves.iter() {
                        if target != arg {
                            let target: NodeIndex = graph.add_node(target);
                            let arg: NodeIndex = graph.add_node(arg);
                            graph.add_edge(target, arg, ());
                        }
                    }
                    let mut scc = petgraph::algo::kosaraju_scc(&graph);
                    scc.reverse(); // topological order
                    for scc in scc.into_iter() {
                        if scc.len() == 1 {
                            let target = graph[scc[0]];
                            if let Some(&arg) = moves.get(&target) {
                                block.push(Inst::Mv(target, arg));
                            }
                        } else {
                            let mut target = graph[scc[0]];
                            let mut arg = moves[&target];
                            block.push(Inst::Mv(RegId::T0, target));
                            for _ in 1..scc.len() {
                                block.push(Inst::Mv(target, arg));
                                target = arg;
                                arg = moves[&target];
                            }
                            let after = moves[&arg];
                            block.push(Inst::Mv(after, RegId::T0));
                        }
                    }
                    // block.push(Inst::Nop); // for debugging
                }

                for (target, arg) in after_moves.into_iter() {
                    Codegen(arg).load_value_to_reg(block, dfg, regs, frame, target)?;
                }

                // Call the function.
                block.push(Inst::Call(func));

                // Fetch return value.
                if !value.ty().is_unit() {
                    let storage = regs.map[&self.0];
                    match storage {
                        Storage::Reg(reg) => block.push(Inst::Mv(reg, RegId::A0)),
                        Storage::Slot(slot) => {
                            let slot = slot.offset(&frame.size);
                            let slot = i12::try_from(slot).unwrap();
                            block.push(Inst::Sw(RegId::A0, slot, RegId::SP));
                        }
                    };
                }

                // Restore registers.
                for (i, reg) in save_regs.iter().enumerate() {
                    let slot = FrameSlot::Saved(i as i32 * 4);
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
                    block.push(Inst::Lw(reg, slot, RegId::SP));
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
    /// If the value is a constant or stored in stack, load it to a temporary
    /// register.
    fn load_value(
        &self,
        block: &mut BlockBuilder,
        dfg: &DataFlowGraph,
        regs: &RegAlloc,
        frame: &Frame,
        temp: RegId,
    ) -> Result<RegId> {
        let data = self.data(dfg);
        if data.is_const() {
            data.load_const(block, temp)
        } else {
            match regs.map[&self.0] {
                Storage::Reg(reg) => Ok(reg),
                Storage::Slot(slot) => {
                    let slot = slot.offset(&frame.size);
                    let slot = i12::try_from(slot).unwrap();
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
        frame: &Frame,
        reg: RegId,
    ) -> Result<()> {
        let res = self.load_value(block, dfg, regs, frame, reg)?;
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
