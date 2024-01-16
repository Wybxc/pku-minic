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

use std::{collections::HashMap, rc::Rc};

use koopa::ir::{dfg::DataFlowGraph, BasicBlock, BinaryOp, Function, TypeKind, ValueKind};
use miette::Result;
#[allow(unused_imports)]
use nolog::*;

mod builder;
pub mod error;
pub mod imm;
mod peephole;
pub mod riscv;

use builder::BlockBuilder;
use riscv::{Inst, RegId};

use crate::{
    analysis::{
        call::FunctionCalls,
        dominators::Dominators,
        frame::Frame,
        global::GlobalValues,
        localvar::LocalVars,
        register::{RegAlloc, Storage},
        Analyzer,
    },
    codegen::{
        imm::i12,
        riscv::{Block, BlockId, FrameSize, FrameSlot, FunctionId, Global},
    },
    utils,
};

/// Code generator for Koopa IR.
#[repr(transparent)]
pub struct Codegen<T>(pub T);

/// Program context in code generation.
pub struct ProgramContext {
    /// Global variables.
    pub global: Rc<GlobalValues>,
    /// Function map.
    func_map: HashMap<Function, FunctionId>,
}

/// Function context in code generation.
pub struct FunctionContext<'a> {
    /// Data flow graph.
    pub dfg: &'a DataFlowGraph,
    /// Dominators tree.
    pub dominators: Rc<Dominators>,
    /// Local variables.
    pub local_vars: Rc<LocalVars>,
    /// Register allocation.
    pub regs: Rc<RegAlloc>,
    /// Function calls.
    pub calls: Rc<FunctionCalls>,
    /// Frame size.
    pub frame: Rc<Frame>,
    /// Basic blocks map.
    pub bb_map: HashMap<BasicBlock, BlockId>,
}

impl Codegen<&koopa::ir::Program> {
    /// Generate code from Koopa IR.
    pub fn generate(self, analyzer: &mut Analyzer, opt_level: u8) -> Result<riscv::Program> {
        let mut program = riscv::Program::new();
        let mut program_ctx = ProgramContext {
            global: analyzer.analyze_global_values(),
            func_map: HashMap::new(),
        };

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
        for &func in self.0.func_layout() {
            trace!(->[0] "GEN " => "Function {:?}", func);
            let func_data = self.0.func(func);
            let id = Codegen(func_data).declare(func, &mut program_ctx);
            // Skip functions declarations.
            if func_data.layout().entry_bb().is_some() {
                // Function context.
                let bbs = func_data.layout().bbs();
                let function_ctx = FunctionContext {
                    dfg: func_data.dfg(),
                    dominators: analyzer.analyze_dominators(func),
                    local_vars: analyzer.analyze_local_vars(func),
                    regs: analyzer.analyze_register_alloc(func),
                    calls: analyzer.analyze_function_calls(func),
                    frame: analyzer.analyze_frame(func),
                    bb_map: bbs.keys().map(|&bb| (bb, BlockId::next_id())).collect(),
                };
                let func =
                    Codegen(func_data).generate(id, &mut program_ctx, &function_ctx, opt_level);
                program.push(func);
            }
        }
        Ok(program)
    }
}

impl Codegen<&koopa::ir::FunctionData> {
    /// Declare a function.
    pub fn declare(&self, func: Function, program_ctx: &mut ProgramContext) -> FunctionId {
        // Add the function to the function map.
        let name = &self.0.name()[1..];
        let func_id = FunctionId::next_id();
        func_id.set_name(name.to_string());
        program_ctx.func_map.insert(func, func_id);
        func_id
    }

    /// Generate code from Koopa IR.
    pub fn generate(
        self,
        id: FunctionId,
        program_ctx: &mut ProgramContext,
        function_ctx: &FunctionContext,
        opt_level: u8,
    ) -> riscv::Function {
        // Generate code for the function.
        let mut func = riscv::Function::new(id);

        // Prologue and epilogue.
        let frame_size = function_ctx.frame.total;
        let mut prologue = imbl::Vector::new();
        let mut epilogue = imbl::Vector::new();
        if frame_size > 0 {
            prologue.extend([
                Inst::Li(RegId::T0, -frame_size),
                Inst::Add(RegId::SP, RegId::SP, RegId::T0),
            ]);
            epilogue.extend([
                Inst::Li(RegId::T0, frame_size),
                Inst::Add(RegId::SP, RegId::SP, RegId::T0),
            ]);
        }
        if !function_ctx.calls.is_leaf {
            let slot = FrameSlot::RetAddr;
            prologue = prologue + slot.store(&function_ctx.frame.size, RegId::RA);
            epilogue = slot.load(&function_ctx.frame.size, RegId::RA) + epilogue;
        }
        let prologue = prologue;
        let epilogue = epilogue;

        // Generate code for the instruction.
        for bb in function_ctx.dominators.iter() {
            let node = self.0.layout().bbs().node(&bb).unwrap();
            let id = function_ctx.bb_map[&bb];

            let mut block = BlockBuilder::new(Block::new());
            if self.0.layout().entry_bb() == Some(bb) {
                for &prologue in prologue.iter() {
                    block.push(prologue);
                }
            }
            for &inst in node.insts().keys() {
                Codegen(inst).generate(&mut block, program_ctx, function_ctx, epilogue.iter());
                block.add_annotation(utils::dbg_inst(inst, function_ctx.dfg));
            }
            let mut block = block.build();

            // peephole optimization.
            peephole::optimize(&mut block, opt_level);

            func.push(id, block);
        }

        func
    }
}

impl FrameSlot {
    /// Insts to load from slot to reg
    pub fn load(self, frame_size: &FrameSize, reg: RegId) -> imbl::Vector<Inst> {
        let slot = self.offset(frame_size);
        if let Ok(slot) = i12::try_from(slot) {
            [Inst::Lw(reg, slot, RegId::SP)].into()
        } else {
            [
                Inst::Li(RegId::T0, slot),
                Inst::Add(RegId::T0, RegId::SP, RegId::T0),
                Inst::Lw(reg, i12::ZERO, RegId::T0),
            ]
            .into()
        }
    }

    /// Insts to store from reg to slot
    pub fn store(self, frame_size: &FrameSize, reg: RegId) -> imbl::Vector<Inst> {
        let slot = self.offset(frame_size);
        if let Ok(slot) = i12::try_from(slot) {
            [Inst::Sw(reg, slot, RegId::SP)].into()
        } else {
            [
                Inst::Li(RegId::T0, slot),
                Inst::Add(RegId::T0, RegId::SP, RegId::T0),
                Inst::Sw(reg, i12::ZERO, RegId::T0),
            ]
            .into()
        }
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
        program_ctx: &ProgramContext,
        function_ctx: &FunctionContext,
        epilogue: impl Iterator<Item = &'a Inst> + 'a,
    ) {
        let dfg = function_ctx.dfg;
        let regs = &function_ctx.regs;
        let global = &program_ctx.global;
        let local_vars = &function_ctx.local_vars;
        let frame = &function_ctx.frame;
        let value = dfg.value(self.0);
        trace!("GEN " => "{}", utils::dbg_inst(self.0, dfg));
        match value.kind() {
            ValueKind::Return(ret) => {
                if let Some(val) = ret.value() {
                    // Load the return value to a0.
                    Codegen(val).load_value_to_reg(block, function_ctx, RegId::A0);
                }
                for &epilogue in epilogue.into_iter() {
                    block.push(epilogue);
                }
                block.push(Inst::Ret);
            }
            ValueKind::Binary(bin) => {
                let lhs = Codegen(bin.lhs()).load_value(block, function_ctx, RegId::T0);
                let rhs = Codegen(bin.rhs()).load_value(block, function_ctx, RegId::T1);

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
                        _ => panic!("unexpected binary op: {:}", bin.op()),
                    }
                }

                // Write the result to stack if necessary.
                if let Storage::Slot(slot) = storage {
                    block.push_batch(slot.store(&frame.size, reg));
                }
            }
            ValueKind::Alloc(_) => {
                // Nothing to do here.
            }
            ValueKind::Store(store) => {
                let val = Codegen(store.value());
                if global.values.contains_key(&store.dest()) {
                    // global variable
                    block.push(Inst::La(RegId::A0, global.values[&store.dest()].id));
                    val.load_value_to_reg(block, function_ctx, RegId::T0);
                    block.push(Inst::Sw(RegId::T0, i12::ZERO, RegId::A0));
                } else if local_vars.map.contains_key(&store.dest()) {
                    // local variable
                    let slot = local_vars.map[&store.dest()];
                    val.load_value_to_reg(block, function_ctx, RegId::A0);
                    block.push_batch(slot.store(&frame.size, RegId::A0));
                } else {
                    // pointer
                    let dest = Codegen(store.dest()).load_value(block, function_ctx, RegId::A0);
                    val.load_value_to_reg(block, function_ctx, RegId::T0);
                    block.push(Inst::Sw(RegId::T0, i12::ZERO, dest));
                }
            }
            ValueKind::Load(load) => {
                if global.values.contains_key(&load.src()) {
                    // global variable
                    block.push(Inst::La(RegId::A0, global.values[&load.src()].id));
                    block.push(Inst::Lw(RegId::A0, i12::ZERO, RegId::A0));
                } else if local_vars.map.contains_key(&load.src()) {
                    // local variable
                    let src = local_vars.map[&load.src()];
                    block.push_batch(src.load(&frame.size, RegId::A0));
                } else {
                    // pointer
                    let src = Codegen(load.src()).load_value(block, function_ctx, RegId::A0);
                    block.push(Inst::Lw(RegId::A0, i12::ZERO, src));
                }
                match regs.map[&self.0] {
                    Storage::Reg(reg) => {
                        // Load the value to the register.
                        block.push(Inst::Mv(reg, RegId::A0));
                    }
                    Storage::Slot(slot) => {
                        // Write the value to stack.
                        block.push_batch(slot.store(&frame.size, RegId::A0));
                    }
                }
            }
            ValueKind::Branch(branch) => {
                let cond = Codegen(branch.cond()).load_value(block, function_ctx, RegId::A0);
                let then = function_ctx.bb_map[&branch.true_bb()];
                let els = function_ctx.bb_map[&branch.false_bb()];
                block.push(Inst::Beqz(cond, els));
                block.push(Inst::J(then));
            }
            ValueKind::Jump(jump) => {
                let target = function_ctx.bb_map[&jump.target()];
                block.push(Inst::J(target));
            }
            ValueKind::Call(call) => {
                let func = program_ctx.func_map[&call.callee()];

                // Save registers.
                let save_regs = function_ctx.calls.save_regs[&self.0];
                for (i, reg) in save_regs.iter().enumerate() {
                    let slot = FrameSlot::Saved(i as i32 * 4);
                    block.push_batch(slot.store(&frame.size, reg))
                }

                // Set up arguments.

                // Save 9th and later arguments to stack.
                for (i, &arg) in call.args().iter().skip(8).enumerate() {
                    let slot = FrameSlot::Arg(i as i32 * 4);
                    // load in reverse order, so a0 is available here
                    Codegen(arg).load_value_to_reg(block, function_ctx, RegId::A0);
                    block.push_batch(slot.store(&frame.size, RegId::A0));
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
                    use petgraph::prelude::*;
                    let mut graph = DiGraphMap::with_capacity(nodes, nodes);
                    for (&target, &arg) in moves.iter() {
                        if target != arg {
                            graph.add_edge(target, arg, ());
                        }
                    }
                    let mut scc = petgraph::algo::tarjan_scc(&graph);
                    scc.reverse(); // topological order
                    trace!("CALL" => "scc: {:?}", scc);
                    for scc in scc.into_iter() {
                        if scc.len() == 1 {
                            let target = scc[0];
                            if let Some(&arg) = moves.get(&target) {
                                block.push(Inst::Mv(target, arg));
                            }
                        } else {
                            let mut target = scc[0];
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
                }

                for (target, arg) in after_moves.into_iter() {
                    Codegen(arg).load_value_to_reg(block, function_ctx, target);
                }

                // Call the function.
                block.push(Inst::Call(func));

                // Fetch return value.
                if !value.ty().is_unit() {
                    let storage = regs.map[&self.0];
                    match storage {
                        Storage::Reg(reg) => block.push(Inst::Mv(reg, RegId::A0)),
                        Storage::Slot(slot) => block.push_batch(slot.store(&frame.size, RegId::A0)),
                    };
                }

                // Restore registers.
                for (i, reg) in save_regs.iter().enumerate() {
                    let slot = FrameSlot::Saved(i as i32 * 4);
                    block.push_batch(slot.load(&frame.size, reg))
                }
            }
            ValueKind::GetElemPtr(get_elem_ptr) => {
                let index = get_elem_ptr.index();
                let src = get_elem_ptr.src();
                let base_size = if dfg.values().contains_key(&src) {
                    let ty = dfg.value(src).ty();
                    match ty.kind() {
                        TypeKind::Pointer(ptr) => match ptr.kind() {
                            TypeKind::Array(base, _) => base.size(),
                            _ => panic!("unexpected type: {:}", ptr),
                        },
                        _ => panic!("unexpected type: {:}", ty),
                    }
                } else {
                    global.values[&src].base_size
                };
                Codegen(src).load_offset(
                    index,
                    base_size,
                    block,
                    program_ctx,
                    function_ctx,
                    regs.map[&self.0],
                );
            }
            ValueKind::GetPtr(get_ptr) => {
                let index = get_ptr.index();
                let src = get_ptr.src();
                let base_size = if dfg.values().contains_key(&src) {
                    let ty = dfg.value(src).ty();
                    match ty.kind() {
                        TypeKind::Pointer(base) => base.size(),
                        _ => panic!("unexpected type: {:}", ty),
                    }
                } else {
                    global.values[&src].base_size
                };
                Codegen(src).load_offset(
                    index,
                    base_size,
                    block,
                    program_ctx,
                    function_ctx,
                    regs.map[&self.0],
                )
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
        function_ctx: &FunctionContext,
        temp: RegId,
    ) -> RegId {
        let data = self.data(function_ctx.dfg);
        if data.is_const() {
            data.load_const(block, temp)
        } else {
            match function_ctx.regs.map[&self.0] {
                Storage::Reg(reg) => reg,
                Storage::Slot(slot) => {
                    let slot = slot.offset(&function_ctx.frame.size);
                    if let Ok(slot) = i12::try_from(slot) {
                        block.push(Inst::Lw(temp, slot, RegId::SP));
                    } else {
                        block.push(Inst::Li(temp, slot));
                        block.push(Inst::Add(temp, RegId::SP, temp));
                        block.push(Inst::Lw(temp, i12::ZERO, temp));
                    }
                    temp
                }
            }
        }
    }

    /// Load value to a register.
    fn load_value_to_reg(
        &self,
        block: &mut BlockBuilder,
        function_ctx: &FunctionContext,
        reg: RegId,
    ) {
        let res = self.load_value(block, function_ctx, reg);
        if res != reg {
            block.push(Inst::Mv(reg, res));
        }
    }

    /// Load offset of pointer value
    fn load_offset(
        self,
        index: koopa::ir::Value,
        base_size: usize,
        block: &mut BlockBuilder,
        program_ctx: &ProgramContext,
        function_ctx: &FunctionContext,
        storage: Storage,
    ) {
        let global = &program_ctx.global;
        let local_vars = &function_ctx.local_vars;
        let frame = &function_ctx.frame;
        let index = Codegen(index).load_value(block, function_ctx, RegId::T0);

        block.push(Inst::Li(RegId::T1, base_size as i32));
        block.push(Inst::Mul(RegId::T0, index, RegId::T1));
        if global.values.contains_key(&self.0) {
            // global variable
            block.push(Inst::La(RegId::A0, global.values[&self.0].id));
        } else if local_vars.map.contains_key(&self.0) {
            // local variable
            let src = local_vars.map[&self.0];
            let src = src.offset(&frame.size);
            block.push(Inst::Li(RegId::T1, src));
            block.push(Inst::Add(RegId::A0, RegId::SP, RegId::T1));
        } else {
            // pointer
            self.load_value_to_reg(block, function_ctx, RegId::A0);
        }

        match storage {
            Storage::Reg(reg) => {
                // Load the value to the register.
                block.push(Inst::Add(reg, RegId::A0, RegId::T0));
            }
            Storage::Slot(slot) => {
                // Write the value to stack.
                let slot = slot.offset(&frame.size);
                block.push(Inst::Li(RegId::T1, slot));
                block.push(Inst::Add(RegId::T1, RegId::SP, RegId::T1));
                block.push(Inst::Add(RegId::A0, RegId::A0, RegId::T0));
                block.push(Inst::Sw(RegId::A0, i12::ZERO, RegId::T1));
            }
        }
    }
}

impl Codegen<&koopa::ir::entities::ValueData> {
    /// Load a constant.
    /// Return the register that contains the constant.
    pub fn load_const(self, block: &mut BlockBuilder, temp: RegId) -> RegId {
        match self.0.kind() {
            ValueKind::Integer(i) => {
                if i.value() == 0 {
                    RegId::X0
                } else {
                    block.push(Inst::Li(temp, i.value()));
                    temp
                }
            }
            ValueKind::ZeroInit(_) => RegId::X0,
            _ => panic!("unexpected value kind: {:?}", self.0),
        }
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
