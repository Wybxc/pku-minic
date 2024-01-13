//! Register allocation.
//!
//! ## Register usage
//!
//! RISC-V has 32 registers in total, 15 of which can be used for
//! local variables.
//!
//! Each IR instruction will use:
//! - 0~1 registers for result
//! - 0~2 registers for temporary values
//! - 0~3 registers for temporary storage of spilled values
//!
//! We allocate 1 register for each instruction with a result, and
//! reserve 3 registers for temporary storage. See [`REG_LOCAL_COUNT`]
//! and [`REG_LOCAL`].
//!
//! ## Algorithm
//!
//! We adopt a static method: once a variable is assigned a register,
//! it will not be changed.
//!
//! First we do a live variable analysis. Each IR instruction may drop
//! at most 2 operands values, and generate at most 1 result value.
//! The generated value can be obviously referred from the instruction,
//! and we keep track of the dropped values.
//!
//! Then we do a linear scan. We scan the instructions in a basic block,
//! and allocate registers for results. If there is no free register,
//! we spill a variable to the stack.
//!
//! ## TODO
//!
//! - Now we always spill the newest variable, which is not optimal. We should
//!   spill the variable that will not be used for a long time.
//! - We do not allocate a0, t0, t1, since they are reserved for return value
//!   and temporary storage. However, we can use them if they are not used in
//!   some time.

use std::collections::HashMap;

use koopa::ir::{FunctionData, Value};
#[allow(unused_imports)]
use nolog::*;

#[allow(unused_imports)]
use crate::utils;
use crate::{
    analysis::{dominators::Dominators, liveliness::Liveliness},
    codegen::riscv::{make_reg_set, FrameSlot, RegId, RegSet},
};

/// Storage for a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    /// Register.
    Reg(RegId),
    /// Stack offset from sp, in bytes.
    Slot(FrameSlot),
}

/// Ids of registers that can be used.
/// There are 15 registers in total (a0..a7, t0..t6), but we reserve
/// a0 for return value, and t0..t1 for temporary storage.
const REG_LOCAL_MASK: u32 = make_reg_set!(A1, A2, A3, A4, A5, A6, A7, T2, T3, T4, T5, T6);
type LocalRegSet = RegSet<REG_LOCAL_MASK>;

impl Storage {
    fn local(self) -> Option<RegId> {
        match self {
            Storage::Reg(reg) => Some(reg),
            _ => None,
        }
    }
}

/// Register allocation.
pub struct RegAlloc {
    /// Map from value to storage.
    pub map: HashMap<Value, Storage>,
    /// Frame size.
    pub frame_size: i32,
}

impl RegAlloc {
    /// Analyse register allocation.
    pub fn analyze(liveliness: &Liveliness, dominators: &Dominators, func: &FunctionData) -> Self {
        // Get live variables.
        let live_in = &liveliness.live_in;
        let live_out = &liveliness.live_out;

        // Linear scan.
        // We adopt a static method: once a variable is assigned a register,
        // it will not be changed. This is not optimal, and might to be improved later.
        let mut map = HashMap::<Value, Storage>::new();
        let mut frame_size = 0;

        for (i, &param) in func.params().iter().enumerate() {
            match i {
                0 => map.insert(param, Storage::Reg(RegId::A0)),
                1 => map.insert(param, Storage::Reg(RegId::A1)),
                2 => map.insert(param, Storage::Reg(RegId::A2)),
                3 => map.insert(param, Storage::Reg(RegId::A3)),
                4 => map.insert(param, Storage::Reg(RegId::A4)),
                5 => map.insert(param, Storage::Reg(RegId::A5)),
                6 => map.insert(param, Storage::Reg(RegId::A6)),
                7 => map.insert(param, Storage::Reg(RegId::A7)),
                _ => map.insert(param, Storage::Slot(FrameSlot::Param((i - 8) as i32 * 4))),
            };
        }

        // Scan basic blocks in topological order.
        let bbs = func.layout().bbs();
        for bb in dominators.iter() {
            let node = bbs.node(&bb).unwrap();

            // Free registers. Initially all registers are free, and we will
            // change the status while scanning the instructions.
            let mut free = LocalRegSet::full();
            for &val in live_in[&bb].iter() {
                // live variables must be allocated before, since basic blocks
                // are in topological order.
                if let Some(reg) = map.get(&val).copied().and_then(Storage::local) {
                    free.remove(reg);
                }
            }

            for &inst in node.insts().keys() {
                // Drop dead variables.
                for val in live_out[&inst].iter() {
                    if let Some(reg) = map.get(val).copied().and_then(Storage::local) {
                        free.insert(reg);
                    }
                }

                // Allocate registers for results.
                let value = func.dfg().value(inst);
                trace!(->[0] "REG " => "allocating for `{}`", utils::dbg_inst(inst, func.dfg()));
                if !value.ty().is_unit() && !matches!(value.kind(), koopa::ir::ValueKind::Alloc(_))
                {
                    // Not a unit type and not an `alloc` instruction, allocate a register.
                    let try_reg = free.iter().next();
                    if let Some(reg) = try_reg {
                        // Found a free register.
                        map.insert(inst, Storage::Reg(reg));
                        free.remove(reg);

                        trace!(
                            "REG " => "allocate `{}` to `{reg}`",
                            utils::ident_inst(inst, func.dfg())
                        );
                    } else {
                        // No free register, spill a variable.
                        map.insert(inst, Storage::Slot(FrameSlot::Spilled(frame_size)));

                        trace!(
                            "REG " => "spill {} to stack sp+{frame_size}",
                            utils::ident_inst(inst, func.dfg())
                        );

                        frame_size += 4;
                    }
                }
            }
        }

        Self { map, frame_size }
    }
}
