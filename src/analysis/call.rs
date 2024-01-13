//! Function-call related analysis.

use std::collections::HashMap;

use koopa::ir::{FunctionData, Value, ValueKind};

use crate::{
    analysis::{
        liveliness::Liveliness,
        register::{RegAlloc, Storage},
    },
    codegen::riscv::RegSet,
};

/// Registers that need to be saved before a function call.
pub struct FunctionCalls {
    /// Map from call instruction to registers that need to be saved.
    pub save_regs: HashMap<Value, RegSet>,
    /// Frame size for saving registers.
    pub save_frame_size: i32,
    /// Frame size for arguments.
    pub arg_frame_size: i32,
    /// Whether this function is a leaf function.
    pub is_leaf: bool,
}

impl FunctionCalls {
    /// Analyse register saving.
    pub fn analyze(liveliness: &Liveliness, reg_alloc: &RegAlloc, func: &FunctionData) -> Self {
        let mut save_regs = HashMap::new();
        let mut save_frame_size = 0;
        let mut arg_frame_size = 0;
        let mut is_leaf = true;

        for (&bb, node) in func.layout().bbs() {
            if let Some(alive) = liveliness.live_in.get(&bb) {
                let mut alive = alive.clone();

                for &inst in node.insts().keys() {
                    let live_out = &liveliness.live_out[&inst];
                    for val in live_out.iter().flatten() {
                        alive.remove(val);
                    }

                    if let ValueKind::Call(call) = func.dfg().value(inst).kind() {
                        // `alive` now contains all the variables that are alive after this
                        // instruction. If this instruction is a call, we
                        // need to save all the variables in `alive`, unless
                        // they are stored in the stack.
                        let mut save = RegSet::new();
                        for &val in alive.iter() {
                            if let Storage::Reg(reg) = &reg_alloc.map[&val] {
                                save.insert(*reg);
                            }
                        }
                        save_frame_size = save_frame_size.max(save.len() as i32 * 4);
                        save_regs.insert(inst, save);

                        // If this function has more than 8 arguments, we need to allocate space
                        // for the remaining arguments.
                        let arg_count = call.args().len();
                        if arg_count > 8 {
                            let size = (arg_count - 8) as i32 * 4;
                            arg_frame_size = arg_frame_size.max(size);
                        }

                        // This function is not a leaf function.
                        is_leaf = false;
                    }

                    alive.insert(inst);
                }
            }
        }

        Self {
            save_regs,
            save_frame_size,
            arg_frame_size,
            is_leaf,
        }
    }
}
