//! Local variable analysis.

use std::collections::HashMap;

use koopa::ir::{FunctionData, Value, ValueKind};

use crate::codegen::riscv::FrameSlot;

/// Local variables.
pub struct LocalVars {
    /// Map from variable to its storage.
    pub map: HashMap<Value, FrameSlot>,
    /// Frame size.
    pub frame_size: i32,
}

impl LocalVars {
    /// Analyse local variables.
    pub fn analyze(func: &FunctionData) -> Self {
        let mut map = HashMap::new();
        let mut frame_size = 0;

        let dfg = func.dfg();
        let bbs = func.layout().bbs();
        for (_, node) in bbs.iter() {
            for &inst in node.insts().keys() {
                if let ValueKind::Alloc(_) = dfg.value(inst).kind() {
                    let size = 4; // i32
                    let slot = FrameSlot::Local(frame_size);
                    frame_size += size;
                    map.insert(inst, slot);
                }
            }
        }

        Self { map, frame_size }
    }
}
