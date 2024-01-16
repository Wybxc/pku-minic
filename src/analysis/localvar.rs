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
                let value = dfg.value(inst);
                if let ValueKind::Alloc(_) = value.kind() {
                    let ty = match value.ty().kind() {
                        koopa::ir::TypeKind::Pointer(ty) => ty,
                        _ => unreachable!(),
                    };
                    let size = ty.size() as i32;
                    let slot = FrameSlot::Local(frame_size);
                    frame_size += size;
                    map.insert(inst, slot);
                }
            }
        }

        Self { map, frame_size }
    }
}
