//! Utility functions for the IR.

use koopa::ir::{dfg::DataFlowGraph, entities::ValueData, Value, ValueKind};

/// Check if a value is a constant.
///
/// Constants are nameless values that need no computation.
pub fn is_const(inst: &ValueData) -> bool {
    matches!(inst.kind(), ValueKind::Integer(_) | ValueKind::ZeroInit(_))
}

/// Debug print an instruction.
pub fn dbg_inst(inst: Value, dfg: &DataFlowGraph) -> String {
    let value = dfg.value(inst);
    let name = value.name().as_deref().unwrap_or("<>");
    format!("{}: {:?}", name, value.kind())
}
