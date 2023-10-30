//! Utility functions for the IR.

use koopa::ir::{dfg::DataFlowGraph, entities::ValueData, Value, ValueKind};

/// Check if a value is a constant.
///
/// Constants are nameless values that need no computation.
pub fn is_const(inst: &ValueData) -> bool {
    matches!(inst.kind(), ValueKind::Integer(_) | ValueKind::ZeroInit(_))
}

/// Get the name of a value for debug printing.
pub fn ident_inst(inst: Value, dfg: &DataFlowGraph) -> String {
    dfg.value(inst)
        .name()
        .clone()
        .unwrap_or_else(|| format!("{:?}", inst))
}

/// Debug print an instruction.
pub fn dbg_inst(inst: Value, dfg: &DataFlowGraph) -> String {
    let value = dfg.value(inst);
    if let Some(name) = value.name().as_deref() {
        format!("{}: {:?} := {:?}", name, inst, value.kind())
    } else {
        format!("{:?} := {:?}", inst, value.kind())
    }
}
