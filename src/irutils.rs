//! Utility functions for the IR.

use koopa::ir::{dfg::DataFlowGraph, entities::ValueData, Value, ValueKind};

/// Check if a value is a constant.
///
/// Constants are nameless values that need no computation.
pub fn is_const(inst: &ValueData) -> bool {
    matches!(inst.kind(), ValueKind::Integer(_) | ValueKind::ZeroInit(_))
}

#[cfg(test)]
mod test_is_const {
    use super::*;

    #[test]
    fn int_is_const() {
        use koopa::ir::builder_traits::*;
        use koopa::ir::*;

        let mut program = Program::new();
        let func = program.new_func(FunctionData::with_param_names(
            "@int_is_const".to_string(),
            vec![],
            Type::get_i32(),
        ));
        let func = program.func_mut(func);
        let dfg = func.dfg_mut();
        let value = dfg.new_value().integer(0);
        assert!(is_const(dfg.value(value)));
    }

    #[test]
    fn expr_is_not_const() {
        use koopa::ir::builder_traits::*;
        use koopa::ir::*;

        let mut program = Program::new();
        let func = program.new_func(FunctionData::with_param_names(
            "@expr_is_not_const".to_string(),
            vec![],
            Type::get_i32(),
        ));
        let func = program.func_mut(func);

        let v1 = func.dfg_mut().new_value().integer(0);
        let v2 = func.dfg_mut().new_value().integer(1);
        let value = func.dfg_mut().new_value().binary(BinaryOp::Add, v1, v2);

        assert!(!is_const(func.dfg().value(value)));
    }
}

/// Check if a value can terminate a basic block.
#[allow(dead_code)]
pub fn is_terminator(inst: &ValueData) -> bool {
    matches!(
        inst.kind(),
        ValueKind::Branch(_) | ValueKind::Return(_) | ValueKind::Jump(_)
    )
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
