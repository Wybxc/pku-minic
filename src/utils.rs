//! Utility functions for the IR.
#![allow(dead_code)]

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

use koopa::ir::{dfg::DataFlowGraph, entities::ValueData, BasicBlock, Value, ValueKind};

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
        use koopa::ir::{builder_traits::*, *};

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
        use koopa::ir::{builder_traits::*, *};

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
pub fn is_terminator(inst: &ValueData) -> bool {
    matches!(
        inst.kind(),
        ValueKind::Branch(_) | ValueKind::Return(_) | ValueKind::Jump(_)
    )
}

thread_local! {
    /// The next unique ID for a value.
    static NEXT_ID: Cell<i32> = Cell::new(0);
    /// Name map for values.
    static NAME_MAP: RefCell<HashMap<Value, String>> = RefCell::new(HashMap::new());
}

/// Get the name of a value for debug printing.
pub fn ident_inst(inst: Value, dfg: &DataFlowGraph) -> String {
    let value = dfg.value(inst);
    value.name().clone().unwrap_or_else(|| {
        NAME_MAP.with(|map| {
            let mut map = map.borrow_mut();
            if let Some(name) = map.get(&inst) {
                return name.clone();
            }
            let name = match value.kind() {
                ValueKind::Integer(i) => format!("{}", i.value()),
                ValueKind::ZeroInit(_) => "0".to_string(),
                _ => format!("%{}", NEXT_ID.replace(NEXT_ID.get() + 1)),
            };
            map.insert(inst, name.clone());
            name
        })
    })
}

/// Debug print an instruction.
pub fn dbg_inst(inst: Value, dfg: &DataFlowGraph) -> String {
    let value = dfg.value(inst);
    let body = match value.kind() {
        ValueKind::Integer(i) => format!("{}", i.value()),
        ValueKind::Alloc(_) => format!("alloc {}", value.ty()),
        ValueKind::Load(v) => format!("load {}", ident_inst(v.src(), dfg)),
        ValueKind::Store(v) => format!(
            "store {}, {}",
            ident_inst(v.value(), dfg),
            ident_inst(v.dest(), dfg)
        ),
        ValueKind::GetPtr(v) => format!(
            "getptr {}, {}",
            ident_inst(v.src(), dfg),
            ident_inst(v.index(), dfg)
        ),
        ValueKind::GetElemPtr(v) => format!(
            "getelemptr {}, {}",
            ident_inst(v.src(), dfg),
            ident_inst(v.index(), dfg)
        ),
        ValueKind::Binary(v) => {
            let op = match v.op() {
                koopa::ir::BinaryOp::NotEq => "!=",
                koopa::ir::BinaryOp::Eq => "==",
                koopa::ir::BinaryOp::Gt => ">",
                koopa::ir::BinaryOp::Lt => "<",
                koopa::ir::BinaryOp::Ge => ">=",
                koopa::ir::BinaryOp::Le => "<=",
                koopa::ir::BinaryOp::Add => "+",
                koopa::ir::BinaryOp::Sub => "-",
                koopa::ir::BinaryOp::Mul => "*",
                koopa::ir::BinaryOp::Div => "/",
                koopa::ir::BinaryOp::Mod => "%",
                koopa::ir::BinaryOp::And => "&",
                koopa::ir::BinaryOp::Or => "|",
                koopa::ir::BinaryOp::Xor => "^",
                koopa::ir::BinaryOp::Shl => "<<",
                koopa::ir::BinaryOp::Shr => ">>",
                koopa::ir::BinaryOp::Sar => ">>>",
            };
            format!(
                "{} {} {}",
                ident_inst(v.lhs(), dfg),
                op,
                ident_inst(v.rhs(), dfg)
            )
        }
        ValueKind::Return(v) => {
            if let Some(value) = v.value() {
                format!("return {}", ident_inst(value, dfg))
            } else {
                "return".to_string()
            }
        }
        _ => todo!(),
    };
    if value.ty().is_unit() {
        body
    } else {
        let name = ident_inst(inst, dfg);
        format!("{} := {}", name, body)
    }
}

/// Get the name of a block for debug printing.
pub fn ident_block(block: BasicBlock, dfg: &DataFlowGraph) -> String {
    dfg.bb(block)
        .name()
        .clone()
        .unwrap_or_else(|| format!("{:?}", block))
}

/// Declare a unique identifier type.
#[macro_export]
macro_rules! declare_u32_id {
    ($name: ident) => {
        /// Unique identifier.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(::std::num::NonZeroU32);

        impl $name {
            /// Next unique identifier.
            pub fn next_id() -> Self {
                use ::std::cell::Cell;
                use ::std::num::NonZeroU32;
                thread_local! {
                    static COUNTER: Cell<NonZeroU32> = Cell::new(unsafe { NonZeroU32::new_unchecked(1) });
                }
                COUNTER.with(|counter| {
                    let id = counter.get();
                    counter.set(NonZeroU32::new(id.get() + 1).unwrap());
                    Self(id)
                })
            }
        }
    }
}

pub use declare_u32_id;
