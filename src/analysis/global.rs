//! Global values.

use std::collections::HashMap;

use koopa::ir::{Program, Value, ValueKind};

use crate::codegen::riscv::GlobalId;

/// Global values.
pub struct GlobalValues {
    /// Global values.
    pub values: HashMap<Value, GlobalValueData>,
}

/// Global value data.
pub struct GlobalValueData {
    /// Global value id.
    pub id: GlobalId,
    /// Global value size.
    pub size: usize,
    /// Initial value.
    pub init: Option<Vec<i32>>,
}

impl GlobalValues {
    /// Analyze global values.
    pub fn analyze(program: &Program) -> Self {
        let mut values = HashMap::new();
        let global_values = program.borrow_values();
        for (&value, data) in global_values.iter() {
            if let ValueKind::GlobalAlloc(alloc) = data.kind() {
                let name = data.name().as_ref().unwrap()[1..].to_string();
                let init = alloc.init();
                let data = &global_values[&init];

                let id = GlobalId::next_id();
                id.set_name(name);
                let size = data.ty().size();
                let init = match data.kind() {
                    ValueKind::Integer(i) => Some(vec![i.value()]),
                    ValueKind::ZeroInit(_) => None,
                    _ => todo!(),
                };
                values.insert(value, GlobalValueData { id, size, init });
            }
        }
        Self { values }
    }
}
