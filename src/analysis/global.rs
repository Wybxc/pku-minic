//! Global values.

use std::collections::HashMap;

use koopa::ir::{entities::ValueData, Program, Value, ValueKind};

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
    /// Size of base type.
    pub base_size: usize,
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
                let size = data.ty().size(); // TODO: multiply overflow
                let base_size = match data.ty().kind() {
                    koopa::ir::TypeKind::Array(ty, _) => ty.size(),
                    _ => size,
                };
                let init = if let ValueKind::ZeroInit(_) = data.kind() {
                    None
                } else {
                    let mut init = Vec::new();
                    Self::make_init(&global_values, data, &mut init);
                    Some(init)
                };
                values.insert(
                    value,
                    GlobalValueData {
                        id,
                        size,
                        base_size,
                        init,
                    },
                );
            }
        }
        Self { values }
    }

    fn make_init(global: &HashMap<Value, ValueData>, data: &ValueData, init: &mut Vec<i32>) {
        match data.kind() {
            ValueKind::Integer(i) => {
                init.push(i.value());
            }
            ValueKind::Aggregate(v) => {
                for value in v.elems() {
                    Self::make_init(global, &global[value], init);
                }
            }
            _ => unreachable!(),
        }
    }
}
