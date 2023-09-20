//! Register allocation.
//!
//! Each IR instruction will use:
//! - 0~1 registers for result
//! - 0~2 registers for temporary values
//! - 0~3 registers for temporary storage of spilled values
//!
//! We allocate 1 register for each instruction with a result, and
//! reserve 3 registers for temporary storage.

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

use koopa::ir::{dfg::DataFlowGraph, FunctionData, Value, ValueKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegId {
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
}

impl Display for RegId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegId::A0 => write!(f, "a0"),
            RegId::A1 => write!(f, "a1"),
            RegId::A2 => write!(f, "a2"),
            RegId::A3 => write!(f, "a3"),
            RegId::A4 => write!(f, "a4"),
            RegId::A5 => write!(f, "a5"),
            RegId::A6 => write!(f, "a6"),
            RegId::A7 => write!(f, "a7"),
            RegId::T0 => write!(f, "t0"),
            RegId::T1 => write!(f, "t1"),
            RegId::T2 => write!(f, "t2"),
            RegId::T3 => write!(f, "t3"),
            RegId::T4 => write!(f, "t4"),
            RegId::T5 => write!(f, "t5"),
            RegId::T6 => write!(f, "t6"),
        }
    }
}

/// Storage for a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    /// Register.
    Reg(RegId),
    /// Stack offset from sp, in bytes.
    Slot(u32),
}

/// Count of registers that can be used.
/// There are 15 registers in total (a0..a7, t0..t6), but we reserve
/// a0 for return value, and t0..t1 for temporary storage.
const REG_LOCAL_COUNT: usize = 12;

/// Ids of registers that can be used.
const REG_LOCAL: [RegId; REG_LOCAL_COUNT] = [
    RegId::A1,
    RegId::A2,
    RegId::A3,
    RegId::A4,
    RegId::A5,
    RegId::A6,
    RegId::A7,
    RegId::T2,
    RegId::T3,
    RegId::T4,
    RegId::T5,
    RegId::T6,
];

impl Storage {
    fn local(self) -> Option<usize> {
        match self {
            Storage::Reg(reg) => REG_LOCAL.iter().position(|&r| r == reg),
            _ => None,
        }
    }
}

pub struct RegAlloc {
    map: HashMap<Value, Storage>,
    frame_size: u32,
}

impl RegAlloc {
    pub fn new(func: &FunctionData) -> Self {
        // Live variable analysis.
        // Currently there is no control flow, so we can do it in a simple way.
        // - live in: Which variables are live when entering a basic block?
        // - live out: Which variables are dropped after the instruction?
        let mut live_in = HashMap::new();
        let mut live_out = HashMap::new();
        for (&bb, node) in func.layout().bbs() {
            live_in.insert(bb, HashSet::<Value>::new()); // Now there is only an entry block, so nothing is live in.

            // In reverse order scan the instructions in a basic block,
            // the last appearing variable is live out.
            let insts = node.insts().keys().collect::<Vec<_>>();
            let mut scanned = HashSet::new();
            for &inst in insts.into_iter().rev() {
                let mut ops = operand_vars(inst, func.dfg());
                for op in ops.iter_mut() {
                    if let Some(val) = op {
                        if !scanned.insert(*val) {
                            // The variable has been scanned, it will not die here.
                            *op = None;
                        }
                    }
                }
                live_out.insert(inst, ops);
            }
        }

        // Linear scan.
        // We adopt a static method: once a variable is assigned a register,
        // it will not be changed. This is not optimal, and might to be improved later.
        let mut map = HashMap::<Value, Storage>::new();
        let mut sp = 0; // Stack pointer.

        // TODO: Basic blocks must in CFG topological order.
        for (&bb, node) in func.layout().bbs() {
            // Free registers. Initially all registers are free, and we will
            // change the status while scanning the instructions.
            let mut free = [true; REG_LOCAL_COUNT];
            for &val in live_in[&bb].iter() {
                // live variables must be allocated before, since basic blocks
                // are in topological order.
                if let Some(reg) = map[&val].local() {
                    free[reg] = false;
                }
            }

            for &inst in node.insts().keys() {
                // Allocate registers for results.
                let value = func.dfg().value(inst);
                if !value.ty().is_unit() {
                    // Not a unit type, allocate a register.
                    if let Some(reg) = free.iter().position(|&f| f) {
                        // Found a free register.
                        map.insert(inst, Storage::Reg(REG_LOCAL[reg]));
                        free[reg] = false;
                    } else {
                        // No free register, spill a variable.
                        map.insert(inst, Storage::Slot(sp));
                        sp += 4;
                    }
                }

                // Drop dead variables.
                for val in live_out[&inst].iter().filter_map(|&v| v) {
                    if let Some(reg) = map[&val].local() {
                        free[reg] = true;
                    }
                }
            }
        }

        Self {
            map,
            frame_size: sp,
        }
    }

    /// Get storage of a value.
    pub fn get(&self, value: Value) -> Storage {
        self.map[&value]
    }

    /// Minimum size of stack frame, in bytes.
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }
}

/// Get operands of an instruction, only return variables.
///
/// Variables are values that have a name.
fn operand_vars(value: Value, dfg: &DataFlowGraph) -> [Option<Value>; 2] {
    let is_var = |&v: &Value| dfg.value(v).name().is_some();
    match dfg.value(value).kind() {
        ValueKind::Return(ret) => {
            let val = ret.value().filter(is_var);
            [val, None]
        }
        ValueKind::Binary(bin) => {
            let lhs = bin.lhs();
            let rhs = bin.rhs();
            [Some(lhs).filter(is_var), Some(rhs).filter(is_var)]
        }
        _ => [None; 2],
    }
}
