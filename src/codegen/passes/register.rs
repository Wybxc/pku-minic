use std::collections::{BTreeMap, HashMap};

#[allow(unused_imports)]
use nolog::*;

use crate::{
    codegen::{
        liveliness::{ControlFlowGraph, Dominators, Liveliness},
        riscv::{Function, Inst, PseudoInst, RegId},
    },
    common::cfg::ControlFlowInst,
};

pub fn register_alloc(function: &mut Function) {
    let cfg = ControlFlowGraph::analyze(
        function.iter().map(|(id, block)| {
            let inst = block.get(block.back().unwrap()).unwrap();
            let inst = match inst {
                Inst::Beq(_, _, bb)
                | Inst::Beqz(_, bb)
                | Inst::Bge(_, _, bb)
                | Inst::Bgeu(_, _, bb)
                | Inst::Bgez(_, bb)
                | Inst::Bgt(_, _, bb)
                | Inst::Bgtu(_, _, bb)
                | Inst::Bgtz(_, bb)
                | Inst::Ble(_, _, bb)
                | Inst::Bleu(_, _, bb)
                | Inst::Blez(_, bb)
                | Inst::Blt(_, _, bb)
                | Inst::Bltu(_, _, bb)
                | Inst::Bltz(_, bb)
                | Inst::Bne(_, _, bb)
                | Inst::Bnez(_, bb) => ControlFlowInst::Branch(bb, function.follow(id).unwrap()),
                Inst::J(bb) => ControlFlowInst::Jump(bb),
                Inst::Ret => ControlFlowInst::Return,
                _ => ControlFlowInst::Jump(function.follow(id).unwrap()),
            };
            (id, inst)
        }),
        function.entry().unwrap(),
    );
    let dominators = Dominators::analyze(&cfg);
    let liveliness = Liveliness::analysis(&cfg, function);

    // Compute live variables
    let mut end_time = HashMap::new();
    let mut i = 0;
    for bb in dominators.iter() {
        let block = function.get(bb).unwrap();
        for (id, _inst) in block.insts() {
            for live_out in liveliness.live_out[&id].iter().flatten() {
                end_time.insert(*live_out, i);
            }
            trace!("REG " => "{}", _inst);
            i += 1
        }
    }

    // Register map
    let mut alloc: BTreeMap<RegId, Option<RegId>> = {
        use RegId::*;
        [A1, A2, A3, A4, A5, A6, A7, T2, T3, T4, T5, T6]
            .map(|reg| (reg, None))
            .into_iter()
            .collect()
    };
    let mut result: HashMap<RegId, RegId> = HashMap::new();

    // Spill variables
    let mut spilled: HashMap<RegId, i32> = HashMap::new();
    let mut offset = 0;
    let mut spill = |reg: RegId| {
        spilled.insert(reg, offset);
        trace!("REG " => "spill {} to {}", reg, offset);
        offset += 4;
    };

    // Allocate registers
    for bb in dominators.iter() {
        let block = function.get(bb).unwrap();

        for (id, inst) in block.insts() {
            if let Some(dest) = inst.dest() {
                let phy_reg = match dest {
                    RegId::Pseudo(_) => {
                        // Logical register
                        if let Some((&phy_reg, _)) = alloc.iter().rev().find(|(_, v)| v.is_none()) {
                            phy_reg
                        } else {
                            // select a register to spill
                            let (phy_reg, selected) = alloc
                                .iter()
                                .map(|(k, v)| (*k, v.unwrap()))
                                .filter(|(_, v)| matches!(v, RegId::Pseudo(_)))
                                .max_by_key(|(_, v)| end_time.get(v).copied().unwrap_or(i32::MAX))
                                .unwrap();
                            spill(selected);
                            phy_reg
                        }
                    }
                    _ => {
                        // Physical register
                        if let Some(Some(reg)) = alloc.get(&dest) {
                            spill(*reg);
                        }
                        dest
                    }
                };
                alloc.insert(phy_reg, Some(dest));
                result.insert(dest, phy_reg);
                trace!(->[0] "REG " => "alloc {} to {}", dest, phy_reg);
            }
            for live_out in liveliness.live_out[&id].iter().flatten() {
                for (reg, val) in alloc.iter_mut() {
                    if reg == live_out {
                        trace!("REG " => "free {:?} from {}", val, reg);
                        *val = None;
                    }
                }
            }
        }
    }

    // Perform spill
    for bb in dominators.iter() {
        let block = function.get_mut(bb).unwrap();
        let mut cursor = block.cursor_mut(block.front().unwrap());

        while !cursor.is_null() {
            let mut inst = cursor.inst().unwrap();
            let mut ops = inst.source_mut();
            if let Some(op) = &mut ops[0] {
                if let Some(&offset) = spilled.get(op) {
                    cursor.insert_before(Inst::Pseudo(PseudoInst::LoadFrameBottom(
                        RegId::T0,
                        offset.try_into().unwrap(),
                    )));
                    **op = RegId::T0;
                }
            }
            if let Some(op) = &mut ops[1] {
                if let Some(&offset) = spilled.get(op) {
                    cursor.insert_before(Inst::Pseudo(PseudoInst::LoadFrameBottom(
                        RegId::T1,
                        offset.try_into().unwrap(),
                    )));
                    **op = RegId::T1;
                }
            }
            if let Some(dest) = inst.dest_mut() {
                if let Some(&offset) = spilled.get(dest) {
                    cursor.insert_after(Inst::Pseudo(PseudoInst::StoreFrameBottom(
                        RegId::T0,
                        offset.try_into().unwrap(),
                    )));
                    *dest = RegId::T0;
                }
            }
            cursor.set_inst(inst);
            cursor.next();
        }
    }

    // Replace registers
    for bb in dominators.iter() {
        let block = function.get_mut(bb).unwrap();
        let mut cursor = block.cursor_mut(block.front().unwrap());

        while !cursor.is_null() {
            let mut inst = cursor.inst().unwrap();
            let mut ops = inst.source_mut();
            if let Some(op) = &mut ops[0] {
                if let Some(&reg) = result.get(op) {
                    **op = reg;
                }
            }
            if let Some(op) = &mut ops[1] {
                if let Some(&reg) = result.get(op) {
                    **op = reg;
                }
            }
            if let Some(dest) = inst.dest_mut() {
                if let Some(&reg) = result.get(dest) {
                    *dest = reg;
                }
            }
            cursor.set_inst(inst);
            cursor.next();
        }
    }
}
