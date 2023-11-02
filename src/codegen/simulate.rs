//! Simulate the codegen output.
//!
//! We use this to test the optimizations we make in the codegen pass.
//! We call an optimization "correct" if it produces the a program that
//! bi-simulates with the original program, i.e. for arbitrary initial
//! states, the two programs produce the same final states.

use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};

use crate::codegen::riscv::{make_reg_set, Program, RegSet};

use super::riscv::{Inst, RegId};

/// Register file.
#[derive(Clone, PartialEq, Eq)]
pub struct Regs([i32; 32]);

impl Regs {
    /// Create a new register file.
    pub fn new(mut init: [i32; 32]) -> Self {
        init[RegId::X0 as usize] = 0;
        Self(init)
    }

    /// Generate an arbitrary register file.
    pub fn arbitrary() -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        proptest::collection::vec(any::<i32>(), 32).prop_map(|v| Self(v.try_into().unwrap()))
    }

    /// Get the value of a register.
    pub fn get(&self, reg: RegId) -> i32 {
        self.0[reg as usize]
    }

    /// Set the value of a register.
    pub fn set(&mut self, reg: RegId, value: i32) {
        if reg != RegId::X0 {
            self.0[reg as usize] = value;
        }
    }
}

impl Debug for Regs {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.0.iter().enumerate() {
            let reg: RegId = unsafe { std::mem::transmute(i as u8) };
            write!(f, "{} = {}, ", reg, v)?;
        }
        write!(f, "]")
    }
}

/// Memory.
///
/// We use a hashmap to simulate the memory. 32-bit addresses are split into
/// two parts: the lower 8 bits are the offset into a 256-byte page, and the
/// upper 24 bits are the page number. This is a simplification of the actual
/// RISC-V memory model, but it is sufficient for our purposes.
///
/// For simplicity, we assume that the memory is initialized to zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    pages: HashMap<i32, [u8; 256]>,
}

impl Memory {
    /// Create a new memory.
    pub fn new() -> Self {
        Self {
            pages: HashMap::new(),
        }
    }

    /// Read memory.
    pub fn read<const N: usize>(&self, addr: i32) -> [u8; N] {
        assert!(N <= 256, "memory read too large");

        let page_idx = addr >> 8;
        let offset = addr & 0xff;

        let start = offset as usize;
        let end = (start + N).min(256);
        let len = end - start;

        let mut result = [0; N];
        match self.pages.get(&page_idx) {
            Some(page) => {
                result[..len].copy_from_slice(&page[start..end]);
                if len < N {
                    if let Some(page) = self.pages.get(&(page_idx + 1)) {
                        result[len..].copy_from_slice(&page[..(N - len)]);
                    }
                }
                result
            }
            None => result,
        }
    }

    /// Write memory.
    pub fn write<const N: usize>(&mut self, addr: i32, value: [u8; N]) {
        assert!(N <= 256, "memory write too large");

        let page_idx = addr >> 8;
        let offset = addr & 0xff;

        let start = offset as usize;
        let end = (start + N).min(256);
        let len = end - start;

        self.pages.entry(page_idx).or_insert([0; 256])[start..end].copy_from_slice(&value[..len]);
        if len < N {
            self.pages.entry(page_idx + 1).or_insert([0; 256])[..(N - len)]
                .copy_from_slice(&value[len..]);
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

/// A simulator for the codegen output.
///
/// # Panics
/// - If the program does not contain a `main` function.
pub fn simulate(code: &Program, regs: &mut Regs, mem: &mut Memory) {
    let main = code.functions.iter().find(|f| f.name == "main").unwrap();

    for block in &main.blocks {
        for (_, inst) in block.insts() {
            match inst {
                Inst::Li(rd, imm) => regs.set(rd, imm),
                Inst::Mv(rd, rs) => regs.set(rd, regs.get(rs)),
                Inst::Add(rd, rs1, rs2) => regs.set(rd, regs.get(rs1).wrapping_add(regs.get(rs2))),
                Inst::Addi(rd, rs, imm) => regs.set(rd, regs.get(rs).wrapping_add(imm.value())),
                Inst::Sub(rd, rs1, rs2) => regs.set(rd, regs.get(rs1).wrapping_sub(regs.get(rs2))),
                Inst::Neg(rd, rs) => regs.set(rd, 0i32.wrapping_sub(regs.get(rs))),
                Inst::Mul(rd, rs1, rs2) => regs.set(rd, regs.get(rs1).wrapping_mul(regs.get(rs2))),
                Inst::Mulh(rd, rs1, rs2) => {
                    let result = (regs.get(rs1) as i64 * regs.get(rs2) as i64) >> 32;
                    regs.set(rd, result as i32);
                }
                Inst::Div(rd, rs1, rs2) => {
                    regs.set(rd, regs.get(rs1).checked_div(regs.get(rs2)).unwrap_or(-1))
                }
                Inst::Divu(rd, rs1, rs2) => {
                    let result = (regs.get(rs1) as u32).checked_div(regs.get(rs2) as u32);
                    regs.set(rd, result.map_or(-1, |r| r as i32))
                }
                Inst::Rem(rd, rs1, rs2) => {
                    regs.set(rd, regs.get(rs1).checked_rem(regs.get(rs2)).unwrap_or(-1))
                }
                Inst::Remu(rd, rs1, rs2) => {
                    let result = (regs.get(rs1) as u32).checked_rem(regs.get(rs2) as u32);
                    regs.set(rd, result.map_or(-1, |r| r as i32))
                }
                Inst::Not(rd, rs) => regs.set(rd, !regs.get(rs)),
                Inst::And(rd, rs1, rs2) => regs.set(rd, regs.get(rs1) & regs.get(rs2)),
                Inst::Andi(rd, rs, imm) => regs.set(rd, regs.get(rs) & imm.value()),
                Inst::Or(rd, rs1, rs2) => regs.set(rd, regs.get(rs1) | regs.get(rs2)),
                Inst::Ori(rd, rs, imm) => regs.set(rd, regs.get(rs) | imm.value()),
                Inst::Xor(rd, rs1, rs2) => regs.set(rd, regs.get(rs1) ^ regs.get(rs2)),
                Inst::Xori(rd, rs, imm) => regs.set(rd, regs.get(rs) ^ imm.value()),
                Inst::Sll(rd, rs1, rs2) => regs.set(rd, regs.get(rs1) << regs.get(rs2)),
                Inst::Slli(rd, rs, imm) => regs.set(rd, regs.get(rs) << imm.value()),
                Inst::Srl(rd, rs1, rs2) => {
                    regs.set(rd, ((regs.get(rs1) as u32) >> regs.get(rs2)) as i32)
                }
                Inst::Srli(rd, rs, imm) => {
                    regs.set(rd, ((regs.get(rs) as u32) >> imm.value()) as i32)
                }
                Inst::Sra(rd, rs1, rs2) => regs.set(rd, regs.get(rs1) >> regs.get(rs2)),
                Inst::Srai(rd, rs, imm) => regs.set(rd, regs.get(rs) >> imm.value()),
                Inst::Slt(rd, rs1, rs2) => regs.set(rd, (regs.get(rs1) < regs.get(rs2)) as i32),
                Inst::Slti(rd, rs, imm) => regs.set(rd, (regs.get(rs) < imm.value()) as i32),
                Inst::Sltu(rd, rs1, rs2) => {
                    regs.set(rd, ((regs.get(rs1) as u32) < regs.get(rs2) as u32) as i32)
                }
                Inst::Sltiu(rd, rs, imm) => {
                    regs.set(rd, ((regs.get(rs) as u32) < imm.value() as u32) as i32)
                }
                Inst::Seqz(rd, rs) => regs.set(rd, (regs.get(rs) == 0) as i32),
                Inst::Snez(rd, rs) => regs.set(rd, (regs.get(rs) != 0) as i32),
                Inst::Sltz(rd, rs) => regs.set(rd, (regs.get(rs) < 0) as i32),
                Inst::Sgtz(rd, rs) => regs.set(rd, (regs.get(rs) > 0) as i32),
                Inst::Lb(rd, imm, rs) => {
                    let addr = regs.get(rs) + imm.value();
                    regs.set(rd, mem.read::<1>(addr)[0] as i8 as i32);
                }
                Inst::Lbu(rd, imm, rs) => {
                    let addr = regs.get(rs) + imm.value();
                    regs.set(rd, mem.read::<1>(addr)[0] as i32);
                }
                Inst::Lh(rd, imm, rs) => {
                    let addr = regs.get(rs) + imm.value();
                    regs.set(rd, i16::from_le_bytes(mem.read::<2>(addr)) as i32);
                }
                Inst::Lhu(rd, imm, rs) => {
                    let addr = regs.get(rs) + imm.value();
                    regs.set(rd, u16::from_le_bytes(mem.read::<2>(addr)) as i32);
                }
                Inst::Lw(rd, imm, rs) => {
                    let addr = regs.get(rs) + imm.value();
                    regs.set(rd, i32::from_le_bytes(mem.read::<4>(addr)));
                }
                Inst::Sb(rs2, imm, rs1) => {
                    let addr = regs.get(rs1) + imm.value();
                    mem.write::<1>(addr, [regs.get(rs2) as u8]);
                }
                Inst::Sh(rs2, imm, rs1) => {
                    let addr = regs.get(rs1) + imm.value();
                    mem.write::<2>(addr, (regs.get(rs2) as i16).to_le_bytes());
                }
                Inst::Sw(rs2, imm, rs1) => {
                    let addr = regs.get(rs1) + imm.value();
                    mem.write::<4>(addr, regs.get(rs2).to_le_bytes());
                }
                Inst::Nop => (),
                Inst::Ret => return,
            }
        }
    }
}

/// Erase temporary registers.
pub fn erase_temp_regs(regs: &mut Regs) {
    const TEMP_REGS: RegSet = RegSet::from_bitset(make_reg_set!(
        A1, A2, A3, A4, A5, A6, A7, T0, T1, T2, T3, T4, T5, T6
    ));
    for reg in TEMP_REGS.iter() {
        regs.set(reg, 0);
    }
}
