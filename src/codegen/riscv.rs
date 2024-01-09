//! RISC-V specification.
//!
//! This module contains types for representing RISC-V instructions and
//! programs.

use std::{collections::HashMap, fmt::Display};

use key_node_list::{impl_node, KeyNodeList};

use super::imm::i12;
use crate::utils;

/// RISC-V register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(dead_code, missing_docs)]
pub enum RegId {
    X0 = 0, // zero
    RA,     // return address
    SP,     // stack pointer
    GP,     // global pointer
    TP,     // thread pointer
    A0,     // arguments / return values
    A1,
    A2, // arguments
    A3,
    A4,
    A5,
    A6,
    A7,
    T0, // temporaries
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    S0, // saved register / frame pointer
    S1, // saved registers
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
}

impl Display for RegId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegId::X0 => write!(f, "x0"),
            RegId::RA => write!(f, "ra"),
            RegId::SP => write!(f, "sp"),
            RegId::GP => write!(f, "gp"),
            RegId::TP => write!(f, "tp"),
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
            RegId::S0 => write!(f, "s0"),
            RegId::S1 => write!(f, "s1"),
            RegId::S2 => write!(f, "s2"),
            RegId::S3 => write!(f, "s3"),
            RegId::S4 => write!(f, "s4"),
            RegId::S5 => write!(f, "s5"),
            RegId::S6 => write!(f, "s6"),
            RegId::S7 => write!(f, "s7"),
            RegId::S8 => write!(f, "s8"),
            RegId::S9 => write!(f, "s9"),
            RegId::S10 => write!(f, "s10"),
            RegId::S11 => write!(f, "s11"),
        }
    }
}

/// A set of registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegSet<const MASK: u32 = { u32::MAX }> {
    bitset: u32,
}

macro_rules! make_reg_set {
    ($($reg:ident),+) => {
        {
            let mut set = 0u32;
            $(set |= 1 << RegId::$reg as u32;)*
            set
        }
    };
}

pub(crate) use make_reg_set;

impl<const MASK: u32> RegSet<MASK> {
    /// Create a new empty register set.
    pub const fn new() -> Self {
        Self { bitset: 0 }
    }

    /// Create a register set from a bitset.
    pub const fn from_bitset(bitset: u32) -> Self {
        Self { bitset }
    }

    /// Create a full register set.
    pub const fn full() -> Self {
        Self { bitset: MASK }
    }

    /// Insert a register into the set.
    pub fn insert(&mut self, reg: RegId) {
        self.bitset |= 1 << reg as u32;
        self.bitset &= MASK;
    }

    /// Remove a register from the set.
    pub fn remove(&mut self, reg: RegId) {
        self.bitset &= !(1 << reg as u32);
    }

    /// Check if the set contains a register.
    pub fn contains(&self, reg: RegId) -> bool {
        self.bitset & (1 << reg as u32) != 0
    }

    /// Get an iterator over the registers in the set.
    pub fn iter(&self) -> impl Iterator<Item = RegId> + '_ {
        (0..32u8).filter_map(move |i| {
            if self.bitset & (1 << i) != 0 {
                // SAFETY: `i` is in the range of `RegId`.
                Some(unsafe { std::mem::transmute(i) })
            } else {
                None
            }
        })
    }
}

/// RISC-V instruction (RV32I, RV32M).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum Inst {
    // Data transfer
    // --------------------
    /// Load immediate.
    Li(RegId, i32),
    /// Move.
    Mv(RegId, RegId),

    // Arithmetic
    // --------------------
    /// Add.
    Add(RegId, RegId, RegId),
    /// Add immediate.
    Addi(RegId, RegId, i12),
    /// Subtract.
    Sub(RegId, RegId, RegId),
    /// Negate.
    Neg(RegId, RegId),
    /// Multiply.
    Mul(RegId, RegId, RegId),
    /// Multiply high.
    Mulh(RegId, RegId, RegId),
    /// Divide.
    Div(RegId, RegId, RegId),
    /// Divide unsigned.
    Divu(RegId, RegId, RegId),
    /// Remainder.
    Rem(RegId, RegId, RegId),
    /// Remainder unsigned.
    Remu(RegId, RegId, RegId),

    // Bitwise
    // --------------------
    /// Not.
    Not(RegId, RegId),
    /// And.
    And(RegId, RegId, RegId),
    /// And immediate.
    Andi(RegId, RegId, i12),
    /// Or.
    Or(RegId, RegId, RegId),
    /// Or immediate.
    Ori(RegId, RegId, i12),
    /// Xor.
    Xor(RegId, RegId, RegId),
    /// Xor immediate.
    Xori(RegId, RegId, i12),

    // Shift
    // --------------------
    /// Shift left logical.
    Sll(RegId, RegId, RegId),
    /// Shift left logical immediate.
    Slli(RegId, RegId, i12),
    /// Shift right arithmetic.
    Sra(RegId, RegId, RegId),
    /// Shift right arithmetic immediate.
    Srai(RegId, RegId, i12),
    /// Shift right logical.
    Srl(RegId, RegId, RegId),
    /// Shift right logical immediate.
    Srli(RegId, RegId, i12),

    // Compare
    // --------------------
    /// Set if less than.
    Slt(RegId, RegId, RegId),
    /// Set if less than unsigned.
    Sltu(RegId, RegId, RegId),
    /// Set if less than immediate.
    Slti(RegId, RegId, i12),
    /// Set if less than immediate unsigned.
    Sltiu(RegId, RegId, i12),
    /// Set if equal to zero.
    Seqz(RegId, RegId),
    /// Set if not equal to zero.
    Snez(RegId, RegId),
    /// Set if less than zero.
    Sltz(RegId, RegId),
    /// Set if greater than zero.
    Sgtz(RegId, RegId),

    // Memory
    // --------------------
    /// Load byte.
    Lb(RegId, i12, RegId),
    /// Load half word.
    Lh(RegId, i12, RegId),
    /// Load word.
    Lw(RegId, i12, RegId),
    /// Store byte.
    Sb(RegId, i12, RegId),
    /// Store half word.
    Sh(RegId, i12, RegId),
    /// Store word.
    Sw(RegId, i12, RegId),
    /// Load byte unsigned.
    Lbu(RegId, i12, RegId),
    /// Load half word unsigned.
    Lhu(RegId, i12, RegId),

    // Control flow
    // --------------------
    /// Return.
    Ret,
    /// Jump.
    J(BlockId),
    /// Branch if equal.
    Beq(RegId, RegId, BlockId),
    /// Branch if equal to zero.
    Beqz(RegId, BlockId),
    /// Branch if greater than or equal.
    Bge(RegId, RegId, BlockId),
    /// Branch if greater than or equal unsigned.
    Bgeu(RegId, RegId, BlockId),
    /// Branch if greater than or equal to zero.
    Bgez(RegId, BlockId),
    /// Branch if greater than.
    Bgt(RegId, RegId, BlockId),
    /// Branch if greater than unsigned.
    Bgtu(RegId, RegId, BlockId),
    /// Branch if greater than zero.
    Bgtz(RegId, BlockId),
    /// Branch if less than or equal.
    Ble(RegId, RegId, BlockId),
    /// Branch if less than or equal unsigned.
    Bleu(RegId, RegId, BlockId),
    /// Branch if less than or equal to zero.
    Blez(RegId, BlockId),
    /// Branch if less than.
    Blt(RegId, RegId, BlockId),
    /// Branch if less than unsigned.
    Bltu(RegId, RegId, BlockId),
    /// Branch if less than zero.
    Bltz(RegId, BlockId),
    /// Branch if not equal.
    Bne(RegId, RegId, BlockId),
    /// Branch if not equal to zero.
    Bnez(RegId, BlockId),

    // Misc
    // --------------------
    /// Nope.
    Nop,
}

impl Display for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inst::Li(rd, imm) => write!(f, "li {}, {}", rd, imm),
            Inst::Mv(rd, rs) => write!(f, "mv {}, {}", rd, rs),
            Inst::Add(rd, rs1, rs2) => write!(f, "add {}, {}, {}", rd, rs1, rs2),
            Inst::Addi(rd, rs, imm) => write!(f, "addi {}, {}, {}", rd, rs, imm),
            Inst::Sub(rd, rs1, rs2) => write!(f, "sub {}, {}, {}", rd, rs1, rs2),
            Inst::Neg(rd, rs) => write!(f, "neg {}, {}", rd, rs),
            Inst::Mul(rd, rs1, rs2) => write!(f, "mul {}, {}, {}", rd, rs1, rs2),
            Inst::Mulh(rd, rs1, rs2) => write!(f, "mulh {}, {}, {}", rd, rs1, rs2),
            Inst::Div(rd, rs1, rs2) => write!(f, "div {}, {}, {}", rd, rs1, rs2),
            Inst::Divu(rd, rs1, rs2) => write!(f, "divu {}, {}, {}", rd, rs1, rs2),
            Inst::Rem(rd, rs1, rs2) => write!(f, "rem {}, {}, {}", rd, rs1, rs2),
            Inst::Remu(rd, rs1, rs2) => write!(f, "remu {}, {}, {}", rd, rs1, rs2),
            Inst::Not(rd, rs) => write!(f, "not {}, {}", rd, rs),
            Inst::And(rd, rs1, rs2) => write!(f, "and {}, {}, {}", rd, rs1, rs2),
            Inst::Andi(rd, rs, imm) => write!(f, "andi {}, {}, {}", rd, rs, imm),
            Inst::Or(rd, rs1, rs2) => write!(f, "or {}, {}, {}", rd, rs1, rs2),
            Inst::Ori(rd, rs, imm) => write!(f, "ori {}, {}, {}", rd, rs, imm),
            Inst::Xor(rd, rs1, rs2) => write!(f, "xor {}, {}, {}", rd, rs1, rs2),
            Inst::Xori(rd, rs, imm) => write!(f, "xori {}, {}, {}", rd, rs, imm),
            Inst::Sll(rd, rs1, rs2) => write!(f, "sll {}, {}, {}", rd, rs1, rs2),
            Inst::Slli(rd, rs, imm) => write!(f, "slli {}, {}, {}", rd, rs, imm),
            Inst::Sra(rd, rs1, rs2) => write!(f, "sra {}, {}, {}", rd, rs1, rs2),
            Inst::Srai(rd, rs, imm) => write!(f, "srai {}, {}, {}", rd, rs, imm),
            Inst::Srl(rd, rs1, rs2) => write!(f, "srl {}, {}, {}", rd, rs1, rs2),
            Inst::Srli(rd, rs, imm) => write!(f, "srli {}, {}, {}", rd, rs, imm),
            Inst::Slt(rd, rs1, rs2) => write!(f, "slt {}, {}, {}", rd, rs1, rs2),
            Inst::Sltu(rd, rs1, rs2) => write!(f, "sltu {}, {}, {}", rd, rs1, rs2),
            Inst::Slti(rd, rs, imm) => write!(f, "slti {}, {}, {}", rd, rs, imm),
            Inst::Sltiu(rd, rs, imm) => write!(f, "sltiu {}, {}, {}", rd, rs, imm),
            Inst::Seqz(rd, rs) => write!(f, "seqz {}, {}", rd, rs),
            Inst::Snez(rd, rs) => write!(f, "snez {}, {}", rd, rs),
            Inst::Sltz(rd, rs) => write!(f, "sltz {}, {}", rd, rs),
            Inst::Sgtz(rd, rs) => write!(f, "sgtz {}, {}", rd, rs),
            Inst::Lb(rd, imm, rs) => write!(f, "lb {}, {}({})", rd, imm, rs),
            Inst::Lh(rd, imm, rs) => write!(f, "lh {}, {}({})", rd, imm, rs),
            Inst::Lw(rd, imm, rs) => write!(f, "lw {}, {}({})", rd, imm, rs),
            Inst::Sb(rd, imm, rs) => write!(f, "sb {}, {}({})", rd, imm, rs),
            Inst::Sh(rd, imm, rs) => write!(f, "sh {}, {}({})", rd, imm, rs),
            Inst::Sw(rd, imm, rs) => write!(f, "sw {}, {}({})", rd, imm, rs),
            Inst::Lbu(rd, imm, rs) => write!(f, "lbu {}, {}({})", rd, imm, rs),
            Inst::Lhu(rd, imm, rs) => write!(f, "lhu {}, {}({})", rd, imm, rs),
            Inst::Ret => write!(f, "ret"),
            Inst::J(label) => write!(f, "j {}", label),
            Inst::Beq(rs1, rs2, label) => write!(f, "beq {}, {}, {}", rs1, rs2, label),
            Inst::Beqz(rs, label) => write!(f, "beqz {}, {}", rs, label),
            Inst::Bge(rs1, rs2, label) => write!(f, "bge {}, {}, {}", rs1, rs2, label),
            Inst::Bgeu(rs1, rs2, label) => write!(f, "bgeu {}, {}, {}", rs1, rs2, label),
            Inst::Bgez(rs, label) => write!(f, "bgez {}, {}", rs, label),
            Inst::Bgt(rs1, rs2, label) => write!(f, "bgt {}, {}, {}", rs1, rs2, label),
            Inst::Bgtu(rs1, rs2, label) => write!(f, "bgtu {}, {}, {}", rs1, rs2, label),
            Inst::Bgtz(rs, label) => write!(f, "bgtz {}, {}", rs, label),
            Inst::Ble(rs1, rs2, label) => write!(f, "ble {}, {}, {}", rs1, rs2, label),
            Inst::Bleu(rs1, rs2, label) => write!(f, "bleu {}, {}, {}", rs1, rs2, label),
            Inst::Blez(rs, label) => write!(f, "blez {}, {}", rs, label),
            Inst::Blt(rs1, rs2, label) => write!(f, "blt {}, {}, {}", rs1, rs2, label),
            Inst::Bltu(rs1, rs2, label) => write!(f, "bltu {}, {}, {}", rs1, rs2, label),
            Inst::Bltz(rs, label) => write!(f, "bltz {}, {}", rs, label),
            Inst::Bne(rs1, rs2, label) => write!(f, "bne {}, {}, {}", rs1, rs2, label),
            Inst::Bnez(rs, label) => write!(f, "bnez {}, {}", rs, label),
            Inst::Nop => write!(f, "nop"),
        }
    }
}

impl Inst {
    /// Get the destination register of the instruction.
    pub fn dest(self) -> Option<RegId> {
        match self {
            Inst::Li(rd, _) => Some(rd),
            Inst::Mv(rd, _) => Some(rd),
            Inst::Add(rd, _, _) => Some(rd),
            Inst::Addi(rd, _, _) => Some(rd),
            Inst::Sub(rd, _, _) => Some(rd),
            Inst::Neg(rd, _) => Some(rd),
            Inst::Mul(rd, _, _) => Some(rd),
            Inst::Mulh(rd, _, _) => Some(rd),
            Inst::Div(rd, _, _) => Some(rd),
            Inst::Divu(rd, _, _) => Some(rd),
            Inst::Rem(rd, _, _) => Some(rd),
            Inst::Remu(rd, _, _) => Some(rd),
            Inst::Not(rd, _) => Some(rd),
            Inst::And(rd, _, _) => Some(rd),
            Inst::Andi(rd, _, _) => Some(rd),
            Inst::Or(rd, _, _) => Some(rd),
            Inst::Ori(rd, _, _) => Some(rd),
            Inst::Xor(rd, _, _) => Some(rd),
            Inst::Xori(rd, _, _) => Some(rd),
            Inst::Sll(rd, _, _) => Some(rd),
            Inst::Slli(rd, _, _) => Some(rd),
            Inst::Sra(rd, _, _) => Some(rd),
            Inst::Srai(rd, _, _) => Some(rd),
            Inst::Srl(rd, _, _) => Some(rd),
            Inst::Srli(rd, _, _) => Some(rd),
            Inst::Slt(rd, _, _) => Some(rd),
            Inst::Sltu(rd, _, _) => Some(rd),
            Inst::Slti(rd, _, _) => Some(rd),
            Inst::Sltiu(rd, _, _) => Some(rd),
            Inst::Seqz(rd, _) => Some(rd),
            Inst::Snez(rd, _) => Some(rd),
            Inst::Sltz(rd, _) => Some(rd),
            Inst::Sgtz(rd, _) => Some(rd),
            Inst::Lb(rd, _, _) => Some(rd),
            Inst::Lh(rd, _, _) => Some(rd),
            Inst::Lw(rd, _, _) => Some(rd),
            Inst::Sb(_, _, _) => None,
            Inst::Sh(_, _, _) => None,
            Inst::Sw(_, _, _) => None,
            Inst::Lbu(rd, _, _) => Some(rd),
            Inst::Lhu(rd, _, _) => Some(rd),
            Inst::Ret => None,
            Inst::J(_) => None,
            Inst::Beq(_, _, _) => None,
            Inst::Beqz(_, _) => None,
            Inst::Bge(_, _, _) => None,
            Inst::Bgeu(_, _, _) => None,
            Inst::Bgez(_, _) => None,
            Inst::Bgt(_, _, _) => None,
            Inst::Bgtu(_, _, _) => None,
            Inst::Bgtz(_, _) => None,
            Inst::Ble(_, _, _) => None,
            Inst::Bleu(_, _, _) => None,
            Inst::Blez(_, _) => None,
            Inst::Blt(_, _, _) => None,
            Inst::Bltu(_, _, _) => None,
            Inst::Bltz(_, _) => None,
            Inst::Bne(_, _, _) => None,
            Inst::Bnez(_, _) => None,
            Inst::Nop => None,
        }
    }

    /// Get the source registers of the instruction.
    pub fn source(mut self) -> [Option<RegId>; 2] {
        self.source_mut().map(|r| r.copied())
    }

    /// Get the mutable source registers of the instruction.
    pub fn source_mut(&mut self) -> [Option<&mut RegId>; 2] {
        match self {
            Inst::Li(_, _) => [None, None],
            Inst::Mv(_, rs) => [None, Some(rs)],
            Inst::Add(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Addi(_, rs, _) => [Some(rs), None],
            Inst::Sub(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Neg(_, rs) => [None, Some(rs)],
            Inst::Mul(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Mulh(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Div(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Divu(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Rem(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Remu(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Not(_, rs) => [None, Some(rs)],
            Inst::And(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Andi(_, rs, _) => [Some(rs), None],
            Inst::Or(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Ori(_, rs, _) => [Some(rs), None],
            Inst::Xor(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Xori(_, rs, _) => [Some(rs), None],
            Inst::Sll(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Slli(_, rs, _) => [Some(rs), None],
            Inst::Sra(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Srai(_, rs, _) => [Some(rs), None],
            Inst::Srl(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Srli(_, rs, _) => [Some(rs), None],
            Inst::Slt(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Sltu(_, rs1, rs2) => [Some(rs1), Some(rs2)],
            Inst::Slti(_, rs, _) => [Some(rs), None],
            Inst::Sltiu(_, rs, _) => [Some(rs), None],
            Inst::Seqz(_, rs) => [None, Some(rs)],
            Inst::Snez(_, rs) => [None, Some(rs)],
            Inst::Sltz(_, rs) => [None, Some(rs)],
            Inst::Sgtz(_, rs) => [None, Some(rs)],
            Inst::Lb(_, _, rs) => [None, Some(rs)],
            Inst::Lh(_, _, rs) => [None, Some(rs)],
            Inst::Lw(_, _, rs) => [None, Some(rs)],
            Inst::Sb(rs1, _, rs2) => [Some(rs1), Some(rs2)],
            Inst::Sh(rs1, _, rs2) => [Some(rs1), Some(rs2)],
            Inst::Sw(rs1, _, rs2) => [Some(rs1), Some(rs2)],
            Inst::Lbu(_, _, rs) => [None, Some(rs)],
            Inst::Lhu(_, _, rs) => [None, Some(rs)],
            Inst::Ret => [None, None],
            Inst::J(_) => [None, None],
            Inst::Beq(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Beqz(rs, _) => [None, Some(rs)],
            Inst::Bge(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bgeu(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bgez(rs, _) => [None, Some(rs)],
            Inst::Bgt(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bgtu(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bgtz(rs, _) => [None, Some(rs)],
            Inst::Ble(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bleu(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Blez(rs, _) => [None, Some(rs)],
            Inst::Blt(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bltu(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bltz(rs, _) => [None, Some(rs)],
            Inst::Bne(rs1, rs2, _) => [Some(rs1), Some(rs2)],
            Inst::Bnez(rs, _) => [None, Some(rs)],
            Inst::Nop => [None, None],
        }
    }
}

utils::declare_u32_id!(InstId);

/// Node in the instruction list.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InstNode {
    pub inst: Inst,
    prev: Option<InstId>,
    next: Option<InstId>,
}

impl_node!(InstNode { Key = InstId, prev = prev, next = next });

impl InstNode {
    /// Create a new instruction node.
    pub fn new(inst: Inst) -> Self {
        Self {
            inst,
            prev: None,
            next: None,
        }
    }
}

type InstList = KeyNodeList<InstId, InstNode, HashMap<InstId, InstNode>>;

/// RISC-V basic block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Instructions in the basic block.
    instructions: InstList,
}

impl Block {
    /// Create a new basic block.
    pub fn new() -> Self {
        Self {
            instructions: InstList::new(),
        }
    }

    /// Add an instruction to the basic block.
    pub fn push(&mut self, inst: Inst) -> InstId {
        let id = InstId::next_id();
        let node = InstNode::new(inst);
        self.instructions.push_back(id, node).unwrap();
        id
    }

    /// Get the instruction with the given id.
    pub fn get(&self, id: InstId) -> Option<Inst> {
        self.instructions.cursor(id).node().map(|node| node.inst)
    }

    /// The number of instructions in the basic block.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if the basic block is empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// The first instruction id in the basic block.
    pub fn front(&self) -> Option<InstId> {
        self.instructions.front_key().copied()
    }

    /// The last instruction id in the basic block.
    pub fn back(&self) -> Option<InstId> {
        self.instructions.back_key().copied()
    }

    /// Provide a cursor with mutable access to the instruction with the given
    /// id.
    pub fn cursor_mut(&mut self, id: InstId) -> InstCursorMut<'_> {
        InstCursorMut {
            cursor: self.instructions.cursor_mut(id),
        }
    }

    /// Get an iterator over the instructions in the basic block.
    pub fn insts(&self) -> impl Iterator<Item = (InstId, Inst)> + '_ {
        self.instructions.iter().map(|(&id, node)| (id, node.inst))
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for node in self.instructions.nodes() {
            writeln!(f, "    {}", node.inst)?;
        }
        Ok(())
    }
}

/// Cursor with mutable access to an instruction in a basic block.
pub struct InstCursorMut<'a> {
    cursor: key_node_list::CursorMut<'a, InstId, InstNode, HashMap<InstId, InstNode>>,
}

impl<'a> InstCursorMut<'a> {
    /// Check if the cursor is null.
    pub fn is_null(&self) -> bool {
        self.cursor.is_null()
    }

    /// Get the instruction id.
    pub fn id(&self) -> Option<InstId> {
        self.cursor.key().cloned()
    }

    /// Get the instruction.
    pub fn inst(&self) -> Option<Inst> {
        self.cursor.node().map(|node| node.inst)
    }

    /// Set the instruction. If the cursor is null, this function does nothing.
    pub fn set_inst(&mut self, inst: Inst) {
        if let Some(node) = self.cursor.node_mut() {
            node.inst = inst;
        }
    }

    /// Move the cursor to the previous instruction.
    pub fn prev(&mut self) {
        self.cursor.move_prev();
    }

    /// Move the cursor to the next instruction.
    pub fn next(&mut self) {
        self.cursor.move_next();
    }

    /// Get an iterator over the following instructions, not including the
    /// current instruction.
    pub fn followings(&self) -> impl Iterator<Item = (InstId, Inst)> + '_ {
        let mut cursor = self.cursor.as_cursor();
        cursor.move_next();
        FollowInsts { cursor }
    }

    /// Insert a new instruction before the current instruction.
    pub fn insert_before(&mut self, inst: Inst) -> InstId {
        let id = InstId::next_id();
        let node = InstNode::new(inst);
        self.cursor.insert_before(id, node).unwrap();
        id
    }

    /// Insert a new instruction after the current instruction.
    pub fn insert_after(&mut self, inst: Inst) -> InstId {
        let id = InstId::next_id();
        let node = InstNode::new(inst);
        self.cursor.insert_after(id, node).unwrap();
        id
    }

    /// Remove the current instruction, and move the cursor to the next
    /// instruction.
    ///
    /// If the cursor is null, this function does nothing.
    pub fn remove_and_next(&mut self) {
        self.cursor.remove_current();
    }

    /// Remove the current instruction, and move the cursor to the previous
    /// instruction.
    ///
    /// If the cursor is null, this function does nothing.
    pub fn remove_and_prev(&mut self) {
        self.cursor.remove_current();
        if !self.cursor.is_null() {
            self.cursor.move_prev();
        }
    }
}

struct FollowInsts<'a> {
    cursor: key_node_list::Cursor<'a, InstId, InstNode, HashMap<InstId, InstNode>>,
}

impl Iterator for FollowInsts<'_> {
    type Item = (InstId, Inst);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.is_null() {
            None
        } else {
            let id = *self.cursor.key().unwrap();
            let inst = self.cursor.node().unwrap().inst;
            self.cursor.move_next();
            Some((id, inst))
        }
    }
}

utils::declare_u32_id!(BlockId);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".L{}", self.0)
    }
}

/// Node in the instruction list.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockNode {
    pub block: Block,
    prev: Option<BlockId>,
    next: Option<BlockId>,
}

impl_node!(BlockNode { Key = BlockId, prev = prev, next = next });

impl BlockNode {
    /// Create a new instruction node.
    pub fn new(block: Block) -> Self {
        Self {
            block,
            prev: None,
            next: None,
        }
    }
}

type BlockList = KeyNodeList<BlockId, BlockNode, HashMap<BlockId, BlockNode>>;

/// RISC-V function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// Name of the function.
    pub name: String,
    /// Basic blocks in the function.
    blocks: BlockList,
}

impl Function {
    /// Create a new function.
    pub fn new(name: String) -> Self {
        Self {
            name,
            blocks: BlockList::new(),
        }
    }

    /// Add a basic block to the function.
    pub fn push(&mut self, id: BlockId, block: Block) -> BlockId {
        let node = BlockNode::new(block);
        self.blocks.push_back(id, node).unwrap();
        id
    }

    /// Get the basic block with the given id.
    pub fn get(&self, id: BlockId) -> Option<&Block> {
        self.blocks.node(&id).map(|node| &node.block)
    }

    /// Number of basic blocks in the function.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the function is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Iterator over the basic blocks in the function.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &Block)> + '_ {
        self.blocks.iter().map(|(&id, node)| (id, &node.block))
    }

    /// Provide a cursor to the basic block with the given id.
    pub fn cursor(&self, id: BlockId) -> BlockCursor<'_> {
        BlockCursor {
            cursor: self.blocks.cursor(id),
        }
    }

    /// The first basic block id in the function.
    pub fn entry(&self) -> Option<BlockId> {
        self.blocks.front_key().copied()
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "    .text")?;
        writeln!(f, "    .globl {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        for (id, block) in self.iter() {
            writeln!(f, "{}:", id)?;
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

/// Cursor to a basic block in a function.
pub struct BlockCursor<'a> {
    cursor: key_node_list::Cursor<'a, BlockId, BlockNode, HashMap<BlockId, BlockNode>>,
}

impl<'a> BlockCursor<'a> {
    /// Check if the cursor is null.
    pub fn is_null(&self) -> bool {
        self.cursor.is_null()
    }

    /// Get the basic block id.
    pub fn id(&self) -> Option<BlockId> {
        self.cursor.key().copied()
    }

    /// Get the basic block.
    pub fn block(&self) -> Option<&Block> {
        self.cursor.node().map(|node| &node.block)
    }

    /// Move the cursor to the previous basic block.
    pub fn prev(&mut self) {
        self.cursor.move_prev();
    }

    /// Move the cursor to the next basic block.
    pub fn next(&mut self) {
        self.cursor.move_next();
    }
}

/// RISC-V program.
pub struct Program {
    /// Functions in the program.
    pub functions: Vec<Function>,
}

impl Program {
    /// Create a new program.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    /// Add a function to the program.
    pub fn push(&mut self, function: Function) {
        self.functions.push(function);
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for function in &self.functions {
            write!(f, "{}", function)?;
        }
        Ok(())
    }
}
