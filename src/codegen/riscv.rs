use std::fmt::Display;

use super::imm::{i12, i20};

/// RISC-V register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum RegId {
    X0, // zero
    RA, // return address
    SP, // stack pointer
    GP, // global pointer
    TP, // thread pointer
    A0, // arguments / return values
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
    Slti(RegId, RegId, i20),
    /// Set if less than immediate unsigned.
    Sltiu(RegId, RegId, i20),
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
            Inst::Nop => write!(f, "nop"),
        }
    }
}

/// RISC-V basic block.
#[derive(Debug, Clone)]
pub struct Block {
    pub label: String,
    pub instructions: Vec<Inst>,
}

impl Block {
    /// Create a new basic block.
    pub fn new(label: String) -> Self {
        Self {
            label,
            instructions: Vec::new(),
        }
    }

    /// Add an instruction to the basic block.
    pub fn push(&mut self, inst: Inst) {
        self.instructions.push(inst);
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for inst in &self.instructions {
            writeln!(f, "    {}", inst)?;
        }
        Ok(())
    }
}

/// RISC-V function.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub blocks: Vec<Block>,
}

impl Function {
    /// Create a new function.
    pub fn new(name: String) -> Self {
        Self {
            name,
            blocks: Vec::new(),
        }
    }

    /// Add a basic block to the function.
    pub fn push(&mut self, block: Block) {
        self.blocks.push(block);
    }

    /// Number of basic blocks in the function.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Create a new block with a unique label.
    pub fn new_block(&self) -> Block {
        let label = if self.len() > 0 {
            format!("{}{}", self.name, self.len())
        } else {
            self.name.clone()
        };
        Block::new(label)
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "    .text")?;
        writeln!(f, "    .globl {}", self.name)?;
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

/// RISC-V program.
#[derive(Debug, Clone)]
pub struct Program {
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

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for function in &self.functions {
            write!(f, "{}", function)?;
        }
        Ok(())
    }
}