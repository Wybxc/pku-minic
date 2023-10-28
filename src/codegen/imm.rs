//! Short immediate values.
//!
//! RISC-V has a number of instructions that take immediate values. These
//! immediate values are often limited in size, such as 12-bit or 20-bit
//! immediates. This module provides a type for representing these immediate
//! values.

use std::fmt::Display;

/// N-bit immediate value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Imm<const N: usize>(i32);

impl<const N: usize> Imm<N> {
    /// Create a new immediate value.
    pub fn new(value: i32) -> Result<Self, &'static str> {
        Self::try_from(value)
    }

    /// Get the value of the immediate.
    pub fn value(self) -> i32 {
        self.0
    }
}

impl<const N: usize> TryFrom<i32> for Imm<N> {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if !(-(1 << (N - 1))..((1 << (N - 1)) - 1)).contains(&value) {
            Err("imm out of range")
        } else {
            Ok(Imm(value))
        }
    }
}

impl<const N: usize> From<Imm<N>> for i32 {
    fn from(value: Imm<N>) -> Self {
        value.0
    }
}

impl<const N: usize> Display for Imm<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 12-bit immediate.
#[allow(non_camel_case_types)]
pub type i12 = Imm<12>;

/// 20-bit immediate.
#[allow(non_camel_case_types)]
pub type i20 = Imm<20>;
