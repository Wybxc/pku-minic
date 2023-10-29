//! Short immediate values.
//!
//! RISC-V has a number of instructions that take immediate values. These
//! immediate values are often limited in size, such as 12-bit or 20-bit
//! immediates. This module provides a type for representing these immediate
//! values.

use std::{fmt::Display, ops::Neg};

/// N-bit immediate value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Imm<const N: usize>(i32);

impl<const N: usize> Imm<N> {
    /// Create a new immediate value.
    #[inline]
    pub fn new(value: i32) -> Result<Self, &'static str> {
        if !(Self::MIN..Self::MAX).contains(&value) {
            Err("imm out of range")
        } else {
            Ok(Imm(value))
        }
    }

    /// Get the value of the immediate.
    #[inline]
    pub fn value(self) -> i32 {
        self.0
    }

    /// Minimum value of the immediate.
    pub const MIN: i32 = -(1 << (N - 1));

    /// Maximum value of the immediate.
    pub const MAX: i32 = (1 << (N - 1)) - 1;

    /// Negate the immediate.
    ///
    /// # Panics
    /// Panics if the negation overflows.
    #[inline]
    pub fn neg(self) -> Self {
        if self.0 == Self::MIN {
            panic!("negation overflow");
        }
        Self(-self.0)
    }
}

impl Neg for Imm<12> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

impl<const N: usize> PartialEq<i32> for Imm<N> {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl<const N: usize> PartialOrd<i32> for Imm<N> {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl<const N: usize> TryFrom<i32> for Imm<N> {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::new(value)
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
