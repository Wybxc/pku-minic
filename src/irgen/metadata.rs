//! Metadata for the IR.
//!
//! This module contains metadata for the IR. This is used to store syntactic
//! information about the IR, which is not stored in the IR itself, for example,
//! the name of a function.

use std::collections::HashMap;

use koopa::ir::{Function, FunctionData};

use crate::{
    ast::Span,
    irgen::cfg::{self, ControlFlowGraph, Dominators},
};

/// Program metadata.
pub struct ProgramMetadata {
    /// Metadata for functions.
    pub functions: HashMap<Function, FunctionMetadata>,
}

impl ProgramMetadata {
    /// Create a new program metadata.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}

impl Default for ProgramMetadata {
    fn default() -> Self { Self::new() }
}

/// Function metadata.
pub struct FunctionMetadata {
    /// Name of the function.
    pub name: Span<String>,
    /// Control flow graph.
    pub cfg: ControlFlowGraph,
    /// Dominator tree.
    pub dominators: Dominators,
}

impl FunctionMetadata {
    /// Create a new function metadata.
    pub fn new(name: Span<String>, function: &FunctionData) -> Self {
        let (cfg, dominators) = cfg::analyze(function);
        Self {
            name,
            cfg,
            dominators,
        }
    }
}
