use std::collections::HashMap;

use koopa::ir::Function;

use crate::ast::Span;

/// Program metadata.
pub struct ProgramMetadata {
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
    fn default() -> Self {
        Self::new()
    }
}

/// Function metadata.
pub struct FunctionMetadata {
    /// Name of the function.
    pub name: Span<String>,
}

impl FunctionMetadata {
    /// Create a new function metadata.
    pub fn new(name: Span<String>) -> Self {
        Self { name }
    }
}
