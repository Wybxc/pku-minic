//! Local variable analysis.

use std::collections::HashMap;

use koopa::ir::{FunctionData, Value, ValueKind};
use miette::Result;

use crate::{
    analysis::error::AnalysisError, ast::Spanned, codegen::imm::i12,
    irgen::metadata::FunctionMetadata,
};

/// Local variables.
pub struct LocalVars {
    /// Map from variable to its storage.
    pub map: HashMap<Value, i12>,
    /// Frame size.
    pub frame_size: i12,
}

impl LocalVars {
    /// Analyse local variables.
    pub fn analyze(func: &FunctionData, metadata: &FunctionMetadata) -> Result<Self> {
        let mut map = HashMap::new();
        let mut frame_size = 0;

        let dfg = func.dfg();
        let bbs = func.layout().bbs();
        for (_, node) in bbs.iter() {
            for &inst in node.insts().keys() {
                if let ValueKind::Alloc(_) = dfg.value(inst).kind() {
                    let size = 4; // i32
                    let offset =
                        i12::try_from(frame_size).map_err(|_| AnalysisError::TooManyLocals {
                            span: metadata.name.span().into(),
                        })?;
                    frame_size += size;
                    map.insert(inst, offset);
                }
            }
        }

        let frame_size = i12::try_from(frame_size).map_err(|_| AnalysisError::TooManyLocals {
            span: metadata.name.span().into(),
        })?;
        Ok(Self { map, frame_size })
    }
}
