//! Frame size analysis.

use miette::Result;

use crate::{
    analysis::{
        call::FunctionCalls, error::AnalysisError, localvar::LocalVars, register::RegAlloc,
    },
    ast::Spanned,
    codegen::{imm::i12, riscv::FrameSize},
    irgen::metadata::FunctionMetadata,
};

/// Frame size analysis.
pub struct Frame {
    /// Frame size.
    pub size: FrameSize,
    /// Total frame size.
    pub total: i12,
}

impl Frame {
    /// Analyse frame size.
    pub fn analyze(
        localvar: &LocalVars,
        reg_alloc: &RegAlloc,
        function_call: &FunctionCalls,
        metadata: &FunctionMetadata,
    ) -> Result<Self> {
        let ra = !function_call.is_leaf;
        let local = localvar.frame_size;
        let spilled = reg_alloc.frame_size;
        let saved = function_call.save_frame_size;
        let arg = function_call.arg_frame_size;
        let size = FrameSize::new(ra, local, spilled, saved, arg);

        let total = size.total();
        let total = i12::try_from(total).map_err(|_| AnalysisError::TooManyLocals {
            span: metadata.name.span().into(),
        })?;
        Ok(Self { size, total })
    }
}
