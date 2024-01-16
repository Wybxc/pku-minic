//! Frame size analysis.

use crate::{
    analysis::{call::FunctionCalls, localvar::LocalVars, register::RegAlloc},
    codegen::riscv::FrameSize,
};

/// Frame size analysis.
pub struct Frame {
    /// Frame size.
    pub size: FrameSize,
    /// Total frame size.
    pub total: i32,
}

impl Frame {
    /// Analyse frame size.
    pub fn analyze(
        localvar: &LocalVars,
        reg_alloc: &RegAlloc,
        function_call: &FunctionCalls,
    ) -> Self {
        let ra = !function_call.is_leaf;
        let local = localvar.frame_size;
        let spilled = reg_alloc.frame_size;
        let saved = function_call.save_frame_size;
        let arg = function_call.arg_frame_size;
        let size = FrameSize::new(ra, local, spilled, saved, arg);

        let total = size.total();
        Self { size, total }
    }
}
