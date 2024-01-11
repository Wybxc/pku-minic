use koopa::ir::BasicBlock;

/// Loop boundary.
#[derive(Debug)]
pub struct LoopBoundary {
    /// The entry block of the loop.
    pub entry: BasicBlock,
    /// The block to jump to when the loop condition is false.
    pub exit: BasicBlock,
}

/// Context.
#[derive(Debug, Default)]
pub struct Context {
    loop_bounds: Vec<LoopBoundary>,
}

impl Context {
    /// Create a new context.
    pub fn new() -> Self {
        Self {
            loop_bounds: vec![],
        }
    }

    /// Push a new loop boundary.
    pub fn push_loop_boundary(&mut self, entry: BasicBlock, exit: BasicBlock) {
        self.loop_bounds.push(LoopBoundary { entry, exit });
    }

    /// Pop a loop boundary.
    pub fn pop_loop_boundary(&mut self) -> Option<LoopBoundary> { self.loop_bounds.pop() }

    /// Get the current loop boundary.
    pub fn current_loop_boundary(&self) -> Option<&LoopBoundary> { self.loop_bounds.last() }
}
