#[must_use]
/// Whether the current basic block is terminated.
pub struct Terminated(bool);

impl From<bool> for Terminated {
    fn from(terminated: bool) -> Self { Terminated(terminated) }
}

impl Terminated {
    /// Create a new `Terminated` with the given value.
    pub fn new(b: bool) -> Self { Self(b) }

    /// Whether the current basic block is terminated.
    pub fn is_terminated(&self) -> bool { self.0 }
}
