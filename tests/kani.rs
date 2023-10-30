#[cfg(kani)]
mod verification {
    #[kani::proof]
    pub fn has_kani() {
        assert!(true);
    }
}
