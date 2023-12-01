//! Program analysis.

use std::collections::HashMap;
use std::rc::Rc;

use crate::analysis::cfg::ControlFlowGraph;
use crate::analysis::liveliness::Liveliness;
use koopa::ir::{Function, Program};

pub mod cfg;
pub mod liveliness;

/// Program analyser.
pub struct Analyzer<'a> {
    program: &'a Program,
    cfg_cache: HashMap<Function, Rc<ControlFlowGraph>>,
    liveliness_cache: HashMap<Function, Rc<Liveliness>>,
}

impl<'a> Analyzer<'a> {
    /// Create a new program analyser.
    pub fn new(program: &'a Program) -> Self {
        Self {
            program,
            cfg_cache: HashMap::new(),
            liveliness_cache: HashMap::new(),
        }
    }

    /// Get the program being analysed.
    pub fn program(&self) -> &Program {
        self.program
    }

    /// Analyse the control flow graph of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze_cfg(&mut self, func: Function) -> Rc<ControlFlowGraph> {
        self.cfg_cache
            .entry(func)
            .or_insert_with(|| {
                let cfg = ControlFlowGraph::analyze(self.program.func(func));
                Rc::new(cfg)
            })
            .clone()
    }

    /// Analyse variable liveliness of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze_liveliness(&mut self, func: Function) -> Rc<Liveliness> {
        match self.liveliness_cache.get(&func) {
            Some(l) => l.clone(),
            None => {
                let cfg = self.analyze_cfg(func);
                let liveliness = Liveliness::analysis(cfg.as_ref(), self.program.func(func));
                let liveliness = Rc::new(liveliness);
                self.liveliness_cache.insert(func, liveliness.clone());
                liveliness
            }
        }
    }
}
