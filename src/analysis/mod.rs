//! Program analysis.

use std::{collections::HashMap, rc::Rc};

use koopa::ir::{Function, Program};
use miette::Result;
#[allow(unused_imports)]
use nolog::*;

use crate::{
    analysis::{
        cfg::ControlFlowGraph, dominators::Dominators, liveliness::Liveliness, register::RegAlloc,
    },
    irgen::metadata::ProgramMetadata,
};

pub mod cfg;
pub mod dominators;
pub mod error;
pub mod liveliness;
pub mod register;

/// Program analyser.
pub struct Analyzer<'a> {
    program: &'a Program,
    metadata: &'a ProgramMetadata,
    cfg_cache: HashMap<Function, Rc<ControlFlowGraph>>,
    dominators_cache: HashMap<Function, Rc<Dominators>>,
    liveliness_cache: HashMap<Function, Rc<Liveliness>>,
}

impl<'a> Analyzer<'a> {
    /// Create a new program analyser.
    pub fn new(program: &'a Program, metadata: &'a ProgramMetadata) -> Self {
        Self {
            program,
            metadata,
            cfg_cache: HashMap::new(),
            dominators_cache: HashMap::new(),
            liveliness_cache: HashMap::new(),
        }
    }

    /// Get the program being analysed.
    pub fn program(&self) -> &Program { self.program }

    /// Analyse the control flow graph of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze_cfg(&mut self, func: Function) -> Rc<ControlFlowGraph> {
        self.cfg_cache
            .entry(func)
            .or_insert_with(|| {
                trace!(->[0] "MGR " => "Analyzing CFG of {:?}", func);
                let cfg = ControlFlowGraph::analyze(self.program.func(func));
                Rc::new(cfg)
            })
            .clone()
    }

    /// Analyse dominators tree of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze_dominators(&mut self, func: Function) -> Rc<Dominators> {
        let cfg = self.analyze_cfg(func);
        self.dominators_cache
            .entry(func)
            .or_insert_with(|| {
                trace!(->[0] "MGR " => "Analyzing dominators of {:?}", func);
                let dominators = Dominators::analyze(cfg.as_ref());
                Rc::new(dominators)
            })
            .clone()
    }

    /// Analyse variable liveliness of a function.
    ///
    /// # Panics
    /// Panics if the function has no basic blocks, i.e. a function declaration.
    pub fn analyze_liveliness(&mut self, func: Function) -> Rc<Liveliness> {
        let cfg = self.analyze_cfg(func);
        self.liveliness_cache
            .entry(func)
            .or_insert_with(|| {
                trace!(->[0] "MGR " => "Analyzing liveliness of {:?}", func);
                let liveliness = Liveliness::analysis(cfg.as_ref(), self.program.func(func));
                Rc::new(liveliness)
            })
            .clone()
    }

    /// Analyse register allocation of a function.
    ///
    /// Note: this analysis does not have a cache.
    pub fn analyze_register_alloc(&mut self, func: Function) -> Result<RegAlloc> {
        trace!(->[0] "MGR " => "Analyzing register allocation of {:?}", func);
        let liveliness = self.analyze_liveliness(func);
        let dominators = self.analyze_dominators(func);
        RegAlloc::analyze(
            liveliness.as_ref(),
            dominators.as_ref(),
            self.program.func(func),
            &self.metadata.functions[&func],
        )
    }
}
