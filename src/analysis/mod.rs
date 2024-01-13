//! Program analysis.

use std::{cell::OnceCell, collections::HashMap, rc::Rc};

use koopa::ir::{Function, Program};
use miette::Result;
#[allow(unused_imports)]
use nolog::*;

use crate::{
    analysis::{
        call::FunctionCalls, cfg::ControlFlowGraph, dominators::Dominators, frame::Frame,
        global::GlobalValues, liveliness::Liveliness, localvar::LocalVars, register::RegAlloc,
    },
    irgen::metadata::ProgramMetadata,
};

pub mod call;
pub mod cfg;
pub mod dominators;
pub mod error;
pub mod frame;
pub mod global;
pub mod liveliness;
pub mod localvar;
pub mod register;

/// Program analyser.
pub struct Analyzer<'a> {
    program: &'a Program,
    metadata: &'a ProgramMetadata,
    global_values: OnceCell<Rc<GlobalValues>>,
    cfg_cache: HashMap<Function, Rc<ControlFlowGraph>>,
    dominators_cache: HashMap<Function, Rc<Dominators>>,
    liveliness_cache: HashMap<Function, Rc<Liveliness>>,
    local_vars_cache: HashMap<Function, Rc<LocalVars>>,
    regalloc_cache: HashMap<Function, Rc<RegAlloc>>,
    calls_cache: HashMap<Function, Rc<FunctionCalls>>,
    frame_cache: HashMap<Function, Rc<Frame>>,
}

impl<'a> Analyzer<'a> {
    /// Create a new program analyser.
    pub fn new(program: &'a Program, metadata: &'a ProgramMetadata) -> Self {
        Self {
            program,
            metadata,
            global_values: OnceCell::new(),
            cfg_cache: HashMap::new(),
            dominators_cache: HashMap::new(),
            liveliness_cache: HashMap::new(),
            local_vars_cache: HashMap::new(),
            regalloc_cache: HashMap::new(),
            calls_cache: HashMap::new(),
            frame_cache: HashMap::new(),
        }
    }

    /// Get the program being analysed.
    pub fn program(&self) -> &Program {
        self.program
    }

    /// Analyse global values.
    pub fn analyze_global_values(&self) -> Rc<GlobalValues> {
        self.global_values
            .get_or_init(|| {
                trace!(->[0] "MGR " => "Analyzing global values");
                let global_values = GlobalValues::analyze(self.program);
                Rc::new(global_values)
            })
            .clone()
    }

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

    /// Analyse local variables of a function.
    pub fn analyze_local_vars(&mut self, func: Function) -> Rc<LocalVars> {
        if self.local_vars_cache.contains_key(&func) {
            return self.local_vars_cache[&func].clone();
        }
        trace!(->[0] "MGR " => "Analyzing local variables of {:?}", func);
        let local_vars = LocalVars::analyze(self.program.func(func));
        let local_vars = Rc::new(local_vars);
        self.local_vars_cache.insert(func, local_vars.clone());
        local_vars
    }

    /// Analyse register allocation of a function.
    pub fn analyze_register_alloc(&mut self, func: Function) -> Rc<RegAlloc> {
        if self.regalloc_cache.contains_key(&func) {
            return self.regalloc_cache[&func].clone();
        }
        trace!(->[0] "MGR " => "Analyzing register allocation of {:?}", func);
        let liveliness = self.analyze_liveliness(func);
        let dominators = self.analyze_dominators(func);

        let reg_alloc = RegAlloc::analyze(
            liveliness.as_ref(),
            dominators.as_ref(),
            self.program.func(func),
        );
        let reg_alloc = Rc::new(reg_alloc);
        self.regalloc_cache.insert(func, reg_alloc.clone());
        reg_alloc
    }

    /// Analyse function calls of a function.
    pub fn analyze_function_calls(&mut self, func: Function) -> Rc<FunctionCalls> {
        if self.calls_cache.contains_key(&func) {
            return self.calls_cache[&func].clone();
        }
        trace!(->[0] "MGR " => "Analyzing function calls of {:?}", func);
        let liveliness = self.analyze_liveliness(func);
        let reg_alloc = self.analyze_register_alloc(func);
        let function_calls = call::FunctionCalls::analyze(
            liveliness.as_ref(),
            reg_alloc.as_ref(),
            self.program.func(func),
        );
        let function_calls = Rc::new(function_calls);
        self.calls_cache.insert(func, function_calls.clone());
        function_calls
    }

    /// Analyse frame size of a function.
    pub fn analyze_frame(&mut self, func: Function) -> Result<Rc<Frame>> {
        if self.frame_cache.contains_key(&func) {
            return Ok(self.frame_cache[&func].clone());
        }
        trace!(->[0] "MGR " => "Analyzing frame size of {:?}", func);
        let local_vars = self.analyze_local_vars(func);
        let reg_alloc = self.analyze_register_alloc(func);
        let function_calls = self.analyze_function_calls(func);
        let frame = Frame::analyze(
            local_vars.as_ref(),
            reg_alloc.as_ref(),
            function_calls.as_ref(),
            &self.metadata.functions[&func],
        )?;
        let frame = Rc::new(frame);
        self.frame_cache.insert(func, frame.clone());
        Ok(frame)
    }
}
