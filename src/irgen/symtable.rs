//! Symbol table.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use koopa::ir::Value;

/// Symbol, can be const or variable.
pub enum Symbol {
    Const(i32),
    Var(Value),
}

/// Symbol table.
pub struct SymbolTable {
    chain_map: ChainMap<String, Symbol>,
}

impl SymbolTable {
    /// Create a new symbol table.
    pub fn new() -> Self {
        let mut chain_map = ChainMap::new();
        chain_map.push();
        Self { chain_map }
    }

    /// Push a new scope.
    #[allow(dead_code)]
    pub fn push(&mut self) {
        self.chain_map.push();
    }

    /// Pop a scope.
    #[allow(dead_code)]
    pub fn pop(&mut self) {
        self.chain_map.pop();
    }

    /// Insert a symbol.
    pub fn insert_var(&mut self, ident: String, symbol: Symbol) {
        self.chain_map.insert(ident, symbol)
    }

    /// Get a symbol.
    pub fn get_var(&self, ident: &str) -> Option<&Symbol> {
        self.chain_map.get(ident)
    }
}

struct ChainMap<K: Eq + Hash, V> {
    maps: Vec<HashMap<K, V>>,
}

impl<K: Eq + Hash, V> ChainMap<K, V> {
    fn new() -> Self {
        Self { maps: vec![] }
    }

    fn push(&mut self) {
        self.maps.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.maps.pop();
    }

    fn insert(&mut self, key: K, value: V) {
        self.maps.last_mut().unwrap().insert(key, value);
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: ?Sized,
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        for map in self.maps.iter().rev() {
            if let Some(value) = map.get(key) {
                return Some(value);
            }
        }
        None
    }
}
