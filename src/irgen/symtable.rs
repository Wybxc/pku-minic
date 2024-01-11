//! Symbol table.

use std::{borrow::Borrow, collections::HashMap, hash::Hash};

use koopa::ir::{Function, Value};

/// Symbol, can be const or variable.
pub enum Symbol {
    Const(i32),
    Var(Value),
    Func(Function),
}

/// Symbol table.
pub struct SymbolTable {
    chain_map: ChainMap<String, Symbol>,
}

impl SymbolTable {
    /// Create a new symbol table.
    pub fn new() -> Self {
        let chain_map = ChainMap::new();
        Self { chain_map }
    }

    /// Push a new scope.
    pub fn push(&mut self) { self.chain_map.push(); }

    /// Pop a scope.
    pub fn pop(&mut self) { self.chain_map.pop(); }

    /// Insert a symbol, return the old symbol in same scope if exists.
    ///
    /// # Panics
    /// Panics if there is no scope.
    #[must_use]
    pub fn insert_var(&mut self, ident: String, symbol: Symbol) -> Option<Symbol> {
        self.chain_map.insert(ident, symbol)
    }

    /// Get a symbol.
    pub fn get_var(&self, ident: &str) -> Option<&Symbol> { self.chain_map.get(ident) }
}

impl Default for SymbolTable {
    fn default() -> Self { Self::new() }
}

/// Chain map.
///
/// A chain map is a stack of hash maps.
struct ChainMap<K: Eq + Hash, V> {
    maps: Vec<HashMap<K, V>>,
}

impl<K: Eq + Hash, V> ChainMap<K, V> {
    /// Create a new chain map with no scopes.
    fn new() -> Self { Self { maps: vec![] } }

    /// Push a new scope.
    fn push(&mut self) { self.maps.push(HashMap::new()); }

    /// Pop a scope.
    fn pop(&mut self) { self.maps.pop(); }

    /// Insert a key-value pair, return the old value in same scope if exists.
    ///
    /// # Panics
    /// Panics if there is no scope.
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.maps
            .last_mut()
            .expect("chain map is empty")
            .insert(key, value)
    }

    /// Get a value.
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
