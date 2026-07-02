use anyhow::{anyhow, Result};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::tla::hashed_arc::HashedArc;

/// TLA+ values with Arc-wrapped collection types for zero-copy cloning.
/// Cloning a TlaValue with nested collections now only increments reference counts.
///
/// Collection variants (Set/Seq/Record/Function) use `HashedArc` instead of
/// plain `Arc` to cache a `u64` fingerprint hash at construction time. This
/// makes `Ord::cmp` between two collection-bearing TlaValues fast-fail on
/// hash inequality (the common case during BTreeMap/BTreeSet inserts during
/// state exploration), instead of recursing structurally on every compare —
/// the hotspot the eval-perf profiling on MCBinarySearch surfaced. See
/// `src/tla/hashed_arc.rs` for the soundness invariants.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TlaValue {
    Bool(bool),
    Int(i64),
    String(String),
    ModelValue(String),
    Set(HashedArc<BTreeSet<TlaValue>>),
    Seq(HashedArc<Vec<TlaValue>>),
    Record(HashedArc<BTreeMap<String, TlaValue>>),
    Function(HashedArc<BTreeMap<TlaValue, TlaValue>>),
    Lambda {
        params: Arc<Vec<String>>,
        body: String,
        captured_locals: Arc<BTreeMap<String, TlaValue>>,
    },
    Undefined,
}

#[derive(Clone, Debug)]
pub struct StateSchema {
    pub names: Vec<Arc<str>>,
    slot_of: HashMap<Arc<str>, u32>,
}

thread_local! {
    static ACTIVE_SCHEMA: RefCell<Option<Arc<StateSchema>>> = RefCell::new(None);
}

impl StateSchema {
    pub fn new(mut names: Vec<Arc<str>>) -> Self {
        names.sort();
        names.dedup();
        assert!(
            names.len() <= u32::MAX as usize,
            "TlaState schema has too many slots"
        );
        let slot_of = names
            .iter()
            .enumerate()
            .map(|(slot, name)| (Arc::clone(name), slot as u32))
            .collect();
        Self { names, slot_of }
    }

    pub fn same_names(&self, other: &Self) -> bool {
        self.names == other.names
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

pub fn set_active_schema(names: Vec<Arc<str>>) {
    ACTIVE_SCHEMA.with(|active| {
        *active.borrow_mut() = Some(Arc::new(StateSchema::new(names)));
    });
}

pub fn clear_active_schema() {
    ACTIVE_SCHEMA.with(|active| {
        *active.borrow_mut() = None;
    });
}

fn schema_for_names(names: Vec<Arc<str>>) -> Arc<StateSchema> {
    let schema = StateSchema::new(names);
    ACTIVE_SCHEMA.with(|active| {
        if let Some(active_schema) = active.borrow().as_ref() {
            if active_schema.same_names(&schema) {
                return Arc::clone(active_schema);
            }
        }
        Arc::new(schema)
    })
}

#[derive(Clone)]
pub struct TlaState {
    schema: Arc<StateSchema>,
    values: Vec<TlaValue>,
}

#[derive(Clone, Debug, Default)]
pub struct StateBuilder {
    entries: BTreeMap<Arc<str>, TlaValue>,
}

pub struct TlaStateIter<'a> {
    names: std::slice::Iter<'a, Arc<str>>,
    values: std::slice::Iter<'a, TlaValue>,
}

pub enum TlaStateEntry<'a> {
    Occupied(&'a mut TlaValue),
    Vacant {
        state: &'a mut TlaState,
        name: Arc<str>,
    },
}

impl<'a> Iterator for TlaStateIter<'a> {
    type Item = (&'a Arc<str>, &'a TlaValue);

    fn next(&mut self) -> Option<Self::Item> {
        Some((self.names.next()?, self.values.next()?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl ExactSizeIterator for TlaStateIter<'_> {}

impl StateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: Arc<str>, value: TlaValue) -> Option<TlaValue> {
        self.entries.insert(name, value)
    }

    pub fn remove(&mut self, name: &str) -> Option<TlaValue> {
        self.entries.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&TlaValue> {
        self.entries.get(name)
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn finish(self) -> TlaState {
        TlaState::from_entries(self.entries)
    }
}

impl FromIterator<(Arc<str>, TlaValue)> for StateBuilder {
    fn from_iter<T: IntoIterator<Item = (Arc<str>, TlaValue)>>(iter: T) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl TlaState {
    pub fn new() -> Self {
        Self {
            schema: schema_for_names(Vec::new()),
            values: Vec::new(),
        }
    }

    fn from_entries(entries: BTreeMap<Arc<str>, TlaValue>) -> Self {
        let names: Vec<Arc<str>> = entries.keys().cloned().collect();
        let schema = schema_for_names(names);
        let values = schema
            .names
            .iter()
            .map(|name| {
                entries
                    .get(name)
                    .expect("schema names must match StateBuilder entries")
                    .clone()
            })
            .collect();
        Self { schema, values }
    }

    fn same_schema_names(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.schema, &other.schema) || self.schema.same_names(&other.schema)
    }

    fn as_btree_map(&self) -> BTreeMap<Arc<str>, TlaValue> {
        self.iter()
            .map(|(k, v)| (Arc::clone(k), v.clone()))
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&TlaValue> {
        self.slot_of(name)
            .and_then(|slot| self.values.get(slot as usize))
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut TlaValue> {
        self.slot_of(name)
            .and_then(|slot| self.values.get_mut(slot as usize))
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.slot_of(name).is_some()
    }

    pub fn insert(&mut self, name: Arc<str>, value: TlaValue) -> Option<TlaValue> {
        if let Some(slot) = self.slot_of(name.as_ref()) {
            return Some(std::mem::replace(&mut self.values[slot as usize], value));
        }

        let mut entries = self.as_btree_map();
        entries.insert(name, value);
        *self = Self::from_entries(entries);
        None
    }

    pub fn entry(&mut self, name: Arc<str>) -> TlaStateEntry<'_> {
        match self.slot_of(name.as_ref()) {
            Some(slot) => TlaStateEntry::Occupied(&mut self.values[slot as usize]),
            None => TlaStateEntry::Vacant { state: self, name },
        }
    }

    pub fn remove(&mut self, name: &str) -> Option<TlaValue> {
        self.slot_of(name)?;
        let mut entries = self.as_btree_map();
        let removed = entries.remove(name);
        *self = Self::from_entries(entries);
        removed
    }

    pub fn iter(&self) -> TlaStateIter<'_> {
        TlaStateIter {
            names: self.schema.names.iter(),
            values: self.values.iter(),
        }
    }

    pub fn keys(&self) -> std::slice::Iter<'_, Arc<str>> {
        self.schema.names.iter()
    }

    pub fn values(&self) -> std::slice::Iter<'_, TlaValue> {
        self.values.iter()
    }

    pub fn values_mut(&mut self) -> std::slice::IterMut<'_, TlaValue> {
        self.values.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get_slot(&self, slot: u32) -> Option<&TlaValue> {
        self.values.get(slot as usize)
    }

    pub fn slot_of(&self, name: &str) -> Option<u32> {
        self.schema.slot_of.get(name).copied()
    }

    /// Build a new state by mapping each value, PRESERVING the schema `Arc` and
    /// slot order. This is the hot-path way to transform a state whose variable
    /// set is unchanged (e.g. symmetry permutes the ModelValues *inside* values
    /// but never the variable names): it avoids the `iter().collect()` round-trip
    /// through `StateBuilder`/`from_entries`, which would re-derive a schema
    /// (sort + HashMap build) and clone every key+value twice.
    pub fn map_values(&self, f: impl Fn(&TlaValue) -> TlaValue) -> TlaState {
        TlaState {
            schema: Arc::clone(&self.schema),
            values: self.values.iter().map(f).collect(),
        }
    }
}

impl<'a> TlaStateEntry<'a> {
    pub fn or_insert(self, default: TlaValue) -> &'a mut TlaValue {
        match self {
            Self::Occupied(value) => value,
            Self::Vacant { state, name } => {
                state.insert(Arc::clone(&name), default);
                state
                    .get_mut(name.as_ref())
                    .expect("inserted TlaState entry must be present")
            }
        }
    }

    pub fn or_insert_with(self, default: impl FnOnce() -> TlaValue) -> &'a mut TlaValue {
        match self {
            Self::Occupied(value) => value,
            Self::Vacant { state, name } => {
                state.insert(Arc::clone(&name), default());
                state
                    .get_mut(name.as_ref())
                    .expect("inserted TlaState entry must be present")
            }
        }
    }
}

impl Default for TlaState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TlaState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl PartialEq for TlaState {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.same_schema_names(other) {
            return self.values == other.values;
        }
        self.iter().eq(other.iter())
    }
}

impl Eq for TlaState {}

impl PartialOrd for TlaState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TlaState {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.same_schema_names(other) {
            return self.values.cmp(&other.values);
        }
        self.iter().cmp(other.iter())
    }
}

impl Hash for TlaState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the (name, value) pairs in schema (sorted) order — consistent
        // with `Eq`/`Ord` (equal states hash equally) and with `BTreeMap`'s
        // length-then-entries scheme. Avoids rebuilding a `BTreeMap` per hash
        // (which cloned every Arc<str> key + value). `Hash` is only used for
        // in-memory HashSet/HashMap keys within a run, so the exact value need
        // not match the old impl — only Eq-consistency matters.
        state.write_usize(self.len());
        for (name, value) in self.iter() {
            name.hash(state);
            value.hash(state);
        }
    }
}

impl std::ops::Index<&str> for TlaState {
    type Output = TlaValue;

    fn index(&self, index: &str) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("TlaState missing key '{index}'"))
    }
}

impl Serialize for TlaState {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (name, value) in self.iter() {
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for TlaState {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let entries = BTreeMap::<Arc<str>, TlaValue>::deserialize(deserializer)?;
        Ok(Self::from_entries(entries))
    }
}

impl FromIterator<(Arc<str>, TlaValue)> for TlaState {
    fn from_iter<T: IntoIterator<Item = (Arc<str>, TlaValue)>>(iter: T) -> Self {
        StateBuilder::from_iter(iter).finish()
    }
}

impl IntoIterator for TlaState {
    type Item = (Arc<str>, TlaValue);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.schema
            .names
            .clone()
            .into_iter()
            .zip(self.values)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a> IntoIterator for &'a TlaState {
    type Item = (&'a Arc<str>, &'a TlaValue);
    type IntoIter = TlaStateIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Build a [`TlaState`] from `(&str, TlaValue)` pairs, converting keys to `Arc<str>`.
pub fn tla_state<const N: usize>(pairs: [(&str, TlaValue); N]) -> TlaState {
    pairs.into_iter().map(|(k, v)| (Arc::from(k), v)).collect()
}

/// Normalize every `1..n`-domain function in a state to its `Seq` form (see
/// [`TlaValue::normalize_seq_functions`]), returning the rewritten state — but
/// only when at least one variable actually changes. Returns `None` for an
/// already-normal state so hot-path callers (state fingerprinting) avoid
/// cloning and reallocating untouched states.
pub fn normalize_state_if_changed(state: &TlaState) -> Option<TlaState> {
    let mut values = state.values.clone();
    let mut changed = false;
    for (idx, value) in state.values.iter().enumerate() {
        if let Some(normalized) = value.normalize_seq_changed() {
            values[idx] = normalized;
            changed = true;
        }
    }
    if changed {
        Some(TlaState {
            schema: Arc::clone(&state.schema),
            values,
        })
    } else {
        None
    }
}

impl TlaValue {
    /// TLA+ value equality, treating a `Seq` and a `Function` whose domain is
    /// `1..Len` as equal — because in TLA+ a sequence IS a function over
    /// `1..Len`, so `<<a,b,c>> = [i \in 1..3 |-> ...]` is TRUE. Our `TlaValue`
    /// keeps `Seq` and `Function` as distinct variants, so the derived `==`
    /// reports them unequal; this is the semantics the `=`/`#`/`/=` operators
    /// must use. Recurses through Set/Seq/Record/Function so nested sequences
    /// (e.g. a record field or set element built two different ways) also
    /// compare equal.
    pub fn semantic_eq(&self, other: &TlaValue) -> bool {
        match (self, other) {
            (TlaValue::Seq(s), TlaValue::Function(f))
            | (TlaValue::Function(f), TlaValue::Seq(s)) => Self::seq_eq_function(s, f),
            (TlaValue::Seq(a), TlaValue::Seq(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.semantic_eq(y))
            }
            (TlaValue::Function(a), TlaValue::Function(b)) => {
                // Keys of equal functions sort identically (function keys are
                // Int/ModelValue/String/etc., whose order is representation-
                // independent), so positional comparison is sound and avoids an
                // O(n^2) cross match on the hot path.
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((k1, v1), (k2, v2))| k1.semantic_eq(k2) && v1.semantic_eq(v2))
            }
            (TlaValue::Record(a), TlaValue::Record(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((k1, v1), (k2, v2))| k1 == k2 && v1.semantic_eq(v2))
            }
            (TlaValue::Set(a), TlaValue::Set(b)) => {
                // Set equality is by membership; a Seq element and its Function
                // twin must match, so we can't rely on `BTreeSet ==`.
                a.len() == b.len() && a.iter().all(|x| b.iter().any(|y| x.semantic_eq(y)))
            }
            // Primitives and genuinely-different variants: derived equality.
            _ => self == other,
        }
    }

    /// True when sequence `s` equals function `f` viewed as a TLA+ sequence:
    /// `f`'s domain is exactly `1..s.len()` and `f[i]` equals `s[i-1]`.
    fn seq_eq_function(s: &[TlaValue], f: &BTreeMap<TlaValue, TlaValue>) -> bool {
        if s.len() != f.len() {
            return false;
        }
        for (i, elem) in s.iter().enumerate() {
            match f.get(&TlaValue::Int(i as i64 + 1)) {
                Some(v) if elem.semantic_eq(v) => {}
                _ => return false,
            }
        }
        true
    }

    /// In TLA+ a sequence IS a function whose domain is `1..Len`, so
    /// `<<a,b,c>>` and `[i \in 1..3 |-> ...]` are *equal*. Our `TlaValue` keeps
    /// them as distinct variants (`Seq` vs `Function`), so logically-equal
    /// values built two ways (e.g. a log as `[i \in 1..n |-> e]` vs via
    /// `Append`) compare unequal and hash differently — inflating the state
    /// space because the fingerprint store fails to dedup them.
    ///
    /// `normalize_seq_functions` rewrites every function with domain exactly
    /// `{1,...,n}` (n >= 1) into its `Seq` form, recursively, giving one
    /// canonical representation. Returns `None` when nothing changed so callers
    /// on the hot path (state canonicalization) avoid reallocating untouched
    /// states. The empty function is left as-is (conservative; non-empty
    /// 1..n-domain functions are the ones that cause the dedup split in practice).
    pub(crate) fn normalize_seq_changed(&self) -> Option<TlaValue> {
        match self {
            TlaValue::Function(map) => {
                let mut any = false;
                let mut nmap: BTreeMap<TlaValue, TlaValue> = BTreeMap::new();
                for (k, v) in map.iter() {
                    let nk = k.normalize_seq_changed();
                    let nv = v.normalize_seq_changed();
                    any |= nk.is_some() || nv.is_some();
                    nmap.insert(
                        nk.unwrap_or_else(|| k.clone()),
                        nv.unwrap_or_else(|| v.clone()),
                    );
                }
                if Self::is_one_to_n_domain(&nmap) {
                    Some(TlaValue::Seq(HashedArc::new(nmap.into_values().collect())))
                } else if any {
                    Some(TlaValue::Function(HashedArc::new(nmap)))
                } else {
                    None
                }
            }
            TlaValue::Seq(items) => {
                let mut any = false;
                let nitems: Vec<TlaValue> = items
                    .iter()
                    .map(|v| match v.normalize_seq_changed() {
                        Some(n) => {
                            any = true;
                            n
                        }
                        None => v.clone(),
                    })
                    .collect();
                any.then(|| TlaValue::Seq(HashedArc::new(nitems)))
            }
            TlaValue::Set(items) => {
                let mut any = false;
                let nitems: BTreeSet<TlaValue> = items
                    .iter()
                    .map(|v| match v.normalize_seq_changed() {
                        Some(n) => {
                            any = true;
                            n
                        }
                        None => v.clone(),
                    })
                    .collect();
                any.then(|| TlaValue::Set(HashedArc::new(nitems)))
            }
            TlaValue::Record(fields) => {
                let mut any = false;
                let nfields: BTreeMap<String, TlaValue> = fields
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            match v.normalize_seq_changed() {
                                Some(n) => {
                                    any = true;
                                    n
                                }
                                None => v.clone(),
                            },
                        )
                    })
                    .collect();
                any.then(|| TlaValue::Record(HashedArc::new(nfields)))
            }
            _ => None,
        }
    }

    /// Owned-return convenience wrapper around [`Self::normalize_seq_changed`].
    pub fn normalize_seq_functions(&self) -> TlaValue {
        self.normalize_seq_changed().unwrap_or_else(|| self.clone())
    }

    /// True when `map`'s keys are exactly `Int(1), Int(2), ..., Int(n)` for some
    /// `n >= 1` — i.e. the function is a TLA+ sequence. `BTreeMap` iterates keys
    /// in sorted order and all `Int` keys sort numerically within the `Int`
    /// variant, so positional comparison is sound.
    fn is_one_to_n_domain(map: &BTreeMap<TlaValue, TlaValue>) -> bool {
        if map.is_empty() {
            return false;
        }
        for (i, k) in map.keys().enumerate() {
            match k {
                TlaValue::Int(v) if *v == (i as i64 + 1) => {}
                _ => return false,
            }
        }
        true
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(anyhow!("expected BOOLEAN, got {self:?}")),
        }
    }

    pub fn as_int(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            _ => Err(anyhow!("expected Int, got {self:?}")),
        }
    }

    pub fn as_set(&self) -> Result<&BTreeSet<TlaValue>> {
        match self {
            Self::Set(v) => Ok(v.as_ref()),
            _ => Err(anyhow!("expected Set, got {self:?}")),
        }
    }

    pub fn as_seq(&self) -> Result<&Vec<TlaValue>> {
        match self {
            Self::Seq(v) => Ok(v.as_ref()),
            _ => Err(anyhow!("expected Seq, got {self:?}")),
        }
    }

    pub fn as_record(&self) -> Result<&BTreeMap<String, TlaValue>> {
        match self {
            Self::Record(v) => Ok(v.as_ref()),
            _ => Err(anyhow!("expected Record, got {self:?}")),
        }
    }

    pub fn as_function(&self) -> Result<&BTreeMap<TlaValue, TlaValue>> {
        match self {
            Self::Function(v) => Ok(v.as_ref()),
            _ => Err(anyhow!("expected Function, got {self:?}")),
        }
    }

    pub fn set_union(&self, other: &Self) -> Result<Self> {
        let mut out = self.as_set()?.clone();
        for value in other.as_set()? {
            out.insert(value.clone());
        }
        Ok(Self::Set(HashedArc::new(out)))
    }

    pub fn set_intersection(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(HashedArc::new(out)))
    }

    pub fn set_minus(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| !rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(HashedArc::new(out)))
    }

    pub fn contains(&self, value: &Self) -> Result<bool> {
        Ok(self.as_set()?.contains(value))
    }

    pub fn len(&self) -> Result<usize> {
        match self {
            Self::Set(v) => Ok(v.len()),
            Self::Seq(v) => Ok(v.len()),
            Self::Record(v) => Ok(v.len()),
            Self::Function(v) => Ok(v.len()),
            _ => Err(anyhow!("Len undefined for value {self:?}")),
        }
    }

    pub fn is_empty(&self) -> Result<bool> {
        match self {
            Self::Set(v) => Ok(v.is_empty()),
            Self::Seq(v) => Ok(v.is_empty()),
            Self::Record(v) => Ok(v.is_empty()),
            Self::Function(v) => Ok(v.is_empty()),
            _ => Err(anyhow!("is_empty undefined for value {self:?}")),
        }
    }

    pub fn select_key(&self, key: &str) -> Result<&Self> {
        match self {
            Self::Record(map) => map
                .get(key)
                .ok_or_else(|| anyhow!("record missing key '{key}'")),
            _ => Err(anyhow!("record access on non-record value {self:?}")),
        }
    }

    pub fn apply(&self, key: &Self) -> Result<&Self> {
        match self {
            Self::Function(map) => map
                .get(key)
                .ok_or_else(|| anyhow!("function missing key {key:?}")),
            Self::Record(map) => {
                let key = match key {
                    Self::String(v) | Self::ModelValue(v) => v,
                    _ => {
                        return Err(anyhow!(
                            "record index must be string/model value, got {key:?}"
                        ));
                    }
                };
                map.get(key)
                    .ok_or_else(|| anyhow!("record missing key '{key}'"))
            }
            Self::Seq(items) => {
                let idx = key.as_int()?;
                if idx <= 0 {
                    return Err(anyhow!("sequence index must be >= 1, got {idx}"));
                }
                let zero = (idx - 1) as usize;
                items
                    .get(zero)
                    .ok_or_else(|| anyhow!("sequence index out of range: {idx}"))
            }
            _ => Err(anyhow!(
                "function application unsupported for value {self:?}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A TLA+ sequence IS a function over `1..Len`, so `=` must treat them as
    /// equal even though our `TlaValue` keeps them as distinct variants.
    #[test]
    fn semantic_eq_treats_seq_as_function_over_one_to_n() {
        let seq = TlaValue::Seq(HashedArc::new(vec![
            TlaValue::Int(7),
            TlaValue::Int(8),
            TlaValue::Int(9),
        ]));
        let func = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(7)),
            (TlaValue::Int(2), TlaValue::Int(8)),
            (TlaValue::Int(3), TlaValue::Int(9)),
        ])));
        // Equal both directions; the derived `==` (which we must NOT use for `=`)
        // would report them unequal.
        assert!(seq.semantic_eq(&func));
        assert!(func.semantic_eq(&seq));
        assert_ne!(
            seq, func,
            "derived eq distinguishes the variants (as expected)"
        );

        // Different values / wrong domain / different length are NOT equal.
        let func_diff = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(1), TlaValue::Int(7)),
            (TlaValue::Int(2), TlaValue::Int(8)),
            (TlaValue::Int(3), TlaValue::Int(0)),
        ])));
        assert!(!seq.semantic_eq(&func_diff));
        let func_0based = TlaValue::Function(HashedArc::new(BTreeMap::from([
            (TlaValue::Int(0), TlaValue::Int(7)),
            (TlaValue::Int(1), TlaValue::Int(8)),
            (TlaValue::Int(2), TlaValue::Int(9)),
        ])));
        assert!(
            !seq.semantic_eq(&func_0based),
            "0-based function is not this sequence"
        );

        // Nested: a record/set carrying the two forms compares equal too.
        let rec_seq = TlaValue::Record(HashedArc::new(BTreeMap::from([(
            "log".to_string(),
            seq.clone(),
        )])));
        let rec_func = TlaValue::Record(HashedArc::new(BTreeMap::from([(
            "log".to_string(),
            func.clone(),
        )])));
        assert!(rec_seq.semantic_eq(&rec_func));
        let set_seq = TlaValue::Set(HashedArc::new(BTreeSet::from([seq.clone()])));
        let set_func = TlaValue::Set(HashedArc::new(BTreeSet::from([func.clone()])));
        assert!(set_seq.semantic_eq(&set_func));
    }

    #[test]
    fn set_ops_work() {
        let a = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
        ])));
        let b = TlaValue::Set(HashedArc::new(BTreeSet::from([
            TlaValue::Int(2),
            TlaValue::Int(3),
        ])));

        let u = a.set_union(&b).expect("union should work");
        let i = a.set_intersection(&b).expect("intersection should work");
        let d = a.set_minus(&b).expect("minus should work");

        assert_eq!(u.len().expect("len should work"), 3);
        assert_eq!(i.len().expect("len should work"), 1);
        assert_eq!(d.len().expect("len should work"), 1);
    }

    #[test]
    fn sequence_and_record_access_work() {
        let seq = TlaValue::Seq(HashedArc::new(vec![TlaValue::ModelValue("a".to_string())]));
        assert_eq!(
            seq.apply(&TlaValue::Int(1))
                .expect("sequence index should work"),
            &TlaValue::ModelValue("a".to_string())
        );

        let rec = TlaValue::Record(HashedArc::new(BTreeMap::from([(
            "x".to_string(),
            TlaValue::Int(7),
        )])));
        assert_eq!(
            rec.select_key("x").expect("record key should exist"),
            &TlaValue::Int(7)
        );
    }

    #[test]
    fn state_matches_old_btreemap_eq_hash_order_and_serialization() {
        use std::collections::hash_map::DefaultHasher;

        fn hash_of<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        let old: BTreeMap<Arc<str>, TlaValue> = BTreeMap::from([
            (Arc::from("x"), TlaValue::Int(1)),
            (Arc::from("y"), TlaValue::Bool(true)),
        ]);
        let old_larger: BTreeMap<Arc<str>, TlaValue> = BTreeMap::from([
            (Arc::from("x"), TlaValue::Int(2)),
            (Arc::from("y"), TlaValue::Bool(true)),
        ]);

        let state_from_map: TlaState = old.clone().into_iter().collect();
        let state_from_pairs = tla_state([("y", TlaValue::Bool(true)), ("x", TlaValue::Int(1))]);
        let state_larger: TlaState = old_larger.clone().into_iter().collect();

        assert_eq!(state_from_map, state_from_pairs);
        assert_eq!(hash_of(&old), hash_of(&state_from_map));
        assert_eq!(old.cmp(&old_larger), state_from_map.cmp(&state_larger));
        assert_eq!(
            bincode::serialize(&old).expect("old map should serialize"),
            bincode::serialize(&state_from_map).expect("state should serialize")
        );
    }

    #[test]
    fn state_names_are_part_of_identity() {
        let x = tla_state([("x", TlaValue::Int(1))]);
        let y = tla_state([("y", TlaValue::Int(1))]);

        assert_ne!(x, y);
    }

    #[test]
    fn state_bincode_round_trip_preserves_entries() {
        let state = tla_state([("x", TlaValue::Int(1)), ("y", TlaValue::Bool(false))]);
        let bytes = bincode::serialize(&state).expect("state should serialize");
        let decoded: TlaState = bincode::deserialize(&bytes).expect("state should deserialize");

        assert_eq!(decoded, state);
        assert_eq!(decoded.get("x"), Some(&TlaValue::Int(1)));
        assert_eq!(decoded.get("y"), Some(&TlaValue::Bool(false)));
    }

    #[test]
    fn state_slots_are_available_after_building() {
        let state = tla_state([("y", TlaValue::Bool(true)), ("x", TlaValue::Int(5))]);

        let x_slot = state.slot_of("x").expect("x slot should exist");
        let y_slot = state.slot_of("y").expect("y slot should exist");

        assert_eq!(x_slot, 0);
        assert_eq!(y_slot, 1);
        assert_eq!(state.get_slot(x_slot), Some(&TlaValue::Int(5)));
        assert_eq!(state.get_slot(y_slot), Some(&TlaValue::Bool(true)));
    }
}

/// Property-based tests using proptest
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate arbitrary TlaValue for property testing
    fn arb_tla_value() -> impl Strategy<Value = TlaValue> {
        prop_oneof![
            // Primitive values
            any::<bool>().prop_map(TlaValue::Bool),
            (-1000i64..1000).prop_map(TlaValue::Int),
            "[a-z]{0,5}".prop_map(TlaValue::String),
            "[A-Z]{1,3}".prop_map(TlaValue::ModelValue),
        ]
    }

    /// Generate a set of TlaValues
    fn arb_tla_set() -> impl Strategy<Value = TlaValue> {
        prop::collection::btree_set(arb_tla_value(), 0..10)
            .prop_map(|s| TlaValue::Set(HashedArc::new(s)))
    }

    /// Generate a sequence of TlaValues
    #[allow(dead_code)]
    fn arb_tla_seq() -> impl Strategy<Value = TlaValue> {
        prop::collection::vec(arb_tla_value(), 0..10).prop_map(|v| TlaValue::Seq(HashedArc::new(v)))
    }

    proptest! {
        /// Set union is commutative: A ∪ B = B ∪ A
        #[test]
        fn set_union_commutative(a in arb_tla_set(), b in arb_tla_set()) {
            let ab = a.set_union(&b).unwrap();
            let ba = b.set_union(&a).unwrap();
            prop_assert_eq!(ab, ba);
        }

        /// Set union is associative: (A ∪ B) ∪ C = A ∪ (B ∪ C)
        #[test]
        fn set_union_associative(a in arb_tla_set(), b in arb_tla_set(), c in arb_tla_set()) {
            let ab_c = a.set_union(&b).unwrap().set_union(&c).unwrap();
            let a_bc = a.set_union(&b.set_union(&c).unwrap()).unwrap();
            prop_assert_eq!(ab_c, a_bc);
        }

        /// Set intersection is commutative: A ∩ B = B ∩ A
        #[test]
        fn set_intersection_commutative(a in arb_tla_set(), b in arb_tla_set()) {
            let ab = a.set_intersection(&b).unwrap();
            let ba = b.set_intersection(&a).unwrap();
            prop_assert_eq!(ab, ba);
        }

        /// Set intersection is associative: (A ∩ B) ∩ C = A ∩ (B ∩ C)
        #[test]
        fn set_intersection_associative(a in arb_tla_set(), b in arb_tla_set(), c in arb_tla_set()) {
            let ab_c = a.set_intersection(&b).unwrap().set_intersection(&c).unwrap();
            let a_bc = a.set_intersection(&b.set_intersection(&c).unwrap()).unwrap();
            prop_assert_eq!(ab_c, a_bc);
        }

        /// Set difference: |A \ B| <= |A|
        #[test]
        fn set_minus_shrinks(a in arb_tla_set(), b in arb_tla_set()) {
            let diff = a.set_minus(&b).unwrap();
            prop_assert!(diff.len().unwrap() <= a.len().unwrap());
        }

        /// Union contains both operands: A ⊆ (A ∪ B) and B ⊆ (A ∪ B)
        #[test]
        fn set_union_contains_operands(a in arb_tla_set(), b in arb_tla_set()) {
            let union = a.set_union(&b).unwrap();
            for elem in a.as_set().unwrap() {
                prop_assert!(union.contains(elem).unwrap());
            }
            for elem in b.as_set().unwrap() {
                prop_assert!(union.contains(elem).unwrap());
            }
        }

        /// Intersection is subset of both: (A ∩ B) ⊆ A and (A ∩ B) ⊆ B
        #[test]
        fn set_intersection_is_subset(a in arb_tla_set(), b in arb_tla_set()) {
            let inter = a.set_intersection(&b).unwrap();
            for elem in inter.as_set().unwrap() {
                prop_assert!(a.contains(elem).unwrap());
                prop_assert!(b.contains(elem).unwrap());
            }
        }

        /// Difference has no elements from B: (A \ B) ∩ B = ∅
        #[test]
        fn set_minus_disjoint(a in arb_tla_set(), b in arb_tla_set()) {
            let diff = a.set_minus(&b).unwrap();
            let inter = diff.set_intersection(&b).unwrap();
            prop_assert!(inter.is_empty().unwrap());
        }

        /// De Morgan: A \ (B ∪ C) = (A \ B) ∩ (A \ C)
        #[test]
        fn set_demorgan(a in arb_tla_set(), b in arb_tla_set(), c in arb_tla_set()) {
            let bc = b.set_union(&c).unwrap();
            let lhs = a.set_minus(&bc).unwrap();
            let ab = a.set_minus(&b).unwrap();
            let ac = a.set_minus(&c).unwrap();
            let rhs = ab.set_intersection(&ac).unwrap();
            prop_assert_eq!(lhs, rhs);
        }

        /// Sequence length matches vec length
        #[test]
        fn seq_length_correct(items in prop::collection::vec(arb_tla_value(), 0..20)) {
            let seq = TlaValue::Seq(HashedArc::new(items.clone()));
            prop_assert_eq!(seq.len().unwrap(), items.len());
        }

        /// Sequence indexing (1-based)
        #[test]
        fn seq_indexing_works(items in prop::collection::vec(arb_tla_value(), 1..10)) {
            let seq = TlaValue::Seq(HashedArc::new(items.clone()));
            for (i, expected) in items.iter().enumerate() {
                let idx = TlaValue::Int((i + 1) as i64); // 1-based indexing
                let actual = seq.apply(&idx).unwrap();
                prop_assert_eq!(actual, expected);
            }
        }

        /// Bool roundtrip
        #[test]
        fn bool_roundtrip(b in any::<bool>()) {
            let val = TlaValue::Bool(b);
            prop_assert_eq!(val.as_bool().unwrap(), b);
        }

        /// Int roundtrip
        #[test]
        fn int_roundtrip(i in any::<i64>()) {
            let val = TlaValue::Int(i);
            prop_assert_eq!(val.as_int().unwrap(), i);
        }
    }
}
