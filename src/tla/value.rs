use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// TLA+ values with Arc-wrapped collection types for zero-copy cloning.
/// Cloning a TlaValue with nested collections now only increments reference counts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TlaValue {
    Bool(bool),
    Int(i64),
    String(String),
    ModelValue(String),
    Set(Arc<BTreeSet<TlaValue>>),
    Seq(Arc<Vec<TlaValue>>),
    Record(Arc<BTreeMap<String, TlaValue>>),
    Function(Arc<BTreeMap<TlaValue, TlaValue>>),
    Lambda {
        params: Arc<Vec<String>>,
        body: String,
        captured_locals: Arc<BTreeMap<String, TlaValue>>,
    },
    Undefined,
}

pub type TlaState = BTreeMap<String, TlaValue>;

impl TlaValue {
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
        Ok(Self::Set(Arc::new(out)))
    }

    pub fn set_intersection(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(Arc::new(out)))
    }

    pub fn set_minus(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| !rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(Arc::new(out)))
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

    #[test]
    fn set_ops_work() {
        let a = TlaValue::Set(Arc::new(BTreeSet::from([
            TlaValue::Int(1),
            TlaValue::Int(2),
        ])));
        let b = TlaValue::Set(Arc::new(BTreeSet::from([
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
        let seq = TlaValue::Seq(Arc::new(vec![TlaValue::ModelValue("a".to_string())]));
        assert_eq!(
            seq.apply(&TlaValue::Int(1))
                .expect("sequence index should work"),
            &TlaValue::ModelValue("a".to_string())
        );

        let rec = TlaValue::Record(Arc::new(BTreeMap::from([(
            "x".to_string(),
            TlaValue::Int(7),
        )])));
        assert_eq!(
            rec.select_key("x").expect("record key should exist"),
            &TlaValue::Int(7)
        );
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
        prop::collection::btree_set(arb_tla_value(), 0..10).prop_map(|s| TlaValue::Set(Arc::new(s)))
    }

    /// Generate a sequence of TlaValues
    fn arb_tla_seq() -> impl Strategy<Value = TlaValue> {
        prop::collection::vec(arb_tla_value(), 0..10).prop_map(|v| TlaValue::Seq(Arc::new(v)))
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
            let seq = TlaValue::Seq(Arc::new(items.clone()));
            prop_assert_eq!(seq.len().unwrap(), items.len());
        }

        /// Sequence indexing (1-based)
        #[test]
        fn seq_indexing_works(items in prop::collection::vec(arb_tla_value(), 1..10)) {
            let seq = TlaValue::Seq(Arc::new(items.clone()));
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
