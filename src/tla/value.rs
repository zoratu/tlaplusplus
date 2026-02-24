use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TlaValue {
    Bool(bool),
    Int(i64),
    String(String),
    ModelValue(String),
    Set(BTreeSet<TlaValue>),
    Seq(Vec<TlaValue>),
    Record(BTreeMap<String, TlaValue>),
    Function(BTreeMap<TlaValue, TlaValue>),
    Lambda {
        params: Vec<String>,
        body: String,
        captured_locals: BTreeMap<String, TlaValue>,
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
            Self::Set(v) => Ok(v),
            _ => Err(anyhow!("expected Set, got {self:?}")),
        }
    }

    pub fn set_union(&self, other: &Self) -> Result<Self> {
        let mut out = self.as_set()?.clone();
        for value in other.as_set()? {
            out.insert(value.clone());
        }
        Ok(Self::Set(out))
    }

    pub fn set_intersection(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(out))
    }

    pub fn set_minus(&self, other: &Self) -> Result<Self> {
        let lhs = self.as_set()?;
        let rhs = other.as_set()?;
        let out = lhs
            .iter()
            .filter(|v| !rhs.contains(*v))
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(Self::Set(out))
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
        let a = TlaValue::Set(BTreeSet::from([TlaValue::Int(1), TlaValue::Int(2)]));
        let b = TlaValue::Set(BTreeSet::from([TlaValue::Int(2), TlaValue::Int(3)]));

        let u = a.set_union(&b).expect("union should work");
        let i = a.set_intersection(&b).expect("intersection should work");
        let d = a.set_minus(&b).expect("minus should work");

        assert_eq!(u.len().expect("len should work"), 3);
        assert_eq!(i.len().expect("len should work"), 1);
        assert_eq!(d.len().expect("len should work"), 1);
    }

    #[test]
    fn sequence_and_record_access_work() {
        let seq = TlaValue::Seq(vec![TlaValue::ModelValue("a".to_string())]);
        assert_eq!(
            seq.apply(&TlaValue::Int(1))
                .expect("sequence index should work"),
            &TlaValue::ModelValue("a".to_string())
        );

        let rec = TlaValue::Record(BTreeMap::from([("x".to_string(), TlaValue::Int(7))]));
        assert_eq!(
            rec.select_key("x").expect("record key should exist"),
            &TlaValue::Int(7)
        );
    }
}
