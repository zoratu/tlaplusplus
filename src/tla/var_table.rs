use std::collections::HashMap;

/// Maps variable names to compact integer indices for O(1) state access.
/// Populated once at module parse time, then shared immutably across all workers.
#[derive(Clone, Debug)]
pub struct VarTable {
    name_to_id: HashMap<String, u16>,
    id_to_name: Vec<String>,
}

impl VarTable {
    pub fn new(variable_names: &[String]) -> Self {
        let mut name_to_id = HashMap::with_capacity(variable_names.len());
        let mut id_to_name = Vec::with_capacity(variable_names.len());
        for (i, name) in variable_names.iter().enumerate() {
            name_to_id.insert(name.clone(), i as u16);
            id_to_name.push(name.clone());
        }
        Self {
            name_to_id,
            id_to_name,
        }
    }

    pub fn resolve(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    pub fn name(&self, id: u16) -> &str {
        &self.id_to_name[id as usize]
    }

    pub fn len(&self) -> usize {
        self.id_to_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_name.is_empty()
    }
}
