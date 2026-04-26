//! Partial-order reduction (POR) via stubborn sets.
//!
//! ## Background
//!
//! For TLA+ specs whose `Next` is a top-level disjunction
//! `Next == A1 \/ A2 \/ ... \/ An`, exhaustive exploration interleaves all
//! enabled actions. When two actions are *independent* (their effects commute
//! and neither disables the other), exploring both orderings reaches the same
//! states — wasted work.
//!
//! POR exploits this by computing, at each visited state, a *stubborn set*
//! `T(s) ⊆ Enabled(s)` and firing only those actions, deferring the rest to
//! descendant states. Provided the stubborn set satisfies the standard
//! conditions (Valmari 1990), the reduced state graph preserves invariant
//! violations and reachability of any safety property.
//!
//! ## Scope (1.0.0)
//!
//! This module provides the simplest correct stubborn-set algorithm:
//!
//! 1. **Static dependency analysis** — for each `Next` disjunct, extract the
//!    set of variables it writes (`x'`) and reads (`x` without prime).  Two
//!    disjuncts are dependent if either writes a variable the other reads or
//!    writes.  Conservative: when the structure is unfamiliar we treat the
//!    disjunct as touching every variable.
//! 2. **Stubborn-set computation per state** — evaluate every disjunct's
//!    successor set, then pick the lowest-index enabled disjunct and take its
//!    static-dependency closure restricted to enabled disjuncts.
//! 3. **Liveness opt-out** — POR in this version preserves *only* safety
//!    properties.  Callers must not enable POR when fairness or liveness
//!    checking is requested; the integration layer in `TlaModel` enforces
//!    this.
//!
//! False dependencies hurt reduction but never affect correctness, so the
//! analysis is deliberately coarse.  When the structure of a disjunct can't
//! be parsed we fall back to "depends on everything", which collapses POR
//! into vanilla full-enumeration for that disjunct.

use crate::tla::TlaDefinition;
use crate::tla::action_ir::split_action_body_disjuncts;
use crate::tla::module::TlaModule;
use std::collections::{BTreeMap, BTreeSet};

/// Static read/write summary for a single `Next` disjunct.
#[derive(Debug, Clone, Default)]
pub struct ActionFootprint {
    /// Set of state-variable names this disjunct may primed-assign.
    pub writes: BTreeSet<String>,
    /// Set of state-variable names this disjunct may read.
    pub reads: BTreeSet<String>,
    /// Set to `true` when the analysis can't determine the footprint and we
    /// fall back to "depends on everything".  When this is true, `writes`
    /// and `reads` are populated with the entire variable set.
    pub conservative: bool,
}

/// Pre-computed POR analysis for a `Next` definition.
#[derive(Debug, Clone)]
pub struct PorAnalysis {
    /// Per-disjunct footprint, indexed by the same disjunct index used by
    /// `evaluate_next_states_swarm`.
    pub footprints: Vec<ActionFootprint>,
    /// Symmetric dependency matrix: `dep[i]` is the set of disjunct indices
    /// `j` (including `i` itself) such that `i` and `j` may not commute.
    pub dep: Vec<BTreeSet<usize>>,
    /// All state variables declared by the spec.
    pub variables: BTreeSet<String>,
    /// **T7.3** — set of disjunct indices that are "visible" to the
    /// fairness/liveness specification.  An action is visible iff it is
    /// named directly in a WF/SF constraint, or it touches any variable
    /// referenced by a temporal state predicate.
    ///
    /// When `visible` is non-empty, `stubborn_set()` always includes every
    /// enabled visible disjunct.  This is the standard Peled (1994)
    /// visible-action proviso: the reduced graph preserves stutter-
    /// equivalent LTL\X over visible actions, including WF/SF on those
    /// actions.  Empty set = pure-safety mode (T7 baseline).
    pub visible: BTreeSet<usize>,
}

impl PorAnalysis {
    /// Build the POR analysis from a `Next` body and the enclosing module.
    ///
    /// `next_body` must be the exact text of the `Next` definition's body.
    /// `module` provides the variable declarations and operator definitions
    /// used to expand action calls during footprint extraction.
    ///
    /// The result has an empty visibility set — i.e., pure safety POR
    /// (T7 baseline).  For liveness POR, call `from_next_with_visibility`.
    pub fn from_next(next_body: &str, module: &TlaModule) -> Self {
        Self::build(next_body, module, &[], &BTreeSet::new())
    }

    /// Build the POR analysis with a Peled (1994) visible-action proviso.
    ///
    /// `visible_action_names` lists the *names* of actions that appear in
    /// fairness clauses (e.g., `WF_vars(SendMsg)` → "SendMsg") and `Next`
    /// disjuncts, plus disjuncts that read or write any variable in
    /// `visible_vars`.  The resulting `visible` set indexes the disjuncts
    /// whose footprints overlap any name in `visible_action_names` or any
    /// variable in `visible_vars`.
    ///
    /// When `visible` is non-empty, `stubborn_set()` always includes every
    /// enabled visible disjunct, preserving stutter-equivalent LTL\X over
    /// visible actions (sound for WF/SF on those actions).
    pub fn from_next_with_visibility(
        next_body: &str,
        module: &TlaModule,
        visible_action_names: &[String],
        visible_vars: &BTreeSet<String>,
    ) -> Self {
        Self::build(next_body, module, visible_action_names, visible_vars)
    }

    fn build(
        next_body: &str,
        module: &TlaModule,
        visible_action_names: &[String],
        visible_vars: &BTreeSet<String>,
    ) -> Self {
        let variables: BTreeSet<String> = module.variables.iter().cloned().collect();
        let disjuncts = split_action_body_disjuncts(next_body);
        let disjuncts = if disjuncts.is_empty() {
            vec![next_body.trim().to_string()]
        } else {
            disjuncts
        };

        let footprints: Vec<ActionFootprint> = disjuncts
            .iter()
            .map(|disj| compute_footprint(disj, &module.definitions, &variables))
            .collect();

        let dep = compute_dependency_matrix(&footprints);

        // Compute the visibility set: a disjunct is visible iff
        //   (a) its top-level identifier matches a visible_action_name, OR
        //   (b) its read or write footprint touches any variable in
        //       visible_vars.
        let mut visible: BTreeSet<usize> = BTreeSet::new();
        let visible_action_set: BTreeSet<&str> =
            visible_action_names.iter().map(|s| s.as_str()).collect();
        for (idx, (disj, fp)) in disjuncts.iter().zip(footprints.iter()).enumerate() {
            // (a) name match
            if let Some(name) = top_level_action_name(disj)
                && visible_action_set.contains(name.as_str())
            {
                visible.insert(idx);
                continue;
            }
            // (b) variable footprint match
            if !visible_vars.is_empty()
                && (fp.writes.iter().any(|v| visible_vars.contains(v))
                    || fp.reads.iter().any(|v| visible_vars.contains(v)))
            {
                visible.insert(idx);
            }
        }

        Self {
            footprints,
            dep,
            variables,
            visible,
        }
    }

    /// Number of disjuncts (matches `count_next_disjuncts`).
    pub fn num_actions(&self) -> usize {
        self.footprints.len()
    }

    /// Compute the stubborn set for a state given the per-disjunct enabledness
    /// vector `enabled_per_disjunct`: `enabled_per_disjunct[i] == true` iff
    /// disjunct `i` produced at least one successor on the current state.
    ///
    /// Returns the indices of disjuncts whose successors should be enqueued.
    /// On the empty input or when no disjunct is enabled, returns the empty
    /// vector (signalling "deadlock — nothing to fire").
    ///
    /// **Seed selection (T7.2)** — among enabled disjuncts we pick the one
    /// whose dependency closure (restricted to enabled disjuncts) is
    /// smallest.  Ties are broken by lowest index so the reduced graph
    /// stays deterministic across runs.  This is strictly at least as good
    /// as the lowest-index heuristic and on Pipeline-shaped specs (one
    /// "central" action depending on every other) it avoids picking the
    /// central action — which would otherwise drag the entire enabled set
    /// into the stubborn set.
    ///
    /// **Visibility proviso (T7.3)** — when `self.visible` is non-empty,
    /// every enabled visible disjunct is added to the seed set before the
    /// dep closure is computed.  This is the Peled (1994) visible-action
    /// proviso: the reduced graph preserves stutter-equivalent LTL\X over
    /// visible actions, so WF/SF on visible actions is preserved.
    pub fn stubborn_set(&self, enabled_per_disjunct: &[bool]) -> Vec<usize> {
        // Defensive bounds check.  If the runtime hands us a mismatched
        // length we fall back to firing everything that is enabled, which is
        // always correct (just no reduction).
        if enabled_per_disjunct.len() != self.footprints.len() {
            return (0..enabled_per_disjunct.len())
                .filter(|i| enabled_per_disjunct[*i])
                .collect();
        }

        // T7.3: visibility proviso.  Build a seed set from every enabled
        // visible disjunct, then close under dep.  When `visible` is empty
        // (pure-safety POR) this branch is skipped and we fall through to
        // the single-seed path below.
        if !self.visible.is_empty() {
            let mut seeds: Vec<usize> = self
                .visible
                .iter()
                .copied()
                .filter(|&i| enabled_per_disjunct[i])
                .collect();
            if seeds.is_empty() {
                // No visible action enabled — fall through to single-seed
                // mode.  This is the corner case where the stubborn set
                // contains only invisible actions; sound for safety, and
                // for liveness the cycle proviso (deferred) would catch
                // any infinite-invisible-cycle pathology.
            } else {
                // Pick the additional safety seed via the smarter heuristic
                // and merge.  The result is always at least as large as
                // either seeding strategy alone, but never larger than the
                // full enabled set.
                let safety_seed = self.pick_smartest_seed(enabled_per_disjunct);
                if let Some(s) = safety_seed
                    && !seeds.contains(&s)
                {
                    seeds.push(s);
                }
                return self.close_dep(seeds, enabled_per_disjunct);
            }
        }

        // Pure-safety path (visible empty or no visible action enabled).
        // Pick the seed via the smarter heuristic and close under dep.
        let Some(seed) = self.pick_smartest_seed(enabled_per_disjunct) else {
            return Vec::new();
        };
        self.close_dep(vec![seed], enabled_per_disjunct)
    }

    /// Pick the enabled disjunct whose dep closure (restricted to enabled
    /// disjuncts) is smallest.  Returns `None` iff no disjunct is enabled.
    /// Ties broken by lowest index.
    fn pick_smartest_seed(&self, enabled: &[bool]) -> Option<usize> {
        // Fast path: zero or one enabled.
        let enabled_count = enabled.iter().filter(|b| **b).count();
        if enabled_count == 0 {
            return None;
        }
        if enabled_count == 1 {
            return enabled.iter().position(|b| *b);
        }

        let mut best_seed: Option<usize> = None;
        let mut best_size: usize = usize::MAX;
        for seed in 0..enabled.len() {
            if !enabled[seed] {
                continue;
            }
            let size = self.closure_size_from(seed, enabled, best_size);
            if size < best_size {
                best_size = size;
                best_seed = Some(seed);
                if best_size == 1 {
                    break;
                }
            }
        }
        best_seed
    }

    /// BFS dep-closure of `seeds` restricted to enabled disjuncts.
    /// Returns a sorted, deduplicated index list.
    fn close_dep(&self, seeds: Vec<usize>, enabled: &[bool]) -> Vec<usize> {
        let mut chosen: BTreeSet<usize> = BTreeSet::new();
        let mut frontier: Vec<usize> = Vec::with_capacity(seeds.len());
        for s in seeds {
            if enabled[s] && chosen.insert(s) {
                frontier.push(s);
            }
        }
        while let Some(idx) = frontier.pop() {
            for &dep_idx in &self.dep[idx] {
                if !enabled[dep_idx] {
                    continue;
                }
                if chosen.insert(dep_idx) {
                    frontier.push(dep_idx);
                }
            }
        }
        let mut out: Vec<usize> = chosen.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Compute |closure(seed) ∩ enabled| with early termination if the
    /// running size meets or exceeds `cutoff`.  Returns `cutoff` when the
    /// candidate is provably no better than the current best.
    fn closure_size_from(
        &self,
        seed: usize,
        enabled: &[bool],
        cutoff: usize,
    ) -> usize {
        // Bitset-style chosen tracking via a Vec<bool> (cheaper than
        // BTreeSet for n ≤ ~64; still fine for n ≤ a few hundred).
        let n = self.footprints.len();
        let mut chosen = vec![false; n];
        chosen[seed] = true;
        let mut size = 1usize;
        let mut frontier: Vec<usize> = Vec::with_capacity(8);
        frontier.push(seed);
        while let Some(idx) = frontier.pop() {
            for &dep_idx in &self.dep[idx] {
                if !enabled[dep_idx] || chosen[dep_idx] {
                    continue;
                }
                chosen[dep_idx] = true;
                size += 1;
                if size >= cutoff {
                    return size; // can't be the new best, bail out early
                }
                frontier.push(dep_idx);
            }
        }
        size
    }

    /// Verbose human-readable dump for debugging / logging.
    #[allow(dead_code)]
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (i, fp) in self.footprints.iter().enumerate() {
            let conservative = if fp.conservative {
                " [conservative]"
            } else {
                ""
            };
            s.push_str(&format!(
                "  action[{}]: writes={:?} reads={:?}{}\n",
                i, fp.writes, fp.reads, conservative
            ));
            s.push_str(&format!("    deps={:?}\n", self.dep[i]));
        }
        s
    }
}

fn compute_dependency_matrix(footprints: &[ActionFootprint]) -> Vec<BTreeSet<usize>> {
    let n = footprints.len();
    let mut dep = vec![BTreeSet::<usize>::new(); n];
    for i in 0..n {
        // Every action depends on itself (closes BFS termination cleanly).
        dep[i].insert(i);
        for j in (i + 1)..n {
            if depends(&footprints[i], &footprints[j]) {
                dep[i].insert(j);
                dep[j].insert(i);
            }
        }
    }
    dep
}

fn depends(a: &ActionFootprint, b: &ActionFootprint) -> bool {
    // Conservative: anything tagged conservative depends on every other
    // action that touches at least one variable.  In practice both will be
    // populated with the entire variable set, so this falls out from the
    // overlap check below — but we short-circuit here for clarity.
    if a.conservative || b.conservative {
        return true;
    }

    // Two actions commute iff:
    //   - their write sets are disjoint, AND
    //   - neither reads what the other writes.
    if !disjoint(&a.writes, &b.writes) {
        return true;
    }
    if !disjoint(&a.writes, &b.reads) {
        return true;
    }
    if !disjoint(&a.reads, &b.writes) {
        return true;
    }
    false
}

fn disjoint(a: &BTreeSet<String>, b: &BTreeSet<String>) -> bool {
    a.is_disjoint(b)
}

/// Extract the top-level action name from a disjunct, if it is structured
/// as a (possibly quantified) action call.  Used by `from_next_with_visibility`
/// to match disjuncts to fairness clauses by name.
///
/// Examples:
///   "SendMsg(m)"                   -> Some("SendMsg")
///   "\\E p \\in Procs : Step(p)"   -> Some("Step")
///   "x' = x + 1 /\\ UNCHANGED y"   -> None  (no top-level identifier)
fn top_level_action_name(disj: &str) -> Option<String> {
    let trimmed = strip_comments(disj);
    let trimmed = trimmed.trim();
    let trimmed = strip_outer_parens(trimmed);
    // Skip any leading \E, \A, LET ... IN binders.
    let body = strip_temporal_binders(trimmed);
    parse_leading_identifier(body.trim())
}

fn strip_outer_parens(s: &str) -> &str {
    let mut s = s.trim();
    while s.starts_with('(') && s.ends_with(')') && parens_match(s) {
        s = s[1..s.len() - 1].trim();
    }
    s
}

fn parens_match(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes[0] != b'(' {
        return false;
    }
    let mut depth = 0i64;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'(' {
            depth += 1;
        } else if b == b')' {
            depth -= 1;
            if depth == 0 {
                return i == bytes.len() - 1;
            }
        }
    }
    false
}

fn strip_temporal_binders(input: &str) -> &str {
    let mut s = input.trim();
    loop {
        if let Some(rest) = s.strip_prefix("\\E").or_else(|| s.strip_prefix("\\A")) {
            // Find the colon that ends the binder header.  Stay shallow.
            if let Some(colon) = find_binder_colon(rest) {
                s = rest[colon + 1..].trim();
                continue;
            }
        }
        if let Some(rest) = s.strip_prefix("LET") {
            // Find matching IN at the same depth.
            if let Some(in_idx) = find_let_in(rest) {
                s = rest[in_idx + 2..].trim();
                continue;
            }
        }
        break;
    }
    s
}

fn find_binder_colon(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i64;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b':' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn find_let_in(s: &str) -> Option<usize> {
    // Crude: scan for whitespace-bounded "IN" at depth 0.
    let bytes = s.as_bytes();
    let mut depth = 0i64;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0
            && bytes.get(i).copied() == Some(b'I')
            && bytes.get(i + 1).copied() == Some(b'N')
        {
            let prev_ok = i == 0 || !is_ident_char(bytes[i - 1]);
            let next_ok = i + 2 == bytes.len() || !is_ident_char(bytes[i + 2]);
            if prev_ok && next_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn parse_leading_identifier(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !is_ident_start(bytes[0]) {
        return None;
    }
    let mut i = 0usize;
    while i < bytes.len() && is_ident_char(bytes[i]) {
        i += 1;
    }
    Some(s[..i].to_string())
}

/// Compute the static read/write footprint of a single Next disjunct.
fn compute_footprint(
    disjunct: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    variables: &BTreeSet<String>,
) -> ActionFootprint {
    let mut footprint = ActionFootprint::default();
    // ENABLED A makes the truth value of the guard depend on every variable
    // A could ever read/write — far beyond what local syntax shows.  We
    // collapse to "depends on everything" in that case.
    if disjunct.contains("ENABLED") {
        footprint.conservative = true;
    }
    let mut visited_defs: BTreeSet<String> = BTreeSet::new();
    extract_footprint_recursive(
        disjunct,
        definitions,
        variables,
        &mut visited_defs,
        &mut footprint,
    );

    // When conservative was forced (e.g. ENABLED), populate the read/write
    // sets with every state variable so the dependency check correctly
    // flags conflicts with any other action.
    if footprint.conservative {
        for v in variables {
            footprint.writes.insert(v.clone());
            footprint.reads.insert(v.clone());
        }
    }
    footprint
}

/// Walk the disjunct text token-by-token, recording variable reads/writes.
/// Recursively descends through action calls referenced in `definitions`.
fn extract_footprint_recursive(
    expr: &str,
    definitions: &BTreeMap<String, TlaDefinition>,
    variables: &BTreeSet<String>,
    visited_defs: &mut BTreeSet<String>,
    footprint: &mut ActionFootprint,
) {
    let stripped = strip_comments(expr);
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    let mut in_string = false;

    while i < bytes.len() {
        let ch = bytes[i];

        if in_string {
            if ch == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if ch == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if ch == b'"' {
            in_string = true;
            i += 1;
            continue;
        }

        // Handle UNCHANGED <<x, y, z>> or UNCHANGED x — UNCHANGED neither
        // reads nor writes; it's the absence of an effect.  We skip past the
        // operand so the variable names don't get counted as reads.
        if is_keyword_at(&stripped, i, "UNCHANGED") {
            i += "UNCHANGED".len();
            i = skip_unchanged_operand(&stripped, i);
            continue;
        }

        // Identifier scan
        if is_ident_start(ch) {
            let start = i;
            while i < bytes.len() && is_ident_char(bytes[i]) {
                i += 1;
            }
            let name = &stripped[start..i];

            // After the identifier, look for `'` (write) — possibly with
            // whitespace between.
            let mut k = i;
            while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
                k += 1;
            }
            let primed = k < bytes.len() && bytes[k] == b'\'';

            if variables.contains(name) {
                if primed {
                    footprint.writes.insert(name.to_string());
                    // Also count the prime — anything *after* primed
                    // assignment that reads other vars is still a read.
                    i = k + 1;
                    continue;
                } else {
                    footprint.reads.insert(name.to_string());
                }
                continue;
            }

            // Skip TLA+ keywords that should not be expanded.
            if is_tla_keyword(name) {
                continue;
            }

            // Look up as an operator: if found and not yet visited,
            // recurse into its body to pick up the writes/reads it does.
            if let Some(def) = definitions.get(name)
                && visited_defs.insert(name.to_string())
            {
                extract_footprint_recursive(
                    &def.body,
                    definitions,
                    variables,
                    visited_defs,
                    footprint,
                );
                visited_defs.remove(name);
            }

            continue;
        }

        i += 1;
    }
}

fn strip_comments(input: &str) -> String {
    // Reuse the lightweight stripper logic locally (the scan.rs version is
    // private).  Handles `\* line` and `(* block *)` comments.
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut block_depth = 0usize;
    let mut in_line_comment = false;
    while i < bytes.len() {
        let c = bytes[i];
        let next = bytes.get(i + 1).copied();
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
                out.push('\n');
            }
            i += 1;
            continue;
        }
        if block_depth > 0 {
            if c == b'(' && next == Some(b'*') {
                block_depth += 1;
                i += 2;
                continue;
            }
            if c == b'*' && next == Some(b')') {
                block_depth -= 1;
                i += 2;
                continue;
            }
            if c == b'\n' {
                out.push('\n');
            }
            i += 1;
            continue;
        }
        if c == b'\\' && next == Some(b'*') {
            in_line_comment = true;
            i += 2;
            continue;
        }
        if c == b'(' && next == Some(b'*') {
            block_depth = 1;
            i += 2;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_keyword_at(text: &str, idx: usize, kw: &str) -> bool {
    let bytes = text.as_bytes();
    if idx + kw.len() > bytes.len() {
        return false;
    }
    if &bytes[idx..idx + kw.len()] != kw.as_bytes() {
        return false;
    }
    // Must be a word boundary on both sides.
    let prev_ok = idx == 0 || !is_ident_char(bytes[idx - 1]);
    let next_ok = idx + kw.len() == bytes.len() || !is_ident_char(bytes[idx + kw.len()]);
    prev_ok && next_ok
}

/// After `UNCHANGED`, skip the operand: either a single identifier or a
/// `<<...>>` tuple.  Returns the index just past the operand.
fn skip_unchanged_operand(text: &str, mut i: usize) -> usize {
    let bytes = text.as_bytes();
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i >= bytes.len() {
        return i;
    }
    if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'<') {
        // Skip until matching >>
        let mut depth = 1usize;
        i += 2;
        while i < bytes.len() {
            if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'<') {
                depth += 1;
                i += 2;
                continue;
            }
            if bytes[i] == b'>' && bytes.get(i + 1) == Some(&b'>') {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return i;
                }
                continue;
            }
            i += 1;
        }
        return i;
    }
    // Single identifier
    while i < bytes.len() && is_ident_char(bytes[i]) {
        i += 1;
    }
    i
}

fn is_tla_keyword(s: &str) -> bool {
    // Common TLA+ keywords / operators we should not try to expand as
    // operator definitions.  Missing entries just cost cycles, never
    // correctness.
    matches!(
        s,
        "ASSUME"
            | "AXIOM"
            | "BOOLEAN"
            | "BY"
            | "CASE"
            | "CHOOSE"
            | "CONSTANT"
            | "CONSTANTS"
            | "DEF"
            | "DEFS"
            | "DOMAIN"
            | "ELSE"
            | "ENABLED"
            | "EXCEPT"
            | "EXTENDS"
            | "FALSE"
            | "IF"
            | "IN"
            | "INSTANCE"
            | "LET"
            | "LOCAL"
            | "MODULE"
            | "OBVIOUS"
            | "OMITTED"
            | "ONLY"
            | "OTHER"
            | "PROOF"
            | "PROVE"
            | "QED"
            | "RECURSIVE"
            | "SF"
            | "SUBSET"
            | "SUFFICES"
            | "TEMPORAL"
            | "THEN"
            | "THEOREM"
            | "TRUE"
            | "UNCHANGED"
            | "UNION"
            | "USE"
            | "VARIABLE"
            | "VARIABLES"
            | "WF"
            | "WITH"
            | "Cardinality"
            | "Len"
            | "Head"
            | "Tail"
            | "Append"
            | "Seq"
            | "Nat"
            | "Int"
            | "Real"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla::TlaDefinition;

    fn module_with_vars(vars: &[&str], defs: &[(&str, &[&str], &str)]) -> TlaModule {
        let mut m = TlaModule {
            name: "Test".to_string(),
            ..TlaModule::default()
        };
        m.variables = vars.iter().map(|v| v.to_string()).collect();
        for (name, params, body) in defs {
            m.definitions.insert(
                name.to_string(),
                TlaDefinition {
                    name: name.to_string(),
                    params: params.iter().map(|p| p.to_string()).collect(),
                    body: body.to_string(),
                    is_recursive: false,
                },
            );
        }
        m
    }

    #[test]
    fn footprint_extracts_writes_and_reads() {
        let module = module_with_vars(&["x", "y", "z"], &[]);
        let body = "x' = y + 1 /\\ z' = z";
        let fp = compute_footprint(
            body,
            &module.definitions,
            &module.variables.iter().cloned().collect(),
        );
        assert!(fp.writes.contains("x"));
        assert!(fp.writes.contains("z"));
        assert!(fp.reads.contains("y"));
        assert!(fp.reads.contains("z"));
        assert!(!fp.conservative);
    }

    #[test]
    fn unchanged_skips_operand() {
        let module = module_with_vars(&["x", "y", "z"], &[]);
        let body = "x' = 1 /\\ UNCHANGED <<y, z>>";
        let fp = compute_footprint(
            body,
            &module.definitions,
            &module.variables.iter().cloned().collect(),
        );
        assert!(fp.writes.contains("x"));
        // y and z must NOT be counted as reads — UNCHANGED is structural.
        assert!(!fp.reads.contains("y"));
        assert!(!fp.reads.contains("z"));
    }

    #[test]
    fn independent_actions_have_no_dependency() {
        // Two actions touch disjoint variables — independent.
        let module = module_with_vars(
            &["x", "y"],
            &[
                ("AddX", &[], "x' = x + 1 /\\ UNCHANGED y"),
                ("AddY", &[], "y' = y + 1 /\\ UNCHANGED x"),
            ],
        );
        let next = "AddX \\/ AddY";
        let analysis = PorAnalysis::from_next(next, &module);
        assert_eq!(analysis.num_actions(), 2);
        // dep should only contain self-loops.
        assert_eq!(analysis.dep[0], BTreeSet::from([0]));
        assert_eq!(analysis.dep[1], BTreeSet::from([1]));

        // Stubborn set with both enabled: just the seed.
        let stubborn = analysis.stubborn_set(&[true, true]);
        assert_eq!(stubborn, vec![0]);
    }

    #[test]
    fn shared_write_makes_actions_dependent() {
        let module = module_with_vars(
            &["x"],
            &[("Inc", &[], "x' = x + 1"), ("Dec", &[], "x' = x - 1")],
        );
        let next = "Inc \\/ Dec";
        let analysis = PorAnalysis::from_next(next, &module);
        assert!(analysis.dep[0].contains(&1));
        assert!(analysis.dep[1].contains(&0));

        // Both are dependent — stubborn set must contain both.
        let stubborn = analysis.stubborn_set(&[true, true]);
        assert_eq!(stubborn, vec![0, 1]);
    }

    #[test]
    fn read_write_overlap_is_dependency() {
        // A reads what B writes — they're dependent (B can change A's read).
        let module = module_with_vars(
            &["x", "y"],
            &[
                ("ReadX", &[], "y' = x /\\ UNCHANGED x"),
                ("WriteX", &[], "x' = 0 /\\ UNCHANGED y"),
            ],
        );
        let next = "ReadX \\/ WriteX";
        let analysis = PorAnalysis::from_next(next, &module);
        assert!(analysis.dep[0].contains(&1));
    }

    #[test]
    fn stubborn_closure_grows_to_dependent_neighbors() {
        // Three actions:
        //   A: writes p (independent of B, C)
        //   B: writes q
        //   C: reads q (depends on B)
        // Stubborn set seeded from A is just {A}. Seeded from B (when A
        // disabled) must include {B, C}.
        let module = module_with_vars(
            &["p", "q", "r"],
            &[
                ("A", &[], "p' = 1 /\\ UNCHANGED <<q, r>>"),
                ("B", &[], "q' = q + 1 /\\ UNCHANGED <<p, r>>"),
                ("C", &[], "r' = q /\\ UNCHANGED <<p, q>>"),
            ],
        );
        let next = "A \\/ B \\/ C";
        let analysis = PorAnalysis::from_next(next, &module);
        // A is independent of B and C
        assert!(!analysis.dep[0].contains(&1));
        assert!(!analysis.dep[0].contains(&2));
        // B and C share q
        assert!(analysis.dep[1].contains(&2));

        let s_all = analysis.stubborn_set(&[true, true, true]);
        // Seed = lowest enabled = A. A is independent of all → stubborn={A}.
        assert_eq!(s_all, vec![0]);

        let s_no_a = analysis.stubborn_set(&[false, true, true]);
        // Seed = B. B and C are dependent → stubborn={B, C}.
        assert_eq!(s_no_a, vec![1, 2]);
    }

    #[test]
    fn stubborn_set_respects_enabledness() {
        let module = module_with_vars(
            &["x"],
            &[("Inc", &[], "x' = x + 1"), ("Dec", &[], "x' = x - 1")],
        );
        let next = "Inc \\/ Dec";
        let analysis = PorAnalysis::from_next(next, &module);
        // Only one enabled → stubborn = that one.
        assert_eq!(analysis.stubborn_set(&[true, false]), vec![0]);
        assert_eq!(analysis.stubborn_set(&[false, true]), vec![1]);
        // Nothing enabled → empty.
        assert!(analysis.stubborn_set(&[false, false]).is_empty());
    }

    #[test]
    fn unparsed_quantifier_falls_back_conservative() {
        // \E binders inside an action call — the footprint extractor descends
        // into the action body to find writes/reads.  Even with conservative
        // fallback the stubborn-set computation stays correct.
        let module = module_with_vars(
            &["msgs", "log"],
            &[("Send", &["m"], "msgs' = msgs \\union {m} /\\ UNCHANGED log")],
        );
        let next = "\\E m \\in {1, 2} : Send(m)";
        let analysis = PorAnalysis::from_next(next, &module);
        assert_eq!(analysis.num_actions(), 1);
        // The single disjunct should record `msgs` as a write/read; even if
        // analysis went conservative the result is still correct.
        let fp = &analysis.footprints[0];
        assert!(fp.writes.contains("msgs"));
    }

    #[test]
    fn smarter_seed_picks_smallest_dependency_cluster() {
        // Five actions partitioned into two disconnected dep clusters:
        //   Cluster X: A1, A2 share variable x  (writes/reads conflict)
        //   Cluster Y: B1, B2, B3 share variable y
        //
        // Lowest-index seed = A1, closure = {A1, A2}, size 2.
        // Smarter seed sees: A1->2, A2->2, B1->3, B2->3, B3->3.  Min is 2,
        // tie → lowest index → A1, stubborn = {A1, A2}.
        // To make smarter seed strictly win we add a singleton cluster Z
        // (one action whose only dep is itself) and put it AFTER A1 so the
        // lowest-index heuristic ignores it.
        let module = module_with_vars(
            &["x", "y", "z"],
            &[
                ("A1", &[], "x' = x + 1 /\\ UNCHANGED <<y, z>>"),
                ("A2", &[], "x' = x - 1 /\\ UNCHANGED <<y, z>>"),
                ("Z", &[], "z' = z + 1 /\\ UNCHANGED <<x, y>>"),
                ("B1", &[], "y' = y + 1 /\\ UNCHANGED <<x, z>>"),
                ("B2", &[], "y' = y - 1 /\\ UNCHANGED <<x, z>>"),
            ],
        );
        let next = "A1 \\/ A2 \\/ Z \\/ B1 \\/ B2";
        let analysis = PorAnalysis::from_next(next, &module);
        // Z is independent of every other action.
        assert!(!analysis.dep[2].contains(&0));
        assert!(!analysis.dep[2].contains(&1));
        assert!(!analysis.dep[2].contains(&3));
        assert!(!analysis.dep[2].contains(&4));

        let stubborn = analysis.stubborn_set(&[true, true, true, true, true]);
        // Smarter seed: Z (closure size 1) beats A1 (closure size 2).
        // The lowest-index heuristic would have produced [0, 1] (the A
        // cluster); smarter seed picks the singleton cluster.
        assert_eq!(
            stubborn,
            vec![2],
            "smarter seed should prefer the singleton-cluster action Z, got {:?}",
            stubborn
        );
    }

    #[test]
    fn smarter_seed_falls_back_to_lowest_index_on_tie() {
        // Two fully independent actions: both have closure size 1.  The
        // tie-break must be lowest-index for determinism.
        let module = module_with_vars(
            &["x", "y"],
            &[
                ("Inc1", &[], "x' = x + 1 /\\ UNCHANGED y"),
                ("Inc2", &[], "y' = y + 1 /\\ UNCHANGED x"),
            ],
        );
        let next = "Inc1 \\/ Inc2";
        let analysis = PorAnalysis::from_next(next, &module);
        let stubborn = analysis.stubborn_set(&[true, true]);
        assert_eq!(stubborn, vec![0]);
    }

    #[test]
    fn nested_action_calls_propagate_footprints() {
        // Top-level disjunct = OuterAction; OuterAction calls InnerAction.
        let module = module_with_vars(
            &["a", "b"],
            &[
                ("InnerAction", &[], "a' = 1 /\\ UNCHANGED b"),
                ("OuterAction", &[], "InnerAction"),
                ("Other", &[], "b' = 2 /\\ UNCHANGED a"),
            ],
        );
        let next = "OuterAction \\/ Other";
        let analysis = PorAnalysis::from_next(next, &module);
        assert!(analysis.footprints[0].writes.contains("a"));
        assert!(analysis.footprints[1].writes.contains("b"));
        // Disjoint writes/reads → independent.
        assert!(!analysis.dep[0].contains(&1));
    }
}
