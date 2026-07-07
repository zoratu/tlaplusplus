//! Shadow-compare harness for the `expr_v2` parser (Phase 0 de-risk deliverable).
//!
//! For a large set of real TLA+ expressions (operator / definition bodies
//! extracted from the tlaplus/Examples corpus via the existing module parser),
//! parse each with BOTH the old `compile_expr_v1` and the new
//! `expr_v2::parse_and_lower`, canonicalize both `CompiledExpr`s to a normalized
//! debug form, and compare. Reports AGREE / DIFFER counts and categorizes the
//! DIFFERs.
//!
//! This test is OPT-IN: it only runs its full body when
//! `EXPR_V2_SHADOW=1` is set (so `cargo test` in CI stays fast and this never
//! gates on corpus availability). Point it at a corpus with
//! `EXPR_V2_CORPUS=/path/to/Examples` (defaults to `~/exmpl` then the vendored
//! `corpus/`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tlaplusplus::tla::compiled_expr::{compile_expr_v1, CompiledExpr};
use tlaplusplus::tla::expr_v2;
use tlaplusplus::tla::module::parse_tla_module_text;

/// Canonical structural rendering of a `CompiledExpr`, ignoring incidental
/// representation differences we don't care about for a shape comparison.
/// Whitespace inside `Unparsed`/`Var`/`String` payloads is normalized so that
/// e.g. `Unparsed("A  =  B")` vs `Unparsed("A = B")` compare equal.
fn canon(e: &CompiledExpr) -> String {
    fn norm_text(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    match e {
        CompiledExpr::Bool(b) => format!("Bool({b})"),
        CompiledExpr::Int(n) => format!("Int({n})"),
        CompiledExpr::String(s) => format!("Str({:?})", norm_text(s)),
        CompiledExpr::ModelValue(s) => format!("MV({s})"),
        CompiledExpr::Var(s) => format!("Var({s})"),
        CompiledExpr::StateVar { name, .. } => format!("Var({name})"),
        CompiledExpr::PrimedVar(s) => format!("Primed({s})"),
        CompiledExpr::SelfRef => "SelfRef".into(),
        CompiledExpr::And(xs) => format!("And[{}]", join(xs)),
        CompiledExpr::Or(xs) => format!("Or[{}]", join(xs)),
        CompiledExpr::Not(x) => format!("Not({})", canon(x)),
        CompiledExpr::Implies(a, b) => format!("Implies({},{})", canon(a), canon(b)),
        CompiledExpr::Iff(a, b) => format!("Iff({},{})", canon(a), canon(b)),
        CompiledExpr::Eq(a, b) => format!("Eq({},{})", canon(a), canon(b)),
        CompiledExpr::Neq(a, b) => format!("Neq({},{})", canon(a), canon(b)),
        CompiledExpr::Lt(a, b) => format!("Lt({},{})", canon(a), canon(b)),
        CompiledExpr::Le(a, b) => format!("Le({},{})", canon(a), canon(b)),
        CompiledExpr::Gt(a, b) => format!("Gt({},{})", canon(a), canon(b)),
        CompiledExpr::Ge(a, b) => format!("Ge({},{})", canon(a), canon(b)),
        CompiledExpr::In(a, b) => format!("In({},{})", canon(a), canon(b)),
        CompiledExpr::NotIn(a, b) => format!("NotIn({},{})", canon(a), canon(b)),
        CompiledExpr::Add(a, b) => format!("Add({},{})", canon(a), canon(b)),
        CompiledExpr::Sub(a, b) => format!("Sub({},{})", canon(a), canon(b)),
        CompiledExpr::Mul(a, b) => format!("Mul({},{})", canon(a), canon(b)),
        CompiledExpr::Pow(a, b) => format!("Pow({},{})", canon(a), canon(b)),
        CompiledExpr::Div(a, b) => format!("Div({},{})", canon(a), canon(b)),
        CompiledExpr::Mod(a, b) => format!("Mod({},{})", canon(a), canon(b)),
        CompiledExpr::Neg(a) => format!("Neg({})", canon(a)),
        CompiledExpr::Union(a, b) => format!("Union({},{})", canon(a), canon(b)),
        CompiledExpr::Intersect(a, b) => format!("Intersect({},{})", canon(a), canon(b)),
        CompiledExpr::SetMinus(a, b) => format!("SetMinus({},{})", canon(a), canon(b)),
        CompiledExpr::Concat(a, b) => format!("Concat({},{})", canon(a), canon(b)),
        CompiledExpr::CartesianProduct(a, b) => format!("X({},{})", canon(a), canon(b)),
        CompiledExpr::If {
            cond,
            then_branch,
            else_branch,
        } => format!(
            "If({},{},{})",
            canon(cond),
            canon(then_branch),
            canon(else_branch)
        ),
        CompiledExpr::Let { bindings, body } => {
            let bs: Vec<String> = bindings
                .iter()
                .map(|(n, v)| format!("{n}={}", canon(v)))
                .collect();
            format!("Let([{}],{})", bs.join(","), canon(body))
        }
        CompiledExpr::Exists { var, domain, body } => {
            format!("Exists({var},{},{})", canon(domain), canon(body))
        }
        CompiledExpr::Forall { var, domain, body } => {
            format!("Forall({var},{},{})", canon(domain), canon(body))
        }
        CompiledExpr::Choose { var, domain, body } => {
            format!("Choose({var},{},{})", canon(domain), canon(body))
        }
        CompiledExpr::OpCall { name, args } => format!("Op({name},[{}])", join(args)),
        CompiledExpr::FuncApply(f, args) => format!("App({},[{}])", canon(f), join(args)),
        CompiledExpr::RecordAccess(x, field) => format!("Dot({},{field})", canon(x)),
        CompiledExpr::Unparsed(s) => format!("Unparsed({:?})", norm_text(s)),
        // For any other node we fall back to normalized debug (rare in bodies).
        other => norm_text(&format!("{other:?}")),
    }
}

fn join(xs: &[CompiledExpr]) -> String {
    xs.iter().map(canon).collect::<Vec<_>>().join(",")
}

/// Collect candidate expression strings: every definition body plus every
/// invariant-shaped body from the corpus modules.
fn collect_exprs(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some("tla") {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    if let Ok(module) = parse_tla_module_text(&text) {
                        for (name, def) in &module.definitions {
                            let body = def.body.trim();
                            if body.is_empty() {
                                continue;
                            }
                            out.push((
                                format!("{}::{name}", p.file_name().unwrap().to_string_lossy()),
                                def.body.clone(),
                            ));
                        }
                    }
                }
            }
        }
    }
    out
}

fn corpus_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("EXPR_V2_CORPUS") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    for cand in [
        dirs_home().map(|h| h.join("exmpl")),
        Some(PathBuf::from("corpus")),
    ]
    .into_iter()
    .flatten()
    {
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[test]
fn shadow_compare_corpus() {
    if std::env::var("EXPR_V2_SHADOW").ok().as_deref() != Some("1") {
        eprintln!("expr_v2 shadow-compare skipped (set EXPR_V2_SHADOW=1 to run)");
        return;
    }
    let Some(root) = corpus_root() else {
        eprintln!("no corpus found (set EXPR_V2_CORPUS); skipping");
        return;
    };
    eprintln!("shadow-compare corpus root: {}", root.display());

    let exprs = collect_exprs(&root);

    // Probe mode: if EXPR_V2_PROBE_MODULE=<substr> is set, dump v1-vs-v2 for
    // every def whose label matches, then return. Used to debug a specific
    // divergence (e.g. a spec that regresses under the v2 flag).
    if let Ok(needle) = std::env::var("EXPR_V2_PROBE_MODULE") {
        for (label, body) in &exprs {
            if !label.contains(&needle) {
                continue;
            }
            let v1 = canon(&compile_expr_v1(body));
            let v2 = match expr_v2::parse_and_lower(body) {
                Ok(c) => canon(&c),
                Err(e) => format!("<PARSE-ERR: {e}>"),
            };
            let tag = if v1 == v2 { "SAME" } else { "DIFF" };
            eprintln!("[{tag}] {label}");
            eprintln!("  src: {}", body.replace('\n', "\\n"));
            eprintln!("  v1 : {v1}");
            eprintln!("  v2 : {v2}");
        }
        return;
    }

    let total = exprs.len();
    let mut agree = 0usize;
    let mut v2_err = 0usize;
    let mut differ = 0usize;
    // categorize differs
    let mut cat: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut samples: Vec<(String, String, String, String)> = Vec::new();

    for (label, body) in &exprs {
        let v1 = canon(&compile_expr_v1(body));
        match expr_v2::parse_and_lower(body) {
            Err(_) => {
                v2_err += 1;
            }
            Ok(c2) => {
                let v2 = canon(&c2);
                if v1 == v2 {
                    agree += 1;
                } else {
                    differ += 1;
                    let category = categorize(&v1, &v2);
                    *cat.entry(category).or_default() += 1;
                    if samples.len() < 40 {
                        samples.push((label.clone(), body.clone(), v1, v2));
                    }
                }
            }
        }
    }

    eprintln!("=== expr_v2 shadow-compare ===");
    eprintln!("total exprs: {total}");
    eprintln!("AGREE:       {agree}");
    eprintln!("DIFFER:      {differ}");
    eprintln!("v2 parse-err (fell back to v1): {v2_err}");
    eprintln!("--- DIFFER categories ---");
    for (k, v) in &cat {
        eprintln!("  {k}: {v}");
    }
    eprintln!("--- sample DIFFERs (up to 40) ---");
    for (label, body, v1, v2) in &samples {
        eprintln!("[{label}]");
        eprintln!("  src: {}", body.replace('\n', "\\n"));
        eprintln!("  v1 : {v1}");
        eprintln!("  v2 : {v2}");
    }

    // The harness itself never fails — it is a REPORT. (Phase 0 expects some
    // structural DIFFERs where v2 is correct; the numbers are the deliverable.)
    assert!(total > 0, "collected no expressions from corpus");
}

/// Best-effort category for a DIFFER: is v2 producing junction/implication/quant
/// structure where v1 flattened to Unparsed/And-split, or is it a leaf-shape
/// mismatch (v2 wrapped something as Atom that v1 compiled richer)?
fn categorize(v1: &str, v2: &str) -> &'static str {
    let v1_unparsed = v1.contains("Unparsed(");
    let v2_unparsed = v2.contains("Unparsed(");
    if v1_unparsed && !v2_unparsed {
        "v2-structured-where-v1-unparsed"
    } else if !v1_unparsed && v2_unparsed {
        "v2-atom-where-v1-compiled-leaf"
    } else if v1.starts_with("And[") != v2.starts_with("And[")
        || v1.starts_with("Or[") != v2.starts_with("Or[")
    {
        "junction-grouping-diff"
    } else if v1.starts_with("Implies(") != v2.starts_with("Implies(") {
        "implication-attachment-diff"
    } else {
        "other-structural-diff"
    }
}
