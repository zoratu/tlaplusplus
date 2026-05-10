//! Evaluation budget plumbing.
//!
//! Provides a thread-local `ACTIVE_EVAL_BUDGET` that newly created
//! `EvalContext`s can inherit so all evaluation on the thread shares a
//! single cumulative element limit.
//!
//! Also hosts `normalize_param_name`, the helper that strips higher-order
//! sugar like `P(_)` to bare `P` for binder/parameter resolution.

use std::cell::Cell;
use std::rc::Rc;


thread_local! {
    /// Thread-local active evaluation budget. When set, newly created EvalContexts
    /// inherit this budget so that all evaluation on the thread shares a single
    /// cumulative element limit. This avoids threading a budget parameter through
    /// every function that creates an EvalContext.
    static ACTIVE_EVAL_BUDGET: Cell<Option<Rc<Cell<usize>>>> = const { Cell::new(None) };
}

/// Set a thread-local evaluation budget that all new EvalContexts will inherit.
/// Returns the previous budget (if any) so it can be restored.
pub fn set_active_eval_budget(limit: usize) -> Option<Rc<Cell<usize>>> {
    let budget = Rc::new(Cell::new(limit));
    let prev = ACTIVE_EVAL_BUDGET.with(|b| b.replace(Some(budget.clone())));
    prev
}

/// Clear the thread-local evaluation budget, restoring an optional previous one.
pub fn restore_eval_budget(prev: Option<Rc<Cell<usize>>>) {
    ACTIVE_EVAL_BUDGET.with(|b| b.set(prev));
}

/// Get the currently active thread-local budget (if any).
pub(super) fn get_active_eval_budget() -> Option<Rc<Cell<usize>>> {
    ACTIVE_EVAL_BUDGET.with(|b| {
        let val = b.take();
        b.set(val.clone());
        val
    })
}

pub(super) const MAX_EVAL_DEPTH: usize = 256;

/// Normalize a higher-order operator parameter name.
/// TLA+ allows parameters like `P(_)` or `Op(_, _)` for higher-order operators.
/// When binding arguments to these parameters, we need just the base name.
/// E.g., "P(_)" -> "P", "Op(_, _)" -> "Op", "x" -> "x"
pub fn normalize_param_name(param: &str) -> &str {
    let param = param.trim();
    let param = if let Some(in_pos) = param.find("\\in") {
        param[..in_pos].trim()
    } else {
        param
    };
    if let Some(paren_pos) = param.find('(') {
        param[..paren_pos].trim()
    } else {
        param.trim()
    }
}
