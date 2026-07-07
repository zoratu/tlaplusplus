//! Hand-written recursive-descent + Pratt parser for `expr_v2`.
//!
//! Owns a *layout-fence* stack encoded in the [`Stop`] passed down the parse.
//! The critical rule lives in [`Parser::should_stop`]: a `/\`/`\/` bullet that
//! begins a new line AND sits at or to the LEFT of the current junction's bullet
//! column terminates the current sub-expression (it belongs to an enclosing
//! junction). Body-extending forms (`=>`, `\A x : body`, `LET .. IN body`,
//! `IF .. THEN .. ELSE ..`) parse their bodies with the CALLER's `Stop`, so they
//! extend maximally; a junction ITEM installs a stricter fence at its own bullet
//! column.
//!
//! Everything the grammar does not model structurally is gathered into an
//! `Atom` run of leaf tokens (reconstructed verbatim from the source span) and
//! later lowered via the existing `compile_expr`.

use super::ast::*;
use super::lexer::{Tok, Token};

/// A stop condition threaded through the Pratt parser.
#[derive(Debug, Clone)]
pub struct Stop {
    /// Hard terminator token kinds: parsing stops *before* consuming any of
    /// these (used for `)`, `THEN`, `ELSE`, `IN`, `:`, `,`, `Eof`, etc.).
    hard_terms: Vec<Tok>,
    /// If set, a same-or-more-shallow bullet on a new line terminates.
    bullet_fence_col: Option<u32>,
}

impl Stop {
    fn top() -> Self {
        Stop {
            hard_terms: vec![Tok::Eof],
            bullet_fence_col: None,
        }
    }

    fn with_hard(&self, extra: &[Tok]) -> Self {
        let mut hard = self.hard_terms.clone();
        for e in extra {
            if !hard.contains(e) {
                hard.push(e.clone());
            }
        }
        Stop {
            hard_terms: hard,
            bullet_fence_col: self.bullet_fence_col,
        }
    }

    fn with_bullet_fence(&self, col: u32) -> Self {
        Stop {
            hard_terms: self.hard_terms.clone(),
            bullet_fence_col: Some(col),
        }
    }
}

pub struct Parser<'a> {
    src: &'a str,
    toks: Vec<Token>,
    pos: usize,
    /// A hard token-index barrier: no parse routine may consume tokens at or
    /// beyond this index while it is set. Used to bound a LET binding value to
    /// the start of the next binding without heuristics. `usize::MAX` = none.
    barrier: usize,
}

type PResult<T> = Result<T, String>;

impl<'a> Parser<'a> {
    pub fn new(src: &'a str, toks: Vec<Token>) -> Self {
        Parser {
            src,
            toks,
            pos: 0,
            barrier: usize::MAX,
        }
    }

    fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn peek_kind(&self) -> &Tok {
        &self.peek().kind
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), Tok::Eof)
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    /// Reconstruct the source text for a byte range [start, end).
    fn text(&self, start: usize, end: usize) -> String {
        self.src[start..end].to_string()
    }

    // -- the STOP predicate --------------------------------------------------

    fn at_barrier(&self) -> bool {
        self.pos >= self.barrier
    }

    fn should_stop(&self, stop: &Stop) -> bool {
        if self.at_barrier() {
            return true;
        }
        let t = self.peek();
        if stop.hard_terms.iter().any(|k| k == &t.kind) {
            return true;
        }
        if let Some(col) = stop.bullet_fence_col {
            if matches!(t.kind, Tok::AndBullet | Tok::OrBullet)
                && t.had_newline_before
                && t.span.start_col <= col
            {
                return true;
            }
        }
        false
    }

    // -- entry ---------------------------------------------------------------

    pub fn parse_expr_top(&mut self) -> PResult<Expr> {
        let e = self.parse_expr(&Stop::top())?;
        if !self.at_eof() {
            return Err(format!(
                "trailing tokens after expr near {:?}",
                self.peek_kind()
            ));
        }
        Ok(e)
    }

    /// Pratt entry: lowest-precedence binding starts here.
    ///
    /// Fix #1 (entry-stop): before descending into `parse_bin`/`parse_prefix` we
    /// must honor the active fence. If the very next token already satisfies the
    /// caller's `Stop` (a hard terminator, the barrier, OR a new-line bullet at or
    /// shallower than the fence column), then this sub-expression is EMPTY — the
    /// bullet/terminator belongs to an ENCLOSING construct. Descending anyway
    /// would (a) swallow a sibling/enclosing bullet as a fresh junction, or (b)
    /// collapse an empty junction item to its neighbor. Both are wrong parses, so
    /// we return an error which triggers the v2→v1 fallback rather than producing
    /// a silently-wrong `CompiledExpr`.
    fn parse_expr(&mut self, stop: &Stop) -> PResult<Expr> {
        if self.should_stop(stop) {
            return Err(format!(
                "empty sub-expression at fence (token {:?})",
                self.peek_kind()
            ));
        }
        self.parse_bin(stop, 0)
    }

    // Precedence levels (low -> high):
    //  0: <=> / =>        (right-assoc, body-extending)
    //  1: \/ (infix disjunction, rare — junctions usually prefix)
    //  2: /\ (infix conjunction)
    //  3: comparisons / set-membership  (= # < > <= >= \in \notin)
    //  4: set ops (\union \intersect \ \o) and additive (+ -)
    //  5: multiplicative (* \div % ^)
    // function application / .field / f[x] handled in parse_postfix (highest)
    fn bin_op_at(&self, level: u32) -> Option<(BinOp, bool /*right assoc*/)> {
        let k = self.peek_kind();
        match level {
            // v2 owns ONLY the top-level body-extending logical operators.
            // Comparisons / arithmetic / set operators are LEFT INSIDE the atom
            // and lowered by v1 (so their precedence — incl. `..` vs `+` — is
            // exactly the old parser's). See the atom-reader comment for why.
            0 => match k {
                Tok::Iff => Some((BinOp::Iff, true)),
                Tok::Implies => Some((BinOp::Implies, true)),
                _ => None,
            },
            _ => None,
        }
    }

    // Levels: 0 = <=>/=>, 1 = \/, 2 = /\ (junction levels), 3+ = leaf operators
    // handled inside atoms by v1. We still descend through 3..=MAX so the
    // recursion bottoms out at parse_unary.
    const MAX_BIN_LEVEL: u32 = 2;

    fn parse_bin(&mut self, stop: &Stop, level: u32) -> PResult<Expr> {
        if level > Self::MAX_BIN_LEVEL {
            return self.parse_unary(stop);
        }
        // Levels 1 (\/) and 2 (/\) are the junction levels. A leading bullet is
        // handled by prefix parsing (parse_junction); here we handle the INFIX
        // form `A \/ B` / `A /\ B` (operands separated by a bullet mid-
        // expression, e.g. inside a quantifier body `Phase1a \/ Phase2a`). We
        // collect a flat list and build a Junction so it matches the prefix
        // form's shape.
        if level == 1 || level == 2 {
            let (bullet, jop) = if level == 1 {
                (Tok::OrBullet, JunctionOp::Or)
            } else {
                (Tok::AndBullet, JunctionOp::And)
            };
            let first = self.parse_bin(stop, level + 1)?;
            // If the next token is the infix bullet (and the fence doesn't stop
            // us), gather a disjunction/conjunction list.
            if !self.should_stop(stop) && self.peek_kind() == &bullet {
                let mut items = vec![first];
                let mut span = items[0].span();
                let col = self.peek().span.start_col;
                while !self.should_stop(stop) && self.peek_kind() == &bullet {
                    self.advance(); // bullet
                    let it = self.parse_bin(stop, level + 1)?;
                    span = span.merge(it.span());
                    items.push(it);
                }
                // Flatten: if any item is itself a same-op Junction, splice it in
                // (keeps `A \/ (B \/ C)` and prefix/infix mixes flat).
                let mut flat = Vec::new();
                for it in items {
                    match it {
                        Expr::Junction { op, items: inner, .. } if op == jop => {
                            flat.extend(inner)
                        }
                        other => flat.push(other),
                    }
                }
                return Ok(Expr::Junction {
                    op: jop,
                    col,
                    items: flat,
                    span,
                });
            }
            return Ok(first);
        }
        let mut lhs = self.parse_bin(stop, level + 1)?;

        loop {
            if self.should_stop(stop) {
                break;
            }
            let Some((op, right_assoc)) = self.bin_op_at(level) else {
                break;
            };
            self.advance(); // operator
            // Fix #1 (entry-stop): a body-extending operator (`=>`/`<=>`) whose
            // consequent would be EMPTY — i.e. the next token already satisfies
            // the caller's fence (a sibling/enclosing bullet at or shallower than
            // the fence column, or a hard terminator) — is a WRONG parse. E.g.
            //   /\ A =>
            //   /\ B
            //   /\ C
            // the `=>` consequent must stop at the second `/\` (same fence
            // column); parsing an RHS here would swallow B and C. Reject → v1
            // fallback rather than silently mis-attach.
            if self.should_stop(stop) {
                return Err(format!(
                    "empty consequent for {:?} at fence (token {:?})",
                    op,
                    self.peek_kind()
                ));
            }
            // Body-extending operators (=> / <=>) parse RHS with the CALLER's
            // stop so they extend maximally down/right (this is what pulls a
            // following LET-with-junction into the consequent).
            let rhs = if right_assoc {
                // right-assoc: recurse at the SAME level for chaining.
                self.parse_bin(stop, level)?
            } else {
                self.parse_bin(stop, level + 1)?
            };
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
            if !right_assoc {
                continue;
            } else {
                // right-assoc handled by the recursion above; stop the loop.
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self, stop: &Stop) -> PResult<Expr> {
        if matches!(self.peek_kind(), Tok::Not) {
            let start_tok = self.advance();
            let operand = self.parse_unary(stop)?;
            let span = start_tok.span.merge(operand.span());
            return Ok(Expr::Not {
                operand: Box::new(operand),
                span,
            });
        }
        self.parse_prefix(stop)
    }

    // -- prefix / primary ----------------------------------------------------

    fn parse_prefix(&mut self, stop: &Stop) -> PResult<Expr> {
        match self.peek_kind() {
            Tok::AndBullet => self.parse_junction(JunctionOp::And, stop),
            Tok::OrBullet => self.parse_junction(JunctionOp::Or, stop),
            Tok::Forall => self.parse_quant(QuantKind::Forall, stop),
            Tok::Exists => self.parse_quant(QuantKind::Exists, stop),
            Tok::Let => self.parse_let(stop),
            Tok::If => self.parse_if(stop),
            Tok::LParen => self.parse_paren(stop),
            _ => self.parse_atom(stop),
        }
    }

    /// Parse a bulleted junction: bullets aligned at the column of the FIRST
    /// bullet, each item parsed under a stricter fence at that column.
    fn parse_junction(&mut self, op: JunctionOp, outer: &Stop) -> PResult<Expr> {
        let bullet_tok_kind = match op {
            JunctionOp::And => Tok::AndBullet,
            JunctionOp::Or => Tok::OrBullet,
        };
        let first = self.peek().clone();
        let col = first.span.start_col;
        let item_stop = outer.with_bullet_fence(col);

        let mut items = Vec::new();
        let mut full_span = first.span;

        loop {
            // Expect the bullet at this column.
            if self.peek_kind() != &bullet_tok_kind {
                break;
            }
            let this = self.peek().clone();
            // For continuation bullets (not the first): accept either an aligned
            // bullet on a NEW line (the layout list form) OR an inline bullet on
            // the SAME line (infix conjunction `/\ A /\ B`). A new-line bullet at
            // a DIFFERENT column belongs to an enclosing/inner list and ends this
            // one — but the fence in `should_stop` already terminates item
            // parsing before a shallower bullet, so here we only need to reject a
            // new-line bullet that is not aligned to this column.
            if !items.is_empty() && this.had_newline_before && this.span.start_col != col {
                break;
            }
            self.advance(); // consume bullet
            let item = self.parse_expr(&item_stop)?;
            full_span = full_span.merge(item.span());
            items.push(item);
        }

        // A single-item "junction" is just its item (e.g. `/\ X` alone). But we
        // keep the wrapper only if there are >=2 items; a lone bullet lowers to
        // the item directly to match old-parser semantics.
        if items.len() == 1 {
            return Ok(items.pop().unwrap());
        }
        Ok(Expr::Junction {
            op,
            col,
            items,
            span: full_span,
        })
    }

    /// `\A x \in S : body` / `\A x, y \in S, z \in T : body`.
    fn parse_quant(&mut self, kind: QuantKind, outer: &Stop) -> PResult<Expr> {
        let head = self.advance(); // \A / \E
        let mut bounds = Vec::new();

        // Parse comma-separated bounds until `:`.
        loop {
            // Collect variable names until `\in` or `:` or (for unbounded) `:`.
            let mut vars = Vec::new();
            loop {
                match self.peek_kind() {
                    Tok::Ident(name) => {
                        vars.push(name.clone());
                        self.advance();
                        if matches!(self.peek_kind(), Tok::Comma) {
                            // could be more vars for the same domain OR next bound
                            // Peek past the comma: if the token after is an ident
                            // and then `\in`/`,` we keep collecting into vars.
                            self.advance();
                            continue;
                        } else {
                            break;
                        }
                    }
                    Tok::Other(s) if s == "<<" => {
                        // tuple binder like <<a,b>> — treat opaquely: gather to \in
                        let start = self.peek().span.start;
                        while !matches!(self.peek_kind(), Tok::ElemOf | Tok::Colon | Tok::Eof)
                        {
                            self.advance();
                        }
                        let end = self.peek().span.start;
                        vars.push(self.text(start, end).trim().to_string());
                        break;
                    }
                    _ => break,
                }
            }
            // Domain, if bounded.
            let domain = if matches!(self.peek_kind(), Tok::ElemOf) {
                self.advance(); // \in
                let dom_stop = Stop::top().with_hard(&[Tok::Colon, Tok::Comma]);
                let d = self.parse_expr(&dom_stop)?;
                Box::new(d)
            } else {
                // Unbounded quantifier `\A x : body`.
                Box::new(Expr::Atom {
                    text: "__UNBOUNDED__".to_string(),
                    span: head.span,
                })
            };
            bounds.push(QuantBound { vars, domain });

            if matches!(self.peek_kind(), Tok::Comma) {
                self.advance();
                continue;
            }
            break;
        }

        // Expect `:`
        if !matches!(self.peek_kind(), Tok::Colon) {
            return Err(format!(
                "expected ':' in quantifier, found {:?}",
                self.peek_kind()
            ));
        }
        self.advance(); // :

        // Body extends with the CALLER's stop (maximal).
        let body = self.parse_expr(outer)?;
        let span = head.span.merge(body.span());
        Ok(Expr::Quant {
            kind,
            bounds,
            body: Box::new(body),
            span,
        })
    }

    /// `LET def1 def2 ... IN body`, parsed STRUCTURALLY to the matching `IN`.
    fn parse_let(&mut self, outer: &Stop) -> PResult<Expr> {
        let head = self.advance(); // LET
        let mut defs = Vec::new();

        loop {
            if matches!(self.peek_kind(), Tok::In) {
                break;
            }
            if self.at_eof() {
                return Err("unterminated LET (no IN)".to_string());
            }
            // A definition: Name [ (params) | [args] ] == value
            let name = match self.peek_kind() {
                Tok::Ident(n) => {
                    let n = n.clone();
                    self.advance();
                    n
                }
                other => return Err(format!("expected LET binding name, found {:?}", other)),
            };
            let mut params = Vec::new();
            let mut func_args = Vec::new();
            // Operator params `Op(a,b)`
            if matches!(self.peek_kind(), Tok::LParen) {
                self.advance();
                loop {
                    match self.peek_kind() {
                        Tok::Ident(p) => {
                            params.push(p.clone());
                            self.advance();
                        }
                        Tok::Comma => {
                            self.advance();
                        }
                        Tok::RParen => {
                            self.advance();
                            break;
                        }
                        _ => {
                            // opaque param shape: skip to matching ')'
                            let mut depth = 1;
                            while depth > 0 && !self.at_eof() {
                                match self.peek_kind() {
                                    Tok::LParen => depth += 1,
                                    Tok::RParen => depth -= 1,
                                    _ => {}
                                }
                                self.advance();
                            }
                            break;
                        }
                    }
                }
            }
            // Function args `f[x]`
            if matches!(self.peek_kind(), Tok::LBracket) {
                self.advance();
                loop {
                    match self.peek_kind() {
                        Tok::Ident(p) => {
                            func_args.push(p.clone());
                            self.advance();
                        }
                        Tok::ElemOf => {
                            // `f[x \in S]` — skip the domain part opaquely to ]
                            self.advance();
                            while !matches!(
                                self.peek_kind(),
                                Tok::RBracket | Tok::Comma | Tok::Eof
                            ) {
                                self.advance();
                            }
                        }
                        Tok::Comma => {
                            self.advance();
                        }
                        Tok::RBracket => {
                            self.advance();
                            break;
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
            }
            // Expect `==`
            if !matches!(self.peek_kind(), Tok::EqEq) {
                return Err(format!(
                    "expected '==' in LET binding, found {:?}",
                    self.peek_kind()
                ));
            }
            self.advance(); // ==

            // Value: parse structurally, bounded by a token barrier at the START
            // of the next binding (or the matching IN). This avoids heuristics:
            // we scan forward for the next top-level `Ident (…|[…])? ==` or `IN`
            // and hard-stop the value parse there.
            let value_barrier = self.scan_next_binding_or_in();
            let saved = self.barrier;
            self.barrier = value_barrier.min(saved);
            let value_stop = Stop::top().with_hard(&[Tok::In]);
            let value = self.parse_expr(&value_stop)?;
            self.barrier = saved;

            // Fix #3: v2 has NO faithful lowering for a parameterized operator
            // def `Op(a,b) == ...` or a function def `f[x] == ...` — the target
            // `CompiledExpr::Let` only stores `(name, value)` with no binders, so
            // structurally lowering such a def would DROP the parameters and emit
            // a silently-WRONG CompiledExpr (`LET Op(x) == x+1 IN Op(2)` would
            // mis-resolve `Op`). The old string compiler deliberately returns
            // `Unparsed(whole LET)` for these and lets the interpreter bind
            // params at call sites. So we REJECT here → v2 falls back to v1,
            // which produces that correct Unparsed form. A safe fallback beats a
            // wrong lowering.
            if !params.is_empty() || !func_args.is_empty() {
                return Err(format!(
                    "parameterized/function LET binding '{}' not supported by v2 \
                     (falling back to v1)",
                    name
                ));
            }

            defs.push(LetDef {
                name,
                params,
                func_args,
                value: Box::new(value),
            });
        }

        // consume IN
        if !matches!(self.peek_kind(), Tok::In) {
            return Err("expected IN in LET".to_string());
        }
        self.advance();

        // body extends with caller's stop (maximal).
        let body = self.parse_expr(outer)?;
        let span = head.span.merge(body.span());
        Ok(Expr::Let {
            defs,
            body: Box::new(body),
            span,
        })
    }

    /// Scan forward from the current position (just after a binding's `==`) for
    /// the token index that ends this binding's value: either the start of the
    /// NEXT binding (`Ident (…|[…])? ==`) or the matching top-level `IN`. Returns
    /// the token index of that boundary (exclusive barrier for the value parse).
    /// Depth-aware so nested `LET … IN …` / bracketed `==` don't fool it.
    fn scan_next_binding_or_in(&self) -> usize {
        let mut i = self.pos;
        let mut depth: i32 = 0;
        let mut let_depth: i32 = 0;
        while i < self.toks.len() {
            match &self.toks[i].kind {
                Tok::Eof => return i,
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Let => {
                    let_depth += 1;
                }
                Tok::In => {
                    if depth == 0 && let_depth == 0 {
                        return i; // our LET's IN
                    }
                    if let_depth > 0 {
                        let_depth -= 1;
                    }
                }
                Tok::Ident(_) if depth == 0 && let_depth == 0 => {
                    // Possible next-binding start: Ident (params)? [args]? ==
                    let mut j = i + 1;
                    // skip a balanced (...) param list
                    if j < self.toks.len() && matches!(self.toks[j].kind, Tok::LParen) {
                        let mut d = 0;
                        while j < self.toks.len() {
                            match self.toks[j].kind {
                                Tok::LParen => d += 1,
                                Tok::RParen => {
                                    d -= 1;
                                    if d == 0 {
                                        j += 1;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                    }
                    // skip a balanced [...] func-arg list
                    if j < self.toks.len() && matches!(self.toks[j].kind, Tok::LBracket) {
                        let mut d = 0;
                        while j < self.toks.len() {
                            match self.toks[j].kind {
                                Tok::LBracket => d += 1,
                                Tok::RBracket => {
                                    d -= 1;
                                    if d == 0 {
                                        j += 1;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                    }
                    if j < self.toks.len() && matches!(self.toks[j].kind, Tok::EqEq) {
                        // But this is only a NEW binding if it is NOT the very
                        // first token of the value (i > self.pos).
                        if i > self.pos {
                            return i;
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
        i
    }

    /// `IF cond THEN a ELSE b`. Bodies extend with the caller's stop.
    ///
    /// Fix #2: thread the caller's `outer` Stop through every branch so the
    /// enclosing bullet fence is preserved. Using `Stop::top()` here would LOSE
    /// the fence, letting a sibling bullet that follows THEN/ELSE be swallowed
    /// into the branch. We only ADD the branch-delimiter hard terminators
    /// (`THEN`/`ELSE`) on top of `outer`; the else-branch keeps `outer` as-is.
    fn parse_if(&mut self, outer: &Stop) -> PResult<Expr> {
        let head = self.advance(); // IF
        let cond_stop = outer.with_hard(&[Tok::Then]);
        let cond = self.parse_expr(&cond_stop)?;
        if !matches!(self.peek_kind(), Tok::Then) {
            return Err("expected THEN in IF".to_string());
        }
        self.advance();
        let then_stop = outer.with_hard(&[Tok::Else]);
        let then_ = self.parse_expr(&then_stop)?;
        if !matches!(self.peek_kind(), Tok::Else) {
            return Err("expected ELSE in IF".to_string());
        }
        self.advance();
        // else-branch extends with the caller's fence intact (Fix #2).
        let else_ = self.parse_expr(outer)?;
        let span = head.span.merge(else_.span());
        Ok(Expr::If {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
            span,
        })
    }

    fn parse_paren(&mut self, _outer: &Stop) -> PResult<Expr> {
        let open = self.advance(); // (
        let inner_stop = Stop::top().with_hard(&[Tok::RParen]);
        let inner = self.parse_expr(&inner_stop)?;
        if !matches!(self.peek_kind(), Tok::RParen) {
            return Err("expected ')'".to_string());
        }
        let close = self.advance();
        Ok(Expr::Paren {
            inner: Box::new(inner),
            span: open.span.merge(close.span),
        })
    }

    /// Parse an opaque atom: gather a run of leaf tokens (no structural keyword,
    /// no top-level junction/`=>`/`<=>` at the current fence) into a verbatim
    /// text slice, then wrap in `Atom`.
    ///
    /// This is where the "structural boundary" is enforced: we stop the atom run
    /// at any token that a higher grammar level owns (bullets under the fence,
    /// `=>`, `<=>`, `THEN`/`ELSE`/`IN`/`:` when in the stop set, quantifier /
    /// LET / IF keywords, and comparison/set/arith operators handled by the
    /// Pratt layer — those must NOT be swallowed into the atom).
    fn parse_atom(&mut self, stop: &Stop) -> PResult<Expr> {
        let start_tok = self.peek().clone();
        let start = start_tok.span.start;
        let mut end = start;
        let mut consumed = false;
        // bracket/paren depth so an operator INSIDE brackets stays in the atom.
        let mut depth: i32 = 0;

        while !self.at_eof() {
            if depth == 0 && self.at_barrier() {
                break;
            }
            let t = self.peek();
            // Respect the stop fence (bullets / hard terms) only at depth 0.
            if depth == 0 && self.should_stop(stop) {
                break;
            }
            // At depth 0, ONLY the structural tokens v2 truly owns end an atom.
            // Deliberately NOT included: comparison / arithmetic / set / membership
            // operators (= # < > <= >= \in \notin + - * \div % ^ \union \intersect
            // \ \o) and the range `..`. Those are LEAF operators — v1's compiler
            // already gets their precedence right (e.g. `..` binds tighter than
            // `+` in `0..MaxSegmentId+1`), whereas re-deriving it in v2 introduced
            // precedence bugs. Keeping them inside the atom means v1 lowers them,
            // so leaf semantics stay byte-for-byte identical. v2 owns only
            // junctions, `=>`/`<=>`, and the quantifier/LET/IF keyword structure.
            if depth == 0 && consumed {
                match &t.kind {
                    // body-extending logical structure v2 owns
                    Tok::Implies
                    | Tok::Iff
                    // structural keywords / bullets
                    | Tok::AndBullet
                    | Tok::OrBullet
                    | Tok::Forall
                    | Tok::Exists
                    | Tok::Let
                    | Tok::If
                    | Tok::Then
                    | Tok::Else
                    | Tok::In
                    | Tok::Colon
                    | Tok::Comma
                    | Tok::EqEq => break,
                    _ => {}
                }
            }
            // Track bracket depth (so `f[a => b]`—rare—or `Op(x => y)` keep the
            // inner operator inside the atom; also parenthesized groups).
            match &t.kind {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    if depth == 0 {
                        break; // unbalanced closer belongs to an outer group
                    }
                    depth -= 1;
                }
                _ => {}
            }
            let tk = self.advance();
            end = tk.span.end;
            consumed = true;
        }

        if !consumed {
            return Err(format!(
                "empty atom at {:?} (unexpected token)",
                self.peek_kind()
            ));
        }
        // Fix #4: the atom text is reconstructed from the ORIGINAL byte slice, so
        // any `\*`/`(* *)` comment the lexer already stripped would otherwise be
        // reintroduced (e.g. `x (* c *) + y` would reach v1 WITH the comment).
        // Strip comments from the materialized slice (replacing each with a single
        // space to preserve token separation) so v1 lowers clean text.
        let text = strip_comments(&self.text(start, end));
        Ok(Expr::Atom {
            text,
            span: start_tok.span.merge(self.toks[self.pos.saturating_sub(1)].span),
        })
    }
}

/// Strip TLA+ comments from an atom text slice: `\* ...` to end-of-line and
/// nested `(* ... *)`. Each comment is replaced by a single space so adjacent
/// tokens don't fuse (e.g. `a(* c *)b` -> `a b`). Comment markers inside a
/// double-quoted string literal are left untouched. Mirrors the lexer's trivia
/// rules so the text handed to `compile_expr_v1` matches what v2 tokenized.
fn strip_comments(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    let mut i = 0usize;
    while i < b.len() {
        // String literal: copy verbatim (with `\"` escapes) — no comment scan.
        if b[i] == b'"' {
            out.push(b'"');
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' && i + 1 < b.len() {
                    out.push(b[i]);
                    out.push(b[i + 1]);
                    i += 2;
                    continue;
                }
                out.push(b[i]);
                let done = b[i] == b'"';
                i += 1;
                if done {
                    break;
                }
            }
            continue;
        }
        // Line comment `\* ... ` to end of line (keep the newline).
        if b[i] == b'\\' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            out.push(b' ');
            continue;
        }
        // Nested block comment `(* ... *)`.
        if b[i] == b'(' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i < b.len() && depth > 0 {
                if b[i] == b'(' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b')' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            out.push(b' ');
            continue;
        }
        // Regular byte: copy verbatim (preserves multi-byte UTF-8 unchanged; we
        // only special-case ASCII comment/string markers above).
        out.push(b[i]);
        i += 1;
    }
    // `out` is `s` with whole ASCII-delimited comment regions removed, so it
    // remains valid UTF-8.
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

/// Parse a source string into an [`Expr`] AST.
pub fn parse(src: &str) -> PResult<Expr> {
    let toks = super::lexer::tokenize(src);
    // Fix #5: a lex error (e.g. unterminated block comment) invalidates the whole
    // token stream — bail so the caller falls back to v1 rather than parsing a
    // truncated prefix.
    if let Some(t) = toks.iter().find(|t| matches!(t.kind, Tok::LexError(_))) {
        if let Tok::LexError(msg) = &t.kind {
            return Err(format!("lex error: {msg}"));
        }
    }
    let mut p = Parser::new(src, toks);
    p.parse_expr_top()
}
