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
        // Fix #1 (entry-stop, ROBUST single-point guard): `parse_prefix` is the
        // ONLY place a leading `/\`/`\/` bullet becomes a fresh junction
        // (`parse_junction`). Every operand-parsing entry — `parse_expr`,
        // `parse_unary`'s operand (after `~`), and the infix-bullet RHS in
        // `parse_bin` — reaches here to obtain its leading prefix/atom. So a
        // single `should_stop` check on a leading bullet here makes it
        // STRUCTURALLY IMPOSSIBLE to consume a STOP bullet (one on a new line at
        // or shallower than the active fence column, or past the barrier/a hard
        // terminator) as a junction-start, regardless of the calling path.
        //
        // Without this, `~` (parse_unary) and the infix `/\`/`\/` RHS would call
        // parse_prefix on a stop bullet and `parse_junction` would swallow it:
        //   /\ A => ~        the `/\ B` after `~` becomes `~`'s operand
        //   /\ B
        //   /\ C
        //   /\ A /\          the trailing infix `/\` RHS eats the sibling `/\ B`
        //   /\ B
        //   /\ C
        // Rejecting → v2→v1 fallback (correct) rather than a silently-wrong parse.
        if matches!(self.peek_kind(), Tok::AndBullet | Tok::OrBullet)
            && self.should_stop(stop)
        {
            return Err(format!(
                "stop bullet at fence cannot start a junction (token {:?})",
                self.peek_kind()
            ));
        }
        // Classify the leading token into an owned category first (avoids
        // binding a reference out of the scrutinee into arms that need `&mut
        // self`), then dispatch. The Phase-1 container arms are attempted
        // STRUCTURALLY only when the container is a "standalone" prefix — i.e.
        // after its balanced span the next token is a v2 boundary (a stop bullet
        // / hard terminator / EOF, or a v2-owned `=>`/`<=>`). If instead a LEAF
        // operator follows (e.g. `{1,2} \union {3}`, `<<a>>[1]`, `r.field`), v2
        // does NOT own that operator, so a structural container prefix would
        // strand it → a trailing-token error. In that case we fall through to
        // `parse_atom`, which absorbs the WHOLE expression (container + trailing
        // leaf ops) as one Atom for v1 to lower — identical to v1's own parse.
        enum Lead {
            AndB,
            OrB,
            Forall,
            Exists,
            Let,
            If,
            LParen,
            LBrace,
            LBracket,
            AngleOpen,
            Choose,
            Case,
            Atom,
        }
        let lead = match self.peek_kind() {
            Tok::AndBullet => Lead::AndB,
            Tok::OrBullet => Lead::OrB,
            Tok::Forall => Lead::Forall,
            Tok::Exists => Lead::Exists,
            Tok::Let => Lead::Let,
            Tok::If => Lead::If,
            Tok::LParen => Lead::LParen,
            Tok::LBrace => Lead::LBrace,
            Tok::LBracket => Lead::LBracket,
            Tok::Choose => Lead::Choose,
            Tok::Other(s) if s == "<<" => Lead::AngleOpen,
            Tok::Ident(w) if w == "CASE" => Lead::Case,
            _ => Lead::Atom,
        };
        match lead {
            Lead::AndB => self.parse_junction(JunctionOp::And, stop),
            Lead::OrB => self.parse_junction(JunctionOp::Or, stop),
            Lead::Forall => self.parse_quant(QuantKind::Forall, stop),
            Lead::Exists => self.parse_quant(QuantKind::Exists, stop),
            Lead::Let => self.parse_let(stop),
            Lead::If => self.parse_if(stop),
            Lead::LParen => self.parse_paren(stop),
            Lead::LBrace if self.container_is_standalone(stop, Tok::LBrace, Tok::RBrace) => {
                self.parse_set(stop)
            }
            Lead::AngleOpen if self.tuple_is_standalone(stop) => self.parse_tuple(stop),
            Lead::LBracket if self.container_is_standalone(stop, Tok::LBracket, Tok::RBracket) => {
                self.parse_bracket(stop)
            }
            Lead::Choose => self.parse_choose(stop),
            Lead::Case if self.case_is_standalone(stop) => self.parse_case(stop),
            _ => self.parse_atom(stop),
        }
    }

    // ======================= Phase 1 containers =========================

    /// Scan forward from the current `open` token to its matching `close`,
    /// tracking `()[]{}` and `<<`/`>>` depth, and return the token index just
    /// AFTER the matching close (or `None` if unbalanced). Used by the
    /// `*_is_standalone` predicates and by the container element scanners.
    fn matching_close_index(&self, open: Tok, close: Tok) -> Option<usize> {
        let mut i = self.pos;
        let mut depth: i32 = 0;
        while i < self.toks.len() {
            let k = &self.toks[i].kind;
            match k {
                Tok::Eof => return None,
                _ if *k == open => depth += 1,
                _ if *k == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                // Nested brackets of a DIFFERENT kind still need balancing so a
                // `}` inside `[..]` (or vice versa) doesn't fool us.
                Tok::LParen | Tok::LBracket | Tok::LBrace if *k != open => {
                    // balance this nested group opaquely
                    if let Some(j) = self.balanced_from(i) {
                        i = j;
                        continue;
                    } else {
                        return None;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// Return the token index just after the balanced bracket group that STARTS
    /// at token index `i` (whose kind must be one of `(`/`[`/`{`). `<<`/`>>`
    /// are also tracked as a pair. Returns `None` if unbalanced.
    fn balanced_from(&self, i: usize) -> Option<usize> {
        let mut depth: i32 = 0;
        let mut j = i;
        while j < self.toks.len() {
            match &self.toks[j].kind {
                Tok::Eof => return None,
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(j + 1);
                    }
                }
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(j + 1);
                    }
                }
                _ => {}
            }
            j += 1;
        }
        None
    }

    /// Is the token at index `idx` a v2-owned boundary — i.e. something after
    /// which a structural container prefix is complete and nothing v2 doesn't
    /// own is stranded? A boundary is: EOF, a hard terminator in `stop`, a stop
    /// bullet at the fence, or a v2-owned logical operator (`=>`/`<=>`). Any
    /// OTHER trailing token (a leaf operator, `.field`, `[`, function-apply,
    /// etc.) means the container is an OPERAND and must be absorbed as an atom.
    fn is_v2_boundary_at(&self, idx: usize, stop: &Stop) -> bool {
        // A token at/after the active barrier is a boundary: the enclosing
        // caller (a container element, a LET value) fenced the parse there, so a
        // container ending exactly at the barrier is standalone and can be
        // recursed structurally (e.g. a nested comprehension inside a set
        // element). Without this a nested container inside an element would be
        // absorbed as an atom instead of getting v2 fencing.
        if idx >= self.barrier {
            return true;
        }
        let t = &self.toks[idx.min(self.toks.len() - 1)];
        match &t.kind {
            Tok::Eof | Tok::Implies | Tok::Iff => true,
            k if stop.hard_terms.iter().any(|h| h == k) => true,
            Tok::AndBullet | Tok::OrBullet => {
                if let Some(col) = stop.bullet_fence_col {
                    t.had_newline_before && t.span.start_col <= col
                } else {
                    // No fence: a following bullet begins a NEW junction that
                    // would consume the container as its first item — not a clean
                    // boundary, so treat as non-standalone (absorb as atom).
                    false
                }
            }
            _ => false,
        }
    }

    fn container_is_standalone(&self, stop: &Stop, open: Tok, close: Tok) -> bool {
        match self.matching_close_index(open, close) {
            Some(after) => self.is_v2_boundary_at(after, stop),
            None => false,
        }
    }

    fn tuple_is_standalone(&self, stop: &Stop) -> bool {
        // Balance `<<` .. `>>` starting at the current token.
        let mut i = self.pos;
        let mut depth: i32 = 0;
        while i < self.toks.len() {
            match &self.toks[i].kind {
                Tok::Eof => return false,
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => {
                    depth -= 1;
                    if depth == 0 {
                        return self.is_v2_boundary_at(i + 1, stop);
                    }
                }
                Tok::LParen | Tok::LBracket | Tok::LBrace => {
                    if let Some(j) = self.balanced_from(i) {
                        i = j;
                        continue;
                    }
                    return false;
                }
                _ => {}
            }
            i += 1;
        }
        false
    }

    fn case_is_standalone(&self, stop: &Stop) -> bool {
        // A CASE runs from `CASE` to the enclosing v2 boundary; unlike bracketed
        // containers it has no explicit closer. Scan forward, respecting bracket
        // depth, and stop at the first depth-0 v2 boundary; a CASE is standalone
        // iff that boundary is a clean one (it always is, since the scan itself
        // ends at a boundary). We simply require that the scan finds SOME arm
        // structure (`->`) before the boundary; `parse_case` re-validates.
        let mut i = self.pos + 1; // skip CASE
        let mut depth: i32 = 0;
        let mut saw_arrow = false;
        while i < self.toks.len() {
            let t = &self.toks[i];
            match &t.kind {
                Tok::Eof => break,
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => depth -= 1,
                Tok::Other(s) if s == "->" && depth == 0 => saw_arrow = true,
                _ if depth == 0 && self.is_v2_boundary_at(i, stop) => break,
                _ => {}
            }
            i += 1;
        }
        saw_arrow
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
            // Fix #2: track SYNTACTIC PRESENCE of a param/func-arg list, not just
            // whether we collected any names. `Op() == ..` and `f[] == ..` open a
            // list but collect ZERO names, yet are STILL parameterized defs that
            // v2 cannot faithfully lower (the binder would be dropped). Keying the
            // reject off `params/func_args` emptiness lets those leak to
            // `lower_let` (a release build only `debug_assert`s). So we set a flag
            // the instant a `(`/`[` list is OPENED after the def name, and reject
            // on the flag — regardless of name count.
            let mut saw_param_list = false;
            let mut saw_func_arg_list = false;
            // Operator params `Op(a,b)`
            if matches!(self.peek_kind(), Tok::LParen) {
                saw_param_list = true;
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
                saw_func_arg_list = true;
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

            // Fix #2/#3: v2 has NO faithful lowering for a parameterized operator
            // def `Op(a,b) == ...` or a function def `f[x] == ...` — the target
            // `CompiledExpr::Let` only stores `(name, value)` with no binders, so
            // structurally lowering such a def would DROP the parameters and emit
            // a silently-WRONG CompiledExpr (`LET Op(x) == x+1 IN Op(2)` would
            // mis-resolve `Op`). The old string compiler deliberately returns
            // `Unparsed(whole LET)` for these and lets the interpreter bind
            // params at call sites. So we REJECT here → v2 falls back to v1,
            // which produces that correct Unparsed form. A safe fallback beats a
            // wrong lowering.
            //
            // Fix #2: reject on the SYNTACTIC-PRESENCE flags, not on collected
            // name counts. `Op() == 1` / `f[] == 1` open a list but collect zero
            // names — keying off `params/func_args` emptiness would leak the
            // binder past this reject into `lower_let` (which only `debug_assert`s
            // in release). The flags fire the instant a `(`/`[` list is opened.
            if saw_param_list || saw_func_arg_list || !params.is_empty() || !func_args.is_empty() {
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

    // ---- container element helpers ----

    /// Return `true` iff the token at index `i` (which must be inside the
    /// current container at bracket-depth 0 relative to the container's open)
    /// is a top-level occurrence of `needle` for the purpose of splitting.
    /// Callers scan the token range explicitly instead; this documents intent.
    ///
    /// Scan the container interior from `self.pos` (just AFTER the opener) up to
    /// (but not including) the token index `end_excl` (the closer), and return
    /// the token indices of every DEPTH-0 top-level occurrence of a token kind
    /// matched by `pred`. Depth tracks `()[]{}` and `<<`/`>>`.
    fn top_level_positions<F: Fn(&Tok) -> bool>(&self, end_excl: usize, pred: F) -> Vec<usize> {
        let mut out = Vec::new();
        let mut i = self.pos;
        let mut depth: i32 = 0;
        while i < end_excl {
            match &self.toks[i].kind {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => depth -= 1,
                k if depth == 0 && pred(k) => out.push(i),
                _ => {}
            }
            i += 1;
        }
        out
    }

    /// Parse a sub-expression occupying the token range `[self.pos, end_excl)`
    /// EXACTLY (the caller has already located the boundaries). Consumes tokens
    /// up to `end_excl`; errors if the sub-parse leaves tokens before `end_excl`.
    /// A temporary barrier at `end_excl` fences the inner `parse_expr` so any
    /// junction/`=>` inside is correctly scoped to this element.
    fn parse_subexpr_until(&mut self, end_excl: usize) -> PResult<Expr> {
        if self.pos >= end_excl {
            return Err("empty container sub-expression".to_string());
        }
        let saved = self.barrier;
        self.barrier = end_excl.min(saved);
        let inner_stop = Stop::top();
        let e = self.parse_expr(&inner_stop);
        self.barrier = saved;
        let e = e?;
        if self.pos != end_excl {
            return Err(format!(
                "container sub-expression did not consume its full range \
                 (at {:?})",
                self.peek_kind()
            ));
        }
        Ok(e)
    }

    /// Set: `{a, b, c}` (enum), `{x \in S : P}` (filter), `{e : x \in S}` (map).
    fn parse_set(&mut self, _outer: &Stop) -> PResult<Expr> {
        let open = self.advance(); // {
        let close_idx = self
            .matching_close_index_from(self.pos - 1, Tok::LBrace, Tok::RBrace)
            .ok_or_else(|| "unbalanced '{'".to_string())?;
        let closer = close_idx - 1; // index of the `}`

        // Empty set.
        if self.pos == closer {
            let close = self.advance();
            return Ok(Expr::SetEnum {
                items: Vec::new(),
                span: open.span.merge(close.span),
            });
        }

        // A top-level `:` distinguishes a comprehension from an enumeration.
        let colons = self.top_level_positions(closer, |k| matches!(k, Tok::Colon));
        if let Some(&colon_i) = colons.first() {
            // Exactly one top-level colon → comprehension; more than one is an
            // ambiguous/unsupported shape → fall back (atom).
            if colons.len() != 1 {
                return Err("multi-colon set shape unsupported by v2".to_string());
            }
            // Try FILTER form `{ x \in S : P }` — left side is a single binder.
            if let Some((var, dom_start, dom_end)) = self.match_single_in_binding(self.pos, colon_i)
            {
                // pred = tokens (colon_i, closer)
                let domain = {
                    // re-scan domain sub-range
                    self.pos = dom_start;
                    self.parse_subexpr_until(dom_end)?
                };
                // Move to just after the colon for the predicate.
                self.pos = colon_i + 1;
                let pred = self.parse_subexpr_until(closer)?;
                let close = self.advance(); // }
                return Ok(Expr::SetFilter {
                    var,
                    domain: Box::new(domain),
                    pred: Box::new(pred),
                    span: open.span.merge(close.span),
                });
            }
            // Otherwise MAP form `{ e : x \in S }` — right side is a binder.
            if let Some((var, dom_start, dom_end)) =
                self.match_single_in_binding(colon_i + 1, closer)
            {
                // body = tokens (self.pos, colon_i)
                let body = self.parse_subexpr_until(colon_i)?;
                // domain sub-range
                self.pos = dom_start;
                let domain = self.parse_subexpr_until(dom_end)?;
                debug_assert_eq!(dom_end, closer);
                self.pos = closer;
                let close = self.advance(); // }
                return Ok(Expr::SetMap {
                    var,
                    domain: Box::new(domain),
                    body: Box::new(body),
                    span: open.span.merge(close.span),
                });
            }
            // Neither single-binder form (e.g. multi-binder `{<<k,v>>: k \in K,
            // v \in V}`): v1 defers these to Unparsed, so we must NOT lower a
            // partial structure → fall back to atom.
            return Err("multi/tuple-binder comprehension unsupported by v2".to_string());
        }

        // Enumeration `{a, b, c}`: split on top-level commas.
        let commas = self.top_level_positions(closer, |k| matches!(k, Tok::Comma));
        let items = self.parse_comma_elements(closer, &commas)?;
        let close = self.advance(); // }
        Ok(Expr::SetEnum {
            items,
            span: open.span.merge(close.span),
        })
    }

    /// Tuple / sequence literal `<<a, b, c>>`.
    fn parse_tuple(&mut self, _outer: &Stop) -> PResult<Expr> {
        let open = self.advance(); // <<
        // find matching >> (depth over << >> and ()[]{})
        let closer = {
            let mut i = self.pos;
            let mut depth: i32 = 0;
            let mut found = None;
            while i < self.toks.len() {
                match &self.toks[i].kind {
                    Tok::Eof => break,
                    Tok::Other(s) if s == "<<" => depth += 1,
                    Tok::Other(s) if s == ">>" => {
                        if depth == 0 {
                            found = Some(i);
                            break;
                        }
                        depth -= 1;
                    }
                    Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                    Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            found.ok_or_else(|| "unbalanced '<<'".to_string())?
        };

        if self.pos == closer {
            let close = self.advance(); // >>
            return Ok(Expr::Tuple {
                items: Vec::new(),
                span: open.span.merge(close.span),
            });
        }
        let commas = self.top_level_positions(closer, |k| matches!(k, Tok::Comma));
        let items = self.parse_comma_elements(closer, &commas)?;
        let close = self.advance(); // >>
        Ok(Expr::Tuple {
            items,
            span: open.span.merge(close.span),
        })
    }

    /// Given the closer index and the token indices of top-level commas, parse
    /// each comma-separated element as a full sub-expression. `self.pos` must be
    /// at the first element token; on return it is at `closer`.
    fn parse_comma_elements(
        &mut self,
        closer: usize,
        commas: &[usize],
    ) -> PResult<Vec<Expr>> {
        let mut items = Vec::new();
        let mut bounds: Vec<usize> = Vec::with_capacity(commas.len() + 1);
        bounds.extend_from_slice(commas);
        bounds.push(closer);
        for &end in &bounds {
            let e = self.parse_subexpr_until(end)?;
            items.push(e);
            // Skip the comma (if we're at one).
            if self.pos < closer && matches!(self.peek_kind(), Tok::Comma) {
                self.advance();
            }
        }
        Ok(items)
    }

    /// Match a single `x \in S` binder occupying the token range `[start, end)`:
    /// an identifier, then `\in`, then a non-empty domain. Returns the var name
    /// plus the domain token range `(var, dom_start, dom_end=end)`. Returns
    /// `None` for tuple binders, multi-binders (a comma or a second `\in` in the
    /// range), or any non-single-binder shape — the caller then falls back.
    fn match_single_in_binding(&self, start: usize, end: usize) -> Option<(String, usize, usize)> {
        // Must start with an identifier.
        let name = match &self.toks[start].kind {
            Tok::Ident(n) => n.clone(),
            _ => return None,
        };
        // Next token must be `\in`.
        if start + 1 >= end || !matches!(self.toks[start + 1].kind, Tok::ElemOf) {
            return None;
        }
        let dom_start = start + 2;
        if dom_start >= end {
            return None; // empty domain
        }
        // Reject multi-binder / extra `\in` / top-level comma inside the range.
        let mut i = dom_start;
        let mut depth: i32 = 0;
        while i < end {
            match &self.toks[i].kind {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => depth -= 1,
                Tok::ElemOf | Tok::Comma if depth == 0 => return None,
                _ => {}
            }
            i += 1;
        }
        Some((name, dom_start, end))
    }

    /// Like `matching_close_index` but starting from an explicit opener index.
    fn matching_close_index_from(&self, open_idx: usize, open: Tok, close: Tok) -> Option<usize> {
        let mut i = open_idx;
        let mut depth: i32 = 0;
        while i < self.toks.len() {
            let k = &self.toks[i].kind;
            match k {
                Tok::Eof => return None,
                _ if *k == open => depth += 1,
                _ if *k == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                Tok::LParen | Tok::LBracket | Tok::LBrace if *k != open => {
                    if let Some(j) = self.balanced_from(i) {
                        i = j;
                        continue;
                    }
                    return None;
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// The `[...]` family dispatch. Decides — by looking INSIDE the brackets at
    /// top level — which of the six forms this is:
    ///   * `]_v` suffix  → action/stuttering box `[A]_v` → NOT a data bracket;
    ///     v2 does not model temporal structure, so ABSORB as atom (Err → v1).
    ///   * ` EXCEPT `    → `[f EXCEPT ![k] = v]` → absorb as atom (Err → v1):
    ///     the EXCEPT update-path grammar is intricate; v1 lowers it faithfully.
    ///   * `|->`         → record literal `[a |-> e, ...]` (structural).
    ///   * top-level `->`→ function set `[D -> R]` (structural).
    ///   * `x \in S |->` → function construct — handled under `|->` branch.
    ///   * top-level `:` → record set `[a : S, ...]` (structural).
    /// Anything else → Err → atom fallback.
    fn parse_bracket(&mut self, _outer: &Stop) -> PResult<Expr> {
        let open_idx = self.pos;
        let close_idx = self
            .matching_close_index_from(open_idx, Tok::LBracket, Tok::RBracket)
            .ok_or_else(|| "unbalanced '['".to_string())?;
        let closer = close_idx - 1; // index of `]`
        // pos is at `[`; interior is (open_idx+1 .. closer).
        self.advance(); // consume `[`
        let interior_start = self.pos;

        if interior_start == closer {
            // Empty `[]` — not a data bracket (it's the temporal box / CASE
            // separator handled elsewhere). Absorb as atom.
            return Err("empty '[]' not a data bracket".to_string());
        }

        // Action box `[A]_v`: the `]` is immediately followed by `_`. In our
        // lexer `_` is an ident-continue, so `]_v` tokenizes the `_v` as part of
        // the following ident. Detect: token right after `]` is an Ident that
        // starts with `_`, OR an `Other("_")`. Either way → temporal → atom.
        if let Some(tafter) = self.toks.get(close_idx) {
            let is_subscript = match &tafter.kind {
                Tok::Ident(s) => s.starts_with('_'),
                Tok::Other(s) => s == "_",
                _ => false,
            } && !tafter.had_newline_before
                && tafter.span.start == self.toks[closer].span.end;
            if is_subscript {
                return Err("action box [A]_v not modeled by v2".to_string());
            }
        }

        // EXCEPT keyword at top level → atom fallback.
        {
            let except = self.top_level_positions(closer, |k| match k {
                Tok::Ident(s) => s == "EXCEPT",
                _ => false,
            });
            if !except.is_empty() {
                return Err("EXCEPT bracket lowered by v1".to_string());
            }
        }

        // `|->` present at top level → record literal OR function construct.
        let mapsto = self.top_level_positions(closer, |k| matches!(k, Tok::MapsTo));
        if !mapsto.is_empty() {
            // Function construct `[x \in S |-> e]` has exactly one `|->` and a
            // single-binder left side. Otherwise a record literal.
            if mapsto.len() == 1 {
                let m = mapsto[0];
                if let Some((var, dom_start, dom_end)) =
                    self.match_single_in_binding(interior_start, m)
                {
                    self.pos = dom_start;
                    let domain = self.parse_subexpr_until(dom_end)?;
                    self.pos = m + 1;
                    let body = self.parse_subexpr_until(closer)?;
                    let close = self.advance(); // ]
                    return Ok(Expr::FuncConstruct {
                        var,
                        domain: Box::new(domain),
                        body: Box::new(body),
                        span: self.toks[open_idx].span.merge(close.span),
                    });
                }
            }
            // Record literal `[a |-> e, b |-> f]`.
            return self.parse_record_literal(open_idx, closer, &mapsto);
        }

        // Top-level `->` (not `|->`, already handled) → function set `[D -> R]`.
        let arrows = self.top_level_arrow_positions(closer);
        if let Some(&a) = arrows.first() {
            if arrows.len() != 1 {
                return Err("multi-arrow function set unsupported by v2".to_string());
            }
            let domain = self.parse_subexpr_until(a)?;
            self.pos = a + 1;
            let range = self.parse_subexpr_until(closer)?;
            let close = self.advance(); // ]
            return Ok(Expr::FunctionSet {
                domain: Box::new(domain),
                range: Box::new(range),
                span: self.toks[open_idx].span.merge(close.span),
            });
        }

        // Top-level `:` → record set `[a : S, b : T]`.
        let colons = self.top_level_positions(closer, |k| matches!(k, Tok::Colon));
        if !colons.is_empty() {
            return self.parse_record_set(open_idx, closer, &colons);
        }

        // Anything else in `[...]` (e.g. `[f]`-shaped, function apply, etc.) is
        // not a v2-modeled bracket form → absorb as atom.
        Err("unrecognized '[...]' shape for v2".to_string())
    }

    /// Top-level `->` positions inside a bracket interior, ignoring `|->`.
    fn top_level_arrow_positions(&self, end_excl: usize) -> Vec<usize> {
        // `->` tokenizes as `Tok::Other("->")`; `|->` is a distinct `Tok::MapsTo`.
        self.top_level_positions(end_excl, |k| match k {
            Tok::Other(s) => s == "->",
            _ => false,
        })
    }

    /// Record literal from the `|->` positions. Each field is
    /// `name |-> value`; the name must be a single identifier immediately
    /// before its `|->`. `self.pos` is at the first interior token; on return
    /// it is at `closer`.
    fn parse_record_literal(
        &mut self,
        open_idx: usize,
        closer: usize,
        mapsto: &[usize],
    ) -> PResult<Expr> {
        // Split the interior into comma-separated field entries, then within
        // each entry find its `|->`.
        let commas = self.top_level_positions(closer, |k| matches!(k, Tok::Comma));
        let mut entry_bounds: Vec<usize> = Vec::with_capacity(commas.len() + 1);
        entry_bounds.extend_from_slice(&commas);
        entry_bounds.push(closer);
        // Sanity: one `|->` per entry.
        if mapsto.len() != entry_bounds.len() {
            return Err("record literal field/|-> count mismatch".to_string());
        }
        let mut fields = Vec::new();
        for (idx, &entry_end) in entry_bounds.iter().enumerate() {
            let entry_start = self.pos;
            let m = mapsto[idx];
            if m <= entry_start || m >= entry_end {
                return Err("malformed record field".to_string());
            }
            // Field name: the tokens [entry_start, m) must be a single ident.
            if m - entry_start != 1 {
                return Err("record field name is not a single identifier".to_string());
            }
            let name = match &self.toks[entry_start].kind {
                Tok::Ident(n) => n.clone(),
                _ => return Err("record field name not an identifier".to_string()),
            };
            self.pos = m + 1;
            let value = self.parse_subexpr_until(entry_end)?;
            fields.push((name, value));
            if self.pos < closer && matches!(self.peek_kind(), Tok::Comma) {
                self.advance();
            }
        }
        if fields.is_empty() {
            return Err("empty record literal".to_string());
        }
        let close = self.advance(); // ]
        Ok(Expr::RecordLit {
            fields,
            span: self.toks[open_idx].span.merge(close.span),
        })
    }

    /// Record set `[a : S, b : T]` from the top-level colon positions.
    fn parse_record_set(
        &mut self,
        open_idx: usize,
        closer: usize,
        _colons: &[usize],
    ) -> PResult<Expr> {
        let commas = self.top_level_positions(closer, |k| matches!(k, Tok::Comma));
        let mut entry_bounds: Vec<usize> = Vec::with_capacity(commas.len() + 1);
        entry_bounds.extend_from_slice(&commas);
        entry_bounds.push(closer);
        let mut fields = Vec::new();
        for &entry_end in &entry_bounds {
            let entry_start = self.pos;
            // Find the colon within this entry (must be exactly one at depth 0).
            let entry_colons: Vec<usize> = self
                .top_level_positions(entry_end, |k| matches!(k, Tok::Colon))
                .into_iter()
                .filter(|&c| c >= entry_start)
                .collect();
            if entry_colons.len() != 1 {
                return Err("record set entry needs exactly one ':'".to_string());
            }
            let c = entry_colons[0];
            // Field name: single ident in [entry_start, c).
            if c - entry_start != 1 {
                return Err("record set field name not a single identifier".to_string());
            }
            let name = match &self.toks[entry_start].kind {
                Tok::Ident(n) => n.clone(),
                _ => return Err("record set field name not an identifier".to_string()),
            };
            self.pos = c + 1;
            let set_expr = self.parse_subexpr_until(entry_end)?;
            fields.push((name, set_expr));
            if self.pos < closer && matches!(self.peek_kind(), Tok::Comma) {
                self.advance();
            }
        }
        if fields.is_empty() {
            return Err("empty record set".to_string());
        }
        let close = self.advance(); // ]
        Ok(Expr::RecordSet {
            fields,
            span: self.toks[open_idx].span.merge(close.span),
        })
    }

    /// `CHOOSE x \in S : P` (bounded, single non-tuple binder) or
    /// `CHOOSE x : P` (unbounded). v1 lowers the UNBOUNDED form and any
    /// TUPLE-binder form to `Unparsed`, so v2 falls back (Err → atom) for those
    /// and only lowers the bounded single-binder `Choose`.
    fn parse_choose(&mut self, outer: &Stop) -> PResult<Expr> {
        let head = self.advance(); // CHOOSE
        // Variable name.
        let var = match self.peek_kind() {
            Tok::Ident(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            // Tuple binder `<<a,b>>` or anything else → v1's Unparsed shape.
            _ => return Err("CHOOSE non-identifier binder → v1".to_string()),
        };
        // Bounded form requires `\in Domain`.
        if !matches!(self.peek_kind(), Tok::ElemOf) {
            // Unbounded `CHOOSE x : P` — v1 returns Unparsed → fall back.
            return Err("unbounded CHOOSE → v1".to_string());
        }
        self.advance(); // \in
        let dom_stop = outer.with_hard(&[Tok::Colon]);
        let domain = self.parse_expr(&dom_stop)?;
        if !matches!(self.peek_kind(), Tok::Colon) {
            return Err("expected ':' in CHOOSE".to_string());
        }
        self.advance(); // :
        // Predicate extends with the caller's stop (maximal), like a quantifier.
        let pred = self.parse_expr(outer)?;
        let span = head.span.merge(pred.span());
        Ok(Expr::Choose {
            var,
            domain: Some(Box::new(domain)),
            pred: Box::new(pred),
            span,
        })
    }

    /// `CASE p1 -> e1 [] p2 -> e2 [] OTHER -> e3`. The `[]` is the arm
    /// separator (context-sensitive: here it is NOT the temporal box). Arms and
    /// results recurse. Lowers to `CompiledExpr::Case { arms, other }`.
    fn parse_case(&mut self, outer: &Stop) -> PResult<Expr> {
        let head = self.advance(); // CASE (an Ident)
        // Determine the CASE extent: from here to the enclosing v2 boundary at
        // depth 0.
        let end_excl = {
            let mut i = self.pos;
            let mut depth: i32 = 0;
            loop {
                if i >= self.toks.len() {
                    break i;
                }
                let t = &self.toks[i];
                match &t.kind {
                    Tok::Eof => break i,
                    Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                    Tok::RParen | Tok::RBracket | Tok::RBrace => {
                        if depth == 0 {
                            break i;
                        }
                        depth -= 1;
                    }
                    Tok::Other(s) if s == "<<" => depth += 1,
                    Tok::Other(s) if s == ">>" => depth -= 1,
                    _ if depth == 0 && self.is_v2_boundary_at(i, outer) => break i,
                    _ => {}
                }
                i += 1;
            }
        };

        // Arm separators `[]` at depth 0 within [self.pos, end_excl).
        let seps = self.top_level_positions(end_excl, |k| match k {
            Tok::Other(s) => s == "[]",
            _ => false,
        });
        // Build the arm-segment bounds.
        let mut seg_starts = vec![self.pos];
        for &s in &seps {
            seg_starts.push(s + 1);
        }
        let mut seg_ends: Vec<usize> = seps.clone();
        seg_ends.push(end_excl);

        let mut arms: Vec<(Expr, Expr)> = Vec::new();
        let mut other: Option<Box<Expr>> = None;

        for (seg_start, seg_end) in seg_starts.into_iter().zip(seg_ends.into_iter()) {
            self.pos = seg_start;
            // An arm is `guard -> result`; find the depth-0 `->` in this segment.
            let arrows: Vec<usize> = self
                .top_level_positions(seg_end, |k| match k {
                    Tok::Other(s) => s == "->",
                    _ => false,
                })
                .into_iter()
                .filter(|&a| a >= seg_start)
                .collect();
            let a = *arrows.first().ok_or_else(|| "CASE arm missing '->'".to_string())?;
            // Is the guard `OTHER`?
            let is_other = seg_start + 1 == a
                && matches!(&self.toks[seg_start].kind, Tok::Ident(s) if s == "OTHER");
            if is_other {
                self.pos = a + 1;
                let res = self.parse_subexpr_until(seg_end)?;
                other = Some(Box::new(res));
            } else {
                let guard = self.parse_subexpr_until(a)?;
                self.pos = a + 1;
                let res = self.parse_subexpr_until(seg_end)?;
                arms.push((guard, res));
            }
        }
        if arms.is_empty() && other.is_none() {
            return Err("empty CASE".to_string());
        }
        self.pos = end_excl;
        let span = head.span.merge(self.toks[end_excl.saturating_sub(1)].span);
        Ok(Expr::Case { arms, other, span })
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
            // inner operator inside the atom; also parenthesized groups). `<<`
            // and `>>` are ALSO tracked as a pair so a tuple `<<a, b>>` absorbed
            // into an atom (when it is an operand, not a standalone prefix) keeps
            // its interior commas from splitting the atom — v1 lowers the whole
            // `SeqLiteral`.
            match &t.kind {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    if depth == 0 {
                        break; // unbalanced closer belongs to an outer group
                    }
                    depth -= 1;
                }
                Tok::Other(s) if s == "<<" => depth += 1,
                Tok::Other(s) if s == ">>" => {
                    if depth > 0 {
                        depth -= 1;
                    }
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
            // Fix #3 + Codex carry-over: a block comment may span newlines.
            // Collapsing it to a single space would DESTROY line structure — the
            // Atom text handed to v1 would have fewer lines than the source the
            // lexer saw, so v1's own layout-sensitive parsing could interpret the
            // (now single-line) remainder differently. AND collapsing the final
            // line to a single space would shift the COLUMN of the next token on
            // that line, which is equally layout-breaking. So we replace the
            // comment with whitespace that preserves BOTH: emit the interior
            // newlines, then pad the final-line portion with as many spaces as
            // the comment occupied on its LAST line (its own char WIDTH from the
            // `(*` opener — or from column 0 after an interior newline — through
            // the `*)` closer, inclusive). Because the text BEFORE the comment is
            // copied verbatim, emitting the comment's own width keeps every token
            // AFTER the comment at its original column. String-literal contents
            // are left untouched (handled above; the scan never enters strings).
            let mut newlines = 0usize;
            // Char-width the comment occupies on its FINAL line. Seeded with the
            // 2-col `(*` opener; reset to 0 at each interior newline.
            let mut final_line_cols = 2usize;
            while i < b.len() && depth > 0 {
                if b[i] == b'(' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                    final_line_cols += 2;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b')' {
                    depth -= 1;
                    i += 2;
                    final_line_cols += 2;
                } else {
                    if b[i] == b'\n' {
                        newlines += 1;
                        final_line_cols = 0; // column resets on a newline
                    } else if (b[i] & 0xC0) != 0x80 {
                        final_line_cols += 1; // count chars, not bytes
                    }
                    i += 1;
                }
            }
            for _ in 0..newlines {
                out.push(b'\n');
            }
            // Pad the final line to the exact column width the comment occupied so
            // the next token keeps its source column.
            for _ in 0..final_line_cols {
                out.push(b' ');
            }
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

/// Test-only accessor for the comment stripper (column-preservation golden).
#[cfg(test)]
pub(crate) fn strip_comments_for_test(s: &str) -> String {
    strip_comments(s)
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
