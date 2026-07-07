//! Hand-written lexer for the `expr_v2` TLA+ expression parser.
//!
//! Responsibilities:
//! - Strip `\*` line comments and `(* ... *)` block comments (nested), WITHOUT
//!   perturbing the line/column of real tokens (comments are replaced by
//!   whitespace-equivalent skipping; we track byte position + line/col as we go).
//! - Emit [`Token`]s with a [`Span`] (byte offsets + line/col) and a
//!   `had_newline_before` flag (true if a newline appeared between the previous
//!   token and this one) — the layout fence needs this to distinguish a bullet
//!   that starts a new line from one mid-line.
//! - Longest-match tokenization of the multi-char operators the parser cares
//!   about (`/\`, `\/`, `=>`, `<=>`, `|->`, `[]`, `<>`, `<<`, `>>`, `->`, `..`,
//!   `::`, `:>`, `@@`, `/=`, `<=`, `>=`, `\in`, `\notin`, `\A`, `\E`, ...).
//!
//! Anything the lexer does not recognize as a structural token is emitted as an
//! `Ident`/`Number`/`Str`/`Other` leaf token carrying its raw text; the parser
//! assembles runs of leaf tokens into `Atom` sub-expressions.

use super::ast::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tok {
    // Layout junction bullets
    AndBullet, // /\  (and \land)
    OrBullet,  // \/  (and \lor)

    // Logical
    Implies, // =>
    Iff,     // <=> and \equiv
    Not,     // ~  and \lnot / \neg

    // Quantifiers / binders
    Forall, // \A  and \forall
    Exists, // \E  and \exists
    Choose, // CHOOSE (kept as keyword; parser treats structurally-lite)

    // Keywords
    Let,
    In,
    If,
    Then,
    Else,

    // Punctuation the parser threads structurally
    LParen,
    RParen,
    LBracket, // [
    RBracket, // ]
    LBrace,   // {
    RBrace,   // }
    Colon,    // :
    Comma,    // ,
    EqEq,     // ==  (definition)
    MapsTo,   // |->

    // Comparison / membership leaf-ish operators (still surfaced so the Pratt
    // layer can build Binary nodes for the ones we lower structurally).
    Eq,     // =
    Neq,    // #  or  /=
    Lt,     // <
    Gt,     // >
    Le,     // <=  or  =<  or \leq
    Ge,     // >=  or \geq
    ElemOf, // \in
    NotIn,  // \notin

    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash, // /  (division; also part of /\ handled first)
    Div,   // \div
    Mod,   // %
    Caret, // ^

    // Set / seq
    Union,     // \union / \cup
    Intersect, // \intersect / \cap
    Backslash, // \  (set minus) — only when NOT followed by an ident (\in etc.)
    Concat,    // \o / \circ

    // Leaf atoms
    Ident(String),
    Number(String),
    Str(String),

    /// Any single character / token the parser treats opaquely (kept so an Atom
    /// run can be reconstructed verbatim from the source span).
    Other(String),

    /// A lexing error (e.g. an unterminated `(* ... *)` block comment). Emitted
    /// in place of further tokens so any parse consuming it fails → v1 fallback,
    /// rather than silently truncating the input. Carries a diagnostic message.
    LexError(String),

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: Tok,
    pub span: Span,
    pub had_newline_before: bool,
}

pub struct Lexer<'a> {
    src: &'a [u8],
    /// Byte position.
    pos: usize,
    /// 1-based current line.
    line: u32,
    /// 0-based current column (in chars — TLA+ is ASCII in practice for layout).
    col: u32,
    /// Whether a newline has been seen since the last emitted token.
    pending_newline: bool,
    /// Set when trivia-skipping hits an unrecoverable error (unterminated block
    /// comment). Surfaced as a `Tok::LexError` token at the next `next_token`.
    lex_error: Option<String>,
}

/// The full set of multi-char structural operators, longest first, mapped to a
/// token. Backslash-word operators (`\in`, `\A`, ...) are handled separately in
/// `read_backslash` so we can distinguish `\` (set minus) from `\in`.
const SYMBOL_OPS: &[(&str, fn() -> Tok)] = &[
    ("<=>", || Tok::Iff),
    ("|->", || Tok::MapsTo),
    ("=>", || Tok::Implies),
    ("==", || Tok::EqEq),
    ("/=", || Tok::Neq),
    ("<=", || Tok::Le),
    ("=<", || Tok::Le),
    (">=", || Tok::Ge),
    // (multi-char opaque operators handled as Other below)
];

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 0,
            pending_newline: false,
            lex_error: None,
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn byte_at(&self, off: usize) -> Option<u8> {
        self.src.get(self.pos + off).copied()
    }

    fn starts_with(&self, s: &str) -> bool {
        self.src[self.pos..].starts_with(s.as_bytes())
    }

    /// Width of a hard tab in columns for layout purposes. TLA+ specs SHOULD use
    /// spaces; if a tab appears we advance to the next multiple of this so the
    /// bullet column stays consistent rather than counting the tab as 1 byte.
    const TAB_WIDTH: u32 = 8;

    /// Advance one byte, maintaining line/col. Newlines reset the column and set
    /// `pending_newline`.
    ///
    /// Fix #6: column tracking is now CHAR- and tab-aware, not raw-byte:
    ///  - A UTF-8 continuation byte (`0b10xx_xxxx`) does NOT advance the column,
    ///    so a multi-byte char in a preceding comment/string can't shift a later
    ///    bullet's `start_col` (which the junction fence relies on).
    ///  - A hard tab advances to the next `TAB_WIDTH` tab-stop instead of +1, so
    ///    a tab-indented bullet lands on a stable column.
    fn bump(&mut self) {
        if let Some(b) = self.peek_byte() {
            self.pos += 1;
            if b == b'\n' {
                self.line += 1;
                self.col = 0;
                self.pending_newline = true;
            } else if b == b'\t' {
                self.col = (self.col / Self::TAB_WIDTH + 1) * Self::TAB_WIDTH;
            } else if (b & 0xC0) == 0x80 {
                // UTF-8 continuation byte: part of the previous char — no column
                // advance (the leading byte already counted the char once).
            } else {
                self.col += 1;
            }
        }
    }

    fn bump_n(&mut self, n: usize) {
        for _ in 0..n {
            self.bump();
        }
    }

    /// Skip whitespace and comments (line `\*` + nested block `(* *)`), updating
    /// `pending_newline` appropriately.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek_byte() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => self.bump(),
                Some(b'\\') if self.byte_at(1) == Some(b'*') => {
                    // Line comment: skip to end of line (do NOT consume newline
                    // here so pending_newline flips on the newline itself).
                    while let Some(b) = self.peek_byte() {
                        if b == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some(b'(') if self.byte_at(1) == Some(b'*') => {
                    // Block comment (nested).
                    self.bump_n(2);
                    let mut depth = 1usize;
                    while depth > 0 {
                        match self.peek_byte() {
                            None => {
                                // Fix #5: unterminated block comment. Record a lex
                                // error instead of silently breaking (which would
                                // truncate the input and let v2 lower a prefix).
                                self.lex_error = Some(
                                    "unterminated block comment (missing `*)`)".to_string(),
                                );
                                break;
                            }
                            Some(b'(') if self.byte_at(1) == Some(b'*') => {
                                self.bump_n(2);
                                depth += 1;
                            }
                            Some(b'*') if self.byte_at(1) == Some(b')') => {
                                self.bump_n(2);
                                depth -= 1;
                            }
                            _ => self.bump(),
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn span_from(&self, start: usize, sl: u32, sc: u32) -> Span {
        Span {
            start,
            end: self.pos,
            start_line: sl,
            start_col: sc,
            end_line: self.line,
            end_col: self.col,
        }
    }

    fn is_ident_start(b: u8) -> bool {
        b.is_ascii_alphabetic() || b == b'_'
    }
    fn is_ident_continue(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(b) = self.peek_byte() {
            if Self::is_ident_continue(b) {
                self.bump();
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.src[start..self.pos]).into_owned()
    }

    fn keyword_tok(word: &str) -> Option<Tok> {
        Some(match word {
            "LET" => Tok::Let,
            "IN" => Tok::In,
            "IF" => Tok::If,
            "THEN" => Tok::Then,
            "ELSE" => Tok::Else,
            "CHOOSE" => Tok::Choose,
            _ => return None,
        })
    }

    /// Handle a backslash: could be `\/` (or bullet), a named operator
    /// (`\in`, `\A`, `\notin`, `\land`, ...), or bare `\` (set minus).
    fn read_backslash(&mut self, start: usize, sl: u32, sc: u32) -> Token {
        // `\/` bullet
        if self.byte_at(1) == Some(b'/') {
            self.bump_n(2);
            return self.mk(Tok::OrBullet, start, sl, sc);
        }
        // Named operators: `\` followed by ident letters.
        if let Some(b) = self.byte_at(1) {
            if b.is_ascii_alphabetic() {
                // read the word after backslash
                self.bump(); // consume '\'
                let word = self.read_ident();
                let tok = match word.as_str() {
                    "in" => Tok::ElemOf,
                    "notin" => Tok::NotIn,
                    "A" | "forall" => Tok::Forall,
                    "E" | "exists" => Tok::Exists,
                    "land" => Tok::AndBullet,
                    "lor" => Tok::OrBullet,
                    "lnot" | "neg" => Tok::Not,
                    "equiv" => Tok::Iff,
                    "union" | "cup" => Tok::Union,
                    "intersect" | "cap" => Tok::Intersect,
                    "div" => Tok::Div,
                    "o" | "circ" => Tok::Concat,
                    "leq" => Tok::Le,
                    "geq" => Tok::Ge,
                    // Everything else (\X, \times, \subseteq, \E-less words,
                    // \cdot, \prec, ...) stays opaque.
                    other => Tok::Other(format!("\\{}", other)),
                };
                return self.mk(tok, start, sl, sc);
            }
        }
        // Bare backslash: set minus.
        self.bump();
        self.mk(Tok::Backslash, start, sl, sc)
    }

    fn mk(&self, kind: Tok, start: usize, sl: u32, sc: u32) -> Token {
        Token {
            kind,
            span: self.span_from(start, sl, sc),
            had_newline_before: false, // filled in by next_token
        }
    }

    /// Read a string literal (double-quoted, with `\"` escapes). Returns the raw
    /// text INCLUDING quotes as the Str payload's source; the parser keeps it as
    /// an Atom leaf.
    fn read_string(&mut self, start: usize, sl: u32, sc: u32) -> Token {
        self.bump(); // opening quote
        while let Some(b) = self.peek_byte() {
            if b == b'\\' {
                self.bump();
                self.bump();
                continue;
            }
            if b == b'"' {
                self.bump();
                break;
            }
            self.bump();
        }
        let text = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
        self.mk(Tok::Str(text), start, sl, sc)
    }

    pub fn next_token(&mut self) -> Token {
        self.pending_newline = false;
        self.skip_trivia();
        let had_newline = self.pending_newline;

        let start = self.pos;
        let sl = self.line;
        let sc = self.col;

        // Fix #5: surface a trivia-scanning error (unterminated block comment) as
        // a distinct token so the parser fails and falls back to v1.
        if let Some(msg) = self.lex_error.take() {
            let mut tok = self.mk(Tok::LexError(msg), start, sl, sc);
            tok.had_newline_before = had_newline;
            return tok;
        }

        let mut tok = match self.peek_byte() {
            None => self.mk(Tok::Eof, start, sl, sc),
            Some(b'/') if self.byte_at(1) == Some(b'\\') => {
                self.bump_n(2);
                self.mk(Tok::AndBullet, start, sl, sc)
            }
            Some(b'\\') => self.read_backslash(start, sl, sc),
            Some(b'"') => self.read_string(start, sl, sc),
            Some(b) if Self::is_ident_start(b) => {
                let word = self.read_ident();
                match Self::keyword_tok(&word) {
                    Some(k) => self.mk(k, start, sl, sc),
                    None => self.mk(Tok::Ident(word), start, sl, sc),
                }
            }
            Some(b) if b.is_ascii_digit() => {
                while let Some(d) = self.peek_byte() {
                    if d.is_ascii_digit() {
                        self.bump();
                    } else {
                        break;
                    }
                }
                let text =
                    String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                self.mk(Tok::Number(text), start, sl, sc)
            }
            Some(_) => self.read_symbol(start, sl, sc),
        };
        tok.had_newline_before = had_newline;
        tok
    }

    /// Read a symbolic (non-ident, non-backslash) token, longest-match.
    fn read_symbol(&mut self, start: usize, sl: u32, sc: u32) -> Token {
        // Longest-match structural multi-char operators.
        for (s, f) in SYMBOL_OPS {
            if self.starts_with(s) {
                self.bump_n(s.len());
                return self.mk(f(), start, sl, sc);
            }
        }
        // Opaque multi-char operators the parser keeps as `Other` but must not
        // split (so a `<<` doesn't tokenize as `<` `<`).
        for s in ["|->", "<<", ">>", "->", "..", "::", ":>", "@@", "[]", "<>"].iter() {
            if self.starts_with(s) {
                self.bump_n(s.len());
                return self.mk(Tok::Other(s.to_string()), start, sl, sc);
            }
        }
        // Single-char tokens.
        let b = self.peek_byte().unwrap();
        self.bump();
        let kind = match b {
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b':' => Tok::Colon,
            b',' => Tok::Comma,
            b'=' => Tok::Eq,
            b'#' => Tok::Neq,
            b'<' => Tok::Lt,
            b'>' => Tok::Gt,
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'%' => Tok::Mod,
            b'^' => Tok::Caret,
            b'~' => Tok::Not,
            other => Tok::Other((other as char).to_string()),
        };
        self.mk(kind, start, sl, sc)
    }

    /// Tokenize the whole input into a Vec (ending with an Eof token).
    pub fn tokenize(mut self) -> Vec<Token> {
        let mut out = Vec::new();
        loop {
            let t = self.next_token();
            let is_eof = matches!(t.kind, Tok::Eof);
            out.push(t);
            if is_eof {
                break;
            }
        }
        out
    }
}

/// Convenience: tokenize a source string.
pub fn tokenize(src: &str) -> Vec<Token> {
    Lexer::new(src).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        tokenize(src).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn longest_match_bullets_and_ops() {
        assert_eq!(
            kinds("/\\ x <=> y"),
            vec![
                Tok::AndBullet,
                Tok::Ident("x".into()),
                Tok::Iff,
                Tok::Ident("y".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn distinguishes_in_from_setminus() {
        assert_eq!(kinds("x \\in S")[1], Tok::ElemOf);
        assert_eq!(kinds("S \\ T")[1], Tok::Backslash);
    }

    #[test]
    fn strips_comments_preserves_columns() {
        let toks = tokenize("x \\* comment\n /\\ y");
        // after the comment + newline, the /\ bullet should have had_newline set
        // and its column should be 1 (one leading space on line 2).
        let bullet = toks.iter().find(|t| t.kind == Tok::AndBullet).unwrap();
        assert!(bullet.had_newline_before);
        assert_eq!(bullet.span.start_col, 1);
        assert_eq!(bullet.span.start_line, 2);
    }

    #[test]
    fn block_comment_nested() {
        let toks = tokenize("a (* outer (* inner *) still *) b");
        let idents: Vec<_> = toks
            .iter()
            .filter_map(|t| match &t.kind {
                Tok::Ident(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(idents, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn mid_line_bullet_has_no_newline_flag() {
        let toks = tokenize("Foo == /\\ A");
        let bullet = toks.iter().find(|t| t.kind == Tok::AndBullet).unwrap();
        assert!(!bullet.had_newline_before);
    }

    #[test]
    fn opaque_multichar_not_split() {
        let toks = tokenize("<<1, 2>>");
        assert_eq!(toks[0].kind, Tok::Other("<<".into()));
    }
}
