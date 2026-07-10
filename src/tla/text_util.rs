//! Small text/layout utilities shared across the TLA+ frontend (the expr_v2
//! parser and the action-IR splitter). Extracted here to avoid duplication.

/// Return the first *logical* line of `src` — i.e. skip leading whitespace,
/// `\*` line comments, and nested `(* ... *)` block comments, then return from
/// the first real (non-comment, non-blank) character to the end of that line,
/// `trim_start`ed. Returns `""` if `src` is entirely blank/comments.
///
/// Starting the returned slice at the first real char (not the raw line start)
/// means an inline trailing block comment before real code — `(* c *) /\ A` —
/// still yields `/\ A`, so a leading-bullet (`/\`/`\/`) check on the result
/// fires correctly. Used by the layout-aware fences in both `expr_v2` and the
/// action-IR conjunct/disjunct splitters.
pub(crate) fn first_logical_line_skipping_comments(src: &str) -> &str {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    let mut depth: usize = 0; // block-comment nesting depth

    while i < n {
        if depth > 0 {
            // Inside a block comment: look for `*)` (may nest with `(*`).
            if i + 1 < n && bytes[i] == b'(' && bytes[i + 1] == b'*' {
                depth += 1;
                i += 2;
            } else if i + 1 < n && bytes[i] == b'*' && bytes[i + 1] == b')' {
                depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'(' if i + 1 < n && bytes[i + 1] == b'*' => {
                depth += 1;
                i += 2;
            }
            b'\\' if i + 1 < n && bytes[i + 1] == b'*' => {
                // `\*` line comment: skip to end of line.
                while i < n && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            _ => {
                let line_end = src[i..].find('\n').map(|p| i + p).unwrap_or(n);
                return src[i..line_end].trim_start();
            }
        }
    }
    ""
}

#[cfg(test)]
mod tests {
    use super::first_logical_line_skipping_comments as f;

    #[test]
    fn skips_leading_line_and_block_comments() {
        assert_eq!(f("(* c *)\n/\\ A"), "/\\ A");
        assert_eq!(f("\\* line\n(* blk *)\n\\/ X"), "\\/ X");
        assert_eq!(f("(* c *) /\\ A"), "/\\ A");
        assert_eq!(f("  \n  /\\ A"), "/\\ A");
        assert_eq!(f("(* line one\n line two *)\n/\\ A"), "/\\ A");
        assert_eq!(f("   \n\t(* c *)  "), "");
    }
}
