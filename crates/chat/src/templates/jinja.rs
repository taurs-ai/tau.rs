//! Jinja compilation, message-content classification and JSON formatting.

use minijinja::machinery::{
    ast::{Expr, Stmt},
    parse,
};
use minijinja::syntax::SyntaxConfig;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde::Serialize;
use serde_json::ser::Formatter;
use std::io::{self, Write};

/// Trims block whitespace without adding HTML escaping.
fn syntax() -> SyntaxConfig {
    SyntaxConfig::builder()
        .trim_blocks(true)
        .lstrip_blocks(true)
        .build()
        .expect("default Jinja syntax")
}

/// Compiles owned source with Python methods, JSON formatting and template errors.
pub(super) fn compile(source: String) -> Result<Environment<'static>, Error> {
    let mut env = Environment::new();
    env.set_syntax(syntax());
    env.set_unknown_method_callback(
        minijinja_contrib::pycompat::unknown_method_callback,
    );
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_filter("tojson", to_json);
    // MiniJinja 3 already emits Python-style True/False/None. Keep its default
    // formatter; JSON separator and exponent formatting need an adapter here.
    env.add_template_owned("chat", source)?;
    Ok(env)
}

/// Matches Python json.dumps separators, preserving Unicode and HTML characters.
fn to_json(value: Value) -> Result<String, Error> {
    if value.is_undefined() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "cannot JSON serialize an undefined value",
        ));
    }
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonJson);
    value
        .serialize(&mut serializer)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    String::from_utf8(bytes)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))
}

/// Adds Python's default separator spaces without pretty-printing JSON.
struct PythonJson;
impl Formatter for PythonJson {
    /// Python repr and Rust Debug use the same shortest-float notation here,
    /// except Python includes an exponent sign and at least two exponent digits.
    /// These bytes matter when tojson output becomes part of the model prompt.
    fn write_f64<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> io::Result<()> {
        let representation = format!("{value:?}");
        if let Some((mantissa, exponent)) = representation.split_once('e') {
            let exponent: i32 = exponent.parse().map_err(io::Error::other)?;
            write!(writer, "{mantissa}e{exponent:+03}")
        } else {
            writer.write_all(representation.as_bytes())
        }
    }

    /// Separates array values with the comma-space used by Python JSON.
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    /// Separates object entries without changing their insertion order.
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    /// Matches Python's colon-space separator between keys and values.
    fn begin_object_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
    ) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Media keywords and message-content loops select structured content parts.
/// This classification behavior matches SGLang.
pub(super) fn structured_content(source: &str) -> Result<bool, Error> {
    if ["image", "audio", "video", "vision"]
        .iter()
        .any(|word| source.contains(word))
    {
        return Ok(true);
    }
    let ast = parse(source, "chat", syntax())?;
    Ok(has_content_loop(&ast))
}

/// Visits nested blocks so macros/conditionals cannot hide a content loop.
fn has_content_loop(statement: &Stmt<'_>) -> bool {
    let any = |body: &[Stmt<'_>]| body.iter().any(has_content_loop);
    match statement {
        Stmt::Template(n) => any(&n.children),
        Stmt::ForLoop(n) => {
            content_access(&n.iter) || any(&n.body) || any(&n.else_body)
        }
        Stmt::IfCond(n) => any(&n.true_body) || any(&n.false_body),
        Stmt::WithBlock(n) => any(&n.body),
        Stmt::SetBlock(n) => any(&n.body),
        Stmt::AutoEscape(n) => any(&n.body),
        Stmt::FilterBlock(n) => any(&n.body),
        Stmt::Macro(n) => any(&n.body),
        Stmt::CallBlock(n) => any(&n.macro_decl.body),
        _ => false,
    }
}

/// Recognizes message/msg/m.content and equivalent constant-key indexing.
fn content_access(expression: &Expr<'_>) -> bool {
    let base = match expression {
        Expr::GetAttr(n) if n.name == "content" => Some(&n.expr),
        Expr::GetItem(n) => match &n.subscript_expr {
            Expr::Const(key) if key.value.as_str() == Some("content") => Some(&n.expr),
            _ => None,
        },
        _ => None,
    };
    matches!(base, Some(Expr::Var(n)) if matches!(n.id, "message" | "msg" | "m"))
}
