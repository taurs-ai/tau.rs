use super::*;
use crate::ChatTemplateOptions;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Owns temporary template assets independently of GPU/model fixture storage.
struct Assets(std::path::PathBuf);
impl Assets {
    /// Creates an owned, unique directory for each independent asset test.
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tau-chat-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    /// Writes one synthetic asset without modifying any model fixture.
    fn write(&self, name: &str, value: &str) {
        std::fs::write(self.0.join(name), value).unwrap();
    }
}
impl Drop for Assets {
    /// Cleans up owned assets, including when an assertion is already unwinding.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Validates macros, fresh namespaces, Python methods and default bool rendering.
#[test]
fn executes_jinja_and_reuses_compilation_with_fresh_context() {
    let assets = Assets::new();
    assets.write(
        "chat_template.jinja",
        concat!(
            "{%- macro row(m) -%}{{m.role|upper}}={{m.content.strip()|tojson}};",
            "{%- endmacro -%}{%- set ns=namespace(n=0) -%}",
            "{%- for m in messages -%}{%- set ns.n=ns.n+1 -%}{{ row(m) }}",
            "{%- endfor -%}COUNT={{ns.n}};GEN={{add_generation_prompt}}",
        ),
    );
    let template = load_file(&assets.0, None).unwrap();
    for content in ["  héllo <&>  ", "next", "  héllo <&>  "] {
        let result = template
            .render_default(&[ChatMessage::new("user", content)])
            .unwrap();
        assert_eq!(
            result,
            format!(
                "USER={};COUNT=1;GEN=True",
                serde_json::to_string(content.trim()).unwrap()
            )
        );
    }
    assets.write("chat_template.jinja", "CHANGED");
    assert_ne!(
        template
            .render_default(&[ChatMessage::new("user", "x")])
            .unwrap(),
        "CHANGED"
    );
    assert_eq!(
        load_file(&assets.0, None)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "x")])
            .unwrap(),
        "CHANGED"
    );
}

/// Automatic loading preserves checkpoint bytes; explicit files normalize newlines
/// and take priority over standalone/embedded source.
#[test]
fn resolves_sources_and_reports_invalid_or_missing_templates() {
    let a = Assets::new();
    a.write(
        "tokenizer_config.json",
        concat!(
            r#"{"chat_template":"EMBEDDED {{messages[0].content}}","#,
            r#""bos_token":{"content":"BOS"}}"#,
        ),
    );
    let render = |override_path| {
        load_file(&a.0, override_path)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "hello")])
            .unwrap()
    };
    assert_eq!(render(None), "EMBEDDED hello");
    a.write("chat_template.jinja", "STANDALONE {{bos_token}}");
    assert_eq!(render(None), "STANDALONE BOS");
    a.write("override.jinja", "\nOVERRIDE\\n{{messages[0].content}}\n");
    assert_eq!(render(Some(&a.0.join("override.jinja"))), "OVERRIDE\nhello");
    assert!(load_file(&a.0, Some(&a.0.join("absent"))).is_err());
    a.write("chat_template.jinja", "{% if %}");
    assert!(load_file(&a.0, None).is_err());
    std::fs::remove_file(a.0.join("chat_template.jinja")).unwrap();
    a.write(
        "tokenizer_config.json",
        r#"{"chat_template":[{"name":"default","template":"named"}]}"#,
    );
    assert_eq!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "x")])
            .unwrap(),
        "named"
    );
    std::fs::remove_file(a.0.join("tokenizer_config.json")).unwrap();
    assert!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "x")])
            .is_err()
    );
}

/// Verifies string/list content normalization and trailing-assistant handling.
#[test]
fn renders_text_parts_and_template_exceptions() {
    let a = Assets::new();
    a.write(
        "chat_template.jinja",
        concat!(
            "{{tools is none}}/{{documents is none}}/",
            "{{enable_thinking is defined}}/{{reasoning_effort is defined}}",
        ),
    );
    assert_eq!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "hello")])
            .unwrap(),
        "True/True/False/False"
    );
    let message =
        ChatMessage::from_text_parts("user", vec!["a".to_owned(), "b".to_owned()]);
    a.write(
        "chat_template.jinja",
        "{{messages[0].content}}/{{messages[-1].role}}",
    );
    assert_eq!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[message.clone(), ChatMessage::new("assistant", "answer")])
            .unwrap(),
        "a b/user"
    );
    a.write(
        "chat_template.jinja",
        concat!(
            "{% for m in messages %}{% for part in m.content %}",
            "{{part.text}}{% endfor %}{% endfor %}",
        ),
    );
    assert_eq!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[message])
            .unwrap(),
        "ab"
    );
    a.write("chat_template.jinja", "{{raise_exception('stop')}}");
    assert!(
        load_file(&a.0, None)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "x")])
            .unwrap_err()
            .to_string()
            .contains("stop")
    );
}

impl CheckpointChatTemplate {
    /// Exercises production rendering with defaults for the asset tests.
    fn render_default(
        &self,
        messages: &[ChatMessage],
    ) -> Result<String, ChatTemplateError> {
        self.render(messages, &ChatTemplateOptions::default())
    }
}

/// Numeric custom variables must use Python's exact JSON spelling, including
/// exponent signs/zero padding and negative zero, before prompt tokenization.
#[test]
fn renders_custom_numeric_json_like_python() {
    let assets = Assets::new();
    assets.write("chat_template.jinja", "{{custom|tojson}}");
    let kwargs = json!({"custom":[1e-7, 1e-6, 1e-5, 0.0001, 1e15, 1e16, 1e20, -0.0]});
    let options =
        ChatTemplateOptions::from_request_fields(None, None, Some(&kwargs)).unwrap();
    let template = load_file(&assets.0, None).unwrap();
    assert_eq!(
        template
            .render(&[ChatMessage::new("user", "hi")], &options)
            .unwrap(),
        "[1e-07, 1e-06, 1e-05, 0.0001, 1000000000000000.0, 1e+16, 1e+20, -0.0]",
    );
}

/// Collection order is semantic: a later entry called default is not selected
/// implicitly under the current source policy. Only the chosen source compiles.
#[test]
fn selects_named_entries_without_compiling_unused_sources() {
    let assets = Assets::new();
    assets.write(
        "tokenizer_config.json",
        concat!(
            r#"{"chat_template":{"alternate":"ALT {{messages[0].content}}","#,
            r#""default":"DEFAULT","broken":"{% if %}"}}"#,
        ),
    );
    let render = |selection: ChatTemplateSelection| {
        CheckpointChatTemplate::load(&assets.0, &selection).and_then(|template| {
            template.render_default(&[ChatMessage::new("user", "hello")])
        })
    };
    assert_eq!(render(ChatTemplateSelection::new()).unwrap(), "ALT hello");
    assert_eq!(
        render(ChatTemplateSelection::new().with_name("default")).unwrap(),
        "DEFAULT"
    );
    assert!(render(ChatTemplateSelection::new().with_name("broken")).is_err());
    assert!(render(ChatTemplateSelection::new().with_name("missing")).is_err());
    assets.write("override.jinja", "FILE");
    assert_eq!(
        render(
            ChatTemplateSelection::new()
                .with_name("missing")
                .with_file(assets.0.join("override.jinja"))
        )
        .unwrap(),
        "FILE"
    );
}

/// File catalogs replace embedded entries wholesale. A second default source
/// replaces its value while retaining the original position.
#[test]
fn loads_file_collections_with_source_precedence() {
    let assets = Assets::new();
    assets.write(
        "tokenizer_config.json",
        r#"{"chat_template":{"embedded":"WRONG"}}"#,
    );
    assets.write("chat_template.jinja", "ROOT");
    std::fs::create_dir(assets.0.join("additional_chat_templates")).unwrap();
    assets.write("additional_chat_templates/alternate.jinja", "ALT");
    assets.write("additional_chat_templates/default.jinja", "EXTRA DEFAULT");
    let render = |selection| {
        CheckpointChatTemplate::load(&assets.0, &selection)
            .unwrap()
            .render_default(&[ChatMessage::new("user", "hello")])
            .unwrap()
    };
    assert_eq!(render(ChatTemplateSelection::new()), "EXTRA DEFAULT");
    assert_eq!(
        render(ChatTemplateSelection::new().with_name("alternate")),
        "ALT"
    );
    assert!(
        CheckpointChatTemplate::load(
            &assets.0,
            &ChatTemplateSelection::new().with_name("embedded")
        )
        .is_err()
    );
}

/// Malformed collections fail during construction; a missing optional catalog
/// still permits raw inference unless the caller explicitly requested a name.
#[test]
fn rejects_invalid_collections_and_missing_explicit_names() {
    let assets = Assets::new();
    for templates in [
        json!([]),
        json!({}),
        json!({"default":3}),
        json!([{"name":"x"}]),
        json!([{"template":"x"}]),
        json!([{"name":"x","template":"a"},{"name":"x","template":"b"}]),
    ] {
        assets.write(
            "tokenizer_config.json",
            &json!({"chat_template":templates}).to_string(),
        );
        assert!(
            CheckpointChatTemplate::load(&assets.0, &ChatTemplateSelection::new())
                .is_err()
        );
    }
    assets.write("tokenizer_config.json", "{}");
    assert!(
        CheckpointChatTemplate::load(&assets.0, &ChatTemplateSelection::new()).is_ok()
    );
    assert!(
        CheckpointChatTemplate::load(
            &assets.0,
            &ChatTemplateSelection::new().with_name("default")
        )
        .is_err()
    );
}

/// Preserves readability of file/default asset tests while using the typed source API.
fn load_file(
    root: &Path,
    file: Option<&Path>,
) -> Result<CheckpointChatTemplate, ChatTemplateError> {
    let selection = file
        .map(|path| ChatTemplateSelection::new().with_file(path))
        .unwrap_or_default();
    CheckpointChatTemplate::load(root, &selection)
}
