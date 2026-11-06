use super::*;
use serde_json::json;

/// A request's explicit kwargs win without modifying the caller's JSON object.
#[test]
fn normalizes_precedence_and_preserves_explicit_nulls() {
    let kwargs = json!({"reasoning_effort":"medium", "enable_thinking":null});
    let before = kwargs.clone();
    let options = ChatTemplateOptions::from_request_fields(
        Some(&json!("xhigh")),
        Some(&json!({"effort":"low"})),
        Some(&kwargs),
    )
    .unwrap();
    assert_eq!(
        options.variables(),
        json!({
            "reasoning_effort":"medium", "enable_thinking":null, "thinking":true,
        })
        .as_object()
        .unwrap()
    );
    assert_eq!(kwargs, before);
}

/// Omitted settings retain checkpoint defaults; effort=none supplies false flags
/// but an explicit kwarg remains authoritative.
#[test]
fn keeps_defaults_undefined_and_effort_none_overridable() {
    for fields in [
        (None, None, None),
        (Some(&Value::Null), None, Some(&Value::Null)),
    ] {
        assert!(
            ChatTemplateOptions::from_request_fields(fields.0, fields.1, fields.2)
                .unwrap()
                .variables()
                .is_empty()
        );
    }
    let options = ChatTemplateOptions::from_request_fields(
        Some(&json!("none")),
        None,
        Some(&json!({"enable_thinking":true})),
    )
    .unwrap();
    assert_eq!(options.variables()["thinking"], false);
    assert_eq!(options.variables()["enable_thinking"], true);
}

/// The generic API accepts documented effort tiers and finite fractional values;
/// individual templates remain responsible for stricter checkpoint restrictions.
#[test]
fn validates_effort_and_rejects_legacy_coercion_and_reserved_controls() {
    for value in [
        json!(true),
        json!(-0.1),
        json!(1.0),
        json!("invalid"),
        json!([]),
    ] {
        assert!(
            ChatTemplateOptions::from_request_fields(Some(&value), None, None).is_err()
        );
    }
    for value in [
        json!("max"),
        json!("minimal"),
        json!(0.0),
        json!(0.99),
        json!("0.5"),
    ] {
        assert!(
            ChatTemplateOptions::from_request_fields(Some(&value), None, None).is_ok()
        );
    }
    assert!(
        ChatTemplateOptions::from_request_fields(
            None,
            Some(&json!({"enabled":"yes"})),
            None
        )
        .is_err()
    );
    assert!(
        ChatTemplateOptions::from_request_fields(
            Some(&json!("low")),
            None,
            Some(&json!("bad"))
        )
        .is_err()
    );
    for key in [
        "messages",
        "tools",
        "documents",
        "continue_final_message",
        "tokenize",
    ] {
        assert!(
            ChatTemplateOptions::from_request_fields(
                None,
                None,
                Some(&json!({key:null}))
            )
            .is_err()
        );
    }
}

/// Repeated construction and legacy-engine rejection cover direct Rust callers,
/// independently of the HTTP preprocessing and end-to-end oracle tests.
#[test]
fn preserves_nested_values_and_rejects_unsupported_engines() {
    let kwargs: Value = serde_json::from_str(
        r#"{"custom":{"z":1,"a":1.0,"large":9007199254740993},"list":[null,false]}"#,
    )
    .unwrap();
    let options =
        ChatTemplateOptions::from_request_fields(None, None, Some(&kwargs)).unwrap();
    assert_eq!(options.variables(), kwargs.as_object().unwrap());
    assert_eq!(
        options.variables()["custom"]["large"].as_u64(),
        Some(9007199254740993)
    );
    assert_eq!(
        options.variables()["custom"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a", "large"]
    );
    assert!(options.require_supported(false).is_err());
    assert!(
        ChatTemplateOptions::default()
            .require_supported(false)
            .is_ok()
    );
}

/// Independent SGLang normalization catches precedence changes even when two
/// option combinations happen to produce identical text for one checkpoint.
#[test]
fn matches_pinned_sglang_normalized_context() {
    let reference: Value =
        serde_json::from_str(include_str!("../../tests/data/template_controls.json"))
            .unwrap();
    let mut checked = 0;
    for case in reference["cases"].as_array().unwrap() {
        let expected = &case["expected"];
        let request = &case["request"];
        let actual = ChatTemplateOptions::from_request_fields(
            request.get("reasoning_effort"),
            request.get("reasoning"),
            request.get("chat_template_kwargs"),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(
            actual.variables(),
            expected.as_object().unwrap(),
            "{}",
            case["name"]
        );
        checked += 1;
    }
    assert!(
        checked >= 30,
        "normalization reference must cover distinct option combinations"
    );
}
