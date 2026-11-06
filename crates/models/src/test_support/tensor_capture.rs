//! Test-only observations of the operators actually used by a feed-forward call.
//!
//! Production builds contain none of these callbacks. Tensor handles retain the
//! observed outputs until readback; tests separately verify observation does not
//! change the call's output. Host routing metadata is copied without GPU work.

use candle::Tensor;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[derive(Clone)]
pub(crate) enum Value {
    Tensor(Tensor),
    Host { shape: Vec<usize>, values: Vec<f32> },
}

struct Capture {
    values: BTreeMap<String, Value>,
    root_prefix: String,
    prefix: String,
    router_override: Option<Tensor>,
    enabled: bool,
    overrides: BTreeMap<String, Tensor>,
    deferred: BTreeMap<String, Rc<dyn Fn() -> BTreeMap<String, Value>>>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

/// Starts one isolated observation; nested captures would mix unrelated calls.
pub(crate) fn begin(router_override: Option<Tensor>) {
    ACTIVE.with_borrow_mut(|active| {
        assert!(active.is_none(), "nested tensor capture");
        *active = Some(Capture {
            router_override,
            values: BTreeMap::new(),
            root_prefix: String::new(),
            prefix: String::new(),
            enabled: true,
            overrides: BTreeMap::new(),
            deferred: BTreeMap::new(),
        });
    });
}

/// Registers scratch readback during program construction without reading it
/// inside graph capture. Snapshot callers evaluate it after the selected replay.
pub(crate) fn defer(name: &str, read: impl Fn() -> BTreeMap<String, Value> + 'static) {
    ACTIVE.with_borrow_mut(|active| {
        if let Some(capture) = active.as_mut().filter(|capture| capture.enabled) {
            let key = format!("{}{}{name}", capture.root_prefix, capture.prefix);
            assert!(capture.deferred.insert(key, Rc::new(read)).is_none());
        }
    });
}

/// Owns CPU copies of the current graph slots and deferred scratch before another
/// replay can overwrite them. Registration remains active for later snapshots.
pub(crate) fn snapshot() -> BTreeMap<String, Value> {
    let (mut values, deferred) = ACTIVE.with_borrow(|active| {
        let capture = active.as_ref().expect("capture not active");
        (capture.values.clone(), capture.deferred.clone())
    });
    for (prefix, read) in deferred {
        for (name, value) in read() {
            assert!(values.insert(format!("{prefix}-{name}"), value).is_none());
        }
    }
    for value in values.values_mut() {
        if let Value::Tensor(tensor) = value {
            *tensor = tensor
                .to_device(&candle::Device::Cpu)
                .unwrap()
                .copy()
                .unwrap();
        }
    }
    values
}

/// Returns registered tensor/host handles and clears state. Call `snapshot`
/// first when deferred scratch or CPU-owned values are needed; `finish` itself
/// performs no device readback and does not evaluate deferred readers.
pub(crate) fn finish() -> BTreeMap<String, Value> {
    ACTIVE.with_borrow_mut(|active| active.take().expect("capture not active").values)
}

/// Replaces one explicitly named operator output in a diagnostic forward.
/// Layer-qualified names keep a counterfactual isolated to its intended stage;
/// this mechanism is test-only and does not select serving arithmetic.
pub(crate) fn override_tensor(name: String, value: Tensor) {
    ACTIVE.with_borrow_mut(|active| {
        assert!(
            active
                .as_mut()
                .expect("capture not active")
                .overrides
                .insert(name, value)
                .is_none()
        );
    });
}

/// Applies a diagnostic replacement after the original operator has run. The
/// ordinary computation/state update is retained, allowing one boundary's
/// downstream effect to be isolated without changing production dispatch.
pub(crate) fn substitute(name: &str, actual: Tensor) -> Tensor {
    ACTIVE.with_borrow(|active| {
        let Some(capture) = active else { return actual };
        let key = format!("{}{}{name}", capture.root_prefix, capture.prefix);
        let Some(value) = capture.overrides.get(&key) else {
            return actual;
        };
        assert_eq!(value.shape(), actual.shape());
        assert_eq!(value.dtype(), actual.dtype());
        assert!(value.device().same_device(actual.device()));
        value.clone()
    })
}

/// Substitutes explicit router scores only in controlled validation fixtures.
pub(crate) fn router_logits(projected: Tensor) -> Tensor {
    ACTIVE.with_borrow(|active| {
        active
            .as_ref()
            .and_then(|capture| capture.router_override.clone())
            .unwrap_or(projected)
    })
}

/// Retains a tensor at its actual operator boundary, without evaluating it again.
pub(crate) fn tensor(name: &str, value: &Tensor) {
    ACTIVE.with_borrow_mut(|active| {
        if let Some(capture) = active.as_mut().filter(|capture| capture.enabled) {
            let key = format!("{}{}{name}", capture.root_prefix, capture.prefix);
            assert!(
                capture
                    .values
                    .insert(key, Value::Tensor(value.clone()))
                    .is_none()
            );
        }
    });
}

/// Stores routing/group metadata which the current fallback already owns on CPU.
pub(crate) fn host(name: &str, shape: Vec<usize>, values: Vec<f32>) {
    ACTIVE.with_borrow_mut(|active| {
        if let Some(capture) = active.as_mut().filter(|capture| capture.enabled) {
            let key = format!("{}{}{name}", capture.root_prefix, capture.prefix);
            assert!(
                capture
                    .values
                    .insert(key, Value::Host { shape, values })
                    .is_none()
            );
        }
    });
}

/// Appends values in the actual operator's iteration order, preserving route rank.
pub(crate) fn append_host(name: &str, shape: Vec<usize>, value: f32) {
    ACTIVE.with_borrow_mut(|active| {
        if let Some(capture) = active.as_mut().filter(|capture| capture.enabled) {
            let entry = capture
                .values
                .entry(format!("{}{}{name}", capture.root_prefix, capture.prefix))
                .or_insert_with(|| Value::Host {
                    shape,
                    values: Vec::new(),
                });
            let Value::Host { values, .. } = entry else {
                panic!("host capture type changed")
            };
            values.push(value);
        }
    });
}

/// Reports whether observers should collect optional, potentially costly metadata.
pub(crate) fn active() -> bool {
    ACTIVE.with_borrow(|active| active.as_ref().is_some_and(|capture| capture.enabled))
}

/// Disables observation while retaining a controlled routing fixture for replay.
pub(crate) fn pause() {
    ACTIVE.with_borrow_mut(|active| {
        active.as_mut().expect("capture not active").enabled = false
    });
}

pub(crate) struct Scope(String);

pub(crate) struct RootScope(String);

/// Keeps a decoder-layer label outside component-local scopes. Whole-model
/// diagnostics use this so identically named shared/expert stages in successive
/// layers remain distinct; standalone operator captures keep their usual names.
pub(crate) fn root_scope(prefix: String) -> RootScope {
    RootScope(ACTIVE.with_borrow_mut(|active| {
        active
            .as_mut()
            .map(|capture| std::mem::replace(&mut capture.root_prefix, prefix))
            .unwrap_or_default()
    }))
}

impl Drop for RootScope {
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|active| {
            if let Some(capture) = active {
                capture.root_prefix = self.0.clone();
            }
        });
    }
}

/// Labels nested dense MLP operations with their expert or shared-branch owner.
pub(crate) fn scope(prefix: String) -> Scope {
    Scope(ACTIVE.with_borrow_mut(|active| {
        active
            .as_mut()
            .map(|capture| std::mem::replace(&mut capture.prefix, prefix))
            .unwrap_or_default()
    }))
}

impl Drop for Scope {
    /// Restores the caller's label even when an observed operator returns an error.
    fn drop(&mut self) {
        ACTIVE.with_borrow_mut(|active| {
            if let Some(capture) = active {
                capture.prefix = self.0.clone();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_freeze_deferred_values_before_the_next_replay() {
        let slot = candle::Var::from_tensor(
            &Tensor::new(&[1f32], &candle::Device::Cpu).unwrap(),
        )
        .unwrap();
        begin(None);
        tensor("input", &slot.as_detached_tensor());
        let read_slot = slot.clone();
        defer("scratch", move || {
            BTreeMap::from([(
                "output".to_owned(),
                Value::Tensor(read_slot.as_detached_tensor()),
            )])
        });
        let first = snapshot();
        slot.set(&Tensor::new(&[2f32], &candle::Device::Cpu).unwrap())
            .unwrap();
        let second = snapshot();
        finish();
        for (values, expected) in [(first, 1.), (second, 2.)] {
            for name in ["input", "scratch-output"] {
                let Value::Tensor(value) = &values[name] else {
                    panic!("tensor expected")
                };
                assert_eq!(value.to_vec1::<f32>().unwrap(), [expected]);
            }
        }
    }

    #[test]
    fn substitution_is_layer_scoped_and_survives_observer_pause() {
        let original = Tensor::new(&[1f32], &candle::Device::Cpu).unwrap();
        let replacement = Tensor::new(&[2f32], &candle::Device::Cpu).unwrap();
        begin(None);
        override_tensor("layer-0-output".to_owned(), replacement.clone());
        pause();
        for layer in 0..2 {
            let _scope = root_scope(format!("layer-{layer}-"));
            let value = substitute("output", original.clone());
            assert_eq!(
                value.to_vec1::<f32>().unwrap(),
                if layer == 0 { vec![2.] } else { vec![1.] }
            );
        }
        assert!(finish().is_empty());
        assert_eq!(substitute("output", original.clone()).id(), original.id());
    }

    #[test]
    fn layer_labels_survive_component_scope_changes() {
        begin(None);
        for layer in 0..2 {
            let _layer = root_scope(format!("layer-{layer}-"));
            {
                let _component = scope("shared-".to_owned());
                host("output", vec![1], vec![layer as f32]);
            }
            host("output", vec![1], vec![layer as f32]);
        }
        host("output", vec![1], vec![2.]);
        let observed = finish();
        assert_eq!(
            observed.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "layer-0-output",
                "layer-0-shared-output",
                "layer-1-output",
                "layer-1-shared-output",
                "output"
            ]
        );
    }
}
