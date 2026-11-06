//! Model-independent safetensors header contract validation.
//!
//! Model-family adapters derive the exact names, shapes, and accepted storage
//! dtypes consumed by their constructors. Storage defaults to the checkpoint's
//! declared dtype; a requirement may explicitly also allow FP32. This module checks
//! every loaded shard before Candle allocates model weights. Additional tensors
//! remain accepted so wrappers may carry vision, MTP, or other unused subtrees.

use crate::dtype::ModelDType;
use crate::{ModelLoadError, ModelShard};
use std::collections::HashMap;
use std::path::PathBuf;

/// Storage precision accepted for one tensor, independently of runtime dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageDType {
    Checkpoint,
    CheckpointOrF32,
}

/// One tensor name, shape, and storage policy required by a model constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    storage_dtype: StorageDType,
}

impl TensorSpec {
    /// Requires an exact name and shape, defaulting to the declared checkpoint dtype.
    pub(crate) fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: Into::<String>::into(name),
            shape: Into::<Vec<usize>>::into(shape),
            storage_dtype: StorageDType::Checkpoint,
        }
    }

    /// Also accepts an FP32 master parameter in a reduced-precision checkpoint.
    pub(crate) fn with_f32_storage(mut self) -> Self {
        self.storage_dtype = StorageDType::CheckpointOrF32;
        self
    }

    /// Returns the exact checkpoint tensor name.
    #[cfg(test)]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Returns the dimensions expected by the model constructor.
    #[cfg(test)]
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Header metadata retained while checking names across all shards.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorHeader {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    shard: PathBuf,
}

/// Validates required names, shapes, and storage dtypes across all model shards.
pub(crate) fn validate_tensor_contract(
    family: &str,
    expected_dtype: ModelDType,
    shards: &[ModelShard],
    required: &[TensorSpec],
) -> Result<(), ModelLoadError> {
    let headers = shards
        .iter()
        .flat_map(|shard| {
            shard.tensors().iter().map(|tensor| TensorHeader {
                name: tensor.name().to_owned(),
                dtype: tensor.dtype().to_owned(),
                shape: tensor.shape().to_vec(),
                shard: shard.path().to_path_buf(),
            })
        })
        .collect::<Vec<_>>();
    validate_tensor_headers(family, expected_dtype, &headers, required)
}

/// Checks required headers while deliberately allowing unconsumed tensor names.
fn validate_tensor_headers(
    family: &str,
    expected_dtype: ModelDType,
    headers: &[TensorHeader],
    required: &[TensorSpec],
) -> Result<(), ModelLoadError> {
    let mut tensors = HashMap::with_capacity(headers.len());
    for header in headers {
        if let Some(previous) = tensors.insert(header.name.as_str(), header) {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} tensor {:?} appears in both {} and {}",
                header.name,
                previous.shard.display(),
                header.shard.display()
            )));
        }
    }

    let expected_dtype = safetensors_dtype(expected_dtype);
    for requirement in required {
        let header = tensors.get(requirement.name.as_str()).ok_or_else(|| {
            ModelLoadError::InvalidConfig(format!(
                "{family} checkpoint is missing required tensor {:?}",
                requirement.name
            ))
        })?;
        if header.shape != requirement.shape {
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} tensor {:?} in {} has shape {:?}, expected {:?}",
                requirement.name,
                header.shard.display(),
                header.shape,
                requirement.shape
            )));
        }
        let allows_f32 = requirement.storage_dtype == StorageDType::CheckpointOrF32;
        if header.dtype != expected_dtype && !(allows_f32 && header.dtype == "F32") {
            let allowed = if allows_f32 && expected_dtype != "F32" {
                format!("{expected_dtype} or F32")
            } else {
                expected_dtype.to_owned()
            };
            return Err(ModelLoadError::InvalidConfig(format!(
                "{family} tensor {:?} in {} has dtype {}, expected {} from \
                 checkpoint metadata",
                requirement.name,
                header.shard.display(),
                header.dtype,
                allowed
            )));
        }
    }
    Ok(())
}

/// Maps the declared checkpoint dtype to safetensors' stable header spelling.
fn safetensors_dtype(dtype: ModelDType) -> &'static str {
    match dtype {
        ModelDType::F32 => "F32",
        ModelDType::BF16 => "BF16",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the minimal required tensor set shared by contract tests.
    fn required_tensors() -> Vec<TensorSpec> {
        vec![
            TensorSpec::new("model.embed_tokens.weight", vec![64, 16]),
            TensorSpec::new("model.norm.weight", vec![16]),
        ]
    }

    /// Returns valid headers corresponding to the required test tensors.
    fn valid_headers() -> Vec<TensorHeader> {
        required_tensors()
            .into_iter()
            .map(|requirement| TensorHeader {
                name: requirement.name,
                dtype: "BF16".to_owned(),
                shape: requirement.shape,
                shard: PathBuf::from("model-00001-of-00002.safetensors"),
            })
            .collect()
    }

    /// Verifies required tensors pass while unrelated auxiliary tensors are ignored.
    #[test]
    fn validates_required_headers_and_ignores_auxiliary_tensors() {
        let required = required_tensors();
        let mut headers = valid_headers();
        headers.push(TensorHeader {
            name: "model.visual.patch_embed.weight".to_owned(),
            dtype: "BF16".to_owned(),
            shape: vec![8, 3, 16, 16],
            shard: PathBuf::from("model-00002-of-00002.safetensors"),
        });

        validate_tensor_headers("test model", ModelDType::BF16, &headers, &required)
            .unwrap();
    }

    /// Verifies that a missing required tensor is reported.
    #[test]
    fn rejects_missing_required_tensor() {
        let required = required_tensors();
        let mut headers = valid_headers();
        let missing = headers.pop().unwrap().name;

        let error = validate_tensor_headers(
            "test model",
            ModelDType::BF16,
            &headers,
            &required,
        )
        .unwrap_err();

        assert!(error.to_string().contains("missing required tensor"));
        assert!(error.to_string().contains(&missing));
    }

    /// Verifies that an incorrect required shape is reported.
    #[test]
    fn rejects_wrong_required_tensor_shape() {
        let required = required_tensors();
        let mut headers = valid_headers();
        headers[0].shape = vec![1, 2, 3];
        let name = headers[0].name.clone();

        let error = validate_tensor_headers(
            "test model",
            ModelDType::BF16,
            &headers,
            &required,
        )
        .unwrap_err();

        assert!(error.to_string().contains(&name));
        assert!(error.to_string().contains("expected"));
    }

    /// Verifies that an incorrect required dtype is reported.
    #[test]
    fn rejects_wrong_required_tensor_dtype() {
        let required = required_tensors();
        let mut headers = valid_headers();
        headers[0].dtype = "F32".to_owned();
        let name = headers[0].name.clone();

        let error = validate_tensor_headers(
            "test model",
            ModelDType::BF16,
            &headers,
            &required,
        )
        .unwrap_err();

        assert!(error.to_string().contains(&name));
        assert!(error.to_string().contains("expected BF16"));
    }

    /// Verifies duplicate names across shards are rejected before allocation.
    #[test]
    fn rejects_duplicate_tensor_names_across_shards() {
        let required = required_tensors();
        let mut headers = valid_headers();
        let mut duplicate = headers[0].clone();
        duplicate.shard = PathBuf::from("model-00002-of-00002.safetensors");
        let name = duplicate.name.clone();
        headers.push(duplicate);

        let error = validate_tensor_headers(
            "test model",
            ModelDType::BF16,
            &headers,
            &required,
        )
        .unwrap_err();

        assert!(error.to_string().contains(&name));
        assert!(error.to_string().contains("appears in both"));
    }

    /// Verifies the contract validator accepts F32 tensor storage.
    #[test]
    fn validates_f32_storage_dtype() {
        let required = required_tensors();
        let mut headers = valid_headers();
        for header in &mut headers {
            header.dtype = "F32".to_owned();
        }

        validate_tensor_headers("test model", ModelDType::F32, &headers, &required)
            .unwrap();
    }

    #[test]
    fn accepts_fp32_only_for_parameters_with_an_explicit_storage_policy() {
        let mut required = required_tensors();
        required[1] = required[1].clone().with_f32_storage();
        let mut headers = valid_headers();
        for dtype in ["BF16", "F32"] {
            headers[1].dtype = dtype.to_owned();
            validate_tensor_headers("test", ModelDType::BF16, &headers, &required)
                .unwrap();
        }
        headers[0].dtype = "F32".to_owned();
        assert!(
            validate_tensor_headers("test", ModelDType::BF16, &headers, &required)
                .unwrap_err()
                .to_string()
                .contains("model.embed_tokens.weight")
        );
        headers[0].dtype = "BF16".to_owned();
        for dtype in ["F16", "F64", "I32"] {
            headers[1].dtype = dtype.to_owned();
            let error =
                validate_tensor_headers("test", ModelDType::BF16, &headers, &required)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("expected BF16 or F32"), "{error}");
        }
        headers[0].dtype = "F32".to_owned();
        headers[1].dtype = "BF16".to_owned();
        assert!(
            validate_tensor_headers("test", ModelDType::F32, &headers, &required)
                .is_err()
        );
    }
}
