//! Safetensors file loading and metadata validation.

use crate::ModelLoadError;
use memmap2::{Mmap, MmapOptions};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

pub struct ModelShard {
    path: PathBuf,
    byte_len: usize,
    mmap: Mmap,
    tensors: Vec<TensorInfo>,
}

impl ModelShard {
    /// Returns the safetensors shard path.
    ///
    /// The method borrows the shard and returns the path loaded during
    /// construction. It cannot fail.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the mapped shard byte length.
    ///
    /// The method borrows the shard metadata and returns the file size. It cannot
    /// fail after construction succeeds.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Returns the number of tensors in this shard.
    ///
    /// The method borrows the shard metadata and returns the tensor count. It
    /// cannot fail after safetensors parsing succeeds.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns tensor metadata collected from this shard.
    ///
    /// The method borrows metadata for names, dtypes, shapes, and byte sizes. It
    /// cannot fail after construction succeeds.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Deserializes safetensors views over this shard's memory map.
    ///
    /// The method borrows the shard and returns zero-copy tensor views. It returns
    /// `ModelLoadError` if the mapped file no longer parses as safetensors.
    pub fn safetensors(
        &self,
    ) -> Result<::safetensors::SafeTensors<'_>, ModelLoadError> {
        ::safetensors::SafeTensors::deserialize(&self.mmap)
            .map_err(|source| ModelLoadError::Safetensors(self.path.clone(), source))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    byte_len: usize,
}

impl TensorInfo {
    /// Returns the tensor name from the safetensors header.
    ///
    /// The method borrows metadata and returns the tensor key. It cannot fail after
    /// safetensors validation succeeds.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor dtype from the safetensors header.
    ///
    /// The method borrows metadata and returns dtype text such as `BF16`. It cannot
    /// fail after safetensors validation succeeds.
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Returns the tensor shape from the safetensors header.
    ///
    /// The method borrows metadata and returns the dimensions. It cannot fail after
    /// safetensors validation succeeds.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor byte length from the safetensors header.
    ///
    /// The method borrows metadata and returns the tensor data length in bytes. It
    /// cannot fail after safetensors validation succeeds.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Loads all safetensors shards from a model directory.
///
/// The input root is scanned for files with the `.safetensors` extension. The
/// function returns loaded shard metadata sorted by path or `ModelLoadError` for
/// missing, unreadable, or malformed safetensors files.
pub(crate) fn load_safetensors_shards(
    root: &Path,
) -> Result<Vec<ModelShard>, ModelLoadError> {
    let mut paths = fs::read_dir(root)
        .map_err(|source| ModelLoadError::Io(root.to_path_buf(), source))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|source| ModelLoadError::Io(root.to_path_buf(), source))
        })
        .collect::<Result<Vec<_>, _>>()?;

    paths.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    paths.sort();

    if paths.is_empty() {
        return Err(ModelLoadError::MissingFile(root.join("*.safetensors")));
    }

    paths
        .iter()
        .map(|path| load_safetensors_shard(path))
        .collect()
}

/// Verifies that loaded safetensors shards contain model tensors.
///
/// The input shards are already parsed metadata. The function returns `Ok(())`
/// when at least one tensor exists or `ModelLoadError::InvalidConfig` when the
/// files are structurally valid but empty.
pub(crate) fn validate_safetensors_shards(
    shards: &[ModelShard],
) -> Result<(), ModelLoadError> {
    let tensor_count = shards.iter().map(ModelShard::tensor_count).sum::<usize>();
    if tensor_count == 0 {
        Err(ModelLoadError::InvalidConfig(
            "safetensors model contains no tensors".to_owned(),
        ))
    } else {
        Ok(())
    }
}

/// Loads and validates one safetensors shard.
///
/// The input path is memory-mapped and parsed with the safetensors crate. The
/// function returns shard metadata and keeps the map alive for future inference, or
/// `ModelLoadError` for I/O and safetensors parsing failures.
fn load_safetensors_shard(path: &Path) -> Result<ModelShard, ModelLoadError> {
    require_safetensors_file(path)?;
    let file = File::open(path)
        .map_err(|source| ModelLoadError::Io(path.to_path_buf(), source))?;

    let mmap = {
        // Safety: the file is opened read-only and the mapping is kept alive inside
        // `ModelShard`; this code never mutates the mapped file contents.
        unsafe { MmapOptions::new().map(&file) }
    }
    .map_err(|source| ModelLoadError::Io(path.to_path_buf(), source))?;
    let tensors = ::safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|source| ModelLoadError::Safetensors(path.to_path_buf(), source))?;
    let tensor_infos = tensors
        .iter()
        .map(|(name, view)| TensorInfo {
            name: name.to_owned(),
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            byte_len: view.data().len(),
        })
        .collect::<Vec<_>>();

    Ok(ModelShard {
        path: path.to_path_buf(),
        byte_len: mmap.len(),
        mmap,
        tensors: tensor_infos,
    })
}

/// Ensures a safetensors path exists as a file.
///
/// The input path is checked with filesystem metadata. The function returns
/// `Ok(())` for files and `ModelLoadError::MissingFile` for absent paths or
/// directories.
fn require_safetensors_file(path: &Path) -> Result<(), ModelLoadError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(ModelLoadError::MissingFile(path.to_path_buf()))
    }
}
