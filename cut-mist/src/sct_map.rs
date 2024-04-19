use std::{
    fs::File,
    path::{Path, PathBuf},
};

use chrono::Utc;
use eyre::OptionExt;
use memmap2::{Mmap, MmapOptions};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

pub struct SafeTensorsDirectoryMap(Vec<(Mmap, PathBuf)>);

impl SafeTensorsDirectoryMap {
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>>(
        tensors: P,
        out_dir: Q,
        nrows: usize,
        ncols: usize,
        parameters: &impl core::fmt::Display,
    ) -> eyre::Result<Self> {
        let safe_tensor_files = tensors
            .as_ref()
            .read_dir()?
            .filter_map(|entry| {
                entry
                    .ok()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext.eq("safetensors")))
            })
            .map(|file| -> eyre::Result<_> {
                let stem = file.file_stem().ok_or_eyre("Failed to read stem.")?;
                let file = File::open(&file)?;
                let buffer = unsafe { MmapOptions::new().map(&file) }?;
                let out_dir = out_dir
                    .as_ref()
                    .join(stem)
                    .join(format!("nrows.{nrows}.ncols.{ncols}"))
                    .join(format!("{parameters}"));
                Ok((buffer, out_dir))
            })
            .collect::<eyre::Result<_>>()?;
        Ok(Self(safe_tensor_files))
    }

    pub fn deserialize(&self) -> eyre::Result<DeserializedMap<'_>> {
        let buffered_tensors = self
            .0
            .iter()
            .map(|(buffer, out_dir)| {
                let tensors = SafeTensors::deserialize(buffer)?;
                Ok((tensors, out_dir))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        Ok(DeserializedMap(buffered_tensors))
    }
}

pub struct DeserializedMap<'a>(Vec<(SafeTensors<'a>, &'a PathBuf)>);

impl<'a> DeserializedMap<'a> {
    pub fn filter_tensors(&self, nrows: usize, ncols: usize) -> eyre::Result<MappedTensors> {
        let now = Utc::now().to_rfc3339();
        let tensors = self
            .0
            .iter()
            .flat_map(|(t, o)| {
                t.tensors()
                    .into_iter()
                    .filter(|(_, view)| {
                        assert_eq!(view.dtype(), Dtype::BF16);
                        view.shape().eq(&[nrows, ncols])
                    })
                    .map(|(name, view)| MappedTensor {
                        name: name.clone(),
                        old_tensor: view,
                        out_dir: o.join(name).join(&now),
                    })
            })
            .collect::<Vec<_>>();
        Ok(MappedTensors(tensors))
    }
}

pub struct MappedTensor<'a> {
    name: String,
    old_tensor: TensorView<'a>,
    out_dir: PathBuf,
}

impl<'a> MappedTensor<'a> {
    pub fn view(&self) -> &TensorView {
        &self.old_tensor
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn out_dir(&self) -> &PathBuf {
        &self.out_dir
    }
}

pub struct MappedTensors<'a>(Vec<MappedTensor<'a>>);

impl<'a> MappedTensors<'a> {
    pub fn tensors(&self) -> &[MappedTensor<'a>] {
        self.0.as_slice()
    }
}
