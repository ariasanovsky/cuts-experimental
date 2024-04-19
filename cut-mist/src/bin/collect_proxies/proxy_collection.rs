use std::{
    collections::HashMap,
    ffi::OsString,
    fs::{create_dir_all, DirEntry, File},
    path::Path,
};

use memmap2::{Mmap, MmapOptions};
use safetensors::{serialize_to_file, SafeTensors};

pub struct ModelOutDirs(HashMap<OsString, DirEntry>);

impl ModelOutDirs {
    pub fn new<P: AsRef<Path>>(out_dir: P) -> eyre::Result<Self> {
        let models = out_dir
            .as_ref()
            .read_dir()?
            .map(|model_entry| {
                let model_dir = model_entry?;
                let model_name = model_dir.file_name();
                // dbg!(&model_name);
                Ok((model_name, model_dir))
            })
            .collect::<eyre::Result<HashMap<_, _>>>()?;
        Ok(Self(models))
    }

    pub fn tensor_paths(&self, time: &str) -> eyre::Result<TensorPaths> {
        let tensor_paths = self
            .0
            .iter()
            .map(
                |(model_name, model_dir)| -> eyre::Result<(_, Vec<TensorPath>)> {
                    let tensor_paths = model_dir
                        .path()
                        .read_dir()?
                        .flat_map(|entry| {
                            let shape_entry = entry.unwrap();
                            // dbg!(&shape_entry);
                            let parameters =
                                shape_entry.path().read_dir().unwrap().flat_map(|entry| {
                                    let parameter_entry = entry.unwrap();
                                    // dbg!(&parameter_entry);
                                    parameter_entry
                                        .path()
                                        .read_dir()
                                        .unwrap()
                                        .map(|entry| entry.unwrap())
                                });
                            parameters.filter_map(|tensor_entry| {
                                let time_dir = tensor_entry.path().join(time);
                                if time_dir.exists() {
                                    let tensor_name = tensor_entry.file_name();
                                    let sct_path = time_dir.join("mats.safetensors");
                                    // dbg!(&tensor_name, &sct_path);
                                    let sct_file = File::open(sct_path).unwrap();
                                    let buffer =
                                        unsafe { MmapOptions::new().map(&sct_file) }.unwrap();
                                    Some(TensorPath(tensor_name, buffer))
                                } else {
                                    None
                                }
                            })
                        })
                        .collect();
                    Ok((model_name, tensor_paths))
                },
            )
            .collect::<eyre::Result<HashMap<_, _>>>()?;
        Ok(TensorPaths(tensor_paths))
    }
}

pub struct TensorPaths<'a>(HashMap<&'a OsString, Vec<TensorPath>>);

pub struct TensorPath(OsString, Mmap);

impl<'a> TensorPaths<'a> {
    pub fn safetensors(&self) -> eyre::Result<SafeTensorsProxies> {
        let proxies = self
            .0
            .iter()
            .map(|(model_name, tensor_paths)| {
                let proxy_tensors: Vec<_> = tensor_paths
                    .iter()
                    .map(|tensor_path| {
                        let TensorPath(tensor_name, buffer) = tensor_path;
                        let tensors = SafeTensors::deserialize(buffer).unwrap();
                        ProxySafeTensors(tensor_name, tensors)
                    })
                    .collect();
                (model_name as _, proxy_tensors)
            })
            .collect();
        Ok(SafeTensorsProxies(proxies))
    }
}

pub struct SafeTensorsProxies<'a>(HashMap<&'a OsString, Vec<ProxySafeTensors<'a>>>);

pub struct ProxySafeTensors<'a>(&'a OsString, SafeTensors<'a>);

impl<'a> SafeTensorsProxies<'a> {
    pub fn write_together<P: AsRef<Path>>(&self, proxy_dir: P, time: &str) -> eyre::Result<()> {
        let views = self
            .0
            .iter()
            .map(|(map_name, tensors)| {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        let ProxySafeTensors(tensor_name, tensors) = t;
                        (tensor_name as &OsString, tensors.tensor("a_bf16").unwrap())
                    })
                    .collect::<Vec<_>>();
                (map_name as &OsString, tensors)
            })
            .collect::<HashMap<_, _>>();
        let proxy_dir = proxy_dir.as_ref().join(time);
        create_dir_all(&proxy_dir)?;
        for (model_name, views) in views {
            let proxy_path = proxy_dir.join(model_name).with_extension("safetensors");
            let data = views
                .into_iter()
                .map(|(tensor_name, view)| (tensor_name.to_string_lossy(), view));
            serialize_to_file(data, &None, &proxy_path)?
        }
        Ok(())
    }
}
