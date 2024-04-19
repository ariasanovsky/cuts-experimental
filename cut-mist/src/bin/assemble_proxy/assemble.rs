use std::{
    collections::{BTreeMap, HashMap},
    ffi::OsString,
    fs::File,
    path::Path,
};

use eyre::OptionExt;
use memmap2::{Mmap, MmapOptions};
use safetensors::{serialize_to_file, SafeTensors};

pub struct OriginalTensors(HashMap<OsString, Mmap>);

impl OriginalTensors {
    pub fn new<P: AsRef<Path>>(tensor_dir: P) -> eyre::Result<Self> {
        let original_tensors = tensor_dir
            .as_ref()
            .read_dir()?
            .filter_map(|entry| {
                let entry = entry.unwrap().path();
                if entry.extension().is_some_and(|ext| ext.eq("safetensors")) {
                    Some(entry)
                } else {
                    None
                }
            })
            .map(|tensor_path| -> eyre::Result<_> {
                let stem = tensor_path.file_stem().ok_or_eyre("Failed to read stem.")?;
                let file = File::open(&tensor_path)?;
                let buffer = unsafe { MmapOptions::new().map(&file) }?;
                Ok((stem.to_owned(), buffer))
            })
            .collect::<eyre::Result<_>>()?;
        Ok(Self(original_tensors))
    }
}

/// Maps a time to a map from model name to buffer.
pub struct ProxyTensors(HashMap<OsString, HashMap<OsString, Mmap>>);

impl ProxyTensors {
    pub fn new<P: AsRef<Path>>(proxy_tensors: P) -> eyre::Result<Self> {
        let proxy_tensors = proxy_tensors
            .as_ref()
            .read_dir()?
            .map(|entry| {
                let time_dir = entry.unwrap();
                // dbg!(&time_dir);
                let time = time_dir.file_name();
                // dbg!(&time);
                let proxies = time_dir
                    .path()
                    .read_dir()
                    .unwrap()
                    .filter_map(|proxy| {
                        let proxy_tensors = proxy.unwrap().path();
                        // dbg!(&proxy_tensors);
                        if proxy_tensors
                            .extension()
                            .is_some_and(|ext| ext.eq("safetensors"))
                        {
                            // dbg!();
                            Some(proxy_tensors)
                        } else {
                            // dbg!();
                            None
                        }
                    })
                    .map(|tensor_path| {
                        let stem = tensor_path.file_stem().ok_or_eyre("Failed to read stem.")?;
                        let file = File::open(&tensor_path)?;
                        let buffer = unsafe { MmapOptions::new().map(&file) }?;
                        Ok((stem.to_owned(), buffer))
                    })
                    .collect::<eyre::Result<HashMap<_, _>>>()?;
                // dbg!(proxies.len());
                Ok((time, proxies))
            })
            .collect::<eyre::Result<HashMap<_, _>>>()?;
        // dbg!(proxy_tensors.len());
        Ok(Self(proxy_tensors))
    }
}

/// For each model `m`, writes a new `m.safetensors` inside `out_dir`.
/// For each tensor `t` in `m`, writes the newest proxy tensor for `t`.
/// Falls back to the original `t` in `m` when not found.
pub fn write_proxy<P: AsRef<Path>>(
    tensors: OriginalTensors,
    proxies: ProxyTensors,
    out_dir: P,
) -> eyre::Result<()> {
    let original_tensors = tensors
        .0
        .iter()
        .map(|(model, buffer)| {
            Ok((
                model,
                (
                    SafeTensors::deserialize(buffer)?,
                    SafeTensors::read_metadata(buffer)?,
                ),
            ))
        })
        .collect::<eyre::Result<HashMap<_, _>>>()?;
    let tensor_views = original_tensors
        .iter()
        .map(|(m, (tensors, metadata))| {
            let tensors = tensors.tensors().into_iter().collect::<HashMap<_, _>>();
            (m as &OsString, (tensors, metadata))
        })
        .collect::<HashMap<_, _>>();
    // for (m, views) in tensor_views.iter() {
    //     println!("{m:?}");
    //     for (tensor, views) in views.iter().filter(|(_, v)| {
    //         v.shape().contains(&1024)
    //     }) {
    //         println!("\t{tensor}")
    //     }
    // }
    // dbg!(proxies.0.len());
    let proxy_tensors = proxies
        .0
        .iter()
        .map(|(t, buffers)| {
            let views = buffers
                .iter()
                .map(|(m, buffer)| Ok((m, SafeTensors::deserialize(buffer)?)))
                .collect::<eyre::Result<HashMap<_, _>>>()?;
            // dbg!(views.len());
            Ok((t, views))
        })
        .collect::<eyre::Result<HashMap<_, _>>>()?;
    // dbg!(proxy_tensors.len());
    let proxy_views = proxy_tensors
        .iter()
        .map(|(t, tensors)| {
            let views = tensors
                .iter()
                .map(|(m, tensors)| {
                    let tensors = tensors.tensors().into_iter().collect::<HashMap<_, _>>();
                    Ok((m as &OsString, tensors))
                })
                .collect::<eyre::Result<HashMap<_, _>>>()?;
            Ok((t as &OsString, views))
        })
        .collect::<eyre::Result<BTreeMap<_, _>>>()?;
    // for (t, views) in proxy_views.iter() {
    //     println!("{t:?}: {} models", views.len());
    //     for (m, views) in views.iter() {
    //         println!("\t{m:?}: {} tensors", views.len());
    //         for (tensor, views) in views.iter().filter(|(_, v)| {
    //             true
    //         }) {
    //             println!("\t\t{tensor} w/ shape {:?}", views.shape())
    //         }
    //     }
    // }
    tensor_views
        .into_iter()
        .try_for_each(|(m, (views, metadata))| -> eyre::Result<_> {
            let data = views.iter().map(|(tensor, original_view)| {
                let youngest_proxy = proxy_views
                    .iter()
                    .rev()
                    .find_map(|(_, views)| views.get(m).and_then(|tensors| tensors.get(tensor)));
                // if youngest_proxy.is_some() {
                //     println!("proxy found: {tensor}")
                // }
                (tensor, youngest_proxy.unwrap_or(original_view))
            });
            let proxy_path = out_dir.as_ref().join(m).with_extension("safetensors");
            Ok(serialize_to_file(data, metadata.1.metadata(), &proxy_path)?)
        })
}
