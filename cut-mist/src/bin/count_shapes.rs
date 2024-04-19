use std::{collections::BTreeMap, fs::File};

use clap::Parser;
use hira::half_point;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct SafeTensorsPath {
    #[arg(short = 's', value_name = "SOURCE")]
    source: String,
}

fn main() -> eyre::Result<()> {
    // TODO! lol
    let SafeTensorsPath { source } = SafeTensorsPath::try_parse()?;
    assert!(source.ends_with(".safetensors"));
    let mut names_by_shape: BTreeMap<&[usize], Vec<&str>> = BTreeMap::new();
    let file = File::open(source)?;
    let buffer = unsafe { MmapOptions::new().map(&file) }?;
    let tensors = SafeTensors::deserialize(&buffer)?;
    let (size, metadata) = SafeTensors::read_metadata(&buffer)?;
    dbg!(size, metadata.metadata());
    let t = tensors.tensors();
    t.iter().for_each(|(name, view)| {
        let entry = names_by_shape.entry(view.shape());
        entry
            .and_modify(|names| names.push(name))
            .or_insert_with(|| vec![name]);
    });
    // TODO! `https://crates.io/crates/cli-table`
    for (shape, names) in names_by_shape {
        let half_point: String = if let &[nrows, ncols] = shape {
            format!("{}", half_point::<half::bf16>(nrows, ncols))
        } else {
            String::from("N/A")
        };
        println!(
            "{:>3} with shape {:?} (k{{0.5}} = {}): \t[\"{}\", ...]",
            names.len(),
            shape,
            half_point,
            names[0],
        );
    }
    Ok(())
}
