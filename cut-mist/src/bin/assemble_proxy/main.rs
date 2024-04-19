use clap::Parser;

use crate::assemble::{write_proxy, OriginalTensors, ProxyTensors};

mod assemble;

#[derive(Debug, Parser)]
#[command(name = "Collect proxies")]
// #[command(version = "0.1.0")]
#[command(about = "Collects `a_bf16` matrices generated from a run at a given time", long_about = None)]
struct Args {
    /// Directory with the original `safetensors`
    #[arg(short = 't')]
    tensor_dir: std::path::PathBuf,
    /// Directory where `saftensors` proxies were assmebled
    #[arg(short = 'p')]
    proxy_dir: std::path::PathBuf,
    /// Directory where `saftensors` proxy will be written
    #[arg(short = 'o')]
    out_dir: std::path::PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    // dbg!(&args);
    let Args {
        tensor_dir,
        proxy_dir,
        out_dir,
    } = args;
    let original_tensors = OriginalTensors::new(tensor_dir)?;
    let proxy_tensors = ProxyTensors::new(proxy_dir)?;
    write_proxy(original_tensors, proxy_tensors, out_dir)
}
