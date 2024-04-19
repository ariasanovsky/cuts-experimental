pub mod proxy_collection;

use clap::Parser;
use proxy_collection::ModelOutDirs;

#[derive(Debug, Parser)]
#[command(name = "Collect proxies")]
// #[command(version = "0.1.0")]
#[command(about = "Collects `a_bf16` matrices generated from a run at a given time", long_about = None)]
struct Args {
    /// Time `t` when `safetensors` were generated
    #[arg(short = 't')]
    time: String,
    /// Ouput directory where `safetensors` were written at time `t`
    #[arg(short = 'o')]
    out_dir: std::path::PathBuf,
    /// Proxy directory to assembly `saftensors` from time `t`
    #[arg(short = 'p')]
    proxy_dir: std::path::PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    // dbg!(&args);
    let Args {
        time,
        out_dir,
        proxy_dir,
    } = args;
    let model_outdirs = ModelOutDirs::new(out_dir)?;
    let tensor_paths = model_outdirs.tensor_paths(&time)?;
    let proxies = tensor_paths.safetensors()?;
    proxies.write_together(proxy_dir, &time)
}
