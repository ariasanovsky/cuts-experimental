use clap::Parser;
use cuts::{inplace_sct_signed::CutHelper, sct_helper::Sct};
use rand::thread_rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{borrow::Cow, time::SystemTime};
use tensorboard_writer::TensorboardWriter;
use std::sync::Mutex;
use linya::{Bar, Progress};

use hira::{
    faer::flat_mat,
    half_point,
    hira_helpers::{HiraLogger, HiraParameters},
    safetensors::{serialize_mats, serialize_sct},
    sct_map::SafeTensorsDirectoryMap,
};

#[derive(Debug, Parser)]
#[command(name = "HIRA")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates safetensors with cutsets approximation", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 't')]
    tensors: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    out_dir: std::path::PathBuf,
    /// The number of rows in our target matrices
    #[arg(short = 'm')]
    nrows: usize,
    /// The number of columns in our target matrices
    #[arg(short = 'n')]
    ncols: usize,
    /// Also writes `a`, `r`, `s`, and `t` as `f64` matrices
    // #[arg(short = 'h')]
    // huge_mats: bool,
    // /// Also writes `a`, `r`, `s`, and `t` as `f64` matrices
    #[arg(short = 'b')]
    big_mats: bool,
    /// Also writes `a` and `r` as `bf16` matrices
    #[arg(short = 's')]
    small_mats: bool,
    // /// Suppresses tensorboard event logs
    // #[arg(short = 'L')]
    // no_log: bool,
    /// The rank (defaults to the half point)
    #[arg(short = 'r')]
    rank: Option<usize>,
    /// The number of iterations (default: 40)
    #[arg(short = 'i')]
    iters: Option<usize>,
    /// The number of tensors to process in parallel (default: maximum available)
    #[arg(short = 'T')]
    threads: Option<usize>,
    /// The number of threads to use in mat{vec/mat} (default: to 0)
    #[arg(short = 'P')]
    par: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    // dbg!(&args);
    let Args {
        tensors,
        out_dir,
        nrows,
        ncols,
        big_mats,
        small_mats,
        // no_log,
        rank,
        iters,
        threads,
        par,
    } = args;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?
    }
    let parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank: rank.unwrap_or_else(|| half_point::<half::bf16>(nrows, ncols)),
        parallelism: par.map_or(faer::Parallelism::None, |par| faer::Parallelism::Rayon(par)),
    };
    // dbg!(parameters.rank);
    let directory = SafeTensorsDirectoryMap::new(tensors, out_dir, nrows, ncols, &parameters)?;
    let deserialized_tensors = directory.deserialize()?;
    let filtered_tensors = deserialized_tensors.filter_tensors(nrows, ncols)?;
    let progress = Mutex::new(Progress::new());
    filtered_tensors
        .tensors()
        .into_par_iter()
        .enumerate()
        .try_for_each_init(thread_rng, |rng, (_i, mapped_tensor)| -> eyre::Result<_> {
            let name = mapped_tensor.name();
            let view = mapped_tensor.view();
            let out_dir = mapped_tensor.out_dir();
            let title: Cow<_> = if let Some(t) = rayon::current_thread_index() {
                format!("[{t}] {name}").into()
            } else {
                name.into()
            };
            let bar: Bar = progress.lock().unwrap().bar(parameters.rank, title);
            let checkpoint = (parameters.rank / 100).max(1);
            // let MappedTensor {
            //     old_tensor,
            //     out_dir,
            // } = mapped_tensor;
            // if !out_dir.as_os_str().to_string_lossy().contains("model.layers.4.self_attn.k_proj.weight") {
            //     return Ok(())
            // }
            // eprintln!("tensor[{i}] to {out_dir:?}");
            let mut remainder = flat_mat(view);
            let mut cut_helper = CutHelper::new(nrows, ncols);
            let mut sct_helper = Sct::new(nrows, ncols, parameters.rank);
            let (mut logger, init_summary) = HiraLogger::new(remainder.as_ref());
            let mut writer = TensorboardWriter::new(out_dir)?;
            writer.write_file_version()?;
            writer.write_summary(SystemTime::now(), 0, init_summary)?;
            writer.flush()?;
            for i in 1..=parameters.rank {
                // if i % 1_024 == 0 {
                //     if let Some(t) = rayon::current_thread_index() {
                //         eprintln!("i = {i} (thread {t})");
                //     } else {
                //         eprintln!("i = {i} (thread ?)");
                //     }
                // }
                let cut = cut_helper.cut_mat(
                    remainder.as_mut(),
                    // approximat.as_mut(),
                    rng,
                    parameters.max_iters,
                    parameters.parallelism,
                );
                let s_signs = cut_helper.s_signs();
                let t_signs = cut_helper.t_signs();
                sct_helper.extend_with(s_signs, t_signs, cut.value);
                let log = logger.log_cut(&cut);
                writer.write_summary(SystemTime::now(), i as _, log)?;
                writer.flush()?;
                if i % checkpoint == 0 {
                    progress.lock().unwrap().set_and_draw(&bar, i)
                }
            }
            serialize_sct(out_dir, &sct_helper, big_mats)?;
            let mut approximat = flat_mat(view).to_faer();
            approximat -= remainder.as_ref();
            serialize_mats(
                out_dir,
                approximat.as_ref(),
                remainder.as_ref(),
                big_mats,
                small_mats,
            )?;
            Ok(())
        })
}
