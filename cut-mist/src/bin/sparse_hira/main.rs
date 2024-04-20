use clap::Parser;
use cuts::{inplace_sct_signed::CutHelper, sparse_cut::helpers::{SparseParameters, SparseSct}};
use rand::thread_rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{borrow::Cow, time::SystemTime};
use tensorboard_writer::TensorboardWriter;
use std::sync::Mutex;
use linya::{Bar, Progress};

use cut_mist::{
    faer::flat_mat,
    hira_helpers::SparseHiraLogger,
    safetensors::serialize_mats,
    sct_map::SafeTensorsDirectoryMap,
};
use cut_mist::safetensors::SerializeSct;

#[derive(Debug, Parser)]
#[command(name = "Sparse HIRA")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates safetensors with sparse sct", long_about = None)]
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
    /// The max number of bits (defaults to the half point)
    #[arg(short = 'z')]
    num_bits: Option<usize>,
    /// The number of iterations (default: 100_000)
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
        num_bits,
        iters,
        threads,
        par,
    } = args;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?
    }
    // TODO! hard-coded for bf16
    let parameters = SparseParameters {
        max_iters: iters.unwrap_or(100_000),
        max_bits: num_bits.unwrap_or(64 * nrows * ncols / 2),
        parallelism: par.map_or(faer::Parallelism::None, |par| faer::Parallelism::Rayon(par)),
    };
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
            if !name.eq("model.layers.11.self_attn.v_proj.weight") {
                return Ok(())
            }
            let view = mapped_tensor.view();
            let out_dir = mapped_tensor.out_dir();
            let title: Cow<_> = if let Some(t) = rayon::current_thread_index() {
                format!("[{t}] {name}").into()
            } else {
                name.into()
            };
            let bar: Bar = progress.lock().unwrap().bar(parameters.max_bits, title);
            let checkpoint = (parameters.max_bits / 100).max(1);
            let mut remainder = flat_mat(view);
            let mut cut_helper = CutHelper::new(nrows, ncols);
            let mut sct_helper = SparseSct::new(0);
            let (mut logger, init_summary) = SparseHiraLogger::new(remainder.as_ref());
            let mut writer = TensorboardWriter::new(out_dir)?;
            writer.write_file_version()?;
            writer.write_summary(SystemTime::now(), 0, init_summary)?;
            writer.flush()?;
            // TODO! hard-coded for `bf16`
            let mut num_bits = 0;
            for i in 0.. {
                let cut = cut_helper.sparse_cut_mat(
                    remainder.as_mut(),
                    rng,
                    parameters.max_iters,
                    parameters.parallelism,
                );
                // println!("l2^2 = {}", remainder.as_ref().squared_norm_l2());
                num_bits += cut.num_bits();
                dbg!(num_bits, parameters.max_bits);
                if num_bits > parameters.max_bits {
                    break
                }
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
            sct_helper.serialize(out_dir, big_mats)?;
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
