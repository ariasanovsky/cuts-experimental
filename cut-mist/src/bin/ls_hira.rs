use std::time::SystemTime;

use clap::Parser;
use cuts::{
    ls_sct::{CutSetBuffers, CutSetLdl},
    sct_helper::Sct,
};

use hira::{
    faer::flat_mat,
    half_point,
    hira_helpers::{HiraParameters, LdlHiraLogger},
    safetensors::{serialize_mats, serialize_sct},
    sct_map::SafeTensorsDirectoryMap,
};
use rand::thread_rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tensorboard_writer::TensorboardWriter;

#[derive(Debug, Parser)]
#[command(name = "Least Squares HIRA")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates safetensors with a least-squares HIRA", long_about = None)]
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
    /// The number of iterations (defaults to 40)
    #[arg(short = 'i')]
    iters: Option<usize>,
    /// The number of threads to use (defaults to the maximum available)
    #[arg(short = 'T')]
    threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    dbg!(&args);
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
    } = args;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?
    }
    let mut parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank: rank.unwrap_or_else(|| half_point::<half::bf16>(nrows, ncols)),
        parallelism: faer::Parallelism::None,
    };
    if parameters.rank % 4 != 0 {
        let r = parameters.rank;
        parameters.rank = parameters.rank.next_multiple_of(4);
        eprintln!("rank rounded from {r} to {} (multiple of 4)", parameters.rank);
    }
    let directory = SafeTensorsDirectoryMap::new(tensors, out_dir, nrows, ncols, &parameters)?;
    let deserialized_tensors = directory.deserialize()?;
    let filtered_tensors = deserialized_tensors.filter_tensors(nrows, ncols)?;
    filtered_tensors
        .tensors()
        .into_par_iter()
        .enumerate()
        .try_for_each_init(thread_rng, |rng, (i, mapped_tensor)| -> eyre::Result<_> {
            // let MappedTensor {
            //     old_tensor,
            //     out_dir,
            // } = mapped_tensor;
            let _name = mapped_tensor.name();
            let view = mapped_tensor.view();
            let out_dir = mapped_tensor.out_dir();
            
            let mut remainder = flat_mat(view);
            eprintln!("tensor[{i}] to {out_dir:?}");
            let mut ldl = CutSetLdl::new(nrows, ncols, parameters.rank);
            let mut sct = Sct::new(nrows, ncols, parameters.rank);
            let mut cutset = CutSetBuffers::new(nrows, ncols);
            let (mut logger, init_summary) = LdlHiraLogger::new(remainder.as_ref());
            let mut writer = TensorboardWriter::new(out_dir)?;
            writer.write_file_version()?;
            writer.write_summary(SystemTime::now(), 0, init_summary)?;
            writer.flush()?;
            for i in 1..=parameters.rank {
                if i % 1_024 == 0 {
                    if let Some(t) = rayon::current_thread_index() {
                        eprintln!("i = {i} (thread {t})");
                    } else {
                        eprintln!("i = {i} (thread ?)");
                    }
                }
                let cut = cutset.write_cut(remainder.as_ref(), rng, parameters.max_iters);
                sct.extend_with(cutset.s_signs(), cutset.t_signs(), cut.value);
                let mut projection_buffer = faer::Col::zeros(i - 1);
                let improved_cut = ldl.add_column(&mut projection_buffer, &sct, &cut, &mut remainder);
                let log = logger.log_cut(&cut, &improved_cut);
                writer.write_summary(SystemTime::now(), i as _, log)?;
                // assert_eq!(logger.remainder_norm_squared(), remainder.as_ref().squared_norm_l2());
                writer.flush()?;
            }
            serialize_sct(out_dir, &sct, big_mats)?;
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
