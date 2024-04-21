use std::time::SystemTime;

use chrono::Utc;
use clap::Parser;
use cut_mist::{half_point, hira_helpers::{HiraLogger, HiraParameters}};
use cuts::{inplace_sct_signed::CutHelper, sct_helper::Sct};
use faer::Mat;
use rand::distributions::Distribution;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tensorboard_writer::{proto::tensorboard::Summary, SummaryBuilder, TensorboardWriter};

#[derive(Debug, Parser)]
#[command(name = "Unitary SCT")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates unitary matrices with sct decompositions", long_about = None)]
struct Args {
    #[arg(short = 'o')]
    out_dir: std::path::PathBuf,
    /// The number of rows in our target matrices
    #[arg(short = 'd')]
    dim: usize,
    /// The rank (defaults to the half point)
    #[arg(short = 'r')]
    rank: Option<usize>,
    /// The number of iterations (default: 40)
    #[arg(short = 'i')]
    iters: Option<usize>,
    /// The number of threads to use in mat{vec/mat} (default: to 0)
    #[arg(short = 'P')]
    par: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    let Args {
        out_dir,
        dim,
        rank,
        iters,
        par,
    } = args;
    let parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank: rank.unwrap_or_else(|| half_point::<f64>(dim, dim)),
        parallelism: par.map_or(faer::Parallelism::None, |par| faer::Parallelism::Rayon(par)),
    };
    let time = Utc::now().to_rfc3339();
    let out_dir = out_dir
        .join(format!("dim.{dim}"))
        .join(format!("{parameters}"));

    let unitary: Mat<f64> = faer::stats::UnitaryMat {
        dimension: dim,
    }.sample(&mut rand::thread_rng());
    let remainder_f32 = Mat::from_fn(dim, dim, |row, col| {
        let a = unitary[(row, col)];
        let b = a as f32;
        a - b as f64
    }).norm_l2();
    let mut writer = TensorboardWriter::new(out_dir.join("remainder_f32").join(&time))?;
    writer.write_file_version()?;
    for i in 1..=parameters.rank {
        let log = log_remainder(remainder_f32);
        let time = SystemTime::now();
        writer.write_summary(time.clone(), i as _, log)?;
    }
    writer.flush()?;
    let remainder_bf16 = Mat::from_fn(dim, dim, |row, col| {
        let a = unitary[(row, col)];
        let b = half::bf16::from_f64(a);
        a - b.to_f64()
    }).norm_l2();
    let mut writer = TensorboardWriter::new(out_dir.join("remainder_bf16").join(&time))?;
    writer.write_file_version()?;
    for i in 1..=parameters.rank {
        let log = log_remainder(remainder_bf16);
        let time = SystemTime::now();
        writer.write_summary(time.clone(), i as _, log)?;
    }
    writer.flush()?;
    
    (0..4).into_par_iter().try_for_each_init(rand::thread_rng, |rng, i| -> eyre::Result<()> {
        let (mut remainder, name): (Mat<f64>, &str) = match i {
            0 => (Mat::identity(dim, dim), "identity"),
            1 => (faer::stats::UnitaryMat {
                dimension: dim,
            }.sample(rng), "unitary"),
            2 => {
                let half_graph = Mat::from_fn(dim, dim, |row, col| {
                    if row + col < dim {
                        1.0
                    } else {
                        0.0
                    }
                });
                (half_graph, "half")
            }
            3 => {
                let sum = Mat::from_fn(dim, dim, |row, col| {
                    (row + col) as f64
                });
                (sum, "sum")
            },
            _ => unreachable!(),
        };
        let out_dir = out_dir.join(name).join(&time);
        let mut cut_helper = CutHelper::new(dim, dim);
        let mut sct_helper = Sct::new(dim, dim, parameters.rank);
        let (mut logger, init_summary) = HiraLogger::new(remainder.as_ref());
        let mut writer = TensorboardWriter::new(out_dir)?;
        let time = SystemTime::now();
        writer.write_file_version()?;
        writer.write_summary(time.clone(), 0, init_summary)?;
        let sigma = remainder.singular_values()[0];
        let sigma = log_singular_value(sigma);
        writer.write_summary(time.clone(), 0, sigma)?;
        writer.flush()?;
        for i in 1..=parameters.rank {
            let cut = cut_helper.cut_mat(
                remainder.as_mut(),
                rng,
                parameters.max_iters,
                parameters.parallelism,
            );
            let s_signs = cut_helper.s_signs();
            let t_signs = cut_helper.t_signs();
            sct_helper.extend_with(s_signs, t_signs, cut.value);
            let log = logger.log_cut(&cut);
            let time = SystemTime::now();
            writer.write_summary(time.clone(), i as _, log)?;
            if i % 64 == 0 {
                let sigma = remainder.singular_values()[0];
                let sigma = log_singular_value(sigma);
                writer.write_summary(time.clone(), i as _, sigma)?;
            }
            writer.flush()?;
            if i % 1024 == 0 {
                eprintln!("{i}")
            }        
        }
        
        Ok(())
    })?;
    Ok(())
}

fn log_singular_value(sigma: f64) -> Summary {
    let summary = SummaryBuilder::new()
        .scalar("singular_value/value", sigma as _);
    summary.build()
}

fn log_remainder(remainder: f64) -> Summary {
    let summary = SummaryBuilder::new()
        .scalar("remainder_norm/value", remainder as _)
        .scalar("remainder_norm/value_over_init", remainder as _);
    summary.build()
}
