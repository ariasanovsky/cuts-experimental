use clap::Parser;
use cut_mist::hira_helpers::HiraParameters;
use cuts::inplace_sct_signed::CutHelperV2;
use faer::Mat;
use rand::distributions::Distribution;
use rand::prelude::*;

#[derive(Debug, Parser)]
#[command(name = "Unitary SCT")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates unitary matrices with sct decompositions", long_about = None)]
struct Args {
    #[arg(short = 'd')]
    dim: usize,
    /// The number of iterations (default: 40)
    #[arg(short = 'i')]
    iters: Option<usize>,
    /// The number of threads to use in mat{vec/mat} (default: to 1)
    #[arg(short = 'P')]
    par: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    let Args {
        dim: max_dim,
        iters,
        par,
    } = args;
    let parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank: usize::MAX,
        parallelism: par.map_or(faer::Parallelism::None, |par| faer::Parallelism::Rayon(par)),
    };

    let mut dim = 0usize;
    let incr = max_dim;
    // let incr = 64;
    while dim < max_dim {
        dim += incr;

        let rng = &mut StdRng::seed_from_u64(0);
        let nrows = dim;
        let ncols = dim;
        // let mut remainder: Mat<f64> = faer::stats::UnitaryMat { dimension: dim }.sample(rng);
        let mut remainder: Mat<f64> = faer::stats::StandardNormalMat { nrows, ncols }.sample(rng);
        let mut remainder_transposed = remainder.transpose().to_owned();
        let init_norm = remainder.squared_norm_l2();
        let remainder_bf16 = Mat::from_fn(nrows, ncols, |row, col| {
            let a = remainder[(row, col)];
            let b = half::bf16::from_f64(a);
            a - b.to_f64()
        })
        .squared_norm_l2();

        let mut cut_helper = CutHelperV2::new(remainder.as_ref());

        let mut cur_norm = init_norm;
        let mut iter = 0;

        let instant = std::time::Instant::now();
        let mut finished = false;
        while iter < parameters.rank {
            let cut = cut_helper.cut_mat(
                remainder.as_mut(),
                remainder_transposed.as_mut(),
                rng,
                parameters.max_iters,
                parameters.parallelism,
            );

            cur_norm -= (cut * cut) / (nrows * ncols) as f64;
            if cur_norm <= remainder_bf16 {
                finished = true;
                break;
            }

            iter += 1;
        }
        dbg!(instant.elapsed());
        let break_even_point = iter;

        let sct_bytes = ((nrows / 8 + ncols / 8 + 8) * break_even_point) as f64;
        let f64_bytes = (nrows * ncols * 8) as f64;
        let f16_bytes = (nrows * ncols * 2) as f64;
        if finished {
            eprintln!("dim: ({nrows}*{ncols}), break even at: {break_even_point} (dim * {:.3}), f64 compression rate: {:.3}, bf16 compression rate: {:.3}", break_even_point as f64 / dim as f64, sct_bytes / f64_bytes, sct_bytes / f16_bytes);
        }
    }

    Ok(())
}
