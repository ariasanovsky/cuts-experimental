#![allow(non_snake_case)]
use clap::Parser;
use cut_mist::hira_helpers::HiraParameters;
use cuts::inplace_sct_signed::CutHelperV2;
use faer::dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::linalg::{matmul, temp_mat_req};
use faer::reborrow::*;
use faer::{Col, Mat};
use rand::distributions::Distribution;
use rand::prelude::*;

#[derive(Debug, Parser)]
#[command(name = "Unitary SCT")]
// #[command(version = "0.1.0")]
#[command(about = "Approximates unitary matrices with sct decompositions", long_about = None)]
struct Args {
    #[arg(short = 'd')]
    dim: usize,
    /// The decomposition block size (default: 32)
    #[arg(short = 'b')]
    blocksize: Option<usize>,
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
        blocksize,
        iters,
        par,
    } = args;
    let parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank: usize::MAX,
        parallelism: par.map_or(faer::Parallelism::None, |par| faer::Parallelism::Rayon(par)),
    };

    let blocksize = blocksize.unwrap_or(32);
    let mut dim = 0usize;
    let incr = max_dim;
    // let incr = 64;
    while dim < max_dim {
        dim += incr;

        let rng = &mut StdRng::seed_from_u64(0);
        let nrows = dim;
        let ncols = dim;
        // let mut remainder: Mat<f64> = faer::stats::UnitaryMat { dimension: dim }.sample(rng);
        let mat: Mat<f64> = faer::stats::UnitaryMat { dimension: dim }.sample(rng);
        let init_norm = mat.squared_norm_l2();
        let remainder_bf16 = Mat::from_fn(nrows, ncols, |row, col| {
            let a = mat[(row, col)];
            let b = half::bf16::from_f64(a);
            a - b.to_f64()
        })
        .squared_norm_l2();

        let mut cut_helper = CutHelperV2::new(mat.as_ref());
        let mut two_remainder = faer::scale(2.0) * &mat;
        let mut two_remainder_transposed = two_remainder.transpose().to_owned();

        let mut remainder_norm = init_norm;
        let mut iter = 0;

        let instant = std::time::Instant::now();
        let mut finished = false;

        let mut S = Mat::<f64>::zeros(mat.nrows(), blocksize);
        let mut T = Mat::<f64>::zeros(mat.ncols(), blocksize);
        let mut C = Col::<f64>::zeros(blocksize);
        let mut how_full = 0usize;
        let mut mem = GlobalPodBuffer::new(
            StackReq::new::<u64>(Ord::max(nrows, ncols))
                .and(temp_mat_req::<f64>(blocksize, 1).unwrap()),
        );
        let mut stack = PodStack::new(&mut mem);
        while iter < parameters.rank {
            if how_full == blocksize {
                let tmp = &S * C.as_ref().column_vector_as_diagonal();
                matmul::matmul(
                    two_remainder.as_mut(), // acc
                    tmp.as_ref(),           // lhs
                    T.transpose(),          // rhs
                    Some(1.0),              // alpha
                    2.0,                    // beta
                    parameters.parallelism, // parallelism
                );
                two_remainder_transposed
                    .as_mut()
                    .copy_from(two_remainder.transpose());
                how_full = 0;
            }

            how_full += 1;
            let cut = cut_helper.cut_mat(
                two_remainder.as_ref(),
                two_remainder_transposed.as_ref(),
                S.as_mut().get_mut(.., ..how_full),
                C.as_mut().get_mut(..how_full),
                T.as_mut().get_mut(.., ..how_full),
                rng,
                parameters.max_iters,
                parameters.parallelism,
                stack.rb_mut(),
            );

            remainder_norm -= (cut * cut) / (nrows * ncols) as f64;
            if remainder_norm <= remainder_bf16 {
                let tmp = S.get(.., ..how_full) * C.get(..how_full).column_vector_as_diagonal();
                matmul::matmul(
                    two_remainder.as_mut(),            // acc
                    tmp.as_ref(),                      // lhs
                    T.get(.., ..how_full).transpose(), // rhs
                    Some(1.0),                         // alpha
                    2.0,                               // beta
                    parameters.parallelism,            // parallelism
                );
                two_remainder_transposed.as_mut().copy_from(mat.transpose());
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
