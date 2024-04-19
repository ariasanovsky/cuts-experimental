use candle_core::safetensors::{load, Load};
use clap::Parser;
use cuts::{inplace_sct_signed::CutHelper, sct_helper::Sct};
use half::bf16;
use rand::thread_rng;
use std::time::SystemTime;
use tensorboard_writer::TensorboardWriter;

use hira::{
    faer::flat_mat,
    hira_helpers::{HiraLogger, HiraParameters},
    safetensors::{bits_to_signs, serialize_mats, serialize_sct},
    sct_map::SafeTensorsDirectoryMap,
};

#[derive(Debug, Parser)]
#[command(name = "HIRA")]
// #[command(version = "0.1.0")]
#[command(about = "Tests every matrix in a `sct` approximation", long_about = None)]
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
    /// The number of matrices to test (defaults to 1)
    #[arg(short = 'x')]
    matrices: usize,
    /// Also writes `a`, `r`, `s`, and `t` as `f64` matrices
    // #[arg(short = 'b')]
    // big_mats: bool,
    // /// Also writes `a` and `r` as `bf16` matrices
    // #[arg(short = 's')]
    // small_mats: bool,
    // /// Suppresses tensorboard event logs
    // #[arg(short = 'L')]
    // no_log: bool,
    /// The rank
    #[arg(short = 'r')]
    rank: usize,
    /// The number of iterations (defaults 40)
    #[arg(short = 'i')]
    iters: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let args = Args::try_parse()?;
    dbg!(&args);
    let Args {
        tensors,
        out_dir,
        nrows,
        ncols,
        matrices,
        // big_mats,
        // small_mats,
        rank,
        iters,
    } = args;
    let parameters = HiraParameters {
        max_iters: iters.unwrap_or(40),
        rank,
        parallelism: faer::Parallelism::None,
    };
    let directory = SafeTensorsDirectoryMap::new(tensors, out_dir, nrows, ncols, &parameters)?;
    let deserialized_tensors = directory.deserialize()?;
    let filtered_tensors = deserialized_tensors.filter_tensors(nrows, ncols)?;
    let mut rng = thread_rng();
    filtered_tensors
        .tensors()
        .iter()
        .take(matrices)
        .enumerate()
        .try_for_each(|(i, mapped_tensor)| -> eyre::Result<()> {
            let _name = mapped_tensor.name();
            let view = mapped_tensor.view();
            let out_dir = mapped_tensor.out_dir();
            // let MappedTensor {
            //     old_tensor,
            //     out_dir,
            // } = mapped_tensor;
            eprintln!("tensor[{i}] to {out_dir:?}");
            let mat = flat_mat(view);
            let mut remainder = mat.clone();
            let mut cut_helper = CutHelper::new(nrows, ncols);
            let mut sct_helper = Sct::new(nrows, ncols, parameters.rank);
            let (mut logger, init_summary) = HiraLogger::new(remainder.as_ref());
            let mut writer = TensorboardWriter::new(out_dir)?;
            writer.write_file_version()?;
            writer.write_summary(SystemTime::now(), 0, init_summary)?;
            writer.flush()?;
            for i in 1..=parameters.rank {
                if i % 10 == 0 {
                    eprintln!("r = {i}...")
                }
                let cut = cut_helper.cut_mat(
                    remainder.as_mut(),
                    // approximat.as_mut(),
                    &mut rng,
                    parameters.max_iters,
                    parameters.parallelism,
                );
                let s_signs = cut_helper.s_signs();
                let t_signs = cut_helper.t_signs();
                sct_helper.extend_with(s_signs, t_signs, cut.value);
                let log = logger.log_cut(&cut);
                writer.write_summary(SystemTime::now(), i as _, log)?;
                writer.flush()?;
            }
            serialize_sct(out_dir, &sct_helper, true)?;
            let mut approximat = flat_mat(view).to_faer();
            approximat -= remainder.as_ref();
            serialize_mats(out_dir, approximat.as_ref(), remainder.as_ref(), true, true)?;
            // load matrices
            let device = candle_core::Device::Cpu;
            let sct_mats = load(out_dir.join("sct.safetensors"), &device)?;
            assert_eq!(sct_mats.len(), 6);
            let mats = load(out_dir.join("mats.safetensors"), &device)?;
            assert_eq!(mats.len(), 4);
            let s_f64 = &sct_mats["s_f64"];
            let t_f64 = &sct_mats["t_f64"];
            let c_f64 = &sct_mats["c_f64"];
            let a_f64 = &mats["a_f64"];
            // check `mn * A_k ~ S_k * C_k * T_k^*` up to `f64` precision
            let normalization = (nrows * ncols) as f64;
            let lhs = (a_f64 * normalization)?;
            println!("Let L_k := mn * A_k");
            let l_min: f64 = lhs.min(0)?.min(0)?.to_vec0()?;
            let l_max: f64 = lhs.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{l_min} .. {l_max}] entries");
            let l_squared = (&lhs * &lhs)?;
            let l_frob = (l_squared.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{l_frob} Frobenius norm");
            let sc = s_f64.transpose(0, 1)?.broadcast_mul(c_f64)?;
            let sct = sc.matmul(t_f64)?;
            let rhs = sct; //.transpose(0, 1)?;
            println!("Let R_k := S_k * C_k * T_k^*");
            let r_min: f64 = rhs.min(0)?.min(0)?.to_vec0()?;
            let r_max: f64 = rhs.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{r_min} .. {r_max}] entries");
            let r_squared = (&rhs * &rhs)?;
            let r_frob = (r_squared.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{r_frob} Frobenius norm");
            let diff = (lhs - rhs)?;
            println!("Let D_k := L_k - R_k");
            let d_min: f64 = diff.min(0)?.min(0)?.to_vec0()?;
            let d_max: f64 = diff.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{d_min} .. {d_max}] entries");
            let d_square = (&diff * &diff)?;
            let d_frob = (d_square.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{d_frob} Frobenius norm");
            // check signs
            let s_f64_vecs: Vec<Vec<f64>> = s_f64.to_vec2()?;
            assert_eq!(s_f64_vecs.len(), rank);
            let s_sign_vecs: Vec<Vec<u8>> = sct_mats["s_signs"].to_vec2()?;
            assert_eq!(s_sign_vecs.len(), rank);
            s_f64_vecs
                .into_iter()
                .zip(s_sign_vecs)
                .for_each(|(s_f64, s_signs)| {
                    assert_eq!(s_f64.len(), nrows);
                    assert_eq!(s_signs.len(), sct_helper.dimensions().num_s_bytes());
                    s_f64
                        .into_iter()
                        .zip(bits_to_signs(&s_signs, nrows))
                        .for_each(|(s, sign)| assert_eq!(s, sign))
                });
            let t_f64_vecs: Vec<Vec<f64>> = t_f64.to_vec2()?;
            assert_eq!(t_f64_vecs.len(), rank);
            let t_sign_vecs: Vec<Vec<u8>> = sct_mats["t_signs"].to_vec2()?;
            assert_eq!(t_sign_vecs.len(), rank);
            t_f64_vecs
                .into_iter()
                .zip(t_sign_vecs)
                .for_each(|(t_f64, t_signs)| {
                    assert_eq!(t_f64.len(), ncols);
                    assert_eq!(t_signs.len(), sct_helper.dimensions().num_t_bytes());
                    t_f64
                        .into_iter()
                        .zip(bits_to_signs(&t_signs, ncols))
                        .for_each(|(t, sign)| assert_eq!(t, sign))
                });
            // check `R_k + A_k = A_0` up to `f64` precision
            let lhs = view
                .load(&device)?
                .to_dtype(candle_core::DType::F64)?;
            println!("Let L := A");
            let l_min: f64 = lhs.min(0)?.min(0)?.to_vec0()?;
            let l_max: f64 = lhs.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{l_min} .. {l_max}] entries");
            let l_squared = (&lhs * &lhs)?;
            let l_frob = (l_squared.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{l_frob} Frobenius norm");
            let r_f64 = &mats["r_f64"];
            let rhs = (a_f64 + r_f64)?;
            println!("Let R := A_k + R_k");
            let r_min: f64 = rhs.min(0)?.min(0)?.to_vec0()?;
            let r_max: f64 = rhs.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{r_min} .. {r_max}] entries");
            let r_squared = (&rhs * &rhs)?;
            let r_frob = (r_squared.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{r_frob} Frobenius norm");
            let diff = (lhs - rhs)?;
            println!("Let D := L - R");
            let d_min: f64 = diff.min(0)?.min(0)?.to_vec0()?;
            let d_max: f64 = diff.max(0)?.max(0)?.to_vec0()?;
            println!("\t[{d_min} .. {d_max}] entries");
            let d_square = (&diff * &diff)?;
            let d_frob = (d_square.sum_all()?.to_vec0::<f64>()?).sqrt();
            println!("\t{d_frob} Frobenius norm");
            // compare `A_k`, `R_k`, and `C_k` values as `bf16` and `f64`
            let a_bf16: Vec<Vec<bf16>> = mats["a_bf16"].to_vec2()?;
            let a_f64_to_bf16: Vec<Vec<bf16>> =
                a_f64.to_dtype(candle_core::DType::BF16)?.to_vec2()?;
            assert_eq!(a_bf16.len(), nrows);
            assert_eq!(a_f64_to_bf16.len(), nrows);
            a_bf16
                .into_iter()
                .zip(a_f64_to_bf16)
                .enumerate()
                .for_each(|(i, (x, y))| {
                    assert_eq!(x.len(), ncols, "{i}");
                    assert_eq!(y.len(), ncols, "{i}");
                    x.into_iter().zip(y).enumerate().for_each(|(j, (x, y))| {
                        assert_eq!(x, y, "{i}, {j}");
                    });
                });
            let r_bf16: Vec<Vec<bf16>> = mats["r_bf16"].to_vec2()?;
            let r_f64_to_bf16: Vec<Vec<bf16>> =
                r_f64.to_dtype(candle_core::DType::BF16)?.to_vec2()?;
            assert_eq!(r_bf16.len(), nrows);
            assert_eq!(r_f64_to_bf16.len(), nrows);
            r_bf16
                .into_iter()
                .zip(r_f64_to_bf16)
                .enumerate()
                .for_each(|(i, (x, y))| {
                    assert_eq!(x.len(), ncols, "{i}");
                    assert_eq!(y.len(), ncols, "{i}");
                    x.into_iter().zip(y).enumerate().for_each(|(j, (x, y))| {
                        assert_eq!(x, y, "{i}, {j}");
                    });
                });
            Ok(())
        })
}
